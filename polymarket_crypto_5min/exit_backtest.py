"""Exit-aware backtesting for five-minute prediction markets.

This module tests the user's live-trading idea: enter when a calibrated trend is
confirmed, then try to de-risk before settlement by taking profit, stopping out,
or holding only when no exit signal appears.

The output is intentionally verbose: every simulated entry has a row in the
trade log with entry/exit timestamps, prices, reason, PnL, and losing-trade
flags so bad trades can be audited individually.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from .features import shares_for_stake, taker_fee_per_share
from .metrics import performance_metrics
from .walk_forward import WalkForwardConfig, WalkForwardRule, calibrate_candidates, generate_rule_grid, rule_mask


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """Rules for leaving a position after entry.

    Prices are CLOB/trade-price probabilities in [0, 1]. ``take_profit`` and
    ``stop_loss`` are absolute price moves, not percentages. ``target_price`` is
    an absolute price cap such as 0.90. ``max_hold_seconds=None`` means hold to
    settlement if no take-profit/stop is triggered.
    """

    take_profit: float | None = 0.10
    target_price: float | None = 0.90
    stop_loss: float | None = 0.05
    max_hold_seconds: int | None = None
    exit_fee_rate: float = 0.07


DEFAULT_EXIT_POLICY_GRID = {
    "take_profit": (0.03, 0.05, 0.08, 0.10, 0.15, None),
    "target_price": (0.75, 0.80, 0.85, 0.90, 0.95, None),
    "stop_loss": (0.03, 0.05, 0.08, 0.10, None),
    "max_hold_seconds": (10, 20, 30, None),
}


def generate_exit_policy_grid(policy_grid: dict[str, Iterable[float | int | None]] | None = None) -> list[ExitPolicy]:
    grid = policy_grid or DEFAULT_EXIT_POLICY_GRID
    policies: list[ExitPolicy] = []
    for take_profit, target_price, stop_loss, max_hold in product(
        grid["take_profit"], grid["target_price"], grid["stop_loss"], grid["max_hold_seconds"]
    ):
        if take_profit is None and target_price is None and max_hold is None:
            # This is pure hold-to-settlement; keep it out of the exit-policy
            # grid because it is already covered by the base walk-forward test.
            continue
        policies.append(
            ExitPolicy(
                take_profit=None if take_profit is None else float(take_profit),
                target_price=None if target_price is None else float(target_price),
                stop_loss=None if stop_loss is None else float(stop_loss),
                max_hold_seconds=None if max_hold is None else int(max_hold),
            )
        )
    return policies


def prepare_price_history(price_history: pd.DataFrame) -> pd.DataFrame:
    """Normalize CLOB/trade price history for exit simulations."""
    if price_history is None or price_history.empty:
        return pd.DataFrame(columns=["condition_id", "asset_id", "ts", "p"])
    hist = price_history.copy()
    if "ts" in hist.columns:
        hist["ts"] = pd.to_datetime(hist["ts"], utc=True, errors="coerce")
    elif "t" in hist.columns:
        hist["ts"] = pd.to_datetime(hist["t"], unit="s", utc=True, errors="coerce")
    else:
        raise ValueError("price history must include ts or t")
    hist["condition_id"] = hist["condition_id"].astype(str)
    hist["asset_id"] = hist["asset_id"].astype(str)
    hist["p"] = pd.to_numeric(hist["p"], errors="coerce")
    return hist.dropna(subset=["condition_id", "asset_id", "ts", "p"]).sort_values(
        ["condition_id", "asset_id", "ts"]
    ).reset_index(drop=True)


def attach_candidate_asset_ids(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = candidates.copy()
    rows["asset_id"] = np.where(rows["side"].eq("UP"), rows.get("up_asset_id"), rows.get("down_asset_id"))
    rows["asset_id"] = rows["asset_id"].astype(str)
    return rows


def simulate_exit_policy(
    entries: pd.DataFrame,
    price_history: pd.DataFrame,
    policy: ExitPolicy,
    *,
    stake_usdc: float = 10.0,
) -> pd.DataFrame:
    """Simulate one exit policy for a set of already-selected entries.

    ``entries`` should be candidate rows selected by a walk-forward entry rule.
    The function emits one row per entry, even if no post-entry price history is
    available. Missing path rows default to holding to settlement.
    """
    if entries.empty:
        return pd.DataFrame()
    hist = prepare_price_history(price_history)
    selected = attach_candidate_asset_ids(entries).copy()
    selected["condition_id"] = selected["condition_id"].astype(str)
    selected["snapshot_dt"] = pd.to_datetime(selected["snapshot_dt"], utc=True, errors="coerce")
    selected["end_dt"] = pd.to_datetime(selected["end_dt"], utc=True, errors="coerce")
    records: list[dict[str, object]] = []
    for _, entry in selected.iterrows():
        records.append(_simulate_one_exit(entry, hist, policy, stake_usdc=stake_usdc))
    return pd.DataFrame(records).sort_values(["entry_dt", "condition_id", "side"]).reset_index(drop=True)


def _simulate_one_exit(entry: pd.Series, hist: pd.DataFrame, policy: ExitPolicy, *, stake_usdc: float) -> dict[str, object]:
    entry_price = float(entry["market_price"])
    entry_fee = float(taker_fee_per_share(entry_price))
    shares = shares_for_stake(stake_usdc, entry_price)
    entry_dt = pd.to_datetime(entry["snapshot_dt"], utc=True)
    end_dt = pd.to_datetime(entry["end_dt"], utc=True)
    side = str(entry["side"])
    condition_id = str(entry["condition_id"])
    asset_id = str(entry["asset_id"])
    won = bool(entry["won"])

    deadline = end_dt
    if policy.max_hold_seconds is not None:
        deadline = min(deadline, entry_dt + pd.Timedelta(seconds=int(policy.max_hold_seconds)))

    path = hist[
        hist["condition_id"].eq(condition_id)
        & hist["asset_id"].astype(str).eq(asset_id)
        & hist["ts"].gt(entry_dt)
        & hist["ts"].le(deadline)
    ].sort_values("ts")

    exit_reason = "HOLD_TO_SETTLEMENT"
    exit_dt = end_dt
    exit_price = 1.0 if won else 0.0
    exit_fee = 0.0
    path_points_seen = int(len(path))

    for _, point in path.iterrows():
        price = float(point["p"])
        reason = _exit_reason(entry_price, price, policy)
        if reason is not None:
            exit_reason = reason
            exit_dt = pd.to_datetime(point["ts"], utc=True)
            exit_price = price
            exit_fee = float(taker_fee_per_share(exit_price, fee_rate=policy.exit_fee_rate))
            break
    else:
        if policy.max_hold_seconds is not None and not path.empty:
            last = path.iloc[-1]
            exit_reason = "MAX_HOLD_EXIT"
            exit_dt = pd.to_datetime(last["ts"], utc=True)
            exit_price = float(last["p"])
            exit_fee = float(taker_fee_per_share(exit_price, fee_rate=policy.exit_fee_rate))

    if exit_reason == "HOLD_TO_SETTLEMENT":
        pnl_per_share = (1.0 - entry_price - entry_fee) if won else (-entry_price - entry_fee)
        realized_settlement = True
    else:
        # Conservative approximation: pay taker-like fee on exit too. Selling at
        # exit_price receives exit_price minus fee. If the exit is maker, actual
        # PnL can be better.
        pnl_per_share = exit_price - entry_price - entry_fee - exit_fee
        realized_settlement = False

    pnl_usdc = shares * pnl_per_share
    return {
        "condition_id": condition_id,
        "market_id": entry.get("market_id"),
        "question": entry.get("question"),
        "side": side,
        "entry_dt": entry_dt,
        "exit_dt": exit_dt,
        "end_dt": end_dt,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_fee_per_share": entry_fee,
        "exit_fee_per_share": exit_fee,
        "all_in_entry_cost": entry_price + entry_fee,
        "shares": shares,
        "stake_usdc": stake_usdc,
        "pnl_usdc": pnl_usdc,
        "return_on_stake": pnl_usdc / stake_usdc if stake_usdc else math.nan,
        "won_at_settlement": won,
        "exit_reason": exit_reason,
        "realized_settlement": realized_settlement,
        "loss_trade": pnl_usdc < 0,
        "path_points_seen": path_points_seen,
        "take_profit": policy.take_profit,
        "target_price": policy.target_price,
        "stop_loss": policy.stop_loss,
        "max_hold_seconds": policy.max_hold_seconds,
        "p_calibrated": entry.get("p_calibrated"),
        "p_lower": entry.get("p_lower"),
        "edge_lower": entry.get("edge_lower"),
        "market_price": entry_price,
        "score_bps": entry.get("score_bps"),
        "abs_score_bps": entry.get("abs_score_bps"),
        "calibration_n": entry.get("calibration_n"),
        "calibration_scope": entry.get("calibration_scope"),
        "fold": entry.get("fold"),
    }


def _exit_reason(entry_price: float, price: float, policy: ExitPolicy) -> str | None:
    if policy.stop_loss is not None and price <= entry_price - policy.stop_loss:
        return "STOP_LOSS"
    if policy.target_price is not None and price >= policy.target_price:
        return "TARGET_PRICE"
    if policy.take_profit is not None and price >= entry_price + policy.take_profit:
        return "TAKE_PROFIT"
    return None


def select_entry_exit_rules_from_training(
    calibrated_train: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    config: WalkForwardConfig,
    entry_rules: list[WalkForwardRule] | None = None,
    exit_policies: list[ExitPolicy] | None = None,
) -> tuple[WalkForwardRule | None, ExitPolicy | None, dict[str, float | int]]:
    """Select an entry rule and exit policy using the training fold only."""
    best_entry: WalkForwardRule | None = None
    best_exit: ExitPolicy | None = None
    best_score = -np.inf
    best_stats: dict[str, float | int] = {"score": -np.inf, "train_trades": 0}
    for entry_rule in entry_rules or generate_rule_grid():
        entries = calibrated_train[rule_mask(calibrated_train, entry_rule)].copy()
        if len(entries) < config.min_train_trades:
            continue
        for exit_policy in exit_policies or generate_exit_policy_grid():
            trades = simulate_exit_policy(entries, price_history, exit_policy, stake_usdc=config.stake_usdc)
            if len(trades) < config.min_train_trades:
                continue
            metrics = performance_metrics(_to_metrics_frame(trades), initial_capital=config.initial_capital)
            if metrics.empty:
                continue
            all_metrics = metrics.iloc[0]
            sharpe = _finite_or(float(all_metrics.get("trade_sharpe", np.nan)), -5.0)
            roi = _finite_or(float(all_metrics.get("roi_on_stake", np.nan)), 0.0)
            mdd = abs(_finite_or(float(all_metrics.get("max_drawdown_pct", 0.0)), 0.0))
            loss_rate = float((trades["pnl_usdc"] < 0).mean())
            score = sharpe + 0.25 * roi - 4.0 * mdd - 0.75 * loss_rate
            if score > best_score:
                best_score = score
                best_entry = entry_rule
                best_exit = exit_policy
                best_stats = {
                    "score": float(score),
                    "train_trades": int(len(trades)),
                    "train_roi": float(all_metrics.get("roi_on_stake", np.nan)),
                    "train_trade_sharpe": float(all_metrics.get("trade_sharpe", np.nan)),
                    "train_mdd_pct": float(all_metrics.get("max_drawdown_pct", np.nan)),
                    "train_loss_rate": loss_rate,
                }
    return best_entry, best_exit, best_stats


def walk_forward_exit_backtest(
    candidates: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    config: WalkForwardConfig | None = None,
    entry_rules: list[WalkForwardRule] | None = None,
    exit_policies: list[ExitPolicy] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a strict walk-forward backtest with learned entry and exit rules."""
    cfg = config or WalkForwardConfig()
    rows = attach_candidate_asset_ids(candidates).copy()
    rows["end_dt"] = pd.to_datetime(rows["end_dt"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["condition_id", "end_dt", "market_price", "won", "snapshot_dt"]).sort_values(
        ["end_dt", "condition_id", "side"]
    )
    ordered_markets = (
        rows[["condition_id", "end_dt"]].drop_duplicates("condition_id").sort_values("end_dt").reset_index(drop=True)
    )
    if len(ordered_markets) <= cfg.train_markets:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    fold = 0
    start = cfg.train_markets
    while start < len(ordered_markets):
        fold += 1
        end = min(start + cfg.test_markets, len(ordered_markets))
        train_ids = set(ordered_markets.iloc[:start]["condition_id"].astype(str))
        test_ids = set(ordered_markets.iloc[start:end]["condition_id"].astype(str))
        train = rows[rows["condition_id"].astype(str).isin(train_ids)].copy()
        test = rows[rows["condition_id"].astype(str).isin(test_ids)].copy()
        train_end = train["end_dt"].max()
        test_start = test["end_dt"].min()
        leakage_ok = bool(train_end < test_start)
        calibrated_train = calibrate_candidates(train, train, alpha=cfg.calibration_alpha, min_group_observations=cfg.min_bin_observations)
        entry_rule, exit_policy, train_stats = select_entry_exit_rules_from_training(
            calibrated_train,
            price_history,
            config=cfg,
            entry_rules=entry_rules,
            exit_policies=exit_policies,
        )
        if entry_rule is None or exit_policy is None:
            fold_rows.append(
                {
                    "fold": fold,
                    "selected": False,
                    "train_markets": len(train_ids),
                    "test_markets": len(test_ids),
                    "train_end_dt": train_end,
                    "test_start_dt": test_start,
                    "leakage_check_passed": leakage_ok,
                    "test_trades": 0,
                    "test_pnl_usdc": 0.0,
                }
            )
            start = end
            continue
        calibrated_test = calibrate_candidates(train, test, alpha=cfg.calibration_alpha, min_group_observations=cfg.min_bin_observations)
        test_entries = calibrated_test[rule_mask(calibrated_test, entry_rule)].copy()
        trades = simulate_exit_policy(test_entries, price_history, exit_policy, stake_usdc=cfg.stake_usdc)
        if not trades.empty:
            trades["fold"] = fold
            all_trades.append(trades)
        fold_rows.append(
            {
                "fold": fold,
                "selected": True,
                "train_markets": len(train_ids),
                "test_markets": len(test_ids),
                "train_end_dt": train_end,
                "test_start_dt": test_start,
                "test_end_dt": test["end_dt"].max(),
                "leakage_check_passed": leakage_ok,
                "test_trades": int(len(trades)),
                "test_pnl_usdc": float(trades["pnl_usdc"].sum()) if not trades.empty else 0.0,
                **train_stats,
            }
        )
        rule_rows.append({"fold": fold, **asdict(entry_rule), **asdict(exit_policy), **train_stats})
        start = end

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return trades_out, pd.DataFrame(fold_rows), pd.DataFrame(rule_rows)


def _to_metrics_frame(exit_trades: pd.DataFrame) -> pd.DataFrame:
    rows = exit_trades.copy()
    rows["bucket"] = "EXIT_AWARE"
    rows["end_dt"] = rows["exit_dt"]
    rows["won"] = rows["pnl_usdc"].gt(0)
    return rows


def losing_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Return a sortable losing-trade audit table."""
    if trades.empty:
        return trades.copy()
    losses = trades[trades["pnl_usdc"].lt(0)].copy()
    return losses.sort_values(["pnl_usdc", "entry_dt"], ascending=[True, True]).reset_index(drop=True)


def _finite_or(value: float, default: float) -> float:
    return value if math.isfinite(value) else default
