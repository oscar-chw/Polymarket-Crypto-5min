"""Walk-forward calibration and backtesting.

This module avoids hard-wired strategy thresholds in reported results. Each test
fold is evaluated with a calibrator and rule selected only from markets that
closed before the test fold.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
import pandas as pd

from .features import profit_per_share, shares_for_stake, taker_fee_per_share
from .metrics import performance_metrics


@dataclass(frozen=True, slots=True)
class WalkForwardRule:
    min_edge_lower: float
    min_p_lower: float
    min_price: float
    max_price: float
    min_abs_score_bps: float


@dataclass(slots=True)
class WalkForwardConfig:
    initial_capital: float = 2_000.0
    stake_usdc: float = 10.0
    train_markets: int = 400
    test_markets: int = 100
    min_train_trades: int = 25
    min_bin_observations: int = 30
    calibration_alpha: float = 8.0
    lower_bound_z: float = 1.64
    max_price_for_taker: float = 0.985


DEFAULT_RULE_GRID = {
    "min_edge_lower": (0.000, 0.0025, 0.005, 0.010, 0.015, 0.020),
    "min_p_lower": (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
    "min_price": (0.50, 0.60, 0.70, 0.80, 0.90),
    "max_price": (0.95, 0.975, 0.985),
    "min_abs_score_bps": (0.0, 4.0, 8.0, 12.0, 20.0),
}

PRICE_BINS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 1.0)
SCORE_BINS = (-np.inf, -20.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0, 20.0, np.inf)


def make_side_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand each market row into UP and DOWN candidate rows.

    This prevents the backtest from hard-wiring a single model-chosen direction.
    The walk-forward selection can learn whether either side is worth trading.
    """
    required = {"condition_id", "realized_direction", "market_up_price", "market_down_price", "model_prob_up"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature frame missing required columns: {sorted(missing)}")

    up = _candidate_side(frame, side="UP")
    down = _candidate_side(frame, side="DOWN")
    candidates = pd.concat([up, down], ignore_index=True)
    candidates = candidates.dropna(subset=["market_price", "raw_model_prob", "won", "end_dt", "snapshot_dt"])
    candidates = candidates[candidates["market_price"].between(0.001, 0.999, inclusive="both")].copy()
    candidates["fee_per_share"] = taker_fee_per_share(candidates["market_price"])
    candidates["all_in_cost"] = candidates["market_price"] + candidates["fee_per_share"]
    candidates["raw_model_edge"] = candidates["raw_model_prob"] - candidates["all_in_cost"]
    candidates["profit_per_share"] = profit_per_share(candidates["market_price"], candidates["won"])
    return candidates.sort_values(["end_dt", "condition_id", "side"]).reset_index(drop=True)


def _candidate_side(frame: pd.DataFrame, *, side: str) -> pd.DataFrame:
    rows = frame.copy()
    rows["side"] = side
    if side == "UP":
        rows["market_price"] = pd.to_numeric(rows["market_up_price"], errors="coerce")
        rows["raw_model_prob"] = pd.to_numeric(rows["model_prob_up"], errors="coerce")
        rows["score_aligned_bps"] = pd.to_numeric(rows["score_bps"], errors="coerce")
        rows["momentum_1m_aligned_bps"] = pd.to_numeric(rows["momentum_1m_bps"], errors="coerce")
        rows["momentum_3m_aligned_bps"] = pd.to_numeric(rows["momentum_3m_bps"], errors="coerce")
    else:
        rows["market_price"] = pd.to_numeric(rows["market_down_price"], errors="coerce")
        rows["raw_model_prob"] = 1.0 - pd.to_numeric(rows["model_prob_up"], errors="coerce")
        rows["score_aligned_bps"] = -pd.to_numeric(rows["score_bps"], errors="coerce")
        rows["momentum_1m_aligned_bps"] = -pd.to_numeric(rows["momentum_1m_bps"], errors="coerce")
        rows["momentum_3m_aligned_bps"] = -pd.to_numeric(rows["momentum_3m_bps"], errors="coerce")
    rows["won"] = rows["realized_direction"].eq(side)
    rows["abs_score_bps"] = pd.to_numeric(rows["score_bps"], errors="coerce").abs()
    rows["end_dt"] = pd.to_datetime(rows["end_dt"], utc=True, errors="coerce")
    rows["snapshot_dt"] = pd.to_datetime(rows["snapshot_dt"], utc=True, errors="coerce")
    return rows


def calibrate_candidates(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    *,
    alpha: float = 8.0,
    min_group_observations: int = 30,
    lower_bound_z: float = 1.64,
) -> pd.DataFrame:
    """Calibrate target candidate probabilities from a prior reference set only."""
    ref = _with_bins(reference).copy()
    tgt = _with_bins(target).copy()
    if ref.empty:
        tgt["p_calibrated"] = np.nan
        tgt["p_lower"] = np.nan
        tgt["calibration_n"] = 0
        tgt["calibration_scope"] = "none"
        tgt["edge_calibrated"] = np.nan
        tgt["edge_lower"] = np.nan
        return tgt

    global_wins = int(ref["won"].sum())
    global_n = int(len(ref))
    global_p = _smoothed_rate(global_wins, global_n, prior=0.5, alpha=alpha)

    by_price_score = _group_stats(ref, ["price_bin", "score_bin"])
    by_price = _group_stats(ref, ["price_bin"])

    p_values: list[float] = []
    lower_values: list[float] = []
    n_values: list[int] = []
    scopes: list[str] = []

    for _, row in tgt.iterrows():
        stats = by_price_score.get((row["price_bin"], row["score_bin"]))
        scope = "price_score_bin"
        if stats is None or stats["n"] < min_group_observations:
            stats = by_price.get((row["price_bin"],))
            scope = "price_bin"
        if stats is None or stats["n"] < min_group_observations:
            stats = {"wins": global_wins, "n": global_n}
            scope = "global"
        p = _smoothed_rate(int(stats["wins"]), int(stats["n"]), prior=global_p, alpha=alpha)
        n_eff = max(float(stats["n"]) + alpha, 1.0)
        lower = max(0.0, p - lower_bound_z * math.sqrt(max(p * (1.0 - p), 0.0) / n_eff))
        p_values.append(float(p))
        lower_values.append(float(min(lower, p)))
        n_values.append(int(stats["n"]))
        scopes.append(scope)

    tgt["p_calibrated"] = p_values
    tgt["p_lower"] = lower_values
    tgt["calibration_n"] = n_values
    tgt["calibration_scope"] = scopes
    tgt["edge_calibrated"] = tgt["p_calibrated"] - tgt["all_in_cost"]
    tgt["edge_lower"] = tgt["p_lower"] - tgt["all_in_cost"]
    return tgt


def _with_bins(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["price_bin"] = pd.cut(rows["market_price"], bins=PRICE_BINS, include_lowest=True, right=True).astype(str)
    rows["score_bin"] = pd.cut(rows["score_aligned_bps"], bins=SCORE_BINS, include_lowest=True, right=True).astype(str)
    return rows


def _group_stats(frame: pd.DataFrame, columns: list[str]) -> dict[tuple[str, ...], dict[str, int]]:
    stats: dict[tuple[str, ...], dict[str, int]] = {}
    if frame.empty:
        return stats
    grouped = frame.groupby(columns, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        stats[tuple(str(item) for item in key)] = {"wins": int(group["won"].sum()), "n": int(len(group))}
    return stats


def _smoothed_rate(wins: int, n: int, *, prior: float, alpha: float) -> float:
    if n <= 0:
        return float(prior)
    return float((wins + alpha * prior) / (n + alpha))


def rule_mask(candidates: pd.DataFrame, rule: WalkForwardRule) -> pd.Series:
    return (
        candidates["edge_lower"].ge(rule.min_edge_lower)
        & candidates["p_lower"].ge(rule.min_p_lower)
        & candidates["market_price"].ge(rule.min_price)
        & candidates["market_price"].le(rule.max_price)
        & candidates["abs_score_bps"].ge(rule.min_abs_score_bps)
    )


def simulate_candidate_trades(
    candidates: pd.DataFrame,
    rule: WalkForwardRule,
    *,
    stake_usdc: float = 10.0,
) -> pd.DataFrame:
    trades = candidates[rule_mask(candidates, rule)].copy()
    if trades.empty:
        return trades
    trades["bucket"] = "WALK_FORWARD_CALIBRATED"
    trades["chosen_direction"] = trades["side"]
    trades["chosen_prob"] = trades["p_calibrated"]
    trades["expected_value_per_share"] = trades["edge_calibrated"]
    trades["shares"] = [shares_for_stake(stake_usdc, price) for price in trades["market_price"]]
    trades["stake_usdc"] = stake_usdc
    trades["pnl_usdc"] = trades["shares"] * trades["profit_per_share"]
    trades["return_on_stake"] = trades["pnl_usdc"] / stake_usdc
    return trades.reset_index(drop=True)


def generate_rule_grid(rule_grid: Mapping[str, Iterable[float]] | None = None) -> list[WalkForwardRule]:
    grid = rule_grid or DEFAULT_RULE_GRID
    rules: list[WalkForwardRule] = []
    for edge, p_lower, min_price, max_price, abs_bps in product(
        grid["min_edge_lower"],
        grid["min_p_lower"],
        grid["min_price"],
        grid["max_price"],
        grid["min_abs_score_bps"],
    ):
        if min_price > max_price:
            continue
        rules.append(
            WalkForwardRule(
                min_edge_lower=float(edge),
                min_p_lower=float(p_lower),
                min_price=float(min_price),
                max_price=float(max_price),
                min_abs_score_bps=float(abs_bps),
            )
        )
    return rules


def select_rule_from_training(
    calibrated_train: pd.DataFrame,
    *,
    config: WalkForwardConfig,
    rules: list[WalkForwardRule] | None = None,
) -> tuple[WalkForwardRule | None, dict[str, float | int]]:
    """Select a rule using training data only."""
    best_rule: WalkForwardRule | None = None
    best_score = -np.inf
    best_stats: dict[str, float | int] = {"score": -np.inf, "train_trades": 0}
    for rule in rules or generate_rule_grid():
        trades = simulate_candidate_trades(calibrated_train, rule, stake_usdc=config.stake_usdc)
        if len(trades) < config.min_train_trades:
            continue
        metrics = performance_metrics(trades, initial_capital=config.initial_capital)
        if metrics.empty:
            continue
        all_metrics = metrics.iloc[0]
        sharpe = _finite_or(float(all_metrics.get("trade_sharpe", np.nan)), -5.0)
        roi = _finite_or(float(all_metrics.get("roi_on_stake", np.nan)), 0.0)
        mdd = abs(_finite_or(float(all_metrics.get("max_drawdown_pct", 0.0)), 0.0))
        profit_factor = _finite_or(float(all_metrics.get("profit_factor", np.nan)), 1.0)
        score = sharpe + 0.25 * roi + 0.05 * min(profit_factor, 10.0) - 4.0 * mdd
        if score > best_score:
            best_score = score
            best_rule = rule
            best_stats = {
                "score": float(score),
                "train_trades": int(len(trades)),
                "train_roi": float(all_metrics.get("roi_on_stake", np.nan)),
                "train_trade_sharpe": float(all_metrics.get("trade_sharpe", np.nan)),
                "train_mdd_pct": float(all_metrics.get("max_drawdown_pct", np.nan)),
            }
    return best_rule, best_stats


def _finite_or(value: float, default: float) -> float:
    return value if math.isfinite(value) else default


def walk_forward_backtest(
    candidates: pd.DataFrame,
    *,
    config: WalkForwardConfig | None = None,
    rule_grid: Mapping[str, Iterable[float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a strict expanding-window walk-forward backtest.

    Returns ``(trades, fold_report, selected_rules)``. ``leakage_check_passed``
    requires every training label to settle before the test fold's first
    decision: ``train_end_dt < test_first_decision_dt``.
    """
    cfg = config or WalkForwardConfig()
    rules = generate_rule_grid(rule_grid)
    rows = candidates.copy()
    rows["end_dt"] = pd.to_datetime(rows["end_dt"], utc=True, errors="coerce")
    rows["snapshot_dt"] = pd.to_datetime(rows["snapshot_dt"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["condition_id", "end_dt", "market_price", "won"]).sort_values(["end_dt", "condition_id"])
    ordered_markets = (
        rows[["condition_id", "end_dt"]].drop_duplicates("condition_id").sort_values("end_dt").reset_index(drop=True)
    )
    n_markets = len(ordered_markets)
    if n_markets <= cfg.train_markets:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_trades: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    fold = 0
    test_start_idx = cfg.train_markets
    while test_start_idx < n_markets:
        fold += 1
        test_end_idx = min(test_start_idx + cfg.test_markets, n_markets)
        train_ids = set(ordered_markets.iloc[:test_start_idx]["condition_id"].astype(str))
        test_ids = set(ordered_markets.iloc[test_start_idx:test_end_idx]["condition_id"].astype(str))
        train = rows[rows["condition_id"].astype(str).isin(train_ids)].copy()
        test = rows[rows["condition_id"].astype(str).isin(test_ids)].copy()
        if train.empty or test.empty:
            test_start_idx = test_end_idx
            continue

        calibrated_train = calibrate_candidates(
            train,
            train,
            alpha=cfg.calibration_alpha,
            min_group_observations=cfg.min_bin_observations,
            lower_bound_z=cfg.lower_bound_z,
        )
        selected_rule, train_stats = select_rule_from_training(calibrated_train, config=cfg, rules=rules)
        train_end = train["end_dt"].max()
        test_start = test["end_dt"].min()
        # Folds are cut by market count, so end_dt ties or overlapping windows
        # can put a training label after a test decision; compare label time
        # with decision time, not with test settlement time.
        test_first_decision = test["snapshot_dt"].min()
        leakage_ok = bool(train_end < test_first_decision)
        if selected_rule is None:
            fold_rows.append(
                {
                    "fold": fold,
                    "train_markets": len(train_ids),
                    "test_markets": len(test_ids),
                    "train_end_dt": train_end,
                    "test_start_dt": test_start,
                    "test_first_decision_dt": test_first_decision,
                    "leakage_check_passed": leakage_ok,
                    "selected": False,
                    "test_trades": 0,
                    "test_pnl_usdc": 0.0,
                }
            )
            test_start_idx = test_end_idx
            continue

        calibrated_test = calibrate_candidates(
            train,
            test,
            alpha=cfg.calibration_alpha,
            min_group_observations=cfg.min_bin_observations,
            lower_bound_z=cfg.lower_bound_z,
        )
        trades = simulate_candidate_trades(calibrated_test, selected_rule, stake_usdc=cfg.stake_usdc)
        if not trades.empty:
            trades["fold"] = fold
            all_trades.append(trades)
        fold_rows.append(
            {
                "fold": fold,
                "train_markets": len(train_ids),
                "test_markets": len(test_ids),
                "train_start_dt": train["end_dt"].min(),
                "train_end_dt": train_end,
                "test_start_dt": test_start,
                "test_end_dt": test["end_dt"].max(),
                "test_first_decision_dt": test_first_decision,
                "leakage_check_passed": leakage_ok,
                "selected": True,
                "test_trades": int(len(trades)),
                "test_pnl_usdc": float(trades["pnl_usdc"].sum()) if not trades.empty else 0.0,
                **train_stats,
            }
        )
        rule_rows.append({"fold": fold, **asdict(selected_rule), **train_stats})
        test_start_idx = test_end_idx

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return trades_out, pd.DataFrame(fold_rows), pd.DataFrame(rule_rows)
