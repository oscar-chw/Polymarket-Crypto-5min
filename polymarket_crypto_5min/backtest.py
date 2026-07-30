"""Backtesting and threshold search for the 5-minute strategy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from .features import shares_for_stake


@dataclass(slots=True)
class StrategyThresholds:
    """Thresholds for the two research buckets.

    confirm_min_abs_bps:
        Minimum BTC distance from market-start price. This is the "almost
        confirmed" bucket.
    value_max_market_price / value_min_model_prob / min_ev_per_share:
        The "market says unlikely, model still likes it" bucket.
    """

    confirm_min_abs_bps: float = 12.0
    confirm_min_model_prob: float = 0.72
    value_max_market_price: float = 0.40
    value_min_model_prob: float = 0.60
    min_ev_per_share: float = 0.02


def classify_opportunity(frame: pd.DataFrame, thresholds: StrategyThresholds) -> pd.Series:
    confirm = (
        frame["abs_score_bps"].ge(thresholds.confirm_min_abs_bps)
        & frame["chosen_prob"].ge(thresholds.confirm_min_model_prob)
        & frame["expected_value_per_share"].ge(thresholds.min_ev_per_share)
    )
    value = (
        frame["market_price"].le(thresholds.value_max_market_price)
        & frame["chosen_prob"].ge(thresholds.value_min_model_prob)
        & frame["expected_value_per_share"].ge(thresholds.min_ev_per_share)
    )
    bucket = pd.Series("SKIP", index=frame.index, dtype="object")
    bucket.loc[value] = "VALUE_MISMATCH"
    bucket.loc[confirm] = "CONFIRMATION"
    return bucket


def simulate_strategy(
    frame: pd.DataFrame,
    thresholds: StrategyThresholds,
    *,
    stake_usdc: float = 10.0,
) -> pd.DataFrame:
    """Return per-trade simulated results using fixed USDC stake per trade."""
    trades = frame.copy()
    trades["bucket"] = classify_opportunity(trades, thresholds)
    trades = trades[trades["bucket"].ne("SKIP")].copy()
    if trades.empty:
        return trades
    trades["shares"] = [shares_for_stake(stake_usdc, price) for price in trades["market_price"]]
    trades["stake_usdc"] = stake_usdc
    trades["pnl_usdc"] = trades["shares"] * trades["profit_per_share"]
    trades["return_on_stake"] = trades["pnl_usdc"] / stake_usdc
    return trades.reset_index(drop=True)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=["bucket", "trades", "win_rate", "avg_price", "avg_ev", "pnl_usdc", "roi", "avg_abs_bps"]
        )
    grouped = trades.groupby("bucket", dropna=False)
    summary = grouped.agg(
        trades=("bucket", "size"),
        win_rate=("won", "mean"),
        avg_price=("market_price", "mean"),
        avg_model_prob=("chosen_prob", "mean"),
        avg_ev=("expected_value_per_share", "mean"),
        pnl_usdc=("pnl_usdc", "sum"),
        stake_usdc=("stake_usdc", "sum"),
        avg_abs_bps=("abs_score_bps", "mean"),
    ).reset_index()
    summary["roi"] = summary["pnl_usdc"] / summary["stake_usdc"]
    return summary.sort_values("pnl_usdc", ascending=False).reset_index(drop=True)


def threshold_report(
    frame: pd.DataFrame,
    *,
    confirm_bps_grid: Iterable[float] = (4, 6, 8, 10, 12, 15, 20, 30),
    confirm_prob_grid: Iterable[float] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85),
    value_price_grid: Iterable[float] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    value_prob_grid: Iterable[float] = (0.55, 0.60, 0.65, 0.70, 0.75),
    ev_grid: Iterable[float] = (0.00, 0.01, 0.02, 0.03, 0.05),
    stake_usdc: float = 10.0,
    min_trades: int = 20,
) -> pd.DataFrame:
    """Grid-search thresholds and return sorted performance rows."""
    rows: list[dict[str, float | int | str]] = []
    for bps, prob, ev in product(confirm_bps_grid, confirm_prob_grid, ev_grid):
        thresholds = StrategyThresholds(
            confirm_min_abs_bps=bps,
            confirm_min_model_prob=prob,
            min_ev_per_share=ev,
            # Disable value bucket for this grid row.
            value_max_market_price=-1,
            value_min_model_prob=1.1,
        )
        trades = simulate_strategy(frame, thresholds, stake_usdc=stake_usdc)
        _append_grid_row(
            rows,
            trades,
            bucket="CONFIRMATION",
            min_trades=min_trades,
            params={"confirm_bps": bps, "confirm_prob": prob, "min_ev": ev},
        )

    for price, prob, ev in product(value_price_grid, value_prob_grid, ev_grid):
        thresholds = StrategyThresholds(
            value_max_market_price=price,
            value_min_model_prob=prob,
            min_ev_per_share=ev,
            # Disable confirmation bucket for this grid row.
            confirm_min_abs_bps=1e9,
            confirm_min_model_prob=1.1,
        )
        trades = simulate_strategy(frame, thresholds, stake_usdc=stake_usdc)
        _append_grid_row(
            rows,
            trades,
            bucket="VALUE_MISMATCH",
            min_trades=min_trades,
            params={"value_max_price": price, "value_prob": prob, "min_ev": ev},
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["roi", "trades"], ascending=[False, False]).reset_index(drop=True)


def _append_grid_row(
    rows: list[dict[str, float | int | str]],
    trades: pd.DataFrame,
    *,
    bucket: str,
    min_trades: int,
    params: dict[str, float],
) -> None:
    if trades.empty or len(trades) < min_trades:
        return
    wins = float(trades["won"].mean())
    stake = float(trades["stake_usdc"].sum())
    pnl = float(trades["pnl_usdc"].sum())
    rows.append(
        {
            "bucket": bucket,
            "trades": int(len(trades)),
            "win_rate": wins,
            "avg_market_price": float(trades["market_price"].mean()),
            "avg_model_prob": float(trades["chosen_prob"].mean()),
            "avg_ev_per_share": float(trades["expected_value_per_share"].mean()),
            "avg_abs_score_bps": float(trades["abs_score_bps"].mean()),
            "pnl_usdc": pnl,
            "stake_usdc": stake,
            "roi": pnl / stake if stake else np.nan,
            **params,
        }
    )
