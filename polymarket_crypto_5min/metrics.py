"""Performance metrics for strategy backtests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MetricConfig:
    """Assumptions used for performance statistics.

    ``initial_capital`` is needed for equity/drawdown percentages and daily
    return calculations. ``periods_per_year`` defaults to 365 because crypto
    trades every day.
    """

    initial_capital: float = 1_000.0
    periods_per_year: int = 365


def equity_curve(trades: pd.DataFrame, *, initial_capital: float = 1_000.0) -> pd.DataFrame:
    """Return a realized equity curve from per-trade PnL.

    The curve is anchored to an explicit starting row at ``initial_capital`` so a
    first-trade loss is counted as drawdown. This is realized equity only: it
    does not mark open positions to market between entry and resolution.
    """
    columns = [
        "time",
        "pnl_usdc",
        "cum_pnl_usdc",
        "equity",
        "running_peak",
        "drawdown_usdc",
        "drawdown_pct",
        "is_initial_row",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    frame = trades.copy()
    if "end_dt" in frame.columns:
        frame["time"] = pd.to_datetime(frame["end_dt"], utc=True, errors="coerce")
    elif "snapshot_dt" in frame.columns:
        frame["time"] = pd.to_datetime(frame["snapshot_dt"], utc=True, errors="coerce")
    else:
        frame["time"] = pd.RangeIndex(1, len(frame) + 1)
    frame["pnl_usdc"] = pd.to_numeric(frame["pnl_usdc"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("time").reset_index(drop=True)

    curve = frame[["time", "pnl_usdc"]].copy()
    initial_time = _initial_time(curve["time"])
    initial_row = pd.DataFrame(
        {
            "time": [initial_time],
            "pnl_usdc": [0.0],
            "is_initial_row": [True],
        }
    )
    curve["is_initial_row"] = False
    curve = pd.concat([initial_row, curve], ignore_index=True).sort_values("time", kind="stable").reset_index(drop=True)
    curve["cum_pnl_usdc"] = curve["pnl_usdc"].cumsum()
    curve["equity"] = initial_capital + curve["cum_pnl_usdc"]
    curve["running_peak"] = curve["equity"].cummax()
    # The initial row ensures running_peak starts at initial_capital, not after a
    # first loss. This fixes the common MDD understatement bug.
    curve["drawdown_usdc"] = curve["equity"] - curve["running_peak"]
    curve["drawdown_pct"] = curve["drawdown_usdc"] / curve["running_peak"].replace(0, np.nan)
    return curve[columns]


def _initial_time(times: pd.Series) -> Any:
    if len(times) == 0:
        return 0
    first = times.iloc[0]
    if isinstance(first, pd.Timestamp):
        return first - pd.Timedelta(microseconds=1)
    if pd.api.types.is_datetime64_any_dtype(times):
        first_ts = pd.to_datetime(first, utc=True)
        return first_ts - pd.Timedelta(microseconds=1)
    try:
        return first - 1
    except TypeError:
        return -1


def performance_metrics(
    trades: pd.DataFrame,
    *,
    initial_capital: float = 1_000.0,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """Compute overall and per-bucket Sharpe/MDD style metrics.

    Metrics include:

    - ``trade_sharpe``: mean per-trade return divided by per-trade return std.
      This is not annualized and is often the safest first look for 5-minute bets.
    - ``daily_sharpe``: Sharpe on daily realized equity returns, annualized by
      sqrt(365).
    - ``max_drawdown_pct`` / ``max_drawdown_usdc``: worst peak-to-trough loss on
      the realized equity curve anchored at initial capital.
    """
    rows: list[dict[str, float | int | str]] = []
    if trades.empty:
        return pd.DataFrame(rows)
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", trades)]
    if "bucket" in trades.columns:
        groups.extend((str(bucket), group) for bucket, group in trades.groupby("bucket", dropna=False))
    for label, group in groups:
        rows.append(
            _metrics_for_group(label, group, initial_capital=initial_capital, periods_per_year=periods_per_year)
        )
    return pd.DataFrame(rows)


def _metrics_for_group(
    label: str,
    trades: pd.DataFrame,
    *,
    initial_capital: float,
    periods_per_year: int,
) -> dict[str, float | int | str]:
    pnl = pd.to_numeric(trades.get("pnl_usdc", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    stake = pd.to_numeric(trades.get("stake_usdc", pd.Series(dtype=float)), errors="coerce").replace(0, np.nan)
    returns = pd.to_numeric(trades.get("return_on_stake", pnl / stake), errors="coerce").dropna()
    curve = equity_curve(trades, initial_capital=initial_capital)
    curve_no_initial = curve[~curve.get("is_initial_row", pd.Series(False, index=curve.index)).astype(bool)]
    daily_returns = _daily_equity_returns(curve, initial_capital=initial_capital)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    total_stake = float(
        pd.to_numeric(trades.get("stake_usdc", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
    )
    total_pnl = float(pnl.sum())
    max_dd_usdc = float(curve["drawdown_usdc"].min()) if not curve.empty else math.nan
    max_dd_pct = float(curve["drawdown_pct"].min()) if not curve.empty else math.nan
    return {
        "bucket": label,
        "trades": int(len(trades)),
        "wins": int(pd.to_numeric(trades.get("won", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        if "won" in trades.columns
        else math.nan,
        "win_rate": _safe_mean(pd.to_numeric(trades.get("won", pd.Series(dtype=float)), errors="coerce")),
        "total_pnl_usdc": total_pnl,
        "total_stake_usdc": total_stake,
        "roi_on_stake": total_pnl / total_stake if total_stake else math.nan,
        "return_on_initial_capital": total_pnl / initial_capital if initial_capital else math.nan,
        "avg_trade_return": _safe_mean(returns),
        "trade_return_std": _safe_std(returns),
        "trade_sharpe": _sharpe(returns, annualization=None),
        "daily_sharpe": _sharpe(daily_returns, annualization=math.sqrt(periods_per_year)),
        "daily_sortino": _sortino(daily_returns, annualization=math.sqrt(periods_per_year)),
        "max_drawdown_usdc": max_dd_usdc,
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": gross_profit / gross_loss if gross_loss else math.inf,
        "gross_profit_usdc": gross_profit,
        "gross_loss_usdc": gross_loss,
        "avg_market_price": _safe_mean(
            pd.to_numeric(trades.get("market_price", pd.Series(dtype=float)), errors="coerce")
        ),
        "avg_ev_per_share": _safe_mean(
            pd.to_numeric(trades.get("expected_value_per_share", pd.Series(dtype=float)), errors="coerce")
        ),
        "realized_equity_curve_only": True,
        "mark_to_market_drawdown_available": False,
        "min_realized_equity": float(curve_no_initial["equity"].min()) if not curve_no_initial.empty else math.nan,
    }


def _daily_equity_returns(curve: pd.DataFrame, *, initial_capital: float) -> pd.Series:
    if curve.empty or "time" not in curve.columns:
        return pd.Series(dtype=float)
    curve = curve[~curve.get("is_initial_row", pd.Series(False, index=curve.index)).astype(bool)].copy()
    if curve.empty or not pd.api.types.is_datetime64_any_dtype(curve["time"]):
        return pd.Series(dtype=float)
    daily_pnl = curve.set_index("time")["pnl_usdc"].resample("1D").sum()
    equity_start = initial_capital + daily_pnl.cumsum().shift(1).fillna(0.0)
    return (daily_pnl / equity_start.replace(0, np.nan)).dropna()


def _sharpe(returns: pd.Series, *, annualization: float | None) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if len(returns) < 2:
        return math.nan
    std = returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return math.nan
    value = returns.mean() / std
    return float(value * annualization) if annualization is not None else float(value)


def _sortino(returns: pd.Series, *, annualization: float | None) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    downside = returns[returns < 0]
    if len(returns) < 2 or len(downside) < 1:
        return math.nan
    downside_std = downside.std(ddof=1) if len(downside) > 1 else abs(float(downside.iloc[0]))
    if downside_std == 0 or math.isnan(downside_std):
        return math.nan
    value = returns.mean() / downside_std
    return float(value * annualization) if annualization is not None else float(value)


def _safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else math.nan


def _safe_std(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else math.nan
