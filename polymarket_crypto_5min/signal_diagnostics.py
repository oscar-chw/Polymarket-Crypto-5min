"""Point-in-time signal diagnostics for BTC five-minute markets.

These diagnostics measure association between the raw heuristic probability and
the settled UP/DOWN label.  They deliberately do not reuse test outcomes for
calibration and they are not execution or PnL evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .features import build_training_frame

DEFAULT_HORIZONS_SECONDS = (300, 240, 180, 120, 90, 60, 45, 30, 15)


def build_horizon_diagnostics(
    markets: pd.DataFrame,
    candles: pd.DataFrame,
    *,
    horizons_seconds: Iterable[int] = DEFAULT_HORIZONS_SECONDS,
    fold_market_count: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return horizon rows, full chronological-fold ICs, summary, and metadata."""
    if fold_market_count < 2:
        raise ValueError("fold_market_count must be at least two")
    detail_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for raw_horizon in horizons_seconds:
        horizon = int(raw_horizon)
        if horizon <= 0:
            raise ValueError("horizons must be positive seconds")
        frame = build_training_frame(
            markets,
            candles,
            None,
            snapshot_seconds_before_close=horizon,
            require_resolved_outcome=True,
            allow_gamma_prices=False,
        )
        detail = _detail_rows(frame, horizon=horizon)
        detail_frames.append(detail)
        folds = chronological_fold_ic(detail, fold_market_count=fold_market_count)
        fold_frames.append(folds)
        summaries.append(_summarize_horizon(detail, folds, horizon=horizon))

    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    folds = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    summary = pd.DataFrame(summaries).sort_values("horizon_seconds", ascending=False).reset_index(drop=True)
    metadata = _summary_metadata(summary, details, fold_market_count=fold_market_count)
    return details, folds, summary, metadata


def chronological_fold_ic(detail: pd.DataFrame, *, fold_market_count: int = 100) -> pd.DataFrame:
    """Compute IC only on complete, non-overlapping chronological market blocks."""
    ordered = detail.sort_values(["end_dt", "condition_id"]).reset_index(drop=True)
    complete = len(ordered) // fold_market_count
    records: list[dict[str, Any]] = []
    for fold_index in range(complete):
        block = ordered.iloc[fold_index * fold_market_count : (fold_index + 1) * fold_market_count]
        y = block["direction_up"].astype(float)
        p = block["model_prob_up"].astype(float)
        records.append(
            {
                "horizon_seconds": int(block["horizon_seconds"].iloc[0]),
                "fold": fold_index + 1,
                "n_markets": int(len(block)),
                "start_utc": block["end_dt"].min(),
                "end_utc": block["end_dt"].max(),
                "pearson_ic_prob_vs_direction": _corr(p, y, method="pearson"),
                "rank_ic_prob_vs_direction": _corr(p, y, method="spearman"),
            }
        )
    return pd.DataFrame(records)


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas and non-finite values to strict-JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return value


def _detail_rows(frame: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
    columns = [
        "condition_id",
        "market_id",
        "question",
        "end_dt",
        "score_bps",
        "momentum_1m_bps",
        "momentum_3m_bps",
        "model_prob_up",
        "realized_direction",
        "realized_direction_source",
        "feature_available_at",
        "feature_availability_passed",
    ]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"horizon frame missing columns: {sorted(missing)}")
    detail = frame[columns].copy()
    detail["horizon_seconds"] = int(horizon)
    detail["direction_up"] = detail["realized_direction"].eq("UP").astype(int)
    detail["pred_direction"] = np.where(detail["model_prob_up"].ge(0.5), "UP", "DOWN")
    detail["correct"] = detail["pred_direction"].eq(detail["realized_direction"])
    detail["end_dt"] = pd.to_datetime(detail["end_dt"], utc=True, errors="coerce")
    detail["feature_available_at"] = pd.to_datetime(detail["feature_available_at"], utc=True, errors="coerce")
    if not detail["feature_availability_passed"].fillna(False).all():
        raise ValueError("horizon diagnostics contain unavailable feature observations")
    return detail.sort_values(["end_dt", "condition_id"]).reset_index(drop=True)


def _summarize_horizon(detail: pd.DataFrame, folds: pd.DataFrame, *, horizon: int) -> dict[str, Any]:
    y = detail["direction_up"].astype(float)
    p = detail["model_prob_up"].astype(float)
    score = detail["score_bps"].astype(float)
    fold_ic = folds.get("pearson_ic_prob_vs_direction", pd.Series(dtype=float)).dropna().astype(float)
    fold_rank = folds.get("rank_ic_prob_vs_direction", pd.Series(dtype=float)).dropna().astype(float)
    icir = _mean_over_sample_std(fold_ic)
    rank_icir = _mean_over_sample_std(fold_rank)
    return {
        "horizon_seconds": int(horizon),
        "n_markets_resolved": int(len(detail)),
        "start_utc": detail["end_dt"].min(),
        "end_utc": detail["end_dt"].max(),
        "pearson_ic_prob_vs_direction": _corr(p, y, method="pearson"),
        "rank_ic_prob_vs_direction": _corr(p, y, method="spearman"),
        "score_bps_pearson_ic": _corr(score, y, method="pearson"),
        "score_bps_rank_ic": _corr(score, y, method="spearman"),
        "icir_100market_folds": icir,
        "rank_icir_100market_folds": rank_icir,
        "ic_tstat_100market_folds": icir * math.sqrt(len(fold_ic)) if _finite(icir) else math.nan,
        "fold_count": int(len(fold_ic)),
        "accuracy": float(detail["correct"].mean()),
        "up_rate": float(y.mean()),
        "mean_abs_score_bps": float(score.abs().mean()),
        "feature_availability_passed": bool(detail["feature_availability_passed"].all()),
    }


def _summary_metadata(summary: pd.DataFrame, details: pd.DataFrame, *, fold_market_count: int) -> dict[str, Any]:
    valid = summary.dropna(subset=["pearson_ic_prob_vs_direction"]).copy()
    positive = valid.loc[valid["pearson_ic_prob_vs_direction"].gt(0)].copy()
    half_life = math.nan
    fit_r2 = math.nan
    if len(positive) >= 3:
        x = positive["horizon_seconds"].to_numpy(float)
        y = np.log(positive["pearson_ic_prob_vs_direction"].to_numpy(float))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        denominator = float(((y - y.mean()) ** 2).sum())
        fit_r2 = 1.0 - float(((y - fitted) ** 2).sum()) / denominator if denominator > 0 else math.nan
        if slope < 0:
            half_life = math.log(2.0) / -float(slope)
    best_ic = valid.loc[valid["pearson_ic_prob_vs_direction"].idxmax()] if not valid.empty else None
    best_rank = valid.loc[valid["rank_ic_prob_vs_direction"].idxmax()] if not valid.empty else None
    useful = None
    if best_ic is not None:
        threshold = float(best_ic["pearson_ic_prob_vs_direction"]) / 2.0
        eligible = valid.loc[valid["pearson_ic_prob_vs_direction"].ge(threshold), "horizon_seconds"]
        useful = int(eligible.max()) if not eligible.empty else None
    default = summary.loc[summary["horizon_seconds"].eq(45)]
    return {
        "schema_version": "btc_signal_horizon_diagnostics_v2_close_availability",
        "status": "point_in_time_raw_signal_diagnostic_not_execution_or_pnl",
        "horizons_seconds": [int(value) for value in summary["horizon_seconds"]],
        "resolved_markets_by_horizon": int(summary["n_markets_resolved"].min()) if not summary.empty else 0,
        "date_start_utc": details["end_dt"].min() if not details.empty else None,
        "date_end_utc": details["end_dt"].max() if not details.empty else None,
        "fold_market_count": int(fold_market_count),
        "best_ic_horizon_seconds": int(best_ic["horizon_seconds"]) if best_ic is not None else None,
        "best_rank_ic_horizon_seconds": int(best_rank["horizon_seconds"]) if best_rank is not None else None,
        "useful_life_seconds_half_max_ic": useful,
        "exponential_decay_half_life_seconds_if_valid": half_life,
        "exponential_decay_fit_r2": fit_r2,
        "default_45s": default.iloc[0].to_dict() if len(default) == 1 else None,
        "timestamp_semantics": "Binance final candle values indexed at close availability, never candle open",
        "feature_availability_gate_passed": bool(details["feature_availability_passed"].all())
        if not details.empty
        else False,
        "calibration_scope": "raw heuristic probability; prior-only empirical calibration is evaluated separately inside each trading fold",
        "interpretation": (
            "Forecast-horizon association only. It is not calendar decay, executable edge, "
            "paid income, or selected-policy PnL."
        ),
    }


def _corr(left: pd.Series, right: pd.Series, *, method: str) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return math.nan
    if method == "pearson":
        return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="pearson"))
    if method == "spearman":
        # Spearman correlation is Pearson correlation of average ranks.  Doing
        # the transform explicitly avoids pandas' optional SciPy dependency,
        # so the locked package can regenerate diagnostics without an
        # undeclared environment dependency.
        ranked = pair.rank(method="average")
        return float(ranked.iloc[:, 0].corr(ranked.iloc[:, 1], method="pearson"))
    raise ValueError(f"unsupported correlation method: {method}")


def _mean_over_sample_std(values: pd.Series) -> float:
    if len(values) < 2:
        return math.nan
    std = float(values.std(ddof=1))
    return float(values.mean()) / std if std > 0 else math.nan


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "DEFAULT_HORIZONS_SECONDS",
    "build_horizon_diagnostics",
    "chronological_fold_ic",
    "json_safe",
]
