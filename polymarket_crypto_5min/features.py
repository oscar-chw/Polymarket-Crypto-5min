"""Feature engineering for five-minute direction markets."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CRYPTO_TAKER_FEE_RATE
from .resolution import append_resolved_outcomes
from .utils import parse_jsonish, sigmoid, to_float

LOGGER = logging.getLogger(__name__)

# One-minute candles indexed at close availability are at most 60 s old when
# none is missing. An older join means a gap in the candle file, and the
# feature would measure drift over the gap rather than over the market.
DEFAULT_MAX_BTC_CANDLE_AGE_SECONDS = 60.0
BTC_FEATURE_LOOKUPS = ("start", "snapshot", "1m_ago", "3m_ago")
# The backtest has no historical order book, so it cannot fill at the ask that
# ``live_signal.py`` pays. It fills at the last CLOB/trade print inside the
# market window, at or before the decision, and no older than the configured
# age. That is optimistic against live fills by at least half the spread.
ENTRY_PRICE_RULE = "last_print_in_window"
DEFAULT_MAX_ENTRY_PRICE_AGE_SECONDS = 60.0


def load_csv(path: str | Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in parse_dates or []:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    return frame


def load_markets(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in ["outcomes", "outcome_prices", "clob_token_ids"]:
        if col in frame.columns:
            frame[col] = frame[col].apply(lambda value: parse_jsonish(value, []))
    for col in [
        "start_ts",
        "end_ts",
        "duration_seconds",
        "gamma_up_price",
        "gamma_down_price",
        "best_bid",
        "best_ask",
        "last_trade_price",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_candles(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "ts" not in frame.columns:
        raise ValueError("candle CSV must have a ts column")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    semantics = frame.get("timestamp_semantics")
    if semantics is None:
        # Files emitted by package versions before the availability fix used
        # Binance open time as ``ts`` while storing the interval's final close.
        # Detect only that exact legacy Binance schema and migrate it
        # conservatively.  Unidentified timestamp semantics fail closed.
        required_legacy = {"symbol", "interval", "open", "high", "low", "close", "volume"}
        if not required_legacy.issubset(frame.columns):
            raise ValueError(
                "candle timestamp semantics unavailable; provide close-availability "
                "timestamps or the legacy Binance symbol/interval schema"
            )
        intervals = frame["interval"].dropna().astype(str).unique().tolist()
        if len(intervals) != 1:
            raise ValueError("legacy candle file must contain exactly one interval")
        delta = _candle_interval_delta(intervals[0])
        frame["open_ts"] = frame["ts"]
        frame["close_ts"] = frame["open_ts"] + delta - pd.to_timedelta(1, unit="ms")
        frame["available_at"] = frame["open_ts"] + delta
        frame["ts"] = frame["available_at"]
        frame["timestamp_semantics"] = "legacy_binance_open_time_shifted_to_close_available_at"
    else:
        allowed = {
            "close_available_at",
            "legacy_binance_open_time_shifted_to_close_available_at",
        }
        observed = set(semantics.dropna().astype(str).unique())
        if not observed or not observed.issubset(allowed):
            raise ValueError(f"unsupported candle timestamp semantics: {sorted(observed)}")
        if "available_at" in frame.columns:
            frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
            if not frame["available_at"].equals(frame["ts"]):
                raise ValueError("candle ts must equal available_at")
        else:
            frame["available_at"] = frame["ts"]
        for col in ("open_ts", "close_ts"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)


def _candle_interval_delta(interval: str) -> pd.Timedelta:
    token = str(interval).strip().lower()
    if len(token) < 2 or not token[:-1].isdigit():
        raise ValueError(f"unsupported candle interval: {interval}")
    amount = int(token[:-1])
    units = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    unit = units.get(token[-1])
    if unit is None or amount <= 0:
        raise ValueError(f"unsupported candle interval: {interval}")
    return pd.to_timedelta(amount, unit=unit)


def load_poly_prices(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    elif "t" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["t"], unit="s", utc=True, errors="coerce")
    else:
        raise ValueError("Polymarket price CSV must have ts or t column")
    frame["p"] = pd.to_numeric(frame["p"], errors="coerce")
    return frame.dropna(subset=["condition_id", "ts", "p"]).sort_values(["condition_id", "ts"]).reset_index(drop=True)


def build_training_frame(
    markets: pd.DataFrame,
    btc_candles: pd.DataFrame,
    poly_price_history: pd.DataFrame | None = None,
    *,
    snapshot_seconds_before_close: int = 45,
    probability_noise_bps: float = 8.0,
    fee_rate: float = DEFAULT_CRYPTO_TAKER_FEE_RATE,
    allow_gamma_prices: bool = False,
    require_resolved_outcome: bool = True,
    allow_external_btc_outcome_fallback: bool = False,
    max_entry_price_age_seconds: float = DEFAULT_MAX_ENTRY_PRICE_AGE_SECONDS,
    max_btc_candle_age_seconds: float = DEFAULT_MAX_BTC_CANDLE_AGE_SECONDS,
) -> pd.DataFrame:
    """Build one point-in-time row per market for threshold research.

    Important correctness rule: the payout label comes from settled Polymarket
    outcome data, not from external BTC candles. BTC candles are features only.
    Set ``allow_external_btc_outcome_fallback=True`` only for diagnostics.

    By default, market entry prices are read from point-in-time CLOB/trade price
    history. Set ``allow_gamma_prices=True`` only for exploratory debugging;
    closed-market Gamma prices may contain post-resolution information.

    Entry prices follow ``ENTRY_PRICE_RULE``: the last print with
    ``start_dt <= ts <= snapshot_dt`` and age at most
    ``max_entry_price_age_seconds``. Rows with no such print get no price.

    Rows whose start, snapshot, 1m-ago or 3m-ago BTC candle is missing or
    older than ``max_btc_candle_age_seconds`` are dropped; the count is logged
    and kept in ``frame.attrs["btc_stale_rows_dropped"]``.
    """
    required = {"condition_id", "start_ts", "end_ts"}
    missing = required - set(markets.columns)
    if missing:
        raise ValueError(f"markets missing required columns: {sorted(missing)}")
    candles = _prep_candles(btc_candles)
    rows = markets.copy()
    rows["start_ts"] = pd.to_numeric(rows["start_ts"], errors="coerce")
    rows["end_ts"] = pd.to_numeric(rows["end_ts"], errors="coerce")
    rows = rows.dropna(subset=["condition_id", "start_ts", "end_ts"]).reset_index(drop=True).copy()
    rows["start_dt"] = pd.to_datetime(rows["start_ts"], unit="s", utc=True)
    rows["end_dt"] = pd.to_datetime(rows["end_ts"], unit="s", utc=True)
    rows["snapshot_ts"] = (rows["end_ts"] - snapshot_seconds_before_close).clip(lower=rows["start_ts"])
    rows["snapshot_dt"] = pd.to_datetime(rows["snapshot_ts"], unit="s", utc=True)
    rows["seconds_left"] = rows["end_ts"] - rows["snapshot_ts"]

    lookups = {
        "start": rows["start_dt"],
        "snapshot": rows["snapshot_dt"],
        "end": rows["end_dt"],
        "1m_ago": rows["snapshot_dt"] - pd.to_timedelta(60, unit="s"),
        "3m_ago": rows["snapshot_dt"] - pd.to_timedelta(180, unit="s"),
    }
    for name, when in lookups.items():
        observed = asof_candle(candles, when)
        rows[f"btc_{name}"] = observed["close"]
        rows[f"btc_{name}_available_at"] = observed["available_at"]
        rows[f"btc_{name}_age_seconds"] = (
            pd.to_datetime(when, utc=True).reset_index(drop=True) - pd.to_datetime(observed["available_at"], utc=True)
        ).dt.total_seconds()

    fresh = pd.Series(True, index=rows.index)
    for name in BTC_FEATURE_LOOKUPS:
        fresh &= rows[f"btc_{name}_age_seconds"].le(max_btc_candle_age_seconds)
    stale_rows = int((~fresh).sum())
    if stale_rows:
        LOGGER.warning(
            "Dropping %d market rows with BTC candles missing or older than %.0f s",
            stale_rows,
            max_btc_candle_age_seconds,
        )
    rows = rows[fresh].reset_index(drop=True)

    feature_availability = rows[
        [
            "btc_start_available_at",
            "btc_snapshot_available_at",
            "btc_1m_ago_available_at",
            "btc_3m_ago_available_at",
        ]
    ].max(axis=1)
    rows["feature_available_at"] = feature_availability
    rows["feature_availability_passed"] = feature_availability.le(rows["snapshot_dt"])
    invalid = rows["feature_available_at"].notna() & ~rows["feature_availability_passed"]
    if invalid.any():
        raise ValueError("BTC feature row uses a candle unavailable at the decision timestamp")

    rows["score_bps"] = (rows["btc_snapshot"] / rows["btc_start"] - 1.0) * 10_000
    rows["abs_score_bps"] = rows["score_bps"].abs()
    rows["momentum_1m_bps"] = (rows["btc_snapshot"] / rows["btc_1m_ago"] - 1.0) * 10_000
    rows["momentum_3m_bps"] = (rows["btc_snapshot"] / rows["btc_3m_ago"] - 1.0) * 10_000
    rows["btc_external_direction"] = np.where(rows["btc_end"] > rows["btc_start"], "UP", "DOWN")

    rows = append_resolved_outcomes(
        rows,
        require_resolved=require_resolved_outcome,
        allow_external_btc_fallback=allow_external_btc_outcome_fallback,
        external_direction_col="btc_external_direction",
    )
    rows.attrs["btc_stale_rows_dropped"] = stale_rows
    if rows.empty:
        return rows

    up_hist = market_price_asof(
        poly_price_history, rows, asset_col="up_asset_id", max_age_seconds=max_entry_price_age_seconds
    )
    rows["market_up_price"] = up_hist["p"]
    rows["market_up_price_ts"] = up_hist["ts"]
    if "market_down_price" not in rows:
        rows["market_down_price"] = np.nan
    down_hist = market_price_asof(
        poly_price_history, rows, asset_col="down_asset_id", max_age_seconds=max_entry_price_age_seconds
    )
    rows["market_down_price"] = rows["market_down_price"].where(rows["market_down_price"].notna(), down_hist["p"])
    rows["market_down_price_ts"] = down_hist["ts"]
    rows["entry_price_rule"] = ENTRY_PRICE_RULE

    if allow_gamma_prices:
        rows["market_up_price"] = rows["market_up_price"].where(
            rows["market_up_price"].notna(), rows.get("gamma_up_price")
        )
        rows["market_down_price"] = rows["market_down_price"].where(
            rows["market_down_price"].notna(), rows.get("gamma_down_price")
        )
    rows["market_down_price"] = rows["market_down_price"].where(
        rows["market_down_price"].notna(), 1.0 - rows["market_up_price"]
    )

    rows["model_prob_up"] = model_probability_up(
        score_bps=rows["score_bps"],
        momentum_1m_bps=rows["momentum_1m_bps"],
        momentum_3m_bps=rows["momentum_3m_bps"],
        seconds_left=rows["seconds_left"],
        noise_bps=probability_noise_bps,
    )
    rows["chosen_direction"] = np.where(rows["model_prob_up"] >= 0.5, "UP", "DOWN")
    rows["chosen_prob"] = np.where(
        rows["chosen_direction"].eq("UP"), rows["model_prob_up"], 1.0 - rows["model_prob_up"]
    )
    rows["market_price"] = np.where(
        rows["chosen_direction"].eq("UP"), rows["market_up_price"], rows["market_down_price"]
    )
    rows["fee_per_share"] = taker_fee_per_share(rows["market_price"], fee_rate=fee_rate)
    rows["expected_value_per_share"] = rows["chosen_prob"] - rows["market_price"] - rows["fee_per_share"]
    rows["won"] = rows["chosen_direction"].eq(rows["realized_direction"])
    rows["profit_per_share"] = profit_per_share(
        price=rows["market_price"],
        won=rows["won"],
        fee_rate=fee_rate,
    )
    return rows.reset_index(drop=True)


def _prep_candles(candles: pd.DataFrame) -> pd.DataFrame:
    frame = candles.copy()
    if "ts" not in frame:
        raise ValueError("candles must include ts column")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)


def asof_close(candles: pd.DataFrame, when: pd.Series) -> pd.Series:
    return asof_candle(candles, when)["close"]


def asof_candle(candles: pd.DataFrame, when: pd.Series) -> pd.DataFrame:
    lookup = pd.DataFrame({"_row": np.arange(len(when)), "ts": _datetime64ns_utc(pd.to_datetime(when, utc=True))})
    lookup = lookup.sort_values("ts")
    right = candles[["ts", "close"]].rename(columns={"ts": "available_at"}).copy()
    right["available_at"] = _datetime64ns_utc(pd.to_datetime(right["available_at"], utc=True, errors="coerce"))
    merged = pd.merge_asof(
        lookup,
        right.sort_values("available_at"),
        left_on="ts",
        right_on="available_at",
        direction="backward",
    )
    merged = merged.sort_values("_row").reset_index(drop=True)
    return pd.DataFrame({"close": merged["close"], "available_at": merged["available_at"]})


def _datetime64ns_utc(values: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Normalize timestamps for ``merge_asof`` across pandas/numpy backends."""
    series = pd.Series(values)
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = pd.to_datetime(series, utc=True, errors="coerce")
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    return series.astype("datetime64[ns, UTC]")


def market_price_asof(
    poly_price_history: pd.DataFrame | None,
    markets: pd.DataFrame,
    *,
    asset_col: str,
    max_age_seconds: float = DEFAULT_MAX_ENTRY_PRICE_AGE_SECONDS,
) -> pd.DataFrame:
    """Return ``p`` and its print time ``ts`` under ``ENTRY_PRICE_RULE``.

    History is downloaded with padding before the window opens; without the
    window and age bounds a print from before the market existed could become
    the entry price.
    """
    result = pd.DataFrame(
        {"p": np.nan, "ts": pd.Series(pd.NaT, index=markets.index, dtype="datetime64[ns, UTC]")},
        index=markets.index,
    )
    if poly_price_history is None or poly_price_history.empty or asset_col not in markets.columns:
        return result
    hist = poly_price_history.copy()
    if not {"condition_id", "asset_id", "ts", "p"}.issubset(hist.columns):
        return result
    hist["condition_id"] = hist["condition_id"].astype(str)
    hist["asset_id"] = hist["asset_id"].astype(str)
    hist["ts"] = _datetime64ns_utc(pd.to_datetime(hist["ts"], utc=True, errors="coerce"))
    hist["p"] = pd.to_numeric(hist["p"], errors="coerce")
    hist = hist.dropna(subset=["condition_id", "asset_id", "ts", "p"])
    tolerance = pd.Timedelta(seconds=float(max_age_seconds))
    # Groupwise merge_asof is robust and easy to audit for this dataset size.
    for condition_id, group in markets.groupby(markets["condition_id"].astype(str)):
        h = hist[hist["condition_id"].eq(condition_id)]
        if h.empty:
            continue
        for asset_id, idx in group.groupby(group[asset_col].astype(str)).groups.items():
            if asset_id in {"nan", "None", ""}:
                continue
            h_asset = h[h["asset_id"].eq(asset_id)].sort_values("ts")
            if h_asset.empty:
                continue
            lookup = pd.DataFrame(
                {
                    "_index": list(idx),
                    "ts": _datetime64ns_utc(markets.loc[idx, "snapshot_dt"]).reset_index(drop=True),
                    "_window_start": _datetime64ns_utc(markets.loc[idx, "start_dt"]).reset_index(drop=True),
                }
            ).sort_values("ts")
            right = h_asset[["ts", "p"]].assign(price_ts=h_asset["ts"])
            merged = pd.merge_asof(lookup, right, on="ts", direction="backward", tolerance=tolerance)
            merged = merged[merged["price_ts"].ge(merged["_window_start"])]
            result.loc[merged["_index"].to_numpy(), "p"] = merged["p"].to_numpy()
            result.loc[merged["_index"].to_numpy(), "ts"] = merged["price_ts"].to_numpy()
    return result


def model_probability_up(
    *,
    score_bps: pd.Series,
    momentum_1m_bps: pd.Series,
    momentum_3m_bps: pd.Series,
    seconds_left: pd.Series,
    noise_bps: float = 8.0,
) -> pd.Series:
    """Heuristic probability that the final outcome resolves UP.

    This is intentionally simple and should be calibrated by walk-forward
    backtesting. The most important driver is distance from the market's start
    BTC price. Momentum is a small secondary term.
    """
    time_scale = np.sqrt(np.maximum(pd.to_numeric(seconds_left, errors="coerce"), 1.0) / 60.0)
    denom = np.maximum(noise_bps * time_scale, 1.0)
    z = (
        pd.to_numeric(score_bps, errors="coerce").fillna(0.0) / denom
        + 0.20 * pd.to_numeric(momentum_1m_bps, errors="coerce").fillna(0.0) / max(noise_bps, 1.0)
        + 0.08 * pd.to_numeric(momentum_3m_bps, errors="coerce").fillna(0.0) / max(noise_bps, 1.0)
    )
    return z.apply(lambda value: min(max(sigmoid(float(value)), 0.001), 0.999))


def taker_fee_per_share(price: Any, *, fee_rate: float = DEFAULT_CRYPTO_TAKER_FEE_RATE) -> Any:
    """Polymarket taker fee per share under ``fee_rate * p * (1 - p)``."""
    if isinstance(price, pd.Series):
        p = pd.to_numeric(price, errors="coerce")
        return fee_rate * p * (1.0 - p)
    p = to_float(price)
    if math.isnan(p):
        return math.nan
    return fee_rate * p * (1.0 - p)


def profit_per_share(price: Any, won: Any, *, fee_rate: float = DEFAULT_CRYPTO_TAKER_FEE_RATE) -> Any:
    p = pd.to_numeric(price, errors="coerce") if isinstance(price, pd.Series) else to_float(price)
    fee = taker_fee_per_share(p, fee_rate=fee_rate)
    if isinstance(price, pd.Series):
        won_series = (
            pd.Series(won, index=price.index).astype(bool) if not isinstance(won, pd.Series) else won.astype(bool)
        )
        return np.where(won_series, 1.0 - p - fee, -p - fee)
    return (1.0 - p - fee) if bool(won) else (-p - fee)


def shares_for_stake(stake_usdc: float, price: float, fee_rate: float = DEFAULT_CRYPTO_TAKER_FEE_RATE) -> float:
    fee = taker_fee_per_share(price, fee_rate=fee_rate)
    all_in = price + fee
    if not all_in or math.isnan(all_in) or all_in <= 0:
        return 0.0
    return stake_usdc / all_in
