"""Feature engineering for five-minute direction markets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CRYPTO_TAKER_FEE_RATE
from .utils import parse_jsonish, sigmoid, to_float


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
    for col in ["start_ts", "end_ts", "gamma_up_price", "gamma_down_price", "best_bid", "best_ask", "last_trade_price"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def load_candles(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "ts" not in frame.columns:
        raise ValueError("candle CSV must have a ts column")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)


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
) -> pd.DataFrame:
    """Build one point-in-time row per market for threshold research.

    By default, the function only uses CLOB price history for historical market
    prices. Set ``allow_gamma_prices=True`` for exploratory work only; Gamma
    prices on closed markets may be post-resolution and therefore look-ahead.
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

    rows["btc_start"] = asof_close(candles, rows["start_dt"])
    rows["btc_snapshot"] = asof_close(candles, rows["snapshot_dt"])
    rows["btc_end"] = asof_close(candles, rows["end_dt"])
    rows["btc_1m_ago"] = asof_close(candles, rows["snapshot_dt"] - pd.to_timedelta(60, unit="s"))
    rows["btc_3m_ago"] = asof_close(candles, rows["snapshot_dt"] - pd.to_timedelta(180, unit="s"))

    rows["score_bps"] = (rows["btc_snapshot"] / rows["btc_start"] - 1.0) * 10_000
    rows["abs_score_bps"] = rows["score_bps"].abs()
    rows["momentum_1m_bps"] = (rows["btc_snapshot"] / rows["btc_1m_ago"] - 1.0) * 10_000
    rows["momentum_3m_bps"] = (rows["btc_snapshot"] / rows["btc_3m_ago"] - 1.0) * 10_000
    rows["realized_up"] = rows["btc_end"] > rows["btc_start"]
    rows["realized_direction"] = np.where(rows["realized_up"], "UP", "DOWN")

    rows["market_up_price"] = market_price_asof(poly_price_history, rows, asset_col="up_asset_id")
    if "market_down_price" not in rows:
        rows["market_down_price"] = np.nan
    down_hist = market_price_asof(poly_price_history, rows, asset_col="down_asset_id")
    rows["market_down_price"] = rows["market_down_price"].where(rows["market_down_price"].notna(), down_hist)

    if allow_gamma_prices:
        rows["market_up_price"] = rows["market_up_price"].where(rows["market_up_price"].notna(), rows.get("gamma_up_price"))
        rows["market_down_price"] = rows["market_down_price"].where(rows["market_down_price"].notna(), rows.get("gamma_down_price"))
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
    rows["chosen_prob"] = np.where(rows["chosen_direction"].eq("UP"), rows["model_prob_up"], 1.0 - rows["model_prob_up"])
    rows["market_price"] = np.where(rows["chosen_direction"].eq("UP"), rows["market_up_price"], rows["market_down_price"])
    rows["fee_per_share"] = taker_fee_per_share(rows["market_price"], fee_rate=fee_rate)
    rows["expected_value_per_share"] = rows["chosen_prob"] - rows["market_price"] - rows["fee_per_share"]
    rows["won"] = rows["chosen_direction"].eq(rows["realized_direction"])
    rows["profit_per_share"] = np.where(
        rows["won"],
        1.0 - rows["market_price"] - rows["fee_per_share"],
        -rows["market_price"] - rows["fee_per_share"],
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
    lookup = pd.DataFrame({"_row": np.arange(len(when)), "ts": pd.to_datetime(when, utc=True)})
    lookup = lookup.sort_values("ts")
    merged = pd.merge_asof(lookup, candles[["ts", "close"]].sort_values("ts"), on="ts", direction="backward")
    return merged.sort_values("_row")["close"].reset_index(drop=True)


def market_price_asof(poly_price_history: pd.DataFrame | None, markets: pd.DataFrame, *, asset_col: str) -> pd.Series:
    if poly_price_history is None or poly_price_history.empty or asset_col not in markets.columns:
        return pd.Series(np.nan, index=markets.index)
    hist = poly_price_history.copy()
    hist["condition_id"] = hist["condition_id"].astype(str)
    hist["asset_id"] = hist["asset_id"].astype(str)
    hist["ts"] = pd.to_datetime(hist["ts"], utc=True, errors="coerce")
    hist["p"] = pd.to_numeric(hist["p"], errors="coerce")
    result = pd.Series(np.nan, index=markets.index, dtype="float64")
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
            lookup = pd.DataFrame({"_index": list(idx), "ts": markets.loc[idx, "snapshot_dt"]}).sort_values("ts")
            merged = pd.merge_asof(lookup, h_asset[["ts", "p"]], on="ts", direction="backward")
            result.loc[merged["_index"].to_numpy()] = merged["p"].to_numpy()
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

    This is intentionally simple and should be calibrated by backtest. The most
    important driver is distance from the market's start BTC price. Momentum is a
    small secondary term.
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


def shares_for_stake(stake_usdc: float, price: float, fee_rate: float = DEFAULT_CRYPTO_TAKER_FEE_RATE) -> float:
    fee = taker_fee_per_share(price, fee_rate=fee_rate)
    all_in = price + fee
    if not all_in or math.isnan(all_in) or all_in <= 0:
        return 0.0
    return stake_usdc / all_in
