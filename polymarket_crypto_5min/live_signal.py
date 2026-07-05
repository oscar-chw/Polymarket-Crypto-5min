"""Dry-run live signal generation for active Bitcoin five-minute markets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .clients import BinanceClient, ClobClient, GammaClient
from .config import DEFAULT_CRYPTO_TAKER_FEE_RATE
from .downloader import flatten_event_markets, is_bitcoin_5min_event
from .features import model_probability_up, taker_fee_per_share
from .utils import parse_dt, to_float


@dataclass(slots=True)
class LiveSignal:
    condition_id: str
    question: str
    end_iso: str | None
    seconds_left: float
    direction: str
    btc_start: float
    btc_now: float
    score_bps: float
    model_prob: float
    buy_price: float
    fee_per_share: float
    expected_value_per_share: float
    bucket: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_active_bitcoin_5min_markets(
    *,
    gamma: GammaClient | None = None,
    tag_slug: str | None = "crypto",
    title_search: str | None = "Bitcoin",
    max_pages: int | None = 2,
) -> pd.DataFrame:
    """Return flattened live markets that look like Bitcoin 5-minute markets."""
    gamma = gamma or GammaClient()
    rows: list[dict[str, Any]] = []
    for event in gamma.iter_events_keyset(
        limit=500,
        max_pages=max_pages,
        closed=False,
        tag_slug=tag_slug,
        title_search=title_search,
        order="endDate",
        ascending=True,
    ):
        if not is_bitcoin_5min_event(event):
            continue
        rows.extend(flatten_event_markets(event))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    now_ts = datetime.now(timezone.utc).timestamp()
    frame = frame[pd.to_numeric(frame["end_ts"], errors="coerce").gt(now_ts)].copy()
    return frame.sort_values("end_ts").reset_index(drop=True)


def compute_live_signals(
    markets: pd.DataFrame,
    *,
    binance: BinanceClient | None = None,
    clob: ClobClient | None = None,
    symbol: str = "BTCUSDT",
    confirm_min_abs_bps: float = 12.0,
    confirm_min_prob: float = 0.72,
    value_max_market_price: float = 0.40,
    value_min_prob: float = 0.60,
    min_ev_per_share: float = 0.02,
    fee_rate: float = DEFAULT_CRYPTO_TAKER_FEE_RATE,
) -> list[LiveSignal]:
    """Compute dry-run live signals.

    The function does not place orders. It assumes taker execution at current
    best ask for the selected outcome, which is conservative for a fast strategy.
    """
    if markets.empty:
        return []
    binance = binance or BinanceClient()
    clob = clob or ClobClient()
    now = datetime.now(timezone.utc)
    min_start = int(pd.to_numeric(markets["start_ts"], errors="coerce").min()) - 300
    candles = binance.klines_df(symbol=symbol, interval="1m", start=min_start, end=int(now.timestamp()))
    if candles.empty:
        return []
    now_price = float(candles.iloc[-1]["close"])
    signals: list[LiveSignal] = []

    for _, market in markets.iterrows():
        start_dt = parse_dt(market.get("start_ts"))
        end_dt = parse_dt(market.get("end_ts"))
        if start_dt is None or end_dt is None or end_dt <= now:
            continue
        start_price = _price_at_or_before(candles, start_dt)
        if start_price != start_price or start_price <= 0:
            continue
        score_bps = (now_price / start_price - 1.0) * 10_000
        direction = "UP" if score_bps >= 0 else "DOWN"
        seconds_left = (end_dt - now).total_seconds()
        model_prob_up_value = model_probability_up(
            score_bps=pd.Series([score_bps]),
            momentum_1m_bps=pd.Series([_momentum_bps(candles, now_price, minutes=1)]),
            momentum_3m_bps=pd.Series([_momentum_bps(candles, now_price, minutes=3)]),
            seconds_left=pd.Series([seconds_left]),
        ).iloc[0]
        model_prob = float(model_prob_up_value if direction == "UP" else 1.0 - model_prob_up_value)
        asset_id = market.get("up_asset_id") if direction == "UP" else market.get("down_asset_id")
        if not asset_id or str(asset_id) == "nan":
            continue
        buy_price = best_ask(clob.book(str(asset_id)))
        if buy_price != buy_price or buy_price <= 0:
            continue
        fee = float(taker_fee_per_share(buy_price, fee_rate=fee_rate))
        ev = model_prob - buy_price - fee
        bucket = classify_live_bucket(
            abs_score_bps=abs(score_bps),
            model_prob=model_prob,
            market_price=buy_price,
            ev=ev,
            confirm_min_abs_bps=confirm_min_abs_bps,
            confirm_min_prob=confirm_min_prob,
            value_max_market_price=value_max_market_price,
            value_min_prob=value_min_prob,
            min_ev_per_share=min_ev_per_share,
        )
        action = "PAPER_BUY_" + direction if bucket != "SKIP" else "WATCH"
        signals.append(
            LiveSignal(
                condition_id=str(market.get("condition_id")),
                question=str(market.get("question") or market.get("event_title") or ""),
                end_iso=market.get("end_iso"),
                seconds_left=seconds_left,
                direction=direction,
                btc_start=float(start_price),
                btc_now=now_price,
                score_bps=float(score_bps),
                model_prob=model_prob,
                buy_price=float(buy_price),
                fee_per_share=fee,
                expected_value_per_share=float(ev),
                bucket=bucket,
                action=action,
            )
        )
    return signals


def classify_live_bucket(
    *,
    abs_score_bps: float,
    model_prob: float,
    market_price: float,
    ev: float,
    confirm_min_abs_bps: float,
    confirm_min_prob: float,
    value_max_market_price: float,
    value_min_prob: float,
    min_ev_per_share: float,
) -> str:
    if ev < min_ev_per_share:
        return "SKIP"
    if abs_score_bps >= confirm_min_abs_bps and model_prob >= confirm_min_prob:
        return "CONFIRMATION"
    if market_price <= value_max_market_price and model_prob >= value_min_prob:
        return "VALUE_MISMATCH"
    return "SKIP"


def best_ask(book: dict[str, Any]) -> float:
    asks = book.get("asks") or []
    prices = [to_float(level.get("price") if isinstance(level, dict) else level[0]) for level in asks]
    prices = [price for price in prices if price == price]
    return min(prices) if prices else float("nan")


def _price_at_or_before(candles: pd.DataFrame, when: datetime) -> float:
    subset = candles[candles["ts"].le(pd.Timestamp(when))]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[-1]["close"])


def _momentum_bps(candles: pd.DataFrame, now_price: float, *, minutes: int) -> float:
    if candles.empty:
        return 0.0
    cutoff = candles.iloc[-1]["ts"] - pd.Timedelta(minutes=minutes)
    subset = candles[candles["ts"].le(cutoff)]
    if subset.empty:
        return 0.0
    base = float(subset.iloc[-1]["close"])
    if base <= 0:
        return 0.0
    return (now_price / base - 1.0) * 10_000
