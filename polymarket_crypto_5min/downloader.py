"""Historical data download helpers for Bitcoin five-minute markets."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .clients import ApiError, BinanceClient, ClobClient, DataApiClient, GammaClient
from .utils import any_substring, iso_utc, normalize_text, parse_dt, parse_jsonish, to_float, unix_seconds

LOGGER = logging.getLogger(__name__)

MAX_DATA_API_LIMIT = 10_000
MAX_DATA_API_OFFSET = 10_000
NY_TZ = ZoneInfo("America/New_York")

BITCOIN_NEEDLES = ("bitcoin", "btc")
FIVE_MIN_NEEDLES = (
    "5m",
    "5-min",
    "5 min",
    "5mins",
    "5 mins",
    "5-minute",
    "5 minutes",
    "five-minute",
    "five minutes",
)
DIRECTION_NEEDLES = ("up or down", "up/down", "higher", "lower", "above", "below", "up", "down")
WINDOW_RE = re.compile(
    r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s*"
    r"(?P<shour>\d{1,2})(?::(?P<sminute>\d{2}))?\s*(?P<sampm>AM|PM)\s*[-–]\s*"
    r"(?P<ehour>\d{1,2})(?::(?P<eminute>\d{2}))?\s*(?P<eampm>AM|PM)\s*(?:ET|EST|EDT)?",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DownloadSummary:
    pages_seen: int
    events_seen: int
    markets_matched: int
    output_path: str | None = None


def is_bitcoin_5min_event(event: dict[str, Any]) -> bool:
    """Return True for likely Bitcoin five-minute up/down events."""
    market_text = " ".join(
        normalize_text(
            market.get("question"),
            market.get("slug"),
            market.get("groupItemTitle"),
            market.get("outcomes"),
        )
        for market in event.get("markets", []) or []
    )
    tag_text = " ".join(
        normalize_text(tag.get("label"), tag.get("slug"))
        for tag in event.get("tags", []) or []
        if isinstance(tag, dict)
    )
    text = normalize_text(
        event.get("title"),
        event.get("slug"),
        event.get("ticker"),
        event.get("description"),
        event.get("category"),
        tag_text,
        market_text,
    )
    has_five_min_hint = any_substring(text, FIVE_MIN_NEEDLES) or WINDOW_RE.search(text)
    return any_substring(text, BITCOIN_NEEDLES) and bool(has_five_min_hint) and any_substring(text, DIRECTION_NEEDLES)


def flatten_event_markets(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one Gamma event into one row per nested market."""
    rows: list[dict[str, Any]] = []
    markets = event.get("markets") or []
    if not markets:
        markets = [event]
    for market in markets:
        if not isinstance(market, dict):
            continue
        row = flatten_market(event, market)
        if row:
            rows.append(row)
    return rows


def flatten_market(event: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    outcomes = [str(x) for x in parse_jsonish(market.get("outcomes"), [])]
    outcome_prices = [to_float(x) for x in parse_jsonish(market.get("outcomePrices"), [])]
    clob_token_ids = [str(x) for x in parse_jsonish(market.get("clobTokenIds"), [])]
    if not clob_token_ids:
        clob_token_ids = [str(x) for x in parse_jsonish(market.get("clob_token_ids"), [])]

    up_asset_id, down_asset_id = infer_up_down_assets(
        outcomes=outcomes,
        token_ids=clob_token_ids,
        question=market.get("question") or event.get("title") or "",
    )
    up_price = infer_price_for_asset(up_asset_id, clob_token_ids, outcome_prices)
    down_price = infer_price_for_asset(down_asset_id, clob_token_ids, outcome_prices)

    start_date = market.get("startDateIso") or market.get("startDate") or event.get("startDate")
    end_date = market.get("endDateIso") or market.get("endDate") or event.get("endDate")
    closed_time = market.get("closedTime") or event.get("closedTime")
    condition_id = market.get("conditionId") or market.get("condition_id") or market.get("market")

    question_text = market.get("question") or event.get("title") or market.get("slug") or event.get("slug") or ""
    fallback_year = infer_year(event, market)
    parsed_window = parse_market_window_from_text(question_text, fallback_year=fallback_year)
    if parsed_window is not None:
        start_date, end_date = parsed_window

    start_ts = unix_seconds(start_date)
    end_ts = unix_seconds(end_date)
    duration_seconds = end_ts - start_ts if start_ts is not None and end_ts is not None else None

    return {
        "event_id": event.get("id"),
        "event_slug": event.get("slug"),
        "event_title": event.get("title"),
        "event_closed": event.get("closed"),
        "event_active": event.get("active"),
        "event_start": iso_utc(event.get("startDate")),
        "event_end": iso_utc(event.get("endDate")),
        "market_id": market.get("id"),
        "condition_id": condition_id,
        "market_slug": market.get("slug"),
        "question": market.get("question"),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_seconds": duration_seconds,
        "start_iso": iso_utc(start_date),
        "end_iso": iso_utc(end_date),
        "closed_time": iso_utc(closed_time),
        "active": market.get("active"),
        "closed": market.get("closed"),
        "accepting_orders": market.get("acceptingOrders"),
        "enable_order_book": market.get("enableOrderBook"),
        "fees_enabled": market.get("feesEnabled"),
        "volume": to_float(market.get("volumeNum") or market.get("volume")),
        "liquidity": to_float(market.get("liquidityNum") or market.get("liquidity")),
        "best_bid": to_float(market.get("bestBid")),
        "best_ask": to_float(market.get("bestAsk")),
        "last_trade_price": to_float(market.get("lastTradePrice")),
        "outcomes": outcomes,
        "outcome_prices": outcome_prices,
        "clob_token_ids": clob_token_ids,
        "up_asset_id": up_asset_id,
        "down_asset_id": down_asset_id,
        "gamma_up_price": up_price,
        "gamma_down_price": down_price,
        "winner_outcome": market.get("resolutionOutcome") or market.get("winner") or market.get("winningOutcome"),
        "raw_market_type": market.get("marketType"),
    }


def parse_market_window_from_text(text: str, *, fallback_year: int | None = None) -> tuple[datetime, datetime] | None:
    """Parse titles like ``Bitcoin Up or Down - July 2, 5:30PM-5:35PM ET``.

    Gamma often stores the event start/end as a full UTC day for these markets;
    the actual tradable five-minute window is embedded in the market title.
    """
    match = WINDOW_RE.search(str(text or ""))
    if not match:
        return None
    year = fallback_year or datetime.now(timezone.utc).year
    month_name = match.group("month").title()
    try:
        month_num = datetime.strptime(month_name[:3], "%b").month
    except ValueError:
        return None
    day = int(match.group("day"))
    shour = _to_24h(int(match.group("shour")), match.group("sampm"))
    ehour = _to_24h(int(match.group("ehour")), match.group("eampm"))
    sminute = int(match.group("sminute") or 0)
    eminute = int(match.group("eminute") or 0)
    try:
        start_local = datetime(year, month_num, day, shour, sminute, tzinfo=NY_TZ)
        end_local = datetime(year, month_num, day, ehour, eminute, tzinfo=NY_TZ)
    except ValueError:
        return None
    if end_local <= start_local:
        end_local += timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def infer_year(event: dict[str, Any], market: dict[str, Any]) -> int | None:
    for value in (
        market.get("endDateIso"),
        market.get("endDate"),
        event.get("endDate"),
        market.get("closedTime"),
        event.get("closedTime"),
        event.get("startDate"),
    ):
        dt = parse_dt(value)
        if dt is not None:
            return dt.year
    return None


def _to_24h(hour: int, ampm: str) -> int:
    hour = hour % 12
    return hour + (12 if ampm.upper() == "PM" else 0)


def infer_up_down_assets(*, outcomes: list[str], token_ids: list[str], question: str) -> tuple[str | None, str | None]:
    """Map outcome labels/token IDs to UP and DOWN assets."""
    if len(outcomes) != len(token_ids) or len(token_ids) < 2:
        return None, None

    labels = [label.lower() for label in outcomes]
    up_idx = _first_index(labels, ("up", "higher", "above", "yes up"))
    down_idx = _first_index(labels, ("down", "lower", "below", "yes down"))

    question_lc = question.lower()
    if up_idx is None and down_idx is None and set(labels[:2]) >= {"yes", "no"}:
        yes_idx = labels.index("yes")
        no_idx = labels.index("no")
        if any(word in question_lc for word in ("up", "higher", "above")):
            up_idx, down_idx = yes_idx, no_idx
        elif any(word in question_lc for word in ("down", "lower", "below")):
            down_idx, up_idx = yes_idx, no_idx

    up_asset = token_ids[up_idx] if up_idx is not None and up_idx < len(token_ids) else None
    down_asset = token_ids[down_idx] if down_idx is not None and down_idx < len(token_ids) else None
    if up_asset is None and down_asset is not None and len(token_ids) == 2:
        up_asset = token_ids[1 - token_ids.index(down_asset)]
    if down_asset is None and up_asset is not None and len(token_ids) == 2:
        down_asset = token_ids[1 - token_ids.index(up_asset)]
    return up_asset, down_asset


def _first_index(labels: list[str], terms: Iterable[str]) -> int | None:
    for idx, label in enumerate(labels):
        if any(term in label for term in terms):
            return idx
    return None


def infer_price_for_asset(asset_id: str | None, token_ids: list[str], prices: list[float]) -> float:
    if not asset_id or len(token_ids) != len(prices):
        return float("nan")
    try:
        return prices[token_ids.index(asset_id)]
    except ValueError:
        return float("nan")


def download_bitcoin_5min_markets(
    *,
    out_path: str | Path | None = None,
    gamma: GammaClient | None = None,
    closed: bool = True,
    tag_slug: str | None = "crypto",
    title_search: str | None = "Bitcoin",
    start_date_min: str | None = None,
    start_date_max: str | None = None,
    end_date_min: str | None = None,
    end_date_max: str | None = None,
    max_pages: int | None = None,
    sleep_s: float = 0.0,
) -> tuple[pd.DataFrame, DownloadSummary]:
    """Download and flatten likely historical Bitcoin five-minute markets."""
    gamma = gamma or GammaClient()
    params = {
        "closed": closed,
        "tag_slug": tag_slug,
        "title_search": title_search,
        "start_date_min": start_date_min,
        "start_date_max": start_date_max,
        "end_date_min": end_date_min,
        "end_date_max": end_date_max,
        "order": "endDate",
        "ascending": False,
    }
    rows: list[dict[str, Any]] = []
    events_seen = 0
    pages_seen = 0
    for event in gamma.iter_events_keyset(limit=500, max_pages=max_pages, **params):
        events_seen += 1
        if events_seen % 500 == 1:
            pages_seen += 1
        if not is_bitcoin_5min_event(event):
            continue
        rows.extend(flatten_event_markets(event))
        if sleep_s:
            time.sleep(sleep_s)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["condition_id", "market_id", "up_asset_id"])
        frame["duration_seconds"] = pd.to_numeric(frame["duration_seconds"], errors="coerce")
        frame = frame[frame["duration_seconds"].between(240, 420, inclusive="both")].copy()
        frame = frame.sort_values(["end_ts", "condition_id"], na_position="last").reset_index(drop=True)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    summary = DownloadSummary(
        pages_seen=pages_seen,
        events_seen=events_seen,
        markets_matched=len(frame),
        output_path=str(out_path) if out_path else None,
    )
    return frame, summary


def download_polymarket_price_history(
    markets: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
    clob: ClobClient | None = None,
    interval: str = "1m",
    fidelity: int = 1,
    asset_column: str = "up_asset_id",
    pad_seconds: int = 600,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    """Download CLOB price history for each market's selected token."""
    clob = clob or ClobClient()
    records: list[dict[str, Any]] = []
    required = {asset_column, "condition_id", "start_ts", "end_ts"}
    missing = required - set(markets.columns)
    if missing:
        raise ValueError(f"markets frame missing required columns: {sorted(missing)}")

    for idx, row in markets.iterrows():
        asset_id = row.get(asset_column)
        start_ts = row.get("start_ts")
        end_ts = row.get("end_ts")
        if pd.isna(asset_id) or pd.isna(start_ts) or pd.isna(end_ts):
            continue
        try:
            history = clob.prices_history(
                str(asset_id),
                start_ts=int(start_ts) - pad_seconds,
                end_ts=int(end_ts) + pad_seconds,
                interval=interval,
                fidelity=fidelity,
            )
        except ApiError as exc:
            LOGGER.warning("Skipping price history for %s: %s", asset_id, exc)
            continue
        for point in history:
            records.append(
                {
                    "condition_id": row.get("condition_id"),
                    "market_id": row.get("market_id"),
                    "asset_id": asset_id,
                    "asset_role": asset_column.replace("_asset_id", ""),
                    "t": point.get("t"),
                    "p": point.get("p"),
                }
            )
        LOGGER.info("Downloaded %d price points for %s (%d/%d)", len(history), asset_id, idx + 1, len(markets))
        if sleep_s:
            time.sleep(sleep_s)

    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["t"], unit="s", utc=True, errors="coerce")
        frame["p"] = pd.to_numeric(frame["p"], errors="coerce")
        frame = frame.sort_values(["condition_id", "ts"]).reset_index(drop=True)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    return frame


def download_data_api_trades(
    markets: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
    data_api: DataApiClient | None = None,
    batch_size: int = 20,
    sleep_s: float = 0.1,
) -> pd.DataFrame:
    """Download public Data API trade rows for condition IDs."""
    data_api = data_api or DataApiClient()
    condition_ids = [str(x) for x in markets.get("condition_id", pd.Series(dtype=str)).dropna().unique()]
    all_rows: list[dict[str, Any]] = []
    for start in range(0, len(condition_ids), batch_size):
        batch = condition_ids[start : start + batch_size]
        offset = 0
        while True:
            trades = data_api.trades(market=batch, limit=MAX_DATA_API_LIMIT, offset=offset, taker_only=True)
            all_rows.extend(trades)
            if len(trades) < MAX_DATA_API_LIMIT or offset >= MAX_DATA_API_OFFSET:
                if len(trades) >= MAX_DATA_API_LIMIT and offset >= MAX_DATA_API_OFFSET:
                    LOGGER.warning("Data API offset cap reached for condition ID batch starting at %d", start)
                break
            offset += MAX_DATA_API_LIMIT
        if sleep_s:
            time.sleep(sleep_s)
    frame = pd.DataFrame(all_rows)
    if not frame.empty and "timestamp" in frame:
        frame["ts"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True, errors="coerce")
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    return frame


def download_btc_klines_for_markets(
    markets: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
    binance: BinanceClient | None = None,
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    pad_seconds: int = 900,
) -> pd.DataFrame:
    """Download Binance BTC klines covering the supplied markets."""
    if markets.empty:
        raise ValueError("markets frame is empty")
    start_ts = int(markets["start_ts"].dropna().min()) - pad_seconds
    end_ts = int(markets["end_ts"].dropna().max()) + pad_seconds
    binance = binance or BinanceClient()
    frame = binance.klines_df(symbol=symbol, interval=interval, start=start_ts, end=end_ts)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
    return frame
