"""Thin HTTP clients for public Polymarket and Binance endpoints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

import pandas as pd
import requests

from .config import (
    BINANCE_BASE_URL,
    CLOB_BASE_URL,
    DATA_API_BASE_URL,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    GAMMA_BASE_URL,
)
from .utils import parse_dt, to_float

LOGGER = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """Raised when an upstream API request fails after retries."""


@dataclass(slots=True)
class HttpClient:
    """Small retrying JSON client.

    It uses conservative retry/backoff because historical downloads can span many
    pages and because rate limits may change. The package does not bypass or
    evade upstream controls.
    """

    timeout: int = DEFAULT_HTTP_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    session: requests.Session = field(default_factory=requests.Session)

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=_clean_params(params), timeout=self.timeout)
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    retry_after = response.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else min(2**attempt, 30)
                    LOGGER.warning("Retryable HTTP %s from %s; sleeping %.1fs", response.status_code, url, delay)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(2**attempt, 30)
                LOGGER.warning("Request failed for %s: %s; sleeping %.1fs", url, exc, delay)
                time.sleep(delay)
        raise ApiError(f"GET {url} failed after retries: {last_error}")


def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = "true" if value else "false"
        elif isinstance(value, (list, tuple, set)):
            cleaned[key] = ",".join(str(item) for item in value)
        else:
            cleaned[key] = value
    return cleaned


@dataclass(slots=True)
class GammaClient:
    base_url: str = GAMMA_BASE_URL
    http: HttpClient = field(default_factory=HttpClient)

    def list_events_keyset(self, **params: Any) -> dict[str, Any]:
        """Return one ``/events/keyset`` page."""
        return self.http.get_json(f"{self.base_url}/events/keyset", params=params)

    def iter_events_keyset(
        self,
        *,
        limit: int = 500,
        max_pages: int | None = None,
        **params: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield events across keyset pages.

        ``offset`` is intentionally unsupported by the keyset endpoint; this
        method follows ``next_cursor`` until exhausted or ``max_pages`` is hit.
        """
        cursor: str | None = None
        page = 0
        while True:
            page += 1
            payload = self.list_events_keyset(limit=limit, after_cursor=cursor, **params)
            events = payload.get("events") or payload.get("data") or []
            if not events:
                break
            for event in events:
                yield event
            cursor = payload.get("next_cursor")
            if not cursor or cursor in {"LTE=", "-1"}:
                break
            if max_pages is not None and page >= max_pages:
                break

    def list_events_offset(self, **params: Any) -> list[dict[str, Any]]:
        payload = self.http.get_json(f"{self.base_url}/events", params=params)
        return payload if isinstance(payload, list) else payload.get("events", [])


@dataclass(slots=True)
class ClobClient:
    base_url: str = CLOB_BASE_URL
    http: HttpClient = field(default_factory=HttpClient)

    def prices_history(
        self,
        asset_id: str,
        *,
        start_ts: int | None = None,
        end_ts: int | None = None,
        interval: str | None = "1m",
        fidelity: int = 1,
    ) -> list[dict[str, Any]]:
        params = {
            "market": asset_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
        }
        # The CLOB endpoint rejects some bounded start/end requests when a
        # relative ``interval`` filter is also supplied. For historical
        # point-in-time backtests, the bounded window is the source of truth;
        # use ``interval`` only for unbounded relative lookbacks.
        if interval is not None and start_ts is None and end_ts is None:
            params["interval"] = interval
        payload = self.http.get_json(f"{self.base_url}/prices-history", params=params)
        if isinstance(payload, dict):
            return payload.get("history", [])
        return []

    def book(self, asset_id: str) -> dict[str, Any]:
        return self.http.get_json(f"{self.base_url}/book", params={"token_id": asset_id})

    def midpoint(self, asset_id: str) -> float:
        payload = self.http.get_json(f"{self.base_url}/midpoint", params={"token_id": asset_id})
        return to_float(payload.get("mid") or payload.get("midpoint"))

    def spread(self, asset_id: str) -> float:
        payload = self.http.get_json(f"{self.base_url}/spread", params={"token_id": asset_id})
        return to_float(payload.get("spread"))


@dataclass(slots=True)
class DataApiClient:
    base_url: str = DATA_API_BASE_URL
    http: HttpClient = field(default_factory=HttpClient)

    def trades(
        self,
        *,
        market: list[str] | str | None = None,
        user: str | None = None,
        side: str | None = None,
        limit: int = 10_000,
        offset: int = 0,
        taker_only: bool = True,
    ) -> list[dict[str, Any]]:
        params = {
            "market": market,
            "user": user,
            "side": side,
            "limit": limit,
            "offset": offset,
            "takerOnly": taker_only,
        }
        payload = self.http.get_json(f"{self.base_url}/trades", params=params)
        return payload if isinstance(payload, list) else payload.get("data", [])


@dataclass(slots=True)
class BinanceClient:
    base_url: str = BINANCE_BASE_URL
    http: HttpClient = field(default_factory=HttpClient)

    def klines(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start: datetime | int | None = None,
        end: datetime | int | None = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start is not None:
            params["startTime"] = _to_ms(start)
        if end is not None:
            params["endTime"] = _to_ms(end)
        return self.http.get_json(f"{self.base_url}/api/v3/klines", params=params)

    def iter_klines(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start: datetime | int,
        end: datetime | int,
        limit: int = 1000,
    ) -> Iterator[list[Any]]:
        current_ms = _to_ms(start)
        end_ms = _to_ms(end)
        while current_ms <= end_ms:
            batch = self.klines(symbol=symbol, interval=interval, start=current_ms, end=end_ms, limit=limit)
            if not batch:
                break
            for row in batch:
                yield row
            last_open_ms = int(batch[-1][0])
            next_ms = last_open_ms + _interval_ms(interval)
            if next_ms <= current_ms:
                break
            current_ms = next_ms
            if len(batch) < limit:
                break

    def klines_df(
        self,
        *,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start: datetime | int,
        end: datetime | int,
    ) -> pd.DataFrame:
        rows = list(self.iter_klines(symbol=symbol, interval=interval, start=start, end=end))
        return klines_to_frame(rows, symbol=symbol, interval=interval)


def klines_to_frame(rows: list[list[Any]], *, symbol: str, interval: str) -> pd.DataFrame:
    columns = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns[: len(rows[0])] if rows else columns)
    if frame.empty:
        return pd.DataFrame(columns=["ts", "symbol", "interval", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["ts"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    frame["symbol"] = symbol
    frame["interval"] = interval
    keep = ["ts", "symbol", "interval", "open", "high", "low", "close", "volume", "quote_volume", "num_trades"]
    return frame[[col for col in keep if col in frame]].sort_values("ts").reset_index(drop=True)


def _to_ms(value: datetime | int | float) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    numeric = int(value)
    if numeric < 10_000_000_000:
        return numeric * 1000
    return numeric


def _interval_ms(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    if unit == "m":
        return amount * 60_000
    if unit == "h":
        return amount * 3_600_000
    if unit == "d":
        return amount * 86_400_000
    if unit == "s":
        return amount * 1000
    raise ValueError(f"Unsupported interval for pagination: {interval}")
