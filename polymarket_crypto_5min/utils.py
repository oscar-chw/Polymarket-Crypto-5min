"""Small parsing helpers shared across the package."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as date_parser


def parse_jsonish(value: Any, default: Any | None = None) -> Any:
    """Parse fields that Polymarket sometimes returns as JSON strings.

    Gamma API market fields such as ``outcomes``, ``outcomePrices`` and
    ``clobTokenIds`` are frequently JSON-encoded strings, but can also arrive as
    already-decoded lists depending on endpoint/version. This helper normalizes
    both without throwing on malformed values.
    """
    if default is None:
        default = []
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Some fields may be plain comma-separated strings.
            if "," in value:
                return [part.strip().strip('"') for part in value.split(",") if part.strip()]
            return default
    return default


def to_float(value: Any, default: float | None = math.nan) -> float:
    try:
        if value is None or value == "":
            return float(default) if default is not None else math.nan
        return float(value)
    except (TypeError, ValueError):
        return float(default) if default is not None else math.nan


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_dt(value: Any) -> datetime | None:
    """Parse an ISO datetime or unix timestamp into an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        # Gamma uses ISO strings. WebSocket/CLOB often uses milliseconds.
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif isinstance(value, str) and re.fullmatch(r"\d+(\.\d+)?", value.strip()):
        ts = float(value)
        if ts > 10_000_000_000:
            ts /= 1000.0
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        try:
            dt = date_parser.parse(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def unix_seconds(value: Any) -> int | None:
    dt = parse_dt(value)
    if dt is None:
        return None
    return int(dt.timestamp())


def iso_utc(value: Any) -> str | None:
    dt = parse_dt(value)
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def any_substring(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def sigmoid(x: float) -> float:
    if x >= 35:
        return 1.0
    if x <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def parse_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default
