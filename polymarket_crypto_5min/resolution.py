"""Polymarket resolution/outcome helpers.

Backtests must use the market's settled outcome as the payout label. BTC candles
are features only; using external BTC close vs start as the label can silently
create false wins when the exchange feed or exact settlement timestamp differs
from Polymarket's resolver.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .utils import parse_jsonish, to_float

UP_TERMS = ("up", "higher", "above")
DOWN_TERMS = ("down", "lower", "below")
YES_TERMS = ("yes", "true")
NO_TERMS = ("no", "false")


def label_to_direction(label: Any, *, question: Any = "") -> str | None:
    """Map an outcome label to ``UP`` or ``DOWN``.

    Direct labels such as ``Up``/``Down`` are straightforward. For binary
    Yes/No markets, the question text determines what Yes means.
    """
    text = str(label or "").strip().lower()
    if not text or text == "nan":
        return None
    if any(term in text for term in UP_TERMS):
        return "UP"
    if any(term in text for term in DOWN_TERMS):
        return "DOWN"

    question_lc = str(question or "").lower()
    if text in YES_TERMS:
        if any(term in question_lc for term in UP_TERMS):
            return "UP"
        if any(term in question_lc for term in DOWN_TERMS):
            return "DOWN"
    if text in NO_TERMS:
        if any(term in question_lc for term in UP_TERMS):
            return "DOWN"
        if any(term in question_lc for term in DOWN_TERMS):
            return "UP"
    return None


def resolved_direction_from_row(
    row: pd.Series | dict[str, Any],
    *,
    min_winner_price: float = 0.95,
    max_loser_price: float = 0.05,
) -> tuple[str | None, str]:
    """Infer the settled Polymarket direction for one market row.

    Priority order:

    1. Explicit winner/resolution label fields from Gamma.
    2. Settled ``outcome_prices`` such as ``["1", "0"]`` mapped through
       ``outcomes``.
    3. Settled ``gamma_up_price``/``gamma_down_price`` when available.

    Returns ``(direction, source)``. ``direction`` is ``None`` when the row is not
    confidently resolved.
    """
    get = row.get if isinstance(row, dict) else row.get
    question = get("question") or get("event_title") or ""

    for field in ("winner_outcome", "resolutionOutcome", "resolution_outcome", "winningOutcome", "winning_outcome"):
        direction = label_to_direction(get(field), question=question)
        if direction:
            return direction, field

    outcomes = [str(x) for x in parse_jsonish(get("outcomes"), [])]
    prices = [to_float(x) for x in parse_jsonish(get("outcome_prices"), [])]
    if len(outcomes) >= 2 and len(outcomes) == len(prices):
        finite = [(idx, price) for idx, price in enumerate(prices) if not math.isnan(price)]
        if finite:
            winner_idx, winner_price = max(finite, key=lambda item: item[1])
            loser_prices = [price for idx, price in finite if idx != winner_idx]
            max_other = max(loser_prices) if loser_prices else 0.0
            if winner_price >= min_winner_price and max_other <= max_loser_price:
                direction = label_to_direction(outcomes[winner_idx], question=question)
                if direction:
                    return direction, "outcome_prices"

    up_price = to_float(get("gamma_up_price"))
    down_price = to_float(get("gamma_down_price"))
    if not math.isnan(up_price) and not math.isnan(down_price):
        if up_price >= min_winner_price and down_price <= max_loser_price:
            return "UP", "gamma_up_down_prices"
        if down_price >= min_winner_price and up_price <= max_loser_price:
            return "DOWN", "gamma_up_down_prices"

    return None, "unresolved"


def append_resolved_outcomes(
    frame: pd.DataFrame,
    *,
    require_resolved: bool = True,
    allow_external_btc_fallback: bool = False,
    external_direction_col: str = "btc_external_direction",
) -> pd.DataFrame:
    """Add settled outcome columns to a frame.

    ``allow_external_btc_fallback`` exists only for diagnostics; production
    backtests should leave it False so unresolved markets are dropped instead of
    silently labeled with an external exchange price.
    """
    rows = frame.copy()
    resolved = rows.apply(lambda row: resolved_direction_from_row(row), axis=1)
    rows["realized_direction"] = [item[0] for item in resolved]
    rows["realized_direction_source"] = [item[1] for item in resolved]

    if allow_external_btc_fallback and external_direction_col in rows.columns:
        missing = rows["realized_direction"].isna()
        rows.loc[missing, "realized_direction"] = rows.loc[missing, external_direction_col]
        rows.loc[missing, "realized_direction_source"] = "external_btc_fallback"

    rows["realized_up"] = rows["realized_direction"].eq("UP")
    if require_resolved:
        rows = rows[rows["realized_direction"].isin(["UP", "DOWN"])].copy()
    return rows.reset_index(drop=True)
