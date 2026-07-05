#!/usr/bin/env python3
"""Print dry-run live signals for active Bitcoin five-minute markets."""

from __future__ import annotations

import argparse
import json

from polymarket_crypto_5min.live_signal import compute_live_signals, find_active_bitcoin_5min_markets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-slug", default="crypto")
    parser.add_argument("--title-search", default="Bitcoin")
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--confirm-bps", type=float, default=12.0)
    parser.add_argument("--confirm-prob", type=float, default=0.72)
    parser.add_argument("--value-max-price", type=float, default=0.40)
    parser.add_argument("--value-min-prob", type=float, default=0.60)
    parser.add_argument("--min-ev", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markets = find_active_bitcoin_5min_markets(
        tag_slug=args.tag_slug or None,
        title_search=args.title_search or None,
        max_pages=args.max_pages,
    )
    signals = compute_live_signals(
        markets,
        confirm_min_abs_bps=args.confirm_bps,
        confirm_min_prob=args.confirm_prob,
        value_max_market_price=args.value_max_price,
        value_min_prob=args.value_min_prob,
        min_ev_per_share=args.min_ev,
    )
    for signal in signals:
        print(json.dumps(signal.to_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
