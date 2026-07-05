#!/usr/bin/env python3
"""Download historical Bitcoin 5-minute Polymarket data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from polymarket_crypto_5min.downloader import (
    download_bitcoin_5min_markets,
    download_btc_klines_for_markets,
    download_data_api_trades,
    download_polymarket_price_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw", help="Directory for CSV outputs")
    parser.add_argument("--max-pages", type=int, default=None, help="Limit Gamma keyset pages for testing")
    parser.add_argument("--tag-slug", default="crypto", help="Gamma tag slug filter; set empty string to disable")
    parser.add_argument("--title-search", default="Bitcoin", help="Gamma title search filter")
    parser.add_argument("--start-date-min", default=None, help="Optional ISO lower bound for market start")
    parser.add_argument("--start-date-max", default=None, help="Optional ISO upper bound for market start")
    parser.add_argument("--skip-clob-history", action="store_true", help="Skip CLOB prices-history download")
    parser.add_argument("--skip-trades", action="store_true", help="Skip public Data API trades download")
    parser.add_argument("--skip-binance", action="store_true", help="Skip Binance BTCUSDT 1m candle download")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markets, summary = download_bitcoin_5min_markets(
        out_path=out_dir / "btc_5m_markets.csv",
        closed=True,
        tag_slug=args.tag_slug or None,
        title_search=args.title_search or None,
        start_date_min=args.start_date_min,
        start_date_max=args.start_date_max,
        max_pages=args.max_pages,
    )
    print(summary)
    if markets.empty:
        print("No matching markets found. Try removing --tag-slug or changing --title-search.")
        return

    if not args.skip_clob_history:
        download_polymarket_price_history(markets, out_path=out_dir / "btc_5m_up_price_history.csv")
    if not args.skip_trades:
        download_data_api_trades(markets, out_path=out_dir / "btc_5m_data_api_trades.csv")
    if not args.skip_binance:
        download_btc_klines_for_markets(markets, out_path=out_dir / "btc_usdt_1m.csv")


if __name__ == "__main__":
    main()
