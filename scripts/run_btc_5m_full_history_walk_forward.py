#!/usr/bin/env python3
"""Pull all historical BTC 5-minute markets and run walk-forward validation.

This is the production research path for the current model. It deliberately
scopes to BTC 5-minute crypto markets because the feature builder uses BTC price
features. Do not mix ETH/SOL/XRP markets into this run until asset-specific
candle joins are implemented.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from polymarket_crypto_5min.downloader import (
    download_bitcoin_5min_markets,
    download_btc_klines_for_markets,
    download_data_api_trades,
    download_polymarket_price_history,
)
from polymarket_crypto_5min.features import build_training_frame, load_candles, load_markets, load_poly_prices
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics
from polymarket_crypto_5min.walk_forward import WalkForwardConfig, make_side_candidates, walk_forward_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/full_btc_5m")
    parser.add_argument("--out-dir", default="data/processed/full_btc_5m_walk_forward")
    parser.add_argument(
        "--start-date-min", default=None, help="Optional ISO lower bound; omit for all available history"
    )
    parser.add_argument("--start-date-max", default=None, help="Optional ISO upper bound")
    parser.add_argument("--max-pages", type=int, default=None, help="Debug only; omit for all pages")
    parser.add_argument("--price-source", choices=["clob", "trades", "both"], default="both")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing CSVs when present")
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--initial-capital", type=float, default=2000.0)
    parser.add_argument("--train-markets", type=int, default=400)
    parser.add_argument("--test-markets", type=int, default=100)
    parser.add_argument("--min-train-trades", type=int, default=25)
    parser.add_argument("--min-bin-observations", type=int, default=30)
    parser.add_argument("--snapshot-seconds-before-close", type=int, default=45)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    markets_path = raw_dir / "btc_5m_markets.csv"
    up_prices_path = raw_dir / "btc_5m_up_price_history.csv"
    down_prices_path = raw_dir / "btc_5m_down_price_history.csv"
    trades_path = raw_dir / "btc_5m_data_api_trades.csv"
    candle_path = raw_dir / "btc_usdt_1m.csv"
    combined_price_path = raw_dir / "btc_5m_price_history_combined.csv"

    if args.skip_existing and markets_path.exists():
        markets = load_markets(markets_path)
    else:
        markets, summary = download_bitcoin_5min_markets(
            out_path=markets_path,
            closed=True,
            tag_slug="crypto",
            title_search="Bitcoin",
            start_date_min=args.start_date_min,
            start_date_max=args.start_date_max,
            max_pages=args.max_pages,
        )
        print(summary)
    if markets.empty:
        raise SystemExit("No BTC 5-minute markets found")

    price_frames: list[pd.DataFrame] = []
    if args.price_source in {"clob", "both"}:
        if args.skip_existing and up_prices_path.exists():
            up = pd.read_csv(up_prices_path)
        else:
            up = download_polymarket_price_history(markets, out_path=up_prices_path, asset_column="up_asset_id")
        if args.skip_existing and down_prices_path.exists():
            down = pd.read_csv(down_prices_path)
        else:
            down = download_polymarket_price_history(markets, out_path=down_prices_path, asset_column="down_asset_id")
        price_frames.extend([up, down])

    if args.price_source in {"trades", "both"}:
        if args.skip_existing and trades_path.exists():
            trades = pd.read_csv(trades_path)
        else:
            trades = download_data_api_trades(markets, out_path=trades_path, batch_size=1)
        trade_prices = trades_to_price_history(trades, markets)
        trade_prices.to_csv(raw_dir / "btc_5m_trade_price_history.csv", index=False)
        price_frames.append(trade_prices)

    combined_prices = pd.concat(
        [frame for frame in price_frames if frame is not None and not frame.empty], ignore_index=True
    )
    if not combined_prices.empty:
        combined_prices["ts"] = pd.to_datetime(combined_prices["ts"], utc=True, errors="coerce")
        combined_prices["p"] = pd.to_numeric(combined_prices["p"], errors="coerce")
        combined_prices = combined_prices.dropna(subset=["condition_id", "asset_id", "ts", "p"])
        combined_prices = combined_prices.sort_values(["condition_id", "asset_id", "ts"]).drop_duplicates(
            subset=["condition_id", "asset_id", "ts"], keep="last"
        )
    combined_prices.to_csv(combined_price_path, index=False)

    if args.skip_existing and candle_path.exists():
        candles = load_candles(candle_path)
    else:
        candles = download_btc_klines_for_markets(markets, out_path=candle_path)

    frame = build_training_frame(
        markets,
        candles,
        load_poly_prices(combined_price_path),
        snapshot_seconds_before_close=args.snapshot_seconds_before_close,
        require_resolved_outcome=True,
        allow_gamma_prices=False,
    )
    frame.to_csv(out_dir / "feature_frame.csv", index=False)

    candidates = make_side_candidates(frame)
    candidates.to_csv(out_dir / "side_candidates.csv", index=False)

    config = WalkForwardConfig(
        initial_capital=args.initial_capital,
        stake_usdc=args.stake,
        train_markets=args.train_markets,
        test_markets=args.test_markets,
        min_train_trades=args.min_train_trades,
        min_bin_observations=args.min_bin_observations,
    )
    wf_trades, folds, rules = walk_forward_backtest(candidates, config=config)
    folds.to_csv(out_dir / "walk_forward_folds.csv", index=False)
    leakage_violations = (
        int((~folds["leakage_check_passed"].fillna(False).astype(bool)).sum()) if not folds.empty else None
    )
    if leakage_violations:
        # Fail before writing trades, metrics or equity: a leaking run must not
        # leave results that look like a finished one.
        raise SystemExit(
            f"Leakage check failed: {leakage_violations} fold(s) have train_end_dt >= test_first_decision_dt"
        )
    wf_trades.to_csv(out_dir / "walk_forward_trades.csv", index=False)
    rules.to_csv(out_dir / "walk_forward_selected_rules.csv", index=False)
    metrics = performance_metrics(wf_trades, initial_capital=args.initial_capital)
    metrics.to_csv(out_dir / "walk_forward_metrics.csv", index=False)
    equity_curve(wf_trades, initial_capital=args.initial_capital).to_csv(
        out_dir / "walk_forward_equity_curve.csv", index=False
    )

    manifest = {
        "scope": "BTC 5-minute crypto markets only",
        "markets": int(len(markets)),
        "resolved_feature_rows": int(len(frame)),
        "side_candidates": int(len(candidates)),
        "walk_forward_trades": int(len(wf_trades)),
        "initial_capital": args.initial_capital,
        "stake": args.stake,
        "train_markets": args.train_markets,
        "test_markets": args.test_markets,
        "leakage_violations": leakage_violations,
        "mdd_is_realized_only": True,
        "mark_to_market_mdd": "not available without orderbook snapshots while positions are open",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))
    if not metrics.empty:
        print(metrics.to_string(index=False))


def trades_to_price_history(trades: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["condition_id", "market_id", "asset_id", "asset_role", "t", "p", "ts"])
    market_lookup = markets.set_index(markets["condition_id"].astype(str))
    records: list[dict[str, object]] = []
    for _, trade in trades.iterrows():
        condition_id = (
            trade.get("conditionId") or trade.get("condition_id") or trade.get("market") or trade.get("marketId")
        )
        if condition_id is None:
            continue
        condition_key = str(condition_id)
        if condition_key not in market_lookup.index:
            continue
        market = market_lookup.loc[condition_key]
        if isinstance(market, pd.DataFrame):
            market = market.iloc[0]
        price = trade.get("price")
        timestamp = trade.get("timestamp") or trade.get("time") or trade.get("createdAt")
        if price is None or timestamp is None:
            continue
        ts = pd.to_datetime(timestamp, unit="s", utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        asset = str(trade.get("asset") or trade.get("asset_id") or trade.get("token_id") or trade.get("tokenId") or "")
        outcome = str(trade.get("outcome") or trade.get("outcomeName") or "").strip().lower()
        up_asset = str(market.get("up_asset_id") or "")
        down_asset = str(market.get("down_asset_id") or "")
        asset_role = None
        if asset == up_asset or outcome in {"up", "higher", "yes"}:
            asset = up_asset
            asset_role = "up"
        elif asset == down_asset or outcome in {"down", "lower", "no"}:
            asset = down_asset
            asset_role = "down"
        if not asset or not asset_role:
            continue
        records.append(
            {
                "condition_id": condition_key,
                "market_id": market.get("market_id"),
                "asset_id": asset,
                "asset_role": asset_role,
                "t": int(ts.timestamp()),
                "p": price,
                "ts": ts,
            }
        )
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()
