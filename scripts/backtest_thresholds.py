#!/usr/bin/env python3
"""Build features, search strategy thresholds, and print performance metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from polymarket_crypto_5min.backtest import StrategyThresholds, simulate_strategy, summarize_trades, threshold_report
from polymarket_crypto_5min.features import build_training_frame, load_candles, load_markets, load_poly_prices
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", default="data/raw/btc_5m_markets.csv")
    parser.add_argument("--btc-candles", default="data/raw/btc_usdt_1m.csv")
    parser.add_argument("--poly-prices", default="data/raw/btc_5m_up_price_history.csv")
    parser.add_argument("--snapshot-seconds-before-close", type=int, default=45)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument(
        "--allow-gamma-prices", action="store_true", help="Exploratory only: may use post-resolution prices"
    )
    parser.add_argument("--out-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markets = load_markets(args.markets)
    candles = load_candles(args.btc_candles)
    poly_prices = load_poly_prices(args.poly_prices) if args.poly_prices else None
    frame = build_training_frame(
        markets,
        candles,
        poly_prices,
        snapshot_seconds_before_close=args.snapshot_seconds_before_close,
        allow_gamma_prices=args.allow_gamma_prices,
    )
    frame_path = out_dir / "training_frame.csv"
    frame.to_csv(frame_path, index=False)
    print(f"Wrote {len(frame):,} feature rows to {frame_path}")

    usable = frame.dropna(subset=["market_price"])
    report = threshold_report(usable, stake_usdc=args.stake, min_trades=args.min_trades)
    report_path = out_dir / "threshold_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Wrote threshold report to {report_path}")
    if not report.empty:
        print(report.head(20).to_string(index=False))

    default_trades = simulate_strategy(usable, StrategyThresholds(), stake_usdc=args.stake)
    trades_path = out_dir / "default_strategy_trades.csv"
    default_trades.to_csv(trades_path, index=False)
    print(f"Wrote default strategy trades to {trades_path}")

    summary = summarize_trades(default_trades)
    summary_path = out_dir / "default_strategy_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote default strategy summary to {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))

    metrics = performance_metrics(default_trades, initial_capital=args.initial_capital)
    metrics_path = out_dir / "default_strategy_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote default strategy metrics to {metrics_path}")
    if not metrics.empty:
        print(metrics.to_string(index=False))

    curve = equity_curve(default_trades, initial_capital=args.initial_capital)
    curve_path = out_dir / "default_strategy_equity_curve.csv"
    curve.to_csv(curve_path, index=False)
    print(f"Wrote default strategy equity curve to {curve_path}")


if __name__ == "__main__":
    main()
