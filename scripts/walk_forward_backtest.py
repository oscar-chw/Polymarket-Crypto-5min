#!/usr/bin/env python3
"""Run strict walk-forward calibrated backtests.

The test fold never influences its calibrator or selected rule. Use this script
for reported strategy results; ``backtest_thresholds.py`` is only a simple
threshold-smoke-test helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from polymarket_crypto_5min.features import build_training_frame, load_candles, load_markets, load_poly_prices
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics
from polymarket_crypto_5min.walk_forward import WalkForwardConfig, make_side_candidates, walk_forward_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", default="data/raw/btc_5m_markets.csv")
    parser.add_argument("--btc-candles", default="data/raw/btc_usdt_1m.csv")
    parser.add_argument("--poly-prices", default="data/raw/btc_5m_up_price_history.csv")
    parser.add_argument("--snapshot-seconds-before-close", type=int, default=45)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--initial-capital", type=float, default=2000.0)
    parser.add_argument("--train-markets", type=int, default=400)
    parser.add_argument("--test-markets", type=int, default=100)
    parser.add_argument("--min-train-trades", type=int, default=25)
    parser.add_argument("--min-bin-observations", type=int, default=30)
    parser.add_argument(
        "--allow-gamma-prices", action="store_true", help="Exploratory only: may use post-resolution prices"
    )
    parser.add_argument(
        "--allow-external-btc-outcome-fallback",
        action="store_true",
        help="Diagnostics only: label unresolved markets from external BTC candles",
    )
    parser.add_argument("--out-dir", default="data/processed/walk_forward")
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
        require_resolved_outcome=True,
        allow_external_btc_outcome_fallback=args.allow_external_btc_outcome_fallback,
    )
    frame_path = out_dir / "feature_frame.csv"
    frame.to_csv(frame_path, index=False)
    print(f"Wrote {len(frame):,} resolved feature rows to {frame_path}")

    candidates = make_side_candidates(frame)
    candidates_path = out_dir / "side_candidates.csv"
    candidates.to_csv(candidates_path, index=False)
    print(f"Wrote {len(candidates):,} side candidates to {candidates_path}")

    config = WalkForwardConfig(
        initial_capital=args.initial_capital,
        stake_usdc=args.stake,
        train_markets=args.train_markets,
        test_markets=args.test_markets,
        min_train_trades=args.min_train_trades,
        min_bin_observations=args.min_bin_observations,
    )
    trades, fold_report, selected_rules = walk_forward_backtest(candidates, config=config)

    trades_path = out_dir / "walk_forward_trades.csv"
    fold_path = out_dir / "walk_forward_folds.csv"
    rules_path = out_dir / "walk_forward_selected_rules.csv"
    trades.to_csv(trades_path, index=False)
    fold_report.to_csv(fold_path, index=False)
    selected_rules.to_csv(rules_path, index=False)
    print(f"Wrote walk-forward trades to {trades_path}")
    print(f"Wrote fold report to {fold_path}")
    print(f"Wrote selected rules to {rules_path}")

    metrics = performance_metrics(trades, initial_capital=args.initial_capital)
    metrics_path = out_dir / "walk_forward_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote metrics to {metrics_path}")
    if not metrics.empty:
        print(metrics.to_string(index=False))

    curve = equity_curve(trades, initial_capital=args.initial_capital)
    curve_path = out_dir / "walk_forward_equity_curve.csv"
    curve.to_csv(curve_path, index=False)
    print(f"Wrote equity curve to {curve_path}")

    if not fold_report.empty and not bool(fold_report["leakage_check_passed"].all()):
        raise SystemExit("Leakage check failed: at least one fold has train_end_dt >= test_start_dt")


if __name__ == "__main__":
    main()
