#!/usr/bin/env python3
"""Run exit-aware walk-forward backtests.

This tests the live-trading idea: enter on calibrated confirmation, then take
profit / target-price exit / stop-loss / max-hold exit before settlement when the
post-entry price path allows it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from polymarket_crypto_5min.exit_backtest import losing_trades, walk_forward_exit_backtest
from polymarket_crypto_5min.features import build_training_frame, load_candles, load_markets, load_poly_prices
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics
from polymarket_crypto_5min.walk_forward import WalkForwardConfig, make_side_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", default="data/raw/full_btc_5m/btc_5m_markets.csv")
    parser.add_argument("--btc-candles", default="data/raw/full_btc_5m/btc_usdt_1m.csv")
    parser.add_argument("--poly-prices", default="data/raw/full_btc_5m/btc_5m_price_history_combined.csv")
    parser.add_argument("--snapshot-seconds-before-close", type=int, default=45)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--initial-capital", type=float, default=2000.0)
    parser.add_argument("--train-markets", type=int, default=400)
    parser.add_argument("--test-markets", type=int, default=100)
    parser.add_argument("--min-train-trades", type=int, default=25)
    parser.add_argument("--min-bin-observations", type=int, default=30)
    parser.add_argument("--out-dir", default="data/processed/exit_aware_walk_forward")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markets = load_markets(args.markets)
    candles = load_candles(args.btc_candles)
    prices = load_poly_prices(args.poly_prices)
    frame = build_training_frame(
        markets,
        candles,
        prices,
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
    trades, folds, rules = walk_forward_exit_backtest(candidates, prices, config=config)

    trades.to_csv(out_dir / "exit_aware_trades.csv", index=False)
    losing_trades(trades).to_csv(out_dir / "exit_aware_losing_trades.csv", index=False)
    folds.to_csv(out_dir / "exit_aware_folds.csv", index=False)
    rules.to_csv(out_dir / "exit_aware_selected_rules.csv", index=False)

    metrics_frame = trades.copy()
    if not metrics_frame.empty:
        metrics_frame["bucket"] = "EXIT_AWARE"
        metrics_frame["end_dt"] = metrics_frame["exit_dt"]
        metrics_frame["won"] = metrics_frame["pnl_usdc"].gt(0)
    metrics = performance_metrics(metrics_frame, initial_capital=args.initial_capital)
    metrics.to_csv(out_dir / "exit_aware_metrics.csv", index=False)
    equity_curve(metrics_frame, initial_capital=args.initial_capital).to_csv(out_dir / "exit_aware_equity_curve.csv", index=False)

    detail_cols = [
        "fold",
        "condition_id",
        "market_id",
        "question",
        "side",
        "entry_dt",
        "exit_dt",
        "end_dt",
        "entry_price",
        "exit_price",
        "exit_reason",
        "pnl_usdc",
        "return_on_stake",
        "won_at_settlement",
        "loss_trade",
        "path_points_seen",
        "take_profit",
        "target_price",
        "stop_loss",
        "max_hold_seconds",
        "p_lower",
        "edge_lower",
        "score_bps",
        "abs_score_bps",
    ]
    if not trades.empty:
        trades[[col for col in detail_cols if col in trades.columns]].to_csv(out_dir / "exit_aware_trade_audit.csv", index=False)

    print(f"Wrote exit-aware outputs to {out_dir}")
    if not metrics.empty:
        print(metrics.to_string(index=False))
    if not folds.empty and not bool(folds["leakage_check_passed"].all()):
        raise SystemExit("Leakage check failed")


if __name__ == "__main__":
    main()
