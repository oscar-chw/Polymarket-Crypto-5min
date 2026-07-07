#!/usr/bin/env python3
"""Run exit-aware walk-forward backtests.

This tests the live-trading idea: enter on calibrated confirmation, then take
profit / target-price exit / stop-loss / max-hold exit before settlement when the
post-entry price path allows it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from polymarket_crypto_5min.exit_backtest import DEFAULT_EXIT_POLICY_GRID, generate_exit_policy_grid
from polymarket_crypto_5min.exit_backtest import losing_trades, walk_forward_exit_backtest
from polymarket_crypto_5min.features import build_training_frame, load_candles, load_markets, load_poly_prices
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics
from polymarket_crypto_5min.walk_forward import DEFAULT_RULE_GRID, WalkForwardConfig, generate_rule_grid, make_side_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", default="data/raw/full_btc_5m/btc_5m_markets.csv")
    parser.add_argument("--btc-candles", default="data/raw/full_btc_5m/btc_usdt_1m.csv")
    parser.add_argument("--poly-prices", default="data/raw/full_btc_5m/btc_5m_price_history_combined.csv")
    parser.add_argument("--feature-frame", help="Optional prebuilt feature_frame.csv to reuse")
    parser.add_argument("--side-candidates", help="Optional prebuilt side_candidates.csv to reuse")
    parser.add_argument("--snapshot-seconds-before-close", type=int, default=45)
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--initial-capital", type=float, default=2000.0)
    parser.add_argument("--train-markets", type=int, default=400)
    parser.add_argument("--test-markets", type=int, default=100)
    parser.add_argument("--min-train-trades", type=int, default=25)
    parser.add_argument("--min-bin-observations", type=int, default=30)
    parser.add_argument("--out-dir", default="data/processed/exit_aware_walk_forward")
    parser.add_argument("--grid-min-edge-lower", default=_grid_default(DEFAULT_RULE_GRID["min_edge_lower"]))
    parser.add_argument("--grid-min-p-lower", default=_grid_default(DEFAULT_RULE_GRID["min_p_lower"]))
    parser.add_argument("--grid-min-price", default=_grid_default(DEFAULT_RULE_GRID["min_price"]))
    parser.add_argument("--grid-max-price", default=_grid_default(DEFAULT_RULE_GRID["max_price"]))
    parser.add_argument("--grid-min-abs-score-bps", default=_grid_default(DEFAULT_RULE_GRID["min_abs_score_bps"]))
    parser.add_argument("--grid-take-profit", default=_grid_default(DEFAULT_EXIT_POLICY_GRID["take_profit"]))
    parser.add_argument("--grid-target-price", default=_grid_default(DEFAULT_EXIT_POLICY_GRID["target_price"]))
    parser.add_argument("--grid-stop-loss", default=_grid_default(DEFAULT_EXIT_POLICY_GRID["stop_loss"]))
    parser.add_argument("--grid-max-hold-seconds", default=_grid_default(DEFAULT_EXIT_POLICY_GRID["max_hold_seconds"]))
    parser.add_argument("--run-manifest", default="exit_aware_run_manifest.json")
    return parser.parse_args()


def _grid_default(values: object) -> str:
    return ",".join("none" if value is None else str(value) for value in values)


def _parse_float_grid(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_optional_float_grid(value: str) -> list[float | None]:
    out: list[float | None] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        out.append(None if token in {"none", "null"} else float(token))
    return out


def _parse_optional_int_grid(value: str) -> list[int | None]:
    out: list[int | None] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        out.append(None if token in {"none", "null"} else int(token))
    return out


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_poly_prices(args.poly_prices)
    if args.side_candidates:
        candidates = pd.read_csv(args.side_candidates)
        frame = pd.DataFrame()
    else:
        if args.feature_frame:
            frame = pd.read_csv(args.feature_frame)
        else:
            markets = load_markets(args.markets)
            candles = load_candles(args.btc_candles)
            frame = build_training_frame(
                markets,
                candles,
                prices,
                snapshot_seconds_before_close=args.snapshot_seconds_before_close,
                require_resolved_outcome=True,
                allow_gamma_prices=False,
            )
        candidates = make_side_candidates(frame)
    frame.to_csv(out_dir / "feature_frame.csv", index=False)
    candidates.to_csv(out_dir / "side_candidates.csv", index=False)

    config = WalkForwardConfig(
        initial_capital=args.initial_capital,
        stake_usdc=args.stake,
        train_markets=args.train_markets,
        test_markets=args.test_markets,
        min_train_trades=args.min_train_trades,
        min_bin_observations=args.min_bin_observations,
    )
    rule_grid = {
        "min_edge_lower": _parse_float_grid(args.grid_min_edge_lower),
        "min_p_lower": _parse_float_grid(args.grid_min_p_lower),
        "min_price": _parse_float_grid(args.grid_min_price),
        "max_price": _parse_float_grid(args.grid_max_price),
        "min_abs_score_bps": _parse_float_grid(args.grid_min_abs_score_bps),
    }
    exit_policy_grid = {
        "take_profit": _parse_optional_float_grid(args.grid_take_profit),
        "target_price": _parse_optional_float_grid(args.grid_target_price),
        "stop_loss": _parse_optional_float_grid(args.grid_stop_loss),
        "max_hold_seconds": _parse_optional_int_grid(args.grid_max_hold_seconds),
    }

    entry_rules = generate_rule_grid(rule_grid)
    exit_policies = generate_exit_policy_grid(exit_policy_grid)
    trades, folds, rules = walk_forward_exit_backtest(
        candidates,
        prices,
        config=config,
        entry_rules=entry_rules,
        exit_policies=exit_policies,
    )

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

    manifest = {
        "script": "scripts/exit_aware_walk_forward.py",
        "runtime_seconds": time.perf_counter() - started,
        "inputs": {
            "markets": args.markets,
            "btc_candles": args.btc_candles,
            "poly_prices": args.poly_prices,
            "feature_frame": args.feature_frame,
            "side_candidates": args.side_candidates,
            "snapshot_seconds_before_close": args.snapshot_seconds_before_close,
        },
        "outputs": {
            "out_dir": str(out_dir),
            "feature_frame": str(out_dir / "feature_frame.csv"),
            "side_candidates": str(out_dir / "side_candidates.csv"),
            "trades": str(out_dir / "exit_aware_trades.csv"),
            "metrics": str(out_dir / "exit_aware_metrics.csv"),
            "folds": str(out_dir / "exit_aware_folds.csv"),
            "selected_rules": str(out_dir / "exit_aware_selected_rules.csv"),
        },
        "walk_forward_config": asdict(config),
        "entry_rule_grid": rule_grid,
        "exit_policy_grid": exit_policy_grid,
        "entry_rule_count": len(entry_rules),
        "exit_policy_count": len(exit_policies),
        "candidate_rows": int(len(candidates)),
        "feature_rows": int(len(frame)) if not frame.empty else None,
        "fold_rows": int(len(folds)),
        "trade_rows": int(len(trades)),
        "leakage_passed": bool(folds.empty or folds["leakage_check_passed"].all()),
        "research_status": "pre-deployment research; rerun with pre-registered grid before any live claim",
    }
    (out_dir / args.run_manifest).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"Wrote exit-aware outputs to {out_dir}")
    if not metrics.empty:
        print(metrics.to_string(index=False))
    if not folds.empty and not bool(folds["leakage_check_passed"].all()):
        raise SystemExit("Leakage check failed")


if __name__ == "__main__":
    main()
