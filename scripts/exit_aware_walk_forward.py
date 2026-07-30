#!/usr/bin/env python3
"""Run exit-aware walk-forward backtests.

This tests the live-trading idea: enter on calibrated confirmation, then take
profit / target-price exit / stop-loss / max-hold exit before settlement when the
post-entry price path allows it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from polymarket_crypto_5min.exit_backtest import (
    DEFAULT_EXIT_POLICY_GRID,
    generate_exit_policy_grid,
    losing_trades,
    walk_forward_exit_backtest,
)
from polymarket_crypto_5min.features import build_training_frame, load_candles, load_markets, load_poly_prices
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics
from polymarket_crypto_5min.walk_forward import (
    DEFAULT_RULE_GRID,
    WalkForwardConfig,
    generate_rule_grid,
    make_side_candidates,
)

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path_value: str | Path, *, repo_relative: bool = False) -> dict[str, object]:
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    display = (
        path.relative_to(REPO).as_posix()
        if repo_relative and path.is_relative_to(REPO)
        else str(path_value).replace("\\", "/")
    )
    return {
        "path": display,
        "provenance_runtime_absolute_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


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


def _grid_default(values: Iterable[object]) -> str:
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
    if prices is None:
        raise FileNotFoundError(args.poly_prices)
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
    exit_policy_grid: dict[str, Iterable[float | int | None]] = {
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
    equity_curve(metrics_frame, initial_capital=args.initial_capital).to_csv(
        out_dir / "exit_aware_equity_curve.csv", index=False
    )

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
        "feature_available_at",
        "feature_availability_passed",
        "btc_start_available_at",
        "btc_snapshot_available_at",
        "btc_1m_ago_available_at",
        "btc_3m_ago_available_at",
    ]
    if not trades.empty:
        trades[[col for col in detail_cols if col in trades.columns]].to_csv(
            out_dir / "exit_aware_trade_audit.csv", index=False
        )

    output_paths = {
        "feature_frame": out_dir / "feature_frame.csv",
        "side_candidates": out_dir / "side_candidates.csv",
        "trades": out_dir / "exit_aware_trades.csv",
        "losing_trades": out_dir / "exit_aware_losing_trades.csv",
        "metrics": out_dir / "exit_aware_metrics.csv",
        "equity_curve": out_dir / "exit_aware_equity_curve.csv",
        "folds": out_dir / "exit_aware_folds.csv",
        "selected_rules": out_dir / "exit_aware_selected_rules.csv",
    }
    trade_audit_path = out_dir / "exit_aware_trade_audit.csv"
    if trade_audit_path.is_file():
        output_paths["trade_audit"] = trade_audit_path
    source_paths = [
        Path(__file__),
        REPO / "polymarket_crypto_5min" / "clients.py",
        REPO / "polymarket_crypto_5min" / "features.py",
        REPO / "polymarket_crypto_5min" / "walk_forward.py",
        REPO / "polymarket_crypto_5min" / "exit_backtest.py",
        REPO / "polymarket_crypto_5min" / "metrics.py",
    ]
    feature_availability_present = {
        "feature_available_at",
        "feature_availability_passed",
    }.issubset(frame.columns)
    feature_availability_passed = bool(
        feature_availability_present and not frame.empty and frame["feature_availability_passed"].fillna(False).all()
    )
    manifest = {
        "schema_version": "exit_aware_walk_forward_v2_point_in_time_availability",
        "script": "scripts/exit_aware_walk_forward.py",
        "runtime_seconds": time.perf_counter() - started,
        "inputs": {
            "markets": _file_record(args.markets),
            "btc_candles": _file_record(args.btc_candles),
            "poly_prices": _file_record(args.poly_prices),
            "feature_frame": _file_record(args.feature_frame) if args.feature_frame else None,
            "side_candidates": _file_record(args.side_candidates) if args.side_candidates else None,
            "snapshot_seconds_before_close": args.snapshot_seconds_before_close,
        },
        "outputs": {name: _file_record(path) for name, path in output_paths.items()},
        "source_files": {
            path.relative_to(REPO).as_posix(): _file_record(path, repo_relative=True) for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": __import__("numpy").__version__,
            "pandas": pd.__version__,
            "executable_provenance_absolute_path": sys.executable,
        },
        "walk_forward_config": asdict(config),
        "entry_rule_grid": rule_grid,
        "exit_policy_grid": exit_policy_grid,
        "entry_rule_count": len(entry_rules),
        "exit_policy_count": len(exit_policies),
        "candidate_rows": int(len(candidates)),
        "feature_rows": int(len(frame)) if not frame.empty else None,
        "feature_timestamp_semantics": "Binance final OHLCV indexed at close availability",
        "feature_availability_columns_present": feature_availability_present,
        "feature_availability_passed": feature_availability_passed,
        "feature_availability_failed_rows": (
            int((~frame["feature_availability_passed"].fillna(False)).sum()) if feature_availability_present else None
        ),
        "feature_available_at_max_utc": (
            pd.to_datetime(frame["feature_available_at"], utc=True, errors="coerce").max()
            if feature_availability_present
            else None
        ),
        "fold_rows": int(len(folds)),
        "trade_rows": int(len(trades)),
        "chronological_fold_leakage_passed": bool(folds.empty or folds["leakage_check_passed"].all()),
        "leakage_passed": bool(feature_availability_passed and (folds.empty or folds["leakage_check_passed"].all())),
        "policy_matrix_status": "unavailable_full_configured_policy_by_time_return_matrix",
        "research_status": "research_only_not_deployable",
    }
    (out_dir / args.run_manifest).write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote exit-aware outputs to {out_dir}")
    if not metrics.empty:
        print(metrics.to_string(index=False))
    if not folds.empty and not bool(folds["leakage_check_passed"].all()):
        raise SystemExit("Leakage check failed")


if __name__ == "__main__":
    main()
