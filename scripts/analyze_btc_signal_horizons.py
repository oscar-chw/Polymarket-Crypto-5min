#!/usr/bin/env python3
"""Recompute close-availability-correct BTC five-minute horizon diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from polymarket_crypto_5min.features import load_candles, load_markets
from polymarket_crypto_5min.signal_diagnostics import (
    DEFAULT_HORIZONS_SECONDS,
    build_horizon_diagnostics,
    json_safe,
)

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", required=True)
    parser.add_argument("--btc-candles", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS_SECONDS))
    parser.add_argument("--fold-market-count", type=int, default=100)
    parser.add_argument(
        "--generated-utc",
        help="Optional frozen ISO-8601 generation time for byte-reproducible artifacts",
    )
    return parser.parse_args()


def _file_record(path_value: str | Path) -> dict[str, object]:
    path = Path(path_value).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    display = path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else str(path_value).replace("\\", "/")
    return {
        "path": display,
        "provenance_runtime_absolute_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    horizons = [int(token.strip()) for token in args.horizons.split(",") if token.strip()]
    markets = load_markets(args.markets)
    candles = load_candles(args.btc_candles)
    details, folds, summary, metadata = build_horizon_diagnostics(
        markets,
        candles,
        horizons_seconds=horizons,
        fold_market_count=args.fold_market_count,
    )
    if args.generated_utc:
        generated_at = pd.Timestamp(args.generated_utc)
        generated_at = (
            generated_at.tz_localize("UTC") if generated_at.tzinfo is None else generated_at.tz_convert("UTC")
        )
        generated_utc = generated_at.isoformat()
    else:
        generated_utc = datetime.now(timezone.utc).isoformat()
    metadata.update(
        {
            "generated_utc": generated_utc,
            "inputs": {
                "markets": _file_record(args.markets),
                "btc_candles": _file_record(args.btc_candles),
            },
            "source_files": {
                "scripts/analyze_btc_signal_horizons.py": _file_record(__file__),
                "polymarket_crypto_5min/signal_diagnostics.py": _file_record(
                    Path(__file__).resolve().parents[1] / "polymarket_crypto_5min" / "signal_diagnostics.py"
                ),
                "polymarket_crypto_5min/features.py": _file_record(
                    Path(__file__).resolve().parents[1] / "polymarket_crypto_5min" / "features.py"
                ),
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
    )
    details.to_csv(out / "btc_signal_horizon_rows.csv", index=False)
    folds.to_csv(out / "btc_signal_fold_ic.csv", index=False)
    summary.to_csv(out / "btc_signal_icir_by_horizon.csv", index=False)
    (out / "btc_signal_icir_decay_metrics.json").write_text(
        json.dumps(json_safe(metadata), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(metadata), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
