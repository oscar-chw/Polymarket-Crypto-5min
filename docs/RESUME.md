# Resume checkpoint

Last updated: 2026-07-06 UTC.

## Current scope

The active research scope is **Bitcoin-only Polymarket crypto 5-minute Up/Down markets**.

Do not mix ETH/SOL/XRP markets into the current backtest until asset-specific candle joins are implemented. The current model uses BTC price features, so the correct full-history runner is BTC-only.

## Current repository state

Important files now in the repo:

```text
scripts/run_btc_5m_full_history_walk_forward.py   # all-history BTC 5m data pull + walk-forward
scripts/walk_forward_backtest.py                  # strict walk-forward base runner
scripts/exit_aware_walk_forward.py                # entry timing + early-exit validation
polymarket_crypto_5min/resolution.py              # settled Polymarket/Gamma outcome labels
polymarket_crypto_5min/walk_forward.py            # prior-only calibration/rule selection
polymarket_crypto_5min/exit_backtest.py           # take-profit/target/stop/max-hold exits
polymarket_crypto_5min/metrics.py                 # initial-capital anchored realized MDD
.github/workflows/full-btc-5m-walk-forward.yml    # long CI workflow
docs/backtesting.md                               # correctness notes
```

## Correct backtest principles

1. Payout labels come from settled Polymarket/Gamma outcomes, not external BTC candles.
2. BTC candles are features only.
3. Walk-forward folds must satisfy `train_end_dt < test_start_dt`.
4. Entry and exit parameters are selected on training folds only.
5. Report both base hold-to-settlement and exit-aware variants.
6. MDD is currently realized-equity MDD anchored to initial capital. Mark-to-market MDD requires orderbook snapshots during the open position.

## Last attempted full run

PR:

```text
PR #4: [codex] full BTC 5m walk-forward validation
Branch: codex/full-run
Head SHA: 8e4b52943c8df61dfbfc1aea363dba0c8ad032f2
```

Workflow runs for that head SHA:

```text
backtest-metrics
  run_id: 28752413912
  status: completed
  conclusion: failure
  artifact: backtest-outputs
  artifact_id: 8095066280
  artifact_size_bytes: 9140360
  artifact_expires_at: 2026-10-03T19:34:44Z

full-btc-5m-walk-forward
  run_id: 28752413983
  status: completed
  conclusion: cancelled
  job_id: 85253751959
  artifact: full-btc-5m-walk-forward
  artifact_id: 8097971730
  artifact_size_bytes: 10307335
  artifact_expires_at: 2026-10-03T19:34:44Z
```

The full workflow step status was:

```text
Install package: success
Run tests: success
Run full BTC 5m walk-forward: cancelled
Run exit-aware walk-forward: skipped
Upload outputs: success
```

This means the heavy all-history data pull did not finish, but a partial artifact exists and should be inspected before rerunning from scratch.

## How to resume locally from repo code

Fresh all-history run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

python scripts/run_btc_5m_full_history_walk_forward.py \
  --initial-capital 2000 \
  --stake 10 \
  --train-markets 400 \
  --test-markets 100 \
  --price-source both

python scripts/exit_aware_walk_forward.py \
  --initial-capital 2000 \
  --stake 10 \
  --train-markets 400 \
  --test-markets 100
```

Resume from already-created local CSVs:

```bash
python scripts/run_btc_5m_full_history_walk_forward.py \
  --skip-existing \
  --initial-capital 2000 \
  --stake 10 \
  --train-markets 400 \
  --test-markets 100 \
  --price-source both
```

Expected output directories:

```text
data/raw/full_btc_5m/
data/processed/full_btc_5m_walk_forward/
data/processed/exit_aware_walk_forward/
```

## How to resume from GitHub Actions artifact

The previous long workflow uploaded a partial artifact. Download artifact `8097971730` before it expires, unpack it, and inspect:

```text
data/raw/full_btc_5m/*.csv
data/processed/full_btc_5m_walk_forward/*.csv
data/processed/full_btc_5m_walk_forward/*.json
data/processed/exit_aware_walk_forward/*.csv
data/processed/exit_aware_walk_forward/*.json
```

Then copy the raw CSVs into `data/raw/full_btc_5m/` and run:

```bash
python scripts/run_btc_5m_full_history_walk_forward.py --skip-existing \
  --initial-capital 2000 \
  --stake 10 \
  --train-markets 400 \
  --test-markets 100 \
  --price-source both

python scripts/exit_aware_walk_forward.py \
  --initial-capital 2000 \
  --stake 10 \
  --train-markets 400 \
  --test-markets 100
```

## Strategy currently being validated

Base entry engine:

```text
- Build UP and DOWN candidate rows for each resolved BTC 5m market.
- Calibrate p_win using prior markets only.
- Select entry thresholds on the training fold only:
  - minimum lower-bound edge
  - minimum lower-bound win probability
  - entry price band
  - minimum BTC absolute move
- Apply selected rule to the next unseen test fold.
```

Exit-aware engine:

```text
- Enter only on calibrated confirmation.
- After entry, scan post-entry price history for the same token.
- Exit early by training-selected policy:
  - take-profit
  - target-price
  - stop-loss
  - max-hold seconds
- If no exit triggers, hold to settlement.
```

Current exit policy grid:

```text
take_profit: 0.03, 0.05, 0.08, 0.10, 0.15, disabled
target_price: 0.75, 0.80, 0.85, 0.90, 0.95, disabled
stop_loss: 0.03, 0.05, 0.08, 0.10, disabled
max_hold_seconds: 10, 20, 30, hold to settlement
```

## What to check after the next completed run

Minimum proof gates before scaling real money:

```text
resolved markets >= 2,000
out-of-sample trades >= 300
leakage violations = 0
profitable after +1c and +2c slippage stress
realized MDD on $2,000 < 5-8%
no single day or single trade dominates PnL
exit-aware variant improves downside or Sharpe versus hold-to-settlement
```

Files to inspect first:

```text
data/processed/full_btc_5m_walk_forward/manifest.json
data/processed/full_btc_5m_walk_forward/walk_forward_metrics.csv
data/processed/full_btc_5m_walk_forward/walk_forward_folds.csv
data/processed/full_btc_5m_walk_forward/walk_forward_trades.csv
data/processed/exit_aware_walk_forward/exit_aware_metrics.csv
data/processed/exit_aware_walk_forward/exit_aware_trade_audit.csv
data/processed/exit_aware_walk_forward/exit_aware_losing_trades.csv
```

## Current caveats

- The last full workflow was cancelled before completion, so no full-history result exists yet.
- The partial artifact is useful for resuming/debugging but should not be treated as a completed backtest.
- Realized MDD is fixed to count first-trade losses; mark-to-market MDD still requires orderbook snapshots during open positions.
- If the CLOB price-history crawl remains too slow, split the run by date ranges and concatenate raw CSVs before the walk-forward stage.
