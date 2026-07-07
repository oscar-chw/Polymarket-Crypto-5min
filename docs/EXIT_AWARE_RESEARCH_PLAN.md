# Exit-aware 5-minute research plan

Status: research-only. Do not deploy from this file alone.

## Mechanism

The base 5-minute strategy enters when the walk-forward calibrated probability
exceeds all-in taker cost. Exit-aware research tests whether post-entry
Polymarket price paths can reduce settlement-tail risk or capture profit before
resolution.

## Current evidence

- Earlier compact grid: negative OOS PnL and rejected.
- Grouped/cached bounded-grid smoke: positive OOS, but it was explored during
  research and is not a deployable proof.

## Pre-registered gate run

Use cached side candidates from the recent 3,000-market walk-forward run to
avoid feature-regeneration differences:

```bash
python scripts/exit_aware_walk_forward.py \
  --side-candidates data/processed/full_btc_5m_walk_forward_recent3000/side_candidates.csv \
  --out-dir data/processed/exit_aware_walk_forward_preregistered_grid \
  --grid-min-edge-lower 0,0.0025,0.005,0.01 \
  --grid-min-p-lower 0.60,0.65,0.70,0.75 \
  --grid-min-price 0.50,0.60,0.70,0.80 \
  --grid-max-price 0.95,0.975 \
  --grid-min-abs-score-bps 0,4,8,12 \
  --grid-take-profit 0.03,0.05,0.08,0.10 \
  --grid-target-price 0.80,0.85,0.90,none \
  --grid-stop-loss 0.03,0.05,0.08 \
  --grid-max-hold-seconds 10,20,30,none
```

## Acceptance checks

- Leakage check true on every fold.
- OOS trade count high enough to avoid one/few-trade dominance.
- Positive net PnL after taker entry and conservative taker-like exit fees.
- Max realized drawdown below 2% of USD 2,000 in the research run.
- Profit factor above 1.5 and not dominated by one fold/day.
- Selected rules vary within a stable plateau, not a single brittle parameter.
- No live deployment until executable depth, latency, partial fills, and
  cancellation behavior are measured in paper/live logs.

## 2026-07-07 result

The pre-registered broad grid completed with 512 entry rules x 192 exit
policies over 5,965 cached candidate side rows and 26 walk-forward folds.
Leakage checks passed. OOS result: 58 trades, $52.74 net PnL on a $2,000
capital standard, 2.64% return, $-15.54 max realized drawdown, 2.27 profit
factor, and 91.4% win rate.

Interpretation: candidate research gate passed, but live deployment remains
blocked until executable depth, latency, partial-fill, cancellation, and longer
paper-trade evidence are collected.
