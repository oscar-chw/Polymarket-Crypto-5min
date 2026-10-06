# Backtesting correctness

Use `scripts/walk_forward_backtest.py` for reported results.

Key rules:

1. Payout labels come from settled Polymarket/Gamma market outcome data, not from external BTC candles.
2. External BTC data is used only for features such as start price, snapshot price, momentum, and diagnostic direction.
3. Each walk-forward fold trains on earlier markets and tests later markets only.
4. The fold report includes `leakage_check_passed`; reported runs should require it to be true for every fold.
5. Entry prices follow `ENTRY_PRICE_RULE` in `features.py`: the last CLOB/trade print inside the market window, at or
   before the decision, and no older than `max_entry_price_age_seconds` (default 60 s); otherwise the row has no
   price. This is a print, not the ask that `live_signal.py` buys at, so backtest fills are better than live fills by
   at least half the spread.
6. Fixed-threshold reports from `scripts/backtest_thresholds.py` are smoke tests, not final performance reports.

Example for a $2,000 USD bankroll:

```bash
python scripts/walk_forward_backtest.py \
  --initial-capital 2000 \
  --stake 10 \
  --train-markets 400 \
  --test-markets 100
```

Outputs are written under `data/processed/walk_forward/`:

```text
feature_frame.csv
side_candidates.csv
walk_forward_trades.csv
walk_forward_folds.csv
walk_forward_selected_rules.csv
walk_forward_metrics.csv
walk_forward_equity_curve.csv
```
