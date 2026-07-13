# Polymarket Crypto 5-minute research bot

Research tooling for Bitcoin 5-minute Polymarket **Up/Down** markets.

This repository does **not** promise profit and does **not** place live orders. It gives you a repeatable way to download historical markets, join them with Bitcoin 1-minute price data, backtest two threshold buckets, and print dry-run live signals.

## Strategy idea

The strategy is split into two research buckets:

1. **Confirmation bucket** - BTC has moved far enough above or below the market-start price near the end of the five-minute window. The model retains a direction only when its prior-only calibrated lower bound clears the entry-price and score gates.
2. **Value-mismatch bucket** - Polymarket's observed side price is below the prior-only calibrated probability by enough to clear the fee-adjusted edge gate.

At decision time `d`, with market start `S` and one-minute Binance candles indexed by the first timestamp at which their final values are available:

```text
displacement_bps = 10,000 * (BTC_close_asof(d) / BTC_close_asof(S) - 1)
momentum_1m_bps  = 10,000 * (BTC_close_asof(d) / BTC_close_asof(d - 60s) - 1)
momentum_3m_bps  = 10,000 * (BTC_close_asof(d) / BTC_close_asof(d - 180s) - 1)
time_scale       = sqrt(max(seconds_left, 1) / 60)
z                = displacement_bps / max(8 * time_scale, 1)
                   + 0.20 * momentum_1m_bps / 8
                   + 0.08 * momentum_3m_bps / 8
raw_p_up         = clip(sigmoid(z), 0.001, 0.999)
```

For each chronological test fold, probability calibration and rule selection use only earlier training rows. Price bins and score-magnitude bins are right-closed with the lowest interval included; empirical rates receive beta-binomial smoothing; sparse cells fall back to broader prior-only cells; and a one-sided normal-approximation lower bound is used by the entry gate. Test-fold outcomes never update that fold's calibration or selected rule.

For a candidate buy at price `p`, the research model uses:

```text
fee_per_share = fee_rate * p * (1 - p)
expected_value_per_share = model_probability - p - fee_per_share
```

The default crypto taker fee rate is `0.07`. Makers can be fee-free, but the backtest assumes taker execution because a 5-minute strategy needs conservative fills.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Download historical data

Start small while testing API shape and rate limits:

```bash
python scripts/download_history.py --max-pages 2
```

Full crawl:

```bash
python scripts/download_history.py
```

Outputs:

```text
data/raw/btc_5m_markets.csv              # Gamma market/event metadata
data/raw/btc_5m_up_price_history.csv     # CLOB /prices-history for UP tokens
data/raw/btc_5m_data_api_trades.csv      # public Data API trade rows by condition ID
data/raw/btc_usdt_1m.csv                 # Binance BTCUSDT 1m candles covering markets
```

Useful variants:

```bash
# If tag filtering misses older markets, remove it:
python scripts/download_history.py --tag-slug "" --title-search Bitcoin

# Focus on a date range:
python scripts/download_history.py --start-date-min 2026-01-01T00:00:00Z

# Only fetch market metadata first:
python scripts/download_history.py --skip-clob-history --skip-trades --skip-binance
```

## Backtest thresholds

```bash
python scripts/backtest_thresholds.py
```

Outputs:

```text
data/processed/training_frame.csv
data/processed/threshold_report.csv
data/processed/default_strategy_trades.csv
data/processed/default_strategy_summary.csv
data/processed/default_strategy_metrics.csv
data/processed/default_strategy_equity_curve.csv
```

`default_strategy_metrics.csv` includes overall and per-bucket win rate, ROI, trade Sharpe, daily Sharpe, Sortino, maximum drawdown in USDC and percent, profit factor, and average EV. Use `--initial-capital` to change the denominator used for equity/drawdown percentages and daily returns:

```bash
python scripts/backtest_thresholds.py --initial-capital 1000 --stake 10
```

The report searches both buckets independently. Good candidate threshold rows should have enough trades, positive ROI, realistic average buy prices, and stable performance across nearby thresholds. Do not pick the single best row blindly; that is usually overfitting.

## Exit-aware walk-forward research

Exit-aware tests simulate entering from the calibrated walk-forward signal and leaving before settlement when the price path reaches a take-profit, target, stop-loss, or max-hold rule. This is research-only until a broader pre-registered grid is stable OOS and executable orderbook depth is validated.

```bash
python scripts/exit_aware_walk_forward.py \
  --out-dir data/processed/exit_aware_walk_forward \
  --grid-take-profit 0.03,0.05,0.08,0.10 \
  --grid-target-price 0.80,0.90,none \
  --grid-stop-loss 0.03,0.05,0.08 \
  --grid-max-hold-seconds 10,20,30,none
```

For repeated grid experiments, reuse cached inputs instead of rebuilding features:

```bash
python scripts/exit_aware_walk_forward.py \
  --side-candidates data/processed/full_btc_5m_walk_forward_recent3000/side_candidates.csv \
  --out-dir data/processed/exit_aware_walk_forward_cached
```

Each run writes `exit_aware_run_manifest.json` with input hashes, source hashes, grids, fold count, trade count, feature-availability gates, leakage status, environment versions, and runtime. Legacy Binance files whose `ts` is the candle open are shifted exactly once to close availability; already availability-stamped files are never shifted again.

## Dry-run live signal

```bash
python scripts/live_signal.py
```

Each line is JSON like:

```json
{
  "action": "PAPER_BUY_UP",
  "bucket": "CONFIRMATION",
  "buy_price": 0.67,
  "direction": "UP",
  "expected_value_per_share": 0.03,
  "model_prob": 0.73,
  "score_bps": 14.2,
  "seconds_left": 41.8
}
```

`PAPER_BUY_UP` and `PAPER_BUY_DOWN` are dry-run labels only. No private keys, API keys, signatures, or order placement code are included.

## How the data joins work

The downloader stores one row per Polymarket condition/market with token IDs for UP and DOWN when available. The backtest then:

1. Uses BTCUSDT 1-minute candles to estimate BTC start price, snapshot price, and end price.
2. Uses the CLOB UP-token price history at the same snapshot timestamp. DOWN price is taken from DOWN-token history when present, otherwise approximated as `1 - UP`.
3. Determines the realized outcome from BTC end price versus start price.
4. Computes model probability, fees, expected value, win/loss, and PnL per fixed stake.

By default, closed-market Gamma outcome prices are **not** used in historical backtests because they may include post-resolution information. Use `--allow-gamma-prices` only for debugging.

## Files

```text
polymarket_crypto_5min/
  clients.py       # public HTTP clients for Gamma, Data API, CLOB, Binance
  downloader.py    # market discovery, flattening, price/trade/candle downloads
  features.py      # point-in-time features and EV math
  backtest.py      # threshold grids and fixed-stake simulation
  metrics.py       # Sharpe, Sortino, max drawdown, equity curve
  exit_backtest.py # exit-aware price-path simulation
  live_signal.py   # dry-run live scanner
scripts/
  download_history.py
  backtest_thresholds.py
  exit_aware_walk_forward.py
  live_signal.py
tests/
  test_strategy.py
```

## Safety notes

Use this as a research/backtesting base. The strategy can fail due to stale data, latency, spreads, fees, insufficient liquidity, API delays, market rule nuances, and overfit thresholds. Respect Polymarket availability, jurisdiction, API limits, and terms before trading manually or building execution on top.
