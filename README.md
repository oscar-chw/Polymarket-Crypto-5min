# Polymarket Crypto 5-minute research bot

Research tooling for Bitcoin 5-minute Polymarket **Up/Down** markets.

This repository does **not** promise profit and does **not** place live orders. It gives you a repeatable way to download historical markets, join them with Bitcoin 1-minute price data, backtest two threshold buckets, and print dry-run live signals.

## Strategy idea

The strategy is split into two buckets:

1. **Confirmation bucket** — BTC has moved far enough above or below the market-start price near the end of the 5-minute window. The bot estimates whether the direction is already highly likely and only keeps trades with positive expected value after conservative taker fees.
2. **Value-mismatch bucket** — Polymarket's buy price is still low, but the BTC move and short-term momentum imply a materially higher probability than the market price. This searches for “market says unlikely, external data says likely” situations.

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
data/processed/default_strategy_summary.csv
```

The report searches both buckets independently. Good candidate threshold rows should have enough trades, positive ROI, realistic average buy prices, and stable performance across nearby thresholds. Do not pick the single best row blindly; that is usually overfitting.

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
  live_signal.py   # dry-run live scanner
scripts/
  download_history.py
  backtest_thresholds.py
  live_signal.py
tests/
  test_strategy.py
```

## Safety notes

Use this as a research/backtesting base. The strategy can fail due to stale data, latency, spreads, fees, insufficient liquidity, API delays, market rule nuances, and overfit thresholds. Respect Polymarket availability, jurisdiction, API limits, and terms before trading manually or building execution on top.
