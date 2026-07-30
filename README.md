# Point-in-Time Polymarket Bitcoin Research

Research-only Python tooling for point-in-time, walk-forward analysis of Polymarket Bitcoin five-minute Up/Down markets; it downloads public data and simulates strategies but never places orders.

## Status & honesty

The availability-safe selected result is negative. The exact `ALL` row in the local ignored run artifact `data/processed/exit_aware_walk_forward_availability_safe_grid/exit_aware_metrics.csv:2` records: `trades=16`, `wins=12`, `win_rate=0.75`, `total_pnl_usdc=-12.311260584407641`, `total_stake_usdc=160.0`, `roi_on_stake=-0.07694537865254776`, `return_on_initial_capital=-0.006155630292203821`, `trade_sharpe=-0.1387264673847573`, `daily_sharpe=-7.567719409564401`, `max_drawdown_usdc=-34.032470975482056`, `max_drawdown_pct=-0.016923230624928485`, and `profit_factor=0.692218485389809`.

**Status label:** selected chronological walk-forward OOS; trial-exposed, modeled historical; not live, cash, or executable proof. The corresponding manifest records final Binance OHLCV at candle-close availability, calibration of each test fold from preceding training markets only, exclusion of unresolved markets and post-resolution Gamma prices from features by default, and selection exposed across `512` entry rules and `192` exit policies. PBO and deflated Sharpe are unavailable because the full policy-by-time return matrix was not retained. The earlier positive `$52.74` OOS figure in `docs/EXIT_AWARE_RESEARCH_PLAN.md` predates the candle-availability correction and is withdrawn.

![Selected availability-safe modeled historical OOS equity; trial-exposed and not live evidence](docs/availability_safe_oos_equity_curve.png)

The chart is derived from the ignored local artifact `data/processed/exit_aware_walk_forward_availability_safe_grid/exit_aware_equity_curve.csv` (SHA-256 `88a89e7575bc8a65ead09167d7eb428c46daa7d91b585af2b9c42e2a9cf82d1b`) and is labeled modeled historical OOS, trial-exposed, and non-executable.

## Architecture

- `clients.py` and `downloader.py` retrieve public Gamma market metadata, CLOB price histories, Data API trades, and Binance `BTCUSDT` one-minute candles with bounded retries.
- `features.py` joins each decision to only already-available candle closes, uses settled Polymarket outcomes as labels, excludes unresolved markets, and does not use terminal Gamma prices unless explicitly enabled for diagnostics.
- `walk_forward.py` expands UP/DOWN candidates, estimates prior-only empirical calibration, selects a rule on preceding markets, and evaluates the next non-overlapping chronological fold.
- `exit_backtest.py` evaluates take-profit, target-price, stop-loss, and maximum-hold policies with taker-like fees charged on entry and simulated early exit.
- `metrics.py` and the runner scripts emit trade logs, fold reports, realized equity/drawdown metrics, source hashes, configuration, and leakage checks.

## The interesting decision

Final OHLCV values are indexed at the first timestamp when the exchange candle is complete, not at candle open. That conservative clock is propagated through as-of feature joins, and each test fold is calibrated only from markets that ended earlier. The tradeoff is fewer usable observations and a worse reported result, but the result is auditable without consuming future prices or test-fold outcomes.

## Provenance

- Repository history starts at commit `0543dfdc5ef21612aaede78c55a14bbd98ea538d` on `2026-07-05T22:02:23+08:00`.
- The negative metrics source is a local generated artifact intentionally excluded from Git; SHA-256: `d224a8c990fb62572a90c6ed3eef30d831dd00a29dfbdf291f768edbf31903cb`.
- Its run manifest SHA-256 is `2f215ac951821d34a61bdab0a8c1a61f6e938c306a5e6019067e63153d3b9edd`; it records `3,000` feature rows, `5,965` candidate rows, `26` folds, `16` selected trades, and passing feature-availability and chronological-leakage gates.
- Manifest input hashes: markets `d14b8aeda6b457372616d6395b0accb2d5da259b099f3d1ef587d19f338e1f2b`, Binance candles `a740cb1dfaba38bb39f5d42d0862d43ab8ff7c0a2f3595d3ad0d9399eca25778`, and combined Polymarket prices `fcc590c19fb8b0e094b4e33e34b4c8dfeaa96a7fbfca3dc9d019ab1c23a3d525`.
- Project code is MIT-licensed; the public APIs and downloaded market data remain subject to their providers' terms.

## Run it

```powershell
git clone https://github.com/hihihhi/Polymarket-Crypto-5min.git
cd Polymarket-Crypto-5min
uv sync --extra dev --frozen
uv run pytest -q

# Bounded public-data download for an API-shape check.
uv run python scripts/download_history.py --max-pages 2

# Full public-data acquisition and chronological base walk-forward.
uv run python scripts/run_btc_5m_full_history_walk_forward.py `
  --initial-capital 2000 `
  --stake 10 `
  --train-markets 400 `
  --test-markets 100 `
  --price-source both

# Exit-aware run reusing the full-history inputs.
uv run python scripts/exit_aware_walk_forward.py `
  --initial-capital 2000 `
  --stake 10 `
  --train-markets 400 `
  --test-markets 100 `
  --out-dir data/processed/exit_aware_walk_forward
```

## Limitations

- No order signing, order placement, wallet/account integration, live fills, queue position, cancellation logic, executable depth, or production latency evidence is present.
- Raw and generated datasets are intentionally ignored. The committed plot is a static derivative, not the underlying run artifact; reacquire public inputs and compare hashes before claiming reproduction.
- `15` of the `16` selected trades in the cited run had no post-entry price path, materially limiting exit-policy evidence.
- Drawdown is realized event-step drawdown, not mark-to-market drawdown from order-book snapshots during open positions.
- Rule/policy selection is trial-exposed; without the retained policy-by-time matrix, PBO and deflated Sharpe cannot be reconstructed.
