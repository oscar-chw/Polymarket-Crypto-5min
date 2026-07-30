## One sentence
Research-only tooling for point-in-time, walk-forward analysis of Bitcoin five-minute Polymarket Up/Down markets; it neither places orders nor promises returns.

## Result
No headline result. No finalized fold report is published in this committed tree. IS: **unknown**; OOS: **unknown**; live: **unknown**. The metric fields are defined verbatim in `polymarket_crypto_5min/metrics.py:139-164` and emitted as:

`"bucket"`, `"trades"`, `"wins"`, `"win_rate"`, `"total_pnl_usdc"`, `"total_stake_usdc"`, `"roi_on_stake"`, `"return_on_initial_capital"`, `"avg_trade_return"`, `"trade_return_std"`, `"trade_sharpe"`, `"daily_sharpe"`, `"daily_sortino"`, `"max_drawdown_usdc"`, `"max_drawdown_pct"`, `"profit_factor"`, `"gross_profit_usdc"`, `"gross_loss_usdc"`, `"avg_market_price"`, `"avg_ev_per_share"`, `"realized_equity_curve_only"`, `"mark_to_market_drawdown_available"`, `"min_realized_equity"`.

OOS fold separation is implemented at `polymarket_crypto_5min/walk_forward.py:286-287` and checked at `polymarket_crypto_5min/walk_forward.py:327-370`; no committed fold report or metrics file supplies values in this tree.

Prior OOS history is documented in repo notes (for example `docs/EXIT_AWARE_RESEARCH_PLAN.md` includes `OOS result: 58 trades, $52.74 net PnL on a $2,000`) and the last full run status is captured in `docs/RESUME.md`, but these are not finalized outputs from this commit. Do not describe synthetic, trial-exposed, or dry-run outputs as live.

## How it works
- The downloader retrieves public market metadata, price history, trades, and BTC one-minute candles.
- Feature construction uses values available at each decision time.
- Each walk-forward fold calibrates and selects rules on earlier rows, then evaluates the next chronological fold.
- Fees, fixed-stake PnL, realized equity, and walk-forward diagnostics are calculated locally from supplied historical inputs.

## The interesting decision
The central safeguard is temporal isolation: the test fold does not update its own calibration or selected rule. `polymarket_crypto_5min/walk_forward.py:347-354` calibrates the test set from `train`, while `polymarket_crypto_5min/walk_forward.py:327-329` establishes the training cutoff. This is more important than selecting a favorable threshold because it makes leakage reviewable.

## Run it
```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python scripts/download_history.py --max-pages 2
python scripts/walk_forward_backtest.py
pytest -q
```

All downloaded and generated data remain local and ignored. `scripts/live_signal.py` produces dry-run labels only; `polymarket_crypto_5min/live_signal.py:82-152` creates `PAPER_BUY_*` or `WATCH`, not orders.

## Status
Active research scaffold. No credentials, wallet/account/order/position records, raw dumps, or large binary datasets are tracked. Reproduce results with independently obtained historical inputs, inspect leakage checks, and validate liquidity, fees, latency, and market rules before any manual trading decision.
