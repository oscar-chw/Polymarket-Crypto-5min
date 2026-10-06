# Results: the availability-safe walk-forward and what backs it

**Status label:** selected chronological walk-forward OOS; trial-exposed, modeled historical; not live, cash, or executable proof.

The availability-safe selected result is negative. The `ALL` row of the run's metrics file (`data/processed/exit_aware_walk_forward_availability_safe_grid/exit_aware_metrics.csv:2`, a local generated artifact excluded from Git; hash under [Provenance](#provenance)) records, rounded:

| Metric | Value |
|---|---|
| Selected trades / profitable | 16 / 12 (75% win rate) |
| Net PnL | -$12.31 on $160 staked (-7.7% on stake) |
| Return on $2,000 initial capital | -0.62% |
| Max realized drawdown | -$34.03 (-1.69%) |
| Profit factor | 0.69 |
| Sharpe (per trade / daily) | -0.14 / -7.57 |

**2026-10-06: not re-run after the timing fixes.** These figures come from a July 2026 run, before six timing fixes:
entry prices bounded to fresh prints inside the market window, no exit at the settlement instant, max-hold trades
with no price point marked `MAX_HOLD_NO_PRICE`, fold leakage checked against test decision time (the full-history
runner now fails on it), a staleness limit on BTC candles, and correct ET parsing in the DST fall-back hour and on
Dec 31. The run's inputs are git-ignored local files known only by the hashes under [Provenance](#provenance). They
are no longer available, and the market set cannot be rebuilt exactly from the public APIs, so the figures are left as
published, not regenerated. Known biases that still apply to them:

- Entry at the last print instead of the ask makes them optimistic in expectation, by an unknown amount (a print can be a trade at the ask). Prints could
  also be stale or predate the market window; the direction of that error is not known.
- 15 of the 16 trades were held to settlement because no price point followed entry, so the result is close to a
  hold-to-settlement result and is not evidence that any exit policy works. Only the remaining trade could have exited
  at the settlement instant; the direction of that error is not known.
- The recorded "chronological leakage passed" used the old check (training settlement against test settlement). Whether
  any fold put a training label after a test decision is not known. Any leak would make the figures optimistic.
- Whether stale BTC candles entered any feature is not known, and so the direction is not known.
- Rule selection still uses in-sample calibration of training rows (not fixed). This biases which rule is chosen, not
  the out-of-sample scoring of the test folds.
- The ET parsing fix is not expected to matter: the run used 3,000 markets (about 10 days of 5-minute markets) ending in July 2026,
  a span with no DST change and no Dec 31.

The corresponding manifest records final Binance OHLCV at candle-close availability, calibration of each test fold from preceding training markets only, exclusion of unresolved markets and post-resolution Gamma prices from features by default, and selection exposed across `512` entry rules and `192` exit policies. PBO and deflated Sharpe are unavailable because the full policy-by-time return matrix was not retained.

The earlier positive `$52.74` OOS figure (58 trades) in [EXIT_AWARE_RESEARCH_PLAN.md](EXIT_AWARE_RESEARCH_PLAN.md) predates the candle-availability correction and is withdrawn: it joined final candle values at the candle-open timestamp, before they were available.

![Selected availability-safe modeled historical OOS equity; trial-exposed and not live evidence](availability_safe_oos_equity_curve.png)

The chart is derived from the ignored local artifact `data/processed/exit_aware_walk_forward_availability_safe_grid/exit_aware_equity_curve.csv` (SHA-256 `88a89e7575bc8a65ead09167d7eb428c46daa7d91b585af2b9c42e2a9cf82d1b`) and is labeled modeled historical OOS, trial-exposed, and non-executable.

## Provenance

- Repository history starts at commit `c8e090d9e573eb5df41f9b254d9e30ff50eca0ad` on `2026-07-05T22:02:23+08:00`.
- The negative metrics source is a local generated artifact intentionally excluded from Git; SHA-256: `d224a8c990fb62572a90c6ed3eef30d831dd00a29dfbdf291f768edbf31903cb`.
- Its run manifest SHA-256 is `2f215ac951821d34a61bdab0a8c1a61f6e938c306a5e6019067e63153d3b9edd`; it records `3,000` feature rows, `5,965` candidate rows, `26` folds, `16` selected trades, and passing feature-availability and chronological-leakage gates.
- Manifest input hashes: markets `d14b8aeda6b457372616d6395b0accb2d5da259b099f3d1ef587d19f338e1f2b`, Binance candles `a740cb1dfaba38bb39f5d42d0862d43ab8ff7c0a2f3595d3ad0d9399eca25778`, and combined Polymarket prices `fcc590c19fb8b0e094b4e33e34b4c8dfeaa96a7fbfca3dc9d019ab1c23a3d525`.
