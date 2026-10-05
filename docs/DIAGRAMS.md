# Diagrams

1. [System overview](#1-system-overview)
2. [The candle-timing leak, before and after the fix](#2-the-candle-timing-leak-before-and-after-the-fix)
3. [Walk-forward fold timeline](#3-walk-forward-fold-timeline)
4. [Exit-aware backtest of one trade](#4-exit-aware-backtest-of-one-trade)

If a diagram and the code disagree, the code wins.

## 1. System overview

Public data flows through close-indexed candles and point-in-time features into a walk-forward run whose every test fold is calibrated and selected on earlier markets only; nothing leaves the machine as an order.

```mermaid
flowchart TB
  subgraph SRC["Public APIs, read-only"]
    G["Gamma API<br/>market metadata"]:::ext
    CL["CLOB price history<br/>and Data API trades"]:::ext
    BN["Binance BTCUSDT<br/>1-minute klines"]:::ext
  end
  subgraph ACQ["clients.py, downloader.py"]
    DL["bounded-retry<br/>downloads"]:::step
    KF["klines_to_frame:<br/>ts = close + 1 ms"]:::key
  end
  RAW[("data/raw CSVs<br/>git-ignored")]:::data
  subgraph FEAT["features.py"]
    BF["build_training_frame:<br/>as-of joins at snapshot"]:::key
    AV{"feature_available_at<br/>≤ snapshot?"}:::gate
    STOP["run stops"]:::gate
  end
  subgraph WF["walk_forward.py, exit_backtest.py"]
    SC["make_side_candidates:<br/>UP and DOWN rows"]:::step
    CAL["calibrate_candidates:<br/>earlier markets only"]:::key
    SEL["select on train:<br/>512 entry × 192 exit"]:::step
    TST["evaluate next<br/>test fold"]:::key
  end
  OUT[("trades, folds, metrics,<br/>equity, run manifest")]:::out
  ORD["order placement<br/>not implemented"]:::ext

  G -->|"BTC 5-min markets"| DL
  CL -->|"token price paths"| DL
  BN -->|"OHLCV rows"| KF
  DL -->|"writes CSV"| RAW
  KF ==>|"close-indexed candles"| RAW
  RAW ==>|"markets, candles, prices"| BF
  BF ==>|"feature row"| AV
  AV -->|"no: raises ValueError"| STOP
  AV ==>|"yes, resolved markets only"| SC
  SC ==>|"5,965 candidate rows"| CAL
  CAL -->|"p_lower, edge_lower"| SEL
  SEL ==>|"one rule + policy per fold"| TST
  TST ==>|"26 folds, metrics.py"| OUT
  OUT -.->|"never sent: research-only"| ORD

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `polymarket_crypto_5min/clients.py` (`klines_to_frame`), `polymarket_crypto_5min/downloader.py`, `polymarket_crypto_5min/features.py` (`build_training_frame`, `asof_candle`), `polymarket_crypto_5min/walk_forward.py`, `polymarket_crypto_5min/exit_backtest.py`, `polymarket_crypto_5min/metrics.py`, `scripts/exit_aware_walk_forward.py`.

## 2. The candle-timing leak, before and after the fix

The same one-minute candle, indexed two ways: at its open time an as-of join inside that minute reads a close that does not exist yet; at its close-availability time the join falls back to the previous candle.

```mermaid
flowchart LR
  subgraph WRONG["Before the fix: indexed at open, leaks"]
    direction TB
    W1[("Binance kline<br/>opens at T, final close C")]:::data
    W2["row ts = T<br/>open time"]:::step
    W3["decision at t<br/>inside that minute"]:::step
    W4["feature reads C<br/>known only at T + 1m"]:::gate
    W1 -->|"indexed at open"| W2
    W2 -->|"as-of join: ts ≤ t matches"| W3
    W3 -->|"uses the future close"| W4
  end
  W1 -.->|"same kline, fixed indexing"| R1
  subgraph RIGHT["After the fix: indexed at close"]
    direction TB
    R1[("same kline<br/>opens at T, final close C")]:::data
    R2["row ts = available_at<br/>= close_ts + 1 ms = T + 1m"]:::key
    R3["decision at t<br/>inside that minute"]:::step
    R4["feature reads previous<br/>candle's close"]:::key
    R5{"feature_available_at<br/>≤ snapshot?"}:::gate
    R6["feature row kept"]:::out
    R7["ValueError, run stops"]:::gate
    R1 -->|"klines_to_frame"| R2
    R2 -->|"as-of join: ts ≤ t skips it"| R3
    R3 ==>|"only completed candles"| R4
    R4 ==>|"checked per row"| R5
    R5 ==>|"yes"| R6
    R5 -->|"no"| R7
    LEG[("legacy open-indexed CSV<br/>no timestamp_semantics")]:::ext
    LEG -.->|"load_candles shifts<br/>ts to open + interval"| R2
  end

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `polymarket_crypto_5min/clients.py` (`klines_to_frame`), `polymarket_crypto_5min/features.py` (`load_candles`, `asof_candle`, the availability check in `build_training_frame`); test: `tests/test_research_invariants.py::test_ohlcv_close_is_unavailable_until_after_exchange_close_time`.

## 3. Walk-forward fold timeline

Markets are ordered by end time; each fold trains on every earlier market and tests on the next block, so the test block becomes training data only for the folds after it.

```mermaid
flowchart TB
  MK[("candidates ordered<br/>by market end_dt")]:::data
  subgraph EXP["Expanding window: --train-markets 400, --test-markets 100"]
    direction LR
    F1["Fold 1<br/>train: markets 1–400<br/>test: 401–500"]:::step
    F2["Fold 2<br/>train: 1–500<br/>test: 501–600"]:::step
    FD["Folds 3–25<br/>train grows by 100"]:::step
    F26["Fold 26<br/>train: all earlier markets<br/>test: final block"]:::step
    F1 -->|"test block joins train"| F2
    F2 -->|"+100 markets"| FD
    FD -->|"+100 markets"| F26
  end
  MK -->|"first 400 markets"| F1
  subgraph ONE["Inside every fold"]
    direction LR
    TR[("train candidates")]:::data
    TE[("test candidates")]:::data
    C1["calibrate on train<br/>reference = train"]:::step
    S1["select entry rule and<br/>exit policy on train"]:::step
    C2["calibrate test<br/>reference = train only"]:::key
    EV["simulate test trades"]:::key
    LK{"train ends before<br/>test starts?"}:::gate
    REP[("fold report row")]:::out
    BAD["SystemExit:<br/>Leakage check failed"]:::gate
    TR -->|"wins per price/score bin"| C1
    C1 -->|"p_lower, edge_lower"| S1
    TR -->|"bin win rates"| C2
    TE -->|"rows to score"| C2
    S1 ==>|"frozen rule + policy"| EV
    C2 ==>|"calibrated test rows"| EV
    EV ==>|"test PnL"| REP
    TR -->|"train_end_dt"| LK
    TE -->|"test_start_dt"| LK
    LK -->|"recorded as<br/>leakage_check_passed"| REP
    REP -->|"false in any fold"| BAD
  end
  F1 -.->|"same steps each fold"| ONE

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

Where in the code: `polymarket_crypto_5min/exit_backtest.py` (`walk_forward_exit_backtest`, `select_entry_exit_rules_from_training`), `polymarket_crypto_5min/walk_forward.py` (`calibrate_candidates`, `walk_forward_backtest`), `scripts/exit_aware_walk_forward.py` (the leakage exit); tests: `tests/test_research_invariants.py::test_walk_forward_test_calibration_uses_preceding_markets_only`, `::test_walk_forward_uses_expanding_nonoverlapping_test_boundaries`.

## 4. Exit-aware backtest of one trade

One selected test entry, followed along its own token's price path until an exit rule fires, the hold limit runs out, or the market settles.

```mermaid
flowchart TB
  E0[("test candidate passes<br/>the fold's entry rule")]:::data
  E1["enter at snapshot_dt:<br/>market price + taker fee"]:::step
  PATH[("token price path after entry,<br/>up to the deadline")]:::data
  Q{"next price point p<br/>in time order"}:::gate
  SL["STOP_LOSS<br/>p ≤ entry − stop_loss"]:::step
  TG["TARGET_PRICE<br/>p ≥ target_price"]:::step
  TP["TAKE_PROFIT<br/>p ≥ entry + take_profit"]:::step
  MH["MAX_HOLD_EXIT<br/>at last path price"]:::step
  HS["HOLD_TO_SETTLEMENT<br/>pays 1 or 0"]:::key
  X1[("early-exit PnL:<br/>exit − entry − both fees")]:::out
  X2[("settlement PnL:<br/>payout − entry − entry fee")]:::out

  E0 -->|"rule_mask true"| E1
  E1 -->|"deadline = end_dt or<br/>entry + max_hold"| PATH
  PATH -->|"prices after end_dt<br/>never read"| Q
  Q -->|"checked 1st"| SL
  Q -->|"checked 2nd"| TG
  Q -->|"checked 3rd"| TP
  Q -->|"path ends, no trigger,<br/>max_hold set"| MH
  Q ==>|"no trigger and no max_hold,<br/>or empty path"| HS
  SL -->|"sell at p, exit fee"| X1
  TG -->|"sell at p, exit fee"| X1
  TP -->|"sell at p, exit fee"| X1
  MH -->|"sell at p, exit fee"| X1
  HS ==>|"settled Polymarket outcome"| X2

  classDef data fill:#dbeafe,stroke:#1d4ed8,color:#0b1220
  classDef step fill:#f1f5f9,stroke:#475569,color:#0b1220
  classDef gate fill:#fef3c7,stroke:#b45309,color:#0b1220
  classDef out  fill:#dcfce7,stroke:#15803d,color:#0b1220
  classDef ext  fill:#f8fafc,stroke:#94a3b8,color:#0b1220,stroke-dasharray:4 3
  classDef key  fill:#ede9fe,stroke:#6d28d9,color:#0b1220,stroke-width:2px
```

In the cited availability-safe run, 15 of the 16 selected trades had no post-entry price path, so they took the empty-path branch to settlement (see the [README limitations](../README.md#limitations)).

Where in the code: `polymarket_crypto_5min/exit_backtest.py` (`simulate_exit_policy`, `_simulate_one_exit`, `_slice_exit_path`, `_exit_reason`), `polymarket_crypto_5min/features.py` (`taker_fee_per_share`); test: `tests/test_research_invariants.py::test_exit_pnl_deducts_entry_and_exit_fees_and_ignores_post_resolution_prices`.
