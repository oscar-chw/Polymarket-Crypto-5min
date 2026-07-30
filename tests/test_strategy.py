from __future__ import annotations

import pandas as pd

from polymarket_crypto_5min.backtest import StrategyThresholds, simulate_strategy, summarize_trades
from polymarket_crypto_5min.clients import ClobClient, klines_to_frame
from polymarket_crypto_5min.downloader import infer_up_down_assets, is_bitcoin_5min_event
from polymarket_crypto_5min.features import asof_close, build_training_frame, load_candles, taker_fee_per_share
from polymarket_crypto_5min.exit_backtest import ExitPolicy, _training_score, group_price_history, simulate_exit_policy
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics
from polymarket_crypto_5min.resolution import resolved_direction_from_row
from polymarket_crypto_5min.signal_diagnostics import chronological_fold_ic, json_safe
from polymarket_crypto_5min.walk_forward import WalkForwardConfig, make_side_candidates, walk_forward_backtest


def test_event_detector_and_asset_mapping() -> None:
    event = {
        "title": "Bitcoin Up or Down - July 5, 12:00PM-12:05PM ET",
        "slug": "bitcoin-up-or-down-july-5-1200pm-1205pm-et",
        "tags": [{"slug": "crypto", "label": "Crypto"}],
        "markets": [
            {
                "question": "Bitcoin Up or Down?",
                "outcomes": '["Up", "Down"]',
                "clobTokenIds": '["111", "222"]',
            }
        ],
    }
    assert is_bitcoin_5min_event(event)
    assert infer_up_down_assets(outcomes=["Up", "Down"], token_ids=["111", "222"], question="BTC") == ("111", "222")
    assert infer_up_down_assets(outcomes=["Yes", "No"], token_ids=["aaa", "bbb"], question="Will BTC be higher?") == (
        "aaa",
        "bbb",
    )


def test_resolution_uses_polymarket_outcome_prices_not_external_btc() -> None:
    direction, source = resolved_direction_from_row(
        {
            "question": "Bitcoin Up or Down?",
            "outcomes": ["Up", "Down"],
            "outcome_prices": ["0", "1"],
            "gamma_up_price": 0,
            "gamma_down_price": 1,
        }
    )
    assert direction == "DOWN"
    assert source == "outcome_prices"


def test_fee_formula() -> None:
    assert round(taker_fee_per_share(0.50, fee_rate=0.07), 5) == 0.0175
    assert round(taker_fee_per_share(0.30, fee_rate=0.07), 5) == 0.0147


def test_bounded_clob_price_history_omits_relative_interval_filter() -> None:
    class FakeHttp:
        def __init__(self) -> None:
            self.params = None

        def get_json(self, url: str, params: dict | None = None) -> dict:
            self.params = params
            return {"history": [{"t": 1_700_000_000, "p": 0.5}]}

    http = FakeHttp()
    client = ClobClient(http=http)  # type: ignore[arg-type]
    history = client.prices_history("asset", start_ts=1_700_000_000, end_ts=1_700_000_300)
    assert history
    assert http.params["startTs"] == 1_700_000_000
    assert http.params["endTs"] == 1_700_000_300
    assert "interval" not in http.params


def test_asof_close_normalizes_timestamp_precision() -> None:
    candles = pd.DataFrame(
        {
            "ts": pd.to_datetime([1_700_000_000, 1_700_000_060], unit="s", utc=True).astype("datetime64[us, UTC]"),
            "close": [100.0, 101.0],
        }
    )
    when = pd.Series(pd.to_datetime([1_700_000_030], unit="s", utc=True).astype("datetime64[s, UTC]"))
    result = asof_close(candles, when)
    assert result.iloc[0] == 100.0


def test_binance_final_close_is_timestamped_when_available() -> None:
    frame = klines_to_frame(
        [
            [
                1_700_000_000_000,
                "100",
                "102",
                "99",
                "101",
                "12",
                1_700_000_059_999,
                "1200",
                10,
                "6",
                "600",
                "0",
            ]
        ],
        symbol="BTCUSDT",
        interval="1m",
    )
    assert frame.loc[0, "open_ts"] == pd.Timestamp(1_700_000_000, unit="s", tz="UTC")
    assert frame.loc[0, "ts"] == pd.Timestamp(1_700_000_060, unit="s", tz="UTC")
    assert frame.loc[0, "available_at"] == frame.loc[0, "ts"]
    assert frame.loc[0, "timestamp_semantics"] == "close_available_at"


def test_load_candles_migrates_legacy_binance_open_timestamps(tmp_path) -> None:
    path = tmp_path / "btc.csv"
    pd.DataFrame(
        {
            "ts": ["2026-07-05T12:00:00Z"],
            "symbol": ["BTCUSDT"],
            "interval": ["1m"],
            "open": [100],
            "high": [102],
            "low": [99],
            "close": [101],
            "volume": [12],
        }
    ).to_csv(path, index=False)
    frame = load_candles(path)
    assert frame.loc[0, "open_ts"] == pd.Timestamp("2026-07-05T12:00:00Z")
    assert frame.loc[0, "ts"] == pd.Timestamp("2026-07-05T12:01:00Z")
    assert frame.loc[0, "timestamp_semantics"] == "legacy_binance_open_time_shifted_to_close_available_at"


def test_legacy_and_explicit_close_availability_candles_are_equivalent(tmp_path) -> None:
    legacy_path = tmp_path / "legacy.csv"
    explicit_path = tmp_path / "explicit.csv"
    values = {
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "interval": ["1m", "1m"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [12.0, 14.0],
    }
    open_times = pd.to_datetime(
        ["2026-07-05T12:00:00Z", "2026-07-05T12:01:00Z"], utc=True
    )
    pd.DataFrame({"ts": open_times, **values}).to_csv(legacy_path, index=False)
    pd.DataFrame(
        {
            "ts": open_times + pd.Timedelta(minutes=1),
            "open_ts": open_times,
            "close_ts": open_times + pd.Timedelta(minutes=1) - pd.Timedelta(milliseconds=1),
            "available_at": open_times + pd.Timedelta(minutes=1),
            "timestamp_semantics": ["close_available_at", "close_available_at"],
            **values,
        }
    ).to_csv(explicit_path, index=False)

    legacy = load_candles(legacy_path)
    explicit = load_candles(explicit_path)
    columns = ["ts", "open_ts", "close_ts", "available_at", "open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(
        legacy[columns].reset_index(drop=True),
        explicit[columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_training_frame_excludes_unfinished_candle_close() -> None:
    markets = pd.DataFrame(
        [
            {
                "condition_id": "0x" + "4" * 64,
                "market_id": "4",
                "question": "Bitcoin Up or Down - availability test",
                "start_ts": 1_700_000_000,
                "end_ts": 1_700_000_300,
                "up_asset_id": "up4",
                "down_asset_id": "down4",
                "outcomes": ["Up", "Down"],
                "outcome_prices": [1, 0],
            }
        ]
    )
    # Values are indexed by their availability timestamps.  At snapshot
    # 00:04:15 the 00:04-00:05 candle is still unfinished, so 104 is the
    # newest legal close and 999 must not enter any feature.
    candles = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    1_700_000_000,
                    1_700_000_060,
                    1_700_000_120,
                    1_700_000_180,
                    1_700_000_240,
                    1_700_000_300,
                ],
                unit="s",
                utc=True,
            ),
            "close": [100, 101, 102, 103, 104, 999],
        }
    )
    poly_prices = pd.DataFrame(
        {
            "condition_id": ["0x" + "4" * 64],
            "asset_id": ["up4"],
            "ts": pd.to_datetime([1_700_000_240], unit="s", utc=True),
            "p": [0.65],
        }
    )
    frame = build_training_frame(markets, candles, poly_prices, snapshot_seconds_before_close=45)
    assert frame.loc[0, "btc_snapshot"] == 104
    assert frame.loc[0, "btc_snapshot_available_at"] == pd.Timestamp(1_700_000_240, unit="s", tz="UTC")
    assert frame.loc[0, "feature_availability_passed"]


def test_signal_ic_uses_only_complete_chronological_folds() -> None:
    detail = pd.DataFrame(
        {
            "condition_id": [f"m{i}" for i in range(7)],
            "end_dt": pd.date_range("2026-07-01", periods=7, freq="5min", tz="UTC"),
            "horizon_seconds": [45] * 7,
            "model_prob_up": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.99],
            "direction_up": [0, 1, 0, 1, 0, 1, 0],
        }
    )
    folds = chronological_fold_ic(detail, fold_market_count=3)
    assert len(folds) == 2
    assert folds["n_markets"].eq(3).all()
    assert folds["pearson_ic_prob_vs_direction"].gt(0).all()


def test_signal_rank_ic_handles_ties_without_scipy() -> None:
    detail = pd.DataFrame(
        {
            "condition_id": [f"m{i}" for i in range(6)],
            "end_dt": pd.date_range("2026-07-01", periods=6, freq="5min", tz="UTC"),
            "horizon_seconds": [45] * 6,
            "model_prob_up": [0.2, 0.2, 0.4, 0.6, 0.8, 0.8],
            "direction_up": [0, 0, 0, 1, 1, 1],
        }
    )
    folds = chronological_fold_ic(detail, fold_market_count=6)
    expected = detail["model_prob_up"].rank(method="average").corr(
        detail["direction_up"].rank(method="average")
    )
    assert folds.loc[0, "rank_ic_prob_vs_direction"] == expected


def test_signal_diagnostic_json_is_strict() -> None:
    assert json_safe({"missing": float("nan"), "finite": 1.5}) == {"missing": None, "finite": 1.5}


def test_equity_curve_counts_first_trade_loss_as_drawdown() -> None:
    trades = pd.DataFrame(
        {
            "end_dt": pd.to_datetime([1_700_000_000, 1_700_000_300], unit="s", utc=True),
            "pnl_usdc": [-10.0, 5.0],
            "stake_usdc": [10.0, 10.0],
            "return_on_stake": [-1.0, 0.5],
            "won": [False, True],
        }
    )
    curve = equity_curve(trades, initial_capital=2000)
    assert curve.iloc[0]["is_initial_row"] is True or bool(curve.iloc[0]["is_initial_row"])
    assert curve["drawdown_usdc"].min() == -10.0
    metrics = performance_metrics(trades, initial_capital=2000)
    score = _training_score(trades.assign(exit_dt=trades["end_dt"]), initial_capital=2000)
    assert metrics.loc[0, "max_drawdown_usdc"] == -10.0
    assert round(metrics.loc[0, "max_drawdown_pct"], 4) == -0.005
    assert round(score["train_mdd_pct"], 4) == -0.005
    assert score["train_roi"] == metrics.loc[0, "roi_on_stake"]
    assert metrics.loc[0, "realized_equity_curve_only"] is True or bool(metrics.loc[0, "realized_equity_curve_only"])


def test_build_training_frame_and_simulate() -> None:
    markets = pd.DataFrame(
        [
            {
                "condition_id": "0x" + "1" * 64,
                "market_id": "1",
                "question": "Bitcoin Up or Down - test",
                "start_ts": 1_700_000_000,
                "end_ts": 1_700_000_300,
                "up_asset_id": "up1",
                "down_asset_id": "down1",
                "outcomes": ["Up", "Down"],
                "outcome_prices": [1, 0],
            },
            {
                "condition_id": "0x" + "2" * 64,
                "market_id": "2",
                "question": "Bitcoin Up or Down - test 2",
                "start_ts": 1_700_000_300,
                "end_ts": 1_700_000_600,
                "up_asset_id": "up2",
                "down_asset_id": "down2",
                "outcomes": ["Up", "Down"],
                "outcome_prices": [0, 1],
            },
        ]
    )
    candles = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    1_700_000_000,
                    1_700_000_060,
                    1_700_000_120,
                    1_700_000_180,
                    1_700_000_240,
                    1_700_000_300,
                    1_700_000_360,
                    1_700_000_420,
                    1_700_000_480,
                    1_700_000_540,
                    1_700_000_600,
                ],
                unit="s",
                utc=True,
            ),
            "close": [100, 101, 102, 103, 104, 105, 104, 103, 102, 101, 100],
        }
    )
    poly_prices = pd.DataFrame(
        {
            "condition_id": ["0x" + "1" * 64, "0x" + "2" * 64],
            "asset_id": ["up1", "down2"],
            "ts": pd.to_datetime([1_700_000_240, 1_700_000_540], unit="s", utc=True),
            "p": [0.65, 0.55],
        }
    )
    frame = build_training_frame(markets, candles, poly_prices, snapshot_seconds_before_close=45)
    assert len(frame) == 2
    assert frame.loc[0, "realized_direction"] == "UP"
    assert frame.loc[0, "realized_direction_source"] == "outcome_prices"
    assert frame.loc[1, "realized_direction"] == "DOWN"
    trades = simulate_strategy(
        frame,
        StrategyThresholds(
            confirm_min_abs_bps=1,
            confirm_min_model_prob=0.50,
            value_max_market_price=1,
            value_min_model_prob=0.50,
            min_ev_per_share=-1,
        ),
    )
    assert len(trades) >= 1
    summary = summarize_trades(trades)
    assert not summary.empty

    curve = equity_curve(trades, initial_capital=1000)
    assert {"equity", "drawdown_usdc", "drawdown_pct"}.issubset(curve.columns)
    metrics = performance_metrics(trades, initial_capital=1000)
    assert {"trade_sharpe", "daily_sharpe", "max_drawdown_pct", "max_drawdown_usdc"}.issubset(metrics.columns)
    assert metrics.loc[0, "bucket"] == "ALL"


def test_external_btc_is_not_used_as_label_when_polymarket_resolved_differs() -> None:
    markets = pd.DataFrame(
        [
            {
                "condition_id": "0x" + "3" * 64,
                "market_id": "3",
                "question": "Bitcoin Up or Down - conflict test",
                "start_ts": 1_700_001_000,
                "end_ts": 1_700_001_300,
                "up_asset_id": "up3",
                "down_asset_id": "down3",
                "outcomes": ["Up", "Down"],
                "outcome_prices": [0, 1],
            }
        ]
    )
    candles = pd.DataFrame(
        {
            "ts": pd.to_datetime([1_700_001_000, 1_700_001_240, 1_700_001_300], unit="s", utc=True),
            "close": [100, 110, 120],
        }
    )
    poly_prices = pd.DataFrame(
        {
            "condition_id": ["0x" + "3" * 64],
            "asset_id": ["up3"],
            "ts": pd.to_datetime([1_700_001_240], unit="s", utc=True),
            "p": [0.75],
        }
    )
    frame = build_training_frame(markets, candles, poly_prices, snapshot_seconds_before_close=45)
    assert frame.loc[0, "btc_external_direction"] == "UP"
    assert frame.loc[0, "realized_direction"] == "DOWN"


def test_walk_forward_has_no_train_test_leakage() -> None:
    feature_rows = []
    base = 1_700_010_000
    for i in range(12):
        feature_rows.append(
            {
                "condition_id": f"m{i}",
                "end_dt": pd.Timestamp(base + i * 300, unit="s", tz="UTC"),
                "snapshot_dt": pd.Timestamp(base + i * 300 - 45, unit="s", tz="UTC"),
                "realized_direction": "UP" if i % 2 == 0 else "DOWN",
                "market_up_price": 0.80 if i % 2 == 0 else 0.20,
                "market_down_price": 0.20 if i % 2 == 0 else 0.80,
                "model_prob_up": 0.75 if i % 2 == 0 else 0.25,
                "score_bps": 10 if i % 2 == 0 else -10,
                "momentum_1m_bps": 1,
                "momentum_3m_bps": 1,
            }
        )
    candidates = make_side_candidates(pd.DataFrame(feature_rows))
    trades, fold_report, rules = walk_forward_backtest(
        candidates,
        config=WalkForwardConfig(
            initial_capital=2000,
            stake_usdc=10,
            train_markets=6,
            test_markets=3,
            min_train_trades=1,
            min_bin_observations=1,
        ),
    )
    assert not fold_report.empty
    assert fold_report["leakage_check_passed"].all()
    assert (pd.to_datetime(fold_report["train_end_dt"]) < pd.to_datetime(fold_report["test_start_dt"])).all()
    assert isinstance(trades, pd.DataFrame)
    assert isinstance(rules, pd.DataFrame)


def test_exit_policy_uses_grouped_price_history_path() -> None:
    entry = pd.DataFrame(
        [
            {
                "condition_id": "m1",
                "market_id": "1",
                "question": "Bitcoin Up or Down",
                "side": "UP",
                "snapshot_dt": pd.Timestamp("2026-01-01T00:00:00Z"),
                "end_dt": pd.Timestamp("2026-01-01T00:05:00Z"),
                "market_price": 0.50,
                "won": True,
                "up_asset_id": "up",
                "down_asset_id": "down",
            }
        ]
    )
    history = pd.DataFrame(
        [
            {"condition_id": "m1", "asset_id": "up", "ts": pd.Timestamp("2026-01-01T00:00:10Z"), "p": 0.54},
            {"condition_id": "m1", "asset_id": "up", "ts": pd.Timestamp("2026-01-01T00:00:20Z"), "p": 0.57},
            {"condition_id": "m1", "asset_id": "down", "ts": pd.Timestamp("2026-01-01T00:00:20Z"), "p": 0.43},
        ]
    )
    grouped = group_price_history(history)
    assert ("m1", "up") in grouped
    trades = simulate_exit_policy(entry, history, ExitPolicy(take_profit=0.05, target_price=None, stop_loss=None), stake_usdc=10)
    assert trades.loc[0, "exit_reason"] == "TAKE_PROFIT"
    assert trades.loc[0, "path_points_seen"] == 2
    assert trades.loc[0, "exit_price"] == 0.57
