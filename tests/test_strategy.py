from __future__ import annotations

import pandas as pd

from polymarket_crypto_5min.backtest import StrategyThresholds, simulate_strategy, summarize_trades
from polymarket_crypto_5min.clients import ClobClient
from polymarket_crypto_5min.downloader import infer_up_down_assets, is_bitcoin_5min_event
from polymarket_crypto_5min.features import asof_close, build_training_frame, taker_fee_per_share
from polymarket_crypto_5min.exit_backtest import ExitPolicy, _training_score, group_price_history, simulate_exit_policy
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics
from polymarket_crypto_5min.resolution import resolved_direction_from_row
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
