from __future__ import annotations

import pandas as pd

from polymarket_crypto_5min.backtest import StrategyThresholds, simulate_strategy, summarize_trades
from polymarket_crypto_5min.downloader import infer_up_down_assets, is_bitcoin_5min_event
from polymarket_crypto_5min.features import build_training_frame, taker_fee_per_share
from polymarket_crypto_5min.metrics import equity_curve, performance_metrics


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


def test_fee_formula() -> None:
    assert round(taker_fee_per_share(0.50, fee_rate=0.07), 5) == 0.0175
    assert round(taker_fee_per_share(0.30, fee_rate=0.07), 5) == 0.0147


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
            },
            {
                "condition_id": "0x" + "2" * 64,
                "market_id": "2",
                "question": "Bitcoin Up or Down - test 2",
                "start_ts": 1_700_000_300,
                "end_ts": 1_700_000_600,
                "up_asset_id": "up2",
                "down_asset_id": "down2",
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
