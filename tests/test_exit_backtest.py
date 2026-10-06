from __future__ import annotations

import pandas as pd

from polymarket_crypto_5min.exit_backtest import ExitPolicy, losing_trades, simulate_exit_policy


def test_exit_policy_takes_profit_and_logs_trade_details() -> None:
    entries = pd.DataFrame(
        [
            {
                "condition_id": "m1",
                "market_id": "1",
                "question": "Bitcoin Up or Down?",
                "side": "UP",
                "snapshot_dt": pd.Timestamp("2026-07-05T12:00:00Z"),
                "end_dt": pd.Timestamp("2026-07-05T12:00:45Z"),
                "market_price": 0.65,
                "up_asset_id": "up1",
                "down_asset_id": "down1",
                "won": True,
                "p_lower": 0.80,
                "edge_lower": 0.12,
                "score_bps": 10.0,
                "abs_score_bps": 10.0,
            }
        ]
    )
    prices = pd.DataFrame(
        {
            "condition_id": ["m1", "m1"],
            "asset_id": ["up1", "up1"],
            "ts": pd.to_datetime(["2026-07-05T12:00:05Z", "2026-07-05T12:00:10Z"]),
            "p": [0.68, 0.76],
        }
    )
    trades = simulate_exit_policy(
        entries, prices, ExitPolicy(take_profit=0.08, target_price=None, stop_loss=None), stake_usdc=10
    )
    assert len(trades) == 1
    assert trades.loc[0, "exit_reason"] == "TAKE_PROFIT"
    assert trades.loc[0, "exit_price"] == 0.76
    assert trades.loc[0, "realized_settlement"] is False or not bool(trades.loc[0, "realized_settlement"])
    assert trades.loc[0, "pnl_usdc"] > 0
    assert trades.loc[0, "path_points_seen"] == 2


def test_exit_policy_stop_loss_and_losing_trade_audit() -> None:
    entries = pd.DataFrame(
        [
            {
                "condition_id": "m2",
                "market_id": "2",
                "question": "Bitcoin Up or Down?",
                "side": "DOWN",
                "snapshot_dt": pd.Timestamp("2026-07-05T12:05:00Z"),
                "end_dt": pd.Timestamp("2026-07-05T12:05:45Z"),
                "market_price": 0.70,
                "up_asset_id": "up2",
                "down_asset_id": "down2",
                "won": False,
                "p_lower": 0.82,
                "edge_lower": 0.09,
                "score_bps": -11.0,
                "abs_score_bps": 11.0,
            }
        ]
    )
    prices = pd.DataFrame(
        {
            "condition_id": ["m2"],
            "asset_id": ["down2"],
            "ts": pd.to_datetime(["2026-07-05T12:05:07Z"]),
            "p": [0.63],
        }
    )
    trades = simulate_exit_policy(
        entries, prices, ExitPolicy(take_profit=None, target_price=None, stop_loss=0.05), stake_usdc=10
    )
    assert trades.loc[0, "exit_reason"] == "STOP_LOSS"
    assert trades.loc[0, "loss_trade"] is True or bool(trades.loc[0, "loss_trade"])
    losses = losing_trades(trades)
    assert len(losses) == 1
    assert losses.loc[0, "condition_id"] == "m2"


def test_exit_never_fills_at_the_settlement_instant() -> None:
    entries = pd.DataFrame(
        [
            {
                "condition_id": "m3",
                "side": "UP",
                "snapshot_dt": pd.Timestamp("2026-07-05T12:04:15Z"),
                "end_dt": pd.Timestamp("2026-07-05T12:05:00Z"),
                "market_price": 0.50,
                "up_asset_id": "up3",
                "down_asset_id": "down3",
                "won": False,
            }
        ]
    )
    prices = pd.DataFrame(
        {
            "condition_id": ["m3"],
            "asset_id": ["up3"],
            "ts": pd.to_datetime(["2026-07-05T12:05:00Z"]),
            "p": [0.01],
        }
    )
    trades = simulate_exit_policy(
        entries, prices, ExitPolicy(take_profit=None, target_price=None, stop_loss=0.05), stake_usdc=10
    )
    assert trades.loc[0, "exit_reason"] == "HOLD_TO_SETTLEMENT"
    assert trades.loc[0, "path_points_seen"] == 0
    assert trades.loc[0, "exit_price"] == 0.0
