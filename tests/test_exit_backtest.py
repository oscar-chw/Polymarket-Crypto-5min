from __future__ import annotations

import pandas as pd

from polymarket_crypto_5min.exit_backtest import (
    MAX_HOLD_NO_PRICE,
    ExitPolicy,
    losing_trades,
    simulate_exit_policy,
    walk_forward_exit_backtest,
)
from polymarket_crypto_5min.walk_forward import WalkForwardConfig, WalkForwardRule, make_side_candidates


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


def _max_hold_entry(condition_id: str, snapshot: str, end: str) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "side": "UP",
        "snapshot_dt": pd.Timestamp(snapshot),
        "end_dt": pd.Timestamp(end),
        "market_price": 0.60,
        "up_asset_id": f"up-{condition_id}",
        "down_asset_id": f"down-{condition_id}",
        "won": True,
    }


def test_max_hold_without_price_points_is_marked_not_silently_settled() -> None:
    entries = pd.DataFrame([_max_hold_entry("m4", "2026-07-05T12:04:15Z", "2026-07-05T12:05:00Z")])
    # 1-minute points: nothing lands in (12:04:15, 12:04:45].
    prices = pd.DataFrame(
        {
            "condition_id": ["m4", "m4"],
            "asset_id": ["up-m4", "up-m4"],
            "ts": pd.to_datetime(["2026-07-05T12:04:00Z", "2026-07-05T12:05:00Z"]),
            "p": [0.60, 0.99],
        }
    )
    policy = ExitPolicy(take_profit=None, target_price=None, stop_loss=None, max_hold_seconds=30)
    trades = simulate_exit_policy(entries, prices, policy, stake_usdc=10)
    settled = simulate_exit_policy(
        entries, prices, ExitPolicy(take_profit=None, target_price=None, stop_loss=None), stake_usdc=10
    )

    assert trades.loc[0, "exit_reason"] == MAX_HOLD_NO_PRICE
    assert bool(trades.loc[0, "realized_settlement"])
    assert trades.loc[0, "path_points_seen"] == 0
    assert settled.loc[0, "exit_reason"] == "HOLD_TO_SETTLEMENT"
    assert trades.loc[0, "pnl_usdc"] == settled.loc[0, "pnl_usdc"]


def test_walk_forward_exit_folds_count_max_hold_no_price_trades() -> None:
    rows = []
    for index in range(6):
        end = pd.Timestamp("2026-07-05T12:05:00Z") + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "condition_id": f"m{index}",
                "end_dt": end,
                "snapshot_dt": end - pd.Timedelta(seconds=45),
                "realized_direction": "UP",
                "market_up_price": 0.60,
                "market_down_price": 0.40,
                "model_prob_up": 0.80,
                "score_bps": 10.0,
                "momentum_1m_bps": 1.0,
                "momentum_3m_bps": 1.0,
                "up_asset_id": f"up-m{index}",
                "down_asset_id": f"down-m{index}",
            }
        )
    candidates = make_side_candidates(pd.DataFrame(rows))
    empty_prices = pd.DataFrame(columns=["condition_id", "asset_id", "ts", "p"])
    trades, folds, _ = walk_forward_exit_backtest(
        candidates,
        empty_prices,
        config=WalkForwardConfig(train_markets=4, test_markets=2, min_train_trades=1, min_bin_observations=1),
        entry_rules=[WalkForwardRule(-1.0, 0.0, 0.0, 1.0, 0.0)],
        exit_policies=[ExitPolicy(take_profit=None, target_price=None, stop_loss=None, max_hold_seconds=10)],
    )

    assert len(trades) > 0
    assert trades["exit_reason"].eq(MAX_HOLD_NO_PRICE).all()
    assert int(folds["test_max_hold_no_price_trades"].sum()) == len(trades)
