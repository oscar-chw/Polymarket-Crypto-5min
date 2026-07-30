from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

import polymarket_crypto_5min.walk_forward as walk_forward_module
from polymarket_crypto_5min.clients import BinanceClient
from polymarket_crypto_5min.exit_backtest import ExitPolicy, simulate_exit_policy
from polymarket_crypto_5min.features import shares_for_stake, taker_fee_per_share
from polymarket_crypto_5min.resolution import append_resolved_outcomes
from polymarket_crypto_5min.walk_forward import WalkForwardConfig, WalkForwardRule, make_side_candidates


def _kline(open_ms: int, close_ms: int, close: float) -> list[object]:
    return [open_ms, "100", "110", "90", str(close), "5", close_ms, "500", 1, "2", "200", "0"]


def test_ohlcv_close_is_unavailable_until_after_exchange_close_time() -> None:
    first_open = 1_700_000_000_000
    first_close = first_open + 59_999
    second_open = first_open + 60_000
    second_close = first_open + 119_999

    class StubBinance(BinanceClient):
        def iter_klines(self, **_: object) -> Iterator[list[object]]:
            yield _kline(first_open, first_close, 101.0)
            yield _kline(second_open, second_close, 999.0)

    client = StubBinance()
    before_second_close = client.klines_df(start=first_open, end=second_close)
    at_second_availability = client.klines_df(start=first_open, end=second_close + 1)

    assert before_second_close["close"].tolist() == [101.0]
    assert at_second_availability["close"].tolist() == [101.0, 999.0]
    assert at_second_availability["ts"].tolist() == [
        pd.Timestamp(first_close + 1, unit="ms", tz="UTC"),
        pd.Timestamp(second_close + 1, unit="ms", tz="UTC"),
    ]


def test_walk_forward_test_calibration_uses_preceding_markets_only(monkeypatch) -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "condition_id": f"m{index}",
                "end_dt": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=5 * index),
                "snapshot_dt": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=5 * index, seconds=-45),
                "realized_direction": "UP" if index % 2 == 0 else "DOWN",
                "market_up_price": 0.55,
                "market_down_price": 0.45,
                "model_prob_up": 0.70 if index % 2 == 0 else 0.30,
                "score_bps": 10.0 if index % 2 == 0 else -10.0,
                "momentum_1m_bps": 1.0,
                "momentum_3m_bps": 1.0,
            }
        )
    candidates = make_side_candidates(pd.DataFrame(rows))
    original = walk_forward_module.calibrate_candidates
    calibration_pairs: list[tuple[set[str], set[str]]] = []

    def recording_calibrator(reference: pd.DataFrame, target: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        reference_ids = set(reference["condition_id"].astype(str))
        target_ids = set(target["condition_id"].astype(str))
        if reference_ids != target_ids:
            calibration_pairs.append((reference_ids, target_ids))
        return original(reference, target, **kwargs)

    monkeypatch.setattr(walk_forward_module, "calibrate_candidates", recording_calibrator)
    permissive_rule = {
        "min_edge_lower": [-1.0],
        "min_p_lower": [0.0],
        "min_price": [0.0],
        "max_price": [1.0],
        "min_abs_score_bps": [0.0],
    }
    _, folds, _ = walk_forward_module.walk_forward_backtest(
        candidates,
        config=WalkForwardConfig(train_markets=4, test_markets=2, min_train_trades=1, min_bin_observations=1),
        rule_grid=permissive_rule,
    )

    assert len(calibration_pairs) == 2
    assert [len(reference) for reference, _ in calibration_pairs] == [4, 6]
    assert [len(target) for _, target in calibration_pairs] == [2, 2]
    assert all(reference.isdisjoint(target) for reference, target in calibration_pairs)
    assert folds["leakage_check_passed"].all()


def test_unresolved_markets_are_excluded_without_external_label_fallback() -> None:
    frame = pd.DataFrame(
        [
            {
                "condition_id": "unresolved",
                "outcomes": ["Up", "Down"],
                "outcome_prices": [0.52, 0.48],
                "btc_external_direction": "UP",
            },
            {
                "condition_id": "resolved",
                "outcomes": ["Up", "Down"],
                "outcome_prices": [1.0, 0.0],
                "btc_external_direction": "DOWN",
            },
        ]
    )

    resolved = append_resolved_outcomes(frame, require_resolved=True, allow_external_btc_fallback=False)

    assert resolved["condition_id"].tolist() == ["resolved"]
    assert resolved.loc[0, "realized_direction"] == "UP"
    assert resolved.loc[0, "realized_direction_source"] == "outcome_prices"


def test_exit_pnl_deducts_entry_and_exit_fees_and_ignores_post_resolution_prices() -> None:
    entry_price = 0.50
    exit_price = 0.70
    stake = 10.0
    end = pd.Timestamp("2026-01-01T00:05:00Z")
    entries = pd.DataFrame(
        [
            {
                "condition_id": "m1",
                "side": "UP",
                "snapshot_dt": pd.Timestamp("2026-01-01T00:04:00Z"),
                "end_dt": end,
                "market_price": entry_price,
                "up_asset_id": "up1",
                "down_asset_id": "down1",
                "won": True,
            }
        ]
    )
    prices = pd.DataFrame(
        {
            "condition_id": ["m1", "m1"],
            "asset_id": ["up1", "up1"],
            "ts": [end - pd.Timedelta(seconds=5), end + pd.Timedelta(seconds=1)],
            "p": [exit_price, 0.99],
        }
    )

    trades = simulate_exit_policy(
        entries,
        prices,
        ExitPolicy(take_profit=0.15, target_price=None, stop_loss=None),
        stake_usdc=stake,
    )
    entry_fee = taker_fee_per_share(entry_price)
    exit_fee = taker_fee_per_share(exit_price)
    expected = shares_for_stake(stake, entry_price) * (exit_price - entry_price - entry_fee - exit_fee)

    assert trades.loc[0, "exit_reason"] == "TAKE_PROFIT"
    assert trades.loc[0, "exit_price"] == exit_price
    assert trades.loc[0, "path_points_seen"] == 1
    assert math.isclose(trades.loc[0, "pnl_usdc"], expected, rel_tol=0.0, abs_tol=1e-12)


def test_walk_forward_uses_expanding_nonoverlapping_test_boundaries() -> None:
    rows = []
    for index in range(9):
        rows.append(
            {
                "condition_id": f"m{index}",
                "end_dt": pd.Timestamp("2026-02-01", tz="UTC") + pd.Timedelta(minutes=5 * index),
                "snapshot_dt": pd.Timestamp("2026-02-01", tz="UTC") + pd.Timedelta(minutes=5 * index, seconds=-45),
                "realized_direction": "UP",
                "market_up_price": 0.50,
                "market_down_price": 0.50,
                "model_prob_up": 0.80,
                "score_bps": 20.0,
                "momentum_1m_bps": 1.0,
                "momentum_3m_bps": 1.0,
            }
        )
    candidates = make_side_candidates(pd.DataFrame(rows))
    rule = WalkForwardRule(-1.0, 0.0, 0.0, 1.0, 0.0)
    _, folds, _ = walk_forward_module.walk_forward_backtest(
        candidates,
        config=WalkForwardConfig(train_markets=4, test_markets=2, min_train_trades=1, min_bin_observations=1),
        rule_grid={
            "min_edge_lower": [rule.min_edge_lower],
            "min_p_lower": [rule.min_p_lower],
            "min_price": [rule.min_price],
            "max_price": [rule.max_price],
            "min_abs_score_bps": [rule.min_abs_score_bps],
        },
    )

    assert folds["train_markets"].tolist() == [4, 6, 8]
    assert folds["test_markets"].tolist() == [2, 2, 1]
    assert (pd.to_datetime(folds["train_end_dt"]) < pd.to_datetime(folds["test_start_dt"])).all()
    assert (
        pd.to_datetime(folds["test_start_dt"]).iloc[1:].reset_index(drop=True)
        > pd.to_datetime(folds["test_end_dt"]).iloc[:-1].reset_index(drop=True)
    ).all()


def test_backtest_cli_main_path_imports_and_parses_help() -> None:
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/backtest_thresholds.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0
    assert "Build features, search strategy thresholds" in result.stdout
