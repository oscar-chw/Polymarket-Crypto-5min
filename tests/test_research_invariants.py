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
from polymarket_crypto_5min.exit_backtest import ExitPolicy, simulate_exit_policy, walk_forward_exit_backtest
from polymarket_crypto_5min.features import (
    ENTRY_PRICE_RULE,
    build_training_frame,
    shares_for_stake,
    taker_fee_per_share,
)
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


def _one_market_frame(prints: list[tuple[str, float]], *, snapshot_seconds_before_close: int = 45) -> pd.DataFrame:
    start = pd.Timestamp("2026-07-05T12:00:00Z")
    markets = pd.DataFrame(
        [
            {
                "condition_id": "m1",
                "start_ts": int(start.timestamp()),
                "end_ts": int(start.timestamp()) + 300,
                "up_asset_id": "up1",
                "down_asset_id": "down1",
                "outcomes": ["Up", "Down"],
                "outcome_prices": [1, 0],
            }
        ]
    )
    candles = pd.DataFrame(
        {
            "ts": pd.date_range(start - pd.Timedelta(minutes=5), periods=11, freq="1min", tz="UTC"),
            "close": [100.0] * 11,
        }
    )
    poly_prices = pd.DataFrame(
        {
            "condition_id": ["m1"] * len(prints),
            "asset_id": ["up1"] * len(prints),
            "ts": pd.to_datetime([ts for ts, _ in prints], utc=True),
            "p": [p for _, p in prints],
        }
    )
    return build_training_frame(
        markets, candles, poly_prices, snapshot_seconds_before_close=snapshot_seconds_before_close
    )


def test_entry_price_ignores_prints_from_before_the_window_opened() -> None:
    # Snapshot is 12:04:15. The only print at or before it is from 11:52, when
    # this market's window had not opened; 0.88 at 12:04:20 is after the decision.
    frame = _one_market_frame([("2026-07-05T11:52:00Z", 0.50), ("2026-07-05T12:04:20Z", 0.88)])

    assert pd.isna(frame.loc[0, "market_up_price"])
    assert pd.isna(frame.loc[0, "market_up_price_ts"])


def test_entry_price_ignores_a_fresh_print_from_before_the_window_opened() -> None:
    # Snapshot 12:00:10; the 11:59:50 print is only 20 s old, so the age bound
    # alone would accept it, but the market did not exist yet.
    frame = _one_market_frame([("2026-07-05T11:59:50Z", 0.50)], snapshot_seconds_before_close=290)

    assert pd.isna(frame.loc[0, "market_up_price"])


def test_entry_price_ignores_in_window_prints_older_than_the_age_bound() -> None:
    frame = _one_market_frame([("2026-07-05T12:01:00Z", 0.50)])

    assert pd.isna(frame.loc[0, "market_up_price"])


def test_entry_price_uses_fresh_in_window_print_and_records_the_rule() -> None:
    frame = _one_market_frame([("2026-07-05T12:01:00Z", 0.50), ("2026-07-05T12:04:00Z", 0.61)])

    assert frame.loc[0, "market_up_price"] == 0.61
    assert frame.loc[0, "market_up_price_ts"] == pd.Timestamp("2026-07-05T12:04:00Z")
    assert frame.loc[0, "entry_price_rule"] == ENTRY_PRICE_RULE


def _overlapping_fold_candidates(*, overlap: bool = True) -> pd.DataFrame:
    # Ordered by end_dt, m1 is the last training market and m2 the first test
    # market. With overlap, m2 settles after m1 but decides (12:09:45) before
    # m1 settles (12:10:00), so m1's label is unknown at m2's decision.
    if overlap:
        ends = ["12:05:00", "12:10:00", "12:10:30", "12:15:30"]
    else:
        ends = ["12:05:00", "12:10:00", "12:15:00", "12:20:00"]
    rows = []
    for index, end in enumerate(ends):
        end_dt = pd.Timestamp(f"2026-07-05T{end}Z")
        rows.append(
            {
                "condition_id": f"m{index}",
                "end_dt": end_dt,
                "snapshot_dt": end_dt - pd.Timedelta(seconds=45),
                "realized_direction": "UP",
                "market_up_price": 0.60,
                "market_down_price": 0.40,
                "model_prob_up": 0.80,
                "score_bps": 10.0,
                "momentum_1m_bps": 1.0,
                "momentum_3m_bps": 1.0,
                "up_asset_id": f"up{index}",
                "down_asset_id": f"down{index}",
            }
        )
    return make_side_candidates(pd.DataFrame(rows))


def test_fold_leakage_compares_training_labels_with_test_decision_time() -> None:
    config = WalkForwardConfig(train_markets=2, test_markets=2, min_train_trades=1, min_bin_observations=1)
    candidates = _overlapping_fold_candidates()

    _, folds, _ = walk_forward_module.walk_forward_backtest(
        candidates,
        config=config,
        rule_grid={
            "min_edge_lower": [-1.0],
            "min_p_lower": [0.0],
            "min_price": [0.0],
            "max_price": [1.0],
            "min_abs_score_bps": [0.0],
        },
    )
    _, exit_folds, _ = walk_forward_exit_backtest(
        candidates,
        pd.DataFrame(columns=["condition_id", "asset_id", "ts", "p"]),
        config=config,
        entry_rules=[WalkForwardRule(-1.0, 0.0, 0.0, 1.0, 0.0)],
        exit_policies=[ExitPolicy(take_profit=0.05, target_price=None, stop_loss=None)],
    )

    assert folds["leakage_check_passed"].tolist() == [False]
    assert exit_folds["leakage_check_passed"].tolist() == [False]
    assert folds.loc[0, "test_first_decision_dt"] == pd.Timestamp("2026-07-05T12:09:45Z")


def _run_full_history_runner(tmp_path: Path, *, overlap: bool) -> tuple[subprocess.CompletedProcess[str], Path]:
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    candidates = _overlapping_fold_candidates(overlap=overlap).drop_duplicates("condition_id")
    markets = pd.DataFrame(
        {
            "condition_id": candidates["condition_id"],
            "market_id": candidates["condition_id"],
            "start_ts": (candidates["end_dt"].astype("int64") // 10**9) - 300,
            "end_ts": candidates["end_dt"].astype("int64") // 10**9,
            "up_asset_id": candidates["up_asset_id"],
            "down_asset_id": candidates["down_asset_id"],
            "outcomes": '["Up", "Down"]',
            "outcome_prices": '["1", "0"]',
        }
    )
    markets.to_csv(raw / "btc_5m_markets.csv", index=False)
    print_ts = candidates["snapshot_dt"] - pd.Timedelta(seconds=15)
    for role, price in (("up", 0.60), ("down", 0.40)):
        pd.DataFrame(
            {
                "condition_id": candidates["condition_id"],
                "asset_id": candidates[f"{role}_asset_id"],
                "t": print_ts.astype("int64") // 10**9,
                "p": price,
                "ts": print_ts,
            }
        ).to_csv(raw / f"btc_5m_{role}_price_history.csv", index=False)
    candle_ts = pd.date_range("2026-07-05T11:50:00Z", "2026-07-05T12:25:00Z", freq="1min")
    pd.DataFrame(
        {
            "ts": candle_ts,
            "close": [100.0 + 0.1 * i for i in range(len(candle_ts))],
            "timestamp_semantics": "close_available_at",
        }
    ).to_csv(raw / "btc_usdt_1m.csv", index=False)

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_btc_5m_full_history_walk_forward.py",
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(out),
            "--skip-existing",
            "--price-source",
            "clob",
            "--train-markets",
            "2",
            "--test-markets",
            "2",
            "--min-train-trades",
            "1",
            "--min-bin-observations",
            "1",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result, out


def test_full_history_runner_exits_nonzero_on_fold_leakage(tmp_path: Path) -> None:
    result, out = _run_full_history_runner(tmp_path, overlap=True)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "Leakage check failed" in result.stderr
    assert (out / "side_candidates.csv").is_file()
    assert not (out / "walk_forward_trades.csv").exists()
    assert not (out / "walk_forward_metrics.csv").exists()


def test_full_history_runner_writes_results_when_folds_do_not_leak(tmp_path: Path) -> None:
    result, out = _run_full_history_runner(tmp_path, overlap=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "walk_forward_trades.csv").is_file()
    assert '"leakage_violations": 0' in result.stdout
