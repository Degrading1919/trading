from __future__ import annotations

import numpy as np
import pandas as pd

from swingrank.adaptive.config import AdaptiveConfig
from swingrank.adaptive.schema import (
    ADAPTIVE_FEATURE_COLUMNS,
    TRAINING_SAMPLE_COLUMNS,
)
from swingrank.audit import sha256_json
from swingrank.result import ScreeningResult


def adaptive_test_config(**overrides) -> AdaptiveConfig:
    values = {
        "training_window_calendar_days": 2_000,
        "minimum_training_cohorts": 6,
        "minimum_training_samples": 50,
        "minimum_class_samples": 10,
    }
    values.update(overrides)
    return AdaptiveConfig(**values)


def make_training_panel(
    baseline: ScreeningResult,
    *,
    cohort_count: int = 12,
    start: str = "2022-01-03",
) -> pd.DataFrame:
    symbols = sorted(baseline.candidates["symbol"].astype(str))
    cohort_dates = pd.date_range(start=start, periods=cohort_count, freq="MS", tz="UTC")
    rows = []
    base_values = np.linspace(-1.0, 1.0, len(symbols))
    for cohort_index, formation_day in enumerate(cohort_dates):
        formation_at = formation_day + pd.Timedelta(hours=21)
        label_start = formation_day.tz_localize(None).normalize()
        label_end = label_start + pd.offsets.BDay(42)
        label_end_at = label_end.tz_localize("UTC") + pd.Timedelta(hours=22)
        fingerprint = sha256_json(
            {"formation_at": formation_at.isoformat(), "kind": "test-baseline"}
        )
        factor_values = {
            feature: np.roll(
                base_values,
                cohort_index + feature_index * 2,
            )
            for feature_index, feature in enumerate(ADAPTIVE_FEATURE_COLUMNS)
        }
        for symbol_index, symbol in enumerate(symbols):
            signal = (
                0.06 * factor_values[ADAPTIVE_FEATURE_COLUMNS[0]][symbol_index]
                + 0.02 * factor_values[ADAPTIVE_FEATURE_COLUMNS[1]][symbol_index]
                + 0.001 * np.sin(cohort_index + symbol_index)
            )
            rows.append(
                {
                    "symbol": symbol,
                    "formation_at": formation_at,
                    "label_start_session": label_start,
                    "label_end_session": label_end,
                    "label_end_at": label_end_at,
                    "horizon_sessions": 42,
                    "forward_return": signal,
                    "formation_universe_size": len(symbols),
                    "baseline_methodology_version": baseline.manifest[
                        "methodology_version"
                    ],
                    "baseline_config_hash": baseline.manifest["config_hash"],
                    "baseline_run_fingerprint": fingerprint,
                    "available_at": label_end_at + pd.Timedelta(minutes=5),
                    **{
                        feature: values[symbol_index]
                        for feature, values in factor_values.items()
                    },
                }
            )
    return pd.DataFrame(rows).loc[:, TRAINING_SAMPLE_COLUMNS]
