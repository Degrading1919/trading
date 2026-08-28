"""Adaptive result object with side-by-side baseline artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from swingrank.result import ScreeningResult


@dataclass(frozen=True, slots=True)
class AdaptiveScreeningResult:
    """Adaptive ranks, learned weights, manifest, and untouched baseline run."""

    baseline: ScreeningResult
    candidates: pd.DataFrame
    learned_weights: pd.DataFrame
    manifest: dict[str, Any]

    def write(
        self, output_directory: str | Path, prefix: str = "screen"
    ) -> dict[str, Path]:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        baseline_paths = self.baseline.write(directory, f"{prefix}.baseline")
        paths = {
            "adaptive_candidates": directory / f"{prefix}.adaptive.candidates.csv",
            "adaptive_weights": directory / f"{prefix}.adaptive.weights.csv",
            "adaptive_manifest": directory / f"{prefix}.adaptive.manifest.json",
            **{f"baseline_{key}": path for key, path in baseline_paths.items()},
        }
        self.candidates.to_csv(
            paths["adaptive_candidates"], index=False, float_format="%.12g"
        )
        self.learned_weights.to_csv(
            paths["adaptive_weights"], index=False, float_format="%.12g"
        )
        paths["adaptive_manifest"].write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return paths
