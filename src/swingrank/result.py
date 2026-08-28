"""Result objects and atomic-ish artifact output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """Candidate table, exclusions, and a reproducibility manifest."""

    candidates: pd.DataFrame
    exclusions: pd.DataFrame
    manifest: dict[str, Any]

    def write(
        self, output_directory: str | Path, prefix: str = "screen"
    ) -> dict[str, Path]:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "candidates": directory / f"{prefix}.candidates.csv",
            "exclusions": directory / f"{prefix}.exclusions.csv",
            "manifest": directory / f"{prefix}.manifest.json",
        }
        self.candidates.to_csv(paths["candidates"], index=False, float_format="%.12g")
        self.exclusions.to_csv(paths["exclusions"], index=False, float_format="%.12g")
        paths["manifest"].write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return paths
