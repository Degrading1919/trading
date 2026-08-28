"""Canonical hashing helpers for reproducible run manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_scalar(value: Any) -> str:
    if value is None or value is pd.NA:
        return "null:"
    try:
        if pd.isna(value):
            return "null:"
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC")
        return "timestamp:" + value.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC)
        return "datetime:" + value.isoformat()
    if isinstance(value, date):
        return "date:" + value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return "bool:" + ("1" if bool(value) else "0")
    if isinstance(value, (int, np.integer)):
        return "int:" + str(int(value))
    if isinstance(value, (float, np.floating)):
        return "float:" + format(float(value), ".17g")
    return "str:" + str(value)


def hash_frame(frame: pd.DataFrame) -> str:
    """Hash a frame independent of input row/column order."""

    if frame.empty:
        return sha256(b"<empty>").hexdigest()
    columns = sorted(str(column) for column in frame.columns)
    rows: list[tuple[str, ...]] = []
    ordered = frame.loc[:, columns]
    for row in ordered.itertuples(index=False, name=None):
        rows.append(tuple(_canonical_scalar(value) for value in row))
    rows.sort()
    payload = canonical_json({"columns": columns, "rows": rows})
    return sha256(payload.encode("utf-8")).hexdigest()


def combined_input_hash(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[str, dict[str, str]]:
    component_hashes = {name: hash_frame(frame) for name, frame in frames.items()}
    return sha256_json(component_hashes), component_hashes
