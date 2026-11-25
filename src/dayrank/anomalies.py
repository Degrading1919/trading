"""Rank unusual market observations against information available before the session."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import median

from dayrank.data import ET, timestamp

METRICS = {
    "short_return": "trailing_5m_close_return_bps",
    "gap": "return_since_previous_close_bps",
    "dollars": "trailing_5m_dollar_volume",
    "trades": "trailing_5m_trade_count",
}
COMPARABLE_METADATA = (
    "feed",
    "observation",
    "bar_lag_seconds",
    "price_edges",
    "prior_dollar_volume_edges",
    "feature_windows",
)


@dataclass(frozen=True, slots=True)
class AnomalyConfig:
    lookback: int = 30
    minimum_history: int = 20
    alert_budget: int = 15
    minimum_peer_count: int = 10
    minimum_bars: int = 2
    minimum_trades: int = 5
    maximum_tail_fraction: float = 0.10
    minimum_surprise: float = 3.5
    minimum_peer_surprise: float = 2.0

    def __post_init__(self) -> None:
        if not 5 <= self.minimum_history <= self.lookback or self.alert_budget < 1:
            raise ValueError(
                "Use a positive alert budget and 5 or more historical observations"
            )


@dataclass(frozen=True, slots=True)
class Observation:
    symbol: str
    decision: str
    clock: str
    cohort: str
    bar_status: str
    quote_status: str
    spread: float | None
    bars: int
    values: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class SessionSample:
    manifest: dict
    rows: list[Observation]


@dataclass(frozen=True, slots=True)
class Surprise:
    value: float
    expected: float
    score: float
    tail_fraction: float
    observations: int


def _numeric(raw: str) -> float | None:
    if raw == "":
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("Snapshot contains a non-finite feature")
    return value


def read_sample(folder: Path) -> SessionSample:
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if manifest.get("schema") != 1 or 5 not in manifest["feature_windows"]:
        raise ValueError(f"{folder} is missing the supported snapshot schema")
    day = date.fromisoformat(manifest["date"])
    expected_times = {timestamp(item["decision_at"]) for item in manifest["coverage"]}
    seen = defaultdict(set)
    rows = []
    with gzip.open(
        folder / "snapshots.csv.gz", "rt", encoding="utf-8", newline=""
    ) as file:
        for row in csv.DictReader(file):
            decision = timestamp(row["decision_at"])
            local = decision.tz_convert(ET)
            name = row["symbol"]
            if (
                row["date"] != manifest["date"]
                or row["feed"] != manifest["feed"]
                or local.date() != day
                or decision not in expected_times
                or name in seen[decision]
            ):
                raise ValueError(f"Unexpected or duplicate snapshot in {folder}")
            for field in ("last_bar_observed_at", "quote_time"):
                if row.get(field) and timestamp(row[field]) > decision:
                    raise ValueError(f"Snapshot contains a future {field} in {folder}")
            seen[decision].add(name)
            bars = int(row["trailing_5m_bars"])
            values = {key: _numeric(row[column]) for key, column in METRICS.items()}
            if bars == 0:
                # A complete elapsed window with zero minute bars has zero observed activity.
                values["trades"] = 0.0
            if row["bar_status"] != "fresh":
                values["short_return"] = None
                values["gap"] = None
            rows.append(
                Observation(
                    name,
                    decision.isoformat(),
                    local.strftime("%H:%M"),
                    row["peer_cohort"],
                    row["bar_status"],
                    row["quote_status"],
                    _numeric(row["spread_bps"]),
                    bars,
                    values,
                )
            )
    if (
        len(rows) != manifest["snapshot_rows"]
        or set(seen) != expected_times
        or any(len(names) != manifest["eligible_listings"] for names in seen.values())
        or len({frozenset(names) for names in seen.values()}) != 1
    ):
        raise ValueError(f"Snapshot coverage does not match manifest for {folder}")
    return SessionSample(manifest, rows)


def recent_history(
    target: SessionSample, history: list[SessionSample], lookback: int
) -> list[SessionSample]:
    meta = target.manifest
    dates = [s.manifest["date"] for s in history]
    if len(set(dates)) != len(dates) or any(day >= meta["date"] for day in dates):
        raise ValueError(
            "Baseline sessions must be distinct and precede the target session"
        )
    scope = meta.get("population", "eligible_exchange_listings")
    if any(
        any(s.manifest[key] != meta[key] for key in COMPARABLE_METADATA)
        or s.manifest.get("population", "eligible_exchange_listings") != scope
        for s in history
    ):
        raise ValueError(
            "Baseline and target need the same feed, sampling design, and population"
        )
    return sorted(history, key=lambda sample: sample.manifest["date"])[-lookback:]


def _surprise(
    value: float | None,
    historical: list[float],
    minimum: int,
    *,
    activity: bool = False,
) -> Surprise | None:
    if value is None or len(historical) < minimum:
        return None
    transform = math.log1p if activity else float
    current = transform(value)
    values = [transform(x) for x in historical]
    center = median(values)
    deviation = current - center
    mad = median(abs(x - center) for x in values)
    scale = max(1.4826 * mad, math.log(2) if activity else 10)
    extreme = sum(
        (x >= current if activity else abs(x - center) >= abs(deviation))
        for x in values
    )
    return Surprise(
        value,
        math.expm1(center) if activity else center,
        deviation / scale,
        (extreme + 1) / (len(values) + 1),
        len(values),
    )


def _is_unusual(
    result: Surprise | None, config: AnomalyConfig, *, positive: bool = False
) -> bool:
    return bool(
        result is not None
        and (result.score if positive else abs(result.score)) >= config.minimum_surprise
        and result.tail_fraction <= config.maximum_tail_fraction
    )


def _peer_groups(
    rows: list[Observation], config: AnomalyConfig
) -> dict[tuple, list[tuple[str, float]]]:
    groups = defaultdict(list)
    for row in rows:
        if row.bar_status != "fresh" or row.bars < config.minimum_bars:
            continue
        short = row.values["short_return"]
        if short is not None:
            groups[row.clock, "market", "short_return"].append((row.symbol, short))
        if row.cohort:
            for metric in ("short_return", "gap"):
                value = row.values[metric]
                if value is not None:
                    groups[row.clock, row.cohort, metric].append((row.symbol, value))
    return groups


def _peers(
    row: Observation,
    metric: str,
    groups: dict[tuple, list[tuple[str, float]]],
    config: AnomalyConfig,
) -> Surprise | None:
    others = [
        value
        for symbol, value in groups.get((row.clock, row.cohort, metric), ())
        if symbol != row.symbol
    ]
    return _surprise(row.values[metric], others, config.minimum_peer_count)


def _details(prefix: str, data: Surprise | None) -> dict:
    return {
        f"{prefix}_observed": data.value if data else None,
        f"{prefix}_expected": data.expected if data else None,
        f"{prefix}_score": data.score if data else None,
        f"{prefix}_tail_fraction": data.tail_fraction if data else None,
        f"{prefix}_observations": data.observations if data else 0,
    }


def _reason(metric: str, primary: Surprise, support: Surprise | None = None) -> str:
    if support is not None:
        return (
            f"Five-minute dollar volume was ${primary.value:,.0f} versus a typical "
            f"${primary.expected:,.0f}, with {support.value:,.0f} trades versus "
            f"{support.expected:,.0f}."
        )
    label = {
        "short_return": "Recent return",
        "gap": "Return from the prior close",
        "median_short_return": "Group median recent return",
    }[metric]
    return (
        f"{label} was {primary.value / 100:+.2f}% versus a typical "
        f"{primary.expected / 100:+.2f}% at this time."
    )


def detect(
    target: SessionSample,
    history: list[SessionSample],
    config: AnomalyConfig,
) -> dict:
    history = recent_history(target, history, config.lookback)
    target_clocks = {r.clock for r in target.rows}
    baselines = defaultdict(list)
    group_history = defaultdict(list)
    for prior in history:
        clock_rows = [r for r in prior.rows if r.clock in target_clocks]
        for row in clock_rows:
            for metric, value in row.values.items():
                if value is not None:
                    baselines[row.symbol, row.clock, metric].append(value)
        for (clock, cohort, metric), group in _peer_groups(clock_rows, config).items():
            if metric == "short_return" and len(group) >= config.minimum_peer_count:
                group_history[clock, cohort].append(median(value for _, value in group))

    groups = _peer_groups(target.rows, config)
    meta = target.manifest
    candidates, audit = [], []
    for row in target.rows:
        available = {
            metric: baselines.get((row.symbol, row.clock, metric), [])
            for metric in METRICS
        }
        values = {
            metric: _surprise(
                value,
                available[metric],
                config.minimum_history,
                activity=metric in {"dollars", "trades"},
            )
            for metric, value in row.values.items()
        }
        support = row.values["trades"]
        quality = (
            row.bar_status
            if row.bar_status != "fresh"
            else "thin_activity"
            if row.bars < config.minimum_bars
            or support is None
            or support < config.minimum_trades
            else "insufficient_history"
            if not any(values.values())
            else "scoreable"
        )
        audit_row = dict(
            date=meta["date"],
            feed=meta["feed"],
            symbol=row.symbol,
            decision_at=row.decision,
            clock_et=row.clock,
            data_quality=quality,
            peer_cohort=row.cohort,
            quote_status=row.quote_status,
            spread_bps=row.spread,
            trailing_5m_bars=row.bars,
            trailing_5m_trades=support,
            trailing_5m_dollars=row.values["dollars"],
            candidate_count=0,
            **{
                key: val
                for metric, evidence in values.items()
                for key, val in _details(metric, evidence).items()
            },
            **{
                f"{metric}_available_prior_sessions": len(samples)
                for metric, samples in available.items()
            },
        )
        audit.append(audit_row)
        if quality != "scoreable":
            continue
        for metric, kind in (
            ("short_return", "price_dislocation"),
            ("gap", "gap_dislocation"),
            ("dollars", "participation_surge"),
        ):
            primary = values[metric]
            if not _is_unusual(primary, config, positive=metric == "dollars"):
                continue
            if metric == "dollars":
                support_evidence = values["trades"]
                if not _is_unusual(support_evidence, config, positive=True):
                    continue
                peer = None
                priority = min(primary.score, support_evidence.score)
            else:
                support_evidence = None
                peer = _peers(row, metric, groups, config)
                priority = abs(primary.score)
            aligned = peer is not None and (
                peer.score * primary.score <= 0
                or abs(peer.score) < config.minimum_peer_surprise
            )
            candidates.append(
                dict(
                    date=meta["date"],
                    feed=meta["feed"],
                    decision_at=row.decision,
                    clock_et=row.clock,
                    scope="stock",
                    entity=row.symbol,
                    kind=kind,
                    direction="higher_activity"
                    if metric == "dollars"
                    else "above_normal"
                    if primary.score > 0
                    else "below_normal",
                    score=priority,
                    metric=metric,
                    reason=_reason(metric, primary, support_evidence),
                    peer_cohort=row.cohort,
                    peer_context="aligned"
                    if aligned
                    else "outlying"
                    if peer
                    else "unavailable",
                    quote_status=row.quote_status,
                    spread_bps=row.spread,
                    alert_status="peer_aligned" if aligned else "candidate",
                    **_details("self", primary),
                    **_details("peer", peer),
                    **_details("activity_support", support_evidence),
                )
            )
            audit_row["candidate_count"] += 1

    clock_to_decision = {r.clock: r.decision for r in target.rows}
    for (clock, cohort, metric), group in sorted(groups.items()):
        if metric != "short_return" or len(group) < config.minimum_peer_count:
            continue
        center = median(value for _, value in group)
        primary = _surprise(
            center, group_history.get((clock, cohort), []), config.minimum_history
        )
        if _is_unusual(primary, config):
            candidates.append(
                dict(
                    date=meta["date"],
                    feed=meta["feed"],
                    decision_at=clock_to_decision[clock],
                    clock_et=clock,
                    scope="market" if cohort == "market" else "peer_group",
                    entity=cohort,
                    kind="breadth_shift",
                    direction="above_normal" if primary.score > 0 else "below_normal",
                    score=abs(primary.score),
                    metric="median_short_return",
                    group_members=len(group),
                    reason=_reason("median_short_return", primary),
                    alert_status="candidate",
                    **_details("self", primary),
                )
            )
    return dict(
        audit=audit,
        candidates=candidates,
        manifest=dict(
            schema=1,
            date=meta["date"],
            feed=meta["feed"],
            config=asdict(config),
            baseline_dates=[s.manifest["date"] for s in history],
            target_clocks=sorted(target_clocks),
            snapshot_rows=len(target.rows),
            population=meta.get("population", "eligible_exchange_listings"),
            data_quality=dict(Counter(row["data_quality"] for row in audit)),
            observation=meta["observation"],
            bar_lag_seconds=meta["bar_lag_seconds"],
        ),
    )


def rank_alerts(
    report: dict,
    alert_budget: int,
    splits: set[str],
    split_status: str,
) -> None:
    candidates = sorted(
        report["candidates"],
        key=lambda row: (
            row["decision_at"],
            -row["score"],
            row["scope"],
            row["entity"],
            row["kind"],
        ),
    )
    seen, shown_at, counts, alerts = set(), set(), Counter(), []
    for row in candidates:
        stock = row["scope"] == "stock"
        row["split_check"] = (
            "not_applicable"
            if not stock
            else "reported_split"
            if row["entity"] in splits
            else "no_split_reported"
            if split_status == "available"
            else "unverified"
        )
        key = row["scope"], row["entity"], row["kind"], row["direction"]
        decision = row["decision_at"]
        entity_at = (decision, row["scope"], row["entity"])
        if stock and row["entity"] in splits:
            row["alert_status"] = "split_effective"
        elif row["alert_status"] == "candidate":
            if row["kind"] == "gap_dislocation" and split_status != "available":
                row["alert_status"] = "requires_split_check"
            elif key in seen:
                row["alert_status"] = "repeat"
            elif entity_at in shown_at:
                row["alert_status"] = "coincident_anomaly"
            elif counts[decision] >= alert_budget:
                row["alert_status"] = "below_alert_budget"
            else:
                counts[decision] += 1
                seen.add(key)
                shown_at.add(entity_at)
                row["rank"] = counts[decision]
                row["alert_status"] = "alert"
                alerts.append(row)
    report["candidates"] = candidates
    report["alerts"] = alerts
    checked = {row["entity"] for row in candidates if row["scope"] == "stock"}
    for row in report["audit"]:
        symbol = row["symbol"]
        row["split_check"] = (
            "reported_split"
            if symbol in splits
            else "no_split_reported"
            if split_status == "available" and symbol in checked
            else "unverified"
        )
    report["manifest"].update(
        split_status=split_status,
        candidate_count=len(candidates),
        alert_count=len(alerts),
        alert_status=dict(Counter(row["alert_status"] for row in candidates)),
        alerts_by_kind=dict(Counter(row["kind"] for row in alerts)),
    )
