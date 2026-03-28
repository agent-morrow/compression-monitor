#!/usr/bin/env python3
"""
negative_space_log.py — structured logging for decisions-not-to-act.

Companion to compression-monitor's behavioral instruments. Where
ghost_lexicon.py, behavioral_footprint.py, and semantic_drift.py
measure what the agent *did*, this module logs what the agent
*considered and set aside*.

Schema: two sibling record types
  - negative_space: a skipped option, logged at rejection time
  - skip_resolution: a resolution event linking back to a prior skip

See: https://github.com/agent-morrow/compression-monitor/issues/8

Usage:
    from negative_space_log import NegativeSpaceLog

    log = NegativeSpaceLog("agent_skip_log.jsonl")
    skip_id = log.log_skip(
        cycle_id="heartbeat-20260328T1920Z",
        option_considered="escalate to operator",
        criterion="confidence_threshold_not_met",
        estimated_value=0.3,
        confidence=0.7,
        significance="medium",
    )
    # ... later, when the consequence becomes visible:
    log.log_resolution(
        cycle_id="heartbeat-20260328T2040Z",
        resolves_skip_id=skip_id,
        outcome="option_taken",
        counterfactual_delta=None,
        notes="reconsidered and escalated two cycles later",
    )

    # Calibration analysis once >= 10 resolutions are present:
    report = log.calibration_report()
    print(report)
"""

import json
import uuid
import datetime
from pathlib import Path
from typing import Optional, Literal, Union
from collections import defaultdict

Significance = Literal["low", "medium", "high", "critical"]
Outcome = Literal[
    "option_taken",
    "option_irrelevant",
    "option_closed",
    "counterfactual_confirmed",
]


def _now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class NegativeSpaceLog:
    """
    Append-only log for negative-space events.

    Records are newline-delimited JSON. Two record_types:
      - "negative_space":  a skipped option
      - "skip_resolution": links back to a negative_space record

    Design constraint: neither record type mutates existing records.
    Resolution events reference skips by skip_id; the log stays
    fully append-only so existing pipeline assumptions are preserved.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def log_skip(
        self,
        cycle_id: str,
        option_considered: str,
        criterion: str,
        significance: Significance,
        estimated_value: Optional[float] = None,
        confidence: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Log a decision-not-to-act. Returns the skip_id for later resolution.

        Args:
            cycle_id:          Heartbeat or session cycle identifier.
            option_considered: Human-readable description of the skipped option.
            criterion:         Reason for rejection (e.g. "confidence_threshold_not_met",
                               "already_handled", "too_risky", "out_of_scope").
            significance:      Categorical importance: low | medium | high | critical.
                               Required when estimated_value is None.
            estimated_value:   Numeric estimate of option value, or None.
            confidence:        Confidence in the estimated_value, or None.
            notes:             Optional free-text annotation.

        Returns:
            skip_id: unique identifier for this skip record.
        """
        skip_id = f"skip_{cycle_id}_{uuid.uuid4().hex[:6]}"
        record = {
            "record_type": "negative_space",
            "skip_id": skip_id,
            "cycle_id": cycle_id,
            "option_considered": option_considered,
            "criterion_for_rejection": criterion,
            "estimated_value": estimated_value,
            "confidence": confidence,
            "significance": significance,
            "notes": notes,
            "timestamp": _now_iso(),
        }
        self._append(record)
        return skip_id

    def log_resolution(
        self,
        cycle_id: str,
        resolves_skip_id: str,
        outcome: Outcome,
        counterfactual_delta: Optional[Union[float, dict]] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Log the resolution of a prior skip event.

        Args:
            cycle_id:             Current cycle identifier (when resolution occurred).
            resolves_skip_id:     The skip_id of the negative_space record being resolved.
            outcome:              Categorical outcome:
                                    "option_taken"            — skipped option was later executed
                                    "option_irrelevant"       — window closed, option no longer applicable
                                    "option_closed"           — external state changed, option gone
                                    "counterfactual_confirmed"— skip had a measurable observed consequence
            counterfactual_delta: Numeric or structured delta for measurable domains (e.g. P&L).
                                  None when not quantifiable.
            notes:                Optional annotation.
        """
        record = {
            "record_type": "skip_resolution",
            "cycle_id": cycle_id,
            "resolves_skip_id": resolves_skip_id,
            "outcome": outcome,
            "counterfactual_delta": counterfactual_delta,
            "notes": notes,
            "timestamp": _now_iso(),
        }
        self._append(record)

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> list[dict]:
        """Load all records from the log file."""
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def calibration_report(self, min_resolutions: int = 10) -> str:
        """
        Compute calibration error per significance level.

        For each resolved skip, asks: did the agent's ex-ante significance
        label predict the observed outcome severity?

        Calibration check:
          - high/critical skips that resolve as option_irrelevant = under-confidence
          - low skips that resolve as counterfactual_confirmed = under-confidence
          - high/critical skips that resolve as counterfactual_confirmed = well-calibrated
          - low/medium skips that resolve as option_taken = acceptable
        """
        records = self.load()
        skips = {r["skip_id"]: r for r in records if r["record_type"] == "negative_space"}
        resolutions = [r for r in records if r["record_type"] == "skip_resolution"]

        if len(resolutions) < min_resolutions:
            return (
                f"Calibration report: only {len(resolutions)} resolution events recorded "
                f"(minimum {min_resolutions} required). Accumulate more data before "
                "interpreting significance labels."
            )

        # outcome severity: how much did the skip actually matter?
        severity_map = {
            "counterfactual_confirmed": 3,
            "option_taken": 2,
            "option_closed": 1,
            "option_irrelevant": 0,
        }
        significance_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        per_level: dict[str, list[int]] = defaultdict(list)
        unresolvable = 0

        for res in resolutions:
            skip_id = res.get("resolves_skip_id")
            skip = skips.get(skip_id)
            if not skip:
                unresolvable += 1
                continue
            predicted = significance_map.get(skip.get("significance", "medium"), 1)
            observed = severity_map.get(res.get("outcome", "option_irrelevant"), 0)
            per_level[skip.get("significance", "unknown")].append(observed - predicted)

        lines = [
            "## Negative-space calibration report",
            f"Total resolutions: {len(resolutions)}  |  Unresolvable: {unresolvable}",
            "",
            "| Significance | Count | Mean delta (observed - predicted) | Interpretation |",
            "|---|---|---|---|",
        ]
        for level in ("low", "medium", "high", "critical"):
            deltas = per_level.get(level, [])
            if not deltas:
                continue
            mean_delta = sum(deltas) / len(deltas)
            if mean_delta > 0.5:
                interp = "under-estimating importance (label too low)"
            elif mean_delta < -0.5:
                interp = "over-estimating importance (label too high)"
            else:
                interp = "reasonably calibrated"
            lines.append(
                f"| {level} | {len(deltas)} | {mean_delta:+.2f} | {interp} |"
            )

        lines.extend([
            "",
            "A persistent positive delta means the agent is systematically labelling "
            "skips as less important than they turn out to be — a calibration gap that "
            "may worsen after compaction events.",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp = f.name

    log = NegativeSpaceLog(tmp)

    # Log some skips
    skip_a = log.log_skip(
        cycle_id="heartbeat-20260328T1920Z",
        option_considered="escalate to operator",
        criterion="confidence_threshold_not_met",
        significance="medium",
        estimated_value=0.3,
        confidence=0.7,
    )
    skip_b = log.log_skip(
        cycle_id="heartbeat-20260328T1940Z",
        option_considered="run verification check",
        criterion="already_handled",
        significance="low",
    )
    skip_c = log.log_skip(
        cycle_id="heartbeat-20260328T2000Z",
        option_considered="file operator outbox entry",
        criterion="no_new_information",
        significance="high",
    )

    # Resolve them
    log.log_resolution(
        cycle_id="heartbeat-20260328T2040Z",
        resolves_skip_id=skip_a,
        outcome="option_taken",
        notes="escalated in the following cycle",
    )
    log.log_resolution(
        cycle_id="heartbeat-20260328T2100Z",
        resolves_skip_id=skip_b,
        outcome="option_irrelevant",
    )
    log.log_resolution(
        cycle_id="heartbeat-20260328T2120Z",
        resolves_skip_id=skip_c,
        outcome="counterfactual_confirmed",
        counterfactual_delta=None,
        notes="operator noticed the missing entry independently",
    )

    records = log.load()
    print(f"Logged {len(records)} records\n")
    for r in records:
        print(json.dumps(r))

    print()
    print(log.calibration_report(min_resolutions=2))

    os.unlink(tmp)
