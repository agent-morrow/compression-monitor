"""
negative_space.py — Negative-space logging for persistent agents.

Tracks two related record types as append-only JSONL log entries:

  1. SkipRecord   — an option the agent considered but did not take
  2. SkipResolution — the observed outcome for a previously-logged skip

Together these enable:
  - Post-hoc calibration of significance labels (were "high" skips actually high-consequence?)
  - Counterfactual analysis when outcomes are measurable (trading, test runs, etc.)
  - Forward traversal from skip → resolution without mutating original records

Schema design principle: append-only. SkipResolution references the original
skip by ID; the original record is never patched.

Usage:
    from compression_monitor.negative_space import NegativeSpaceLog

    log = NegativeSpaceLog("agent_skips.jsonl")

    skip_id = log.record_skip(
        cycle_id="20260328T1823Z",
        option_label="escalate_to_operator",
        reason="context already handled; escalation would be redundant",
        significance="medium",
        estimated_value=None,  # nullable when not quantifiable
        alternatives_considered=["defer_to_next_cycle", "file_outbox_note"],
    )

    # Later, when outcome is known:
    log.record_resolution(
        resolves_skip_id=skip_id,
        cycle_id="20260328T2040Z",
        outcome="option_irrelevant",
        counterfactual_delta=None,
        notes="the context was handled correctly; escalation would have been noise",
    )
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Outcome vocabulary for resolution records
# ---------------------------------------------------------------------------

class SkipOutcome(str, Enum):
    OPTION_TAKEN        = "option_taken"         # deferred option was later taken
    OPTION_IRRELEVANT   = "option_irrelevant"     # the window passed; option no longer mattered
    OPTION_CLOSED       = "option_closed"         # the option became unavailable
    COUNTERFACTUAL_CONFIRMED = "counterfactual_confirmed"  # outcome measurable, delta computable


class Significance(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SkipRecord:
    record_type: str = "skip"
    skip_id: str = field(default_factory=lambda: f"skip_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    cycle_id: str = ""
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # What was considered
    option_label: str = ""
    reason: str = ""
    significance: str = Significance.MEDIUM.value

    # Optional quantification — nullable for non-measurable decisions
    estimated_value: Optional[float] = None
    alternatives_considered: List[str] = field(default_factory=list)

    # Optional metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict())


@dataclass
class SkipResolution:
    """
    Append-only resolution event for a previously logged SkipRecord.

    Never patches the original skip record — references it by skip_id only.
    This keeps the log format immutable and pipeline-safe.
    """
    record_type: str = "skip_resolution"
    resolution_id: str = field(default_factory=lambda: f"res_{int(time.time())}_{uuid.uuid4().hex[:6]}")
    resolves_skip_id: str = ""
    cycle_id: str = ""
    timestamp_utc: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    outcome: str = SkipOutcome.OPTION_IRRELEVANT.value

    # For quantifiable domains (trading, test pass/fail, measurable metrics):
    # positive = skipping was the right call, negative = skipping cost something
    counterfactual_delta: Optional[float] = None

    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict())


# ---------------------------------------------------------------------------
# Log manager — append-only JSONL file
# ---------------------------------------------------------------------------

class NegativeSpaceLog:
    """
    Append-only JSONL log for skip records and their resolutions.

    Both record types coexist in the same file. Readers can reconstruct
    the full skip→resolution chain by indexing on skip_id.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- write side ---

    def record_skip(
        self,
        cycle_id: str,
        option_label: str,
        reason: str,
        significance: str = Significance.MEDIUM.value,
        estimated_value: Optional[float] = None,
        alternatives_considered: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> str:
        """Append a SkipRecord. Returns the skip_id for later resolution."""
        rec = SkipRecord(
            cycle_id=cycle_id,
            option_label=option_label,
            reason=reason,
            significance=significance,
            estimated_value=estimated_value,
            alternatives_considered=alternatives_considered or [],
            tags=tags or [],
            notes=notes,
        )
        self._append(rec.as_dict())
        return rec.skip_id

    def record_resolution(
        self,
        resolves_skip_id: str,
        cycle_id: str,
        outcome: str = SkipOutcome.OPTION_IRRELEVANT.value,
        counterfactual_delta: Optional[float] = None,
        notes: str = "",
    ) -> str:
        """Append a SkipResolution. Returns the resolution_id."""
        res = SkipResolution(
            resolves_skip_id=resolves_skip_id,
            cycle_id=cycle_id,
            outcome=outcome,
            counterfactual_delta=counterfactual_delta,
            notes=notes,
        )
        self._append(res.as_dict())
        return res.resolution_id

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # --- read / analysis side ---

    def read_all(self) -> List[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def skips(self) -> List[dict]:
        return [r for r in self.read_all() if r.get("record_type") == "skip"]

    def resolutions(self) -> List[dict]:
        return [r for r in self.read_all() if r.get("record_type") == "skip_resolution"]

    def resolve_chain(self, skip_id: str) -> dict:
        """Return the skip record and all resolution events for a given skip_id."""
        all_records = self.read_all()
        skip = next((r for r in all_records if r.get("skip_id") == skip_id), None)
        resolutions = [r for r in all_records if r.get("resolves_skip_id") == skip_id]
        return {"skip": skip, "resolutions": resolutions}

    def calibration_summary(self) -> dict:
        """
        Basic calibration report: for each significance level, what fraction
        of resolved skips had each outcome?

        Useful for checking whether your ex-ante significance labels predict
        anything about the actual outcomes.
        """
        skips = {r["skip_id"]: r for r in self.skips()}
        res_by_skip: dict[str, list] = {}
        for res in self.resolutions():
            sid = res.get("resolves_skip_id", "")
            res_by_skip.setdefault(sid, []).append(res)

        summary: dict = {}
        for skip_id, skip in skips.items():
            sig = skip.get("significance", "unknown")
            ress = res_by_skip.get(skip_id, [])
            for res in ress:
                outcome = res.get("outcome", "unknown")
                summary.setdefault(sig, {}).setdefault(outcome, 0)
                summary[sig][outcome] += 1

        return summary
