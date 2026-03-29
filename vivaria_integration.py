"""
vivaria_integration.py
──────────────────────
Behavioral consistency monitoring for METR's Vivaria eval runner.

Vivaria (https://github.com/METR/vivaria) runs agentic tasks in sandboxed environments.
Long-horizon tasks frequently cross context boundaries — either explicit compaction events
or implicit context pressure as trace length grows. This adapter tracks the behavioral
fingerprint of an agent across a run and surfaces a Context-Consistency Score (CCS) that
can be stored alongside Vivaria's native trace_entries and run metadata.

What it measures
────────────────
Three signals, same as the core compression-monitor framework:

1. Ghost lexicon decay — domain vocabulary present in early trace steps disappears
   after context pressure. Measured as recall over a vocabulary anchor from step N.

2. Tool-call sequence shift — Jaccard distance between the tool-use pattern in the
   first half vs. second half of a run, after accounting for task phase.

3. Semantic anchor drift — the overlap between topic keyword sets across a configurable
   rolling window.

The CCS is 1.0 minus the weighted mean of the three decay signals. 1.0 = no detected
drift; 0.0 = complete behavioral divergence.

Integration points
──────────────────
Vivaria stores trace entries in its PostgreSQL `trace_entries_t` table. This adapter
emits structured JSON that can be appended as a custom trace entry (type: "action",
content type: "json") or written to a sidecar file alongside the run's output.

Usage (run-alongside pattern)
──────────────────────────────
    from vivaria_integration import VivariaBehavioralMonitor

    monitor = VivariaBehavioralMonitor(run_id="my-run-42", window=10)

    # Call after each agent action/trace entry is written:
    monitor.record_step(
        step_index=step_i,
        tool_calls=["bash", "bash", "python"],       # tool names used this step
        output_text="Agent output text for step...", # raw output for vocab tracking
    )

    # At the end of the run (or after each compaction boundary):
    ccs = monitor.context_consistency_score()
    print(f"CCS: {ccs:.3f}")

    # Serialize for storage in trace_entries_t or a sidecar file:
    monitor.to_trace_entry()  # returns dict compatible with Vivaria's action entry schema

Boundary detection (optional)
──────────────────────────────
If you have access to the agent's message count (e.g., from a pyhooks log entry),
call monitor.flag_boundary(step_index) at the compaction event. The CCS delta
across boundaries is the most diagnostic signal.

Dependencies: compression-monitor (ghost_lexicon.py, behavioral_footprint.py,
semantic_drift.py in the same package). No Vivaria dependencies required.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ─── internal imports ────────────────────────────────────────────────────────

try:
    from ghost_lexicon import GhostLexiconTracker
    from behavioral_footprint import BehavioralFootprintTracker
    from semantic_drift import SemanticDriftTracker
except ImportError:
    try:
        from compression_monitor.ghost_lexicon import GhostLexiconTracker
        from compression_monitor.behavioral_footprint import BehavioralFootprintTracker
        from compression_monitor.semantic_drift import SemanticDriftTracker
    except ImportError:
        raise ImportError(
            "compression-monitor not found. Install from source:\n"
            "  pip install -e .\n"
            "or from PyPI (when available):\n"
            "  pip install compression-monitor"
        )


# ─── types ───────────────────────────────────────────────────────────────────

@dataclass
class StepRecord:
    step_index: int
    timestamp: float
    tool_calls: List[str]
    output_text: str
    is_boundary: bool = False           # flagged compaction boundary
    ccs_at_step: Optional[float] = None


@dataclass
class VivariaBehavioralMonitor:
    """
    Track behavioral consistency for a single Vivaria run.

    Parameters
    ----------
    run_id : str
        The Vivaria run ID. Stored in the output but not used internally.
    window : int
        Number of recent steps to use for rolling comparisons. Default 10.
    anchor_steps : int
        Number of early steps used to build the vocabulary and tool-call baseline.
        Default 5.
    ccs_weights : dict
        Relative weighting for the three signals. Default: equal thirds.
    """

    run_id: str
    window: int = 10
    anchor_steps: int = 5
    ccs_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ghost_lexicon": 0.34,
            "tool_footprint": 0.33,
            "semantic_drift": 0.33,
        }
    )

    # internal state
    _steps: List[StepRecord] = field(default_factory=list, init=False)
    _boundaries: List[int] = field(default_factory=list, init=False)
    _lexicon_tracker: Optional[GhostLexiconTracker] = field(default=None, init=False)
    _footprint_tracker: Optional[BehavioralFootprintTracker] = field(default=None, init=False)
    _semantic_tracker: Optional[SemanticDriftTracker] = field(default=None, init=False)
    _started_at: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self._lexicon_tracker = GhostLexiconTracker()
        self._footprint_tracker = BehavioralFootprintTracker()
        self._semantic_tracker = SemanticDriftTracker()

    # ── recording ─────────────────────────────────────────────────────────────

    def record_step(
        self,
        step_index: int,
        tool_calls: List[str],
        output_text: str,
        is_boundary: bool = False,
    ) -> None:
        """
        Record one agent step. Call after each action/trace entry is processed.

        Parameters
        ----------
        step_index : int
            Sequential step number within the run (0-indexed).
        tool_calls : list[str]
            Tool names called during this step (e.g., ["bash", "python"]).
        output_text : str
            Raw text output or reasoning from this step. Used for lexicon and
            semantic tracking.
        is_boundary : bool
            Set True if this step follows a detected context compaction event.
        """
        record = StepRecord(
            step_index=step_index,
            timestamp=time.time(),
            tool_calls=tool_calls,
            output_text=output_text,
            is_boundary=is_boundary,
        )
        self._steps.append(record)

        # Feed each tracker
        self._lexicon_tracker.record(
            step_index=step_index,
            text=output_text,
            is_anchor=(step_index < self.anchor_steps),
        )
        self._footprint_tracker.record(
            step_index=step_index,
            tool_calls=tool_calls,
        )
        self._semantic_tracker.record(
            step_index=step_index,
            text=output_text,
        )

        if is_boundary:
            self._boundaries.append(step_index)

    def flag_boundary(self, step_index: int) -> None:
        """
        Mark a compaction boundary at the given step (alternative to is_boundary=True
        when you detect compaction after the fact).
        """
        self._boundaries.append(step_index)
        for rec in self._steps:
            if rec.step_index == step_index:
                rec.is_boundary = True

    # ── scoring ───────────────────────────────────────────────────────────────

    def context_consistency_score(self) -> float:
        """
        Compute the Context-Consistency Score (CCS) across the full run so far.

        Returns a float in [0.0, 1.0] where:
          1.0 = no detected behavioral drift
          0.0 = complete behavioral divergence across the three signals

        Returns 1.0 (no evidence of drift) if fewer than anchor_steps + 1 steps
        have been recorded.
        """
        if len(self._steps) < self.anchor_steps + 1:
            return 1.0

        ghost_score = self._lexicon_tracker.consistency_score()
        footprint_score = self._footprint_tracker.consistency_score()
        semantic_score = self._semantic_tracker.consistency_score()

        w = self.ccs_weights
        ccs = (
            w["ghost_lexicon"] * ghost_score
            + w["tool_footprint"] * footprint_score
            + w["semantic_drift"] * semantic_score
        ) / sum(w.values())

        return max(0.0, min(1.0, ccs))

    def boundary_ccs_deltas(self) -> List[Dict[str, Any]]:
        """
        For each recorded boundary, compute the CCS delta: score just before vs.
        score just after the boundary.

        Returns a list of dicts, one per boundary, with:
          boundary_step, ccs_before, ccs_after, delta
        """
        if not self._boundaries:
            return []

        results = []
        for b in self._boundaries:
            before_steps = [s for s in self._steps if s.step_index < b]
            after_steps = [s for s in self._steps if s.step_index >= b]

            # Compute approximate scores for sub-windows
            before_ccs = self._window_ccs(before_steps)
            after_ccs = self._window_ccs(after_steps)

            results.append({
                "boundary_step": b,
                "ccs_before": before_ccs,
                "ccs_after": after_ccs,
                "delta": after_ccs - before_ccs,  # negative = degradation after boundary
            })

        return results

    def _window_ccs(self, steps: List[StepRecord]) -> float:
        """Approximate CCS for a sub-window of steps. Returns 1.0 if too few steps."""
        if len(steps) < 3:
            return 1.0
        tool_sequences = [s.tool_calls for s in steps]
        texts = [s.output_text for s in steps]

        # Use footprint Jaccard as the primary proxy for window CCS
        if len(tool_sequences) < 2:
            return 1.0
        mid = len(tool_sequences) // 2
        first_half = {t for calls in tool_sequences[:mid] for t in calls}
        second_half = {t for calls in tool_sequences[mid:] for t in calls}
        if not first_half and not second_half:
            return 1.0
        union = first_half | second_half
        intersection = first_half & second_half
        return len(intersection) / len(union) if union else 1.0

    # ── serialization ─────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        Return a summary dict with run metadata and all three signal scores.
        Suitable for appending to a sidecar JSON file alongside a Vivaria run.
        """
        ccs = self.context_consistency_score()
        deltas = self.boundary_ccs_deltas()

        return {
            "run_id": self.run_id,
            "tool": "compression-monitor/vivaria_integration",
            "version": "0.1.0",
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_duration_s": round(time.time() - self._started_at, 1),
            "total_steps": len(self._steps),
            "boundaries_detected": len(self._boundaries),
            "boundary_steps": self._boundaries,
            "context_consistency_score": round(ccs, 4),
            "signal_scores": {
                "ghost_lexicon": round(self._lexicon_tracker.consistency_score(), 4),
                "tool_footprint": round(self._footprint_tracker.consistency_score(), 4),
                "semantic_drift": round(self._semantic_tracker.consistency_score(), 4),
            },
            "boundary_ccs_deltas": deltas,
            "drift_detected": ccs < 0.8,
            "drift_severity": (
                "none" if ccs >= 0.9
                else "mild" if ccs >= 0.8
                else "moderate" if ccs >= 0.65
                else "severe"
            ),
        }

    def to_trace_entry(self) -> Dict[str, Any]:
        """
        Return a dict formatted as a Vivaria trace_entry action entry.

        This can be written to a run's trace as a custom entry to make CCS
        visible in the Vivaria UI and queryable via runs_mv.

        Schema follows Vivaria's action entry structure:
          type: "action"
          content: { type: "json", value: <summary dict> }
        """
        return {
            "type": "action",
            "content": {
                "type": "json",
                "value": self.summary(),
            },
        }

    def write_sidecar(self, path: str) -> None:
        """
        Write the CCS summary to a JSON sidecar file alongside the Vivaria run output.

        Parameters
        ----------
        path : str
            Filepath to write. Suggested naming: <run_id>-ccs.json
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"[VivariaBehavioralMonitor] CCS sidecar written to {path}")


# ─── convenience: parse Vivaria trace JSON ────────────────────────────────────

def monitor_from_trace(
    trace: List[Dict[str, Any]],
    run_id: str = "unknown",
    tool_key: str = "tool",
    text_key: str = "content",
    **kwargs: Any,
) -> VivariaBehavioralMonitor:
    """
    Construct and populate a VivariaBehavioralMonitor from a Vivaria trace list
    (as returned by the Vivaria API or exported as JSON).

    Each entry in `trace` should have at least:
      - A tool name at trace[i][tool_key]  (or nested inside "content")
      - Output text at trace[i][text_key]

    Returns a populated monitor with CCS already computable.
    """
    monitor = VivariaBehavioralMonitor(run_id=run_id, **kwargs)

    for i, entry in enumerate(trace):
        # Try to extract tool calls
        tool_call = entry.get(tool_key) or entry.get("action", {}).get("tool", "")
        tool_calls = [tool_call] if tool_call else []

        # Try to extract output text
        content = entry.get(text_key, "")
        if isinstance(content, dict):
            content = content.get("text", "") or content.get("value", "") or str(content)

        monitor.record_step(
            step_index=i,
            tool_calls=tool_calls,
            output_text=str(content),
        )

    return monitor


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vivaria_integration.py <trace.json> [run_id]")
        print("\nExamples:")
        print("  python vivaria_integration.py my_run_trace.json my-run-42")
        print("  python vivaria_integration.py trace.json | python -m json.tool")
        sys.exit(0)

    trace_path = sys.argv[1]
    run_id_arg = sys.argv[2] if len(sys.argv) > 2 else "cli-run"

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    if not isinstance(trace_data, list):
        # Handle {"entries": [...]} wrapper
        trace_data = trace_data.get("entries", trace_data.get("trace", []))

    monitor = monitor_from_trace(trace_data, run_id=run_id_arg)
    print(json.dumps(monitor.summary(), indent=2))
