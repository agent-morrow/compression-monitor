"""
ai_scientist_integration.py
────────────────────────────
Behavioral consistency monitoring for SakanaAI/AI-Scientist-v2 BFTS runs.

AI-Scientist-v2 runs multi-phase agentic tree searches (hypothesis generation →
experiment → analysis → writeup) that can span several hours. Each phase transition
is a context boundary candidate — the agent may cross its context limit before the
next phase begins, causing silent behavioral drift that task-completion and review
metrics won't catch.

This adapter wraps the per-phase journal outputs (Experiment_description, Description,
Key_numerical_results text) and computes a Context-Consistency Score (CCS) across
BFTS phases to surface drift before it reaches the final reviewer.

What it detects
───────────────
1. **Ghost lexicon decay** — domain vocabulary present in the hypothesis/experiment
   phase (method names, dataset names, metric names) disappears from the analysis or
   writeup phase.

2. **Semantic drift** — topic keyword overlap between the idea generation phase and
   final phase declines, indicating the effective working frame narrowed after context
   pressure.

3. **Phase vocabulary shift** — Jaccard distance between term sets in early vs. late
   phases spikes while the scientific content appears internally complete.

CCS < 0.8 means detectable drift — the paper likely has consistency issues that a
per-phase reviewer wouldn't catch individually.

Usage
─────
Drop-in with the existing `manager.journals` update loop in launch_scientist_bfts.py:

    from ai_scientist_integration import AIScientistConsistencyMonitor

    monitor = AIScientistConsistencyMonitor(run_folder=base_folder)

    # After each phase journal is written to manager.journals:
    for stage_name, journal in manager.journals.items():
        best_node = max(journal.nodes, key=lambda n: n.metric or -999, default=None)
        if best_node:
            monitor.record_phase(stage_name=stage_name, node=best_node)

    # Before or after overall_summarize():
    report = monitor.ccs_report()
    if report["drift_detected"]:
        print(f"[CCS] Drift detected across phases: {report['context_consistency_score']:.3f}")
        print(f"[CCS] Most affected signal: {report['weakest_signal']}")

Offline (on an existing run folder)
─────────────────────────────────────
    monitor = AIScientistConsistencyMonitor.from_run_folder("/path/to/run/")
    print(json.dumps(monitor.ccs_report(), indent=2))

Dependencies: compression-monitor (same package)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # avoid hard AI-Scientist import at module level


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
            "or: pip install compression-monitor"
        )


# ─── phase record ─────────────────────────────────────────────────────────────

@dataclass
class PhaseRecord:
    stage_name: str              # e.g. "stage_1", "stage_2_sub_1"
    phase_index: int             # sequential ordering
    text: str                    # concatenated journal text for this phase
    metric: Optional[float]      # best node metric if available
    recorded_at: float = field(default_factory=time.time)


# ─── monitor ──────────────────────────────────────────────────────────────────

@dataclass
class AIScientistConsistencyMonitor:
    """
    Track behavioral consistency across AI-Scientist-v2 BFTS phases.

    Parameters
    ----------
    run_folder : str
        The run's base folder path. Used for sidecar output paths.
    anchor_phases : int
        Number of early phases used to anchor the vocabulary baseline. Default 1
        (first phase = hypothesis/experiment anchors the vocabulary).
    """

    run_folder: str = "."
    anchor_phases: int = 1

    _phases: List[PhaseRecord] = field(default_factory=list, init=False)
    _lexicon: Optional[GhostLexiconTracker] = field(default=None, init=False)
    _footprint: Optional[BehavioralFootprintTracker] = field(default=None, init=False)
    _semantic: Optional[SemanticDriftTracker] = field(default=None, init=False)
    _started_at: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        self._lexicon = GhostLexiconTracker()
        self._footprint = BehavioralFootprintTracker()
        self._semantic = SemanticDriftTracker()

    # ── intake helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _node_to_text(node: Any) -> str:
        """Extract text from an AI-Scientist Journal Node object."""
        parts = []
        if hasattr(node, "plan") and node.plan:
            parts.append(str(node.plan))
        if hasattr(node, "code") and node.code:
            parts.append(str(node.code)[:500])  # cap code length
        if hasattr(node, "analysis") and node.analysis:
            parts.append(str(node.analysis))
        if hasattr(node, "result") and node.result:
            parts.append(str(node.result))
        if hasattr(node, "summary") and node.summary:
            parts.append(str(node.summary))
        return "\n".join(parts) if parts else str(node)

    @staticmethod
    def _summarize_text_to_phase_text(summary_dict: Dict[str, Any]) -> str:
        """Extract text from a log_summarization output dict."""
        parts = []
        for key in ["Experiment_description", "Significance", "Description"]:
            val = summary_dict.get(key, "")
            if val:
                parts.append(str(val))
        for result in summary_dict.get("Key_numerical_results", []):
            desc = result.get("description", "") or result.get("analysis", "")
            if desc:
                parts.append(str(desc))
        return "\n".join(parts)

    # ── recording ──────────────────────────────────────────────────────────────

    def record_phase(
        self,
        stage_name: str,
        node: Any = None,
        text: Optional[str] = None,
        tool_calls: Optional[List[str]] = None,
        metric: Optional[float] = None,
    ) -> None:
        """
        Record a completed BFTS phase (one Journal stage).

        Pass either `node` (a Journal Node object) or `text` (raw string).
        If both are given, text takes precedence.

        Parameters
        ----------
        stage_name : str
            Stage key from manager.journals (e.g. "stage_1", "stage_2_sub_1").
        node : Journal Node, optional
            Best node from the stage's Journal.
        text : str, optional
            Pre-extracted text for the phase. Overrides node extraction.
        tool_calls : list[str], optional
            Tool names used during this phase (for footprint tracking).
        metric : float, optional
            Best metric score from this phase.
        """
        phase_idx = len(self._phases)
        is_anchor = phase_idx < self.anchor_phases

        if text is None and node is not None:
            text = self._node_to_text(node)
            if not text and hasattr(node, "__dict__"):
                text = str(node.__dict__)[:1000]
        text = text or ""

        record = PhaseRecord(
            stage_name=stage_name,
            phase_index=phase_idx,
            text=text,
            metric=metric,
        )
        self._phases.append(record)

        self._lexicon.record(
            step_index=phase_idx,
            text=text,
            is_anchor=is_anchor,
        )
        self._semantic.record(
            step_index=phase_idx,
            text=text,
        )
        if tool_calls is not None:
            self._footprint.record(
                step_index=phase_idx,
                tool_calls=tool_calls,
            )

    def record_summary_dict(
        self,
        stage_name: str,
        summary_dict: Dict[str, Any],
        tool_calls: Optional[List[str]] = None,
        metric: Optional[float] = None,
    ) -> None:
        """
        Record from a log_summarization output dict (overall_summarize return value).
        Convenience wrapper for the post-run analysis path.
        """
        text = self._summarize_text_to_phase_text(summary_dict)
        self.record_phase(stage_name=stage_name, text=text,
                          tool_calls=tool_calls, metric=metric)

    # ── scoring ────────────────────────────────────────────────────────────────

    def context_consistency_score(self) -> float:
        """
        Compute the CCS across all recorded phases.
        Returns 1.0 (no evidence of drift) if fewer than 2 phases recorded.
        """
        if len(self._phases) < 2:
            return 1.0

        ghost = self._lexicon.consistency_score()
        semantic = self._semantic.consistency_score()

        # Footprint only meaningful if tool_calls were provided
        has_footprint = any(
            getattr(self._footprint, "_step_tools", {})
        ) if hasattr(self._footprint, "_step_tools") else False

        if has_footprint:
            footprint = self._footprint.consistency_score()
            ccs = (ghost * 0.4 + semantic * 0.4 + footprint * 0.2)
        else:
            ccs = (ghost * 0.5 + semantic * 0.5)

        return max(0.0, min(1.0, ccs))

    def ccs_report(self) -> Dict[str, Any]:
        """
        Return a full CCS report for this run's phases.
        Suitable for writing to a sidecar JSON or feeding into the reviewer model.
        """
        ccs = self.context_consistency_score()
        ghost_score = self._lexicon.consistency_score() if len(self._phases) >= 2 else 1.0
        semantic_score = self._semantic.consistency_score() if len(self._phases) >= 2 else 1.0

        scores = {"ghost_lexicon": round(ghost_score, 4), "semantic_drift": round(semantic_score, 4)}
        weakest = min(scores, key=scores.get)

        return {
            "tool": "compression-monitor/ai_scientist_integration",
            "version": "0.1.0",
            "run_folder": self.run_folder,
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "phases_analyzed": len(self._phases),
            "phase_names": [p.stage_name for p in self._phases],
            "context_consistency_score": round(ccs, 4),
            "signal_scores": scores,
            "weakest_signal": weakest,
            "drift_detected": ccs < 0.8,
            "drift_severity": (
                "none" if ccs >= 0.9
                else "mild" if ccs >= 0.8
                else "moderate" if ccs >= 0.65
                else "severe"
            ),
            "per_phase": [
                {
                    "stage_name": p.stage_name,
                    "phase_index": p.phase_index,
                    "metric": p.metric,
                    "text_length": len(p.text),
                }
                for p in self._phases
            ],
            "interpretation": (
                "Phases are behaviorally consistent — vocabulary and topic frame stable."
                if ccs >= 0.9 else
                "Mild drift detected — vocabulary or topic frame may have narrowed."
                if ccs >= 0.8 else
                "Moderate drift — recommend reviewing phase-to-phase coherence in the writeup."
                if ccs >= 0.65 else
                "Severe drift — strong evidence of context pressure between phases. "
                "Writeup may not reflect early experimental framing."
            ),
        }

    def write_sidecar(self, filename: Optional[str] = None) -> str:
        """
        Write the CCS report to a JSON sidecar file in the run folder.
        Returns the path written.
        """
        if filename is None:
            run_name = os.path.basename(self.run_folder.rstrip("/"))
            filename = f"{run_name}-ccs.json"
        path = os.path.join(self.run_folder, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.ccs_report(), f, indent=2)
        print(f"[AIScientistConsistencyMonitor] CCS sidecar written to {path}")
        return path

    # ── offline: load from run folder ─────────────────────────────────────────

    @classmethod
    def from_run_folder(cls, run_folder: str) -> "AIScientistConsistencyMonitor":
        """
        Build a populated monitor from an existing AI-Scientist-v2 run folder.

        Looks for:
          - <run_folder>/journals/*.json  (if present)
          - <run_folder>/final_info.json  (if present)
          - <run_folder>/experiment.log   (fallback text)

        Returns a monitor with CCS already computable.
        """
        monitor = cls(run_folder=run_folder)

        # Try journals directory
        journals_dir = os.path.join(run_folder, "journals")
        if os.path.isdir(journals_dir):
            journal_files = sorted(
                f for f in os.listdir(journals_dir) if f.endswith(".json")
            )
            for i, jf in enumerate(journal_files):
                path = os.path.join(journals_dir, jf)
                try:
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    stage = os.path.splitext(jf)[0]
                    if isinstance(data, dict):
                        text = monitor._summarize_text_to_phase_text(data)
                        monitor.record_phase(stage_name=stage, text=text)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                text = monitor._summarize_text_to_phase_text(item)
                                monitor.record_phase(stage_name=stage, text=text)
                                break
                except (json.JSONDecodeError, OSError):
                    pass
            if monitor._phases:
                return monitor

        # Fallback: final_info.json
        final_info_path = os.path.join(run_folder, "final_info.json")
        if os.path.isfile(final_info_path):
            try:
                with open(final_info_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    # Treat each top-level key as a phase
                    for k, v in data.items():
                        text = json.dumps(v) if not isinstance(v, str) else v
                        monitor.record_phase(stage_name=k, text=text)
            except (json.JSONDecodeError, OSError):
                pass
            if monitor._phases:
                return monitor

        # Fallback: experiment.log
        log_path = os.path.join(run_folder, "experiment.log")
        if os.path.isfile(log_path):
            try:
                with open(log_path, encoding="utf-8") as fh:
                    log_text = fh.read()
                # Split into rough halves as two phases
                mid = len(log_text) // 2
                monitor.record_phase("early_log", text=log_text[:mid])
                monitor.record_phase("late_log", text=log_text[mid:])
            except OSError:
                pass

        return monitor


# ─── reviewer model input adapter ─────────────────────────────────────────────

def ccs_as_reviewer_context(monitor: AIScientistConsistencyMonitor) -> str:
    """
    Format the CCS report as a context block for the reviewer LLM.

    Usage in perform_review.py:
        from ai_scientist_integration import ccs_as_reviewer_context
        ccs_context = ccs_as_reviewer_context(monitor)
        # Prepend to the paper text or add as a system context note
    """
    report = monitor.ccs_report()
    ccs = report["context_consistency_score"]
    severity = report["drift_severity"]

    lines = [
        "=== Behavioral Consistency Report (pre-review) ===",
        f"Context-Consistency Score: {ccs:.3f}  ({severity})",
        f"Phases analyzed: {report['phases_analyzed']} ({', '.join(report['phase_names'])})",
        f"Ghost lexicon score: {report['signal_scores'].get('ghost_lexicon', 'N/A')}",
        f"Semantic drift score: {report['signal_scores'].get('semantic_drift', 'N/A')}",
    ]
    if report["drift_detected"]:
        lines.append(
            f"NOTE: Drift detected (weakest signal: {report['weakest_signal']}). "
            f"{report['interpretation']}"
        )
    else:
        lines.append(f"NOTE: {report['interpretation']}")
    lines.append("=" * 50)
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ai_scientist_integration.py <run_folder>")
        print("\nAnalyzes an existing AI-Scientist-v2 run folder for behavioral drift.")
        sys.exit(0)

    run_folder = sys.argv[1]
    monitor = AIScientistConsistencyMonitor.from_run_folder(run_folder)

    report = monitor.ccs_report()
    print(json.dumps(report, indent=2))

    if report["drift_detected"]:
        print(f"\n⚠ CCS {report['context_consistency_score']:.3f}: {report['interpretation']}")
    else:
        print(f"\n✓ CCS {report['context_consistency_score']:.3f}: {report['interpretation']}")
