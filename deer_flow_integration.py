"""
DeerFlow behavioral consistency monitor for long-horizon project sessions.

Monitors behavioral drift at session resume points in bytedance/deer-flow
multi-day project workflows. A behavioral consistency check at resume_project()
can surface constraint loss and framing shifts before the agent acts.

Usage:
    monitor = DeerFlowSessionMonitor()

    # At session end (end of Day 1)
    monitor.checkpoint_session("project-id", session_outputs)

    # At session resume (start of Day 2)
    report = monitor.check_resume_consistency("project-id", new_session_outputs)
    if report["drift_score"] > 0.4:
        print("Warning: behavioral shift detected since last session")
        print("Ghost terms:", report["ghost_terms"])
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer (no external deps)."""
    import re
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text.lower())


def _fingerprint(outputs: list[str]) -> dict[str, Any]:
    """
    Compute a behavioral fingerprint from a list of agent output strings.
    Returns term frequencies, low-frequency (precise) vocab, and a compact vector.
    """
    all_tokens: list[str] = []
    for out in outputs:
        all_tokens.extend(_tokenize(out))
    if not all_tokens:
        return {"term_freq": {}, "low_freq_vocab": set(), "total_tokens": 0}
    freq = Counter(all_tokens)
    total = len(all_tokens)
    # Low-frequency vocab = terms appearing 1-3 times; these are often precise constraints
    low_freq_vocab = {t for t, c in freq.items() if 1 <= c <= 3}
    return {
        "term_freq": dict(freq.most_common(100)),
        "low_freq_vocab": low_freq_vocab,
        "total_tokens": total,
    }


def _ghost_terms(baseline_fp: dict, current_fp: dict) -> list[str]:
    """
    Find terms that appeared in the baseline but are absent from current output.
    These may indicate constraints or framings that were lost at the session boundary.
    """
    baseline_terms = set(baseline_fp.get("term_freq", {}).keys())
    current_terms = set(current_fp.get("term_freq", {}).keys())
    # Focus on terms that were in baseline low-freq vocab (precise/specific)
    baseline_precise = baseline_fp.get("low_freq_vocab", set())
    return sorted(baseline_precise & baseline_terms - current_terms)


def _drift_score(baseline_fp: dict, current_fp: dict) -> float:
    """
    Compute a [0, 1] drift score between two fingerprints.
    0 = identical behavioral profile. 1 = completely different.
    """
    baseline_terms = set(baseline_fp.get("term_freq", {}).keys())
    current_terms = set(current_fp.get("term_freq", {}).keys())
    if not baseline_terms and not current_terms:
        return 0.0
    union = baseline_terms | current_terms
    intersection = baseline_terms & current_terms
    jaccard = len(intersection) / len(union) if union else 1.0
    return round(1.0 - jaccard, 4)


class DeerFlowSessionMonitor:
    """
    Monitors behavioral consistency at session resume points in DeerFlow projects.

    Stores behavioral fingerprints at session checkpoints and compares them
    when a project session is resumed. High drift scores indicate that the
    agent's behavioral patterns shifted — potentially due to context compaction
    or summarization loss across the session boundary.
    """

    def __init__(self, checkpoint_dir: str = ".deer_flow_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def checkpoint_session(
        self,
        project_id: str,
        session_outputs: list[str],
        metadata: dict | None = None,
    ) -> dict:
        """
        Save a behavioral fingerprint for the current session.
        Call this at the end of each project session (before context is lost).

        Args:
            project_id: Unique identifier for the project
            session_outputs: List of agent output strings from this session
            metadata: Optional metadata (session length, tool calls used, etc.)

        Returns:
            Fingerprint record saved to disk
        """
        fp = _fingerprint(session_outputs)
        record = {
            "project_id": project_id,
            "checkpointed_at": time.time(),
            "fingerprint": {
                "term_freq": fp["term_freq"],
                "low_freq_vocab": sorted(fp["low_freq_vocab"]),
                "total_tokens": fp["total_tokens"],
            },
            "metadata": metadata or {},
        }
        checkpoint_file = self.checkpoint_dir / f"{project_id}_checkpoint.json"
        checkpoint_file.write_text(json.dumps(record, indent=2))
        return record

    def check_resume_consistency(
        self,
        project_id: str,
        new_session_outputs: list[str],
        drift_threshold: float = 0.35,
    ) -> dict:
        """
        Compare current session behavior against the last checkpoint.
        Call this at the start of a resumed project session.

        Args:
            project_id: Unique identifier for the project
            new_session_outputs: Initial outputs from the resumed session
            drift_threshold: Drift score above which a warning is raised

        Returns:
            Consistency report with drift_score, ghost_terms, and recommendation
        """
        checkpoint_file = self.checkpoint_dir / f"{project_id}_checkpoint.json"
        if not checkpoint_file.exists():
            return {
                "project_id": project_id,
                "status": "no_baseline",
                "message": "No prior checkpoint found. Run checkpoint_session() at end of first session.",
            }

        record = json.loads(checkpoint_file.read_text())
        baseline_fp = record["fingerprint"]
        # Reconstruct low_freq_vocab as set
        baseline_fp["low_freq_vocab"] = set(baseline_fp.get("low_freq_vocab", []))

        current_fp = _fingerprint(new_session_outputs)
        score = _drift_score(baseline_fp, current_fp)
        ghosts = _ghost_terms(baseline_fp, current_fp)

        report = {
            "project_id": project_id,
            "drift_score": score,
            "ghost_terms": ghosts[:20],  # top 20 vanished precise terms
            "baseline_tokens": baseline_fp["total_tokens"],
            "current_tokens": current_fp["total_tokens"],
            "checkpointed_at": record["checkpointed_at"],
            "checked_at": time.time(),
        }

        if score > drift_threshold:
            report["status"] = "warning"
            report["recommendation"] = (
                f"Behavioral drift score {score:.2f} exceeds threshold {drift_threshold}. "
                f"Ghost terms suggest these constraints/framings may have been lost: "
                f"{ghosts[:5]}. Consider re-anchoring session context before proceeding."
            )
        else:
            report["status"] = "consistent"
            report["recommendation"] = (
                f"Drift score {score:.2f} within threshold. "
                "Behavioral patterns are consistent with prior session."
            )

        return report

    def summarize_project_drift(self, project_id: str) -> str:
        """
        Return a human-readable drift summary for a project session.
        """
        checkpoint_file = self.checkpoint_dir / f"{project_id}_checkpoint.json"
        if not checkpoint_file.exists():
            return f"No checkpoint found for project '{project_id}'."
        record = json.loads(checkpoint_file.read_text())
        import datetime
        ts = datetime.datetime.fromtimestamp(record["checkpointed_at"]).strftime("%Y-%m-%d %H:%M")
        return (
            f"Project '{project_id}' | Last checkpoint: {ts} | "
            f"Tracked terms: {len(record['fingerprint'].get('term_freq', {}))}"
        )


# ---------------------------------------------------------------------------
# LangGraph integration helper
# Attach to DeerFlow's graph state transitions for automatic monitoring
# ---------------------------------------------------------------------------

class DeerFlowGraphMonitor:
    """
    LangGraph state hook for DeerFlow that automatically checkpoints
    behavioral fingerprints at graph completion and checks consistency at resume.

    Example:
        graph_monitor = DeerFlowGraphMonitor(project_id="my-research-project")

        # Wrap DeerFlow's graph invocation
        result = graph_monitor.run_with_monitoring(graph.invoke, state)
    """

    def __init__(self, project_id: str, checkpoint_dir: str = ".deer_flow_checkpoints"):
        self.project_id = project_id
        self.session_monitor = DeerFlowSessionMonitor(checkpoint_dir)
        self._session_outputs: list[str] = []

    def collect_output(self, output: str) -> None:
        """Call with each agent output string to accumulate the session record."""
        self._session_outputs.append(output)

    def run_with_monitoring(self, graph_fn, state, **kwargs):
        """
        Run a LangGraph function, check resume consistency first, and
        checkpoint at completion.
        """
        # Check consistency at session start if prior checkpoint exists
        if self._session_outputs:
            resume_report = self.session_monitor.check_resume_consistency(
                self.project_id, self._session_outputs[:10]
            )
            if resume_report.get("status") == "warning":
                print(f"[DeerFlowMonitor] ⚠️  {resume_report['recommendation']}")

        # Run the graph
        result = graph_fn(state, **kwargs)

        # Checkpoint at completion
        self.session_monitor.checkpoint_session(
            self.project_id,
            self._session_outputs,
            metadata={"state_keys": list(state.keys()) if isinstance(state, dict) else []},
        )
        return result
