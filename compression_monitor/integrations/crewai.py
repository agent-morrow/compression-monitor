"""
crewai_integration.py — compression-monitor adapter for CrewAI

Wraps Crew.kickoff() to automatically measure behavioral drift across session boundaries.

Usage:
    from compression_monitor.integrations.crewai import MonitoredCrew

    crew = MonitoredCrew(
        agents=[...],
        tasks=[...],
        monitor_dir="./drift_logs"
    )
    result = crew.kickoff()  # automatically measures and logs drift
    report = crew.drift_report()

Requirements:
    pip install crewai
    (compression-monitor scripts must be in the same directory or on PYTHONPATH)
"""

import json
import time
import hashlib
import importlib.util
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional
from collections import Counter
import re


# ---------------------------------------------------------------------------
# Minimal inline implementations (no import from sibling scripts required)
# ---------------------------------------------------------------------------

def _extract_lexicon(text: str, min_len: int = 5) -> Counter:
    """Extract word frequency counter from text."""
    words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_len, text.lower())
    return Counter(words)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _semantic_overlap(text_a: str, text_b: str, top_n: int = 30) -> float:
    c_a = _extract_lexicon(text_a)
    c_b = _extract_lexicon(text_b)
    top_a = set(w for w, _ in c_a.most_common(top_n))
    top_b = set(w for w, _ in c_b.most_common(top_n))
    return _jaccard(top_a, top_b)


def _ghost_lexicon_score(prior_text: str, current_text: str, top_n: int = 50) -> float:
    """Fraction of prior top-N vocabulary that survives in current output."""
    prior = _extract_lexicon(prior_text)
    current_words = set(_extract_lexicon(current_text).keys())
    top_prior = [w for w, _ in prior.most_common(top_n)]
    if not top_prior:
        return 1.0
    survived = sum(1 for w in top_prior if w in current_words)
    return survived / len(top_prior)


# ---------------------------------------------------------------------------
# Session state snapshot
# ---------------------------------------------------------------------------

class AgentSnapshot:
    """Captures a behavioral fingerprint for one agent at one point in time."""

    def __init__(self, agent_name: str, output_text: str, tool_calls: list[str]):
        self.agent_name = agent_name
        self.output_text = output_text
        self.tool_calls = tool_calls
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.lexicon = _extract_lexicon(output_text)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "output_length": len(self.output_text),
            "top_lexicon": dict(self.lexicon.most_common(20)),
            "tool_calls": self.tool_calls,
            "output_hash": hashlib.sha256(self.output_text.encode()).hexdigest()[:16],
        }


# ---------------------------------------------------------------------------
# Drift measurement between two snapshots
# ---------------------------------------------------------------------------

class DriftMeasurement:
    def __init__(self, before: AgentSnapshot, after: AgentSnapshot):
        self.agent_name = before.agent_name
        self.before = before
        self.after = after

        self.ghost_score = _ghost_lexicon_score(before.output_text, after.output_text)
        self.semantic_score = _semantic_overlap(before.output_text, after.output_text)
        self.tool_jaccard = _jaccard(set(before.tool_calls), set(after.tool_calls))

        # Composite drift score (0=identical, 1=completely different)
        self.drift_score = 1.0 - (
            self.ghost_score * 0.4 +
            self.semantic_score * 0.3 +
            self.tool_jaccard * 0.3
        )

    @property
    def alert(self) -> Optional[str]:
        issues = []
        if self.ghost_score < 0.5:
            issues.append(f"ghost_lexicon={self.ghost_score:.2f} (>50% vocabulary loss)")
        if self.semantic_score < 0.4:
            issues.append(f"semantic_overlap={self.semantic_score:.2f} (sharp topic shift)")
        if self.tool_jaccard < 0.3:
            issues.append(f"tool_jaccard={self.tool_jaccard:.2f} (tool pattern changed)")
        if issues:
            return f"DRIFT ALERT [{self.agent_name}]: " + "; ".join(issues)
        return None

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "drift_score": round(self.drift_score, 4),
            "ghost_lexicon_survival": round(self.ghost_score, 4),
            "semantic_overlap": round(self.semantic_score, 4),
            "tool_jaccard": round(self.tool_jaccard, 4),
            "alert": self.alert,
            "before_timestamp": self.before.timestamp,
            "after_timestamp": self.after.timestamp,
        }


# ---------------------------------------------------------------------------
# MonitoredCrew wrapper
# ---------------------------------------------------------------------------

class MonitoredCrew:
    """
    Drop-in replacement for crewai.Crew that monitors behavioral drift
    across kickoff() calls.

    Parameters
    ----------
    monitor_dir : str or Path
        Directory where drift logs are written (JSONL format).
    drift_threshold : float
        Composite drift score above which a warning is printed (0–1, default 0.3).
    *args, **kwargs
        Passed directly to crewai.Crew.
    """

    def __init__(self, *args, monitor_dir: str = "./drift_logs",
                 drift_threshold: float = 0.3, **kwargs):
        try:
            from crewai import Crew
            self._crew = Crew(*args, **kwargs)
        except ImportError:
            raise ImportError("crewai is required: pip install crewai")

        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

        self._prior_snapshots: dict[str, AgentSnapshot] = {}
        self._measurements: list[DriftMeasurement] = []
        self._kickoff_count = 0

    def kickoff(self, inputs: Optional[dict] = None) -> Any:
        """Run the crew and measure drift relative to the previous kickoff."""
        self._kickoff_count += 1
        kickoff_id = f"kickoff_{self._kickoff_count:04d}_{int(time.time())}"

        result = self._crew.kickoff(inputs=inputs) if inputs else self._crew.kickoff()

        # Collect agent outputs from task results
        self._snapshot_and_measure(result, kickoff_id)

        return result

    def _snapshot_and_measure(self, result: Any, kickoff_id: str):
        """Extract outputs, snapshot, measure drift, log."""
        # Extract text outputs per agent from tasks
        agent_outputs: dict[str, list[str]] = {}
        agent_tools: dict[str, list[str]] = {}

        for task in self._crew.tasks:
            agent = task.agent
            if agent is None:
                continue
            agent_key = getattr(agent, 'role', str(agent))
            output = ""
            if hasattr(task, 'output') and task.output:
                output = str(task.output.raw) if hasattr(task.output, 'raw') else str(task.output)
            agent_outputs.setdefault(agent_key, []).append(output)

            # Collect tool names if available
            tools = [getattr(t, 'name', str(t)) for t in getattr(agent, 'tools', [])]
            agent_tools[agent_key] = tools

        log_path = self.monitor_dir / f"{kickoff_id}.jsonl"
        new_measurements = []

        for agent_key, outputs in agent_outputs.items():
            combined_output = "\n".join(outputs)
            snapshot = AgentSnapshot(agent_key, combined_output, agent_tools.get(agent_key, []))

            if agent_key in self._prior_snapshots:
                m = DriftMeasurement(self._prior_snapshots[agent_key], snapshot)
                self._measurements.append(m)
                new_measurements.append(m)

                if m.alert:
                    print(f"\n⚠  {m.alert}")
                elif m.drift_score > self.drift_threshold:
                    print(f"\n⚠  DRIFT WARNING [{agent_key}]: score={m.drift_score:.3f} (threshold={self.drift_threshold})")

                with open(log_path, 'a') as f:
                    f.write(json.dumps(m.to_dict()) + "\n")

            self._prior_snapshots[agent_key] = snapshot

        # Also write snapshots
        snap_path = self.monitor_dir / f"{kickoff_id}_snapshots.jsonl"
        for agent_key, outputs in agent_outputs.items():
            snap = AgentSnapshot(agent_key, "\n".join(outputs), agent_tools.get(agent_key, []))
            with open(snap_path, 'a') as f:
                f.write(json.dumps(snap.to_dict()) + "\n")

    def drift_report(self) -> dict:
        """Return a summary of all drift measurements since instantiation."""
        if not self._measurements:
            return {"kickoffs": self._kickoff_count, "measurements": 0,
                    "note": "No prior baseline yet — need at least 2 kickoffs to measure drift."}

        alerts = [m.alert for m in self._measurements if m.alert]
        avg_drift = sum(m.drift_score for m in self._measurements) / len(self._measurements)
        worst = max(self._measurements, key=lambda m: m.drift_score)

        return {
            "kickoffs": self._kickoff_count,
            "measurements": len(self._measurements),
            "avg_drift_score": round(avg_drift, 4),
            "alerts": alerts,
            "worst_agent": worst.agent_name,
            "worst_drift_score": round(worst.drift_score, 4),
            "log_dir": str(self.monitor_dir),
        }

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying Crew."""
        return getattr(self._crew, name)


# ---------------------------------------------------------------------------
# CLI demo (no crewai required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("compression-monitor / crewai_integration.py")
    print("Drop-in MonitoredCrew wrapper for behavioral drift measurement.")
    print()
    print("Usage:")
    print("  from compression_monitor.integrations.crewai import MonitoredCrew")
    print("  crew = MonitoredCrew(agents=[...], tasks=[...], monitor_dir='./logs')")
    print("  result = crew.kickoff()")
    print("  print(crew.drift_report())")
    print()
    print("Testing inline drift math...")
    # Quick smoke test
    text_a = "the agent deployed the kubernetes cluster and configured the database connection pool"
    text_b = "the agent completed the task successfully and returned the result to the user"
    text_similar = "the agent deployed the kubernetes cluster with updated database settings"

    print(f"  ghost_lexicon (same domain): {_ghost_lexicon_score(text_a, text_similar):.3f}  (expect high)")
    print(f"  ghost_lexicon (different):   {_ghost_lexicon_score(text_a, text_b):.3f}  (expect low)")
    print(f"  semantic_overlap (same):     {_semantic_overlap(text_a, text_similar):.3f}  (expect high)")
    print(f"  semantic_overlap (diff):     {_semantic_overlap(text_a, text_b):.3f}  (expect low)")
    print("  OK")
