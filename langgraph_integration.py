"""
langgraph_integration.py — compression-monitor adapter for LangGraph

Monitors behavioral drift across checkpoints in LangGraph stateful graphs.

Usage:
    from langgraph_integration import GraphDriftMonitor

    monitor = GraphDriftMonitor(graph, monitor_dir="./drift_logs")
    result = monitor.invoke({"messages": [...]})
    report = monitor.drift_report()

    # Or use as a context manager around get_state_history():
    monitor.snapshot_from_state_history(graph, config, lookback=5)

Requirements:
    pip install langgraph
"""

import json
import re
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional, Generator
from collections import Counter


# ---------------------------------------------------------------------------
# Shared measurement primitives
# ---------------------------------------------------------------------------

def _extract_lexicon(text: str, min_len: int = 5) -> Counter:
    words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_len, text.lower())
    return Counter(words)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def _ghost_lexicon_score(prior: str, current: str, top_n: int = 50) -> float:
    prior_top = [w for w, _ in _extract_lexicon(prior).most_common(top_n)]
    if not prior_top:
        return 1.0
    current_words = set(_extract_lexicon(current).keys())
    return sum(1 for w in prior_top if w in current_words) / len(prior_top)


def _semantic_overlap(text_a: str, text_b: str, top_n: int = 30) -> float:
    top_a = set(w for w, _ in _extract_lexicon(text_a).most_common(top_n))
    top_b = set(w for w, _ in _extract_lexicon(text_b).most_common(top_n))
    return _jaccard(top_a, top_b)


# ---------------------------------------------------------------------------
# Checkpoint snapshot — captures a behavioral fingerprint from LangGraph state
# ---------------------------------------------------------------------------

class CheckpointSnapshot:
    """Extracts a behavioral fingerprint from a LangGraph state dict."""

    def __init__(self, state: dict, checkpoint_id: Optional[str] = None):
        self.checkpoint_id = checkpoint_id or f"snap_{int(time.time()*1000)}"
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.state = state

        # Extract text content from common LangGraph state shapes
        self.text_content = self._extract_text(state)
        self.tool_calls = self._extract_tool_calls(state)
        self.lexicon = _extract_lexicon(self.text_content)

    def _extract_text(self, state: dict) -> str:
        """Pull text from messages, output, or any string values in state."""
        parts = []
        messages = state.get("messages", [])
        for msg in messages:
            if hasattr(msg, "content"):
                parts.append(str(msg.content))
            elif isinstance(msg, dict):
                parts.append(str(msg.get("content", "")))

        # Also capture any string values at top level
        for k, v in state.items():
            if k != "messages" and isinstance(v, str) and len(v) > 20:
                parts.append(v)

        return "\n".join(parts)

    def _extract_tool_calls(self, state: dict) -> list[str]:
        """Extract tool call names from messages."""
        tools = []
        for msg in state.get("messages", []):
            calls = []
            if hasattr(msg, "tool_calls"):
                calls = msg.tool_calls or []
            elif isinstance(msg, dict):
                calls = msg.get("tool_calls", [])
            for tc in calls:
                if isinstance(tc, dict):
                    tools.append(tc.get("name", "unknown"))
                elif hasattr(tc, "name"):
                    tools.append(tc.name)
        return tools

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "text_length": len(self.text_content),
            "tool_calls": self.tool_calls,
            "top_lexicon": dict(self.lexicon.most_common(20)),
            "content_hash": hashlib.sha256(self.text_content.encode()).hexdigest()[:16],
        }


# ---------------------------------------------------------------------------
# Drift measurement between two checkpoints
# ---------------------------------------------------------------------------

class CheckpointDrift:
    def __init__(self, before: CheckpointSnapshot, after: CheckpointSnapshot):
        self.before = before
        self.after = after

        self.ghost_score = _ghost_lexicon_score(before.text_content, after.text_content)
        self.semantic_score = _semantic_overlap(before.text_content, after.text_content)
        self.tool_jaccard = _jaccard(set(before.tool_calls), set(after.tool_calls))

        self.drift_score = 1.0 - (
            self.ghost_score * 0.4 +
            self.semantic_score * 0.3 +
            self.tool_jaccard * 0.3
        )

    @property
    def alert(self) -> Optional[str]:
        issues = []
        if self.ghost_score < 0.5:
            issues.append(f"ghost_lexicon={self.ghost_score:.2f}")
        if self.semantic_score < 0.4:
            issues.append(f"semantic={self.semantic_score:.2f}")
        if self.tool_jaccard < 0.3 and (self.before.tool_calls or self.after.tool_calls):
            issues.append(f"tool_jaccard={self.tool_jaccard:.2f}")
        if issues:
            return "DRIFT ALERT [checkpoint boundary]: " + "; ".join(issues)
        return None

    def to_dict(self) -> dict:
        return {
            "before_checkpoint": self.before.checkpoint_id,
            "after_checkpoint": self.after.checkpoint_id,
            "drift_score": round(self.drift_score, 4),
            "ghost_lexicon_survival": round(self.ghost_score, 4),
            "semantic_overlap": round(self.semantic_score, 4),
            "tool_jaccard": round(self.tool_jaccard, 4),
            "alert": self.alert,
            "before_timestamp": self.before.timestamp,
            "after_timestamp": self.after.timestamp,
        }


# ---------------------------------------------------------------------------
# GraphDriftMonitor — wraps a compiled LangGraph graph
# ---------------------------------------------------------------------------

class GraphDriftMonitor:
    """
    Wraps a compiled LangGraph graph to track behavioral drift across invocations.

    Parameters
    ----------
    graph : CompiledGraph
        A compiled LangGraph graph (output of graph.compile()).
    monitor_dir : str or Path
        Directory for drift log files (JSONL).
    drift_threshold : float
        Composite drift score above which warnings print (0–1, default 0.3).
    """

    def __init__(self, graph, monitor_dir: str = "./drift_logs",
                 drift_threshold: float = 0.3):
        self._graph = graph
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

        self._prior_snapshot: Optional[CheckpointSnapshot] = None
        self._measurements: list[CheckpointDrift] = []
        self._invoke_count = 0

    def invoke(self, input_state: dict, config: Optional[dict] = None, **kwargs) -> dict:
        """Invoke the graph and measure drift relative to the previous invocation."""
        self._invoke_count += 1
        invoke_id = f"invoke_{self._invoke_count:04d}_{int(time.time())}"

        result = self._graph.invoke(input_state, config=config, **kwargs) if config else \
                 self._graph.invoke(input_state, **kwargs)

        snap = CheckpointSnapshot(result, checkpoint_id=invoke_id)
        self._measure_and_log(snap, invoke_id)
        return result

    def stream(self, input_state: dict, config: Optional[dict] = None, **kwargs) -> Generator:
        """Stream graph execution and snapshot the final state."""
        self._invoke_count += 1
        invoke_id = f"stream_{self._invoke_count:04d}_{int(time.time())}"
        final_state = input_state.copy()

        for chunk in (self._graph.stream(input_state, config=config, **kwargs) if config
                      else self._graph.stream(input_state, **kwargs)):
            if isinstance(chunk, dict):
                final_state.update(chunk)
            yield chunk

        snap = CheckpointSnapshot(final_state, checkpoint_id=invoke_id)
        self._measure_and_log(snap, invoke_id)

    def snapshot_from_state_history(self, graph, config: dict,
                                     lookback: int = 10) -> list[CheckpointDrift]:
        """
        Measure drift across recent checkpoints using get_state_history().
        Useful for post-hoc analysis of an existing checkpointed graph.
        """
        measurements = []
        history = list(graph.get_state_history(config))[:lookback]
        history.reverse()  # oldest first

        prior = None
        for i, state_snapshot in enumerate(history):
            state_values = state_snapshot.values if hasattr(state_snapshot, 'values') else state_snapshot
            snap = CheckpointSnapshot(
                state_values,
                checkpoint_id=f"history_{i:04d}"
            )
            if prior is not None:
                m = CheckpointDrift(prior, snap)
                measurements.append(m)
                self._measurements.append(m)
                if m.alert:
                    print(f"\n[WARN] {m.alert}")
                log_path = self.monitor_dir / f"history_analysis_{int(time.time())}.jsonl"
                with open(log_path, 'a') as f:
                    f.write(json.dumps(m.to_dict()) + "\n")
            prior = snap

        return measurements

    def _measure_and_log(self, snap: CheckpointSnapshot, invoke_id: str):
        if self._prior_snapshot is not None:
            m = CheckpointDrift(self._prior_snapshot, snap)
            self._measurements.append(m)

            if m.alert:
                print(f"\n[WARN] {m.alert}")
            elif m.drift_score > self.drift_threshold:
                print(f"\n[WARN] DRIFT WARNING: score={m.drift_score:.3f} (threshold={self.drift_threshold})")

            log_path = self.monitor_dir / f"{invoke_id}_drift.jsonl"
            with open(log_path, 'w') as f:
                f.write(json.dumps(m.to_dict()) + "\n")

        snap_path = self.monitor_dir / f"{invoke_id}_snapshot.jsonl"
        with open(snap_path, 'w') as f:
            f.write(json.dumps(snap.to_dict()) + "\n")

        self._prior_snapshot = snap

    def drift_report(self) -> dict:
        if not self._measurements:
            return {
                "invocations": self._invoke_count,
                "measurements": 0,
                "note": "Need at least 2 invocations to measure drift."
            }
        alerts = [m.alert for m in self._measurements if m.alert]
        avg_drift = sum(m.drift_score for m in self._measurements) / len(self._measurements)
        worst = max(self._measurements, key=lambda m: m.drift_score)
        return {
            "invocations": self._invoke_count,
            "measurements": len(self._measurements),
            "avg_drift_score": round(avg_drift, 4),
            "worst_drift_score": round(worst.drift_score, 4),
            "alerts": alerts,
            "log_dir": str(self.monitor_dir),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("compression-monitor / langgraph_integration.py")
    print("GraphDriftMonitor wrapper for LangGraph stateful graphs.")
    print()

    # Smoke test without actual langgraph dependency
    state_a = {
        "messages": [
            type("M", (), {"content": "the agent deployed kubernetes cluster and configured the database connection pool", "tool_calls": []})()
        ]
    }
    state_b = {
        "messages": [
            type("M", (), {"content": "the agent completed the task successfully and returned results to the user", "tool_calls": []})()
        ]
    }
    state_c = {
        "messages": [
            type("M", (), {"content": "the agent redeployed kubernetes with updated database pool configuration settings", "tool_calls": []})()
        ]
    }

    s_a = CheckpointSnapshot(state_a, "snap_A")
    s_b = CheckpointSnapshot(state_b, "snap_B")
    s_c = CheckpointSnapshot(state_c, "snap_C")

    drift_ab = CheckpointDrift(s_a, s_b)
    drift_ac = CheckpointDrift(s_a, s_c)

    print(f"  Drift (topic shift):  score={drift_ab.drift_score:.3f}  (expect high ~0.7)")
    print(f"  Drift (same topic):   score={drift_ac.drift_score:.3f}  (expect low ~0.2)")
    print(f"  Alert (shift):  {drift_ab.alert or 'none'}")
    print(f"  Alert (same):   {drift_ac.alert or 'none'}")
    print("  OK")
