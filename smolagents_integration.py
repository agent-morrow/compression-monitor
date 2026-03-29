"""
smolagents_integration.py
─────────────────────────
Behavioral fingerprinting for smolagents MultiStepAgent memory consolidation events.

This adapter detects context compression boundaries in smolagents by monitoring the
agent's message history length across steps. When history shrinks (consolidation) or
crosses a configurable token-budget threshold, it snapshots and compares the behavioral
fingerprint — catching silent behavioral drift that task-completion metrics won't surface.

Usage
─────
    from smolagents import CodeAgent, HfApiModel
    from smolagents_integration import BehavioralFingerprintMonitor

    model = HfApiModel()
    agent = CodeAgent(tools=[], model=model)

    monitor = BehavioralFingerprintMonitor(
        agent=agent,
        history_drop_threshold=5,   # flag if history shrinks by ≥5 messages
        auto_snapshot=True,
    )

    result = agent.run("Write a function that sorts a list of dicts by key 'score'.")
    report = monitor.report()
    if report["drift_detected"]:
        print("Behavioral drift at steps:", report["drift_at_steps"])

Note: this adapter works with the existing step_callbacks surface. It does NOT require
the first-class MemoryConsolidationEvent hook proposed in
https://github.com/huggingface/smolagents/issues/2129 — but would be cleaner with it.

Dependencies: smolagents, compression_monitor (pip install compression-monitor)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from smolagents import MultiStepAgent

try:
    from ghost_lexicon import GhostLexiconTracker
    from behavioral_footprint import BehavioralFootprintTracker
except ImportError:
    try:
        from compression_monitor.ghost_lexicon import GhostLexiconTracker
        from compression_monitor.behavioral_footprint import BehavioralFootprintTracker
    except ImportError:
        raise ImportError(
            "compression-monitor not found. Install with: pip install compression-monitor\n"
            "or run from the compression-monitor source directory."
        )


@dataclass
class FingerprintSnapshot:
    step: int
    history_length: int
    ghost_lexicon: Dict[str, float]
    tool_call_sequence: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class DriftEvent:
    step: int
    reason: str  # "history_drop" | "threshold_crossed"
    history_before: int
    history_after: int
    lexicon_overlap: float        # Jaccard overlap of top-20 domain terms (1.0 = no drift)
    tool_sequence_overlap: float  # Jaccard overlap of tool call set (1.0 = no drift)


class BehavioralFingerprintMonitor:
    """
    Attaches to a smolagents MultiStepAgent via step callbacks and monitors for
    behavioral drift at memory consolidation boundaries.

    Consolidation boundary detection heuristics:
      1. History shrinks between consecutive steps (messages were dropped/summarized)
      2. History crosses a configurable length threshold (approaching context limit)
    """

    def __init__(
        self,
        agent: "MultiStepAgent",
        history_drop_threshold: int = 3,
        history_length_alert: Optional[int] = None,
        lexicon_drift_threshold: float = 0.6,
        tool_drift_threshold: float = 0.5,
        auto_snapshot: bool = True,
        verbose: bool = False,
    ):
        self.agent = agent
        self.history_drop_threshold = history_drop_threshold
        self.history_length_alert = history_length_alert
        self.lexicon_drift_threshold = lexicon_drift_threshold
        self.tool_drift_threshold = tool_drift_threshold
        self.auto_snapshot = auto_snapshot
        self.verbose = verbose

        self._snapshots: List[FingerprintSnapshot] = []
        self._drift_events: List[DriftEvent] = []
        self._step_counter = 0
        self._lexicon = GhostLexiconTracker()
        self._footprint = BehavioralFootprintTracker()
        self._last_history_len: Optional[int] = None

        # Register with agent's step callbacks if available
        self._register_callbacks()

    def _register_callbacks(self) -> None:
        """Wire into smolagents step_callbacks if the agent supports them."""
        if hasattr(self.agent, "step_callbacks"):
            self.agent.step_callbacks.append(self._on_step)
        else:
            import warnings
            warnings.warn(
                "Agent does not expose step_callbacks. "
                "Call monitor.on_step(agent_output, step_logs) manually after each step.",
                stacklevel=2,
            )

    def _extract_history_length(self) -> int:
        """Get current agent message history length."""
        for attr in ("memory", "_memory", "messages", "_messages", "input_messages"):
            obj = getattr(self.agent, attr, None)
            if obj is not None:
                if hasattr(obj, "messages"):
                    return len(obj.messages)
                if isinstance(obj, list):
                    return len(obj)
        return 0

    def _extract_tool_calls(self, step_log: Any) -> List[str]:
        """Pull tool names from a step log entry."""
        tools: List[str] = []
        if step_log is None:
            return tools
        for attr in ("tool_calls", "tool_name", "action"):
            val = getattr(step_log, attr, None)
            if val is None:
                continue
            if isinstance(val, str):
                tools.append(val)
            elif isinstance(val, list):
                for tc in val:
                    name = getattr(tc, "name", None) or getattr(tc, "tool_name", None)
                    if name:
                        tools.append(name)
        return tools

    def _extract_text_output(self, step_log: Any) -> str:
        """Pull text content from a step log entry for lexicon tracking."""
        for attr in ("observations", "llm_output", "output", "content"):
            val = getattr(step_log, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    def _take_snapshot(self, step: int, history_len: int, tool_calls: List[str]) -> FingerprintSnapshot:
        snap = FingerprintSnapshot(
            step=step,
            history_length=history_len,
            ghost_lexicon=self._lexicon.current_distribution(),
            tool_call_sequence=list(tool_calls),
        )
        self._snapshots.append(snap)
        return snap

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        union = a | b
        if not union:
            return 1.0
        return len(a & b) / len(union)

    def _compare_snapshots(self, before: FingerprintSnapshot, after: FingerprintSnapshot, reason: str) -> DriftEvent:
        top_n = 20
        before_terms = set(sorted(before.ghost_lexicon, key=before.ghost_lexicon.get, reverse=True)[:top_n])
        after_terms = set(sorted(after.ghost_lexicon, key=after.ghost_lexicon.get, reverse=True)[:top_n])
        lexicon_overlap = self._jaccard(before_terms, after_terms)

        before_tools = set(before.tool_call_sequence)
        after_tools = set(after.tool_call_sequence)
        tool_overlap = self._jaccard(before_tools, after_tools)

        event = DriftEvent(
            step=after.step,
            reason=reason,
            history_before=before.history_length,
            history_after=after.history_length,
            lexicon_overlap=lexicon_overlap,
            tool_sequence_overlap=tool_overlap,
        )
        self._drift_events.append(event)
        return event

    def _on_step(self, step_log: Any, *args: Any, **kwargs: Any) -> None:
        """Callback fired after each agent step."""
        self._step_counter += 1
        step = self._step_counter

        text = self._extract_text_output(step_log)
        if text:
            self._lexicon.update(text)

        tool_calls = self._extract_tool_calls(step_log)
        for tc in tool_calls:
            self._footprint.record_call(tc)

        current_history_len = self._extract_history_length()

        boundary_reason: Optional[str] = None
        if self._last_history_len is not None:
            drop = self._last_history_len - current_history_len
            if drop >= self.history_drop_threshold:
                boundary_reason = "history_drop"
            elif (
                self.history_length_alert
                and current_history_len >= self.history_length_alert
                and self._last_history_len < self.history_length_alert
            ):
                boundary_reason = "threshold_crossed"

        if boundary_reason is not None and len(self._snapshots) > 0:
            post_snap = self._take_snapshot(step, current_history_len, tool_calls)
            prev = self._snapshots[-2] if len(self._snapshots) >= 2 else self._snapshots[-1]
            event = self._compare_snapshots(prev, post_snap, boundary_reason)

            flagged = (
                event.lexicon_overlap < self.lexicon_drift_threshold
                or event.tool_sequence_overlap < self.tool_drift_threshold
            )

            if self.verbose or flagged:
                print(
                    f"[compression-monitor] Step {step}: {boundary_reason} "
                    f"(history {event.history_before}→{event.history_after}), "
                    f"lexicon_overlap={event.lexicon_overlap:.2f}, "
                    f"tool_overlap={event.tool_sequence_overlap:.2f}"
                    + (" ⚠ DRIFT FLAGGED" if flagged else "")
                )
        elif self.auto_snapshot and (step == 1 or step % 10 == 0):
            self._take_snapshot(step, current_history_len, tool_calls)

        self._last_history_len = current_history_len

    def on_step(self, step_log: Any, *args: Any, **kwargs: Any) -> None:
        """Manual call surface for agents without step_callbacks support."""
        self._on_step(step_log, *args, **kwargs)

    def report(self) -> Dict[str, Any]:
        """Return a summary of detected drift events and snapshot history."""
        flagged = [
            e for e in self._drift_events
            if (
                e.lexicon_overlap < self.lexicon_drift_threshold
                or e.tool_sequence_overlap < self.tool_drift_threshold
            )
        ]
        return {
            "drift_detected": len(flagged) > 0,
            "drift_at_steps": [e.step for e in flagged],
            "drift_events": [
                {
                    "step": e.step,
                    "reason": e.reason,
                    "history_before": e.history_before,
                    "history_after": e.history_after,
                    "lexicon_overlap": round(e.lexicon_overlap, 3),
                    "tool_sequence_overlap": round(e.tool_sequence_overlap, 3),
                }
                for e in flagged
            ],
            "total_steps": self._step_counter,
            "snapshots_taken": len(self._snapshots),
            "boundary_events": len(self._drift_events),
        }
