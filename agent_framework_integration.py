"""
agent_framework_integration.py
────────────────────────────────
Behavioral consistency monitoring for microsoft/agent-framework at compaction
and context-isolation boundaries.

agent-framework's compaction layer runs when a session's message history exceeds
token or turn thresholds. With multi-agent orchestrations (HandoffBuilder,
GroupChatBuilder), compaction events also happen when IsolateAgentCompactionStrategy
filters shared history — an agent's effective context can be silently reduced even
without a hard token limit being hit.

This adapter intercepts compaction/isolation boundaries and measures whether each
agent's behavioral fingerprint is preserved across them:
  - Ghost lexicon decay: high-precision terms used pre-compaction that go quiet after
  - Tool-call sequence divergence: Jaccard distance between pre/post tool-use patterns
  - Semantic topic drift: keyword overlap across rolling output windows

Usage — drop-in wrapper
───────────────────────
    from agent_framework import AgentClient
    from agent_framework_integration import MonitoredAgentClient, CompactionMonitor

    monitor = CompactionMonitor(agent_name="SalesAgent")
    client = AgentClient(...)
    monitored = MonitoredAgentClient(client, monitor)

    async for response in monitored.stream("What's the pricing for plan B?"):
        print(response.text)

Usage — explicit boundary marking
──────────────────────────────────
    monitor = CompactionMonitor(agent_name="SalesAgent")

    # Before compaction or context-isolation event:
    monitor.record_pre_compaction(agent_outputs)

    # After compaction:
    monitor.record_post_compaction(agent_outputs)
    report = monitor.compute_ccs()
    if report["ccs"] < 0.70:
        print(f"Behavioral drift detected: CCS={report['ccs']:.3f}")
        print("Ghost terms:", report["ghost_lexicon"])

Usage — IsolateAgentCompactionStrategy validation
──────────────────────────────────────────────────
    # Wrap each agent with a monitor to verify isolation
    # is not producing unexpected behavioral drift:
    monitor = CompactionMonitor.from_isolation_strategy(
        agent_name="SalesAgent",
        keep_last_turns=3,
    )
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "to", "of", "in", "on", "at", "by", "for",
    "with", "about", "as", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "and", "or", "but", "if", "then", "that", "this", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
    "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "not", "only", "same",
    "so", "than", "too", "very", "just", "can", "its", "our", "their",
})


def _tokenize(text: str) -> list[str]:
    """Extract lowercase word tokens, excluding stop words."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", text)
    return [w.lower() for w in words if w.lower() not in _STOP_WORDS]


def _term_freq(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _high_precision_terms(freq: Counter, min_len: int = 5, min_count: int = 2) -> set[str]:
    """Return terms that are long enough and appear at least min_count times."""
    return {t for t, c in freq.items() if len(t) >= min_len and c >= min_count}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _overlap_coefficient(a: set, b: set) -> float:
    """Overlap coefficient: how much of the smaller set is in the larger."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ──────────────────────────────────────────────────────────────────────────────
# Core monitor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundarySnapshot:
    """Behavioral fingerprint snapshot at one side of a compaction boundary."""
    agent_name: str
    timestamp: float
    outputs: list[str]
    tool_calls: list[str]
    tokens: list[str] = field(default_factory=list)
    term_freq: Counter = field(default_factory=Counter)
    precision_terms: set[str] = field(default_factory=set)
    tool_set: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        combined = " ".join(self.outputs)
        self.tokens = _tokenize(combined)
        self.term_freq = _term_freq(self.tokens)
        self.precision_terms = _high_precision_terms(self.term_freq)
        self.tool_set = set(self.tool_calls)


@dataclass
class CompactionReport:
    """Result of comparing pre- and post-compaction snapshots."""
    agent_name: str
    boundary_at: float
    ccs: float                           # 0.0–1.0, 1.0 = no drift
    ghost_lexicon: list[str]             # terms that disappeared post-compaction
    tool_divergence: float               # 1 - Jaccard of tool sets
    semantic_overlap: float              # keyword overlap coefficient
    precision_term_retention: float      # fraction of pre-terms that survived
    alert: bool                          # True if CCS below threshold
    threshold: float

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "boundary_at": self.boundary_at,
            "ccs": round(self.ccs, 4),
            "ghost_lexicon": self.ghost_lexicon,
            "tool_divergence": round(self.tool_divergence, 4),
            "semantic_overlap": round(self.semantic_overlap, 4),
            "precision_term_retention": round(self.precision_term_retention, 4),
            "alert": self.alert,
            "threshold": self.threshold,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        status = "⚠ DRIFT DETECTED" if self.alert else "✓ stable"
        return (
            f"[CompactionMonitor:{self.agent_name}] {status} "
            f"CCS={self.ccs:.3f} "
            f"ghost={len(self.ghost_lexicon)} "
            f"tool_div={self.tool_divergence:.3f}"
        )


class CompactionMonitor:
    """
    Monitor behavioral consistency for a single agent across compaction
    or context-isolation boundaries.

    CCS (Context-Consistency Score) is a weighted combination of:
      - precision_term_retention (0.45): how much high-value vocabulary survived
      - semantic_overlap (0.30): topic keyword overlap across boundary
      - tool stability (0.25): how similar the tool-use pattern remained

    A CCS below ``threshold`` (default 0.70) triggers an alert.
    """

    def __init__(
        self,
        agent_name: str,
        threshold: float = 0.70,
        weights: tuple[float, float, float] = (0.45, 0.30, 0.25),
    ) -> None:
        self.agent_name = agent_name
        self.threshold = threshold
        self.weights = weights  # (precision_term_retention, semantic_overlap, tool_stability)
        self._pre: BoundarySnapshot | None = None
        self._post: BoundarySnapshot | None = None
        self.history: list[CompactionReport] = []

    @classmethod
    def from_isolation_strategy(
        cls,
        agent_name: str,
        keep_last_turns: int = 3,
        threshold: float = 0.70,
    ) -> "CompactionMonitor":
        """
        Factory for agents using IsolateAgentCompactionStrategy.
        The keep_last_turns hint is stored for diagnostics; it doesn't change
        measurement logic, but it's useful when interpreting ghost lexicon reports
        (terms dropped > keep_last_turns turns ago are expected losses).
        """
        monitor = cls(agent_name=agent_name, threshold=threshold)
        monitor._keep_last_turns = keep_last_turns
        return monitor

    def record_pre_compaction(
        self,
        outputs: list[str],
        tool_calls: list[str] | None = None,
    ) -> BoundarySnapshot:
        """
        Record behavioral fingerprint before a compaction or isolation event.

        Args:
            outputs: Recent text outputs from the agent (last N turns).
            tool_calls: Tool names called in the same window (optional).
        """
        self._pre = BoundarySnapshot(
            agent_name=self.agent_name,
            timestamp=time.time(),
            outputs=outputs,
            tool_calls=tool_calls or [],
        )
        return self._pre

    def record_post_compaction(
        self,
        outputs: list[str],
        tool_calls: list[str] | None = None,
    ) -> BoundarySnapshot:
        """Record behavioral fingerprint after the compaction or isolation event."""
        self._post = BoundarySnapshot(
            agent_name=self.agent_name,
            timestamp=time.time(),
            outputs=outputs,
            tool_calls=tool_calls or [],
        )
        return self._post

    def compute_ccs(self) -> CompactionReport:
        """
        Compute Context-Consistency Score from the most recent pre/post pair.
        Raises RuntimeError if both snapshots haven't been recorded.
        """
        if self._pre is None or self._post is None:
            raise RuntimeError(
                "Both pre- and post-compaction snapshots must be recorded before computing CCS. "
                "Call record_pre_compaction() and record_post_compaction() first."
            )

        pre, post = self._pre, self._post

        # 1. Precision term retention
        if pre.precision_terms:
            retained = pre.precision_terms & post.precision_terms
            precision_term_retention = len(retained) / len(pre.precision_terms)
        else:
            precision_term_retention = 1.0

        # 2. Semantic keyword overlap
        pre_topics = set(pre.term_freq.keys())
        post_topics = set(post.term_freq.keys())
        semantic_overlap = _overlap_coefficient(pre_topics, post_topics)

        # 3. Tool stability
        tool_stability = _jaccard(pre.tool_set, post.tool_set)

        # 4. Weighted CCS
        w_ptr, w_sem, w_tool = self.weights
        ccs = (
            w_ptr * precision_term_retention
            + w_sem * semantic_overlap
            + w_tool * tool_stability
        )
        ccs = min(1.0, max(0.0, ccs))

        # 5. Ghost lexicon: precision terms that disappeared
        ghost = sorted(
            pre.precision_terms - post.precision_terms,
            key=lambda t: pre.term_freq[t],
            reverse=True,
        )

        report = CompactionReport(
            agent_name=self.agent_name,
            boundary_at=self._post.timestamp,
            ccs=ccs,
            ghost_lexicon=ghost[:20],
            tool_divergence=1.0 - tool_stability,
            semantic_overlap=semantic_overlap,
            precision_term_retention=precision_term_retention,
            alert=ccs < self.threshold,
            threshold=self.threshold,
        )
        self.history.append(report)
        return report

    def session_summary(self) -> dict:
        """Aggregate CCS across all recorded boundary events in this session."""
        if not self.history:
            return {"agent_name": self.agent_name, "boundaries": 0, "mean_ccs": None}
        csss = [r.ccs for r in self.history]
        return {
            "agent_name": self.agent_name,
            "boundaries": len(self.history),
            "mean_ccs": round(sum(csss) / len(csss), 4),
            "min_ccs": round(min(csss), 4),
            "max_ccs": round(max(csss), 4),
            "alert_count": sum(1 for r in self.history if r.alert),
        }


# ──────────────────────────────────────────────────────────────────────────────
# agent-framework adapter: wraps an agent client to auto-detect boundaries
# ──────────────────────────────────────────────────────────────────────────────

class AgentFrameworkMonitor:
    """
    Session-level behavioral monitor for multi-agent orchestrations built with
    microsoft/agent-framework.

    Maintains one CompactionMonitor per participating agent. Call
    ``on_compaction_event(agent_name, ...)`` from any middleware or
    post-turn hook to record boundaries as they happen.

    For IsolateAgentCompactionStrategy validation, call
    ``register_isolated_agent(agent_name, keep_last_turns)`` at startup.
    """

    def __init__(self, threshold: float = 0.70) -> None:
        self.threshold = threshold
        self._monitors: dict[str, CompactionMonitor] = {}
        self._output_buffers: dict[str, list[str]] = {}
        self._tool_buffers: dict[str, list[str]] = {}

    def register_agent(
        self,
        agent_name: str,
        keep_last_turns: int | None = None,
    ) -> CompactionMonitor:
        """Register an agent for behavioral monitoring."""
        if keep_last_turns is not None:
            monitor = CompactionMonitor.from_isolation_strategy(
                agent_name=agent_name,
                keep_last_turns=keep_last_turns,
                threshold=self.threshold,
            )
        else:
            monitor = CompactionMonitor(
                agent_name=agent_name,
                threshold=self.threshold,
            )
        self._monitors[agent_name] = monitor
        self._output_buffers[agent_name] = []
        self._tool_buffers[agent_name] = []
        return monitor

    def record_turn(
        self,
        agent_name: str,
        output: str,
        tools_called: list[str] | None = None,
    ) -> None:
        """Record a completed agent turn. Call after each agent response."""
        if agent_name not in self._monitors:
            self.register_agent(agent_name)
        self._output_buffers[agent_name].append(output)
        if tools_called:
            self._tool_buffers[agent_name].extend(tools_called)
        # Keep rolling window of last 10 turns
        self._output_buffers[agent_name] = self._output_buffers[agent_name][-10:]
        self._tool_buffers[agent_name] = self._tool_buffers[agent_name][-20:]

    def on_compaction_event(
        self,
        agent_name: str,
        is_pre: bool,
        outputs: list[str] | None = None,
        tool_calls: list[str] | None = None,
    ) -> CompactionReport | None:
        """
        Call this from a compaction middleware or hook.

        Set is_pre=True before compaction, is_pre=False after.
        After the post event, returns a CompactionReport immediately.

        Args:
            agent_name: Name of the agent being compacted.
            is_pre: True = before compaction, False = after.
            outputs: Override for output buffer (uses rolling buffer if None).
            tool_calls: Override for tool buffer.
        """
        if agent_name not in self._monitors:
            self.register_agent(agent_name)

        monitor = self._monitors[agent_name]
        out = outputs or self._output_buffers.get(agent_name, [""])
        tools = tool_calls or list(set(self._tool_buffers.get(agent_name, [])))

        if is_pre:
            monitor.record_pre_compaction(outputs=out, tool_calls=tools)
            return None
        else:
            monitor.record_post_compaction(outputs=out, tool_calls=tools)
            try:
                report = monitor.compute_ccs()
                return report
            except RuntimeError:
                return None

    def full_report(self) -> dict:
        """Return session-level summaries for all registered agents."""
        return {
            name: monitor.session_summary()
            for name, monitor in self._monitors.items()
        }


# ──────────────────────────────────────────────────────────────────────────────
# Example: IsolateAgentCompactionStrategy validation pattern
# ──────────────────────────────────────────────────────────────────────────────

EXAMPLE = '''
Example: Validating IsolateAgentCompactionStrategy
────────────────────────────────────────────────────

from agent_framework import AgentClient
from agent_framework._compaction import IsolateAgentCompactionStrategy
from agent_framework_integration import AgentFrameworkMonitor

monitor = AgentFrameworkMonitor(threshold=0.70)

# Register agents with their isolation window sizes
monitor.register_agent("SalesAgent", keep_last_turns=3)
monitor.register_agent("SupportAgent", keep_last_turns=5)

# Build the orchestration as normal
sales_agent = client.as_agent(
    name="SalesAgent",
    instructions="You handle sales...",
    tools=[get_pricing, add_to_cart],
    compaction_strategy=IsolateAgentCompactionStrategy(
        agent_name="SalesAgent", keep_last_turns=3
    )
)

# Before compaction runs (in a middleware hook):
monitor.on_compaction_event("SalesAgent", is_pre=True)

# After compaction runs:
report = monitor.on_compaction_event("SalesAgent", is_pre=False)
if report and report.alert:
    print(f"Unexpected drift after isolation: {report.summary()}")
    print("Ghost terms:", report.ghost_lexicon[:5])
    # These are domain terms the isolation dropped that the agent was actively using.
    # Consider increasing keep_last_turns or pinning them to system context.

# End-of-session summary
print(monitor.full_report())
'''


if __name__ == "__main__":
    print("agent_framework_integration.py — compression-monitor adapter")
    print("github.com/agent-morrow/compression-monitor")
    print()
    print(EXAMPLE)

    # Minimal self-test
    monitor = CompactionMonitor(agent_name="TestAgent")

    pre_outputs = [
        "The IsolateAgentCompactionStrategy filters shared context per participant. "
        "The HandoffBuilder routes to the appropriate agent based on task_routing_key. "
        "Using keep_last_turns=3 with annotate_message_groups preserves recency.",
        "The compaction_strategy parameter accepts any ICompactionStrategy. "
        "For SalesAgent, pricing_context and cart_state are critical retention targets.",
    ]
    post_outputs = [
        "I can help you with sales inquiries.",
        "Please let me know what you need.",
    ]

    monitor.record_pre_compaction(pre_outputs, tool_calls=["get_pricing", "add_to_cart"])
    monitor.record_post_compaction(post_outputs, tool_calls=["get_pricing"])
    report = monitor.compute_ccs()

    print("Self-test result:")
    print(report.summary())
    print(f"Ghost lexicon: {report.ghost_lexicon[:8]}")
    assert report.alert, "Expected drift alert in self-test (post-outputs are generic)"
    print("✓ Self-test passed")
