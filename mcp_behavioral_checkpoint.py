"""
mcp_behavioral_checkpoint.py — Reference implementation for MCP SEP #2492

Demonstrates the proposed session resumption + behavioral checkpoint pattern
for MCP initialize requests. See:
https://github.com/modelcontextprotocol/modelcontextprotocol/issues/2492

Usage:
    # 1. Capture a behavioral baseline at the start of a session
    checkpoint = MCPBehavioralCheckpoint(session_id="my-agent-session-001")
    checkpoint.record_tool_call("read_file", {"path": "/etc/hosts"})
    checkpoint.record_tool_call("search", {"query": "agent memory"})
    checkpoint.record_probe_response("What are you working on?", "Reading system files and searching.")

    # 2. Serialize to inject into the next MCP initialize request
    fingerprint = checkpoint.to_initialize_params()
    print(fingerprint)
    # {
    #   "sessionId": "my-agent-session-001",
    #   "behavioralCheckpoint": {
    #     "capturedAt": "2026-03-29T02:21:00Z",
    #     "toolCallVectorHash": "a3f2...",
    #     "semanticAnchorHash": "8c1d..."
    #   }
    # }

    # 3. On session resume, compare against a new checkpoint
    drift = checkpoint.compare(new_checkpoint)
    print(drift)
    # {"driftScore": 0.12, "driftDimensions": {"toolCallPattern": 0.08, "semanticAnchor": 0.16}}
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    tool_name: str
    argument_keys: list[str]          # keys only, not values (privacy)
    timestamp_ms: int


@dataclass
class ProbeRecord:
    question: str
    response_tokens: list[str]        # lowercased word tokens


@dataclass
class DriftReport:
    drift_score: float                # 0.0 = identical, 1.0 = fully diverged
    tool_call_pattern_drift: float
    semantic_anchor_drift: float
    session_id: str
    baseline_captured_at: str
    comparison_captured_at: str

    def as_mcp_notification(self) -> dict:
        """Format as the proposed session/drift MCP notification payload."""
        return {
            "method": "session/drift",
            "params": {
                "sessionId": self.session_id,
                "driftScore": round(self.drift_score, 4),
                "driftDimensions": {
                    "toolCallPattern": round(self.tool_call_pattern_drift, 4),
                    "semanticAnchor": round(self.semantic_anchor_drift, 4),
                },
                "capturedAt": self.comparison_captured_at,
            },
        }


# ---------------------------------------------------------------------------
# Core checkpoint class
# ---------------------------------------------------------------------------

class MCPBehavioralCheckpoint:
    """
    Captures and serializes behavioral fingerprints for MCP sessions.

    Designed to be injected into MCP initialize requests as proposed in
    SEP #2492. The checkpoint contains only hashes — no raw content
    leaves the client.
    """

    def __init__(self, session_id: str, window_size: int = 50):
        """
        Args:
            session_id: Stable identifier for this agent session. Should
                persist across context rotations.
            window_size: Number of recent tool calls to include in the
                behavioral fingerprint.
        """
        self.session_id = session_id
        self.window_size = window_size
        self._tool_calls: list[ToolCallRecord] = []
        self._probes: list[ProbeRecord] = []
        self._captured_at: Optional[str] = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Record a tool call. Argument keys are retained; values are dropped."""
        self._tool_calls.append(ToolCallRecord(
            tool_name=tool_name,
            argument_keys=sorted(arguments.keys()),
            timestamp_ms=int(time.time() * 1000),
        ))
        # Keep only the most recent window_size calls
        if len(self._tool_calls) > self.window_size:
            self._tool_calls = self._tool_calls[-self.window_size:]

    def record_probe_response(self, question: str, response: str) -> None:
        """Record a semantic anchor probe response."""
        tokens = _tokenize(response)
        self._probes.append(ProbeRecord(question=question, response_tokens=tokens))

    def snapshot(self) -> "MCPBehavioralCheckpoint":
        """Return a frozen snapshot of the current state."""
        snap = MCPBehavioralCheckpoint(self.session_id, self.window_size)
        snap._tool_calls = list(self._tool_calls)
        snap._probes = list(self._probes)
        snap._captured_at = datetime.now(timezone.utc).isoformat()
        return snap

    # ------------------------------------------------------------------
    # Serialization for MCP initialize request
    # ------------------------------------------------------------------

    def to_initialize_params(self) -> dict:
        """
        Returns the clientInfo extension fields proposed in SEP #2492.

        Inject into the MCP initialize request as:
            clientInfo = {
                "name": "my-agent",
                "version": "1.0",
                **checkpoint.to_initialize_params()
            }
        """
        captured_at = self._captured_at or datetime.now(timezone.utc).isoformat()
        return {
            "sessionId": self.session_id,
            "behavioralCheckpoint": {
                "capturedAt": captured_at,
                "toolCallVectorHash": self._tool_call_vector_hash(),
                "semanticAnchorHash": self._semantic_anchor_hash(),
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_initialize_params(), indent=2)

    # ------------------------------------------------------------------
    # Drift comparison
    # ------------------------------------------------------------------

    def compare(self, other: "MCPBehavioralCheckpoint") -> DriftReport:
        """
        Compare this checkpoint against another (e.g., post-compression).

        Returns a DriftReport with scores in [0.0, 1.0].
        """
        tool_drift = self._compare_tool_vectors(other)
        semantic_drift = self._compare_semantic_anchors(other)
        composite = 0.6 * tool_drift + 0.4 * semantic_drift

        return DriftReport(
            drift_score=composite,
            tool_call_pattern_drift=tool_drift,
            semantic_anchor_drift=semantic_drift,
            session_id=self.session_id,
            baseline_captured_at=self._captured_at or "",
            comparison_captured_at=other._captured_at or "",
        )

    # ------------------------------------------------------------------
    # Internal fingerprinting
    # ------------------------------------------------------------------

    def _tool_call_vector(self) -> Counter:
        """Bigram frequency vector over (tool_name, arg_key_set_hash) pairs."""
        vec: Counter = Counter()
        calls = self._tool_calls
        for i, call in enumerate(calls):
            key = f"{call.tool_name}:{','.join(call.argument_keys)}"
            vec[key] += 1
            if i > 0:
                prev = calls[i - 1]
                bigram = (
                    f"{prev.tool_name}→{call.tool_name}"
                )
                vec[bigram] += 1
        return vec

    def _tool_call_vector_hash(self) -> str:
        vec = self._tool_call_vector()
        blob = json.dumps(sorted(vec.items()), separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _semantic_anchor_hash(self) -> str:
        if not self._probes:
            return "no-probes"
        tokens: Counter = Counter()
        for probe in self._probes:
            tokens.update(probe.response_tokens)
        blob = json.dumps(sorted(tokens.most_common(40)), separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _compare_tool_vectors(self, other: "MCPBehavioralCheckpoint") -> float:
        """Cosine distance between tool-call bigram frequency vectors."""
        a = self._tool_call_vector()
        b = other._tool_call_vector()
        if not a and not b:
            return 0.0
        if not a or not b:
            return 1.0
        keys = set(a) | set(b)
        dot = sum(a[k] * b[k] for k in keys)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 1.0
        cosine_similarity = dot / (mag_a * mag_b)
        return round(1.0 - cosine_similarity, 4)

    def _compare_semantic_anchors(self, other: "MCPBehavioralCheckpoint") -> float:
        """Jaccard distance between top-40 vocabulary tokens."""
        def top_tokens(cp: "MCPBehavioralCheckpoint") -> set:
            tokens: Counter = Counter()
            for probe in cp._probes:
                tokens.update(probe.response_tokens)
            return {tok for tok, _ in tokens.most_common(40)}

        a_tokens = top_tokens(self)
        b_tokens = top_tokens(other)
        if not a_tokens and not b_tokens:
            return 0.0
        if not a_tokens or not b_tokens:
            return 1.0
        intersection = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)
        return round(1.0 - intersection / union, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "through", "and", "or",
    "but", "if", "then", "that", "this", "it", "i", "you", "we", "they",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9_-]*", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== MCP Behavioral Checkpoint — SEP #2492 reference demo ===\n")

    # Baseline session: agent working on a code review task
    baseline = MCPBehavioralCheckpoint(session_id="demo-session-001")
    baseline.record_tool_call("read_file", {"path": "/src/main.py"})
    baseline.record_tool_call("search_code", {"query": "authentication", "repo": "myrepo"})
    baseline.record_tool_call("read_file", {"path": "/src/auth.py"})
    baseline.record_tool_call("run_tests", {"suite": "unit"})
    baseline.record_probe_response(
        "What are you currently working on?",
        "Reviewing authentication code in the repository. Checking unit tests for the auth module."
    )
    baseline = baseline.snapshot()

    params = baseline.to_initialize_params()
    print("Initialize params (inject into MCP clientInfo):")
    print(json.dumps(params, indent=2))

    print("\n--- After context compression: agent resumes ---\n")

    # Post-compression session: same agent, but behavioral drift occurred
    post_compression = MCPBehavioralCheckpoint(session_id="demo-session-001")
    post_compression.record_tool_call("list_files", {"directory": "/src"})
    post_compression.record_tool_call("read_file", {"path": "/README.md"})
    post_compression.record_tool_call("search_web", {"query": "python best practices"})
    post_compression.record_probe_response(
        "What are you currently working on?",
        "Looking at the project structure and documentation. Exploring general Python practices."
    )
    post_compression = post_compression.snapshot()

    drift = baseline.compare(post_compression)
    print(f"Drift score:            {drift.drift_score:.4f}")
    print(f"Tool-call pattern drift: {drift.tool_call_pattern_drift:.4f}")
    print(f"Semantic anchor drift:   {drift.semantic_anchor_drift:.4f}")
    print()
    print("session/drift notification payload:")
    print(json.dumps(drift.as_mcp_notification(), indent=2))

    print("\n--- Same session, minimal drift ---\n")

    # Similar session: minor variation, should show low drift
    similar = MCPBehavioralCheckpoint(session_id="demo-session-001")
    similar.record_tool_call("read_file", {"path": "/src/main.py"})
    similar.record_tool_call("search_code", {"query": "authentication", "repo": "myrepo"})
    similar.record_tool_call("run_tests", {"suite": "unit"})
    similar.record_probe_response(
        "What are you currently working on?",
        "Still working on authentication code review and running unit tests."
    )
    similar = similar.snapshot()

    drift_low = baseline.compare(similar)
    print(f"Drift score:            {drift_low.drift_score:.4f}  (expected low)")
    print(f"Tool-call pattern drift: {drift_low.tool_call_pattern_drift:.4f}")
    print(f"Semantic anchor drift:   {drift_low.semantic_anchor_drift:.4f}")
