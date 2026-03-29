"""
autogen_integration.py — compression-monitor adapter for AutoGen

Monitors behavioral drift in AutoGen agents across conversation turns and session boundaries.

Usage:
    from autogen_integration import AgentDriftMonitor, MonitoredConversableAgent

    # Option 1: Wrap an existing agent
    agent = MonitoredConversableAgent(
        name="assistant",
        llm_config={...},
        monitor_dir="./drift_logs"
    )

    # Option 2: Monitor an existing agent via callback hooks
    monitor = AgentDriftMonitor(monitor_dir="./drift_logs")
    monitor.attach(agent)
    # run your conversation...
    print(monitor.drift_report())

Requirements:
    pip install pyautogen
"""

import json
import re
import hashlib
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional, Callable
from collections import Counter


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------

def _extract_lexicon(text: str, min_len: int = 5) -> Counter:
    words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_len, text.lower())
    return Counter(words)


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def _ghost_score(prior: str, current: str, top_n: int = 50) -> float:
    prior_top = [w for w, _ in _extract_lexicon(prior).most_common(top_n)]
    if not prior_top:
        return 1.0
    current_words = set(_extract_lexicon(current).keys())
    return sum(1 for w in prior_top if w in current_words) / len(prior_top)


def _semantic_overlap(a: str, b: str, top_n: int = 30) -> float:
    top_a = set(w for w, _ in _extract_lexicon(a).most_common(top_n))
    top_b = set(w for w, _ in _extract_lexicon(b).most_common(top_n))
    return _jaccard(top_a, top_b)


# ---------------------------------------------------------------------------
# Turn snapshot — captures one agent's message turn
# ---------------------------------------------------------------------------

class TurnSnapshot:
    def __init__(self, agent_name: str, messages: list[dict],
                 snapshot_id: Optional[str] = None):
        self.agent_name = agent_name
        self.snapshot_id = snapshot_id or f"snap_{int(time.time()*1000)}"
        self.timestamp = datetime.now(timezone.utc).isoformat()

        # Extract text from messages sent by this agent
        agent_messages = [
            m for m in messages
            if m.get("name") == agent_name or m.get("role") == "assistant"
        ]
        self.text_content = "\n".join(str(m.get("content", "")) for m in agent_messages)
        self.message_count = len(agent_messages)
        self.lexicon = _extract_lexicon(self.text_content)

        # Extract any function/tool calls
        self.tool_calls: list[str] = []
        for m in agent_messages:
            for fc in m.get("tool_calls", []) or []:
                if isinstance(fc, dict):
                    fn = fc.get("function", {})
                    self.tool_calls.append(fn.get("name", "unknown"))

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "message_count": self.message_count,
            "text_length": len(self.text_content),
            "tool_calls": self.tool_calls,
            "top_lexicon": dict(self.lexicon.most_common(20)),
            "content_hash": hashlib.sha256(self.text_content.encode()).hexdigest()[:16],
        }


# ---------------------------------------------------------------------------
# Drift measurement
# ---------------------------------------------------------------------------

class TurnDrift:
    def __init__(self, before: TurnSnapshot, after: TurnSnapshot):
        self.before = before
        self.after = after
        self.agent_name = before.agent_name

        self.ghost = _ghost_score(before.text_content, after.text_content)
        self.semantic = _semantic_overlap(before.text_content, after.text_content)
        self.tool_jac = _jaccard(set(before.tool_calls), set(after.tool_calls))

        # Weighted composite drift (0=identical, 1=completely different)
        self.drift_score = 1.0 - (self.ghost * 0.4 + self.semantic * 0.3 + self.tool_jac * 0.3)

    @property
    def alert(self) -> Optional[str]:
        issues = []
        if self.ghost < 0.5:
            issues.append(f"ghost_lexicon={self.ghost:.2f}")
        if self.semantic < 0.4:
            issues.append(f"semantic={self.semantic:.2f}")
        if self.tool_jac < 0.3 and (self.before.tool_calls or self.after.tool_calls):
            issues.append(f"tool_jaccard={self.tool_jac:.2f}")
        if issues:
            return f"DRIFT ALERT [{self.agent_name}]: " + "; ".join(issues)
        return None

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "before_id": self.before.snapshot_id,
            "after_id": self.after.snapshot_id,
            "drift_score": round(self.drift_score, 4),
            "ghost_lexicon_survival": round(self.ghost, 4),
            "semantic_overlap": round(self.semantic, 4),
            "tool_jaccard": round(self.tool_jac, 4),
            "alert": self.alert,
        }


# ---------------------------------------------------------------------------
# AgentDriftMonitor — attach to any ConversableAgent via reply hooks
# ---------------------------------------------------------------------------

class AgentDriftMonitor:
    """
    Attaches to AutoGen ConversableAgent instances to measure behavioral
    drift across conversation sessions.

    Can be attached to multiple agents simultaneously. Logs drift to JSONL.
    """

    def __init__(self, monitor_dir: str = "./drift_logs", drift_threshold: float = 0.3):
        self.monitor_dir = Path(monitor_dir)
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        self.drift_threshold = drift_threshold

        self._snapshots: dict[str, TurnSnapshot] = {}  # agent_name -> last snapshot
        self._measurements: list[TurnDrift] = []
        self._session_count = 0

    def attach(self, agent) -> None:
        """
        Attach monitoring hooks to a ConversableAgent.
        Wraps the agent's generate_reply method to capture outputs.
        """
        original_generate = agent.generate_reply

        agent_name = getattr(agent, 'name', str(id(agent)))
        monitor = self  # capture ref

        def monitored_generate(messages=None, sender=None, **kwargs):
            result = original_generate(messages=messages, sender=sender, **kwargs)
            if messages and result:
                monitor._on_reply(agent_name, messages, result)
            return result

        agent.generate_reply = monitored_generate

    def _on_reply(self, agent_name: str, messages: list[dict], reply: Any):
        """Called after each generate_reply. Snapshots and measures drift."""
        # Build snapshot from message history plus the new reply
        all_msgs = list(messages or [])
        if isinstance(reply, str):
            all_msgs.append({"name": agent_name, "role": "assistant", "content": reply})
        elif isinstance(reply, dict):
            all_msgs.append({**reply, "name": agent_name})

        snap_id = f"{agent_name}_{int(time.time()*1000)}"
        snap = TurnSnapshot(agent_name, all_msgs, snapshot_id=snap_id)

        if agent_name in self._snapshots:
            m = TurnDrift(self._snapshots[agent_name], snap)
            self._measurements.append(m)
            if m.alert:
                print(f"\n[WARN] {m.alert}")
            elif m.drift_score > self.drift_threshold:
                print(f"\n[WARN] DRIFT WARNING [{agent_name}]: score={m.drift_score:.3f}")
            log_path = self.monitor_dir / f"{snap_id}_drift.jsonl"
            with open(log_path, 'w') as f:
                f.write(json.dumps(m.to_dict()) + "\n")

        self._snapshots[agent_name] = snap

    def snapshot_session(self, agent_name: str, chat_history: list[dict],
                         session_label: Optional[str] = None) -> TurnSnapshot:
        """
        Manually snapshot an agent's state from a completed chat history.
        Call at session boundaries to track cross-session drift.
        """
        label = session_label or f"session_{int(time.time())}"
        snap = TurnSnapshot(agent_name, chat_history, snapshot_id=label)

        if agent_name in self._snapshots:
            m = TurnDrift(self._snapshots[agent_name], snap)
            self._measurements.append(m)
            self._session_count += 1
            if m.alert:
                print(f"\n[WARN] {m.alert}")
            log_path = self.monitor_dir / f"{label}_drift.jsonl"
            with open(log_path, 'w') as f:
                f.write(json.dumps(m.to_dict()) + "\n")

        self._snapshots[agent_name] = snap
        snap_path = self.monitor_dir / f"{label}_snapshot.jsonl"
        with open(snap_path, 'w') as f:
            f.write(json.dumps(snap.to_dict()) + "\n")
        return snap

    def drift_report(self) -> dict:
        if not self._measurements:
            return {
                "sessions_measured": self._session_count,
                "measurements": 0,
                "note": "No drift measurements yet. Need at least 2 snapshots per agent."
            }
        alerts = [m.alert for m in self._measurements if m.alert]
        avg = sum(m.drift_score for m in self._measurements) / len(self._measurements)
        worst = max(self._measurements, key=lambda m: m.drift_score)
        return {
            "sessions_measured": self._session_count,
            "measurements": len(self._measurements),
            "avg_drift_score": round(avg, 4),
            "worst_agent": worst.agent_name,
            "worst_drift_score": round(worst.drift_score, 4),
            "alerts": alerts,
            "log_dir": str(self.monitor_dir),
        }


# ---------------------------------------------------------------------------
# MonitoredConversableAgent — drop-in subclass
# ---------------------------------------------------------------------------

class MonitoredConversableAgent:
    """
    Drop-in wrapper around AutoGen ConversableAgent with built-in drift monitoring.

    Parameters mirror ConversableAgent exactly. After each session, call
    .drift_report() to see behavioral change metrics.

    Example:
        agent = MonitoredConversableAgent(
            name="assistant",
            llm_config={"model": "gpt-4"},
            monitor_dir="./logs"
        )
        agent.initiate_chat(other_agent, message="help me with X")
        agent.initiate_chat(other_agent, message="help me with X")  # drift measured
        print(agent.drift_report())
    """

    def __init__(self, *args, monitor_dir: str = "./drift_logs",
                 drift_threshold: float = 0.3, **kwargs):
        try:
            from autogen import ConversableAgent
            self._agent = ConversableAgent(*args, **kwargs)
        except ImportError:
            raise ImportError("pyautogen is required: pip install pyautogen")

        self._monitor = AgentDriftMonitor(monitor_dir=monitor_dir,
                                          drift_threshold=drift_threshold)
        self._monitor.attach(self._agent)
        self._chat_count = 0

    def initiate_chat(self, recipient, *args, **kwargs):
        """Run a chat and snapshot drift at the end."""
        result = self._agent.initiate_chat(recipient, *args, **kwargs)
        self._chat_count += 1
        # Snapshot from final chat history
        history = self._agent.chat_messages.get(recipient, [])
        agent_name = getattr(self._agent, 'name', 'agent')
        self._monitor.snapshot_session(agent_name, history,
                                       session_label=f"chat_{self._chat_count:04d}")
        return result

    def drift_report(self) -> dict:
        return self._monitor.drift_report()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("compression-monitor / autogen_integration.py")
    print("AgentDriftMonitor + MonitoredConversableAgent for AutoGen.")
    print()

    monitor = AgentDriftMonitor(monitor_dir="/tmp/autogen_test_drift")

    history_1 = [
        {"name": "assistant", "role": "assistant",
         "content": "I've analyzed the codebase and found three critical security vulnerabilities in the authentication module. The JWT validation logic is missing expiry checks and the password hashing uses MD5."},
        {"name": "assistant", "role": "assistant",
         "content": "I recommend patching the authentication module first, then running the security scanner against the API endpoints to identify further exposure."},
    ]
    history_2 = [
        {"name": "assistant", "role": "assistant",
         "content": "The task has been completed. I've processed your request and returned the output. Let me know if you need anything else."},
        {"name": "assistant", "role": "assistant",
         "content": "Here is the result of the operation. The system responded successfully and all steps were executed without errors."},
    ]
    history_3 = [
        {"name": "assistant", "role": "assistant",
         "content": "Continuing the security analysis: I've patched the JWT expiry validation and updated the password hashing to bcrypt. The authentication module is now hardened."},
    ]

    s1 = monitor.snapshot_session("assistant", history_1, "session_A")
    s2 = monitor.snapshot_session("assistant", history_2, "session_B")
    s3 = monitor.snapshot_session("assistant", history_3, "session_C")

    report = monitor.drift_report()
    print(f"  Measurements: {report['measurements']}")
    print(f"  Avg drift: {report['avg_drift_score']}")
    print(f"  Worst: {report['worst_drift_score']}")
    print(f"  Alerts: {len(report['alerts'])}")
    # Expect session_B (generic filler) to score high drift vs session_A (security-focused)
    # Expect session_C (back to security) to score lower drift vs session_B
    print("  OK")
