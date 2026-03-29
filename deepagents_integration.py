"""
deepagents_integration.py — compression-monitor adapter for LangChain Deep Agents

Detects context compression events emitted by DeepAgents' SummarizationMiddleware
and measures behavioral drift across each compaction boundary.

## How it works

DeepAgents' SummarizationMiddleware writes offloaded conversation history to:
    <backend_root>/conversation_history/<thread_id>.md

Each compaction appends a new section to this file. This adapter watches that
file for new sections, captures behavioral snapshots before and after each
compaction, and reports drift across the three core signals.

## Usage

    from deepagents import create_deep_agent
    from deepagents.middleware.summarization import SummarizationMiddleware
    from deepagents.backends import FilesystemBackend
    from deepagents_integration import DeepAgentsDriftMonitor

    backend = FilesystemBackend(root_dir="/data/agent_files")
    summ = SummarizationMiddleware(model="gpt-4o-mini", backend=backend)
    agent = create_deep_agent(middleware=[summ])

    monitor = DeepAgentsDriftMonitor(
        agent=agent,
        backend_root="/data/agent_files",
        thread_id="my_thread",
    )
    result = monitor.invoke({"messages": [HumanMessage(content="...")]})
    print(monitor.drift_report())

## Requirements

    pip install langgraph langchain langchain-deepagents
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Measurement primitives (inline to keep adapter self-contained)
# ---------------------------------------------------------------------------

def _extract_lexicon(text: str, min_len: int = 5) -> Counter:
    words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_len, text.lower())
    return Counter(words)


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _ghost_retention(prior: str, current: str, top_n: int = 50) -> float:
    """Fraction of top-N prior vocabulary still present after compaction."""
    top_prior = [w for w, _ in _extract_lexicon(prior).most_common(top_n)]
    if not top_prior:
        return 1.0
    current_words = set(_extract_lexicon(current).keys())
    return sum(1 for w in top_prior if w in current_words) / len(top_prior)


def _semantic_overlap(a: str, b: str, top_n: int = 30) -> float:
    """Jaccard overlap of top-N vocabulary sets."""
    ta = set(w for w, _ in _extract_lexicon(a).most_common(top_n))
    tb = set(w for w, _ in _extract_lexicon(b).most_common(top_n))
    return _jaccard(ta, tb)


# ---------------------------------------------------------------------------
# Conversation history parser
# ---------------------------------------------------------------------------

def _count_sections(path: Path) -> int:
    """Count markdown sections (## headings) in the history file."""
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r'^##\s+', text, re.MULTILINE))


def _read_history(path: Path) -> str:
    """Return full text of the conversation history file."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class Snapshot:
    """Behavioral fingerprint captured at a point in time."""

    def __init__(self, label: str, text: str, section_count: int):
        self.label = label
        self.text = text
        self.section_count = section_count
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.lexicon = _extract_lexicon(text)
        self.word_count = sum(self.lexicon.values())

    def __repr__(self) -> str:
        return (
            f"Snapshot(label={self.label!r}, "
            f"sections={self.section_count}, words={self.word_count})"
        )


# ---------------------------------------------------------------------------
# DeepAgentsDriftMonitor
# ---------------------------------------------------------------------------

class DeepAgentsDriftMonitor:
    """
    Wraps a DeepAgents agent invocation and measures behavioral drift
    across compaction events detected via the filesystem history file.

    Parameters
    ----------
    agent : CompiledStateGraph
        The agent returned by ``create_deep_agent()``.
    backend_root : str | Path
        The root directory passed to ``FilesystemBackend``.
    thread_id : str
        The LangGraph thread identifier used for the invocation.
    log_dir : str | Path, optional
        Directory to write JSON drift logs. Default: ``./drift_logs``.
    """

    def __init__(
        self,
        agent: Any,
        backend_root: str | Path,
        thread_id: str,
        log_dir: str | Path = "./drift_logs",
    ):
        self.agent = agent
        self.history_path = Path(backend_root) / "conversation_history" / f"{thread_id}.md"
        self.thread_id = thread_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._snapshots: list[Snapshot] = []
        self._events: list[dict] = []

    # ------------------------------------------------------------------ #

    def _capture(self, label: str, output_text: str) -> Snapshot:
        snap = Snapshot(
            label=label,
            text=output_text,
            section_count=_count_sections(self.history_path),
        )
        self._snapshots.append(snap)
        return snap

    def _diff(self, before: Snapshot, after: Snapshot) -> dict:
        return {
            "event": "compaction",
            "thread_id": self.thread_id,
            "before_label": before.label,
            "after_label": after.label,
            "before_sections": before.section_count,
            "after_sections": after.section_count,
            "sections_added": after.section_count - before.section_count,
            "ghost_retention": round(_ghost_retention(before.text, after.text), 4),
            "semantic_overlap": round(_semantic_overlap(before.text, after.text), 4),
            "word_count_delta": after.word_count - before.word_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------ #

    def invoke(self, input_: dict, config: Optional[dict] = None, **kwargs) -> Any:
        """
        Invoke the agent, capturing snapshots before and after each detected
        compaction event, plus a final post-run snapshot.
        """
        config = config or {}
        if "configurable" not in config:
            config["configurable"] = {}
        config["configurable"].setdefault("thread_id", self.thread_id)

        # Pre-run snapshot: use existing history as the baseline text
        pre_text = _read_history(self.history_path) or ""
        pre_sections = _count_sections(self.history_path)
        pre_snap = self._capture("pre_run", pre_text)

        result = self.agent.invoke(input_, config=config, **kwargs)

        # Post-run snapshot: new output text
        post_text = _read_history(self.history_path)
        post_snap = self._capture("post_run", post_text)

        sections_added = post_snap.section_count - pre_snap.section_count
        if sections_added > 0:
            event = self._diff(pre_snap, post_snap)
            event["compactions_detected"] = sections_added
            self._events.append(event)
            self._persist_event(event)

        return result

    def _persist_event(self, event: dict) -> None:
        ts = event["timestamp"].replace(":", "-")[:19]
        path = self.log_dir / f"drift_{self.thread_id}_{ts}.json"
        path.write_text(json.dumps(event, indent=2))

    # ------------------------------------------------------------------ #

    def drift_report(self) -> dict:
        """Return a summary of all detected compaction events and drift metrics."""
        if not self._events:
            return {
                "thread_id": self.thread_id,
                "compaction_events": 0,
                "status": "no_compaction_detected",
            }

        ghost_scores = [e["ghost_retention"] for e in self._events]
        semantic_scores = [e["semantic_overlap"] for e in self._events]
        return {
            "thread_id": self.thread_id,
            "compaction_events": len(self._events),
            "avg_ghost_retention": round(sum(ghost_scores) / len(ghost_scores), 4),
            "min_ghost_retention": round(min(ghost_scores), 4),
            "avg_semantic_overlap": round(sum(semantic_scores) / len(semantic_scores), 4),
            "min_semantic_overlap": round(min(semantic_scores), 4),
            "events": self._events,
        }

    def print_report(self) -> None:
        report = self.drift_report()
        print(f"\n{'='*60}")
        print(f"DeepAgents Drift Report — thread: {report['thread_id']}")
        print(f"Compaction events detected: {report['compaction_events']}")
        if report["compaction_events"] > 0:
            print(f"Avg ghost retention:   {report['avg_ghost_retention']:.2%}")
            print(f"Min ghost retention:   {report['min_ghost_retention']:.2%}")
            print(f"Avg semantic overlap:  {report['avg_semantic_overlap']:.2%}")
            print(f"Min semantic overlap:  {report['min_semantic_overlap']:.2%}")
        print(f"{'='*60}\n")
