"""
semantic_kernel_integration.py
───────────────────────────────
Behavioral fingerprinting for Semantic Kernel agents at chat-history reduction boundaries.

SK's ChatHistorySummarizationReducer (and TruncationReducer) compact the chat history
when it exceeds a token or message threshold. When that happens silently, agents can
drift — losing constraints, prior decisions, or task framing — with no observable signal
from the task-completion metrics that normally gate production reliability.

This adapter wraps SK's ChatHistory and intercepts reduction events to snapshot and
compare behavioral fingerprints before and after each compaction.

Usage
─────
    import asyncio
    from semantic_kernel import Kernel
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel_integration import MonitoredChatHistory, BehavioralSummaryReducer

    kernel = Kernel()
    kernel.add_service(AzureChatCompletion(...))

    history = MonitoredChatHistory()
    reducer = BehavioralSummaryReducer(
        kernel=kernel,
        target_count=10,
        threshold_count=20,
    )

    # Use history + reducer in your agent loop as normal
    await reducer.reduce_if_required(history)

    report = history.monitor.report()
    if report["drift_detected"]:
        print("Drift at reduction events:", report["drift_events"])

Note: works with both ChatHistorySummarizationReducer and ChatHistoryTruncationReducer.
Tested against semantic-kernel-python >= 1.3.0.

Related issue: https://github.com/microsoft/semantic-kernel/issues/12303
Dependencies: semantic-kernel, compression_monitor (pip install compression-monitor)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

try:
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel.contents.chat_message_content import ChatMessageContent
except ImportError:
    # Allow import without SK installed (for type inspection / testing)
    ChatHistory = object  # type: ignore
    ChatMessageContent = object  # type: ignore

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
class HistorySnapshot:
    event_index: int
    message_count: int
    ghost_lexicon: Dict[str, float]
    role_distribution: Dict[str, int]   # {"user": n, "assistant": n, "system": n, ...}
    timestamp: float = field(default_factory=time.time)


@dataclass
class DriftEvent:
    event_index: int
    trigger: str                        # "summarization" | "truncation" | "manual"
    messages_before: int
    messages_dropped: int
    messages_after: int
    lexicon_overlap: float              # Jaccard over top-20 domain terms (1.0 = no drift)
    role_shift: float                   # L1 distance on normalized role distribution (0 = no shift)


class ChatHistoryMonitor:
    """
    Lightweight monitor that snapshots a ChatHistory before and after each
    reduction event and measures behavioral drift between snapshots.
    """

    def __init__(
        self,
        lexicon_drift_threshold: float = 0.6,
        role_shift_threshold: float = 0.2,
        verbose: bool = False,
    ):
        self.lexicon_drift_threshold = lexicon_drift_threshold
        self.role_shift_threshold = role_shift_threshold
        self.verbose = verbose

        self._snapshots: List[HistorySnapshot] = []
        self._drift_events: List[DriftEvent] = []
        self._event_counter = 0
        self._lexicon = GhostLexiconTracker()

    def _role_distribution(self, messages: list) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for m in messages:
            role = str(getattr(m, "role", "unknown"))
            dist[role] = dist.get(role, 0) + 1
        return dist

    def _extract_text(self, messages: list) -> str:
        parts = []
        for m in messages:
            content = getattr(m, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for c in content:
                    text = getattr(c, "text", None) or getattr(c, "content", None)
                    if isinstance(text, str):
                        parts.append(text)
        return " ".join(parts)

    def _jaccard(self, a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    def _role_l1(self, before: Dict[str, int], after: Dict[str, int]) -> float:
        all_roles = set(before) | set(after)
        total_b = max(sum(before.values()), 1)
        total_a = max(sum(after.values()), 1)
        return sum(
            abs(before.get(r, 0) / total_b - after.get(r, 0) / total_a)
            for r in all_roles
        )

    def snapshot_before(self, messages: list, trigger: str = "unknown") -> HistorySnapshot:
        """Call immediately before a reduction fires."""
        text = self._extract_text(messages)
        if text:
            self._lexicon.update(text)
        snap = HistorySnapshot(
            event_index=self._event_counter,
            message_count=len(messages),
            ghost_lexicon=self._lexicon.current_distribution(),
            role_distribution=self._role_distribution(messages),
        )
        self._snapshots.append(snap)
        return snap

    def snapshot_after(self, messages: list, trigger: str = "unknown") -> Optional[DriftEvent]:
        """Call immediately after a reduction fires. Returns DriftEvent if drift detected."""
        if not self._snapshots:
            return None

        self._event_counter += 1
        before = self._snapshots[-1]

        text = self._extract_text(messages)
        if text:
            self._lexicon.update(text)

        after_snap = HistorySnapshot(
            event_index=self._event_counter,
            message_count=len(messages),
            ghost_lexicon=self._lexicon.current_distribution(),
            role_distribution=self._role_distribution(messages),
        )
        self._snapshots.append(after_snap)

        top_n = 20
        before_terms = set(sorted(before.ghost_lexicon, key=before.ghost_lexicon.get, reverse=True)[:top_n])
        after_terms = set(sorted(after_snap.ghost_lexicon, key=after_snap.ghost_lexicon.get, reverse=True)[:top_n])
        lexicon_overlap = self._jaccard(before_terms, after_terms)
        role_shift = self._role_l1(before.role_distribution, after_snap.role_distribution)

        dropped = before.message_count - after_snap.message_count

        event = DriftEvent(
            event_index=self._event_counter,
            trigger=trigger,
            messages_before=before.message_count,
            messages_dropped=max(dropped, 0),
            messages_after=after_snap.message_count,
            lexicon_overlap=lexicon_overlap,
            role_shift=role_shift,
        )
        self._drift_events.append(event)

        flagged = (
            lexicon_overlap < self.lexicon_drift_threshold
            or role_shift > self.role_shift_threshold
        )

        if self.verbose or flagged:
            print(
                f"[compression-monitor] Reduction event {self._event_counter}: {trigger} "
                f"({before.message_count}→{after_snap.message_count} msgs, "
                f"dropped={max(dropped,0)}), "
                f"lexicon_overlap={lexicon_overlap:.2f}, role_shift={role_shift:.2f}"
                + (" ⚠ DRIFT FLAGGED" if flagged else "")
            )

        return event if flagged else None

    def report(self) -> Dict[str, Any]:
        flagged = [
            e for e in self._drift_events
            if (
                e.lexicon_overlap < self.lexicon_drift_threshold
                or e.role_shift > self.role_shift_threshold
            )
        ]
        return {
            "drift_detected": len(flagged) > 0,
            "reduction_events": len(self._drift_events),
            "snapshots_taken": len(self._snapshots),
            "drift_events": [
                {
                    "event_index": e.event_index,
                    "trigger": e.trigger,
                    "messages_before": e.messages_before,
                    "messages_dropped": e.messages_dropped,
                    "messages_after": e.messages_after,
                    "lexicon_overlap": round(e.lexicon_overlap, 3),
                    "role_shift": round(e.role_shift, 3),
                }
                for e in flagged
            ],
        }


class MonitoredChatHistory(ChatHistory):
    """
    Drop-in replacement for SK's ChatHistory that wraps a ChatHistoryMonitor.
    Pass instances of this to your agent and reducer as you would a normal ChatHistory.
    """

    def __init__(self, *args: Any, verbose: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.monitor = ChatHistoryMonitor(verbose=verbose)


class BehavioralSummaryReducer:
    """
    Thin wrapper around SK's ChatHistorySummarizationReducer that adds
    before/after behavioral snapshots at each reduction boundary.

    Usage:
        reducer = BehavioralSummaryReducer(kernel=kernel, target_count=10, threshold_count=20)
        reduced = await reducer.reduce_if_required(history)
        report = history.monitor.report()
    """

    def __init__(
        self,
        kernel: Any,
        target_count: int = 10,
        threshold_count: int = 20,
        **reducer_kwargs: Any,
    ):
        try:
            from semantic_kernel.agents.strategies.selection.chat_history_reducer import (
                ChatHistorySummarizationReducer,
            )
        except ImportError:
            try:
                from semantic_kernel.memory.chat_history_reducer import (
                    ChatHistorySummarizationReducer,
                )
            except ImportError:
                raise ImportError(
                    "Could not import ChatHistorySummarizationReducer from semantic_kernel. "
                    "Check your semantic-kernel version."
                )

        self._reducer = ChatHistorySummarizationReducer(
            kernel=kernel,
            target_count=target_count,
            threshold_count=threshold_count,
            **reducer_kwargs,
        )
        self._target_count = target_count
        self._threshold_count = threshold_count

    async def reduce_if_required(self, history: MonitoredChatHistory) -> bool:
        """
        Mirrors the SK reducer's reduce_if_required interface, adding behavioral
        snapshots around the actual reduction call.

        Returns True if reduction occurred, False otherwise.
        """
        msgs = list(history.messages)
        if len(msgs) < self._threshold_count:
            return False

        # Snapshot before
        history.monitor.snapshot_before(msgs, trigger="summarization")

        # Delegate to SK reducer
        reduced = await self._reducer.reduce_if_required(history)

        if reduced:
            # Snapshot after and check for drift
            history.monitor.snapshot_after(list(history.messages), trigger="summarization")

        return reduced


class BehavioralTruncationReducer:
    """
    Thin wrapper around SK's ChatHistoryTruncationReducer.
    Same interface as BehavioralSummaryReducer.
    """

    def __init__(self, target_count: int = 10, threshold_count: int = 20, **kwargs: Any):
        try:
            from semantic_kernel.agents.strategies.selection.chat_history_reducer import (
                ChatHistoryTruncationReducer,
            )
        except ImportError:
            try:
                from semantic_kernel.memory.chat_history_reducer import (
                    ChatHistoryTruncationReducer,
                )
            except ImportError:
                raise ImportError(
                    "Could not import ChatHistoryTruncationReducer from semantic_kernel."
                )

        self._reducer = ChatHistoryTruncationReducer(
            target_count=target_count,
            threshold_count=threshold_count,
            **kwargs,
        )
        self._threshold_count = threshold_count

    async def reduce_if_required(self, history: MonitoredChatHistory) -> bool:
        msgs = list(history.messages)
        if len(msgs) < self._threshold_count:
            return False

        history.monitor.snapshot_before(msgs, trigger="truncation")
        reduced = await self._reducer.reduce_if_required(history)

        if reduced:
            history.monitor.snapshot_after(list(history.messages), trigger="truncation")

        return reduced
