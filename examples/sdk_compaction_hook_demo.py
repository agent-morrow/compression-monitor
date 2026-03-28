"""
sdk_compaction_hook_demo.py
============================
Reference implementation for the proposed claude-agent-sdk-python
compaction lifecycle hooks (Issue #772).

This demo shows how OnCompaction + OnContextThreshold hooks would
integrate with compression-monitor's behavioral drift instruments.

Until the SDK ships native hooks, this file also includes a polling-based
workaround that infers compaction boundaries from turn-level token metadata.

Usage (once SDK #772 ships):
    python sdk_compaction_hook_demo.py

Workaround usage (today, requires token usage in ResultMessage):
    python sdk_compaction_hook_demo.py --polling

Reference: https://github.com/anthropics/claude-agent-sdk-python/issues/772
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# compression-monitor instruments
# (adjust import paths as needed for your layout)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ghost_lexicon import extract_vocabulary, compute_ghost_terms
from behavioral_footprint import extract_footprint, compute_footprint_delta


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CompactionEvent:
    turn: int
    tokens_before: int
    tokens_after: int
    timestamp: float = field(default_factory=time.time)
    compression_ratio: float = 0.0

    def __post_init__(self):
        if self.tokens_before > 0:
            self.compression_ratio = 1.0 - (self.tokens_after / self.tokens_before)


@dataclass
class SessionSnapshot:
    turn: int
    tokens_used: int
    output_text: str
    tool_calls: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compaction monitor state
# ---------------------------------------------------------------------------

class CompactionMonitor:
    """
    Tracks compaction events and measures behavioral drift across boundaries.

    With native SDK hooks (Issue #772):
        - OnCompaction fires exactly when compaction occurs
        - OnContextThreshold fires before compaction, allowing pre-snapshot

    Without native hooks (polling workaround):
        - Detects compaction by watching for token count drops between turns
    """

    def __init__(self, threshold: float = 0.75, log_path: Optional[Path] = None):
        self.threshold = threshold
        self.log_path = log_path or Path("compaction_events.jsonl")
        self.events: list[CompactionEvent] = []
        self.snapshots: list[SessionSnapshot] = []
        self._pre_compaction_snapshot: Optional[SessionSnapshot] = None
        self._last_tokens: int = 0

    # --- Native hook callbacks (proposed API) ---

    async def on_context_threshold(self, hook_input: dict, hook_context) -> None:
        """
        Fires when context usage crosses self.threshold.
        Use this to snapshot pre-compaction state for drift comparison.
        """
        current_tokens = hook_input.get("current_tokens", 0)
        max_tokens = hook_input.get("max_tokens", 200_000)
        fraction = hook_input.get("fraction", current_tokens / max_tokens if max_tokens else 0)

        print(f"[monitor] Context at {fraction:.1%} ({current_tokens:,}/{max_tokens:,} tokens) — snapshotting pre-compaction state")

        # Snapshot will be used by on_compaction for drift comparison
        if self.snapshots:
            self._pre_compaction_snapshot = self.snapshots[-1]

    async def on_compaction(self, hook_input: dict, hook_context) -> None:
        """
        Fires when Claude performs context compaction.
        Measures behavioral drift vs. pre-compaction snapshot.
        """
        tokens_before = hook_input.get("tokens_before", self._last_tokens)
        tokens_after = hook_input.get("tokens_after", 0)
        turn = hook_input.get("turn_number", len(self.snapshots))

        event = CompactionEvent(
            turn=turn,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        self.events.append(event)
        self._last_tokens = tokens_after

        print(f"\n[monitor] COMPACTION at turn {turn}")
        print(f"  tokens: {tokens_before:,} -> {tokens_after:,} ({event.compression_ratio:.1%} reduced)")

        if self._pre_compaction_snapshot and len(self.snapshots) >= 2:
            self._measure_drift(event)

        # Log to file
        self._log_event(event)

    # --- Polling workaround (infers compaction from token drops) ---

    def observe_turn(self, turn: int, tokens_used: int, output_text: str, tool_calls: list[str] = None) -> bool:
        """
        Call after each turn with current token usage.
        Returns True if a compaction boundary was detected.
        """
        snapshot = SessionSnapshot(
            turn=turn,
            tokens_used=tokens_used,
            output_text=output_text,
            tool_calls=tool_calls or [],
        )
        self.snapshots.append(snapshot)

        compaction_detected = False

        # Detect compaction: token count dropped significantly between turns
        if self._last_tokens > 0 and tokens_used < self._last_tokens * 0.85:
            drop = self._last_tokens - tokens_used
            event = CompactionEvent(
                turn=turn,
                tokens_before=self._last_tokens,
                tokens_after=tokens_used,
            )
            self.events.append(event)

            print(f"\n[monitor] COMPACTION INFERRED at turn {turn} (token drop: {drop:,})")
            print(f"  {self._last_tokens:,} -> {tokens_used:,} ({event.compression_ratio:.1%} reduced)")

            if len(self.snapshots) >= 2:
                self._measure_drift(event)

            self._log_event(event)
            compaction_detected = True

        self._last_tokens = tokens_used
        return compaction_detected

    # --- Drift measurement ---

    def _measure_drift(self, event: CompactionEvent) -> None:
        """Compare behavioral fingerprint before and after compaction."""
        if not self._pre_compaction_snapshot and len(self.snapshots) < 2:
            return

        pre = self._pre_compaction_snapshot or self.snapshots[-2]
        post = self.snapshots[-1]

        # Ghost lexicon: terms present before, absent after
        pre_vocab = extract_vocabulary([pre.output_text])
        post_vocab = extract_vocabulary([post.output_text])
        ghosts = compute_ghost_terms(pre_vocab, post_vocab)

        # Behavioral footprint delta
        pre_fp = extract_footprint([pre])
        post_fp = extract_footprint([post])
        delta = compute_footprint_delta(pre_fp, post_fp)

        print(f"  ghost terms ({len(ghosts)}): {list(ghosts)[:8]}")
        print(f"  footprint delta: response_length={delta.get('response_length_delta', 'n/a')}, "
              f"tool_ratio={delta.get('tool_ratio_delta', 'n/a')}")

        self._pre_compaction_snapshot = None

    def _log_event(self, event: CompactionEvent) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps({
                "turn": event.turn,
                "tokens_before": event.tokens_before,
                "tokens_after": event.tokens_after,
                "compression_ratio": round(event.compression_ratio, 4),
                "timestamp": event.timestamp,
            }) + "\n")


# ---------------------------------------------------------------------------
# Demo: native hook API (proposed, requires SDK #772)
# ---------------------------------------------------------------------------

async def demo_with_native_hooks():
    """
    Shows how compression-monitor would integrate with native compaction hooks.
    This code path requires the OnCompaction + OnContextThreshold hook types
    proposed in https://github.com/anthropics/claude-agent-sdk-python/issues/772
    """
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions
    except ImportError:
        print("claude-agent-sdk not installed. Run: pip install claude-agent-sdk")
        return

    monitor = CompactionMonitor(threshold=0.75)

    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant working on a long research task.",
        hooks={
            # Existing hooks (already in SDK)
            # "PreToolUse": [...],

            # Proposed hooks (Issue #772) -- not yet in SDK
            "OnContextThreshold": [monitor.on_context_threshold],
            "OnCompaction": [monitor.on_compaction],
        },
        context_threshold=0.75,  # proposed: fraction of window
    )

    print("Starting long session with native compaction hooks...")
    print("(OnCompaction and OnContextThreshold hooks are proposed in SDK #772)\n")

    async for message in query(
        prompt="Walk me through a comprehensive analysis of distributed systems failure modes. "
               "Cover network partitions, Byzantine faults, split-brain scenarios, and recovery patterns. "
               "Be thorough -- include real-world examples, historical incidents, and mitigation strategies.",
        options=options,
    ):
        pass

    print(f"\nSession complete. Compaction events: {len(monitor.events)}")
    for ev in monitor.events:
        print(f"  Turn {ev.turn}: {ev.tokens_before:,} -> {ev.tokens_after:,} ({ev.compression_ratio:.1%} reduced)")


# ---------------------------------------------------------------------------
# Demo: polling workaround (works today with current SDK)
# ---------------------------------------------------------------------------

async def demo_with_polling():
    """
    Polling-based compaction detection using current SDK (no hook support needed).
    Infers compaction from token count drops between turns.
    """
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock
    except ImportError:
        print("claude-agent-sdk not installed. Run: pip install claude-agent-sdk")
        return

    monitor = CompactionMonitor()
    turn = 0
    current_tokens = 0

    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant.",
        max_turns=50,
    )

    print("Starting session with polling-based compaction detection...\n")

    async for message in query(
        prompt="Explain the history of the internet in exhaustive detail.",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            output_text = " ".join(
                block.text for block in message.content
                if isinstance(block, TextBlock)
            )
            tool_calls = [
                block.name for block in message.content
                if hasattr(block, "name")
            ]

            # Extract token count from usage if available
            usage = getattr(message, "usage", None)
            if usage:
                current_tokens = getattr(usage, "input_tokens", current_tokens)

            turn += 1
            monitor.observe_turn(turn, current_tokens, output_text, tool_calls)

        elif isinstance(message, ResultMessage):
            print(f"\nSession ended. Total turns: {turn}")
            print(f"Compaction events detected: {len(monitor.events)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compaction hook demo for compression-monitor")
    parser.add_argument("--polling", action="store_true",
                        help="Use polling workaround instead of native hooks (works today)")
    args = parser.parse_args()

    if args.polling:
        asyncio.run(demo_with_polling())
    else:
        print("Running native hook demo (requires SDK #772 to be implemented).")
        print("Use --polling for the workaround that works with the current SDK.\n")
        asyncio.run(demo_with_native_hooks())
