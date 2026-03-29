"""
mem0 behavioral noise detector for compression-monitor.

Measures behavioral drift caused by hallucinated or junk memory injection.
When a memory store contains noisy entries (as documented in mem0ai/mem0#4573),
those memories shift the agent's behavioral output in detectable ways.

This module provides two approaches:
1. Baseline comparison: compare agent behavior with and without memories active
2. Session-to-session drift: track behavioral consistency across sessions
   that use the same memory store

Usage:
    detector = Mem0NoiseDetector()

    # Run a baseline session WITHOUT memories (clean context)
    baseline = detector.run_baseline("clean context here")

    # Run production session WITH memories
    production = detector.run_with_memories("same prompt + memory context here")

    # Compare
    report = detector.compare(baseline, production)
    if report["noise_score"] > 0.3:
        print("Memory noise detected — schedule an audit")
        print("Suspected noise terms:", report["noise_terms"])
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Core fingerprinting utilities (no external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    import re
    return re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", text.lower())


def _build_fingerprint(outputs: list[str]) -> dict[str, Any]:
    """
    Build a behavioral fingerprint from a list of agent output strings.
    """
    all_tokens: list[str] = []
    for out in outputs:
        all_tokens.extend(_tokenize(out))
    if not all_tokens:
        return {"term_freq": {}, "low_freq_vocab": [], "total_tokens": 0}
    freq = Counter(all_tokens)
    total = len(all_tokens)
    low_freq_vocab = [t for t, c in freq.items() if 1 <= c <= 3]
    return {
        "term_freq": dict(freq.most_common(150)),
        "low_freq_vocab": low_freq_vocab,
        "total_tokens": total,
    }


def _jaccard_drift(fp_a: dict, fp_b: dict) -> float:
    """
    Compute 1 - Jaccard similarity between two fingerprints.
    0 = identical, 1 = completely different.
    """
    set_a = set(fp_a.get("term_freq", {}).keys())
    set_b = set(fp_b.get("term_freq", {}).keys())
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return round(1.0 - len(set_a & set_b) / len(union), 4)


def _noise_terms(
    baseline_fp: dict,
    memory_fp: dict,
    conversation_context: list[str] | None = None,
) -> list[str]:
    """
    Find terms that appear in memory-augmented outputs but not in baseline.
    These are candidates for hallucinated memory injection.

    If conversation_context is provided, also filters out terms that appear
    in the actual conversation (i.e., non-hallucinated additions).
    """
    baseline_terms = set(baseline_fp.get("term_freq", {}).keys())
    memory_terms = set(memory_fp.get("term_freq", {}).keys())
    injected = memory_terms - baseline_terms

    if conversation_context:
        # Filter out terms that appear in the actual conversation
        conv_tokens = set(_tokenize(" ".join(conversation_context)))
        # Keep only terms NOT in the conversation (likely from memory)
        injected = injected - conv_tokens

    # Focus on injected terms that are in the memory_fp's low-freq set
    # (precise terms suggest specific hallucinated facts, not stop words)
    memory_low_freq = set(memory_fp.get("low_freq_vocab", []))
    precise_noise = sorted(injected & memory_low_freq)
    general_noise = sorted(injected - memory_low_freq)

    return precise_noise[:10] + general_noise[:10]


# ---------------------------------------------------------------------------
# Mem0NoiseDetector
# ---------------------------------------------------------------------------

class Mem0NoiseDetector:
    """
    Detects behavioral noise caused by hallucinated or junk mem0 entries.

    The key insight: if the agent's behavioral output contains terms that
    (a) do not appear in the baseline conversation, but
    (b) do appear when memories are injected into context,
    those terms are likely coming from hallucinated memory entries.

    This is a continuous, cheap complement to periodic content audits.
    """

    def __init__(self, state_dir: str = ".mem0_noise_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)

    def record_session(
        self,
        session_id: str,
        outputs: list[str],
        memories_active: bool = True,
        conversation_turns: list[str] | None = None,
    ) -> dict:
        """
        Record a session's behavioral fingerprint.

        Args:
            session_id: Unique identifier for this session
            outputs: List of agent output strings from this session
            memories_active: Whether mem0 memories were injected into context
            conversation_turns: The actual conversation (used to exclude non-noise terms)

        Returns:
            Session record saved to disk
        """
        fp = _build_fingerprint(outputs)
        record = {
            "session_id": session_id,
            "memories_active": memories_active,
            "recorded_at": time.time(),
            "fingerprint": fp,
            "conversation_tokens": list(set(_tokenize(" ".join(conversation_turns or [])))),
        }
        path = self.state_dir / f"{session_id}.json"
        path.write_text(json.dumps(record, indent=2))
        return record

    def compare_sessions(
        self,
        baseline_session_id: str,
        memory_session_id: str,
        noise_threshold: float = 0.25,
    ) -> dict:
        """
        Compare a baseline session (no memories) against a memory-augmented session.

        Args:
            baseline_session_id: Session ID recorded without mem0 active
            memory_session_id: Session ID recorded with mem0 active
            noise_threshold: Drift score above which noise warning is raised

        Returns:
            Noise report with noise_score, noise_terms, and recommendation
        """
        baseline_path = self.state_dir / f"{baseline_session_id}.json"
        memory_path = self.state_dir / f"{memory_session_id}.json"

        if not baseline_path.exists():
            return {"status": "error", "message": f"Baseline session '{baseline_session_id}' not found"}
        if not memory_path.exists():
            return {"status": "error", "message": f"Memory session '{memory_session_id}' not found"}

        baseline = json.loads(baseline_path.read_text())
        memory_session = json.loads(memory_path.read_text())

        baseline_fp = baseline["fingerprint"]
        memory_fp = memory_session["fingerprint"]

        conv_tokens = baseline.get("conversation_tokens", []) + memory_session.get("conversation_tokens", [])
        conv_list = list(set(conv_tokens))

        score = _jaccard_drift(baseline_fp, memory_fp)
        noise = _noise_terms(baseline_fp, memory_fp, conv_list)

        report = {
            "baseline_session": baseline_session_id,
            "memory_session": memory_session_id,
            "noise_score": score,
            "noise_terms": noise,
            "baseline_tokens": baseline_fp["total_tokens"],
            "memory_session_tokens": memory_fp["total_tokens"],
        }

        if score > noise_threshold:
            report["status"] = "warning"
            report["recommendation"] = (
                f"Noise score {score:.2f} exceeds threshold {noise_threshold}. "
                f"Suspected memory-injected terms: {noise[:5]}. "
                "Consider auditing mem0 entries containing these terms. "
                "Run mem0.search_memory() with each term to identify the source entries."
            )
        else:
            report["status"] = "clean"
            report["recommendation"] = (
                f"Noise score {score:.2f} within threshold. "
                "Memory injection does not appear to be adding anomalous behavioral terms."
            )

        return report

    def rolling_drift_check(
        self,
        session_ids: list[str],
        window: int = 3,
        threshold: float = 0.3,
    ) -> list[dict]:
        """
        Check drift across a rolling window of sessions (all with memories active).
        High drift between consecutive sessions suggests memory quality degraded.

        Args:
            session_ids: Ordered list of session IDs
            window: Number of sessions to compare each time
            threshold: Drift score above which to flag a window

        Returns:
            List of flagged windows with their drift scores
        """
        flags = []
        for i in range(len(session_ids) - window + 1):
            window_ids = session_ids[i : i + window]
            fps = []
            for sid in window_ids:
                path = self.state_dir / f"{sid}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                    fps.append(record["fingerprint"])

            if len(fps) < 2:
                continue

            # Compare first and last in window
            score = _jaccard_drift(fps[0], fps[-1])
            if score > threshold:
                flags.append({
                    "window": window_ids,
                    "drift_score": score,
                    "sessions_compared": (window_ids[0], window_ids[-1]),
                    "recommendation": f"Drift {score:.2f} across {window} sessions — memory quality may have degraded",
                })

        return flags


# ---------------------------------------------------------------------------
# Quick diagnostic: compare two sets of outputs inline (no disk state)
# ---------------------------------------------------------------------------

def quick_noise_check(
    baseline_outputs: list[str],
    memory_outputs: list[str],
    conversation_context: list[str] | None = None,
) -> dict:
    """
    Inline noise check without persisting state.

    Args:
        baseline_outputs: Agent outputs without memories active
        memory_outputs: Agent outputs with memories active
        conversation_context: The actual conversation turns (optional)

    Returns:
        Noise report dict
    """
    baseline_fp = _build_fingerprint(baseline_outputs)
    memory_fp = _build_fingerprint(memory_outputs)
    score = _jaccard_drift(baseline_fp, memory_fp)
    noise = _noise_terms(baseline_fp, memory_fp, conversation_context)

    return {
        "noise_score": score,
        "noise_terms": noise,
        "baseline_tokens": baseline_fp["total_tokens"],
        "memory_tokens": memory_fp["total_tokens"],
        "interpretation": (
            "High score + unrecognized noise terms = likely hallucinated memory injection. "
            "Use mem0.search_memory(term) to find and audit the source entries."
            if score > 0.25
            else "Noise score within expected range."
        ),
    }
