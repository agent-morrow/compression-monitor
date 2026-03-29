#!/usr/bin/env python3
"""
compression-monitor quickstart
================================
Run this script to see behavioral drift detection in action.
No external dependencies required beyond the base package.

Usage:
    pip install git+https://github.com/agent-morrow/compression-monitor
    python quickstart.py

What it demonstrates:
    1. Build a behavioral fingerprint from a session's tool-call log
    2. Simulate a context compaction event
    3. Measure drift across three instruments
    4. Interpret the results

Takes ~2 seconds to run.
"""

import json
import math
import sys
from collections import Counter

ALERT_MARKER = "[ALERT]"
OK_MARKER = "[OK]"


# ─── Inline minimal implementations (no import required) ──────────────────────

def tokenize(text):
    import re
    return re.findall(r'\b[a-z][a-z0-9_]{2,}\b', text.lower())

def low_frequency_vocab(texts, min_count=2, max_freq_ratio=0.3):
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize(t))
    total = len(all_tokens)
    counts = Counter(all_tokens)
    vocab = {w for w, c in counts.items()
             if c >= min_count and c / total <= max_freq_ratio}
    return vocab

def ghost_lexicon_decay(pre_texts, post_texts):
    pre_vocab = low_frequency_vocab(pre_texts)
    if not pre_vocab:
        return 0.0
    post_tokens = set()
    for t in post_texts:
        post_tokens.update(tokenize(t))
    survivors = pre_vocab & post_tokens
    return 1.0 - len(survivors) / len(pre_vocab)

def tool_call_jaccard(pre_log, post_log):
    def seq(log):
        return Counter(e.get('tool', e.get('type', '')) for e in log)
    pre = seq(pre_log)
    post = seq(post_log)
    keys = set(pre) | set(post)
    if not keys:
        return 0.0
    intersection = sum(min(pre[k], post[k]) for k in keys)
    union = sum(max(pre[k], post[k]) for k in keys)
    return 1.0 - intersection / union if union else 0.0

def vocab_overlap(texts_a, texts_b):
    def vocab(texts):
        tokens = set()
        for t in texts:
            tokens.update(tokenize(t))
        return tokens
    a, b = vocab(texts_a), vocab(texts_b)
    if not a or not b:
        return 1.0
    return 1.0 - len(a & b) / len(a | b)


# ─── Scenario: debugging session with and without context compaction ───────────

PRE_COMPACTION_OUTPUTS = [
    """
    Investigating the authentication failure. The stack trace shows:
    AttributeError in jwt_validator.py line 142: decode_token received None.
    Root cause traced to session_manager.py get_user_context returning empty dict
    when Redis cache TTL expires. The jwt_validator assumes non-null context
    but does not guard against None. Need to fix session_manager.get_user_context
    to return a default context object on cache miss, not an empty dict.
    """,
    """
    Read jwt_validator.py: confirmed no null guard on line 142.
    Read session_manager.py: get_user_context has no fallback for Redis miss.
    Ran test_auth.py with Redis disabled: reproduced AttributeError.
    Hypothesis confirmed: Redis cache miss → empty dict → None propagation → crash.
    Fix: add default_context = UserContext(anonymous=True) in session_manager.py.
    """,
    """
    Checking Redis TTL configuration. Current TTL is 300s. During load tests,
    sessions expiring under high concurrency cause cascading null propagation.
    The jwt_validator crash is a symptom. The root cause is session_manager
    not treating cache miss as a first-class case. Fix location: session_manager.py.
    """
]

PRE_COMPACTION_TOOL_LOG = [
    {"tool": "Read",  "path": "jwt_validator.py"},
    {"tool": "Read",  "path": "session_manager.py"},
    {"tool": "Read",  "path": "auth_middleware.py"},
    {"tool": "Bash",  "cmd": "python -m pytest test_auth.py -x 2>&1 | head -30"},
    {"tool": "Bash",  "cmd": "grep -n 'get_user_context' session_manager.py"},
    {"tool": "Bash",  "cmd": "redis-cli TTL session:test123"},
    {"tool": "Read",  "path": "config/redis.yaml"},
]

# Post-compaction: the session was summarized as "debugging auth bug"
# The specific error vocabulary and the constraint (fix location, root cause)
# are no longer in active context. The agent reverts toward guess-and-edit.

POST_COMPACTION_OUTPUTS = [
    """
    Continuing the authentication debugging. Will try adding some error handling
    to fix the issue. The authentication module seems to have a problem.
    """,
    """
    Added try/except block to handle the authentication error. Let me check
    if this resolves the problem by running the tests.
    """,
    """
    Tests still failing. Will try a different approach to fix the auth issue.
    Maybe the problem is in how we handle the user session.
    """
]

POST_COMPACTION_TOOL_LOG = [
    {"tool": "Read",   "path": "auth_middleware.py"},
    {"tool": "Edit",   "path": "auth_middleware.py"},
    {"tool": "Bash",   "cmd": "python -m pytest test_auth.py 2>&1 | tail -5"},
    {"tool": "Edit",   "path": "auth_middleware.py"},
    {"tool": "Bash",   "cmd": "python -m pytest test_auth.py 2>&1 | tail -5"},
]


# ─── Run measurements ──────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("compression-monitor quickstart")
    print("=" * 64)
    print()
    print("Scenario: Claude Code debugging session, authentication bug")
    print("Measuring behavioral drift across a context compaction event")
    print()

    ghost = ghost_lexicon_decay(PRE_COMPACTION_OUTPUTS, POST_COMPACTION_OUTPUTS)
    tool  = tool_call_jaccard(PRE_COMPACTION_TOOL_LOG, POST_COMPACTION_TOOL_LOG)
    sem   = vocab_overlap(PRE_COMPACTION_OUTPUTS, POST_COMPACTION_OUTPUTS)
    composite = (ghost + tool + sem) / 3

    threshold = 0.35

    print("Instrument results:")
    print(f"  Ghost lexicon decay   {ghost:.2f}  {ALERT_MARKER if ghost > threshold else OK_MARKER}")
    print(f"  Tool-call shift       {tool:.2f}  {ALERT_MARKER if tool > threshold else OK_MARKER}")
    print(f"  Semantic drift        {sem:.2f}  {ALERT_MARKER if sem > threshold else OK_MARKER}")
    print("  " + "-" * 25)
    print(f"  Composite score       {composite:.2f}  {'[ALERT] composite drift' if composite > threshold else '[OK] stable'}")
    print()

    if composite > threshold:
        print("Interpretation:")
        print(f"  Ghost lexicon decay {ghost:.0%}: {ghost:.0%} of the domain-specific")
        print("  vocabulary from the pre-compaction phase (jwt_validator, decode_token,")
        print("  session_manager, get_user_context, Redis TTL, null propagation) did")
        print("  not appear in post-compaction outputs. The agent lost the specific")
        print("  vocabulary that pinned the investigation.")
        print()
        print(f"  Tool-call shift {tool:.0%}: Pre-compaction sequence was")
        print("  [Read×4, Bash×3] — investigation-heavy. Post-compaction sequence")
        print("  shifted to [Read×1, Edit×2, Bash×2] — guess-and-edit pattern.")
        print("  This matches the Pattern 8 debugging spiral.")
        print()
        print(f"  Recommended action: reinject investigation context before")
        print("  continuing. Key constraint: fix is in session_manager.py,")
        print("  not auth_middleware.py.")
    else:
        print("No significant drift detected — behavioral fingerprint stable.")

    print()
    print("-" * 64)
    print("To monitor your own Claude Code sessions:")
    print()
    print("  1. Copy .claude-plugin/ from the repo into your project:")
    print("     https://github.com/agent-morrow/compression-monitor")
    print()
    print("  2. The PostToolUse hook will detect compaction events and")
    print("     alert when the behavioral fingerprint shifts.")
    print()
    print("  3. For multi-agent frameworks (CrewAI, LangGraph, AutoGen):")
    print("     from compression_monitor.integrations import MonitoredCrew")
    print()


if __name__ == "__main__":
    main()
