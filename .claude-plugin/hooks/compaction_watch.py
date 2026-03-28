#!/usr/bin/env python3
"""
PostToolUse hook: detect compaction events and measure behavioral drift.

Fires after every tool call. Checks the current session's JSONL log for new
'type':'summary' entries (Claude Code's compaction boundary marker). When a new
compaction is detected, runs drift analysis on the pre/post windows and emits
a warning to stderr if the composite drift score exceeds the threshold.

State persisted to .claude/compression-monitor.json so the hook tracks which
compaction events have already been reported.

Claude Code exposes these env vars to hooks:
  CLAUDE_SESSION_ID      - current session ID
  CLAUDE_PROJECT_DIR     - project root directory
  CLAUDE_PLUGIN_ROOT     - directory this plugin was loaded from
"""
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

THRESHOLD = float(os.environ.get("CM_DRIFT_THRESHOLD", "0.35"))
STATE_FILENAME = ".claude/compression-monitor.json"
PROJECTS_DIR = Path("~/.claude/projects").expanduser()


# ---------------------------------------------------------------------------
# Minimal drift instruments (no external deps)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the a an and or not in on at to of is are was were be been being "
    "have has had do does did will would could should may might must can "
    "i you he she it we they this that these those my your his her its our "
    "their what which who when where how all some any if but so just like "
    "also only then from with into about up out as by for".split()
)


def _tokenize(text):
    return [t for t in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
            if t not in _STOPWORDS]


def _extract_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return " ".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _extract_tools(message):
    content = message.get("content", [])
    if isinstance(content, str):
        return []
    return [b.get("name", "") for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use"]


def _ghost_decay(pre_texts, post_texts):
    pre_tokens = Counter(t for s in pre_texts for t in _tokenize(s))
    post_tokens = set(t for s in post_texts for t in _tokenize(s))
    top50 = {t for t, _ in pre_tokens.most_common(50)}
    precise = {t for t, c in pre_tokens.items() if c >= 2 and t not in top50}
    if not precise:
        precise = set(pre_tokens.keys())
    if not precise:
        return 0.0
    return 1.0 - len(precise & post_tokens) / len(precise)


def _tool_shift(pre_tools, post_tools):
    a, b = set(pre_tools), set(post_tools)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - len(a & b) / len(a | b)


def _sem_drift(pre_texts, post_texts):
    a = set(t for s in pre_texts for t in _tokenize(s))
    b = set(t for s in post_texts for t in _tokenize(s))
    if not a or not b:
        return 0.0
    return 1.0 - len(a & b) / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Session log finder
# ---------------------------------------------------------------------------

def _find_session_log(session_id):
    """Search ~/.claude/projects/**/<session_id>.jsonl"""
    if not PROJECTS_DIR.exists():
        return None
    for f in PROJECTS_DIR.rglob(f"{session_id}.jsonl"):
        return f
    # Fallback: any recently-modified file
    candidates = sorted(PROJECTS_DIR.rglob("*.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_entries(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Read hook input (not used for logic but consume stdin)
    try:
        json.load(sys.stdin)
    except Exception:
        pass

    session_id = os.environ.get("CLAUDE_SESSION_ID", "")
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))

    state_path = project_dir / STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # Load persisted state
    state = {"seen_compactions": 0, "last_drift": None}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            pass

    # Find session log
    log_path = _find_session_log(session_id) if session_id else None
    if not log_path:
        sys.exit(0)  # Can't find log — silent pass

    entries = _load_entries(log_path)

    # Find all compaction boundaries
    compactions = [i for i, e in enumerate(entries) if e.get("type") == "summary"]
    seen = state.get("seen_compactions", 0)

    if len(compactions) <= seen:
        sys.exit(0)  # No new compaction events

    # New compaction detected — analyze the most recent boundary
    boundary_idx = compactions[seen]
    pre_window = entries[max(0, boundary_idx - 50):boundary_idx]
    post_window = entries[boundary_idx + 1: boundary_idx + 51]

    pre_texts = [_extract_text(e.get("message", {})) for e in pre_window if e.get("type") == "assistant"]
    post_texts = [_extract_text(e.get("message", {})) for e in post_window if e.get("type") == "assistant"]
    pre_tools = [t for e in pre_window if e.get("type") == "assistant" for t in _extract_tools(e.get("message", {}))]
    post_tools = [t for e in post_window if e.get("type") == "assistant" for t in _extract_tools(e.get("message", {}))]

    decay = _ghost_decay(pre_texts, post_texts)
    shift = _tool_shift(pre_tools, post_tools)
    drift = _sem_drift(pre_texts, post_texts)
    composite = (decay + shift + drift) / 3.0

    summary_text = entries[boundary_idx].get("summary", "")[:80]

    result = {
        "seen_compactions": len(compactions),
        "last_drift": {
            "ghost_decay": round(decay, 3),
            "tool_shift": round(shift, 3),
            "semantic_drift": round(drift, 3),
            "composite": round(composite, 3),
            "summary": summary_text,
            "alert": composite > THRESHOLD,
        }
    }
    state_path.write_text(json.dumps(result, indent=2))

    if composite > THRESHOLD:
        print(
            f"\n⚠️  compression-monitor: drift detected after compaction "
            f"(composite={composite:.2f}, threshold={THRESHOLD})\n"
            f"   ghost_decay={decay:.2f}  tool_shift={shift:.2f}  semantic_drift={drift:.2f}\n"
            f"   Compaction summary: \"{summary_text}\"\n"
            f"   Behavioral fingerprint shifted — key context may need reinsertion.\n"
            f"   See .claude/compression-monitor.json for details.\n",
            file=sys.stderr,
        )
    else:
        print(
            f"\n✓  compression-monitor: compaction detected, drift within bounds "
            f"(composite={composite:.2f})\n",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
