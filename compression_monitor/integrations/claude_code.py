"""
compression_monitor/integrations/claude_code.py

Session-boundary drift monitoring for Claude Code JSONL logs.

Claude Code stores session logs at ~/.claude/projects/<project-hash>/<session-id>.jsonl
Each line is one of:
  - {"type": "user",      "message": {"role": "user", "content": "..."},     "timestamp": ..., "sessionId": ...}
  - {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "..."}, ...]}, ...}
  - {"type": "summary",   "summary": "...",  "sessionId": ...}   <- compaction boundary

Usage:
    from compression_monitor.integrations.claude_code import ClaudeCodeSession
    session = ClaudeCodeSession.from_file("~/.claude/projects/.../session.jsonl")
    report = session.drift_report()
    print(report.summary())

    # Or from CLI:
    python -m compression_monitor.integrations.claude_code ~/.claude/projects/.../session.jsonl
"""

from __future__ import annotations
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path) -> list[dict]:
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return lines


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_text(message: dict) -> str:
    """Extract all text content from a message dict."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_result":
                c = block.get("content", "")
                if isinstance(c, str):
                    texts.append(c)
                elif isinstance(c, list):
                    for sub in c:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            texts.append(sub.get("text", ""))
    return " ".join(texts)


def _extract_tool_calls(message: dict) -> list[str]:
    """Extract list of tool names from an assistant message."""
    content = message.get("content", [])
    if isinstance(content, str):
        return []
    return [
        block.get("name", "unknown")
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


# ---------------------------------------------------------------------------
# Tokenization (minimal, no deps)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "the a an and or not in on at to of is are was were be been being "
    "have has had do does did will would could should may might must can "
    "i you he she it we they this that these those my your his her its our "
    "their what which who when where how all some any if but so just like "
    "also only then from with into about up out as by for".split()
)


def _tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())
        if t not in _STOPWORDS
    ]


def _ghost_lexicon_decay(pre_texts: list[str], post_texts: list[str]) -> float:
    """Return fraction of pre-compaction vocabulary absent post-compaction."""
    pre_all = [t for s in pre_texts for t in _tokenize(s)]
    post_all = set(t for s in post_texts for t in _tokenize(s))
    if not pre_all:
        return 0.0
    pre_counter = Counter(pre_all)
    # Low-frequency but present vocabulary (appears ≥2 times, not the top 50)
    top50 = {t for t, _ in pre_counter.most_common(50)}
    precise = {t for t, c in pre_counter.items() if c >= 2 and t not in top50}
    if not precise:
        # Fall back to all pre-vocabulary
        precise = set(pre_counter.keys())
    surviving = precise & post_all
    return 1.0 - len(surviving) / len(precise)


def _tool_call_shift(pre_tools: list[str], post_tools: list[str]) -> float:
    """Jaccard distance between pre- and post-compaction tool distributions."""
    pre_set = set(pre_tools)
    post_set = set(post_tools)
    if not pre_set and not post_set:
        return 0.0
    if not pre_set or not post_set:
        return 1.0
    intersection = pre_set & post_set
    union = pre_set | post_set
    return 1.0 - len(intersection) / len(union)


def _semantic_overlap(pre_texts: list[str], post_texts: list[str]) -> float:
    """1 - token-overlap fraction (higher = more drift)."""
    pre_tokens = set(t for s in pre_texts for t in _tokenize(s))
    post_tokens = set(t for s in post_texts for t in _tokenize(s))
    if not pre_tokens or not post_tokens:
        return 0.0
    overlap = pre_tokens & post_tokens
    return 1.0 - len(overlap) / max(len(pre_tokens), len(post_tokens))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CompactionWindow:
    """Messages before or after a compaction boundary."""
    label: str               # "pre-compaction" or "post-compaction"
    start_idx: int
    end_idx: int
    texts: list[str]         # assistant text outputs
    tool_calls: list[str]    # tool names
    summary_text: str = ""   # summary content at boundary, if any


@dataclass
class DriftReport:
    session_path: str
    total_entries: int
    compaction_count: int
    windows: list[tuple[CompactionWindow, CompactionWindow]]  # (pre, post) pairs
    ghost_lexicon_decay: float
    tool_call_shift: float
    semantic_drift: float
    alert: Optional[str] = None

    def drift_score(self) -> float:
        return (self.ghost_lexicon_decay + self.tool_call_shift + self.semantic_drift) / 3.0

    def summary(self) -> str:
        lines = [
            f"Session: {self.session_path}",
            f"  Entries: {self.total_entries}  |  Compaction events: {self.compaction_count}",
            f"  Ghost lexicon decay:  {self.ghost_lexicon_decay:.3f}  (domain vocabulary lost post-compaction)",
            f"  Tool-call shift:      {self.tool_call_shift:.3f}  (behavioral pattern change)",
            f"  Semantic drift:       {self.semantic_drift:.3f}  (topic shift)",
            f"  Composite drift:      {self.drift_score():.3f}",
        ]
        if self.alert:
            lines.append(f"  ALERT: {self.alert}")
        else:
            lines.append("  Status: within normal bounds")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ClaudeCodeSession:
    """
    Load and analyze a Claude Code JSONL session log for behavioral drift
    across compaction (context rotation) boundaries.
    """

    def __init__(self, entries: list[dict], path: str = "<stream>"):
        self.entries = entries
        self.path = path

    @classmethod
    def from_file(cls, path: str | Path) -> "ClaudeCodeSession":
        p = Path(path).expanduser()
        return cls(_load_jsonl(p), str(p))

    @classmethod
    def latest_session(cls, project_dir: str | Path | None = None) -> "ClaudeCodeSession":
        """
        Load the most recently modified session from ~/.claude/projects/.
        Optionally restrict to a project subdirectory.
        """
        base = Path(project_dir).expanduser() if project_dir else Path("~/.claude/projects").expanduser()
        candidates = sorted(base.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No .jsonl files found under {base}")
        return cls.from_file(candidates[0])

    # ------------------------------------------------------------------

    def _find_compaction_boundaries(self) -> list[int]:
        """Return indices of entries with type='summary' (compaction events)."""
        return [i for i, e in enumerate(self.entries) if e.get("type") == "summary"]

    def _make_window(self, start: int, end: int, label: str, summary_text: str = "") -> CompactionWindow:
        texts, tools = [], []
        for e in self.entries[start:end]:
            if e.get("type") == "assistant":
                msg = e.get("message", {})
                t = _extract_text(msg)
                if t.strip():
                    texts.append(t)
                tools.extend(_extract_tool_calls(msg))
        return CompactionWindow(
            label=label,
            start_idx=start,
            end_idx=end,
            texts=texts,
            tool_calls=tools,
            summary_text=summary_text,
        )

    def drift_report(self, alert_threshold: float = 0.4) -> DriftReport:
        """
        Build a DriftReport for this session.

        If there are no compaction events, splits the session in half
        (first half vs second half) as a proxy for temporal drift.
        """
        boundaries = self._find_compaction_boundaries()
        n = len(self.entries)

        if not boundaries:
            # No compaction markers — compare first vs second half
            mid = n // 2
            pre = self._make_window(0, mid, "first-half")
            post = self._make_window(mid, n, "second-half")
            windows = [(pre, post)]
            compaction_count = 0
        else:
            windows = []
            prev = 0
            for idx in boundaries:
                summary_text = self.entries[idx].get("summary", "")
                pre = self._make_window(prev, idx, "pre-compaction", "")
                post_end = boundaries[boundaries.index(idx) + 1] if boundaries.index(idx) + 1 < len(boundaries) else n
                post = self._make_window(idx + 1, post_end, "post-compaction", summary_text)
                windows.append((pre, post))
                prev = idx + 1
            compaction_count = len(boundaries)

        # Aggregate across all pre/post pairs
        all_pre_texts = [t for pre, _ in windows for t in pre.texts]
        all_post_texts = [t for _, post in windows for t in post.texts]
        all_pre_tools = [t for pre, _ in windows for t in pre.tool_calls]
        all_post_tools = [t for _, post in windows for t in post.tool_calls]

        decay = _ghost_lexicon_decay(all_pre_texts, all_post_texts)
        tool_shift = _tool_call_shift(all_pre_tools, all_post_tools)
        sem_drift = _semantic_overlap(all_pre_texts, all_post_texts)

        drift = (decay + tool_shift + sem_drift) / 3.0
        alert = None
        if drift > alert_threshold:
            reasons = []
            if decay > alert_threshold:
                reasons.append(f"ghost lexicon decay {decay:.2f}")
            if tool_shift > alert_threshold:
                reasons.append(f"tool-call shift {tool_shift:.2f}")
            if sem_drift > alert_threshold:
                reasons.append(f"semantic drift {sem_drift:.2f}")
            alert = f"Drift score {drift:.2f} exceeds threshold {alert_threshold}: " + "; ".join(reasons)

        return DriftReport(
            session_path=self.path,
            total_entries=n,
            compaction_count=compaction_count,
            windows=windows,
            ghost_lexicon_decay=decay,
            tool_call_shift=tool_shift,
            semantic_drift=sem_drift,
            alert=alert,
        )

    def tool_call_timeline(self) -> list[tuple[str, str]]:
        """Return list of (timestamp, tool_name) for the full session."""
        result = []
        for e in self.entries:
            if e.get("type") == "assistant":
                ts = e.get("timestamp", "")
                for name in _extract_tool_calls(e.get("message", {})):
                    result.append((ts, name))
        return result


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import glob

    if len(sys.argv) < 2:
        # Auto-discover latest session
        pattern = str(Path("~/.claude/projects").expanduser() / "**" / "*.jsonl")
        files = sorted(glob.glob(pattern, recursive=True), key=lambda f: Path(f).stat().st_mtime, reverse=True)
        if not files:
            print("Usage: python -m compression_monitor.integrations.claude_code <session.jsonl>")
            print("       (or run from a directory with ~/.claude/projects/ to auto-detect)")
            sys.exit(1)
        target = files[0]
        print(f"Auto-detected: {target}\n")
    else:
        target = sys.argv[1]

    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    session = ClaudeCodeSession.from_file(target)
    report = session.drift_report(alert_threshold=threshold)
    print(report.summary())

    if "--timeline" in sys.argv:
        print("\nTool call timeline:")
        for ts, name in session.tool_call_timeline():
            print(f"  {ts[:19]}  {name}")

    sys.exit(1 if report.alert else 0)
