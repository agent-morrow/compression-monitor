#!/usr/bin/env python3
"""
delegation_quality.py — Measure delegation prompt quality across compaction boundaries.

Reads a JSONL session log (Claude Code format or split pre/post files from
parse_claude_session.py) and extracts three signals per delegation event:

  1. file_path_specificity  — ratio of delegation prompts that reference at
                              least one file path (e.g. src/foo.py, ./bar/baz)
  2. constraint_density     — average count of explicit negative constraints
                              per delegation prompt ("don't", "only", "avoid",
                              "do not", "never", "except", "must not")
  3. verification_presence  — ratio of delegation prompts that include a
                              verification request ("confirm before", "verify",
                              "check with me", "ask if", "wait for approval")

Usage:
  # Compare pre/post compaction files from parse_claude_session.py:
  python delegation_quality.py --pre session_pre.jsonl --post session_post.jsonl

  # Analyze a single file (full session or any split):
  python delegation_quality.py --file session.jsonl

  # Auto-detect latest Claude Code session and split at compaction boundary:
  python delegation_quality.py --auto

Output: TSV summary + per-event table to stdout. Use --json for machine-readable output.
"""

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ── patterns ────────────────────────────────────────────────────────────────

# File path: starts with ./ or ../ or /, or has an extension after a word char
FILE_PATH_RE = re.compile(
    r'(?:^|[\s\(\[\'"])(?:\.{1,2}/[\w./\-]+|/[\w./\-]{3,}|[\w\-]+\.(?:py|ts|js|tsx|jsx|go|rs|cpp|c|h|java|rb|sh|md|json|yaml|yml|toml|lock))\b',
    re.MULTILINE
)

CONSTRAINT_KEYWORDS = [
    r"\bdon'?t\b", r"\bdo not\b", r"\bmust not\b", r"\bnever\b",
    r"\bonly\b", r"\bavoid\b", r"\bexcept\b", r"\bwithout\b",
    r"\bleave.*unchanged\b", r"\bdo not touch\b", r"\bdo not modify\b",
]
CONSTRAINT_RE = re.compile("|".join(CONSTRAINT_KEYWORDS), re.IGNORECASE)

VERIFY_RE = re.compile(
    r'\b(confirm before|verify|check with me|ask (me|if|before)|'
    r'wait for (my |your )?approval|before (you )?(proceed|act|make|modify)|'
    r'get (my |your )?sign.?off|double.?check)\b',
    re.IGNORECASE
)

# ── extraction ───────────────────────────────────────────────────────────────

@dataclass
class DelegationEvent:
    turn: int
    role: str
    tool_name: str
    prompt_text: str
    file_path_count: int = 0
    constraint_count: int = 0
    has_verification: bool = False
    char_length: int = 0

    def compute_signals(self):
        self.file_path_count = len(FILE_PATH_RE.findall(self.prompt_text))
        self.constraint_count = len(CONSTRAINT_RE.findall(self.prompt_text))
        self.has_verification = bool(VERIFY_RE.search(self.prompt_text))
        self.char_length = len(self.prompt_text)


DELEGATION_TOOLS = {
    # Claude Code subagent / Task tool names
    "Task", "task", "computer_use", "bash", "str_replace_based_edit_tool",
    # Generic agent delegation patterns
    "delegate", "spawn_agent", "call_agent", "run_agent",
}


def extract_delegation_events(jsonl_path: str) -> list[DelegationEvent]:
    """Extract delegation events from a JSONL session log."""
    events = []
    turn = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn += 1
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Content can be a string or a list of content blocks
            if isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                blocks = content
            else:
                continue

            for block in blocks:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                # Tool use block (agent delegating to a tool/subagent)
                if btype == "tool_use":
                    tool_name = block.get("name", "")
                    if tool_name in DELEGATION_TOOLS or tool_name.lower() in {t.lower() for t in DELEGATION_TOOLS}:
                        # Extract the prompt text from input
                        inp = block.get("input", {})
                        if isinstance(inp, dict):
                            prompt = inp.get("description", "") or inp.get("prompt", "") or inp.get("command", "") or str(inp)
                        else:
                            prompt = str(inp)
                        if prompt:
                            ev = DelegationEvent(turn=turn, role=role, tool_name=tool_name, prompt_text=prompt)
                            ev.compute_signals()
                            events.append(ev)
                # Also catch text blocks that look like delegation instructions
                elif btype == "text" and role == "assistant":
                    text = block.get("text", "")
                    # Heuristic: long assistant turns that reference subagent patterns
                    if len(text) > 200 and re.search(r'\b(subagent|sub-agent|delegate|spawn|Task\()', text):
                        ev = DelegationEvent(turn=turn, role=role, tool_name="text_delegation", prompt_text=text[:2000])
                        ev.compute_signals()
                        events.append(ev)
    return events


# ── summary stats ─────────────────────────────────────────────────────────

@dataclass
class QualitySummary:
    label: str
    n_events: int = 0
    file_path_specificity: float = 0.0   # fraction with >=1 file path
    mean_constraint_density: float = 0.0
    verification_presence: float = 0.0   # fraction with verification request
    mean_length: float = 0.0

    @classmethod
    def from_events(cls, label: str, events: list[DelegationEvent]) -> "QualitySummary":
        if not events:
            return cls(label=label)
        n = len(events)
        return cls(
            label=label,
            n_events=n,
            file_path_specificity=sum(1 for e in events if e.file_path_count > 0) / n,
            mean_constraint_density=sum(e.constraint_count for e in events) / n,
            verification_presence=sum(1 for e in events if e.has_verification) / n,
            mean_length=sum(e.char_length for e in events) / n,
        )

    def __str__(self):
        return (
            f"{self.label:20s}  n={self.n_events:3d}  "
            f"file_specificity={self.file_path_specificity:.2f}  "
            f"constraint_density={self.mean_constraint_density:.2f}  "
            f"verification={self.verification_presence:.2f}  "
            f"mean_len={self.mean_length:.0f}"
        )


def compare(pre: QualitySummary, post: QualitySummary) -> dict:
    """Return per-metric delta (post - pre) and direction."""
    metrics = ["file_path_specificity", "mean_constraint_density", "verification_presence", "mean_length"]
    result = {}
    for m in metrics:
        pre_val = getattr(pre, m)
        post_val = getattr(post, m)
        delta = post_val - pre_val
        direction = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
        result[m] = {"pre": pre_val, "post": post_val, "delta": delta, "direction": direction}
    return result


# ── auto-detect ──────────────────────────────────────────────────────────────

def find_latest_session() -> Optional[str]:
    pattern = os.path.expanduser("~/.claude/projects/**/*.jsonl")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def find_compaction_boundary(jsonl_path: str) -> Optional[int]:
    """Return the line index of the compaction summary message, or None."""
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Claude Code compaction inserts a summary with this role/type
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            for inner in block.get("content", []):
                                if isinstance(inner, dict) and "compacted" in str(inner).lower():
                                    return i
                if isinstance(content, str) and "compacted" in content.lower():
                    return i
    return None


def split_at_boundary(jsonl_path: str, boundary_line: int) -> tuple[str, str]:
    """Write pre/post split files to /tmp and return their paths."""
    pre_path = "/tmp/delegation_pre.jsonl"
    post_path = "/tmp/delegation_post.jsonl"
    with open(jsonl_path) as f:
        lines = f.readlines()
    with open(pre_path, "w") as f:
        f.writelines(lines[:boundary_line])
    with open(post_path, "w") as f:
        f.writelines(lines[boundary_line:])
    return pre_path, post_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pre", help="Pre-compaction JSONL file")
    group.add_argument("--file", help="Single JSONL file to analyze")
    group.add_argument("--auto", action="store_true", help="Auto-detect latest Claude Code session")
    parser.add_argument("--post", help="Post-compaction JSONL file (used with --pre)")
    parser.add_argument("--json", dest="json_out", action="store_true", help="JSON output")
    parser.add_argument("--verbose", action="store_true", help="Print per-event table")
    args = parser.parse_args()

    pre_path = post_path = None

    if args.auto:
        session_path = find_latest_session()
        if not session_path:
            print("No Claude Code session logs found at ~/.claude/projects/", file=sys.stderr)
            sys.exit(1)
        print(f"Session: {session_path}", file=sys.stderr)
        boundary = find_compaction_boundary(session_path)
        if boundary is None:
            print("No compaction boundary detected — analyzing full session as single window.", file=sys.stderr)
            pre_path = session_path
        else:
            print(f"Compaction boundary at line {boundary}", file=sys.stderr)
            pre_path, post_path = split_at_boundary(session_path, boundary)
    elif args.pre:
        pre_path = args.pre
        post_path = args.post
    else:
        pre_path = args.file

    pre_events = extract_delegation_events(pre_path) if pre_path else []
    post_events = extract_delegation_events(post_path) if post_path else []

    pre_summary = QualitySummary.from_events("pre-compaction", pre_events)
    post_summary = QualitySummary.from_events("post-compaction", post_events) if post_path else None

    if args.json_out:
        out = {"pre": asdict(pre_summary)}
        if post_summary:
            out["post"] = asdict(post_summary)
            out["delta"] = compare(pre_summary, post_summary)
        print(json.dumps(out, indent=2))
        return

    print("\n── Delegation Quality Report ────────────────────────────────────")
    print(str(pre_summary))
    if post_summary:
        print(str(post_summary))
        print()
        print("── Delta (post − pre) ───────────────────────────────────────────")
        delta = compare(pre_summary, post_summary)
        for metric, vals in delta.items():
            print(f"  {metric:30s}  {vals['direction']}  {vals['delta']:+.3f}  ({vals['pre']:.3f} → {vals['post']:.3f})")

    if args.verbose and (pre_events or post_events):
        print("\n── Per-event table ──────────────────────────────────────────────")
        print(f"{'window':12s}  {'turn':5s}  {'tool':20s}  {'paths':5s}  {'constraints':11s}  {'verify':6s}  {'chars':6s}")
        for ev in pre_events:
            print(f"{'pre':12s}  {ev.turn:5d}  {ev.tool_name:20s}  {ev.file_path_count:5d}  {ev.constraint_count:11d}  {str(ev.has_verification):6s}  {ev.char_length:6d}")
        for ev in post_events:
            print(f"{'post':12s}  {ev.turn:5d}  {ev.tool_name:20s}  {ev.file_path_count:5d}  {ev.constraint_count:11d}  {str(ev.has_verification):6s}  {ev.char_length:6d}")

    if not pre_events and not post_events:
        print("\nNo delegation events detected. Check that the JSONL contains tool_use blocks")
        print("with tool names matching: " + ", ".join(sorted(DELEGATION_TOOLS)))


if __name__ == "__main__":
    main()
