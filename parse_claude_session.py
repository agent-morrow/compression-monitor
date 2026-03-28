#!/usr/bin/env python3
"""
parse_claude_session.py — Extract pre/post compaction samples from Claude Code session logs.

Reads a Claude Code session JSONL file (~/.claude/projects/<hash>/<session>.jsonl),
finds the compaction boundary, and writes two output files:
  - <out_prefix>_pre.jsonl  — assistant turns BEFORE the compaction event
  - <out_prefix>_post.jsonl — assistant turns AFTER the compaction event

Output format matches the input format expected by ghost_lexicon.py,
behavioral_footprint.py, and semantic_drift.py.

Usage:
    # Auto-detect most recent session in current project:
    python parse_claude_session.py --auto

    # Explicit session file:
    python parse_claude_session.py --session ~/.claude/projects/<hash>/<uuid>.jsonl

    # Custom output prefix:
    python parse_claude_session.py --auto --out /tmp/myproject

    # Watch mode: poll for a compaction boundary and emit samples when it fires
    python parse_claude_session.py --auto --watch
    python parse_claude_session.py --auto --watch --interval 5

Claude Code session JSONL schema:
    Each line is a JSON object. Compaction events appear as:
        {"type": "summary", ...}  or  {"role": "system", "content": [{"type": "text", "text": "<summary>..."}]}
    Assistant turns appear as:
        {"role": "assistant", "content": [...]}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DEFAULT_WATCH_INTERVAL = 10  # seconds


def find_latest_session() -> Path | None:
    """Return the most recently modified session JSONL in ~/.claude/projects/."""
    candidates = list(CLAUDE_PROJECTS_DIR.glob("*/*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def is_compaction_boundary(record: dict) -> bool:
    """Return True if this record marks a context compaction event."""
    if record.get("type") == "summary":
        return True
    # Some versions emit a system message with a summary tag
    if record.get("role") == "system":
        content = record.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if "<summary>" in text or "context was compressed" in text.lower():
                        return True
        elif isinstance(content, str):
            if "<summary>" in content or "context was compressed" in content.lower():
                return True
    return False


def extract_text(record: dict) -> str | None:
    """Extract assistant output text from a record, or None if not an assistant turn."""
    if record.get("role") != "assistant":
        return None
    content = record.get("content", [])
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "").strip()
                if t:
                    parts.append(t)
        return "\n".join(parts) or None
    return None


def parse_session(session_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Parse session file and return (pre_samples, post_samples).
    Each sample is {"text": "...", "role": "assistant"}.
    If no compaction boundary is found, returns (all_turns, []).
    Defensive: skips malformed/partial JSON lines (e.g. mid-flush writes).
    """
    pre, post = [], []
    found_boundary = False

    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Skip partial lines — can occur when reading a file mid-write
                continue

            if not found_boundary and is_compaction_boundary(record):
                found_boundary = True
                continue

            text = extract_text(record)
            if text is None:
                continue

            sample = {"text": text, "role": "assistant"}
            if found_boundary:
                post.append(sample)
            else:
                pre.append(sample)

    return pre, post


def write_samples(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def watch_loop(session_path: Path, out_prefix: str, interval: int) -> None:
    """
    Poll session_path every `interval` seconds. When a compaction boundary is
    detected (or new content appears after an already-detected boundary), rewrite
    the pre/post output files and print a notification.

    Exits after the first boundary is detected and post-compaction turns are written,
    unless the boundary was already present on first scan (in which case it waits for
    post-compaction content to accumulate).
    """
    out_pre = Path(f"{out_prefix}_pre.jsonl")
    out_post = Path(f"{out_prefix}_post.jsonl")

    last_size = 0
    boundary_seen = False
    last_post_count = -1

    print(f"Watching: {session_path}  (poll every {interval}s)", flush=True)
    print("Waiting for compaction boundary...", flush=True)

    while True:
        try:
            current_size = session_path.stat().st_size
        except FileNotFoundError:
            print(f"Session file disappeared: {session_path}", file=sys.stderr)
            sys.exit(1)

        if current_size != last_size:
            last_size = current_size
            pre, post = parse_session(session_path)

            if not boundary_seen and post:
                # Boundary just appeared
                boundary_seen = True
                write_samples(pre, out_pre)
                write_samples(post, out_post)
                ts = time.strftime("%H:%M:%S")
                print(f"\n[{ts}] Compaction boundary detected!", flush=True)
                print(f"  Pre-compaction turns:  {len(pre):>4}  → {out_pre}", flush=True)
                print(f"  Post-compaction turns: {len(post):>4}  → {out_post}", flush=True)
                last_post_count = len(post)
            elif boundary_seen and len(post) != last_post_count:
                # New post-compaction turns arriving — update output files
                write_samples(pre, out_pre)
                write_samples(post, out_post)
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] Updated: {len(post)} post-compaction turns → {out_post}", flush=True)
                last_post_count = len(post)
            elif not boundary_seen:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] {len(pre)} pre-compaction turns, no boundary yet", flush=True)

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pre/post compaction samples from a Claude Code session log."
    )
    parser.add_argument("--session", type=Path, help="Path to session JSONL file.")
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-detect most recently modified session in ~/.claude/projects/."
    )
    parser.add_argument(
        "--out", type=str, default="./session",
        help="Output file prefix (default: ./session). Writes <prefix>_pre.jsonl and <prefix>_post.jsonl."
    )
    parser.add_argument(
        "--watch", action="store_true",
        help=(
            "Poll the session file for new compaction events and update output files "
            "as they arrive. Useful for live monitoring during an active session. "
            "Handles partial JSON lines defensively."
        )
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_WATCH_INTERVAL,
        help=f"Poll interval in seconds for --watch mode (default: {DEFAULT_WATCH_INTERVAL})."
    )
    args = parser.parse_args()

    if not args.session and not args.auto:
        parser.print_help()
        sys.exit(1)

    if args.auto:
        session_path = find_latest_session()
        if session_path is None:
            print(f"No session files found in {CLAUDE_PROJECTS_DIR}", file=sys.stderr)
            sys.exit(1)
        print(f"Using session: {session_path}")
    else:
        session_path = args.session
        if not session_path.exists():
            print(f"Session file not found: {session_path}", file=sys.stderr)
            sys.exit(1)

    if args.watch:
        watch_loop(session_path, args.out, args.interval)
        return

    pre, post = parse_session(session_path)

    if not pre and not post:
        print("No assistant turns found in session.", file=sys.stderr)
        sys.exit(1)

    if not post:
        print(
            f"Warning: no compaction boundary detected. Found {len(pre)} pre-compaction turns.",
            file=sys.stderr
        )
        print("The session may not have hit a compaction event yet.", file=sys.stderr)

    out_pre = Path(f"{args.out}_pre.jsonl")
    out_post = Path(f"{args.out}_post.jsonl")

    write_samples(pre, out_pre)
    write_samples(post, out_post)

    print(f"Pre-compaction turns:  {len(pre):>4}  → {out_pre}")
    print(f"Post-compaction turns: {len(post):>4}  → {out_post}")

    if pre and post:
        print()
        print("Next steps:")
        print(f"  python ghost_lexicon.py --pre {out_pre} --post {out_post}")
        print(f"  python behavioral_footprint.py --pre {out_pre} --post {out_post}")
        print(f"  python semantic_drift.py --pre {out_pre} --post {out_post}")
    elif pre and not post:
        print()
        print("Run again after a compaction event to get post-compaction data.")


if __name__ == "__main__":
    main()
