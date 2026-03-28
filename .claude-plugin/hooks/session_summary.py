#!/usr/bin/env python3
"""
Stop hook: print a drift summary when the Claude Code session ends.

Reads the persisted .claude/compression-monitor.json state and emits a
one-line summary to stderr so the user sees it before the session closes.
"""
import json
import os
import sys
from pathlib import Path


def main():
    try:
        json.load(sys.stdin)
    except Exception:
        pass

    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    state_path = project_dir / ".claude/compression-monitor.json"

    if not state_path.exists():
        sys.exit(0)

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        sys.exit(0)

    n = state.get("seen_compactions", 0)
    last = state.get("last_drift")

    if n == 0:
        print("\ncompression-monitor: no compaction events in this session.\n", file=sys.stderr)
    elif last:
        status = "⚠️  DRIFT ALERT" if last.get("alert") else "✓  within bounds"
        print(
            f"\ncompression-monitor: {n} compaction event(s) this session. "
            f"Last drift: composite={last.get('composite', '?')} — {status}\n",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
