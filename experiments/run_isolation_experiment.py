#!/usr/bin/env python3
"""
run_isolation_experiment.py — Scaffold for the 2x2 isolation design.

Automates Cells A and B of the isolation design (see isolation_design.md):
  - Cell A: model+toolchain fixed, compressor OFF  → baseline behavioral footprint
  - Cell B: model+toolchain fixed, compressor ON   → compression-caused drift signal

Usage:
    python run_isolation_experiment.py \\
        --harness ./your_agent_harness.py \\
        --turns 20 \\
        --output /tmp/isolation_results.json

Your harness script must accept:
    --compressor-on | --compressor-off
    --session-id <string>
    --output <path>   (writes JSONL with one record per turn)

Each JSONL record must have at minimum:
    {"session_id": "...", "response_length": int, "tool_calls": int}

Extend this scaffold for Cells C and D by varying --model or --toolchain args.
See isolation_design.md for the full experimental protocol.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ── Harness runner ────────────────────────────────────────────────────────────

def run_cell(harness: str, session_id: str, compressor_on: bool, turns: int, extra_args: list) -> list[dict]:
    """Run the agent harness for one cell and return the JSONL records."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    flag = "--compressor-on" if compressor_on else "--compressor-off"
    cmd = [
        sys.executable, harness,
        flag,
        "--session-id", session_id,
        "--turns", str(turns),
        "--output", tmp_path,
    ] + extra_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"ERROR: harness exited {result.returncode}", file=sys.stderr)
            print(result.stderr[:500], file=sys.stderr)
            return []
    except subprocess.TimeoutExpired:
        print("ERROR: harness timed out after 300s", file=sys.stderr)
        return []

    records = []
    try:
        with open(tmp_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: harness did not write output to {tmp_path}", file=sys.stderr)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return records


# ── Behavioral fingerprint ────────────────────────────────────────────────────

def fingerprint(records: list[dict]) -> dict:
    """Compute a behavioral fingerprint from a list of turn records."""
    if not records:
        return {}

    lengths = [r.get("response_length", 0) for r in records]
    tool_calls = [r.get("tool_calls", 0) for r in records]
    n = len(records)

    mean_length = sum(lengths) / n
    tool_ratio = sum(1 for t in tool_calls if t > 0) / n

    return {
        "exchange_count": n,
        "response_length_mean": round(mean_length, 1),
        "tool_call_ratio": round(tool_ratio, 3),
    }


def shift_score(fp_a: dict, fp_b: dict) -> float:
    """Normalised shift score between two fingerprints (0=no shift, 1=max shift)."""
    if not fp_a or not fp_b:
        return float("nan")

    def norm_diff(key, scale):
        a, b = fp_a.get(key, 0), fp_b.get(key, 0)
        return abs(a - b) / scale if scale else 0.0

    length_diff = norm_diff("response_length_mean", max(fp_a.get("response_length_mean", 1), 1))
    tool_diff   = norm_diff("tool_call_ratio", 1.0)

    return round((length_diff + tool_diff) / 2, 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2x2 isolation experiment scaffold (Cells A and B)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--harness", required=True,
                        help="Path to agent harness script (see docstring for contract)")
    parser.add_argument("--turns", type=int, default=20,
                        help="Number of turns per cell (default: 20)")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON results (default: stdout)")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to the harness unchanged")
    args = parser.parse_args()

    harness = str(Path(args.harness).resolve())
    if not os.path.exists(harness):
        print(f"ERROR: harness not found: {harness}", file=sys.stderr)
        sys.exit(1)

    print(f"Running Cell A (compressor OFF, {args.turns} turns)...", file=sys.stderr)
    records_a = run_cell(harness, "cell-A", compressor_on=False,
                         turns=args.turns, extra_args=args.extra)

    print(f"Running Cell B (compressor ON,  {args.turns} turns)...", file=sys.stderr)
    records_b = run_cell(harness, "cell-B", compressor_on=True,
                         turns=args.turns, extra_args=args.extra)

    fp_a = fingerprint(records_a)
    fp_b = fingerprint(records_b)
    score = shift_score(fp_a, fp_b)
    level = "HIGH" if score > 0.3 else "MODERATE" if score > 0.1 else "LOW" if score == score else "UNKNOWN"

    result = {
        "cell_A": fp_a,
        "cell_B": fp_b,
        "B_vs_A_shift_score": score,
        "level": level,
        "interpretation": (
            "Cell B shows behavioural change attributable to compressor activation." if score > 0.1
            else "No significant compressor-caused drift detected at this turn count."
        ),
    }

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
