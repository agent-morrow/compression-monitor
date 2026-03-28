#!/usr/bin/env python3
"""
behavioral_footprint.py — Detect operational consistency shifts across agent sessions.

Usage:
    python behavioral_footprint.py --log agent_session_log.jsonl [--window 50]

The session log JSONL should contain one entry per agent exchange with fields:
    - "session_id": string session identifier
    - "response_length": int (character count of agent response)
    - "tool_calls": int (number of tool calls made, 0 if none)
    - "latency_ms": float (optional, response latency in milliseconds)

Output: per-session behavioral fingerprint and shift detection between consecutive sessions.
"""

import argparse
import json
import sys
from collections import defaultdict
import math


def load_log(path: str) -> dict[str, list[dict]]:
    sessions = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sid = obj.get("session_id", "unknown")
            sessions[sid].append(obj)
    return dict(sessions)


def fingerprint(exchanges: list[dict]) -> dict:
    lengths = [e.get("response_length", 0) for e in exchanges]
    tool_calls = [e.get("tool_calls", 0) for e in exchanges]
    latencies = [e.get("latency_ms") for e in exchanges if e.get("latency_ms") is not None]

    def stats(vals):
        if not vals:
            return {"mean": 0, "std": 0}
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0
        return {"mean": round(mean, 2), "std": round(math.sqrt(variance), 2)}

    tool_call_ratio = sum(1 for t in tool_calls if t > 0) / len(tool_calls) if tool_calls else 0

    return {
        "exchange_count": len(exchanges),
        "response_length": stats(lengths),
        "tool_call_ratio": round(tool_call_ratio, 4),
        "avg_tool_calls": round(sum(tool_calls) / len(tool_calls), 4) if tool_calls else 0,
        "latency_ms": stats(latencies) if latencies else None,
    }


def shift_score(fp_a: dict, fp_b: dict) -> float:
    """
    Compute a normalized shift score between two behavioral fingerprints.
    Returns 0.0 (no shift) to 1.0 (maximal shift).
    """
    scores = []

    # Response length shift (normalized by larger mean)
    mean_a = fp_a["response_length"]["mean"]
    mean_b = fp_b["response_length"]["mean"]
    if max(mean_a, mean_b) > 0:
        scores.append(abs(mean_a - mean_b) / max(mean_a, mean_b))

    # Tool call ratio shift
    scores.append(abs(fp_a["tool_call_ratio"] - fp_b["tool_call_ratio"]))

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description="Behavioral footprint shift detector")
    parser.add_argument("--log", required=True, help="JSONL session log file")
    parser.add_argument("--window", type=int, default=50, help="Exchanges per window for fingerprinting")
    args = parser.parse_args()

    sessions = load_log(args.log)
    if not sessions:
        print("ERROR: No sessions found in log.", file=sys.stderr)
        sys.exit(1)

    session_ids = sorted(sessions.keys())
    fingerprints = {sid: fingerprint(sessions[sid]) for sid in session_ids}

    print(f"sessions_analyzed: {len(session_ids)}")
    print()

    for sid in session_ids:
        fp = fingerprints[sid]
        print(f"session: {sid}")
        print(f"  exchanges: {fp['exchange_count']}")
        print(f"  response_length_mean: {fp['response_length']['mean']}")
        print(f"  tool_call_ratio: {fp['tool_call_ratio']}")
        if fp.get("latency_ms"):
            print(f"  latency_ms_mean: {fp['latency_ms']['mean']}")

    if len(session_ids) >= 2:
        print()
        print("--- shift analysis ---")
        for i in range(len(session_ids) - 1):
            a, b = session_ids[i], session_ids[i + 1]
            score = shift_score(fingerprints[a], fingerprints[b])
            level = "HIGH" if score > 0.3 else "MODERATE" if score > 0.1 else "LOW"
            print(f"{a} → {b}: shift_score={score} [{level}]")


if __name__ == "__main__":
    main()
