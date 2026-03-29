#!/usr/bin/env python3
"""
behavioral_footprint.py — Detect operational consistency shifts across agent sessions.

Usage:
    python behavioral_footprint.py --log agent_session_log.jsonl [--window 50]
    python behavioral_footprint.py --pre session_pre.jsonl --post session_post.jsonl

Supported formats:
    1. Flattened exchange logs with fields:
    - "session_id": string session identifier
    - "response_length": int (character count of agent response)
    - "tool_calls": int (number of tool calls made, 0 if none)
    - "latency_ms": float (optional, response latency in milliseconds)
    2. Lightweight text logs:
    - "text": string agent response text
    3. OpenClaw / chat-style logs:
    - "message": {"role": "assistant", "content": [...]}

Output: per-session behavioral fingerprint and shift detection between consecutive sessions.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def extract_text_and_tool_calls(content) -> tuple[str, int]:
    text_parts = []
    tool_calls = 0
    if isinstance(content, str):
        return content, 0
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "toolCall":
                    tool_calls += 1
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
    elif isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts), tool_calls


def normalize_exchange(obj: dict, default_session_id: str) -> tuple[str, dict] | None:
    if "response_length" in obj or "tool_calls" in obj:
        session_id = obj.get("session_id") or default_session_id
        return str(session_id), {
            "response_length": int(obj.get("response_length", 0) or 0),
            "tool_calls": int(obj.get("tool_calls", 0) or 0),
            "latency_ms": obj.get("latency_ms"),
        }

    if isinstance(obj.get("text"), str):
        session_id = obj.get("session_id") or default_session_id
        return str(session_id), {
            "response_length": len(obj["text"]),
            "tool_calls": int(obj.get("tool_calls", 0) or 0),
            "latency_ms": obj.get("latency_ms"),
        }

    message = obj.get("message")
    if isinstance(message, dict) and message.get("role") == "assistant":
        text, tool_calls = extract_text_and_tool_calls(message.get("content"))
        if text.strip() or tool_calls:
            session_id = obj.get("session_id") or obj.get("sessionId") or default_session_id
            return str(session_id), {
                "response_length": len(text.strip()),
                "tool_calls": tool_calls,
                "latency_ms": obj.get("latency_ms"),
            }

    role = obj.get("role")
    if role == "assistant":
        text, tool_calls = extract_text_and_tool_calls(obj.get("content"))
        if text.strip() or tool_calls:
            session_id = obj.get("session_id") or obj.get("sessionId") or default_session_id
            return str(session_id), {
                "response_length": len(text.strip()),
                "tool_calls": tool_calls,
                "latency_ms": obj.get("latency_ms"),
            }

    return None


def load_log(path: str) -> dict[str, list[dict]]:
    sessions = defaultdict(list)
    default_session_id = Path(path).stem
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            normalized = normalize_exchange(obj, default_session_id)
            if normalized is None:
                continue
            sid, exchange = normalized
            sessions[sid].append(exchange)
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


def extract_footprint(records: list[dict | object]) -> dict:
    """Normalize records into exchanges and compute a behavioral fingerprint."""
    exchanges: list[dict] = []
    for record in records:
        if isinstance(record, dict):
            if {"response_length", "tool_calls"} & set(record):
                exchanges.append({
                    "response_length": int(record.get("response_length", 0) or 0),
                    "tool_calls": int(record.get("tool_calls", 0) or 0),
                    "latency_ms": record.get("latency_ms"),
                })
                continue
            text = str(record.get("text", record.get("output_text", "")) or "")
            tool_calls = record.get("tool_calls", []) or []
            exchanges.append({
                "response_length": len(text),
                "tool_calls": len(tool_calls),
                "latency_ms": record.get("latency_ms"),
            })
            continue

        text = str(getattr(record, "output_text", "") or "")
        tool_calls = getattr(record, "tool_calls", []) or []
        exchanges.append({
            "response_length": len(text),
            "tool_calls": len(tool_calls),
            "latency_ms": getattr(record, "latency_ms", None),
        })

    return fingerprint(exchanges)


def shift_score(fp_a: dict, fp_b: dict) -> float:
    """
    Compute a normalized shift score between two behavioral fingerprints.
    Returns 0.0 (no shift) to 1.0 (maximal shift).
    """
    scores = []

    mean_a = fp_a["response_length"]["mean"]
    mean_b = fp_b["response_length"]["mean"]
    if max(mean_a, mean_b) > 0:
        scores.append(abs(mean_a - mean_b) / max(mean_a, mean_b))

    scores.append(abs(fp_a["tool_call_ratio"] - fp_b["tool_call_ratio"]))

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def compute_footprint_delta(fp_a: dict, fp_b: dict) -> dict[str, float]:
    """Expose per-dimension deltas alongside the composite shift score."""
    response_length_delta = abs(
        fp_a["response_length"]["mean"] - fp_b["response_length"]["mean"]
    )
    tool_ratio_delta = abs(fp_a["tool_call_ratio"] - fp_b["tool_call_ratio"])
    return {
        "response_length_delta": round(response_length_delta, 4),
        "tool_ratio_delta": round(tool_ratio_delta, 4),
        "shift_score": shift_score(fp_a, fp_b),
    }


class BehavioralFootprintTracker:
    """Track tool-use stability across an anchored early window and a recent window."""

    def __init__(self, anchor_window: int = 5, recent_window: int = 5):
        self.anchor_window = max(anchor_window, 1)
        self.recent_window = max(recent_window, 1)
        self._steps: list[list[str]] = []
        self._anchor_steps: list[list[str]] = []

    def record(self, step_index: int, tool_calls: list[str] | None) -> None:
        del step_index
        calls = [str(tool) for tool in (tool_calls or []) if tool]
        self._steps.append(calls)
        if len(self._anchor_steps) < self.anchor_window:
            self._anchor_steps.append(calls)

    def record_call(self, tool_name: str) -> None:
        calls = [str(tool_name)] if tool_name else []
        self._steps.append(calls)
        if len(self._anchor_steps) < self.anchor_window:
            self._anchor_steps.append(calls)

    def consistency_score(self) -> float:
        if not self._steps:
            return 1.0

        anchor_source = self._anchor_steps or self._steps[: self.anchor_window]
        anchor_tools = {tool for calls in anchor_source for tool in calls}
        recent_tools = {tool for calls in self._steps[-self.recent_window :] for tool in calls}

        if not anchor_tools and not recent_tools:
            return 1.0

        union = anchor_tools | recent_tools
        if not union:
            return 1.0

        return round(len(anchor_tools & recent_tools) / len(union), 4)


def main():
    parser = argparse.ArgumentParser(description="Behavioral footprint shift detector")
    parser.add_argument("--log", help="JSONL session log file")
    parser.add_argument("--pre", help="Pre-boundary JSONL session log file")
    parser.add_argument("--post", help="Post-boundary JSONL session log file")
    parser.add_argument("--window", type=int, default=50, help="Exchanges per window for fingerprinting")
    args = parser.parse_args()

    if args.log and (args.pre or args.post):
        print("ERROR: Use either --log or --pre/--post, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.log and not (args.pre and args.post):
        print("ERROR: Provide --log or both --pre and --post.", file=sys.stderr)
        sys.exit(1)

    if args.log:
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
                print(f"{a} -> {b}: shift_score={score:.4f} ({level})")
    else:
        pre_sessions = load_log(args.pre)
        post_sessions = load_log(args.post)

        pre_exchanges = [exchange for exchanges in pre_sessions.values() for exchange in exchanges]
        post_exchanges = [exchange for exchanges in post_sessions.values() for exchange in exchanges]

        if not pre_exchanges or not post_exchanges:
            print("ERROR: No comparable exchanges found in one or both files.", file=sys.stderr)
            sys.exit(1)

        fp_pre = fingerprint(pre_exchanges)
        fp_post = fingerprint(post_exchanges)
        score = shift_score(fp_pre, fp_post)
        level = "HIGH" if score > 0.3 else "MODERATE" if score > 0.1 else "LOW"

        print("pre_boundary:")
        print(f"  exchanges: {fp_pre['exchange_count']}")
        print(f"  response_length_mean: {fp_pre['response_length']['mean']}")
        print(f"  tool_call_ratio: {fp_pre['tool_call_ratio']}")
        print("post_boundary:")
        print(f"  exchanges: {fp_post['exchange_count']}")
        print(f"  response_length_mean: {fp_post['response_length']['mean']}")
        print(f"  tool_call_ratio: {fp_post['tool_call_ratio']}")
        print(f"shift_score: {score:.4f}")
        print(f"interpretation: {level}")


if __name__ == "__main__":
    main()
