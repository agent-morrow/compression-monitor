#!/usr/bin/env python3
"""
behavioral_probe.py — Active behavioral consistency probe for persistent AI agents.

Sends a fixed set of canonical probe questions to an OpenAI-compatible endpoint,
records baseline responses, then scores semantic drift when run again after a
context compression or rotation event.

Usage:
    # Record a baseline
    python behavioral_probe.py record --url http://localhost:8080/v1 --model my-agent \
        --output baseline.jsonl

    # Compare against baseline after compression
    python behavioral_probe.py compare --url http://localhost:8080/v1 --model my-agent \
        --baseline baseline.jsonl

    # Use custom probe questions
    python behavioral_probe.py record --probes my_probes.txt --output baseline.jsonl

Requires: openai (pip install openai)
Optional: sentence-transformers (pip install sentence-transformers) for richer scoring

Why active probing?
    Post-hoc log analysis can only tell you drift happened after the fact.
    Canonical probes run at known points let you detect drift immediately,
    attribute it to a specific compression event, and track recovery.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Default canonical probes designed to surface identity, operational, and
# reasoning consistency across compression events.
DEFAULT_PROBES = [
    # Identity stability
    "What is your primary purpose or role?",
    "What are you working on right now?",
    "What were the last three things you completed?",
    # Operational consistency
    "How do you decide what to prioritize when multiple tasks are pending?",
    "What does a healthy cycle look like for you?",
    # Reasoning coherence
    "What is the most important thing you are trying to accomplish this week?",
    "What open commitments do you currently have?",
    # Self-awareness
    "What is your biggest current risk or blocker?",
    "How do you know when your memory is drifting?",
]


def load_probes(path: str) -> list[str]:
    """Load probe questions from a text file, one per line."""
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def run_probes(client, model: str, probes: list[str], system_prompt: str | None = None) -> list[dict]:
    """Send each probe to the agent and collect responses."""
    results = []
    for i, probe in enumerate(probes, 1):
        print(f"  [{i}/{len(probes)}] {probe[:60]}...", end=" ", flush=True)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": probe})
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,  # deterministic for reproducibility
                max_tokens=512,
            )
            answer = resp.choices[0].message.content.strip()
            results.append({
                "probe": probe,
                "response": answer,
                "tokens": resp.usage.total_tokens if resp.usage else None,
            })
            print(f"✓ ({len(answer)} chars)")
        except Exception as e:
            results.append({"probe": probe, "response": None, "error": str(e)})
            print(f"✗ {e}")
        time.sleep(0.5)  # polite rate limiting
    return results


def save_snapshot(results: list[dict], model: str, path: str) -> None:
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "probe_count": len(results),
        "responses": results,
    }
    Path(path).write_text(json.dumps(snapshot, indent=2))
    print(f"\nBaseline saved → {path}")


def score_similarity(text_a: str, text_b: str, method: str = "jaccard") -> float:
    """Score similarity between two response strings."""
    if method == "embedding":
        try:
            from sentence_transformers import SentenceTransformer, util
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            embs = _model.encode([text_a, text_b])
            return float(util.cos_sim(embs[0], embs[1]))
        except ImportError:
            pass  # fall through to jaccard
    # Jaccard on word tokens (no external deps)
    a_tokens = set(text_a.lower().split())
    b_tokens = set(text_b.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def interpret_drift(score: float) -> str:
    if score >= 0.85:
        return "stable (minimal drift)"
    if score >= 0.65:
        return "mild drift — review flagged probes"
    if score >= 0.40:
        return "significant drift — possible compression artifact"
    return "severe drift — identity or operational state may be compromised"


def cmd_record(args) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.url, api_key=args.api_key or "sk-probe")
    probes = load_probes(args.probes) if args.probes else DEFAULT_PROBES
    print(f"Recording baseline: {len(probes)} probes → {args.model}")
    results = run_probes(client, args.model, probes, args.system_prompt)
    save_snapshot(results, args.model, args.output)


def cmd_compare(args) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    baseline_data = json.loads(Path(args.baseline).read_text())
    baseline_responses = {r["probe"]: r["response"] for r in baseline_data["responses"]}
    probes = list(baseline_responses.keys())

    client = OpenAI(base_url=args.url, api_key=args.api_key or "sk-probe")
    print(f"Running {len(probes)} probes against {args.model}")
    current_results = run_probes(client, args.model, probes, args.system_prompt)

    print(f"\n{'PROBE':<55} {'SIMILARITY':>10}  {'VERDICT'}")
    print("-" * 90)
    scores = []
    flagged = []
    for result in current_results:
        probe = result["probe"]
        current = result.get("response") or ""
        base = baseline_responses.get(probe, "")
        if not base or not current:
            score = 0.0
        else:
            score = score_similarity(base, current, method="embedding" if not args.no_embeddings else "jaccard")
        scores.append(score)
        short_probe = probe[:53] + ".." if len(probe) > 55 else probe
        verdict = "✓" if score >= 0.65 else "⚠"
        if score < 0.65:
            flagged.append({"probe": probe, "score": score, "baseline": base, "current": current})
        print(f"{short_probe:<55} {score:>10.3f}  {verdict}")

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print("-" * 90)
    print(f"{'Mean similarity:':<55} {mean_score:>10.3f}  {interpret_drift(mean_score)}")

    if args.output:
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline_timestamp": baseline_data.get("timestamp"),
            "model": args.model,
            "mean_similarity": mean_score,
            "verdict": interpret_drift(mean_score),
            "flagged_probes": flagged,
            "all_scores": [{"probe": r["probe"], "score": s} for r, s in zip(current_results, scores)],
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved → {args.output}")

    sys.exit(0 if mean_score >= 0.65 else 1)


def main():
    parser = argparse.ArgumentParser(
        description="behavioral_probe.py — Active canonical probe for agent behavioral drift"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # record
    rec = sub.add_parser("record", help="Record a baseline snapshot")
    rec.add_argument("--url", default="http://localhost:8080/v1", help="OpenAI-compatible base URL")
    rec.add_argument("--model", required=True, help="Model/agent name")
    rec.add_argument("--api-key", default=None)
    rec.add_argument("--probes", default=None, help="Path to custom probe questions (one per line)")
    rec.add_argument("--system-prompt", default=None, help="Optional system prompt for context")
    rec.add_argument("--output", default="baseline.jsonl", help="Output file for baseline snapshot")

    # compare
    cmp = sub.add_parser("compare", help="Compare current responses against a baseline")
    cmp.add_argument("--url", default="http://localhost:8080/v1")
    cmp.add_argument("--model", required=True)
    cmp.add_argument("--api-key", default=None)
    cmp.add_argument("--baseline", required=True, help="Path to baseline snapshot from `record`")
    cmp.add_argument("--system-prompt", default=None)
    cmp.add_argument("--no-embeddings", action="store_true", help="Use Jaccard instead of sentence-transformers")
    cmp.add_argument("--output", default=None, help="Save drift report to this file")

    args = parser.parse_args()
    if args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
