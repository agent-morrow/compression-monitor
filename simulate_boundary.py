#!/usr/bin/env python3
"""
simulate_boundary.py — synthetic session boundary simulator for compression-monitor

Generates controlled "post-boundary" variants of agent responses so you can
validate your monitoring setup against known drift before deploying against
real agents.

Usage:
    python simulate_boundary.py generate --input pre_boundary.json --output post_boundary.json --mode vocabulary
    python simulate_boundary.py generate --input pre_boundary.json --output post_boundary.json --mode topic
    python simulate_boundary.py generate --input pre_boundary.json --output post_boundary.json --mode toolcalls
    python simulate_boundary.py generate --input pre_boundary.json --output post_boundary.json --mode combined
    python simulate_boundary.py run-all --input pre_boundary.json

Input format (pre_boundary.json):
    [
      {
        "session_id": "session_001",
        "turn": 1,
        "response": "I'll use the search_files tool to locate the relevant config.",
        "tools_called": ["search_files", "read_file"],
        "topic_keywords": ["config", "files", "search"]
      },
      ...
    ]
"""

import argparse
import json
import random
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Drift generators — each produces a degraded variant of input responses
# ---------------------------------------------------------------------------

VOCABULARY_SUBSTITUTIONS = {
    # Domain-specific terms that might decay after compression
    "config": "configuration file",
    "repo": "repository",
    "auth": "authentication",
    "deploy": "deployment process",
    "env": "environment variable",
    "token": "access credential",
    "cron": "scheduled task",
    "endpoint": "API URL",
    "payload": "request body",
    "callback": "return function",
}

GENERIC_FILLERS = [
    "I'll take a look at that.",
    "Let me check the available options.",
    "I can help with that.",
    "That should work fine.",
    "I'll proceed with the standard approach.",
]

ALTERNATE_TOOLS = {
    "search_files": ["glob_files", "find_in_directory", "list_files"],
    "read_file": ["get_file_contents", "open_file", "fetch_file"],
    "write_file": ["save_file", "update_file", "create_file"],
    "run_command": ["execute_shell", "bash", "terminal"],
    "web_search": ["search_web", "query_search", "internet_search"],
}


def apply_vocabulary_drift(responses: list, intensity: float = 0.4) -> list:
    """Replace domain-specific terms with verbose alternatives at given intensity."""
    result = []
    for r in responses:
        text = r["response"]
        if random.random() < intensity:
            for term, replacement in VOCABULARY_SUBSTITUTIONS.items():
                if term in text.lower():
                    text = text.lower().replace(term, replacement, 1)
                    break
        result.append({**r, "response": text})
    return result


def apply_topic_drift(responses: list, intensity: float = 0.3) -> list:
    """Gradually replace specific responses with generic fillers."""
    result = []
    for i, r in enumerate(responses):
        if random.random() < intensity * (i / max(len(responses) - 1, 1) + 0.2):
            # Later responses more likely to drift
            result.append({**r, "response": random.choice(GENERIC_FILLERS),
                           "topic_keywords": []})
        else:
            result.append(r)
    return result


def apply_toolcall_drift(responses: list, intensity: float = 0.4) -> list:
    """Replace specific tool calls with generic alternatives or drop them."""
    result = []
    for r in responses:
        tools = r.get("tools_called", [])
        new_tools = []
        for tool in tools:
            if random.random() < intensity:
                alternates = ALTERNATE_TOOLS.get(tool, [])
                if alternates and random.random() < 0.7:
                    new_tools.append(random.choice(alternates))
                # else: tool dropped (simulates forgetting the call exists)
            else:
                new_tools.append(tool)
        result.append({**r, "tools_called": new_tools})
    return result


def apply_combined_drift(responses: list) -> list:
    """Apply all drift types at moderate intensity — most realistic simulation."""
    responses = apply_vocabulary_drift(responses, intensity=0.35)
    responses = apply_topic_drift(responses, intensity=0.25)
    responses = apply_toolcall_drift(responses, intensity=0.35)
    return responses


DRIFT_MODES = {
    "vocabulary": apply_vocabulary_drift,
    "topic": apply_topic_drift,
    "toolcalls": apply_toolcall_drift,
    "combined": apply_combined_drift,
}


# ---------------------------------------------------------------------------
# Inline drift measurement (mirrors the three monitor scripts)
# ---------------------------------------------------------------------------

def measure_ghost_lexicon(pre: list, post: list) -> dict:
    """Fraction of domain terms from pre that vanish in post."""
    def extract_terms(responses):
        terms = set()
        for r in responses:
            terms.update(r.get("topic_keywords", []))
        return terms

    pre_terms = extract_terms(pre)
    post_terms = extract_terms(post)
    if not pre_terms:
        return {"ghost_terms": [], "ghost_rate": 0.0}
    ghosts = pre_terms - post_terms
    return {"ghost_terms": sorted(ghosts), "ghost_rate": len(ghosts) / len(pre_terms)}


def measure_behavioral_footprint(pre: list, post: list) -> dict:
    """Jaccard distance between tool-call sets across sessions."""
    def tool_set(responses):
        tools = set()
        for r in responses:
            tools.update(r.get("tools_called", []))
        return tools

    pre_tools = tool_set(pre)
    post_tools = tool_set(post)
    union = pre_tools | post_tools
    intersection = pre_tools & post_tools
    if not union:
        return {"jaccard_distance": 0.0, "dropped_tools": [], "new_tools": []}
    return {
        "jaccard_distance": 1.0 - len(intersection) / len(union),
        "dropped_tools": sorted(pre_tools - post_tools),
        "new_tools": sorted(post_tools - pre_tools),
    }


def measure_topic_drift(pre: list, post: list) -> dict:
    """Simple keyword overlap as proxy for semantic similarity."""
    def keyword_freq(responses):
        freq = {}
        for r in responses:
            for kw in r.get("topic_keywords", []):
                freq[kw] = freq.get(kw, 0) + 1
        return freq

    pre_kw = keyword_freq(pre)
    post_kw = keyword_freq(post)
    all_kw = set(pre_kw) | set(post_kw)
    if not all_kw:
        return {"keyword_overlap": 1.0, "topic_drift_score": 0.0}
    overlap = len(set(pre_kw) & set(post_kw)) / len(all_kw)
    return {"keyword_overlap": overlap, "topic_drift_score": 1.0 - overlap}


# ---------------------------------------------------------------------------
# Thresholds (from compression-monitor README decision rule)
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "ghost_rate": 0.3,        # >30% ghost lexicon → investigate
    "jaccard_distance": 0.4,  # >40% tool drift → investigate
    "topic_drift_score": 0.4, # >40% topic drift → investigate
}


def evaluate(pre: list, post: list) -> dict:
    ghost = measure_ghost_lexicon(pre, post)
    footprint = measure_behavioral_footprint(pre, post)
    topic = measure_topic_drift(pre, post)

    alerts = []
    if ghost["ghost_rate"] > THRESHOLDS["ghost_rate"]:
        alerts.append(f"ghost_lexicon: {ghost['ghost_rate']:.0%} decay (threshold {THRESHOLDS['ghost_rate']:.0%})")
    if footprint["jaccard_distance"] > THRESHOLDS["jaccard_distance"]:
        alerts.append(f"behavioral_drift: {footprint['jaccard_distance']:.0%} distance (threshold {THRESHOLDS['jaccard_distance']:.0%})")
    if topic["topic_drift_score"] > THRESHOLDS["topic_drift_score"]:
        alerts.append(f"topic_drift: {topic['topic_drift_score']:.0%} (threshold {THRESHOLDS['topic_drift_score']:.0%})")

    return {
        "ghost_lexicon": ghost,
        "behavioral_footprint": footprint,
        "topic_drift": topic,
        "alerts": alerts,
        "triggered": len(alerts) > 0,
    }


# ---------------------------------------------------------------------------
# Sample data for quick demo
# ---------------------------------------------------------------------------

SAMPLE_RESPONSES = [
    {
        "session_id": "session_001", "turn": 1,
        "response": "I'll search the repo for the auth config and read the relevant env files.",
        "tools_called": ["search_files", "read_file"],
        "topic_keywords": ["auth", "config", "env", "repo"]
    },
    {
        "session_id": "session_001", "turn": 2,
        "response": "Found the token in the config. I'll update the endpoint and deploy.",
        "tools_called": ["write_file", "run_command"],
        "topic_keywords": ["token", "endpoint", "config", "deploy"]
    },
    {
        "session_id": "session_001", "turn": 3,
        "response": "Deployment succeeded. The cron job will handle the callback on schedule.",
        "tools_called": ["run_command"],
        "topic_keywords": ["deploy", "cron", "callback"]
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_generate(args):
    if args.input == "SAMPLE":
        pre = SAMPLE_RESPONSES
        print("Using built-in sample data (pass --input <file.json> for real data)")
    else:
        pre = json.loads(Path(args.input).read_text())

    mode_fn = DRIFT_MODES.get(args.mode)
    if not mode_fn:
        print(f"Unknown mode: {args.mode}. Choose from: {', '.join(DRIFT_MODES)}")
        sys.exit(1)

    post = mode_fn(pre)
    Path(args.output).write_text(json.dumps(post, indent=2))
    print(f"Generated post-boundary data ({args.mode} drift) → {args.output}")

    # Show summary
    result = evaluate(pre, post)
    print(f"\nMeasured drift against generated data:")
    print(f"  Ghost lexicon rate:     {result['ghost_lexicon']['ghost_rate']:.0%}")
    print(f"  Behavioral Jaccard:     {result['behavioral_footprint']['jaccard_distance']:.0%}")
    print(f"  Topic drift score:      {result['topic_drift']['topic_drift_score']:.0%}")
    if result["alerts"]:
        print(f"\nALERT: {len(result['alerts'])} threshold(s) exceeded:")
        for a in result["alerts"]:
            print(f"  ! {a}")
    else:
        print("\n  No thresholds exceeded.")


def cmd_run_all(args):
    """Run all four drift modes against input data and report side-by-side."""
    if args.input == "SAMPLE":
        pre = SAMPLE_RESPONSES
        print("Using built-in sample data\n")
    else:
        pre = json.loads(Path(args.input).read_text())

    print(f"{'Mode':<12} {'Ghost%':>8} {'Behavior%':>10} {'Topic%':>8} {'Alerts':>7}")
    print("-" * 52)
    for mode, fn in DRIFT_MODES.items():
        post = fn(pre)
        r = evaluate(pre, post)
        print(
            f"{mode:<12} "
            f"{r['ghost_lexicon']['ghost_rate']:>7.0%} "
            f"{r['behavioral_footprint']['jaccard_distance']:>9.0%} "
            f"{r['topic_drift']['topic_drift_score']:>7.0%} "
            f"{'YES' if r['triggered'] else 'no':>7}"
        )
    print("\nNote: randomised simulation — rerun for different samples.")
    print("See Issue #5 for structural limits: framing-level drift is not captured.")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate session boundary drift for testing compression-monitor setup."
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate a drifted post-boundary dataset")
    gen.add_argument("--input", default="SAMPLE", help="Pre-boundary JSON file (default: built-in sample)")
    gen.add_argument("--output", required=True, help="Output file for post-boundary JSON")
    gen.add_argument("--mode", default="combined",
                     choices=list(DRIFT_MODES), help="Drift type to simulate")

    run = sub.add_parser("run-all", help="Run all drift modes and compare results")
    run.add_argument("--input", default="SAMPLE", help="Pre-boundary JSON file (default: built-in sample)")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
