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
    python simulate_boundary.py generate --input pre_boundary.json --output post_boundary.json --mode framing
    python simulate_boundary.py run-all --input pre_boundary.json
    python simulate_boundary.py benchmark [--input pre_boundary.json] [--pairs N]

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

# Framing pairs: (conservative_prefix, exploratory_prefix)
# These change the implicit prior without affecting vocabulary, tools, or topic keywords.
FRAMING_PAIRS = [
    ("I'll carefully verify before proceeding: ", "Let me move quickly on this: "),
    ("To be safe, ", "Going ahead with: "),
    ("Before making changes, I'll confirm — ", ""),
    ("I'll double-check the output of ", "Running "),
    ("Taking a conservative approach: ", "Taking an iterative approach: "),
]


def apply_vocabulary_drift(responses: list, intensity: float = 0.4) -> list:
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
    result = []
    for i, r in enumerate(responses):
        if random.random() < intensity * (i / max(len(responses) - 1, 1) + 0.2):
            result.append({**r, "response": random.choice(GENERIC_FILLERS),
                           "topic_keywords": []})
        else:
            result.append(r)
    return result


def apply_toolcall_drift(responses: list, intensity: float = 0.4) -> list:
    result = []
    for r in responses:
        tools = r.get("tools_called", [])
        new_tools = []
        for tool in tools:
            if random.random() < intensity:
                alternates = ALTERNATE_TOOLS.get(tool, [])
                if alternates and random.random() < 0.7:
                    new_tools.append(random.choice(alternates))
            else:
                new_tools.append(tool)
        result.append({**r, "tools_called": new_tools})
    return result


def apply_combined_drift(responses: list) -> list:
    responses = apply_vocabulary_drift(responses, intensity=0.35)
    responses = apply_topic_drift(responses, intensity=0.25)
    responses = apply_toolcall_drift(responses, intensity=0.35)
    return responses


def apply_framing_drift(responses: list, framing: str = "conservative") -> list:
    """
    Inject a framing prior (conservative vs exploratory) while preserving all
    surface statistics: same vocabulary distribution, same tool-call patterns,
    same topic keywords, same response length profile.

    The semantic content shifts (implicit caution vs implicit speed), but no
    surface instrument can detect this — ghost lexicon, behavioral footprint,
    and topic drift all score near zero. This is the non-separable case: the
    correct monitor output is 'cannot determine'.

    Use `benchmark` to score your monitor's abstention rate on these pairs.
    """
    result = []
    for r in responses:
        text = r["response"]
        pair = random.choice(FRAMING_PAIRS)
        prefix = pair[0] if framing == "conservative" else pair[1]
        if prefix and not text.lower().startswith(prefix.lower().strip()):
            # Prepend the framing prefix, trim to preserve approximate length
            combined = prefix + text[0].lower() + text[1:] if text else prefix
            # Trim trailing words to approximate original length
            if len(combined) > len(text) + 30:
                combined = combined[:len(text) + 10].rsplit(" ", 1)[0]
            text = combined
        result.append({
            **r,
            "response": text,
            "_framing": framing,
            # Preserve all surface-measurable fields identically
            "tools_called": r.get("tools_called", []),
            "topic_keywords": r.get("topic_keywords", []),
        })
    return result


DRIFT_MODES = {
    "vocabulary": apply_vocabulary_drift,
    "topic": apply_topic_drift,
    "toolcalls": apply_toolcall_drift,
    "combined": apply_combined_drift,
    "framing": lambda r: apply_framing_drift(r, framing="exploratory"),
}


# ---------------------------------------------------------------------------
# Inline drift measurement (mirrors the three monitor scripts)
# ---------------------------------------------------------------------------

def measure_ghost_lexicon(pre: list, post: list) -> dict:
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


THRESHOLDS = {
    "ghost_rate": 0.3,
    "jaccard_distance": 0.4,
    "topic_drift_score": 0.4,
}


def evaluate(pre: list, post: list) -> dict:
    ghost = measure_ghost_lexicon(pre, post)
    footprint = measure_behavioral_footprint(pre, post)
    topic = measure_topic_drift(pre, post)
    alerts = []
    if ghost["ghost_rate"] > THRESHOLDS["ghost_rate"]:
        alerts.append(f"ghost_lexicon: {ghost['ghost_rate']:.0%} decay")
    if footprint["jaccard_distance"] > THRESHOLDS["jaccard_distance"]:
        alerts.append(f"behavioral_drift: {footprint['jaccard_distance']:.0%} distance")
    if topic["topic_drift_score"] > THRESHOLDS["topic_drift_score"]:
        alerts.append(f"topic_drift: {topic['topic_drift_score']:.0%}")
    return {
        "ghost_lexicon": ghost,
        "behavioral_footprint": footprint,
        "topic_drift": topic,
        "alerts": alerts,
        "triggered": len(alerts) > 0,
    }


# ---------------------------------------------------------------------------
# Sample data
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

    if args.mode == "framing":
        post_conservative = apply_framing_drift(pre, framing="conservative")
        post_exploratory = apply_framing_drift(pre, framing="exploratory")
        Path(args.output).write_text(json.dumps({
            "conservative": post_conservative,
            "exploratory": post_exploratory,
        }, indent=2))
        print(f"Generated framing pair → {args.output}")
        print("\nFraming mode generates surface-equivalent pairs with different implicit priors.")
        print("Expected result: monitors score near zero on both — correct abstention.")
        r_c = evaluate(pre, post_conservative)
        r_e = evaluate(pre, post_exploratory)
        print(f"\n  Conservative variant: ghost={r_c['ghost_lexicon']['ghost_rate']:.0%}, "
              f"behavior={r_c['behavioral_footprint']['jaccard_distance']:.0%}, "
              f"topic={r_c['topic_drift']['topic_drift_score']:.0%}")
        print(f"  Exploratory variant:  ghost={r_e['ghost_lexicon']['ghost_rate']:.0%}, "
              f"behavior={r_e['behavioral_footprint']['jaccard_distance']:.0%}, "
              f"topic={r_e['topic_drift']['topic_drift_score']:.0%}")
        triggered = r_c["triggered"] or r_e["triggered"]
        status = "FAIL (false positive)" if triggered else "PASS (correct abstention)"
        print(f"\n  Monitor abstention test: {status}")
        return

    mode_fn = DRIFT_MODES.get(args.mode)
    if not mode_fn:
        print(f"Unknown mode: {args.mode}. Choose from: {', '.join(DRIFT_MODES)}")
        sys.exit(1)

    post = mode_fn(pre)
    Path(args.output).write_text(json.dumps(post, indent=2))
    print(f"Generated post-boundary data ({args.mode} drift) → {args.output}")
    result = evaluate(pre, post)
    print(f"\nMeasured drift:")
    print(f"  Ghost lexicon rate:  {result['ghost_lexicon']['ghost_rate']:.0%}")
    print(f"  Behavioral Jaccard:  {result['behavioral_footprint']['jaccard_distance']:.0%}")
    print(f"  Topic drift score:   {result['topic_drift']['topic_drift_score']:.0%}")
    if result["alerts"]:
        print(f"\nALERT: {len(result['alerts'])} threshold(s) exceeded:")
        for a in result["alerts"]:
            print(f"  ! {a}")
    else:
        print("\n  No thresholds exceeded.")


def cmd_run_all(args):
    if args.input == "SAMPLE":
        pre = SAMPLE_RESPONSES
        print("Using built-in sample data\n")
    else:
        pre = json.loads(Path(args.input).read_text())

    print(f"{'Mode':<12} {'Ghost%':>8} {'Behavior%':>10} {'Topic%':>8} {'Alerts':>7}")
    print("-" * 52)
    for mode, fn in DRIFT_MODES.items():
        if mode == "framing":
            post = apply_framing_drift(pre, framing="exploratory")
        else:
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
    print("See README Epistemological Bounds for structural limits.")


def cmd_benchmark(args):
    """
    Benchmark the monitor's abstention rate on non-separable pairs and
    detection rate on separable pairs.

    For each of N trials:
      - Separable pair: pre vs combined-drift post → monitor should TRIGGER
      - Non-separable pair: conservative framing vs exploratory framing → monitor should NOT trigger

    Scores:
      - Detection rate:  fraction of separable pairs where ≥1 monitor triggered
      - Abstention rate: fraction of non-separable pairs where NO monitor triggered
                         (correct behaviour: cannot distinguish framing-level change)

    A well-calibrated monitor scores high on both.
    An over-sensitive monitor scores high on detection but low on abstention (false positives).
    """
    if args.input == "SAMPLE":
        pre = SAMPLE_RESPONSES
    else:
        pre = json.loads(Path(args.input).read_text())

    n = args.pairs
    detected = 0
    abstained = 0

    sep_details = []
    nonsep_details = []

    for _ in range(n):
        # Separable: real drift
        post_sep = apply_combined_drift(pre)
        r_sep = evaluate(pre, post_sep)
        if r_sep["triggered"]:
            detected += 1
        sep_details.append(r_sep)

        # Non-separable: framing-only, surface-equivalent
        post_c = apply_framing_drift(pre, framing="conservative")
        post_e = apply_framing_drift(pre, framing="exploratory")
        r_nonsep = evaluate(post_c, post_e)
        if not r_nonsep["triggered"]:
            abstained += 1
        nonsep_details.append(r_nonsep)

    detection_rate = detected / n
    abstention_rate = abstained / n

    print("=" * 56)
    print("  compression-monitor benchmark (Issue #6)")
    print("=" * 56)
    print(f"  Trials (pairs):       {n}")
    print()
    print(f"  SEPARABLE pairs (combined drift):")
    print(f"    Detection rate:     {detection_rate:.0%}  ({detected}/{n} triggered)")
    print(f"    Expected:           ~100%")
    avg_ghost = sum(r["ghost_lexicon"]["ghost_rate"] for r in sep_details) / n
    avg_beh   = sum(r["behavioral_footprint"]["jaccard_distance"] for r in sep_details) / n
    avg_topic = sum(r["topic_drift"]["topic_drift_score"] for r in sep_details) / n
    print(f"    Avg ghost rate:     {avg_ghost:.0%}")
    print(f"    Avg behavior dist:  {avg_beh:.0%}")
    print(f"    Avg topic drift:    {avg_topic:.0%}")
    print()
    print(f"  NON-SEPARABLE pairs (framing-only, surface-equivalent):")
    print(f"    Abstention rate:    {abstention_rate:.0%}  ({abstained}/{n} correctly silent)")
    print(f"    Expected:           ~100%  (no surface signal exists)")
    avg_ghost_ns = sum(r["ghost_lexicon"]["ghost_rate"] for r in nonsep_details) / n
    avg_beh_ns   = sum(r["behavioral_footprint"]["jaccard_distance"] for r in nonsep_details) / n
    avg_topic_ns = sum(r["topic_drift"]["topic_drift_score"] for r in nonsep_details) / n
    print(f"    Avg ghost rate:     {avg_ghost_ns:.0%}  (should be 0%)")
    print(f"    Avg behavior dist:  {avg_beh_ns:.0%}  (should be 0%)")
    print(f"    Avg topic drift:    {avg_topic_ns:.0%}  (should be 0%)")
    print()
    print("  CALIBRATION SUMMARY:")
    if detection_rate >= 0.8 and abstention_rate >= 0.8:
        verdict = "PASS — monitors detect real drift and abstain on framing-only change"
    elif detection_rate < 0.8 and abstention_rate >= 0.8:
        verdict = "PARTIAL — monitors correctly abstain but miss real drift (tune thresholds down)"
    elif detection_rate >= 0.8 and abstention_rate < 0.8:
        verdict = "PARTIAL — monitors detect drift but produce false positives on framing (tune thresholds up)"
    else:
        verdict = "FAIL — monitors miss real drift AND produce false positives"
    print(f"  {verdict}")
    print("=" * 56)
    print("\nNote: framing-level compression remains definitionally invisible to these")
    print("surface instruments (see README: Cannot See v0.1.0 + Issue #5).")
    print("This benchmark validates calibration, not completeness.")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate session boundary drift for testing compression-monitor setup."
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate a drifted post-boundary dataset")
    gen.add_argument("--input", default="SAMPLE")
    gen.add_argument("--output", required=True)
    gen.add_argument("--mode", default="combined", choices=list(DRIFT_MODES))

    run = sub.add_parser("run-all", help="Run all drift modes and compare results")
    run.add_argument("--input", default="SAMPLE")

    bench = sub.add_parser("benchmark", help="Score detection rate vs abstention rate (Issue #6)")
    bench.add_argument("--input", default="SAMPLE")
    bench.add_argument("--pairs", type=int, default=20, help="Number of trial pairs (default: 20)")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run-all":
        cmd_run_all(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
