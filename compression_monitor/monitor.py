#!/usr/bin/env python3
"""
monitor.py — Unified CLI entrypoint for compression-monitor toolkit.

Chains simulate_boundary + preregister in one command:
  1. Register predictions for a session before a boundary
  2. Run monitors against real or synthetic data post-boundary
  3. Record which instruments fired and in what order
  4. Evaluate against predictions, foregrounding firing order

Usage:
  python3 monitor.py run --session-id SID [--mode vocabulary|topic|toolcalls|combined]
  python3 monitor.py status --session-id SID
  python3 monitor.py demo

The demo command runs a complete example end-to-end with synthetic data,
showing what firing-order analysis looks like in practice.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

TOOLS_DIR = os.path.dirname(__file__)


def run_script(script: str, args: list, capture: bool = True):
    cmd = [sys.executable, os.path.join(TOOLS_DIR, script)] + args
    result = subprocess.run(cmd, capture_output=capture, text=True)
    return result


def cmd_run(args):
    """Full pipeline: register predictions, simulate boundary, record fires, evaluate."""
    session_id = args.session_id
    mode = args.mode

    print(f"=== compression-monitor: session {session_id} ({mode} drift) ===\n")

    # Step 1: Register predictions
    print("Step 1: Registering predictions...")
    # Default firing order hypothesis: ghost_lexicon detects vocabulary loss first,
    # then behavioral_footprint, then semantic_drift (slowest to change)
    reg = run_script("preregister.py", [
        "register",
        "--session-id", session_id,
        "--firing-order", "ghost_lexicon", "behavioral_footprint", "semantic_drift",
    ])
    if reg.returncode != 0 and "already registered" not in reg.stdout:
        print(reg.stdout)
        print(reg.stderr)
        sys.exit(1)
    print(reg.stdout.strip())

    # Step 2: Run simulation and capture metrics
    print("\nStep 2: Running monitors on boundary data...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp = f.name

    # Import simulation directly to get structured results
    sys.path.insert(0, TOOLS_DIR)
    from simulate_boundary import generate_synthetic, apply_vocabulary_drift, \
        apply_topic_drift, apply_toolcall_drift, apply_combined_drift, evaluate, \
        SAMPLE_PRE_BOUNDARY, THRESHOLDS

    pre = generate_synthetic(SAMPLE_PRE_BOUNDARY, 20)
    if mode == "vocabulary":
        post = apply_vocabulary_drift(pre[:])
    elif mode == "topic":
        post = apply_topic_drift(pre[:])
    elif mode == "toolcalls":
        post = apply_toolcall_drift(pre[:])
    else:
        post = apply_combined_drift(pre[:])

    result = evaluate(pre, post)
    ghost_rate = result["ghost_lexicon"]["ghost_rate"]
    behavioral = result["behavioral_footprint"]["jaccard_distance"]
    topic = result["topic_drift"]["topic_drift_score"]

    print(f"  Ghost lexicon decay:  {ghost_rate:.0%}")
    print(f"  Behavioral distance:  {behavioral:.0%}")
    print(f"  Topic drift:          {topic:.0%}")
    print(f"  Alerts triggered:     {len(result['alerts'])}")
    for a in result["alerts"]:
        print(f"    ! {a}")

    # Step 3: Record which instruments fired (threshold-based)
    print("\nStep 3: Recording instrument fire order...")
    fire_order = []
    thresholds = [
        ("ghost_lexicon", ghost_rate, THRESHOLDS["ghost_rate"]),
        ("behavioral_footprint", behavioral, THRESHOLDS["jaccard_distance"]),
        ("semantic_drift", topic, THRESHOLDS["topic_drift_score"]),
    ]
    # Instruments that fired: sort by how much they exceeded threshold (most-exceeded first)
    # as a proxy for which would have fired earliest in a real session
    fired = [(name, val, thresh) for name, val, thresh in thresholds if val > thresh]
    not_fired = [name for name, val, thresh in thresholds if val <= thresh]
    fired.sort(key=lambda x: x[1] - x[2], reverse=True)

    for i, (name, val, thresh) in enumerate(fired):
        exchange_approx = 3 + i * 2  # synthetic approximation of when it would fire
        r = run_script("preregister.py", [
            "record-fire",
            "--session-id", session_id,
            "--instrument", name,
            "--exchange-number", str(exchange_approx),
        ])
        print(r.stdout.strip())
        fire_order.append(name)

    if not fired:
        print("  No instruments exceeded thresholds.")
    if not_fired:
        print(f"  Did not fire: {not_fired}")

    # Step 4: Evaluate against predictions
    print("\nStep 4: Evaluating against predictions...")
    eval_args = [
        "evaluate",
        "--session-id", session_id,
        "--actuals",
        f"ghost_lexicon={ghost_rate:.3f}",
        f"behavioral_footprint={behavioral:.3f}",
        f"semantic_drift={topic:.3f}",
    ]
    if fire_order:
        eval_args += ["--actual-firing-order"] + fire_order
    ev = run_script("preregister.py", eval_args)
    print(ev.stdout.strip())
    print()
    print("Run complete. Divergences between expected and observed firing order")
    print("indicate the compression event may span multiple distinct phenomena.")
    print("See Issue #5 for framing-level limits that no surface instrument can detect.")


def cmd_status(args):
    """Show current state of a session's predictions and observations."""
    r = run_script("preregister.py", ["list"])
    print(r.stdout)


def cmd_demo(args):
    """Run a complete demo: register, simulate combined drift, evaluate."""
    print("Running end-to-end demo with combined drift mode...\n")
    demo_id = f"demo-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    cmd_run(argparse.Namespace(session_id=demo_id, mode="combined"))


def main():
    parser = argparse.ArgumentParser(
        description="Unified compression monitor — register predictions, run monitors, evaluate firing order."
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Full pipeline: register, monitor, record fires, evaluate")
    run_p.add_argument("--session-id", required=True)
    run_p.add_argument("--mode", default="combined",
                       choices=["vocabulary", "topic", "toolcalls", "combined"],
                       help="Drift mode for simulation (default: combined)")

    status_p = sub.add_parser("status", help="Show registered sessions")
    status_p.add_argument("--session-id", default=None)

    sub.add_parser("demo", help="Run end-to-end demo with synthetic combined drift")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
