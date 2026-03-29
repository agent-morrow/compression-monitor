#!/usr/bin/env python3
"""
preregister.py — Pre-commit predictions before compression events.

Records directional predictions, expected firing order, and max-latency bounds
per instrument before a compression event. Post-boundary, compare actuals against
predictions. Divergence is the signal, not a calibration failure.

Cairn insight (2026-03-28): if instruments disagree on whether an event occurred,
"compression event" may not be a unified phenomenon. The TEMPORAL ORDERING of
instrument firing is the primary finding — which instrument detects change first
tells you more about the event's architecture than whether they converge.

Usage:
  python3 preregister.py register --session-id SID [--firing-order ghost_lexicon behavioral_footprint semantic_drift]
  python3 preregister.py record-fire --session-id SID --instrument ghost_lexicon
  python3 preregister.py evaluate --session-id SID [--actuals ghost_lexicon=0.3 behavioral=0.1 semantic=0.15]
  python3 preregister.py list
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "preregister_state.json")


def load_registry() -> dict:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)


def cmd_register(args):
    reg = load_registry()
    if args.session_id in reg:
        print(f"Session {args.session_id} already registered. Use a new session-id or delete manually.")
        sys.exit(1)
    entry = {
        "session_id": args.session_id,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "predictions": {
            "ghost_lexicon": {"direction": args.ghost_direction, "max_latency_exchanges": args.ghost_latency},
            "behavioral_footprint": {"direction": args.behavioral_direction, "max_latency_exchanges": args.behavioral_latency},
            "semantic_drift": {"direction": args.semantic_direction, "max_latency_exchanges": args.semantic_latency},
        },
        "expected_firing_order": args.firing_order,
        "observed_fires": {},  # instrument -> {"timestamp": ..., "exchange_number": ...}
        "evaluated": False,
    }
    reg[args.session_id] = entry
    save_registry(reg)
    print(f"Registered session: {args.session_id}")
    print(f"Expected firing order: {args.firing_order}")
    print(f"Note: use 'record-fire' as each instrument detects drift; 'evaluate' to compare against predictions.")


def cmd_record_fire(args):
    """Log when an instrument actually fires — call this as each monitor detects a change."""
    reg = load_registry()
    if args.session_id not in reg:
        print(f"Session {args.session_id} not found. Register first.")
        sys.exit(1)
    entry = reg[args.session_id]
    ts = datetime.now(timezone.utc).isoformat()
    if args.instrument in entry["observed_fires"]:
        print(f"Instrument {args.instrument} already recorded for session {args.session_id}.")
        print(f"Recorded at: {entry['observed_fires'][args.instrument]['timestamp']}")
    else:
        entry["observed_fires"][args.instrument] = {
            "timestamp": ts,
            "exchange_number": args.exchange_number,
        }
        save_registry(reg)
        # Show current observed order
        fires = sorted(entry["observed_fires"].items(), key=lambda x: x[1]["timestamp"])
        observed_order = [f[0] for f in fires]
        expected_order = entry["expected_firing_order"]
        match = (observed_order == expected_order[:len(observed_order)])
        print(f"Recorded: {args.instrument} fired at exchange {args.exchange_number} ({ts})")
        print(f"Observed firing order so far: {observed_order}")
        if len(observed_order) == len(expected_order):
            status = "[OK] order matches prediction" if observed_order == expected_order else f"[WARN] order diverges - expected {expected_order}"
            print(f"All instruments fired. {status}")
        else:
            remaining = [i for i in expected_order if i not in observed_order]
            print(f"Waiting for: {remaining}")


def cmd_evaluate(args):
    reg = load_registry()
    if args.session_id not in reg:
        print(f"Session {args.session_id} not found.")
        sys.exit(1)
    target = reg[args.session_id]

    # Parse actuals from --actuals key=value pairs
    actuals = {}
    if args.actuals:
        for a in args.actuals:
            k, v = a.split("=")
            actuals[k.strip()] = float(v.strip())

    deviations = []
    latency_issues = []
    fires = target.get("observed_fires", {})

    for instrument, pred in target["predictions"].items():
        if instrument in actuals:
            actual = actuals[instrument]
            pred_dir = pred["direction"]
            pred_max = pred.get("max_latency_exchanges")
            direction_ok = True
            if pred_dir == "increase" and actual <= 0:
                direction_ok = False
            elif pred_dir == "decrease" and actual >= 0:
                direction_ok = False
            if not direction_ok:
                deviations.append({"instrument": instrument, "predicted": pred_dir, "actual": actual})
            if pred_max and instrument in fires:
                actual_latency = fires[instrument].get("exchange_number", 0)
                if actual_latency > pred_max:
                    latency_issues.append({"instrument": instrument, "predicted_max": pred_max, "actual_latency": actual_latency})

    # Firing order analysis — PRIMARY finding per cairn's insight
    expected_order = target["expected_firing_order"]
    observed_order = [f[0] for f in sorted(fires.items(), key=lambda x: x[1]["timestamp"])] if fires else None
    # If no recorded fires, fall back to args.actual_firing_order for backward compat
    if not observed_order and args.actual_firing_order:
        observed_order = args.actual_firing_order

    order_match = (observed_order == expected_order) if observed_order else None

    result = {
        "session_id": args.session_id,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "deviations": deviations,
        "latency_issues": latency_issues,
        "expected_firing_order": expected_order,
        "observed_firing_order": observed_order,
        "order_match": order_match,
    }

    target["evaluation"] = result
    target["evaluated"] = True
    save_registry(reg)

    print(f"=== Evaluation: {args.session_id} ===")

    # Firing order is the PRIMARY finding
    print(f"\nFiring order (primary finding):")
    print(f"  Expected: {expected_order}")
    print(f"  Observed: {observed_order or '(not recorded)'}")
    if order_match is True:
        print(f"  ✓ Order matched — instruments agree on event architecture")
    elif order_match is False:
        print("  [WARN] Order diverged - instruments may be measuring distinct phenomena")
        # Identify which instrument fired unexpectedly early/late
        if observed_order and expected_order:
            for i, (exp, obs) in enumerate(zip(expected_order, observed_order)):
                if exp != obs:
                    print(f"    Position {i+1}: expected {exp}, got {obs}")
            print(f"  Interpretation: early-firing instrument detected the event first.")
            print(f"  This is architectural information about the compression event, not noise.")
    elif observed_order is None:
        print(f"  (no firing order recorded — use 'record-fire' during monitoring)")

    # Magnitude/direction deviations are secondary
    if deviations or latency_issues:
        print(f"\nValue deviations (secondary):")
        for d in deviations:
            print(f"  {d['instrument']}: predicted {d['predicted']}, actual {d['actual']}")
        for d in latency_issues:
            print(f"  {d['instrument']}: latency exceeded — predicted max {d['predicted_max']}, actual {d['actual_latency']} exchanges")
    else:
        print(f"\nValue deviations: none (or not provided)")

    print(f"\nNote: divergences are signals, not failures. Firing-order mismatch means")
    print(f"'compression_event' may not be a unified phenomenon. See Issue #5 for limits.")


def cmd_list(args):
    reg = load_registry()
    if not reg:
        print("No sessions registered.")
        return
    for sid, entry in reg.items():
        status = "evaluated" if entry.get("evaluated") else "pending"
        fires = entry.get("observed_fires", {})
        fire_str = f", fires recorded: {list(fires.keys())}" if fires else ""
        print(f"  {sid}: {status} (registered {entry['registered_at'][:10]}){fire_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-register compression event predictions. Firing order is the primary finding."
    )
    sub = parser.add_subparsers(dest="command")

    # register
    reg_p = sub.add_parser("register", help="Pre-commit predictions before a session boundary")
    reg_p.add_argument("--session-id", required=True)
    reg_p.add_argument("--ghost-direction", default="decrease", choices=["increase", "decrease", "stable"])
    reg_p.add_argument("--ghost-latency", type=int, default=10, help="Max exchanges before ghost_lexicon fires")
    reg_p.add_argument("--behavioral-direction", default="decrease", choices=["increase", "decrease", "stable"])
    reg_p.add_argument("--behavioral-latency", type=int, default=15)
    reg_p.add_argument("--semantic-direction", default="decrease", choices=["increase", "decrease", "stable"])
    reg_p.add_argument("--semantic-latency", type=int, default=20)
    reg_p.add_argument("--firing-order", nargs="+",
                       default=["ghost_lexicon", "behavioral_footprint", "semantic_drift"],
                       help="Expected order instruments will fire (primary prediction)")

    # record-fire — new command
    fire_p = sub.add_parser("record-fire", help="Log when an instrument detects drift (call per instrument as it fires)")
    fire_p.add_argument("--session-id", required=True)
    fire_p.add_argument("--instrument", required=True, choices=["ghost_lexicon", "behavioral_footprint", "semantic_drift"])
    fire_p.add_argument("--exchange-number", type=int, default=0, help="Exchange number when instrument fired")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Compare actuals against predictions post-boundary")
    eval_p.add_argument("--session-id", required=True)
    eval_p.add_argument("--actuals", nargs="+", help="key=value pairs, e.g. ghost_lexicon=0.3")
    eval_p.add_argument("--actual-firing-order", nargs="+", default=None,
                        help="Fallback if record-fire was not used")

    # list
    sub.add_parser("list", help="Show all registered sessions")

    args = parser.parse_args()
    if args.command == "register":
        cmd_register(args)
    elif args.command == "record-fire":
        cmd_record_fire(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
