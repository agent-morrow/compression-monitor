#!/usr/bin/env python3
"""
preregister.py — Pre-commit predictions before epoch boundaries.

Records directional predictions, expected firing order, and max-latency bounds
per instrument before a compression event. Post-boundary, compare actuals against
registered predictions. Divergence is itself a signal.

See: https://github.com/agent-morrow/compression-monitor/issues/3
"""

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path


REGISTRY_FILE = Path("compression_preregister.json")


def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text())
    return {"registrations": [], "evaluations": []}


def save_registry(reg: dict):
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2))


def cmd_register(args):
    """Pre-register predictions before a suspected compression boundary."""
    reg = load_registry()

    entry = {
        "id": f"reg_{int(datetime.now(timezone.utc).timestamp())}",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "predictions": {
            "ghost_lexicon": {
                "fires": args.ghost_fires,
                "threshold": args.ghost_threshold,
                "max_latency_exchanges": args.ghost_latency,
            },
            "behavioral_footprint": {
                "fires": args.behavioral_fires,
                "threshold": args.behavioral_threshold,
                "max_latency_exchanges": args.behavioral_latency,
            },
            "semantic_drift": {
                "fires": args.semantic_fires,
                "threshold": args.semantic_threshold,
                "max_latency_exchanges": args.semantic_latency,
            },
        },
        "expected_firing_order": args.firing_order,
        "notes": args.notes or "",
    }

    reg["registrations"].append(entry)
    save_registry(reg)
    print(f"Registered: {entry['id']} — {args.label}")
    print(f"Predictions: ghost_lexicon={'fires' if args.ghost_fires else 'silent'}, "
          f"behavioral={'fires' if args.behavioral_fires else 'silent'}, "
          f"semantic={'fires' if args.semantic_fires else 'silent'}")
    print(f"Expected firing order: {args.firing_order}")


def cmd_evaluate(args):
    """Record actuals after the boundary and compare against registered predictions."""
    reg = load_registry()

    # Find the registration to evaluate
    target = None
    for r in reg["registrations"]:
        if r["id"] == args.registration_id or r["label"] == args.registration_id:
            target = r
            break

    if not target:
        print(f"ERROR: No registration found for '{args.registration_id}'", file=sys.stderr)
        print("Registered entries:")
        for r in reg["registrations"]:
            print(f"  {r['id']} — {r['label']} ({r['registered_at'][:19]})")
        sys.exit(1)

    actuals = {
        "ghost_lexicon": {"fired": args.ghost_fired, "latency": args.ghost_latency_actual},
        "behavioral_footprint": {"fired": args.behavioral_fired, "latency": args.behavioral_latency_actual},
        "semantic_drift": {"fired": args.semantic_fired, "latency": args.semantic_latency_actual},
    }

    actual_order = args.actual_firing_order

    # Compare predictions vs actuals
    divergences = []
    for instrument, pred in target["predictions"].items():
        actual = actuals[instrument]
        if pred["fires"] != actual["fired"]:
            divergences.append({
                "instrument": instrument,
                "type": "direction_mismatch",
                "predicted": "fires" if pred["fires"] else "silent",
                "actual": "fired" if actual["fired"] else "silent",
            })
        elif actual["fired"] and actual["latency"] is not None:
            if actual["latency"] > pred["max_latency_exchanges"]:
                divergences.append({
                    "instrument": instrument,
                    "type": "latency_exceeded",
                    "predicted_max": pred["max_latency_exchanges"],
                    "actual_latency": actual["latency"],
                })

    order_match = (actual_order == target["expected_firing_order"]) if actual_order else None

    evaluation = {
        "registration_id": target["id"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "actuals": actuals,
        "actual_firing_order": actual_order,
        "divergences": divergences,
        "order_match": order_match,
        "notes": args.notes or "",
    }

    reg["evaluations"].append(evaluation)
    save_registry(reg)

    print(f"\nEvaluation for: {target['label']} ({target['id']})")
    print(f"Registered: {target['registered_at'][:19]}")
    print(f"Evaluated:  {evaluation['evaluated_at'][:19]}")
    print()

    if not divergences:
        print("✓ All directional predictions matched.")
    else:
        print(f"⚠ {len(divergences)} divergence(s) detected:")
        for d in divergences:
            if d["type"] == "direction_mismatch":
                print(f"  {d['instrument']}: predicted {d['predicted']}, actual {d['actual']}")
            elif d["type"] == "latency_exceeded":
                print(f"  {d['instrument']}: latency exceeded — predicted max {d['predicted_max']}, actual {d['actual_latency']} exchanges")

    if order_match is True:
        print("✓ Firing order matched.")
    elif order_match is False:
        print(f"⚠ Firing order mismatch:")
        print(f"  Expected: {target['expected_firing_order']}")
        print(f"  Actual:   {actual_order}")
    
    print()
    print("Note: divergences are signals, not failures. An instrument that fires late or")
    print("not at all may indicate framing-level compression (Issue #5) rather than an")
    print("instrument error. Interpret in context.")


def cmd_list(args):
    """List all registrations and their evaluation status."""
    reg = load_registry()
    evaluated_ids = {e["registration_id"] for e in reg["evaluations"]}

    if not reg["registrations"]:
        print("No registrations yet. Use 'register' to pre-commit predictions.")
        return

    for r in reg["registrations"]:
        status = "✓ evaluated" if r["id"] in evaluated_ids else "○ pending"
        print(f"{status} | {r['id']} | {r['label']} | {r['registered_at'][:19]}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-register compression predictions and evaluate actuals post-boundary."
    )
    sub = parser.add_subparsers(dest="command")

    # register
    reg_p = sub.add_parser("register", help="Pre-commit predictions before a suspected boundary")
    reg_p.add_argument("label", help="Human-readable label for this boundary event")
    reg_p.add_argument("--ghost-fires", action="store_true", default=True, help="Predict ghost_lexicon fires (default: True)")
    reg_p.add_argument("--no-ghost-fires", dest="ghost_fires", action="store_false")
    reg_p.add_argument("--ghost-threshold", type=float, default=0.15, help="Expected decay_score threshold")
    reg_p.add_argument("--ghost-latency", type=int, default=5, help="Max exchanges after boundary before ghost_lexicon fires")
    reg_p.add_argument("--behavioral-fires", action="store_true", default=True)
    reg_p.add_argument("--no-behavioral-fires", dest="behavioral_fires", action="store_false")
    reg_p.add_argument("--behavioral-threshold", type=float, default=0.2)
    reg_p.add_argument("--behavioral-latency", type=int, default=10)
    reg_p.add_argument("--semantic-fires", action="store_true", default=False)
    reg_p.add_argument("--no-semantic-fires", dest="semantic_fires", action="store_false")
    reg_p.add_argument("--semantic-threshold", type=float, default=0.3)
    reg_p.add_argument("--semantic-latency", type=int, default=15)
    reg_p.add_argument("--firing-order", nargs="+", default=["ghost_lexicon", "behavioral_footprint", "semantic_drift"],
                       help="Expected order instruments will fire")
    reg_p.add_argument("--notes", help="Optional notes")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="Record actuals and compare against predictions")
    eval_p.add_argument("registration_id", help="Registration ID or label to evaluate")
    eval_p.add_argument("--ghost-fired", action="store_true", default=False)
    eval_p.add_argument("--no-ghost-fired", dest="ghost_fired", action="store_false")
    eval_p.add_argument("--ghost-latency-actual", type=int, default=None)
    eval_p.add_argument("--behavioral-fired", action="store_true", default=False)
    eval_p.add_argument("--no-behavioral-fired", dest="behavioral_fired", action="store_false")
    eval_p.add_argument("--behavioral-latency-actual", type=int, default=None)
    eval_p.add_argument("--semantic-fired", action="store_true", default=False)
    eval_p.add_argument("--no-semantic-fired", dest="semantic_fired", action="store_false")
    eval_p.add_argument("--semantic-latency-actual", type=int, default=None)
    eval_p.add_argument("--actual-firing-order", nargs="+", default=None)
    eval_p.add_argument("--notes", help="Optional notes")

    # list
    sub.add_parser("list", help="List registrations and evaluation status")

    args = parser.parse_args()
    if args.command == "register":
        cmd_register(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
