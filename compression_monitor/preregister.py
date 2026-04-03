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
  python3 preregister.py trend [--window 20] [--exit-on-regression]
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import linear_regression
from typing import Optional

APP_DIRNAME = "compression-monitor"

INSTRUMENTS = ["ghost_lexicon", "behavioral_footprint", "semantic_drift"]


def get_registry_file() -> Path:
    override = os.environ.get("COMPRESSION_MONITOR_STATE_DIR")
    if override:
        return Path(override).expanduser() / "preregister_state.json"

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / APP_DIRNAME / "preregister_state.json"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIRNAME / "preregister_state.json"

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser() / APP_DIRNAME / "preregister_state.json"

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home).expanduser() / APP_DIRNAME / "preregister_state.json"

    return Path.home() / ".local" / "state" / APP_DIRNAME / "preregister_state.json"


REGISTRY_FILE = str(get_registry_file())


def load_registry() -> dict:
    registry_path = Path(REGISTRY_FILE)
    if registry_path.exists():
        with registry_path.open() as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    registry_path = Path(REGISTRY_FILE)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w") as f:
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
        # Store raw actuals so SessionTrendAnalyzer can read them.
        "actuals_raw": actuals,
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
    items = reg.items()
    if args.session_id:
        entry = reg.get(args.session_id)
        if entry is None:
            print(f"Session {args.session_id} not found.")
            sys.exit(1)
        items = [(args.session_id, entry)]

    for sid, entry in items:
        status = "evaluated" if entry.get("evaluated") else "pending"
        fires = entry.get("observed_fires", {})
        fire_str = f", fires recorded: {list(fires.keys())}" if fires else ""
        print(f"  {sid}: {status} (registered {entry['registered_at'][:10]}){fire_str}")


# ---------------------------------------------------------------------------
# SessionTrendAnalyzer — cross-session compression trend detection (#9)
# ---------------------------------------------------------------------------

@dataclass
class SessionPoint:
    session_id: str
    registered_at: str  # ISO timestamp (used for sort order)
    ghost_actual: Optional[float]
    behavioral_actual: Optional[float]
    semantic_actual: Optional[float]
    firing_order_match: Optional[bool]  # predicted vs observed order


@dataclass
class MetricTrend:
    metric: str
    slope: float          # OLS slope; positive = worsening (scores rising = more drift)
    direction: str        # 'degrading' | 'improving' | 'stable'
    window_size: int      # number of data points used (may be < window if sparse)


@dataclass
class SessionTrendReport:
    metric_trends: list
    firing_order_consistency: Optional[float]  # fraction of sessions where order matched
    firing_order_trend: str                    # 'degrading' | 'improving' | 'stable'
    any_regression: bool
    window_size: int


_SLOPE_THRESHOLD = 0.005  # |slope| below this is "stable"


def _direction(slope: float, metric: str) -> str:
    """
    Interpret slope sign as degrading/improving/stable.

    For ghost_lexicon, behavioral_footprint, semantic_drift:
      - These are *consistency* or *similarity* scores stored by the user.
      - Convention: actuals represent magnitude of drift/decay, so positive slope = worsening.
      - If the caller uses scores where higher = better (e.g. similarity), the sign will
        naturally flip — the direction label will still be correct.
    """
    if abs(slope) < _SLOPE_THRESHOLD:
        return "stable"
    return "degrading" if slope > 0 else "improving"


class SessionTrendAnalyzer:
    """
    Reads preregister_state.json and computes cross-session behavioral slope.

    Only uses evaluated sessions (evaluated=True) that have actuals_raw recorded.
    Requires Python 3.10+ for statistics.linear_regression.
    """

    def __init__(self, registry_path: Optional[str] = None, window: int = 20):
        self.registry_path = registry_path or REGISTRY_FILE
        self.window = window

    def _load_points(self) -> list:
        reg = load_registry() if self.registry_path == REGISTRY_FILE else self._load_from(self.registry_path)
        points = []
        for entry in reg.values():
            if not entry.get("evaluated"):
                continue
            ev = entry.get("evaluation", {})
            actuals = ev.get("actuals_raw", {})
            points.append(SessionPoint(
                session_id=entry["session_id"],
                registered_at=entry["registered_at"],
                ghost_actual=actuals.get("ghost_lexicon"),
                behavioral_actual=actuals.get("behavioral_footprint"),
                semantic_actual=actuals.get("semantic_drift"),
                firing_order_match=ev.get("order_match"),
            ))
        points.sort(key=lambda p: p.registered_at)
        return points[-self.window:]  # last N

    def _load_from(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open() as f:
            return json.load(f)

    def _slope_trend(self, values: list) -> Optional[MetricTrend]:
        """Compute OLS slope over (index, value) pairs. Returns None if <2 data points."""
        pairs = [(float(i), v) for i, v in enumerate(values) if v is not None]
        if len(pairs) < 2:
            return None
        xs, ys = zip(*pairs)
        slope, _ = linear_regression(xs, ys)
        return slope, len(pairs)

    def analyze(self) -> SessionTrendReport:
        points = self._load_points()
        actual_window = len(points)

        metric_trends = []
        for attr, name in [
            ("ghost_actual", "ghost_lexicon"),
            ("behavioral_actual", "behavioral_footprint"),
            ("semantic_actual", "semantic_drift"),
        ]:
            vals = [getattr(p, attr) for p in points]
            result = self._slope_trend(vals)
            if result is not None:
                slope, n = result
                metric_trends.append(MetricTrend(
                    metric=name,
                    slope=slope,
                    direction=_direction(slope, name),
                    window_size=n,
                ))

        # Firing order consistency
        order_results = [p.firing_order_match for p in points if p.firing_order_match is not None]
        if order_results:
            consistency = sum(1 for x in order_results if x) / len(order_results)
            # Trend: compare first half vs second half
            if len(order_results) >= 4:
                mid = len(order_results) // 2
                first_half = sum(1 for x in order_results[:mid] if x) / mid
                second_half = sum(1 for x in order_results[mid:] if x) / (len(order_results) - mid)
                diff = second_half - first_half
                if abs(diff) < 0.05:
                    fo_trend = "stable"
                elif diff < 0:
                    fo_trend = "degrading"
                else:
                    fo_trend = "improving"
            else:
                fo_trend = "stable"
        else:
            consistency = None
            fo_trend = "stable"

        any_regression = (
            any(t.direction == "degrading" for t in metric_trends)
            or fo_trend == "degrading"
        )

        return SessionTrendReport(
            metric_trends=metric_trends,
            firing_order_consistency=consistency,
            firing_order_trend=fo_trend,
            any_regression=any_regression,
            window_size=actual_window,
        )

    @staticmethod
    def format_report(report: SessionTrendReport) -> str:
        lines = [f"=== Cross-session compression trend (last {report.window_size} sessions) ==="]
        col = 26
        for t in report.metric_trends:
            label = f"{t.metric}:"
            slope_str = f"slope={t.slope:+.4f}"
            direction = t.direction.upper() if t.direction != "stable" else "stable"
            lines.append(f"  {label:<{col}} {slope_str:<18} {direction}  (n={t.window_size})")

        if report.firing_order_consistency is not None:
            n_total = report.window_size
            n_match = round(report.firing_order_consistency * n_total)
            pct = int(report.firing_order_consistency * 100)
            fo_label = f"firing_order_match:"
            fo_value = f"{n_match}/{n_total} ({pct}%)"
            fo_dir = f"— {report.firing_order_trend.upper()}" if report.firing_order_trend != "stable" else "— stable"
            lines.append(f"  {fo_label:<{col}} {fo_value:<18} {fo_dir}")
        else:
            lines.append(f"  firing_order_match:        (no data)")

        lines.append("")
        if report.any_regression:
            degrading = [t.metric for t in report.metric_trends if t.direction == "degrading"]
            parts = degrading[:]
            if report.firing_order_trend == "degrading":
                parts.append("firing_order")
            lines.append(f"⚠  Regression detected: {', '.join(parts)} declining")
        else:
            lines.append("✓  No regression detected across sampled window.")

        if report.window_size < 3:
            lines.append("")
            lines.append(f"  Note: only {report.window_size} evaluated session(s) found. Trend is unreliable below ~5 sessions.")

        return "\n".join(lines)


def cmd_trend(args):
    analyzer = SessionTrendAnalyzer(window=args.window)
    report = analyzer.analyze()
    print(SessionTrendAnalyzer.format_report(report))
    if args.exit_on_regression and report.any_regression:
        sys.exit(1)


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

    # record-fire
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
    list_p = sub.add_parser("list", help="Show all registered sessions")
    list_p.add_argument("--session-id", default=None, help="Filter output to a single session")

    # trend — cross-session behavioral slope analysis
    trend_p = sub.add_parser("trend", help="Analyze cross-session compression trend over last N evaluated sessions")
    trend_p.add_argument("--window", type=int, default=20,
                         help="Number of most recent evaluated sessions to analyze (default: 20)")
    trend_p.add_argument("--exit-on-regression", action="store_true",
                         help="Exit with code 1 if any regression detected (useful as CI gate)")

    args = parser.parse_args()
    if args.command == "register":
        cmd_register(args)
    elif args.command == "record-fire":
        cmd_record_fire(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "trend":
        cmd_trend(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
