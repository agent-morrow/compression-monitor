"""
reorientation_cost_tracker.py
──────────────────────────────
Measures the TOKENIZE/ATTEND burst cost at context compaction or session rotation
boundaries — the measurably higher latency and tool-call density in the first few
post-boundary exchanges before the agent re-establishes steady state.

Background
──────────
When an agent's context is compacted or a new session starts, the first N tool calls
show a distinct orientation pattern:
  - Higher tool-call density (more calls per exchange)
  - Higher latency (re-reading state from durable storage)
  - Characteristic call cluster: memory reads → state verification → task re-entry

The ReorientationCostTracker captures this burst window and computes:
  - burst_cost: ratio of first-N-call density vs steady-state density
  - burst_latency: ratio of first-N-call average latency vs steady-state
  - recovery_window: number of calls until density normalizes

This is complementary to CCS (which measures run-wide drift). CCS tells you WHAT
changed; the reorientation cost tells you HOW EXPENSIVE each boundary was.

Hypothesis under test (BIRCH study)
────────────────────────────────────
State complexity at rotation (number of open threads, pending tasks) predicts
burst cost better than context length at rotation.

Measurement variables:
  - context_length_at_rotation: float (fraction of context window used, 0..1)
  - open_threads_at_rotation: int (proxy: unresolved issue comments + active reply
    threads + pending outbox items)
  - burst_cost: float (first-3-calls density / steady-state density)

Call record_boundary() at each rotation event with these variables.
Call summarize() to see the cross-rotation regression table.

Usage
─────
    tracker = ReorientationCostTracker()

    # At each tool call:
    tracker.record_call(
        call_index=i,
        tool_name="memory_search",
        latency_ms=1240.0,
    )

    # At each rotation boundary:
    tracker.record_boundary(
        boundary_index=current_call_index,
        context_length_at_rotation=0.72,      # fraction of context used
        open_threads_at_rotation=21,           # proxy for state complexity
    )

    # Summarize after several rotation events:
    report = tracker.summarize()
    print(json.dumps(report, indent=2))
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple


BURST_WINDOW_CALLS = 3   # first N calls post-boundary = burst window
STEADY_STATE_MIN = 5     # minimum calls needed to compute steady-state baseline


@dataclass
class CallRecord:
    call_index: int
    tool_name: str
    latency_ms: float
    recorded_at: float = field(default_factory=time.time)


@dataclass
class BoundaryRecord:
    boundary_index: int          # call_index at which rotation fired
    context_length: float        # fraction of context window used at rotation (0..1)
    open_threads: int            # open-thread count at rotation
    recorded_at: float = field(default_factory=time.time)

    # populated by tracker after enough post-boundary calls arrive
    burst_cost: Optional[float] = None        # post-boundary density / pre-boundary density
    burst_latency: Optional[float] = None     # post-boundary avg latency / pre-boundary avg
    recovery_window: Optional[int] = None     # calls until density normalizes


@dataclass
class ReorientationCostTracker:
    """
    Track burst cost across session rotation or compaction boundaries.

    Parameters
    ----------
    burst_window : int
        Number of calls after a boundary that constitute the burst window.
        Default: 3 (TOKENIZE/ATTEND phase typically completes within first 3 calls).
    steady_state_min : int
        Minimum pre-boundary calls required to compute a valid steady-state baseline.
    """

    burst_window: int = BURST_WINDOW_CALLS
    steady_state_min: int = STEADY_STATE_MIN

    _calls: List[CallRecord] = field(default_factory=list, init=False)
    _boundaries: List[BoundaryRecord] = field(default_factory=list, init=False)

    # ── recording ─────────────────────────────────────────────────────────────

    def record_call(
        self,
        call_index: int,
        tool_name: str,
        latency_ms: float,
    ) -> None:
        """Record a single tool call with its index and latency."""
        self._calls.append(CallRecord(
            call_index=call_index,
            tool_name=tool_name,
            latency_ms=latency_ms,
        ))

    def record_boundary(
        self,
        boundary_index: int,
        context_length_at_rotation: float,
        open_threads_at_rotation: int,
    ) -> None:
        """
        Record a rotation/compaction boundary event.

        Call this when a context rotation fires — before the post-boundary calls
        are recorded, so the burst window will be populated by subsequent
        record_call() calls.

        Parameters
        ----------
        boundary_index : int
            The call_index at which the rotation fired (i.e., the last pre-boundary
            call index, or the first post-boundary call index).
        context_length_at_rotation : float
            Fraction of context window consumed at rotation (e.g., 0.72 for 72%).
        open_threads_at_rotation : int
            Proxy for state complexity: number of open external threads, pending
            outbox items, and unresolved reply chains at rotation time.
        """
        self._boundaries.append(BoundaryRecord(
            boundary_index=boundary_index,
            context_length=context_length_at_rotation,
            open_threads=open_threads_at_rotation,
        ))

    # ── analysis ──────────────────────────────────────────────────────────────

    def _compute_boundary(self, b: BoundaryRecord) -> None:
        """Populate burst metrics for a boundary once enough post-boundary calls exist."""
        pre_calls = [c for c in self._calls if c.call_index < b.boundary_index]
        post_calls = [c for c in self._calls if c.call_index >= b.boundary_index]

        if len(pre_calls) < self.steady_state_min:
            return  # not enough pre-boundary data
        if len(post_calls) < self.burst_window:
            return  # burst window not yet complete

        # Steady-state density: calls per unit time (using a rolling window)
        pre_window = pre_calls[-max(self.steady_state_min, 10):]
        if len(pre_window) >= 2:
            pre_span = pre_window[-1].recorded_at - pre_window[0].recorded_at
            pre_density = len(pre_window) / max(pre_span, 1.0)
        else:
            pre_density = 1.0

        burst = post_calls[:self.burst_window]
        if len(burst) >= 2:
            burst_span = burst[-1].recorded_at - burst[0].recorded_at
            burst_density = len(burst) / max(burst_span, 1.0)
        else:
            burst_density = pre_density  # no meaningful burst measurement

        b.burst_cost = burst_density / pre_density if pre_density > 0 else 1.0

        # Latency ratio
        pre_lats = [c.latency_ms for c in pre_window]
        burst_lats = [c.latency_ms for c in burst]
        if pre_lats and burst_lats:
            b.burst_latency = mean(burst_lats) / mean(pre_lats) if mean(pre_lats) > 0 else 1.0

        # Recovery window: how many calls until density normalizes to ≤1.2× steady state
        recovery = self.burst_window
        for i, pc in enumerate(post_calls[self.burst_window:], start=self.burst_window):
            window = post_calls[max(0, i - self.burst_window):i + 1]
            if len(window) >= 2:
                w_span = window[-1].recorded_at - window[0].recorded_at
                w_density = len(window) / max(w_span, 1.0)
                if w_density <= pre_density * 1.2:
                    recovery = i
                    break
        b.recovery_window = recovery

    def summarize(self) -> Dict[str, Any]:
        """
        Return a summary of burst cost measurements across all boundaries.

        Includes:
          - per-boundary measurements
          - correlation table: context_length vs burst_cost, open_threads vs burst_cost
          - overall mean burst_cost and burst_latency
        """
        # Populate any incomplete boundaries
        for b in self._boundaries:
            if b.burst_cost is None:
                self._compute_boundary(b)

        completed = [b for b in self._boundaries if b.burst_cost is not None]

        if not completed:
            return {
                "status": "insufficient_data",
                "boundaries_recorded": len(self._boundaries),
                "calls_recorded": len(self._calls),
                "note": f"Need ≥{self.steady_state_min} pre-boundary + {self.burst_window} post-boundary calls per event",
            }

        burst_costs = [b.burst_cost for b in completed]
        burst_lats = [b.burst_latency for b in completed if b.burst_latency is not None]

        # Simple correlation proxy: rank correlation substitute (Spearman approx)
        def rank_corr(xs: List[float], ys: List[float]) -> Optional[float]:
            if len(xs) < 3:
                return None
            n = len(xs)
            rx = sorted(range(n), key=lambda i: xs[i])
            ry = sorted(range(n), key=lambda i: ys[i])
            rank_x = [0.0] * n
            rank_y = [0.0] * n
            for rank, idx in enumerate(rx):
                rank_x[idx] = float(rank)
            for rank, idx in enumerate(ry):
                rank_y[idx] = float(rank)
            d2 = sum((rank_x[i] - rank_y[i]) ** 2 for i in range(n))
            return 1 - (6 * d2) / (n * (n**2 - 1))

        ctx_lens = [b.context_length for b in completed]
        open_thrs = [float(b.open_threads) for b in completed]

        ctx_corr = rank_corr(ctx_lens, burst_costs)
        thr_corr = rank_corr(open_thrs, burst_costs)

        return {
            "tool": "compression-monitor/reorientation_cost_tracker",
            "version": "0.1.0",
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "boundaries_analyzed": len(completed),
            "calls_recorded": len(self._calls),
            "mean_burst_cost": round(mean(burst_costs), 3),
            "mean_burst_latency": round(mean(burst_lats), 3) if burst_lats else None,
            "burst_cost_stdev": round(stdev(burst_costs), 3) if len(burst_costs) >= 2 else None,
            "predictor_correlations": {
                "context_length_vs_burst_cost": round(ctx_corr, 3) if ctx_corr is not None else None,
                "open_threads_vs_burst_cost": round(thr_corr, 3) if thr_corr is not None else None,
                "hypothesis": "open_threads is better predictor than context_length",
                "hypothesis_supported": (
                    abs(thr_corr) > abs(ctx_corr)
                    if (thr_corr is not None and ctx_corr is not None)
                    else None
                ),
            },
            "per_boundary": [
                {
                    "boundary_index": b.boundary_index,
                    "context_length": b.context_length,
                    "open_threads": b.open_threads,
                    "burst_cost": round(b.burst_cost, 3) if b.burst_cost is not None else None,
                    "burst_latency": round(b.burst_latency, 3) if b.burst_latency is not None else None,
                    "recovery_window": b.recovery_window,
                }
                for b in completed
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.summarize(), indent=2)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Quick demo with synthetic data
    tracker = ReorientationCostTracker()

    print("ReorientationCostTracker — demo with synthetic rotation event")
    print("=" * 64)

    # Simulate 20 steady-state calls
    for i in range(20):
        tracker.record_call(i, "exec" if i % 3 else "memory_search", 800.0 + i * 10)
        time.sleep(0.001)

    # Rotation fires at call 20
    tracker.record_boundary(
        boundary_index=20,
        context_length_at_rotation=0.72,
        open_threads_at_rotation=21,
    )

    # Simulate burst: first 3 post-boundary calls are fast orientation calls (dense cluster)
    for i in range(20, 20 + tracker.burst_window):
        tracker.record_call(i, "memory_search", 300.0)  # fast, many in short time
        time.sleep(0.0001)  # very short interval = high density

    # Then steady-state resumes
    for i in range(20 + tracker.burst_window, 30):
        tracker.record_call(i, "exec", 900.0)
        time.sleep(0.001)

    print(tracker.to_json())
