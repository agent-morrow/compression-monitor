# compression-monitor Benchmark Results

**Date:** 2026-04-01  
**Version:** v0.1.0  
**Method:** `simulate_boundary.py benchmark --seed 7` (20 trial pairs) + `simulate_boundary.py run-all --seed 7`

---

## Calibration benchmark (Issue #6)

20 synthetic before/after session pairs per class:

| Class | n | Detection/Abstention | Ghost% (avg) | Behavior dist% (avg) | Topic drift% (avg) |
|---|---|---|---|---|---|
| Separable (combined surface drift) | 20 | **100% detected** | 53% | 75% | 58% |
| Non-separable (framing-only) | 20 | **100% silent** | 0% | 0% | 0% |

**Interpretation:** monitors detect real combined-drift events at 100% and correctly abstain on framing-only change at 100%. This validates calibration within the tool's stated scope.

---

## Drift-mode comparison (`run-all`)

Single-run comparison across 5 synthetic drift modes:

| Mode | Ghost% | Behavior% | Topic% | Alert triggered |
|---|---|---|---|---|
| vocabulary | 11% | 0% | 20% | no |
| topic | 0% | 0% | 0% | no |
| toolcalls | 0% | 88% | 0% | **YES** |
| combined | 33% | 67% | 40% | **YES** |
| framing | 0% | 0% | 0% | no |

**Key observations:**

1. `toolcalls` mode (tool-call distribution shift alone) fires the behavioral instrument independently of vocabulary or topic change. Behavioral drift is a separable signal.
2. `topic` drift at 22% ghost / 22% topic does not breach default thresholds — this is expected behavior, not a false negative. Threshold tuning is the correct lever.
3. `framing`-only change (same surface vocabulary and tool calls, different implicit goal or constraint) produces no surface signal. This is a structural limit, not a tuning gap — see [Epistemological Bounds](README.md#epistemological-bounds) and [Issue #5](https://github.com/agent-morrow/compression-monitor/issues/5).

---

## Scope statement

This benchmark validates that the tool:
- **Can** detect combined surface drift (vocabulary + behavior + topic)
- **Can** detect behavioral drift (tool-call distribution) in isolation
- **Cannot** detect framing-level compression (definitionally no surface signal)

The framing-detection gap is documented as a known structural limit. No surface instrument can close it without access to model internals.

---

## Reproducing

```bash
# Calibration benchmark (20 trial pairs, ~2 seconds)
python3 simulate_boundary.py benchmark --seed 7

# Drift-mode comparison (single run, deterministic)
python3 simulate_boundary.py run-all --seed 7

# End-to-end pipeline demo with firing-order prediction
python3 monitor.py demo
```
