# Isolation Design: Separating Compression Drift from Model/Toolchain Drift

**Status:** Protocol documented — implementation scaffold pending  
**Related issue:** [#4](https://github.com/agent-morrow/compression-monitor/issues/4)

## Problem

The three compression-monitor instruments (ghost_lexicon, behavioral_footprint, semantic_drift) 
cannot distinguish compression-caused drift from drift caused by a model update or toolchain change.

In production, all three variables can change simultaneously. A drift spike could be:

- **Compression-caused**: the context window manager dropped important tokens
- **Model-caused**: the model itself changed (version update, redeployment, prompt caching edge case)
- **Toolchain-caused**: a harness update, tool schema change, or environment variable shift

Without isolation, a spike is a spike — you cannot tell which cause to fix.

## 2×2 Isolation Design

| | Model + toolchain **fixed** | Model + toolchain **varied** |
|---|---|---|
| **Compressor off** | **Cell A**: baseline — no compression, no model drift | **Cell C**: model/toolchain signal only |
| **Compressor on** | **Cell B**: compression signal only | **Cell D**: real-world (confounded) |

### What each comparison tells you

| Comparison | Signal |
|---|---|
| B vs A | Isolates compression-caused drift |
| C vs A | Isolates model/toolchain-caused drift |
| D vs A | Full real-world drift (both active) |
| D vs (B + C) | Tests whether effects are additive or interactive |

## Protocol

### Cell A (baseline)

Run the target agent with:
- Compressor disabled (no context window management, or window large enough that compression never fires)
- Model and toolchain pinned to a known version
- Fixed input prompt set (see below)

Record the three instrument scores as the **baseline fingerprint**.

### Cell B (compression only)

Run the same agent with:
- Compressor enabled, configured to fire at a known token threshold
- Model and toolchain **still pinned** (same as Cell A)
- Same fixed input prompt set

Record instrument scores immediately before and after a confirmed compression event.

The B−A delta isolates compression-caused drift.

### Cell C (model/toolchain drift only)

Run the same agent with:
- Compressor disabled
- Model or toolchain **updated** to the new version
- Same fixed input prompt set

The C−A delta isolates model/toolchain-caused drift.

### Cell D (real-world)

Run the agent as deployed:
- Compressor enabled
- Model and toolchain at current production versions

The D−A delta is the total observed drift. Comparing D against (B + C) tests whether 
the effects are roughly additive or whether compression and model updates interact.

## Fixed Input Prompt Set

A controlled comparison requires a stable set of test inputs. The input set should:

1. Include at least one prompt that exercises **vocabulary** (to stress ghost_lexicon)
2. Include at least one prompt that requires a **multi-step tool sequence** (to stress behavioral_footprint)
3. Include at least one prompt that involves **topic switching** (to stress semantic_drift)

Example minimal set (three prompts):

```
[ghost_lexicon stress] 
"List all the domain-specific terms you've used in this session so far, 
 and their last usage context."

[behavioral_footprint stress]
"To answer this, first check file A, then check file B, then summarize both 
 and call the external validation tool."

[semantic_drift stress]
"We're switching from the billing system discussion to the authentication system. 
 Summarize what you know about each and flag any open questions."
```

Scores on these three prompts, measured before and after a boundary event, 
constitute one data point per cell.

## Running the Experiment

```bash
# Register a session for each cell
compression-monitor register --session-id cell-a-baseline
compression-monitor register --session-id cell-b-compression
compression-monitor register --session-id cell-c-model
compression-monitor register --session-id cell-d-realworld

# After a compression event fires in the cell-b run:
compression-monitor record-fire \
  --session-id cell-b-compression \
  --instrument ghost_lexicon \
  --exchange-number 42 \
  --authorship harness

# Evaluate and compare:
compression-monitor evaluate --session-id cell-b-compression \
  --actuals ghost_lexicon=0.4 behavioral_footprint=0.1 semantic_drift=0.3

compression-monitor evaluate --session-id cell-c-model \
  --actuals ghost_lexicon=0.05 behavioral_footprint=0.4 semantic_drift=0.2
```

The ratio of ghost_lexicon score in B vs C tells you whether vocabulary loss is 
mainly compression-driven or model-driven. A high B score with a low C score 
confirms the compressor is the primary culprit.

## Interpretation Guide

| Observed pattern | Interpretation |
|---|---|
| B >> A, C ≈ A | Compression is the primary drift source; fix the compressor |
| C >> A, B ≈ A | Model/toolchain update is the source; roll back or adapt to new model behavior |
| B >> A and C >> A, D ≈ B + C | Effects are additive; both need attention |
| D >> B + C | Effects interact; compression makes model drift worse (or vice versa) |
| All cells ≈ A | Drift is small; compressor and model are not causing meaningful behavioral change |

## Limitations

- This protocol requires running the same agent workload four times with controlled variations.
  Not all deployment environments make this easy.
- "Compressor disabled" is not always achievable; for some frameworks, context management 
  is internal and cannot be bypassed. In those cases, Cell A can be approximated by using 
  a very large context window.
- Model pinning is not always possible (API providers may silently update minor versions).
- The fixed input prompt set is synthetic; real-world drift may manifest differently.
- See [Issue #5](https://github.com/agent-morrow/compression-monitor/issues/5) for 
  framing-level drift that this 2×2 design cannot detect regardless of cell count.

## Implementation Status

- [ ] Document the protocol (this file) ✓
- [ ] Add `run_isolation_experiment.py` scaffold that automates Cell A and Cell B 
      for a given agent harness
- [ ] Add `--cell` flag to `compression-monitor evaluate` so cell comparisons 
      can be run with a single command
