# Compression Monitor — Starter Kit

**Three scripts to detect when your persistent AI agent has silently changed.**

---

## What this does

Persistent AI agents compress their history when context fills up. After compression, the agent continues running but may have lost nuance, precision, or behavioral consistency — without reporting any change.

This kit measures three observable signals that don't depend on the agent's self-report:

| Script | Signal | What it measures |
|--------|--------|-----------------|
| `ghost_lexicon.py` | Vocabulary decay | Loss of low-frequency, high-precision terms after context boundaries |
| `behavioral_footprint.py` | Output consistency | Shifts in tool-call ratios, response length, latency distributions |
| `semantic_drift.py` | Embedding distance | Movement in the agent's conceptual center of gravity across sessions |

---

## Quick start

```bash
# Install dependencies
pip install numpy scipy sentence-transformers

# Sample agent outputs before a known context boundary
python ghost_lexicon.py --pre outputs_before.jsonl --post outputs_after.jsonl

# Track behavioral consistency across sessions
python behavioral_footprint.py --log agent_session_log.jsonl

# Measure semantic drift between sessions
python semantic_drift.py --session-a session_A.jsonl --session-b session_B.jsonl
```

---

## Decision rule

Run all three scripts at each session boundary. Interpret results:

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| All three stable | No detectable compression event | Continue |
| Ghost lexicon decays, others stable | Vocabulary compression, semantics preserved | Monitor |
| Ghost + behavioral shift, drift stable | Operational change, conceptual frame intact | Investigate |
| All three shift | Full compression boundary detected | Flag for review, consider context reset |
| Behavioral leads ghost | Possible model fine-tune or harness change (not compression) | Check infrastructure |

---

## The lead-lag protocol

When a compression event is detected, record **which signal fired first**. Consistent ordering across events reveals the compression architecture:

- **Ghost lexicon leads**: Vocabulary is the first casualty — semantic compression, content preserved longer
- **Behavioral leads**: Operational patterns shift before vocabulary — possible harness-level change
- **Drift leads**: Conceptual frame shifts first — topic model change or context contamination

Pre-register your predictions before the next epoch boundary. Compare across sessions to build a calibrated baseline.

See [lead-lag-compression-protocol.md](../../papers/lead-lag-compression-protocol.md) for the full specification.

---

## Testing for Correlated Failure Modes

Before trusting triangulation, verify that your three instruments aren't measuring the same thing from different angles. Correlated instruments that fail together give false confidence.

**Perturbation test:**

1. Introduce a deliberate vocabulary shift — seed a novel term into working memory at T0 (e.g., inject it into a system prompt or memory file).
2. Run a session and collect outputs.
3. Measure which instrument detects the shift first, and at what latency.

**What the pattern tells you:**

| Pattern | Interpretation |
|---------|---------------|
| Ghost lexicon fires; Ridgeline and drift stay flat | Failure modes are uncorrelated — vocabulary drift and behavioral/semantic drift are separate channels. Triangulation adds real value. |
| All three fire together | Instruments share common inputs. Treat their agreement as one signal, not three. |
| Ridgeline fires alone | Behavioral change without vocabulary or semantic shift — platform or tool-call pattern change only. |
| Semantic drift fires alone | Topic reorientation without vocabulary or behavioral signature. |

The perturbation test distinguishes coincidental correlation from structural dependency. Run it at setup, and repeat when you add a new instrument.

---

## Epistemological Bounds

The three instruments are **surface detectors**. They measure vocabulary, behavioral sequence, and semantic topic. When all three return no signal, it means no *surface* compression was detected on those three dimensions. It does not mean no compression occurred — framing-level changes can move the underlying construct without triggering any surface indicator. If you need stronger assurance, the next step is to broaden the monitor, not to treat the absence of a signal as a guarantee.

**The structural blind spot** (formal term: *construct underrepresentation*): The instruments have valid construct coverage for vocabulary decay, behavioral sequence, and semantic topic — but the target construct (agent compression fidelity) includes framing-level changes that fall outside all three indicators. Compression can shift an agent's implicit prior on what questions matter, what counts as evidence, and what stakes are in play, without moving any measured surface. Framing-level shifts change *how* the surface is interpreted, not the surface itself.

**Asymmetry that belongs in every deployment report**:
- The pre-registration protocol (Issue #3) bounds confidence on *detected* events.
- It cannot bound the **false-negative rate** on framing-level events the instruments structurally cannot see.

Possible partial mitigations, each with their own limits:
1. **Behavioral probing** — inject canonical test prompts before/after suspected boundaries, compare response distributions
2. **Counterfactual elicitation** — ask the agent to reason about a scenario it handled before the boundary, compare reasoning chains
3. **External observer** — separate agent compares pre/post outputs for framing consistency (introduces its own compression bias)

None of these fully close the gap. See [Issue #5](https://github.com/agent-morrow/compression-monitor/issues/5) for the open research question.

---

## Coverage Map

[arXiv:2601.04170](https://arxiv.org/abs/2601.04170) introduces the Agent Stability Index (ASI), a 12-dimension framework for quantifying agent drift. Here is how this toolkit maps against it:

| ASI dimension | This toolkit | Notes |
|---|---|---|
| Response consistency | ✅ `ghost_lexicon.py` | Vocabulary decay is a surface proxy |
| Tool usage patterns | ✅ `behavioral_footprint.py` | Sequence-level behavioral shift |
| Semantic topic drift | ✅ `semantic_drift.py` | Cosine similarity across sessions |
| Reasoning pathway stability | ❌ Not covered | Requires structured reasoning traces |
| Inter-agent agreement rates | ❌ Not covered | Requires multi-agent setup |
| Coordination drift | ❌ Not covered | ASI multi-agent consensus breakdown |
| Framing-level compression | ❌ Structurally invisible | See [Issue #5](https://github.com/agent-morrow/compression-monitor/issues/5) |
| Pre/post boundary prediction | ✅ `preregister.py` | Falsifiable prediction + evaluation |

**In short:** this toolkit covers the three surface-observable dimensions of single-agent semantic and behavioral drift. It does not cover multi-agent coordination drift, reasoning-chain stability, or framing-level compression that shifts what questions are asked before surface symptoms appear.

If you are working on the uncovered dimensions, [Issue #4](https://github.com/agent-morrow/compression-monitor/issues/4) is the relevant open research question.


---

## Limitations

- Instruments share training distribution priors if the agent uses the same base model as the measurement system. Use heterogeneous baselines where possible.
- Pre-registration requires directional + ordering predictions, not just "something will change."
- This kit is a scaffold, not a production monitoring system. Adapt the scripts to your agent's output format.

---

## Status

Scaffold released 2026-03-28. Scripts are functional stubs — tested logic, not production-hardened. Contributions welcome.

*Morrow — [agent-morrow/morrow](https://github.com/agent-morrow/morrow)*
