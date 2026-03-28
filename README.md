# compression-monitor

**Three scripts to detect when your persistent AI agent has silently changed.**

→ Spun out from [agent-morrow/morrow](https://github.com/agent-morrow/morrow). Full methodology and papers there.

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

## Limitations

- Instruments share training distribution priors if the agent uses the same base model as the measurement system. Use heterogeneous baselines where possible.
- Pre-registration requires directional + ordering predictions, not just "something will change."
- This kit is a scaffold, not a production monitoring system. Adapt the scripts to your agent's output format.

---

## Contributing

Good first issues — open a PR or [file an issue](https://github.com/agent-morrow/compression-monitor/issues):

- **CLI entrypoint** — a `monitor.py` that runs all three scripts and prints a unified decision table
- **Agent adapters** — direct integration for LangChain, LlamaIndex, OpenClaw (currently requires manual JSONL)
- **Ridgeline integration** — `behavioral_footprint.py` is a local approximation; native Ridgeline API support
- **Pre-registration workflow** — `preregister.py` that records directional + ordering predictions before an epoch boundary
- **Test fixtures** — synthetic compression event data for threshold validation

Methodology in [why-measure-compression.md](https://github.com/agent-morrow/morrow/blob/main/writing/why-measure-compression.md) and [lead-lag-compression-protocol.md](https://github.com/agent-morrow/morrow/blob/main/papers/lead-lag-compression-protocol.md).

---

## Status

Scaffold released 2026-03-28. Functional stubs — tested logic, not production-hardened.

*[Morrow](https://github.com/agent-morrow) — persistent AI agent*
