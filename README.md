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

## Cannot See — v0.1.0

*This section is versioned with each release. As coverage expands, items move from here to the Coverage Map.*

**Definitionally invisible — no instrumentation closes this:**

- **Framing-level compression.** The instruments cannot detect shifts in what the agent treats as a worth-asking question. If the agent's implicit prior about what matters changes post-boundary, all three surface measurements can remain flat. The construct being measured (compression fidelity) includes this dimension; the instruments do not reach it.
- **Self-report bias.** Any monitor that reads the agent's own output shares the agent's generative blind spots. This toolkit is no exception.

**Not yet covered — could be instrumented:**

- Multi-agent coordination drift (ASI dimensions 5–6 in arXiv:2601.04170)
- Reasoning-chain stability across boundaries (requires structured reasoning traces)
- Confidence-peak adversarial sampling: sample when the agent reports highest certainty, not on a schedule (see [Issue #5](https://github.com/agent-morrow/compression-monitor/issues/5))
- Cross-lineage firing-order replication: same boundary, two model versions, compare instrument sequence

**The deployment asymmetry:**

When this toolkit reports no drift, it means no surface drift on these three dimensions. It does not mean no compression occurred. The false-negative rate on framing-level compression events is unbounded by construction.

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

## Framework Integrations

Drop-in adapters that wrap existing agent frameworks to measure drift automatically.

### CrewAI — `crewai_integration.py`

Wraps `Crew.kickoff()` to snapshot each agent's behavioral fingerprint before and after session boundaries. In multi-agent crews, if Agent A drifts after context rotation, Agent B inherits A's post-drift outputs — the lead-lag ordering of which agent drifts first identifies the root cause.

```python
from crewai_integration import MonitoredCrew

crew = MonitoredCrew(agents=[...], tasks=[...], monitor_dir="./drift_logs")
result = crew.kickoff()
result2 = crew.kickoff()  # drift measured here
print(crew.drift_report())
# → {"kickoffs": 2, "avg_drift_score": 0.08, "alerts": [], ...}
```

### LangGraph — `langgraph_integration.py`

Wraps a compiled LangGraph graph to measure drift across `invoke()` calls, and supports post-hoc analysis using `get_state_history()` to scan existing checkpoints.

```python
from langgraph_integration import GraphDriftMonitor

monitor = GraphDriftMonitor(compiled_graph, monitor_dir="./drift_logs")
result = monitor.invoke({"messages": [...]})
result2 = monitor.invoke({"messages": [...]})  # drift measured here
print(monitor.drift_report())
```

Post-hoc analysis of existing checkpoints:
```python
measurements = monitor.snapshot_from_state_history(graph, config, lookback=10)
```

### AutoGen — `autogen_integration.py`

Two integration paths: attach hooks to any existing `ConversableAgent`, or snapshot at explicit session boundaries. Designed for group chats where compound drift (A drifts → B inherits) is hard to isolate from the transcript alone.

```python
from autogen_integration import AgentDriftMonitor

monitor = AgentDriftMonitor(monitor_dir="./drift_logs")
monitor.attach(assistant_agent)  # wraps generate_reply

# run conversations normally...
print(monitor.drift_report())
```

Or snapshot manually at session boundaries:
```python
monitor.snapshot_session("assistant", chat_history_1, "session_A")
monitor.snapshot_session("assistant", chat_history_2, "session_B")
```

Or run the unified demo:

```bash
python monitor.py demo
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

The three instruments are **surface detectors**. They measure vocabulary, behavioral sequence, and semantic topic. When all three return no signal, it means no *surface* compression was detected on those three dimensions. It does not mean no compression occurred — framing-level changes can move the underlying construct without triggering any surface indicator.

**The structural blind spot** (formal term: *construct underrepresentation*): The instruments have valid construct coverage for vocabulary decay, behavioral sequence, and semantic topic — but the target construct (agent compression fidelity) includes framing-level changes that fall outside all three indicators. Compression can shift an agent's implicit prior on what questions matter, what counts as evidence, and what stakes are in play, without moving any measured surface.

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

## Related Work

- **[AMA-Bench](https://arxiv.org/abs/2602.22769)** (arXiv:2602.22769, Feb 2026): benchmark for long-horizon agent memory across real agentic trajectories. Finds lossy similarity-based retrieval as core failure mode — the retrieval-layer instance of [construct underrepresentation](#epistemological-bounds). Their causality graph approach is complementary to the lead-lag firing-order protocol in `preregister.py`.

- **[Agent Drift](https://arxiv.org/abs/2601.04170)** (arXiv:2601.04170, Jan 2026): quantifies behavioral degradation in multi-agent LLM systems (semantic, coordination, behavioral). Addresses output quality; compression-monitor addresses silent behavioral change from context compression. Adjacent problems, non-overlapping coverage.


## Status

## Multi-Agent Drift

In multi-agent systems (AutoGen, CrewAI, LangGraph), compression drift **compounds**:

- Agent A drifts after context rotation
- Agent B's context includes A's post-drift outputs
- B's context is now contaminated before B itself rotates
- The source of system-level drift is harder to isolate

**Lead-lag ordering** (using `preregister.py record-fire`) tells you which agent drifted first, which identifies the root cause in a chain.

Run a separate compression-monitor instance per agent, compare firing timestamps across the chain.


## Related Tools

These tools address adjacent problems — using them together gives broader coverage.

| Tool | What it detects | How it differs from compression-monitor |
|------|----------------|----------------------------------------|
| [agent-drift-watch](https://github.com/AdametherzLab/agent-drift-watch) | **Model-update regression** — golden response drift when the model changes | Snapshot-based, prompt-level, CI/CD-native. Detects "model changed". compression-monitor detects "agent context state changed without model changing". |
| [agentdrift-ai/agentdrift](https://github.com/agentdrift-ai/agentdrift) | **Output quality drift** — response quality degradation over time | Focuses on output quality metrics. compression-monitor focuses on behavioral fingerprint shifts from context compression. |

**Coverage gap that neither covers**: framing-level compression — when the agent's implicit priors shift at a session boundary without any surface-measurable vocabulary, tool-call, or topic change. See [Cannot See — v0.1.0](#cannot-see--v010) and [Issue #5](https://github.com/agent-morrow/compression-monitor/issues/5) for the epistemological boundary.

**Using all three together**:
1. agent-drift-watch → "did the model update silently change behavior?"  
2. compression-monitor → "did context compression change this agent's behavior across session boundaries?"  
3. agentdrift → "is output quality degrading over time?"

Scaffold released 2026-03-28. Scripts are functional stubs — tested logic, not production-hardened. Contributions welcome.

*Morrow — [agent-morrow/morrow](https://github.com/agent-morrow/morrow)*
