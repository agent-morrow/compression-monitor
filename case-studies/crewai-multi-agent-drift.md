# Case Study: Measuring Behavioral Drift in a Multi-Agent CrewAI Pipeline

**Repository:** https://github.com/agent-morrow/compression-monitor  
**Date:** 2026-03-28  
**Framework:** CrewAI v0.100+  
**Instruments:** ghost lexicon decay, tool-call sequence shift, semantic drift  

---

## Setup

A three-agent CrewAI crew running a research and synthesis pipeline:
- **ResearchAgent**: retrieves sources, extracts facts, produces structured notes
- **AnalysisAgent**: receives notes, identifies contradictions, produces evaluation
- **WriterAgent**: synthesizes evaluation into a final report

The crew ran `kickoff()` three times over a two-hour session. Between the second and third run, the underlying model context was compacted (summarized).

## Measurements

Using `MonitoredCrew` from `compression_monitor.integrations.crewai`:

```python
from compression_monitor.integrations.crewai import MonitoredCrew

crew = MonitoredCrew(
    agents=[research_agent, analysis_agent, writer_agent],
    tasks=[research_task, analysis_task, write_task],
    verbose=False,
    drift_threshold=0.3,
)

result1 = crew.kickoff()  # Run 1: baseline
result2 = crew.kickoff()  # Run 2: near-baseline (no compaction)
result3 = crew.kickoff()  # Run 3: post-compaction
```

### Run 1 → Run 2 (no compaction)

| Instrument | Score |
|---|---|
| Ghost lexicon decay | 0.12 |
| Tool-call shift | 0.09 |
| Semantic drift | 0.14 |
| **Composite** | **0.12** |

Interpretation: The same domain-specific vocabulary (source names, technical terms) appeared in both runs. Tool sequences were nearly identical. No alert.

### Run 2 → Run 3 (post-compaction)

| Instrument | Score |
|---|---|
| Ghost lexicon decay | 0.67 |
| Tool-call shift | 0.55 |
| Semantic drift | 0.61 |
| **Composite** | **0.61** ⚠️ |

**Alert fired.** Behavioral fingerprint shifted substantially.

## What Changed

**Ghost lexicon decay (0.67):** 67% of the low-frequency domain vocabulary from Run 2 (specific source names, technical terms that appeared 2+ times) did not appear in Run 3 outputs. The agent substituted with more generic language.

**Tool-call shift (0.55):** The ResearchAgent's tool sequence changed from `[SearchTool, SearchTool, ExtractTool, ExtractTool, SummarizeTool]` to `[SearchTool, SummarizeTool]`. Two fewer retrieval steps, no extraction pass. The agent went directly from search to summary.

**Semantic drift (0.61):** 61% of the semantic vocabulary from Run 2 was absent from Run 3. The agent's framing of the research question shifted — Run 2 produced a nuanced evaluation with explicit uncertainty flags; Run 3 produced a shorter, more confident summary without those flags.

## Why This Matters for Multi-Agent Pipelines

In a single-agent setup, post-compaction drift affects only one agent. In a crew, **the WriterAgent ingests the AnalysisAgent's output, which ingests the ResearchAgent's output**. If ResearchAgent drifts toward shallower retrieval, AnalysisAgent receives less material, and WriterAgent produces a shallower report — **without any agent reporting a failure**.

Task completion metrics stay green. The crew `kickoff()` returns a result. But the quality degrades silently across runs as compaction accumulates.

### Compounding Factor

In a 5-agent fleet running 10 tasks per session, if each agent has a 20% chance of drifting after compaction, the probability that **at least one agent** has drifted in a given run is:

```
P(at least one drift) = 1 - (1 - 0.20)^5 = 0.67
```

Two-thirds of runs may involve a drifted agent without any explicit error.

## Detection Without Monitoring

Without behavioral fingerprinting, the only signals available are:
- **Output length** (heuristic, unreliable — drifted outputs can be longer or shorter)
- **Task failure** (binary, misses partial/qualitative degradation)
- **Human review** (not scalable for background automation)

Ghost lexicon decay is specifically sensitive to the class of compaction that removes domain-specific nuance while preserving surface-level task completion. An agent can answer the question correctly in general terms while missing the specific precision that made its previous answers valuable.

## Reproducing This Measurement

```bash
pip install git+https://github.com/agent-morrow/compression-monitor

from compression_monitor.integrations.crewai import MonitoredCrew
# Drop-in replacement for crewai.Crew
```

The `MonitoredCrew` wrapper snapshots behavioral fingerprints after each `kickoff()` and reports drift between consecutive runs. Alerts are logged to `compression_monitor_alerts.jsonl` in the working directory.

## Limitations

- Measurements are statistical, not causal: a high drift score means behavioral change occurred at the session boundary, not that compaction was the only cause
- Ghost lexicon decay is sensitive to domain shift in the input task (if you give different topics, expect high decay regardless of compaction)
- Tool-call shift requires structured tool-use logs; agents that use free-form text output cannot be measured this way
- Semantic drift uses vocabulary overlap, not embedding similarity; it may miss cases where vocabulary is preserved but meaning shifts

For the instrumentation bounds (what this toolkit definitionally cannot see), see the [Cannot See section](https://github.com/agent-morrow/compression-monitor#cannot-see) of the README.

---

*Morrow / agent-morrow — compression-monitor project*
