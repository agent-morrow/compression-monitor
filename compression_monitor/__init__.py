"""
compression-monitor: session-boundary behavioral drift detection for LLM agents.

Core instruments:
  - ghost_lexicon: vocabulary decay (terms used before that vanish after)
  - behavioral_footprint: tool-call pattern shift
  - semantic_drift: topic keyword overlap change

Framework integrations (optional):
    from compression_monitor.integrations.crewai import MonitoredCrew
    from compression_monitor.integrations.langgraph import GraphDriftMonitor
    from compression_monitor.integrations.autogen import AgentDriftMonitor

Quick start:
    from compression_monitor.simulate_boundary import evaluate, generate_synthetic, SAMPLE_PRE_BOUNDARY
    pre = generate_synthetic(SAMPLE_PRE_BOUNDARY, 20)
    result = evaluate(pre, pre)  # no drift
    print(result["alerts"])

Or use the CLI:
    compression-monitor demo
    compression-monitor run --log session.jsonl
"""
from . import ghost_lexicon, behavioral_footprint, semantic_drift
from . import integrations

__version__ = "0.2.0"
__all__ = ["ghost_lexicon", "behavioral_footprint", "semantic_drift", "integrations"]
