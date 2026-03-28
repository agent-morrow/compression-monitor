"""
compression-monitor: session-boundary behavioral drift detection for LLM agents.

Three instruments:
  - ghost_lexicon: vocabulary decay (terms used before that vanish after)
  - behavioral_footprint: tool-call pattern shift
  - semantic_drift: topic keyword overlap change

Quick start:
    from compression_monitor.simulate_boundary import evaluate, generate_synthetic, SAMPLE_PRE_BOUNDARY
    pre = generate_synthetic(SAMPLE_PRE_BOUNDARY, 20)
    result = evaluate(pre, pre)  # no drift
    print(result["alerts"])

Or use the CLI:
    python3 monitor.py demo
"""
from . import ghost_lexicon, behavioral_footprint, semantic_drift
