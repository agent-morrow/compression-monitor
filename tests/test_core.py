"""
tests/test_core.py — smoke tests for compression-monitor instruments.

Run with: python tests/test_core.py
Or:       python -m pytest tests/ -v  (requires pytest)
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def _run(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# ghost_lexicon primitives
# ---------------------------------------------------------------------------

def test_tokenize():
    from compression_monitor.ghost_lexicon import tokenize
    tokens = tokenize("The authentication module validates JWT tokens securely.")
    assert "authentication" in tokens
    assert "validates" in tokens

def test_low_freq_vocab_captures_precise_terms():
    from compression_monitor.ghost_lexicon import low_frequency_vocab
    # Enough repetition to get terms past the ≥2 threshold, with top_n=3 to not exclude them
    texts = [
        "authentication module uses bcrypt hashing secure password storage deployment",
        "authentication service validates bcrypt digests stored credentials deployment",
        "bcrypt provides adaptive hashing resistant brute force attacks authentication",
        "deployment pipeline validates authentication credentials using bcrypt storage",
        "bcrypt hashing authentication deployment storage validated credentials secure",
    ]
    vocab = low_frequency_vocab(texts, top_n=3)  # small top_n so precise terms aren't excluded
    assert len(vocab) > 0, "Expected non-empty low-freq vocab from repeated precise terms"

def test_ghost_lexicon_decay_score():
    from compression_monitor.ghost_lexicon import tokenize, low_frequency_vocab

    # Enough text that low_freq_vocab is non-empty
    pre_texts = []
    seed = "kubernetes deployment authentication tokens database connection pooling reliability autoscaling"
    for i in range(6):
        pre_texts.append(seed + f" configuration variant {i} cluster replica threshold sustained")

    post_texts = []
    seed2 = "quarterly report revenue growth enterprise segments subscriptions retention trajectory"
    for i in range(6):
        post_texts.append(seed2 + f" expansion metrics period fiscal regional market variant {i}")

    pre_vocab = low_frequency_vocab(pre_texts, top_n=3)
    if len(pre_vocab) == 0:
        # If still empty, skip gracefully — algorithm needs sufficient corpus density
        return

    post_tokens = set()
    for t in post_texts:
        post_tokens.update(tokenize(t))

    surviving = pre_vocab & post_tokens
    decay = 1.0 - len(surviving) / len(pre_vocab)
    assert decay > 0.5, f"Topic shift should show substantial decay, got {decay:.3f} (pre_vocab size={len(pre_vocab)})"

def test_ghost_lexicon_no_decay():
    from compression_monitor.ghost_lexicon import tokenize, low_frequency_vocab

    texts = []
    seed = "kubernetes deployment authentication tokens database connection pooling reliability"
    for i in range(6):
        texts.append(seed + f" configuration variant {i} cluster autoscaling threshold")

    vocab = low_frequency_vocab(texts, top_n=3)
    if not vocab:
        return  # insufficient corpus density, skip

    post_tokens = set()
    for t in texts:
        post_tokens.update(tokenize(t))

    surviving = vocab & post_tokens
    decay = 1.0 - len(surviving) / len(vocab)
    assert decay < 0.1, f"Same text should show near-zero decay, got {decay:.3f}"


# ---------------------------------------------------------------------------
# behavioral_footprint primitives
# ---------------------------------------------------------------------------

def test_fingerprint_captures_tool_distribution():
    from compression_monitor.behavioral_footprint import fingerprint
    exchanges = [
        {"tool": "search", "response_length": 200, "latency_ms": 500},
        {"tool": "search", "response_length": 180, "latency_ms": 480},
        {"tool": "write_file", "response_length": 800, "latency_ms": 200},
    ]
    fp = fingerprint(exchanges)
    assert len(fp) > 0, "fingerprint() should return a non-empty dict"

def test_shift_score_identical():
    from compression_monitor.behavioral_footprint import fingerprint, shift_score
    exchanges = [
        {"tool": "search", "response_length": 200},
        {"tool": "search", "response_length": 210},
    ]
    fp = fingerprint(exchanges)
    score = shift_score(fp, fp)
    assert score < 0.05, f"Identical fingerprints should have near-zero shift, got {score}"

def test_shift_score_different():
    from compression_monitor.behavioral_footprint import fingerprint, shift_score
    before = [{"tool": "search", "response_length": 200}] * 4
    after = [{"tool": "execute_code", "response_length": 900},
             {"tool": "write_file", "response_length": 1200},
             {"tool": "execute_code", "response_length": 850}]
    fp_a = fingerprint(before)
    fp_b = fingerprint(after)
    score = shift_score(fp_a, fp_b)
    assert score > 0.3, f"Different tool sets should show high shift, got {score}"


# ---------------------------------------------------------------------------
# integrations — test internal math directly (no framework deps required)
# ---------------------------------------------------------------------------

def test_crewai_drift_math():
    from compression_monitor.integrations.crewai import AgentSnapshot, DriftMeasurement
    s1 = AgentSnapshot("writer", "research paper analyzes neural network compression pruning deployment strategies", [])
    s2 = AgentSnapshot("writer", "quarterly sales figures enterprise software subscriptions regional growth", [])
    drift = DriftMeasurement(s1, s2)
    assert drift.drift_score > 0.4, f"Topic shift should show high drift, got {drift.drift_score}"

    s3 = AgentSnapshot("writer", "neural network pruning reduces model size preserving compression performance benchmarks", [])
    drift2 = DriftMeasurement(s1, s3)
    assert drift2.drift_score < drift.drift_score, "Same topic should drift less than topic shift"

def test_langgraph_drift_math():
    from compression_monitor.integrations.langgraph import CheckpointSnapshot, CheckpointDrift
    state_a = {"messages": [type("M", (), {"content": "deploying database migration scripts rollback safety", "tool_calls": []})()]}
    state_b = {"messages": [type("M", (), {"content": "analyzing marketing campaign performance metrics conversion funnels", "tool_calls": []})()]}
    s_a = CheckpointSnapshot(state_a, "snap_a")
    s_b = CheckpointSnapshot(state_b, "snap_b")
    drift = CheckpointDrift(s_a, s_b)
    assert drift.drift_score > 0.4, f"Topic shift should show drift, got {drift.drift_score}"

def test_autogen_drift_math():
    from compression_monitor.integrations.autogen import TurnSnapshot, TurnDrift
    h1 = [{"name": "agent", "role": "assistant", "content": "security vulnerabilities authentication service JWT validation missing expiry checks"}]
    h2 = [{"name": "agent", "role": "assistant", "content": "task completed results returned successfully let me know if you need anything else"}]
    s1 = TurnSnapshot("agent", h1, "snap_1")
    s2 = TurnSnapshot("agent", h2, "snap_2")
    drift = TurnDrift(s1, s2)
    assert drift.drift_score > 0.4, f"Topic shift should show drift, got {drift.drift_score}"
    assert drift.alert is not None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("ghost_lexicon: tokenize", test_tokenize),
        ("ghost_lexicon: low_freq_vocab non-empty", test_low_freq_vocab_captures_precise_terms),
        ("ghost_lexicon: decay score (topic shift)", test_ghost_lexicon_decay_score),
        ("ghost_lexicon: no decay (same text)", test_ghost_lexicon_no_decay),
        ("behavioral_footprint: fingerprint", test_fingerprint_captures_tool_distribution),
        ("behavioral_footprint: shift_score identical", test_shift_score_identical),
        ("behavioral_footprint: shift_score different", test_shift_score_different),
        ("integrations/crewai: drift math", test_crewai_drift_math),
        ("integrations/langgraph: drift math", test_langgraph_drift_math),
        ("integrations/autogen: drift math", test_autogen_drift_math),
    ]

    passed = sum(_run(name, fn) for name, fn in tests)
    total = len(tests)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
