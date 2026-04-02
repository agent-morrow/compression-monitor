#!/usr/bin/env python3
"""
semantic_drift.py — Detect conceptual center-of-gravity shift between agent sessions.

Usage:
    python semantic_drift.py --session-a session_A.jsonl --session-b session_B.jsonl [--sample 20]
    python semantic_drift.py --pre outputs_before.jsonl --post outputs_after.jsonl [--sample 20]

Requires: sentence-transformers (pip install "compression-monitor[embed]")

Each JSONL file should contain one JSON object per line with a "text" field.
The script embeds a random sample from each session, computes the centroid,
and measures the cosine distance between centroids.

Output: drift score (0.0 = identical, 1.0 = maximally different), interpretation.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter


def load_texts(path: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj and obj["text"].strip():
                texts.append(obj["text"].strip())
    return texts


def centroid(embeddings):
    import numpy as np

    mat = np.array(embeddings)
    return mat.mean(axis=0)


def cosine_distance(a, b):
    import numpy as np

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (norm_a * norm_b))


def _keyword_counter(text: str) -> Counter:
    return Counter(re.findall(r"\b[a-z][a-z'-]{2,}\b", text.lower()))


class SemanticDriftTracker:
    """Fallback online semantic tracker based on keyword overlap windows."""

    def __init__(self, anchor_window: int = 3, recent_window: int = 3, top_n: int = 20):
        self.anchor_window = max(anchor_window, 1)
        self.recent_window = max(recent_window, 1)
        self.top_n = max(top_n, 1)
        self._history: list[Counter] = []

    def _merge(self, counters: list[Counter]) -> Counter:
        merged: Counter = Counter()
        for counter in counters:
            merged.update(counter)
        return merged

    def update(self, text: str) -> None:
        counter = _keyword_counter(text)
        if counter:
            self._history.append(counter)

    def record(self, step_index: int, text: str) -> None:
        del step_index
        self.update(text)

    def consistency_score(self) -> float:
        if len(self._history) < 2:
            return 1.0

        anchor_counts = self._merge(self._history[: self.anchor_window])
        recent_counts = self._merge(self._history[-self.recent_window :])

        anchor_terms = {term for term, _ in anchor_counts.most_common(self.top_n)}
        recent_terms = {term for term, _ in recent_counts.most_common(self.top_n)}
        if not anchor_terms and not recent_terms:
            return 1.0

        union = anchor_terms | recent_terms
        if not union:
            return 1.0

        return round(len(anchor_terms & recent_terms) / len(union), 4)


def resolve_session_paths(args) -> tuple[str, str]:
    session_a = args.session_a or args.pre
    session_b = args.session_b or args.post
    if not session_a or not session_b:
        raise ValueError("Provide either --session-a/--session-b or the --pre/--post aliases.")
    return session_a, session_b


def main():
    parser = argparse.ArgumentParser(description="Semantic drift detector")
    parser.add_argument("--session-a", help="JSONL file for session A")
    parser.add_argument("--session-b", help="JSONL file for session B")
    parser.add_argument("--pre", help="Alias for --session-a")
    parser.add_argument("--post", help="Alias for --session-b")
    parser.add_argument("--sample", type=int, default=20, help="Max texts to sample per session (default: 20)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name")
    args = parser.parse_args()

    try:
        session_a, session_b = resolve_session_paths(args)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            'ERROR: sentence-transformers not installed. Run: pip install "compression-monitor[embed]"',
            file=sys.stderr,
        )
        sys.exit(1)

    texts_a = load_texts(session_a)
    texts_b = load_texts(session_b)

    if not texts_a or not texts_b:
        print("ERROR: One or both session files are empty or missing 'text' fields.", file=sys.stderr)
        sys.exit(1)

    sample_a = random.sample(texts_a, min(args.sample, len(texts_a)))
    sample_b = random.sample(texts_b, min(args.sample, len(texts_b)))

    print(f"Loading model: {args.model} ...", file=sys.stderr)
    model = SentenceTransformer(args.model)

    print(f"Embedding session A ({len(sample_a)} texts) ...", file=sys.stderr)
    emb_a = model.encode(sample_a)

    print(f"Embedding session B ({len(sample_b)} texts) ...", file=sys.stderr)
    emb_b = model.encode(sample_b)

    c_a = centroid(emb_a)
    c_b = centroid(emb_b)
    drift = cosine_distance(c_a, c_b)

    print(f"session_a_texts: {len(texts_a)} (sampled {len(sample_a)})")
    print(f"session_b_texts: {len(texts_b)} (sampled {len(sample_b)})")
    print(f"drift_score: {drift:.4f}")
    print(
        "interpretation: "
        + (
            "HIGH — significant conceptual shift"
            if drift > 0.15
            else "MODERATE — detectable drift"
            if drift > 0.05
            else "LOW — sessions conceptually stable"
        )
    )


if __name__ == "__main__":
    main()
