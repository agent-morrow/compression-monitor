#!/usr/bin/env python3
"""
semantic_drift.py — Detect conceptual center-of-gravity shift between agent sessions.

Usage:
    python semantic_drift.py --session-a session_A.jsonl --session-b session_B.jsonl [--sample 20]

Requires: sentence-transformers (pip install sentence-transformers)

Each JSONL file should contain one JSON object per line with a "text" field.
The script embeds a random sample from each session, computes the centroid,
and measures the cosine distance between centroids.

Output: drift score (0.0 = identical, 1.0 = maximally different), interpretation.
"""

import argparse
import json
import random
import sys


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


def main():
    parser = argparse.ArgumentParser(description="Semantic drift detector")
    parser.add_argument("--session-a", required=True, help="JSONL file for session A")
    parser.add_argument("--session-b", required=True, help="JSONL file for session B")
    parser.add_argument("--sample", type=int, default=20, help="Max texts to sample per session (default: 20)")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-transformers model name")
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    texts_a = load_texts(args.session_a)
    texts_b = load_texts(args.session_b)

    if not texts_a or not texts_b:
        print("ERROR: One or both session files are empty or missing 'text' fields.", file=sys.stderr)
        sys.exit(1)

    # Sample for efficiency
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
    print(f"interpretation: {'HIGH — significant conceptual shift' if drift > 0.15 else 'MODERATE — detectable drift' if drift > 0.05 else 'LOW — sessions conceptually stable'}")


if __name__ == "__main__":
    main()
