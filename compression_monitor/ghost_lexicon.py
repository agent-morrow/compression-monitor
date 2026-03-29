#!/usr/bin/env python3
"""
ghost_lexicon.py — Detect vocabulary decay across agent context boundaries.

Usage:
    python ghost_lexicon.py --pre outputs_before.jsonl --post outputs_after.jsonl [--top 200]

Each JSONL file should contain one JSON object per line with a "text" field
containing the agent's output text for that exchange.

Output: decay score (0.0 = no decay, 1.0 = complete vocabulary loss),
        list of terms present in pre-compression sample but absent post-compression.
"""

import argparse
import json
import re
import sys
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Extract word tokens, lowercased."""
    return re.findall(r"\b[a-z][a-z'-]{2,}\b", text.lower())


def load_texts(path: str) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                texts.append(obj["text"])
    return texts


def low_frequency_vocab(texts: list[str], top_n: int = 200) -> set[str]:
    """
    Return the low-frequency but present vocabulary:
    terms that appear at least twice but are not in the top-N most common.
    These are high-precision terms most vulnerable to compression loss.
    """
    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize(t))
    counts = Counter(all_tokens)
    top_terms = {term for term, _ in counts.most_common(top_n)}
    return {term for term, count in counts.items() if count >= 2 and term not in top_terms}


def extract_vocabulary(texts: list[str], top_n: int = 200) -> set[str]:
    """Return the vocabulary anchor used for ghost-term comparison."""
    return low_frequency_vocab(texts, top_n=top_n)


def compute_ghost_terms(pre_vocab: set[str], post_vocab: set[str]) -> list[str]:
    """Return anchor terms that disappeared after the boundary."""
    return sorted(set(pre_vocab) - set(post_vocab))


def _term_counter(text: str) -> Counter:
    """Return a token counter for a single text fragment."""
    return Counter(tokenize(text))


class GhostLexiconTracker:
    """Lightweight rolling tracker for lexicon survival across boundaries."""

    def __init__(self, anchor_window: int = 3, recent_window: int = 3, top_n: int = 20):
        self.anchor_window = max(anchor_window, 1)
        self.recent_window = max(recent_window, 1)
        self.top_n = max(top_n, 1)
        self._history: list[Counter] = []
        self._anchor_history: list[Counter] = []

    def _merge(self, counters: list[Counter]) -> Counter:
        merged: Counter = Counter()
        for counter in counters:
            merged.update(counter)
        return merged

    def update(self, text: str) -> None:
        counter = _term_counter(text)
        if counter:
            self._history.append(counter)

    def record(self, step_index: int, text: str, is_anchor: bool = False) -> None:
        del step_index  # step ordering matters, absolute value does not.
        counter = _term_counter(text)
        if not counter:
            return
        self._history.append(counter)
        if is_anchor or len(self._anchor_history) < self.anchor_window:
            self._anchor_history.append(counter)

    def current_distribution(self) -> dict[str, int]:
        recent = self._history[-self.recent_window :]
        return dict(self._merge(recent))

    def consistency_score(self) -> float:
        if not self._history:
            return 1.0

        anchor_source = self._anchor_history or self._history[: self.anchor_window]
        anchor_counts = self._merge(anchor_source)
        current_counts = self._merge(self._history[-self.recent_window :])

        anchor_terms = [term for term, _ in anchor_counts.most_common(self.top_n)]
        if not anchor_terms:
            return 1.0

        current_terms = set(current_counts)
        survivors = sum(1 for term in anchor_terms if term in current_terms)
        return round(survivors / len(anchor_terms), 4)


def main():
    parser = argparse.ArgumentParser(description="Ghost lexicon decay detector")
    parser.add_argument("--pre", required=True, help="JSONL file of pre-compression outputs")
    parser.add_argument("--post", required=True, help="JSONL file of post-compression outputs")
    parser.add_argument("--top", type=int, default=200, help="Top-N common terms to exclude (default: 200)")
    args = parser.parse_args()

    pre_texts = load_texts(args.pre)
    post_texts = load_texts(args.post)

    if not pre_texts or not post_texts:
        print("ERROR: One or both input files are empty or missing 'text' fields.", file=sys.stderr)
        sys.exit(1)

    pre_vocab = low_frequency_vocab(pre_texts, args.top)
    post_vocab = low_frequency_vocab(post_texts, args.top)

    if not pre_vocab:
        print("WARNING: No low-frequency vocabulary found in pre-compression sample.", file=sys.stderr)
        print("decay_score: 0.0")
        print("ghost_terms: []")
        return

    ghost_terms = sorted(pre_vocab - post_vocab)
    decay_score = len(ghost_terms) / len(pre_vocab)

    print(f"pre_vocab_size: {len(pre_vocab)}")
    print(f"post_vocab_size: {len(post_vocab)}")
    print(f"ghost_term_count: {len(ghost_terms)}")
    print(f"decay_score: {decay_score:.4f}")
    print(f"interpretation: {'HIGH — likely compression boundary' if decay_score > 0.3 else 'MODERATE — monitor' if decay_score > 0.1 else 'LOW — no clear signal'}")
    print(f"\nghost_terms (first 30): {ghost_terms[:30]}")


if __name__ == "__main__":
    main()
