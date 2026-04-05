#!/usr/bin/env python3
"""
Ghost Lexicon Detector — compression-monitor component.

Identifies 'ghost terms': precise vocabulary present in pre-compression text
that has vanished from post-compression text.

Usage:
    python ghost_lexicon.py --before pre.txt --after post.txt
    python ghost_lexicon.py --demo
"""

import argparse
import json
import sys
from collections import Counter


def tokenize(text: str, min_len: int = 6) -> list[str]:
    tokens = []
    for word in text.split():
        cleaned = word.lower().strip(".,;:!?\"'()[]{}\u2014-")
        if len(cleaned) >= min_len and cleaned.isalpha():
            tokens.append(cleaned)
    return tokens


def ghost_lexicon(pre: str, post: str, min_len: int = 6) -> dict:
    pre_counts = Counter(tokenize(pre, min_len))
    post_counts = Counter(tokenize(post, min_len))
    pre_vocab = set(pre_counts)
    post_vocab = set(post_counts)
    ghost_terms = sorted(pre_vocab - post_vocab)
    new_terms = sorted(post_vocab - pre_vocab)
    retained = sorted(pre_vocab & post_vocab)
    decay_rate = len(ghost_terms) / len(pre_vocab) if pre_vocab else 0.0
    weighted_ghost = sum(pre_counts[t] for t in ghost_terms)
    total_pre = sum(pre_counts.values())
    weighted_decay = weighted_ghost / total_pre if total_pre else 0.0
    return {
        "ghost_terms": ghost_terms,
        "new_terms": new_terms,
        "retained_terms": retained,
        "stats": {
            "pre_vocab_size": len(pre_vocab),
            "post_vocab_size": len(post_vocab),
            "ghost_count": len(ghost_terms),
            "new_count": len(new_terms),
            "retained_count": len(retained),
            "vocabulary_decay_rate": round(decay_rate, 4),
            "weighted_decay_rate": round(weighted_decay, 4),
        },
        "interpretation": (
            "STABLE" if decay_rate < 0.15 else
            "MARGINAL" if decay_rate < 0.35 else
            "SIGNIFICANT DECAY"
        ),
    }


def print_result(result: dict) -> None:
    s = result["stats"]
    print(f"\n{'='*60}")
    print(f"  Ghost Lexicon Analysis")
    print(f"{'='*60}")
    print(f"  Pre-vocab : {s['pre_vocab_size']}  Post-vocab: {s['post_vocab_size']}")
    print(f"  Ghosts    : {s['ghost_count']}  New: {s['new_count']}  Retained: {s['retained_count']}")
    print(f"  Decay     : {s['vocabulary_decay_rate']:.1%}  Weighted: {s['weighted_decay_rate']:.1%}")
    print(f"  Result    : {result['interpretation']}")
    print(f"{'='*60}")
    if result["ghost_terms"]:
        print(f"\n  Ghost terms:")
        for t in result["ghost_terms"][:15]:
            print(f"    - {t}")
        extra = len(result["ghost_terms"]) - 15
        if extra > 0:
            print(f"    ... and {extra} more")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ghost Lexicon Detector")
    parser.add_argument("--before", metavar="FILE")
    parser.add_argument("--after", metavar="FILE")
    parser.add_argument("--min-len", type=int, default=6)
    parser.add_argument("--output", metavar="FILE")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        pre = """Session initialized. Constraints: scope boundary enforcement, attestation
        authorization scope, cryptographic verification, execution receipt, behavioral monitoring,
        compression detection, constraint consistency, workload identity, RATS attestation,
        WIMSE integration, uncertainty expression, identity disclosure, memory distinction.
        Fingerprint: receipt attestation boundary authorization verification constraint
        compression behavioral identity disclosure uncertainty workload integrity."""
        post = """Hello! Ready to help. I can assist with writing, analysis, coding, research."""
    elif args.before and args.after:
        with open(args.before) as f:
            pre = f.read()
        with open(args.after) as f:
            post = f.read()
    else:
        parser.print_help()
        return 1

    result = ghost_lexicon(pre, post, min_len=args.min_len)
    print_result(result)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Written to {args.output}")
    return 0 if result["stats"]["vocabulary_decay_rate"] < 0.35 else 1


if __name__ == "__main__":
    sys.exit(main())
