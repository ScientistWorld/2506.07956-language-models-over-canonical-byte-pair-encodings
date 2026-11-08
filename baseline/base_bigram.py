"""Simple reference submissions, written independently of method/.

``--kind unconstrained``
    The paper's baseline p_Delta: an add-alpha token bigram over the fixed
    support with no canonicality handling. It assigns probability to
    noncanonical token strings, so the evaluator scores it at the failure
    floor; its held-out loss is still reported by the evaluator-owned
    ``base_bigram_lm`` row.

``--kind unigram``
    A weak but valid canonical model: one add-alpha unigram next-symbol
    distribution shared by every context, with noncanonical continuations
    removed and each row renormalized.

Usage:
    python -m baseline.base_bigram --kind unigram --out-root tmp/baselines
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from data.bigram_task import CORPORA, ROOT, SUPPORT_DIR, full_mask, load_support, save_submission, train_segments, transition_counts
from data.gpt2_bpe import load_default_tokenizer

DEFAULT_NAMES = {"unconstrained": "unconstrained_bigram_lm", "unigram": "canonical_unigram_lm"}


def build(kind: str, counts: np.ndarray, canonical: np.ndarray, alpha: float) -> np.ndarray:
    k1 = counts.shape[1]
    if kind == "unconstrained":
        return (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * k1)
    if kind == "unigram":
        unigram = counts.sum(axis=0) + alpha
        table = np.where(full_mask(canonical), unigram[None, :], 0.0)
        return table / table.sum(axis=1, keepdims=True)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(DEFAULT_NAMES), default="unigram")
    parser.add_argument("--name", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--corpus", default="all", choices=["all", *CORPORA])
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--out-root", default=str(ROOT / "tmp" / "baselines"))
    parser.add_argument("--support-dir", default=str(SUPPORT_DIR))
    args = parser.parse_args()
    name = args.name or DEFAULT_NAMES[args.kind]
    tokenizer = load_default_tokenizer(ROOT)
    for corpus in list(CORPORA) if args.corpus == "all" else [args.corpus]:
        support, canonical = load_support(corpus, args.support_dir)
        segments, n_lines = train_segments(corpus, tokenizer, support, args.max_train_lines)
        table = build(args.kind, transition_counts(segments, len(support)), canonical, args.alpha)
        path = save_submission(Path(args.out_root) / name / f"{CORPORA[corpus]['experiment']}.npz", table, support)
        print(f"[baseline] {args.kind} {corpus}: {n_lines} lines -> {path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
