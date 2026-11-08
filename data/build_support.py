"""Build the fixed benchmark support and canonical-bigram table for each corpus.

Support V: the K most frequent GPT-2 BPE tokens of the full training file
(ties broken by the smaller token id).

Canonical bigrams over V x V: a pair (a, b) is canonical when the GPT-2 BPE
bigram round-trip test accepts it (pure byte-level BPE on the concatenated
bytes, or the full tokenizer with GPT-2 pre-tokenization on the decoded pair),
or when the pair occurs adjacently in the tokenized training text. The second
rule covers pre-tokenizer false negatives of the pairwise test. Only training
text is read.

Usage:
    python -m data.build_support --corpus all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from data.bigram_task import CORPORA, ROOT, SUPPORT_DIR, corpus_path, read_lines, sha256_file, split_segments, support_index
from data.gpt2_bpe import load_default_tokenizer


def log(message: str) -> None:
    print(f"[build-support] {message}", file=sys.stderr, flush=True)


def build(corpus: str, k: int, out_dir: Path, max_lines: int | None = None) -> dict:
    tokenizer = load_default_tokenizer(ROOT)
    train_file = corpus_path(corpus, "train")
    lines = read_lines(train_file, max_lines)
    sequences = [tokenizer.encode(line) for line in lines]
    counts = Counter(tok for seq in sequences for tok in seq)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < k:
        raise RuntimeError(f"{corpus}: only {len(ranked)} distinct tokens, cannot build support of {k}")
    support = np.asarray([tok for tok, _ in ranked[:k]], dtype=np.int64)
    index_of = support_index(support)
    segments = split_segments(sequences, index_of)

    observed = np.zeros((k, k), dtype=bool)
    for seg in segments:
        if len(seg) > 1:
            observed[seg[:-1], seg[1:]] = True

    log(f"{corpus}: testing {k * k} support bigrams")
    passes = np.zeros((k, k), dtype=bool)
    ids = [int(tok) for tok in support]
    start = time.time()
    for i, left in enumerate(ids):
        row = passes[i]
        for j, right in enumerate(ids):
            row[j] = tokenizer.canonical_bigram_uncached(left, right)
        if (i + 1) % 256 == 0:
            log(f"{corpus}: {i + 1}/{k} rows in {time.time() - start:.0f}s")
    canonical = passes | observed

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{corpus}_canonical.npz",
        canonical_packed=np.packbits(canonical, axis=1),
        support_size=np.asarray(k, dtype=np.int64),
        support=support,
    )
    total_tokens = sum(len(seq) for seq in sequences)
    in_support = sum(len(seg) for seg in segments)
    meta = {
        "corpus": corpus,
        "experiment": CORPORA[corpus]["experiment"],
        "support_size": k,
        "train_file": CORPORA[corpus]["train"],
        "train_file_sha256": sha256_file(train_file),
        "train_lines": len(lines),
        "train_tokens": total_tokens,
        "in_support_token_fraction": in_support / total_tokens,
        "tokenizer_vocab_sha256": sha256_file(ROOT / "data/downloads/gpt2-tokenizer/vocab.json"),
        "tokenizer_merges_sha256": sha256_file(ROOT / "data/downloads/gpt2-tokenizer/merges.txt"),
        "bigram_test_canonical": int(passes.sum()),
        "observed_train_bigrams": int(observed.sum()),
        "override_bigrams": int((observed & ~passes).sum()),
        "canonical_bigrams": int(canonical.sum()),
        "canonical_fraction": float(canonical.mean()),
        "canonical_table_sha256": sha256_file(out_dir / f"{corpus}_canonical.npz"),
        "rule": "pair passes the GPT-2 BPE bigram round trip (pure BPE or full tokenizer on the decoded pair) or occurs adjacently in tokenized training text",
        "support": [int(tok) for tok in support],
    }
    (out_dir / f"{corpus}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    log(
        f"{corpus}: support={k} canonical={meta['canonical_bigrams']} "
        f"({meta['canonical_fraction']:.4f}) overrides={meta['override_bigrams']} in {time.time() - start:.0f}s"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="all", choices=["all", *CORPORA])
    parser.add_argument("--support-size", type=int, default=2048)
    parser.add_argument("--out-dir", default=str(SUPPORT_DIR))
    parser.add_argument("--max-train-lines", type=int, default=None, help="smoke tests only")
    args = parser.parse_args()
    corpora = list(CORPORA) if args.corpus == "all" else [args.corpus]
    for corpus in corpora:
        build(corpus, args.support_size, Path(args.out_dir), args.max_train_lines)


if __name__ == "__main__":
    main()
