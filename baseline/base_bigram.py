"""Independent unconstrained token bigram baseline.

This intentionally does not import method/. It writes the same artifact shape
as the proposed method, limited to baseline fields, so eval/ can score it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data.gpt2_bpe import load_default_tokenizer
from data.wikitext import read_wikitext_lines

EOS = -1


def log(message: str) -> None:
    print(f"[base-bigram] {message}", flush=True, file=sys.stderr)


class BigramLM:
    def __init__(self, vocab: list[int], alpha: float = 0.05):
        self.vocab = sorted(vocab)
        self.alpha = alpha
        self.context_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.context_totals: Counter[int] = Counter()
        self.denominator_size = len(self.vocab) + 1

    def fit(self, sequences: list[list[int]]) -> None:
        for seq in sequences:
            prev = EOS
            for tok in seq + [EOS]:
                self.context_counts[prev][tok] += 1
                self.context_totals[prev] += 1
                prev = tok

    def prob(self, prev: int, tok: int) -> float:
        return (self.context_counts[prev][tok] + self.alpha) / (
            self.context_totals[prev] + self.alpha * self.denominator_size
        )

    def log2_prob(self, seq: list[int]) -> float:
        total = 0.0
        prev = EOS
        for tok in seq + [EOS]:
            total += math.log2(self.prob(prev, tok))
            prev = tok
        return total

    def sample(self, rng: random.Random, max_tokens: int) -> tuple[list[int], bool]:
        seq: list[int] = []
        prev = EOS
        candidates = [EOS] + self.vocab
        for _ in range(max_tokens):
            weights = [self.prob(prev, tok) for tok in candidates]
            draw = rng.random() * sum(weights)
            acc = 0.0
            chosen = EOS
            for tok, weight in zip(candidates, weights):
                acc += weight
                if draw <= acc:
                    chosen = tok
                    break
            if chosen == EOS:
                return seq, True
            seq.append(chosen)
            prev = chosen
        return seq, False


def split_by_vocab(seq: list[int], vocab: set[int]) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for tok in seq:
        if tok in vocab:
            current.append(tok)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def observed_bigrams(sequences: list[list[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for seq in sequences:
        pairs.update(zip(seq, seq[1:]))
    return pairs


def is_canonical_sequence(seq: list[int], tokenizer, overrides: set[tuple[int, int]]) -> bool:
    prev = EOS
    for tok in seq + [EOS]:
        if tok == EOS or prev == EOS:
            prev = tok
            continue
        if (prev, tok) not in overrides and not tokenizer.tokenizer_canonical_bigram(prev, tok):
            return False
        prev = tok
    return True


def completed_samples(base: BigramLM, rng: random.Random, n: int, max_tokens: int) -> tuple[list[list[int]], int]:
    samples: list[list[int]] = []
    attempts = 0
    max_attempts = max(100, n * 20)
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        sample, completed = base.sample(rng, max_tokens)
        if completed:
            samples.append(sample)
    if len(samples) < n:
        raise RuntimeError(f"only drew {len(samples)}/{n} completed baseline samples in {attempts} attempts")
    return samples, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/downloads/wikitext-2-train.txt")
    parser.add_argument("--test-path", default="data/downloads/wikitext-2-test.txt")
    parser.add_argument("--output", default="artifacts/baseline/wikitext2_bigram.json")
    parser.add_argument("--max-train-lines", type=int, default=6000)
    parser.add_argument("--max-test-lines", type=int, default=600)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--sample-max-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    tokenizer = load_default_tokenizer()
    train_lines = read_wikitext_lines(args.train_path, args.max_train_lines)
    test_lines = read_wikitext_lines(args.test_path, args.max_test_lines)
    train_full = [tokenizer.encode(line) for line in train_lines]
    test_full = [tokenizer.encode(line) for line in test_lines]

    counts = Counter(tok for seq in train_full for tok in seq)
    top_vocab = {tok for tok, _ in counts.most_common(args.vocab_size)}
    train_sequences = [segment for seq in train_full for segment in split_by_vocab(seq, top_vocab)]
    test_sequences = [seq for seq in test_full if seq and all(tok in top_vocab for tok in seq)]
    if not test_sequences:
        raise RuntimeError("no held-out lines are fully representable by the train-derived reduced vocabulary")

    base = BigramLM(sorted(top_vocab), alpha=args.alpha)
    log(f"fitting bigram LM with vocab_size={len(top_vocab)}")
    base.fit(train_sequences)
    overrides = observed_bigrams(train_sequences)

    records = [
        {"id": i, "num_tokens": len(seq), "baseline_log2_prob": base.log2_prob(seq)}
        for i, seq in enumerate(test_sequences)
    ]
    rng = random.Random(args.seed)
    samples, attempts = completed_samples(base, rng, args.samples, args.sample_max_tokens)
    canonical_rate = sum(is_canonical_sequence(seq, tokenizer, overrides) for seq in samples) / len(samples)

    payload = {
        "experiment": "wikitext2_bigram_conditioning",
        "metadata": {
            "baseline_only": True,
            "train_lines": len(train_lines),
            "test_lines": len(test_lines),
            "train_segments": len(train_sequences),
            "scored_test_lines": len(test_sequences),
            "vocab_size": len(top_vocab),
            "alpha": args.alpha,
            "samples": args.samples,
            "base_sample_attempts": attempts,
            "sample_max_tokens": args.sample_max_tokens,
            "seed": args.seed,
            "base_sample_canonical_rate": canonical_rate,
            "canonical_bigram_overrides": len(overrides),
        },
        "records": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
