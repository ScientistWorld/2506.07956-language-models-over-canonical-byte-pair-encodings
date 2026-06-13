"""Reproduce canonicality-by-conditioning for a scaled token n-gram LM.

This implements the paper's computation: a token LM p_Delta, the locally
canonicalized distribution ell that masks noncanonical BPE extensions, and the
global model g estimated from local samples via importance weights.
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
    print(f"[canonical-conditioning] {message}", flush=True, file=sys.stderr)


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

    def sample(self, rng: random.Random, max_tokens: int = 256) -> tuple[list[int], bool]:
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


class LocalCanonicalLM:
    def __init__(self, base: BigramLM, tokenizer, canonical_bigram_overrides: set[tuple[int, int]] | None = None):
        self.base = base
        self.tokenizer = tokenizer
        self.canonical_bigram_overrides = canonical_bigram_overrides or set()
        self._norm_cache: dict[int, float] = {}

    def allowed(self, prev: int, tok: int) -> bool:
        if tok == EOS:
            return True
        if prev == EOS:
            return True
        if (prev, tok) in self.canonical_bigram_overrides:
            return True
        return self.tokenizer.tokenizer_canonical_bigram(prev, tok)

    def norm(self, prev: int) -> float:
        if prev not in self._norm_cache:
            total = self.base.prob(prev, EOS)
            total += sum(self.base.prob(prev, tok) for tok in self.base.vocab if self.allowed(prev, tok))
            self._norm_cache[prev] = total
        return self._norm_cache[prev]

    def log2_probs_and_weight(self, seq: list[int]) -> tuple[float, float, float]:
        base_log2 = 0.0
        local_log2 = 0.0
        log2_weight = 0.0
        prev = EOS
        for tok in seq + [EOS]:
            p = self.base.prob(prev, tok)
            z = self.norm(prev)
            base_log2 += math.log2(p)
            log2_weight += math.log2(z)
            if not self.allowed(prev, tok):
                return base_log2, float("-inf"), float("-inf")
            local_log2 += math.log2(p / z)
            prev = tok
        return base_log2, local_log2, log2_weight

    def sample(self, rng: random.Random, max_tokens: int = 256) -> tuple[list[int], float, bool]:
        seq: list[int] = []
        log2_weight = 0.0
        prev = EOS
        for _ in range(max_tokens):
            candidates = [EOS] + [tok for tok in self.base.vocab if self.allowed(prev, tok)]
            weights = [self.base.prob(prev, tok) for tok in candidates]
            z = sum(weights)
            log2_weight += math.log2(z)
            draw = rng.random() * z
            acc = 0.0
            chosen = EOS
            for tok, weight in zip(candidates, weights):
                acc += weight
                if draw <= acc:
                    chosen = tok
                    break
            if chosen == EOS:
                return seq, log2_weight, True
            seq.append(chosen)
            prev = chosen
        return seq, log2_weight, False


def log2meanexp(log2_values: list[float]) -> float:
    m = max(log2_values)
    return m + math.log2(sum(2 ** (v - m) for v in log2_values) / len(log2_values))


def is_canonical_sequence(seq: list[int], allowed) -> bool:
    prev = EOS
    for tok in seq + [EOS]:
        if not allowed(prev, tok):
            return False
        prev = tok
    return True


def canonical_sample_rate(sequences: list[list[int]], allowed) -> float:
    if not sequences:
        return 0.0
    return sum(is_canonical_sequence(seq, allowed) for seq in sequences) / len(sequences)


def observed_bigrams(sequences: list[list[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for seq in sequences:
        pairs.update(zip(seq, seq[1:]))
    return pairs


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


def completed_base_samples(base: BigramLM, rng: random.Random, n: int, max_tokens: int) -> tuple[list[list[int]], int]:
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


def completed_local_samples(
    local: LocalCanonicalLM, rng: random.Random, n: int, max_tokens: int
) -> tuple[list[list[int]], list[float], int]:
    samples: list[list[int]] = []
    log_weights: list[float] = []
    attempts = 0
    max_attempts = max(100, n * 20)
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        if samples and len(samples) % 5 == 0:
            log(f"drew {len(samples)}/{n} completed local samples; cached_norms={len(local._norm_cache)}")
        sample, logw, completed = local.sample(rng, max_tokens)
        if completed:
            samples.append(sample)
            log_weights.append(logw)
    if len(samples) < n:
        raise RuntimeError(f"only drew {len(samples)}/{n} completed local samples in {attempts} attempts")
    return samples, log_weights, attempts


def importance_resample(
    samples: list[list[int]], log2_weights: list[float], rng: random.Random, n: int
) -> list[list[int]]:
    max_logw = max(log2_weights)
    weights = [2 ** (logw - max_logw) for logw in log2_weights]
    total = sum(weights)
    resampled: list[list[int]] = []
    for _ in range(n):
        draw = rng.random() * total
        acc = 0.0
        for sample, weight in zip(samples, weights):
            acc += weight
            if draw <= acc:
                resampled.append(sample)
                break
    return resampled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/downloads/wikitext-2-train.txt")
    parser.add_argument("--test-path", default="data/downloads/wikitext-2-test.txt")
    parser.add_argument("--output", default="artifacts/canonical_conditioning/wikitext2_bigram.json")
    parser.add_argument("--experiment-name", default="wikitext2_bigram_conditioning")
    parser.add_argument("--max-train-lines", type=int, default=6000)
    parser.add_argument("--max-test-lines", type=int, default=600)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--sample-max-tokens", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    log("loading GPT-2 BPE tokenizer")
    tokenizer = load_default_tokenizer()
    log("reading WikiText-2 lines")
    train_lines = read_wikitext_lines(args.train_path, args.max_train_lines)
    test_lines = read_wikitext_lines(args.test_path, args.max_test_lines)
    log(f"tokenizing {len(train_lines)} train lines and {len(test_lines)} test lines")
    train_sequences_full = [tokenizer.encode(line) for line in train_lines]
    test_sequences_full = [tokenizer.encode(line) for line in test_lines]

    log("building scaled token vocabulary")
    counts = Counter(tok for seq in train_sequences_full for tok in seq)
    top_vocab = {tok for tok, _ in counts.most_common(args.vocab_size)}
    train_sequences = [segment for seq in train_sequences_full for segment in split_by_vocab(seq, top_vocab)]
    test_sequences = [seq for seq in test_sequences_full if seq and all(tok in top_vocab for tok in seq)]
    if not test_sequences:
        raise RuntimeError("no held-out lines are fully representable by the train-derived reduced vocabulary")
    if len(test_sequences) < len(test_sequences_full):
        log(f"kept {len(test_sequences)}/{len(test_sequences_full)} test lines representable by the reduced vocabulary")

    base = BigramLM(sorted(top_vocab), alpha=args.alpha)
    log(f"fitting bigram LM with vocab_size={len(top_vocab)}")
    base.fit(train_sequences)
    # The paper notes that GPT-2's regex pre-tokenizer creates rare false
    # negatives for a pure BPE bigram test and uses corpus-derived overrides.
    # Any adjacent pair emitted by the deterministic tokenizer on text is a
    # canonical pair for that tokenizer implementation.
    overrides = observed_bigrams(train_sequences)
    log(f"using {len(overrides)} corpus-derived canonical bigram overrides")
    local = LocalCanonicalLM(base, tokenizer, overrides)

    records = []
    for i, seq in enumerate(test_sequences):
        if i and i % 5 == 0:
            log(f"evaluated {i}/{len(test_sequences)} test strings; cached_norms={len(local._norm_cache)}")
        base_lp, local_lp, logw = local.log2_probs_and_weight(seq)
        records.append(
            {
                "id": i,
                "num_tokens": len(seq),
                "baseline_log2_prob": base_lp,
                "local_log2_prob": local_lp,
                "local_log2_weight": logw,
            }
        )

    rng = random.Random(args.seed)
    base_samples, base_sample_attempts = completed_base_samples(base, rng, args.samples, args.sample_max_tokens)
    local_samples, sample_logw, local_sample_attempts = completed_local_samples(
        local, rng, args.samples, args.sample_max_tokens
    )
    global_samples = importance_resample(local_samples, sample_logw, rng, args.samples)

    payload = {
        "experiment": args.experiment_name,
        "metadata": {
            "train_lines": len(train_lines),
            "test_lines": len(test_lines),
            "train_segments": len(train_sequences),
            "scored_test_lines": len(test_sequences),
            "vocab_size": len(top_vocab),
            "alpha": args.alpha,
            "samples": args.samples,
            "base_sample_attempts": base_sample_attempts,
            "local_sample_attempts": local_sample_attempts,
            "sample_max_tokens": args.sample_max_tokens,
            "seed": args.seed,
            "log2_z_estimate": log2meanexp(sample_logw),
            "base_sample_canonical_rate": canonical_sample_rate(base_samples, local.allowed),
            "local_sample_canonical_rate": canonical_sample_rate(local_samples, local.allowed),
            "global_sample_canonical_rate": canonical_sample_rate(global_samples, local.allowed),
            "canonical_bigram_overrides": len(overrides),
        },
        "samples": {
            "base_bigram_lm": base_samples,
            "local_canonical_bigram_lm": local_samples,
            "global_canonical_bigram_lm": global_samples,
        },
        "records": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
