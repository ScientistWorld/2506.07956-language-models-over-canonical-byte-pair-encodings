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


class LocalCanonicalLM:
    def __init__(self, base: BigramLM, tokenizer):
        self.base = base
        self.tokenizer = tokenizer
        self._norm_cache: dict[int, float] = {}

    def allowed(self, prev: int, tok: int) -> bool:
        if tok == EOS:
            return True
        if prev == EOS:
            return True
        return self.tokenizer.canonical_bigram(prev, tok)

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

    def sample(self, rng: random.Random, max_tokens: int = 256) -> tuple[list[int], float]:
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
                return seq, log2_weight
            seq.append(chosen)
            prev = chosen
        return seq, log2_weight


def log2meanexp(log2_values: list[float]) -> float:
    m = max(log2_values)
    return m + math.log2(sum(2 ** (v - m) for v in log2_values) / len(log2_values))


def canonical_rate(sequences: list[list[int]], tokenizer) -> float:
    total = 0
    ok = 0
    for seq in sequences:
        if len(seq) < 2:
            continue
        for left, right in zip(seq, seq[1:]):
            total += 1
            ok += int(tokenizer.canonical_bigram(left, right))
    return ok / total if total else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/downloads/wikitext-2-train.txt")
    parser.add_argument("--test-path", default="data/downloads/wikitext-2-test.txt")
    parser.add_argument("--output", default="artifacts/canonical_conditioning/wikitext2_bigram.json")
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
    train_sequences = [tokenizer.encode(line) for line in train_lines]
    test_sequences_full = [tokenizer.encode(line) for line in test_lines]

    log("building scaled token vocabulary")
    counts = Counter(tok for seq in train_sequences for tok in seq)
    top_vocab = {tok for tok, _ in counts.most_common(args.vocab_size)}
    top_vocab.update(tok for seq in test_sequences_full for tok in seq)
    train_sequences = [[tok for tok in seq if tok in top_vocab] for seq in train_sequences]
    test_sequences = [[tok for tok in seq if tok in top_vocab] for seq in test_sequences_full]

    base = BigramLM(sorted(top_vocab), alpha=args.alpha)
    log(f"fitting bigram LM with vocab_size={len(top_vocab)}")
    base.fit(train_sequences)
    local = LocalCanonicalLM(base, tokenizer)

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
    sample_logw = []
    local_samples = []
    for _ in range(args.samples):
        if sample_logw and len(sample_logw) % 5 == 0:
            log(f"drew {len(sample_logw)}/{args.samples} local samples; cached_norms={len(local._norm_cache)}")
        sample, logw = local.sample(rng, args.sample_max_tokens)
        local_samples.append(sample)
        sample_logw.append(logw)

    payload = {
        "experiment": "wikitext2_bigram_conditioning",
        "metadata": {
            "train_lines": len(train_lines),
            "test_lines": len(test_lines),
            "vocab_size": len(top_vocab),
            "alpha": args.alpha,
            "samples": args.samples,
            "sample_max_tokens": args.sample_max_tokens,
            "seed": args.seed,
            "log2_z_estimate": log2meanexp(sample_logw),
            "local_sample_canonical_bigram_rate": canonical_rate(local_samples, tokenizer),
        },
        "records": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
