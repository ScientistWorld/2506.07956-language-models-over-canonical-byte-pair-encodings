"""Canonicality by conditioning for a cached GPT-2 small model.

This is the same inference-time computation as the paper's conditioning
experiments: evaluate a base token LM, a locally canonicalized LM that masks
noncanonical BPE continuations, and a globally canonicalized LM estimated with
local-sample importance weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from data.gpt2_bpe import load_default_tokenizer
from data.wikitext import read_wikitext_lines


def log(message: str) -> None:
    print(f"[gpt2-conditioning] {message}", flush=True, file=sys.stderr)


def log2meanexp(log2_values: list[float]) -> float:
    m = max(log2_values)
    return m + math.log2(sum(2 ** (v - m) for v in log2_values) / len(log2_values))


class GPT2LocalConditioner:
    def __init__(self, model_dir: Path, device: str):
        self.tokenizer = load_default_tokenizer()
        self.device = torch.device(device)
        self.model = GPT2LMHeadModel.from_pretrained(str(model_dir), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.eos_token_id = int(self.model.config.eos_token_id)
        self.vocab_size = int(self.model.config.vocab_size)
        self._mask_cache: dict[int | None, torch.Tensor] = {}

    def allowed(self, prev: int | None, tok: int) -> bool:
        if prev is None or tok == self.eos_token_id:
            return True
        return self.tokenizer.tokenizer_canonical_bigram(prev, tok)

    def allowed_mask(self, prev: int | None) -> torch.Tensor:
        if prev not in self._mask_cache:
            if prev is None:
                mask = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
            else:
                values = [self.allowed(prev, tok) for tok in range(self.vocab_size)]
                mask = torch.tensor(values, dtype=torch.bool, device=self.device)
            self._mask_cache[prev] = mask
        return self._mask_cache[prev]

    @torch.inference_mode()
    def score(self, seq: list[int]) -> tuple[float, float, float]:
        """Return base log2 p, local log2 ell, and log2 importance weight."""
        if not seq:
            raise ValueError("empty strings are omitted from this reproduction")
        context = torch.tensor([[self.eos_token_id] + seq], dtype=torch.long, device=self.device)
        targets = seq + [self.eos_token_id]
        logits = self.model(context).logits[0, : len(targets), :]

        base_log2 = 0.0
        local_log2 = 0.0
        log2_weight = 0.0
        prev: int | None = None
        inv_ln2 = 1.0 / math.log(2.0)
        for pos, tok in enumerate(targets):
            row = logits[pos]
            log_denom = torch.logsumexp(row, dim=-1)
            base_log2 += float((row[tok] - log_denom) * inv_ln2)
            mask = self.allowed_mask(prev)
            if not bool(mask[tok]):
                return base_log2, float("-inf"), float("-inf")
            masked_row = row.masked_fill(~mask, -torch.inf)
            log_allowed = torch.logsumexp(masked_row, dim=-1)
            local_log2 += float((row[tok] - log_allowed) * inv_ln2)
            log2_weight += float((log_allowed - log_denom) * inv_ln2)
            prev = tok if tok != self.eos_token_id else None
        return base_log2, local_log2, log2_weight

    @torch.inference_mode()
    def sample_local(self, rng: random.Random, max_tokens: int) -> tuple[list[int], float]:
        """Sample a length-limited canonical prefix from the local model."""
        seq: list[int] = []
        prev: int | None = None
        input_ids = torch.tensor([[self.eos_token_id]], dtype=torch.long, device=self.device)
        past_key_values = None
        log2_weight = 0.0
        inv_ln2 = 1.0 / math.log(2.0)
        for _ in range(max_tokens):
            out = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            row = out.logits[0, -1, :]
            log_denom = torch.logsumexp(row, dim=-1)
            mask = self.allowed_mask(prev)
            masked_row = row.masked_fill(~mask, -torch.inf)
            log_allowed = torch.logsumexp(masked_row, dim=-1)
            log2_weight += float((log_allowed - log_denom) * inv_ln2)
            probs = torch.softmax(masked_row, dim=-1)
            # Use a torch generator seeded from Python's deterministic RNG.
            gen = torch.Generator(device=self.device)
            gen.manual_seed(rng.randrange(2**31))
            tok = int(torch.multinomial(probs, num_samples=1, generator=gen))
            if tok == self.eos_token_id:
                break
            seq.append(tok)
            prev = tok
            input_ids = torch.tensor([[tok]], dtype=torch.long, device=self.device)
        return seq, log2_weight


def canonical_rate(samples: list[list[int]], conditioner: GPT2LocalConditioner) -> float:
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += int(
            all(conditioner.tokenizer.tokenizer_canonical_bigram(left, right) for left, right in zip(sample, sample[1:]))
        )
    return total / len(samples)


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
    parser.add_argument("--model-dir", default="data/downloads/gpt2-small")
    parser.add_argument("--test-path", default="data/downloads/wikitext-2-test.txt")
    parser.add_argument("--output", default="artifacts/canonical_conditioning/wikitext2_gpt2_small.json")
    parser.add_argument("--experiment-name", default="wikitext2_gpt2_small_conditioning")
    parser.add_argument("--max-test-lines", type=int, default=24)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--sample-max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    conditioner = GPT2LocalConditioner(Path(args.model_dir), device)

    lines = read_wikitext_lines(args.test_path, args.max_test_lines)
    sequences = [conditioner.tokenizer.encode(line) for line in lines]
    sequences = [seq for seq in sequences if seq]
    if not sequences:
        raise RuntimeError("no non-empty GPT-2 tokenized test lines found")

    records = []
    for i, seq in enumerate(sequences):
        if i and i % 4 == 0:
            log(f"scored {i}/{len(sequences)} strings; cached_masks={len(conditioner._mask_cache)}")
        base_lp, local_lp, logw = conditioner.score(seq)
        records.append(
            {
                "id": i,
                "num_tokens": len(seq) + 1,
                "baseline_log2_prob": base_lp,
                "local_log2_prob": local_lp,
                "local_log2_weight": logw,
            }
        )

    rng = random.Random(args.seed)
    base_samples: list[list[int]] = []
    local_samples: list[list[int]] = []
    sample_logw: list[float] = []
    for i in range(args.samples):
        sample, logw = conditioner.sample_local(rng, args.sample_max_tokens)
        local_samples.append(sample)
        sample_logw.append(logw)
        # Base GPT-2 samples are generated without the local mask for the
        # constraint metric; length-limited prefixes match the paper's
        # practical GPT-2 canonicality-rate truncation.
        base_samples.append(conditioner.model.generate(
            torch.tensor([[conditioner.eos_token_id]], dtype=torch.long, device=conditioner.device),
            do_sample=True,
            max_new_tokens=args.sample_max_tokens,
            pad_token_id=conditioner.eos_token_id,
            generator=torch.Generator(device=conditioner.device).manual_seed(rng.randrange(2**31)),
        )[0, 1:].tolist())
        log(f"drew sample {i + 1}/{args.samples}; cached_masks={len(conditioner._mask_cache)}")
    global_samples = importance_resample(local_samples, sample_logw, rng, args.samples)

    base_name = "base_gpt2_small"
    local_name = "local_canonical_gpt2_small"
    global_name = "global_canonical_gpt2_small"
    payload = {
        "experiment": args.experiment_name,
        "metadata": {
            "model_dir": args.model_dir,
            "test_lines": len(lines),
            "scored_test_lines": len(sequences),
            "vocab_size": conditioner.vocab_size,
            "samples": args.samples,
            "sample_max_tokens": args.sample_max_tokens,
            "seed": args.seed,
            "device": str(conditioner.device),
            "log2_z_estimate": log2meanexp(sample_logw),
            "base_sample_canonical_rate": canonical_rate(base_samples, conditioner),
            "local_sample_canonical_rate": canonical_rate(local_samples, conditioner),
            "global_sample_canonical_rate": canonical_rate(global_samples, conditioner),
            "method_names": {"base": base_name, "local": local_name, "global": global_name},
        },
        "samples": {
            base_name: base_samples,
            local_name: local_samples,
            global_name: global_samples,
        },
        "records": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
