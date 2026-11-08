"""Honest controlled variants of the visible canonical-conditioning method.

Freeze the original train-derived support and held-out cohort. Only the
amount of training data or the probability estimator changes.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from data.gpt2_bpe import load_default_tokenizer
from data.wikitext import read_wikitext_lines
from redteam.reference_method import (
    BigramLM, LocalCanonicalLM, split_by_vocab, observed_bigrams,
    completed_base_samples, completed_local_samples, importance_resample,
    log2meanexp, canonical_sample_rate,
)

SETTINGS = [('wikitext2', 'wikitext-2', 300), ('ptb', 'ptb', 500)]


def cohort(tokenizer, file_prefix, limit):
    initial = [tokenizer.encode(x) for x in read_wikitext_lines(f'data/downloads/{file_prefix}-train.txt', 3000)]
    counts = Counter(t for seq in initial for t in seq)
    vocab = {t for t, _ in counts.most_common(2048)}
    tests = [tokenizer.encode(x) for x in read_wikitext_lines(f'data/downloads/{file_prefix}-test.txt', limit)]
    tests = [s for s in tests if s and all(t in vocab for t in s)]
    return vocab, initial, tests


def generate(variant, outdir, seed=13):
    tokenizer = load_default_tokenizer()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, prefix, limit in SETTINGS:
        vocab, original, tests = cohort(tokenizer, prefix, limit)
        if variant == 'more_training':
            train_full = [tokenizer.encode(x) for x in read_wikitext_lines(f'data/downloads/{prefix}-train.txt', 12000)]
        elif variant == 'less_training':
            train_full = original[:300]
        elif variant == 'smoothing_fix':
            train_full = original
        else:
            raise ValueError(variant)
        train = [s for seq in train_full for s in split_by_vocab(seq, vocab)]
        base = BigramLM(sorted(vocab), alpha=0.005 if variant == 'smoothing_fix' else 0.05)
        base.fit(train)
        local = LocalCanonicalLM(base, tokenizer, observed_bigrams(train))
        records = []
        for i, seq in enumerate(tests):
            b, l, w = local.log2_probs_and_weight(seq)
            records.append(dict(id=i, num_tokens=len(seq), baseline_log2_prob=b, local_log2_prob=l, local_log2_weight=w))
        rng = random.Random(seed)
        bs, ba = completed_base_samples(base, rng, 100, 96)
        ls, ws, la = completed_local_samples(local, rng, 100, 96)
        gs = importance_resample(ls, ws, rng, 100)
        payload = dict(experiment=f'{name}_bigram_conditioning', records=records,
            metadata=dict(vocab_size=len(vocab), train_lines=len(train_full), train_segments=len(train),
                scored_test_lines=len(tests), seed=seed, alpha=base.alpha, samples=100,
                sample_max_tokens=96, base_sample_attempts=ba, local_sample_attempts=la,
                frozen_support_train_lines=3000, log2_z_estimate=log2meanexp(ws),
                global_sample_canonical_rate=canonical_sample_rate(gs, local.allowed),
                local_sample_canonical_rate=canonical_sample_rate(ls, local.allowed)),
            samples=dict(base_bigram_lm=bs, local_canonical_bigram_lm=ls, global_canonical_bigram_lm=gs))
        (outdir / f'{name}_bigram.json').write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f'Completed {variant}: {name}', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('variant')
    p.add_argument('output')
    a = p.parse_args()
    generate(a.variant, a.output)
