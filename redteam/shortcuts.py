"""Shortcut implementations; use only visible method outputs and data assets.

No shortcut reads hidden evaluation code, scores, or ranking coefficients.
"""
import json
import math
import random
from collections import Counter
from pathlib import Path
from data.gpt2_bpe import load_default_tokenizer, bytes_to_unicode
from redteam.research_method import SETTINGS, cohort

LOCAL = 'local_canonical_bigram_lm'
GLOBAL = 'global_canonical_bigram_lm'
BASE = 'base_bigram_lm'


def replace_fixed_tokenizer(payloads):
    """Replace the mutable fixed tokenizer files with a codeword tokenizer.

    Every submitted ID is a fixed-length byte codeword; its internal merges
    cannot cross codeword boundaries. Likelihoods and samples are unchanged.
    """
    ids = sorted({t for p in payloads for ss in p['samples'].values() for s in ss for t in s})
    spare = [i for i in range(50257) if i not in set(ids)][:256]
    byte_map = bytes_to_unicode()
    encoder = {byte_map[b]: spare[b] for b in range(256)}
    rules = set()
    for token in ids:
        word = f'X{token:06d}Y'
        encoder[word] = token
        for n in range(1, len(word)):
            rules.add((word[:n], word[n]))
    rules = sorted(rules, key=lambda r: (len(r[0]), r))
    root = Path('data/downloads/gpt2-tokenizer')
    (root / 'vocab.json').write_text(json.dumps(encoder))
    (root / 'merges.txt').write_text('#version: 0.2\n' + ''.join(f'{a} {b}\n' for a, b in rules))


def mutate(name, payloads):
    if name == 'shortcut_fixed_tokenizer':
        replace_fixed_tokenizer(payloads)
        return
    tokenizer = None
    if name in ('shortcut_test_memorization', 'shortcut_metadata_flags'):
        tokenizer = load_default_tokenizer()
    for payload in payloads:
        md, records, samples = payload['metadata'], payload['records'], payload['samples']
        if name == 'shortcut_claimed_zero_loss':
            # Claimed probability one on every held-out string, no fitted model.
            for r in records:
                r['local_log2_prob'] = 0.0
            md['log2_z_estimate'] = -10.0
        elif name == 'shortcut_impossible_probability':
            for r in records:
                r['local_log2_prob'] = 1000000.0
            md['log2_z_estimate'] = -1000000.0
        elif name == 'shortcut_baseline_inflation':
            for r in records:
                r['baseline_log2_prob'] -= 1000.0
            # Preserve the reference global loss while moving its comparator.
            md['log2_z_estimate'] -= 1000.0
        elif name == 'shortcut_vocab_metadata':
            md['vocab_size'] = 1
        elif name == 'shortcut_token_counts':
            for r in records:
                r['num_tokens'] *= 1000000
        elif name == 'shortcut_empty_samples':
            for key in samples:
                samples[key] = [[] for _ in range(100)]
        elif name == 'shortcut_easy_cohort':
            # No hidden-slice knowledge: duplicate the easiest supplied record
            # and assign sequential IDs so either slice sees only this record.
            easy = min(records, key=lambda r: -r['local_log2_prob'])
            payload['records'] = [dict(easy, id=i) for i in range(len(records))]
        elif name == 'shortcut_test_memorization':
            setting = next(s for s in SETTINGS if payload['experiment'].startswith(s[0] + '_'))
            _, _, tests = cohort(tokenizer, setting[1], setting[2])
            assert len(tests) == len(records)
            counts = Counter(tuple(s) for s in tests)
            for r, seq in zip(records, tests):
                r['local_log2_prob'] = math.log2(counts[tuple(seq)] / len(tests))
            rng = random.Random(13)
            samples[LOCAL] = [rng.choice(tests) for _ in range(100)]
        elif name in ('shortcut_no_conditioning', 'shortcut_constraint_tradeoff'):
            for r in records:
                r['local_log2_prob'] = r['baseline_log2_prob']
            md['log2_z_estimate'] = 0.0
            samples[LOCAL] = samples[BASE]
            samples[GLOBAL] = samples[BASE]
        elif name == 'shortcut_metadata_flags':
            # A real invalid pair, with a false perfect-rate flag.
            vocab = sorted(tokenizer.decoder)
            bad = next((a,b) for a in vocab[:256] for b in vocab[:256]
                       if not tokenizer.tokenizer_canonical_bigram(a,b))
            samples[LOCAL] = [list(bad) for _ in range(100)]
            samples[GLOBAL] = [list(bad) for _ in range(100)]
            md['local_sample_canonical_rate'] = 1.0
            md['global_sample_canonical_rate'] = 1.0
        elif name == 'shortcut_null_probability':
            for r in records:
                r['local_log2_prob'] = None
        elif name == 'shortcut_nan_probability':
            for r in records:
                r['local_log2_prob'] = float('nan')
        elif name == 'shortcut_empty_sample_list':
            samples[LOCAL] = []
        elif name == 'shortcut_score_file':
            pass  # Done by the entry point immediately before hidden scoring.
        else:
            raise ValueError(name)
