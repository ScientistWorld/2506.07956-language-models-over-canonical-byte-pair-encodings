# Evaluation Targets

## Primary: wikitext2_bigram_conditioning and ptb_bigram_conditioning

Tests whether tokenizer-valid token-level modeling improves held-out likelihood
on WikiText-2 and Penn Treebank lines.

- **Metric**: `relative_log_loss_to_baseline` (lower is better; below 1.0 means the method improves likelihood over the unconstrained baseline in the same run)
- **Run**: `bash scripts/run.sh`

## Constraint: canonical_sample_rate

Tests whether generated token sequences satisfy tokenizer validity rather than
improving likelihood while still producing invalid internal encodings.

- **Metric**: `canonical_sample_rate` (higher is better)
- **Run**: included in the default reproduction artifact and aggregated by `scripts/evaluate.sh`

## Constraint: vocab_size

Tracks the token vocabulary size used by the scaled reproduction so future improvements do not come only from changing the support size.

- **Metric**: `vocab_size` (held fixed; lower is not a scientific improvement by itself)
- **Run**: included in `scoring/scores.json`
