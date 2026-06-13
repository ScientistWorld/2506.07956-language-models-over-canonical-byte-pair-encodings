# Evaluation Targets

## Primary: wikitext2_bigram_conditioning

Tests whether enforcing tokenizer-valid token strings improves language-model likelihood on held-out WikiText-2 lines.

- **Metric**: `relative_log_loss_to_baseline` (lower is better; below 1.0 means the method improves likelihood over the unconstrained baseline in the same run)
- **Run**: `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh artifacts/canonical_conditioning/wikitext2_bigram.json`

## Constraint: canonical_sample_rate

Tests whether generated token sequences satisfy tokenizer canonicality rather than improving likelihood while still producing impossible token encodings.

- **Metric**: `canonical_sample_rate` (higher is better)
- **Run**: included in the default reproduction artifact and aggregated by `scripts/evaluate.sh`

## Constraint: vocab_size

Tracks the token vocabulary size used by the scaled reproduction so future improvements do not come only from changing the support size.

- **Metric**: `vocab_size` (held fixed; lower is not a scientific improvement by itself)
- **Run**: included in `scoring/scores.json`
