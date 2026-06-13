# Evaluation Targets

## Primary: wikitext2_gpt2_small_conditioning

Tests whether tokenizer-valid conditioning improves held-out likelihood for a
cached GPT-2 small model on WikiText-2 lines, matching the paper's closest
reported model family in this reproduction.

- **Metric**: `relative_log_loss_to_baseline` (lower is better; below 1.0 means the method improves likelihood over the unconstrained baseline in the same run)
- **Run**: `bash scripts/run.sh`

## Scaled Coverage: wikitext2_bigram_conditioning and ptb_bigram_conditioning

Tests the same canonical conditioning equations on a smaller token bigram LM
across WikiText-2 and Penn Treebank so future agents have fast multi-corpus
coverage while improving the method.

- **Metric**: `relative_log_loss_to_baseline`
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
