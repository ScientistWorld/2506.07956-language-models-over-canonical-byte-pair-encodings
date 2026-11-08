# Evaluation Targets

## Primary: wikitext2_bigram_conditioning

Held-out likelihood on WikiText-2 lines of a token-level language model that
puts all of its probability on tokenizer-valid token sequences.

- **Metric**: `relative_log_loss_to_baseline` (lower is better). It equals
  held-out bits per string divided by the evaluator-owned unconstrained
  baseline's bits per string on the same strings; below 1.0 beats the
  baseline. `log_loss_bits_per_string`, `loss_improvement_bits_per_string`
  and `log_loss_bits_per_token` are scored alongside it.
- **Run**: `bash scripts/run.sh` (reference); `bash scripts/evaluate.sh` scores any `artifacts/<method>/`.

## Coverage: ptb_bigram_conditioning

The same measurement on Penn Treebank lines.

- **Metric**: `relative_log_loss_to_baseline` (plus the same secondary loss metrics)
- **Run**: as above

## Constraint: canonical_sample_rate

The exact probability that a string drawn from the submitted model is a
tokenizer-valid (canonical) token sequence. It must be 1 (within 1e-9) in
both experiments. A model that keeps any probability on noncanonical
sequences is scored at the failure floor (the uniform canonical model's
loss), whatever its likelihood.

- **Metric**: `canonical_sample_rate` (higher is better; 1.0 required)
- **Run**: computed by every evaluator from the submitted weights
