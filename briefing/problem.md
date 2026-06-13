# Problem Definition

## Research Question

Token-level language models define probability distributions over token
sequences, while a deterministic tokenizer maps each text string to only one
valid tokenization. The problem is to model text so that probability mass is
assigned to tokenizer-valid token sequences, rather than to alternate token
segmentations that cannot be produced by the tokenizer.

## Why It Matters

Probability assigned to invalid token encodings is unavailable to real text.
This can worsen held-out likelihood and can make sampled internal token
sequences inconsistent with the tokenizer used to train and evaluate the model.
A useful solution improves likelihood on held-out text while preserving support
for the same underlying text strings.

## Success Criteria

A successful system reduces held-out negative log-likelihood relative to an
unconstrained token-level baseline and produces token sequences that are valid
under the deterministic tokenizer. Evaluation must report both the likelihood
benefit and the tokenizer-validity constraint so improvements cannot come from
ignoring valid support.
