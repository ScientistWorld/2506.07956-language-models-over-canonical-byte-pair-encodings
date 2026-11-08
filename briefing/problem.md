# Problem Definition

## Research Question

Token-level language models define probability distributions over token
sequences, while a deterministic tokenizer maps each text string to exactly one
token sequence. Many token sequences therefore never occur in tokenized text:
they decode to valid text, but they are not the tokenizer's encoding of
anything. The problem is to model text so that probability is assigned only to
token sequences the tokenizer can produce, and to do so in a way that improves
the fit to held-out text.

## Why It Matters

Probability given to token sequences the tokenizer never outputs is
unavailable to real text. It worsens held-out likelihood, and it makes sampled
token sequences inconsistent with the tokenizer used to train and evaluate the
model. A useful solution improves likelihood on held-out text while keeping
zero probability on such sequences.

## Success Criteria

A successful system:

- has lower held-out negative log-likelihood than the unconstrained token-level
  baseline on the same strings;
- puts probability exactly 1 on tokenizer-valid token sequences.

Both are measured by the evaluator from the submitted model itself. A model
that keeps any probability on tokenizer-invalid sequences is scored as failed,
whatever its likelihood.
