# Research Direction

## Research Problem

Improve token-level language models by respecting deterministic tokenizer
validity. A tokenizer produces exactly one token sequence per text, so most
token sequences can never appear in tokenized data. The benchmark asks how
well a token model can fit held-out text when it must assign all of its
probability to token sequences the tokenizer can produce.

## Core Contribution

Work must keep the canonicality constraint exact: probability 1 on
tokenizer-valid sequences. Within it, improve held-out likelihood. Gains may
come from how probability removed from invalid sequences is redistributed,
how the model is normalized or trained under the constraint, or how
next-token statistics are estimated from the fixed training text.

## Approach Scope

In bounds: estimation and smoothing of the bigram token model, inference-time
or training-time handling of the validity constraint, normalization schemes,
and hyperparameter selection on the validation split. Everything must be
expressed as the documented bigram weights table over the fixed support.

## Out Of Bounds

- Training on validation, test, or external text, or importing pretrained
  models trained on other data.
- Changing the tokenizer, support, or canonical-bigram table, or relabelling
  tokens.
- Models that keep probability on tokenizer-invalid sequences. They are
  scored as failed.
- Editing evaluators or score files.
