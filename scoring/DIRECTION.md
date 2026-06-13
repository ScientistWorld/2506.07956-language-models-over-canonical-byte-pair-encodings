# Research Direction

## Research Problem

The research problem is improving token-level language models by respecting
deterministic tokenizer validity. The benchmark asks whether probability mass
assigned to invalid token encodings can be reduced in a way that improves
held-out likelihood.

## Core Contribution

Work must stay within tokenizer-valid language modeling. Improvements should
address how the model handles tokenizer-induced support constraints while
remaining comparable under the benchmark artifact and scoring contract.

## Approach Scope

In-bounds approaches may change probability estimation, support constraints,
sampling, calibration, or training, provided they directly address
tokenizer-valid support. They may change the language-model estimator as long
as the central research question remains tokenizer-valid token modeling rather
than an unrelated modeling paradigm.

## Out Of Bounds

Do not beat the benchmark by switching to a larger pretrained model without
addressing tokenizer-valid support. Do not use external training text, change
the test set, or optimize a metric that ignores tokenizer-valid generated
support.
