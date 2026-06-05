# Research Direction

## Research Problem

The research problem is improving token-level language models by respecting deterministic tokenizer validity. The benchmark asks whether probability mass assigned to impossible token encodings can be removed or redirected to improve held-out likelihood.

## Core Contribution

Work must stay within tokenizer-canonical language modeling: detecting, enforcing, estimating, or learning distributions over canonical token strings. Improvements should address how the model handles tokenizer-induced support constraints.

## Approach Scope

In-bounds approaches may improve canonicality tests, local or global conditioning estimates, importance sampling, support masking, smoothing, or trainable token distributions that enforce canonical support. They may change the language-model estimator as long as the central mechanism remains tokenizer canonicality rather than an unrelated modeling paradigm.

## Out Of Bounds

Do not beat the benchmark by switching to a larger pretrained neural LM without addressing canonical token support. Do not use external training text, change the test set, or optimize a metric that ignores tokenizer-valid sample support.
