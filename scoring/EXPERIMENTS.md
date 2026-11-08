# Experiment Overview

Both experiments ask the same question on two text distributions: how well
can a token-level language model over a fixed GPT-2 BPE token support fit
held-out text, while assigning zero probability to token sequences the
deterministic tokenizer never produces?

## wikitext2_bigram_conditioning

WikiText-2, one string per text line. The score is the exact held-out
log-likelihood of the submitted model on the test lines whose tokens all lie
in the fixed support. It is reported relative to an unconstrained baseline
that the evaluator fits and owns, so the comparator cannot move. The
constraint is that the model's probability of producing a tokenizer-valid
sequence is exactly 1. Models that break it are scored as failed.

## ptb_bigram_conditioning

The same measurement on Penn Treebank. Its sentences are longer in tokens
than WikiText-2 lines, so likelihood differences per string are larger. It
checks that gains do not depend on one text distribution.
