# Experiment Overview

## wikitext2_bigram_conditioning

This experiment tests the paper's central likelihood claim in a scaled language-modeling setting. It compares an unconstrained token language model against canonicality-constrained variants on held-out WikiText-2 lines that are representable under a train-derived reduced token support, while also measuring whether samples obey the tokenizer-validity constraint. The primary score is each method's log-loss relative to the unconstrained baseline from the same run, which keeps the paper reference and the scaled reproduction comparable while preserving the likelihood-improvement objective.

## ptb_bigram_conditioning

This experiment repeats the same evaluation on Penn Treebank lines, adding the second corpus axis used by the paper. It checks that the likelihood improvement and canonical-sampling constraint are not specific to the WikiText-2 text distribution.
