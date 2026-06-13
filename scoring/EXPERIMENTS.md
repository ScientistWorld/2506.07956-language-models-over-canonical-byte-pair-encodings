# Experiment Overview

## wikitext2_gpt2_small_conditioning

This experiment measures whether a candidate token-level text model improves
held-out likelihood on WikiText-2 while maintaining tokenizer-valid generated
support using a cached GPT-2 small backbone. It is the closest lightweight
setting here to the paper's reported GPT-2 small WikiText result.

## wikitext2_bigram_conditioning

This experiment measures whether a candidate token-level text model improves
held-out likelihood on WikiText-2 while maintaining tokenizer-valid generated
support. The primary score is log-loss relative to an unconstrained baseline
from the same run, so changes are judged by likelihood gain rather than by
absolute model scale.

## ptb_bigram_conditioning

This experiment repeats the same evaluation on Penn Treebank. It checks that
likelihood improvements and tokenizer-valid generated support are not specific
to the WikiText-2 text distribution.
