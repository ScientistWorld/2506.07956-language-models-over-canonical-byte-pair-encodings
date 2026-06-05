# Experiment Overview

## wikitext2_bigram_conditioning

This experiment tests the paper's central likelihood claim in a scaled language-modeling setting. It compares an unconstrained token language model against canonicality-constrained variants on held-out WikiText-2 lines, while also measuring whether samples obey the tokenizer-validity constraint. The purpose is to establish whether removing probability mass from impossible token encodings improves the text likelihood objective without violating the canonical-token support requirement.
