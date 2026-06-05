# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

Token-level language models define distributions over tokenizer token strings, but deterministic tokenizers only produce a subset of those strings for real text. The problem is to build or evaluate language models whose probability mass aligns with valid deterministic tokenizations rather than wasting mass on token sequences that cannot occur in tokenized training data.

## Why It Matters

If probability mass is assigned to impossible token encodings, held-out likelihood is artificially worse and generation can explore invalid internal representations. A good solution improves likelihood while preserving the ability to represent the same underlying text strings.

## Success Criteria

A successful system lowers held-out negative log-likelihood in bits per string relative to an unconstrained token LM and produces only canonical token strings under the tokenizer. Evaluation should also report the constraint satisfaction rate so improvements cannot come from ignoring tokenizer validity.
