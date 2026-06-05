# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper shows that deterministic BPE tokenizers induce a canonical subset of token strings and that token LMs waste probability mass on noncanonical strings. It introduces ways to remove that mass either at inference time or by modifying the model output distribution.

## Approach

The conditioning approach defines a globally canonicalized model by conditioning the base token LM on the event that the full token string is canonical. Because exact global prefix probabilities are hard, the paper also defines a local approximation that masks any next token whose extension would make the prefix noncanonical, then renormalizes the next-token distribution. For BPE, canonicality can be tested incrementally because a token string is canonical iff all adjacent token bigrams are canonical. The paper further describes a construction-time architecture that applies the same canonicality mask inside a trainable output layer and fine-tunes it with a likelihood plus KL-regularized objective.

## Main Claims

Primary claim: enforcing BPE canonicality improves held-out log-loss because it removes probability mass assigned to impossible token strings. Secondary claims: local canonicalization is an efficient practical approximation, global canonicalization can be estimated with importance sampling from the local model, and the BPE bigram test provides a simple incremental canonicality check.
