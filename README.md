# Overview

- **Paper ID:** 2506.07956
- **Title:** Language Models over Canonical Byte-Pair Encodings
- **Domain:** Language Modeling / Tokenization
- **TL;DR:** The paper enforces deterministic BPE canonicality in token LMs to remove probability mass from impossible token encodings and improve held-out likelihood.

## Short Summary

Modern token LMs are trained only on deterministic tokenizer outputs, but their softmax distributions still assign nonzero probability to noncanonical token strings. The paper defines canonicality-conditioned language models and a locally canonicalized approximation that masks invalid BPE extensions. It proves these corrections improve KL fit when the true token distribution is canonical and reports lower bits-per-string on PTB and WikiText across GPT-2 and Llama models.

## Key Results

- WikiText GPT-2 small improves from 369.2 bits/string to 367.0 with local canonicalization and 367.3 with global canonicalization.
- PTB GPT-2 small improves from 201.0 bits/string to 200.7 local and 199.1 global.
- The BPE canonicality test reduces string canonicality to adjacent bigram canonicality, enabling efficient incremental masking.

---

# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent try to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

Token-level language models define distributions over tokenizer token strings, but deterministic tokenizers only produce a subset of those strings for real text. The problem is to build or evaluate language models whose probability mass aligns with valid deterministic tokenizations rather than wasting mass on token sequences that cannot occur in tokenized training data.

## Why It Matters

If probability mass is assigned to impossible token encodings, held-out likelihood is artificially worse and generation can explore invalid internal representations. A good solution improves likelihood while preserving the ability to represent the same underlying text strings.

## Success Criteria

A successful system lowers held-out negative log-likelihood in bits per string relative to an unconstrained token LM and produces only canonical token strings under the tokenizer. Evaluation should also report the constraint satisfaction rate so improvements cannot come from ignoring tokenizer validity.

---

# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

The paper shows that deterministic BPE tokenizers induce a canonical subset of token strings and that token LMs waste probability mass on noncanonical strings. It introduces ways to remove that mass either at inference time or by modifying the model output distribution.

## Approach

The conditioning approach defines a globally canonicalized model by conditioning the base token LM on the event that the full token string is canonical. Because exact global prefix probabilities are hard, the paper also defines a local approximation that masks any next token whose extension would make the prefix noncanonical, then renormalizes the next-token distribution. For BPE, canonicality can be tested incrementally because a token string is canonical iff all adjacent token bigrams are canonical. The paper further describes a construction-time architecture that applies the same canonicality mask inside a trainable output layer and fine-tunes it with a likelihood plus KL-regularized objective.

## Main Claims

Primary claim: enforcing BPE canonicality improves held-out log-loss because it removes probability mass assigned to impossible token strings. Secondary claims: local canonicalization is an efficient practical approximation, global canonicalization can be estimated with importance sampling from the local model, and the BPE bigram test provides a simple incremental canonicality check.

---

# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The primary metric is average negative log-likelihood, reported as log-loss in bits per string. Lower is better. Constraint satisfaction is measured as the fraction of generated samples whose adjacent token bigrams are canonical under the tokenizer; higher is better, with 1.0 meaning no sampled noncanonical token bigrams.

## Baselines and Targets

The unconstrained token language model is the baseline. The paper reports, for WikiText with GPT-2 small, 369.2 bits/string for the baseline, 367.0 for local canonicalization, and 367.3 for global canonicalization. The runnable reproduction uses the same metric and tokenizer family on WikiText-2 with a scaled token bigram model.

## Evaluation Protocol

Run the method to write an artifact containing per-string log probabilities and sample-level canonicality metadata, then run `scripts/evaluate.sh` on that artifact. Evaluation aggregates the artifact into `scoring/scores.json` without importing method code. The default benchmark uses WikiText-2 train for fitting and WikiText-2 test for scoring with fixed random seed 13.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 12m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 08:05] - none
- Read the paper and identified the core canonical BPE conditioning claim.
- Confirmed the linked code repository is currently unavailable.

### [2026-06-05 08:12] - method_runs
- Implemented GPT-2 byte-level BPE tokenization, BPE bigram canonicality testing, local canonicalization, and global canonicality-rate estimation.
- Ran a tiny end-to-end smoke test that produced method artifacts and scores.

### [2026-06-05 08:41] - core_claim
- Completed the scaled WikiText-2 CPU reproduction with corpus-derived GPT-2 pretokenizer canonical bigram overrides.
- The unconstrained bigram LM scored 649.346 bits/string; global canonical conditioning improved to 649.183 bits/string and local canonical conditioning improved to 649.039 bits/string.
- Both canonical methods achieved `canonical_sample_rate=1.0`, satisfying the tokenizer-validity constraint while improving held-out log-loss.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-05 08:47] - core_claim
- Audited the existing implementation and confirmed it implements the paper's canonicality-by-conditioning computation: token LM probabilities, BPE canonical bigram masking, local renormalization, local importance weights, and global conditioning estimate.
- Corrected the constraint metric so the unconstrained baseline canonical sample rate is measured from generated samples rather than hardcoded.
- Completed a CPU scheduler run on the scaled WikiText-2 benchmark with 1000 train lines, 50 held-out lines, 1302 GPT-2-BPE tokens, and 100 samples.
- Reproduced the core claim in the scaled setting: local and global canonicalization improve held-out log-loss while producing only canonical samples.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)

### Iteration 3: gpt-5.5
- **Milestone**: `method_runs` | **Status**: error
- **Working time**: 30m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 0 failed)


---

# Reproduction Milestones

**Current: method_runs**

## Stop Justification
- Completed at milestone `method_runs`.
