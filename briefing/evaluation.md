# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The primary metric is average negative log-likelihood, reported as log-loss in bits per string. Lower is better. Constraint satisfaction is measured as the fraction of generated samples whose adjacent token bigrams are canonical under the tokenizer; higher is better, with 1.0 meaning no sampled noncanonical token bigrams.

## Baselines and Targets

The unconstrained token language model is the baseline. The paper reports, for WikiText with GPT-2 small, 369.2 bits/string for the baseline, 367.0 for local canonicalization, and 367.3 for global canonicalization. The runnable reproduction uses the same metric and tokenizer family on WikiText-2 with a scaled token bigram model.

## Evaluation Protocol

Run the method to write an artifact containing per-string log probabilities and sample-level canonicality metadata, then run `scripts/evaluate.sh` on that artifact. Evaluation aggregates the artifact into `scoring/scores.json` without importing method code. The default benchmark uses WikiText-2 train for fitting and WikiText-2 test for scoring with fixed random seed 13.
