# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocal for other scientists trying to solve the same problem, where they should not see this paper's solution.

## Metrics

The primary metric is relative log-loss to the unconstrained baseline, computed as each method's bits/string divided by the baseline bits/string in the same run. Lower is better, and values below 1.0 indicate improved likelihood. Absolute average negative log-likelihood in bits per string is also reported as a secondary metric. Constraint satisfaction is measured as the fraction of generated samples whose adjacent token bigrams are canonical under the tokenizer; higher is better, with 1.0 meaning no sampled noncanonical token bigrams.

## Baselines and Targets

The unconstrained token language model is the baseline. The paper reports, for WikiText with GPT-2 small, 369.2 bits/string for the baseline, 367.0 for local canonicalization, and 367.3 for global canonicalization. These correspond to relative log-loss ratios of 1.0, 0.994041 for local canonicalization, and 0.994854 for global canonicalization. The runnable reproduction reports the same absolute metric and uses the relative ratio as the primary comparison because it runs on WikiText-2 with a scaled token bigram model rather than GPT-2-small.

## Evaluation Protocol

Run the method to write an artifact containing per-string log probabilities and sample-level canonicality metadata, then run `scripts/evaluate.sh` on that artifact. Evaluation aggregates the artifact into `scoring/scores.json` without importing method code. The default benchmark uses WikiText-2 train for fitting and WikiText-2 test for scoring with fixed random seed 13.
