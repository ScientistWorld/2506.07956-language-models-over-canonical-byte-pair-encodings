# Evaluation

## Metrics

The primary metric is relative log-loss to an unconstrained token-level
baseline, computed as a candidate system's bits per string divided by the
baseline bits per string in the same run. Lower is better, and values below
1.0 indicate improved likelihood.

The evaluator also reports absolute negative log-likelihood in bits per string
and bits per token. Constraint satisfaction is measured as the fraction of
generated token samples whose adjacent token pairs are valid under the
deterministic tokenizer. Higher is better, with 1.0 meaning no sampled
tokenizer-invalid adjacent pairs.

## Baselines and Targets

The baseline is an unconstrained token-level language model evaluated under the
same tokenizer, train split, held-out split, and output artifact format as the
candidate system. Candidate systems should improve held-out likelihood while
maintaining tokenizer-valid generated support.

The highest-fidelity lightweight target uses cached GPT-2 small on WikiText-2.
The bigram experiments remain as fast multi-corpus coverage for the same
canonicality-by-conditioning claim.

## Evaluation Protocol

Run a candidate system to write one JSON artifact per experiment. Each artifact
must contain per-record log probabilities for the baseline and candidate
outputs, token counts, and generated token samples for validity checking. Then
run `scripts/evaluate.sh` on the artifact directory. Evaluation aggregates the
artifacts into `scoring/scores.json` without importing candidate-method code.

The benchmark uses fixed train/test files downloaded by `scripts/download.sh`.
Any reduced support or fitted statistics must be derived from training text
only.
