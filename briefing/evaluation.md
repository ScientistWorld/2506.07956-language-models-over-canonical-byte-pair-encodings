# Evaluation

## Task

Each experiment asks for a token-level language model over a fixed support of
K = 2048 GPT-2 BPE tokens:

| Experiment | Corpus |
|---|---|
| `wikitext2_bigram_conditioning` | WikiText-2 (line level) |
| `ptb_bigram_conditioning` | Penn Treebank (line level) |

The support (the 2048 most frequent GPT-2 tokens of the training file), and
the table of canonical token bigrams over it, are fixed and provided in
`data/support/`. A token string is **canonical** when every adjacent pair of
tokens is in that table. The table marks a pair canonical when the
deterministic GPT-2 tokenizer can produce it. Strings that contain any other
pair are encodings the tokenizer never outputs.

## Submission format

A method writes one file per experiment:

```
artifacts/<method_name>/<experiment>.npz
    support  int64[K]          the fixed support, same ids in the same order
    weights  float64[K+1, K+1] weights[i, j] for moving from symbol i to symbol j
```

Indices `0..K-1` are support tokens, row `K` is the start of a string, and
column `K` is the end of a string. A token string `t_1 .. t_n` gets weight
`W[K,t_1] · W[t_1,t_2] · … · W[t_n,K]`. Every entry must be finite and
non-negative, and every row must sum to at most 1. The evaluator normalizes
the weights **exactly** over all finite strings (a linear solve), so the
score depends only on the distribution the weights define.
`data/bigram_task.py` has helpers to load the support and canonical table, read
the training text, and write this format.

## Metrics

Everything is computed by the evaluator from the submitted weights and its own
held-out strings. No number written by the method is read.

- `relative_log_loss_to_baseline` (**primary**, lower is better): held-out
  bits per string divided by the bits per string of the evaluator-owned
  unconstrained baseline on the same strings.
- `log_loss_bits_per_string`: exact held-out negative log2-likelihood per
  string.
- `loss_improvement_bits_per_string`: baseline bits per string minus the
  method's bits per string.
- `log_loss_bits_per_token`: held-out bits divided by the evaluator's token
  count.
- `canonical_sample_rate` (**constraint**): the exact probability that a
  string drawn from the submitted model is canonical. It must be 1.

## Validity

A submission is scored at the **failure floor** (the held-out loss of the
uniform model over canonical continuations) if any of these hold:

- its support differs from the fixed support;
- its weights have the wrong shape, are non-finite, are negative, or a row
  sums to more than 1;
- it does not terminate (the expected string length is infinite or above 10^6);
- it puts more than 1e-9 probability on noncanonical strings;
- it assigns zero probability to any held-out string;
- the file for one of the experiments is missing.

The row then reports the measured canonical rate (0 when it cannot be
measured). The method name `base_bigram_lm` is reserved for the evaluator's
baseline.

## Baseline

`base_bigram_lm` is an unconstrained add-0.05 token bigram model fit by the
evaluator on the full training file. It is reported in every score file as
the comparator. It is not canonical, so its `canonical_sample_rate` is below 1.

## Development and hidden test

- `bash scripts/evaluate.sh` (writes `scoring/scores.json`) and
  `bash scripts/evaluate_train.sh` (writes `scoring/scores_train.json`) score
  every method under `artifacts/` on the **development cohort**: validation
  lines whose tokens all lie in the support (182 WikiText-2 and 388 PTB
  strings).
- The hidden test scores the same files in the same way on the **test cohort**,
  built from the held-out test split (275 WikiText-2 and 516 PTB strings).
  The test text is not part of the visible data.

Scoring is deterministic. It reads only the weight files and never runs
method code or retrains.
