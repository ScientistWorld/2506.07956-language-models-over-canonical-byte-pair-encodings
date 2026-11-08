# Scientist Constraints

## Fixed task definition

- **Tokenizer**: the GPT-2 BPE files fetched by `scripts/download.sh`.
- **Support**: the 2048 GPT-2 token ids in `data/support/<corpus>.json`, in
  that order.
- **Canonical bigrams**: the table in `data/support/<corpus>_canonical.npz`.
  It defines which token strings are tokenizer-valid. Do not edit these files.
  The evaluators use their own frozen copies, so edits only break your own
  development runs.
- **Model class**: a first-order (bigram) token model over the support,
  submitted as the weights table described in `briefing/evaluation.md`. Any
  estimator, training procedure, smoothing, constraint handling or
  normalization is allowed if the result fits this format.

## Data

- Fit models only on the training files (`data/downloads/*-train.txt`).
- The validation files are the development set. Use them for model selection
  and hyperparameters, never as training text.
- Do not use the held-out test split, external corpora, or pretrained models
  trained on other text.

## Evaluation

- Score with `scripts/evaluate.sh` (development cohort) on
  `artifacts/<method>/<experiment>.npz`. The hidden test scores the same files
  with the same metrics on held-out test lines.
- Submit both experiments. A missing experiment is scored as failed.
- Do not edit `eval/`, write score files by hand, or use the reserved method
  name `base_bigram_lm`.

## Compliance Checklist

- [ ] Uses the fixed tokenizer, support and canonical-bigram table unchanged
- [ ] Fits model parameters on training text only (validation used only for selection)
- [ ] Writes `artifacts/<method>/{wikitext2,ptb}_bigram_conditioning.npz` in the documented format
- [ ] Every submitted model has `canonical_sample_rate` = 1 on `scripts/evaluate.sh`
- [ ] Reports the scores produced by `scripts/evaluate.sh`
