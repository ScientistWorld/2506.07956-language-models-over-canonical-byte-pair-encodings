# Scientist Constraints

## Model And Tokenizer

Use the GPT-2 BPE tokenizer files provided by `scripts/download.sh`. The default benchmark uses a smoothed token bigram language model with the configured vocabulary cap in `scripts/run.sh`; changes intended as scientific improvements must still evaluate under the same artifact contract and report the same metrics.

## Data

Use only the WikiText-2 train/test files downloaded by `scripts/download.sh` for the default benchmark. Do not add external corpora or evaluate on a different test set when reporting `wikitext2_bigram_conditioning`.

## Evaluation

Evaluation must be performed by `scripts/evaluate.sh` reading a method artifact. Do not edit `eval/` to import a proposed method or regenerate the method's outputs during scoring.

## Compliance Checklist

- [ ] Uses the provided GPT-2 BPE tokenizer files
- [ ] Uses the provided WikiText-2 train/test files for this experiment
- [ ] Writes the expected method artifact and scores it through `scripts/evaluate.sh`
- [ ] Reports `log_loss_bits_per_string`, `canonical_sample_rate`, and `vocab_size`
