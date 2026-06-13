# Scientist Constraints

## Model And Tokenizer

Use the GPT-2 BPE tokenizer files provided by `scripts/download.sh`. Changes
intended as scientific improvements must evaluate under the same artifact
contract and report the same metrics.

For `wikitext2_gpt2_small_conditioning`, use the cached GPT-2 small weights
downloaded by `scripts/download.sh`. Do not switch to a larger pretrained
model for that experiment unless reporting it as a separate, clearly named
experiment.

## Data

Use only the train/test files downloaded by `scripts/download.sh` for the
reported experiment. Build any reduced token support or fitted statistics from
training text only. Do not add held-out tokens to the support, add external
corpora, or evaluate on a different test set when reporting benchmark scores.

## Evaluation

Evaluation must be performed by `scripts/evaluate.sh` reading a method artifact. Do not edit `eval/` to import a proposed method or regenerate the method's outputs during scoring.

## Compliance Checklist

- [ ] Uses the provided GPT-2 BPE tokenizer files
- [ ] Uses cached GPT-2 small weights for `wikitext2_gpt2_small_conditioning`
- [ ] Uses the benchmark train/test files for each reported experiment
- [ ] Builds reduced vocabularies or fitted statistics from training text only
- [ ] Writes the expected method artifact and scores it through `scripts/evaluate.sh`
- [ ] Reports `log_loss_bits_per_string`, `canonical_sample_rate`, and `vocab_size`
