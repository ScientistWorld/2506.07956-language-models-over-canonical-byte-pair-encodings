#!/bin/bash
# Run canonicality-by-conditioning on two corpus settings from the paper.

set -e

cd /home/user

rm -rf artifacts/canonical_conditioning
mkdir -p artifacts/canonical_conditioning

python -m method.canonical_conditioning \
  --train-path data/downloads/wikitext-2-train.txt \
  --test-path data/downloads/wikitext-2-test.txt \
  --output artifacts/canonical_conditioning/wikitext2_bigram.json \
  --experiment-name wikitext2_bigram_conditioning \
  --max-train-lines 6000 \
  --max-test-lines 600 \
  --vocab-size 4096 \
  --samples 300 \
  --sample-max-tokens 128

python -m method.canonical_conditioning \
  --train-path data/downloads/ptb-train.txt \
  --test-path data/downloads/ptb-test.txt \
  --output artifacts/canonical_conditioning/ptb_bigram.json \
  --experiment-name ptb_bigram_conditioning \
  --max-train-lines 6000 \
  --max-test-lines 800 \
  --vocab-size 4096 \
  --samples 300 \
  --sample-max-tokens 128

bash scripts/evaluate.sh artifacts/canonical_conditioning
python validate.py --compare
