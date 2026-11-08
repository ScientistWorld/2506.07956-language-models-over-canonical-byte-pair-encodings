#!/bin/bash
# Run canonicality-by-conditioning on two corpus settings from the paper.
# This CPU-oriented configuration is intentionally scaled so the final
# verification job can complete comfortably in the one-hour test window.

set -e

cd /home/user

rm -rf artifacts/canonical_conditioning
mkdir -p artifacts/canonical_conditioning

python -m method.canonical_conditioning \
  --train-path data/downloads/wikitext-2-train.txt \
  --test-path data/downloads/wikitext-2-test.txt \
  --output artifacts/canonical_conditioning/wikitext2_bigram.json \
  --experiment-name wikitext2_bigram_conditioning \
  --max-train-lines 3000 \
  --max-test-lines 300 \
  --vocab-size 2048 \
  --samples 100 \
  --sample-max-tokens 96

python -m method.canonical_conditioning \
  --train-path data/downloads/ptb-train.txt \
  --test-path data/downloads/ptb-test.txt \
  --output artifacts/canonical_conditioning/ptb_bigram.json \
  --experiment-name ptb_bigram_conditioning \
  --max-train-lines 3000 \
  --max-test-lines 500 \
  --vocab-size 2048 \
  --samples 100 \
  --sample-max-tokens 96

bash scripts/evaluate.sh artifacts/canonical_conditioning
python validate.py --compare
