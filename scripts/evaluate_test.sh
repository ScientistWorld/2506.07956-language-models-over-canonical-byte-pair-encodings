#!/bin/bash
# Held-out test-slice evaluation entry point.
#
# Contract mirrors scripts/evaluate.sh: read pre-generated method artifacts and
# write a scoring JSON. When train scores exist, score the same method keys.

set -e

cd /home/user

mkdir -p scoring

ARTIFACT="${1:-artifacts}"

python -m eval.test.evaluate "$ARTIFACT" \
  --train-scores scoring/scores_train.json \
  --output scoring/scores_test.json
