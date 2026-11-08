#!/bin/bash
# Run a simple baseline into tmp/baselines/ and score it with scripts/evaluate.sh.
#
#   unigram        a canonical unigram model (valid, weak)
#   unconstrained  the paper's baseline p_Delta (no canonicality handling); it
#                  violates the canonicality constraint and is scored at the
#                  failure floor. Its loss is also reported by the
#                  evaluator-owned base_bigram_lm row.
#
# Usage: bash scripts/baseline.sh [unigram|unconstrained]

set -e

cd /home/user

KIND="${1:-unigram}"
OUT=tmp/baselines/$KIND

rm -rf "$OUT"
python -m baseline.base_bigram --kind "$KIND" --out-root "$OUT"
bash scripts/evaluate.sh "$OUT" "$OUT/scores.json"
