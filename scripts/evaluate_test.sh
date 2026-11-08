#!/bin/bash
# Hidden held-out test evaluation.
#
# With no arguments, scores every method found under artifacts/
# (artifacts/<method>/<experiment>.npz) on the held-out test cohort that only
# eval/test/ holds, and writes scoring/scores_test.json. The evaluator-owned
# baseline row (base_bigram_lm) is always included.
#
# Usage: bash scripts/evaluate_test.sh [ARTIFACTS_ROOT]

set -e

cd /home/user

mkdir -p scoring

python -I eval/test/evaluate.py "${1:-artifacts}" --output scoring/scores_test.json
