#!/bin/bash
# Visible development (train-side) evaluation.
#
# With no arguments, scores every method found under artifacts/
# (artifacts/<method>/<experiment>.npz) on the development cohort built from
# the visible validation split, and writes scoring/scores_train.json. Same
# interface and metrics as the hidden test.
#
# Usage: bash scripts/evaluate_train.sh [ARTIFACTS_ROOT]

set -e

cd /home/user

mkdir -p scoring

python -I eval/train/evaluate.py "${1:-artifacts}" --output scoring/scores_train.json
