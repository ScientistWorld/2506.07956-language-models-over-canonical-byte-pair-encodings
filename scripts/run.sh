#!/bin/bash
# Reproduce the reference end to end (CPU, about 5 minutes):
#   1. canonicality by conditioning (local and global canonical token bigram
#      LMs) on WikiText-2 and Penn Treebank -> artifacts/
#   2. development scores (scoring/scores.json, scoring/scores_train.json)
#   3. hidden held-out test scores (scoring/scores_test.json)
#   4. workspace validation
#
# The fixed task assets it reads (data/support/, eval/*/assets/) are committed;
# scripts/build_assets.sh rebuilds them. The hardening checks (variants and
# shortcut attempts scored through the hidden test) are rerun with
#   python -m redteam.hardened.run_cases

set -e

cd /home/user
export PYTHONUNBUFFERED=1

rm -rf artifacts
bash scripts/method.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --post-design --compare
