#!/bin/bash
# Maintainer-only: rebuild the fixed task assets.
#
#   data/support/        fixed token support + canonical-bigram table (from training text)
#   eval/train/assets/   development cohort (validation split), evaluator-owned baseline
#   eval/test/assets/    held-out cohort (test split), evaluator-owned baseline
#
# These are committed; methods and the hidden test only read them. Rebuilding
# needs data/downloads/ and eval/test/downloads/ (scripts/download.sh) and
# takes about 10 CPU-minutes, almost all of it testing every support bigram.

set -e

cd /home/user

python -m data.build_support --corpus all "$@"
python -m eval.train.build_assets --corpus all
python -m eval.test.build_assets --corpus all
