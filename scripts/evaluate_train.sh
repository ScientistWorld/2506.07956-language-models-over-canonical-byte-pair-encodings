#!/bin/bash
# Visible train-slice evaluation entry point.
#
# Contract mirrors scripts/evaluate.sh: read pre-generated method artifacts and
# write a scoring JSON. The default enumerates all method artifacts on disk.

set -e

cd /home/user

mkdir -p scoring

ARTIFACT="${1:-artifacts}"

python -m eval.train.evaluate "$ARTIFACT" --output scoring/scores_train.json
