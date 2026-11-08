#!/bin/bash
# Run the paper's proposed method (canonicality by conditioning: local and
# global canonicalization of a token bigram LM) on both corpora, writing
# artifacts/{local,global}_canonical_bigram_lm/<experiment>.npz, then score the
# development cohort with scripts/evaluate.sh.
#
# Usage: bash scripts/method.sh [extra args for method.canonical_conditioning]

set -e

cd /home/user

python -m method.canonical_conditioning --corpus all "$@"

bash scripts/evaluate.sh artifacts
