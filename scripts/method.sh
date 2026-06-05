#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.canonical_conditioning "$@"
bash scripts/evaluate.sh artifacts/canonical_conditioning/wikitext2_bigram.json
