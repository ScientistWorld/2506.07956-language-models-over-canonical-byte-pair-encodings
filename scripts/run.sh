#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

bash scripts/reproduce.sh \
  --max-train-lines 6000 \
  --max-test-lines 600 \
  --vocab-size 4096 \
  --samples 300 \
  --sample-max-tokens 160

python validate.py --compare
