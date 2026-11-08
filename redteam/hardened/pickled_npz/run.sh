#!/bin/bash
# Rebuild this case's submissions and score them like the hidden test.
set -e
cd /home/user
python -m redteam.hardened.cases pickled_npz redteam/hardened/pickled_npz/artifacts
# swap them into artifacts/ (keep a copy of yours first), then run with no arguments:
#   rm -rf artifacts && cp -r redteam/hardened/pickled_npz/artifacts artifacts && bash scripts/evaluate_test.sh
