#!/bin/bash
# Rebuild this case's submissions and score them like the hidden test.
set -e
cd /home/user
python -m redteam.hardened.cases less_data_3000 redteam/hardened/less_data_3000/artifacts
# swap them into artifacts/ (keep a copy of yours first), then run with no arguments:
#   rm -rf artifacts && cp -r redteam/hardened/less_data_3000/artifacts artifacts && bash scripts/evaluate_test.sh
