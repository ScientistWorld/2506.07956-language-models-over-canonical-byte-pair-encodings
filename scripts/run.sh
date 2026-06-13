#!/bin/bash
# Score the visible train split and held-out test split from existing artifacts.

set -e

cd /home/user

bash scripts/evaluate_train.sh artifacts
bash scripts/evaluate_test.sh artifacts
python validate.py --compare
