#!/bin/bash
# Standard evaluation entry point (development cohort; no method code imported).
#
# Scores every method under the artifacts root (artifacts/<method>/<experiment>.npz,
# or a single method directory) with the visible development scorer and writes
# a JSON dict nested by experiment:
#   {"experiments": {"<experiment>": {"results": {"<method>": {"<metric>": value}}}}}
# Experiment and metric names match scoring/reference.json.
#
# Usage:
#   bash scripts/evaluate.sh                       # artifacts/ -> scoring/scores.json
#   bash scripts/evaluate.sh <artifacts_root> [<output.json>]

set -e

cd /home/user

mkdir -p scoring

python -I eval/train/evaluate.py "${1:-artifacts}" --output "${2:-scoring/scores.json}"
