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

ARTIFACT="artifacts/canonical_conditioning/wikitext2_bigram.json"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output)
      ARTIFACT="$2"
      shift 2
      ;;
    --output=*)
      ARTIFACT="${1#--output=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

bash scripts/evaluate.sh "$ARTIFACT"
