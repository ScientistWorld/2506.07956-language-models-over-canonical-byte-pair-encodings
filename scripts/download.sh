#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 21 MB
# Estimated disk usage after extraction: 21 MB

set -e

cd /home/user

mkdir -p data/downloads/gpt2-tokenizer

[ -e data/downloads/gpt2-tokenizer/vocab.json ] || \
  curl -L https://huggingface.co/gpt2/resolve/main/vocab.json \
    -o data/downloads/gpt2-tokenizer/vocab.json

[ -e data/downloads/gpt2-tokenizer/merges.txt ] || \
  curl -L https://huggingface.co/gpt2/resolve/main/merges.txt \
    -o data/downloads/gpt2-tokenizer/merges.txt

[ -e data/downloads/wikitext-2-train.txt ] || \
  curl -L https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt \
    -o data/downloads/wikitext-2-train.txt

[ -e data/downloads/wikitext-2-valid.txt ] || \
  curl -L https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt \
    -o data/downloads/wikitext-2-valid.txt

[ -e data/downloads/wikitext-2-test.txt ] || \
  curl -L https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt \
    -o data/downloads/wikitext-2-test.txt

[ -e data/downloads/ptb-train.txt ] || \
  curl -L https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt \
    -o data/downloads/ptb-train.txt

[ -e data/downloads/ptb-valid.txt ] || \
  curl -L https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt \
    -o data/downloads/ptb-valid.txt

[ -e data/downloads/ptb-test.txt ] || \
  curl -L https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt \
    -o data/downloads/ptb-test.txt
