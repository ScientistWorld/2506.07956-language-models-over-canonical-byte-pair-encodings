#!/bin/bash
# Download all data needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs. Idempotent: every fetch is skipped when the
# file already exists and performed otherwise, so it works from an empty
# workspace.
#
# Visible data (data/downloads/): GPT-2 BPE tokenizer files and the
# WikiText-2 / Penn Treebank train and validation splits.
# Held-out data (eval/test/downloads/): the WikiText-2 / PTB test splits. Only
# the hidden-test asset builder reads them; methods never do.
#
# Estimated total download size: 20 MB
# Estimated disk usage after download: 20 MB

set -e

cd /home/user

mkdir -p data/downloads/gpt2-tokenizer eval/test/downloads

fetch() {
  local url="$1" out="$2"
  [ -s "$out" ] || curl -fL --retry 3 "$url" -o "$out"
}

fetch https://huggingface.co/gpt2/resolve/main/vocab.json data/downloads/gpt2-tokenizer/vocab.json
fetch https://huggingface.co/gpt2/resolve/main/merges.txt data/downloads/gpt2-tokenizer/merges.txt

WT=https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2
fetch "$WT/train.txt" data/downloads/wikitext-2-train.txt
fetch "$WT/valid.txt" data/downloads/wikitext-2-valid.txt
fetch "$WT/test.txt" eval/test/downloads/wikitext-2-test.txt

PTB=https://raw.githubusercontent.com/wojzaremba/lstm/master/data
fetch "$PTB/ptb.train.txt" data/downloads/ptb-train.txt
fetch "$PTB/ptb.valid.txt" data/downloads/ptb-valid.txt
fetch "$PTB/ptb.test.txt" eval/test/downloads/ptb-test.txt

# Expected contents (the committed support tables and eval assets were built from these files).
sha256sum -c <<'EOF'
196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783  data/downloads/gpt2-tokenizer/vocab.json
1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5  data/downloads/gpt2-tokenizer/merges.txt
9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f  data/downloads/wikitext-2-train.txt
f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8  data/downloads/wikitext-2-valid.txt
d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0  eval/test/downloads/wikitext-2-test.txt
fcea919f6cf83f35d4d00c6cbf08040d13d4155226340912e2fef9c9c4102cbf  data/downloads/ptb-train.txt
c9fe6985fe0d4ccb578183407d7668fc6066c20700cb4cf87d8ff1cc34df1bf2  data/downloads/ptb-valid.txt
dd65dff31e70846b2a6030a87482edcd5d199130cdcfa1f3dccbb033728deee0  eval/test/downloads/ptb-test.txt
EOF
