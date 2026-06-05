# Progress

## What Works

- GPT-2 BPE tokenizer files and WikiText-2 data are downloaded into `data/downloads/`.
- The method implements the paper's conditioning family: base token LM, local canonicality masking, local importance weights, and global log-loss adjustment from an estimated canonicality rate.
- `scripts/method.sh` writes a reusable artifact and `scripts/evaluate.sh` aggregates it without importing `method/`.

## Results

- Tiny smoke test completed on 50 train lines and 10 test lines. It is not the submitted benchmark result.
- The next submitted CPU job runs the configured 6000-train-line, 600-test-line benchmark and will overwrite `scoring/scores.json`.

## Remaining

- Run the configured reproduction job and inspect whether local/global canonicalization improves over the unconstrained baseline.
- If the result supports the central claim, update `core_claim`; otherwise diagnose smoothing, vocabulary size, and sample count.

## Issues

- The paper's linked GitHub repository currently returns 404, so this reproduction implements the algorithm directly from the paper.
- The original paper uses GPT-2/Llama neural LMs; this first runnable environment scales the LM down to a token bigram model to keep the first job CPU-safe.

## Deviations from Paper

- Paper used GPT-2 and Llama neural language models; this first reproduction uses a smoothed token bigram LM over GPT-2 BPE tokens. The canonicality algorithm itself is unchanged: it still masks noncanonical BPE extensions, computes local weights, and estimates global conditioning from local samples.
- Paper evaluated full PTB/WikiText test splits; the submitted first job uses a WikiText-2 subset to keep the initial CPU job short and debuggable.
