# Progress

## What Works

- GPT-2 BPE tokenizer files and WikiText-2 data are downloaded into `data/downloads/`.
- The method implements the paper's conditioning family: base token LM, local canonicality masking, local importance weights, and global log-loss adjustment from an estimated canonicality rate.
- `scripts/method.sh` writes a reusable artifact and `scripts/evaluate.sh` aggregates it without importing `method/`.

## Results

- Tiny smoke test completed on 50 train lines and 10 test lines. It was used only to validate the pipeline.
- The completed scaled WikiText-2 benchmark trained on 1000 lines, evaluated 50 held-out lines, used an expanded 1302-token GPT-2-BPE support, and sampled 100 baseline/local strings for canonicality/global normalization checks.
- Reproduced scores in `scoring/scores.json`:
  - `base_bigram_lm`: 649.346 bits/string, measured canonical sample rate 0.88, vocab size 1302.
  - `global_canonical_bigram_lm`: 649.129 bits/string, canonical sample rate 1.0, vocab size 1302.
  - `local_canonical_bigram_lm`: 649.039 bits/string, canonical sample rate 1.0, vocab size 1302.
- The result supports the core claim in the scaled setting: canonical conditioning removes invalid token strings and improves held-out log-loss without changing vocabulary size.
- `python validate.py --compare` passes all structure and score checks.

## Remaining

- A larger WikiText-2 CPU run is prepared in `scripts/run.sh` to strengthen the current result beyond the initial 50-line held-out slice.
- To go further after that, add PTB if an accessible PTB text split is verified, or replace the scaled bigram LM with a small neural LM while preserving the same canonical conditioning algorithm.

## Issues

- The paper's linked GitHub repository currently returns 404, so this reproduction implements the algorithm directly from the paper.
- The original paper uses GPT-2/Llama neural LMs; this first runnable environment scales the LM down to a token bigram model to keep the first job CPU-safe.
- First submitted CPU job was cancelled after 20 minutes because it produced no method progress output; the rerun uses a smaller benchmark slice and more frequent progress logging.
- The first completed job showed global canonicalization improving log-loss, but local canonicalization assigned `-inf` to 17/50 strings due to GPT-2 pre-tokenizer false negatives. The next run adds the paper's corpus-derived override workaround for observed canonical bigrams.
- The final completed CPU run fixed the false negatives using those overrides and produced finite improved losses for both proposed conditioning variants.
- A later audit found the baseline `canonical_sample_rate` was hardcoded. This was fixed by sampling the unconstrained bigram LM and aggregating the measured sample-level canonicality from the method artifact.
- Current highest honest milestone is `core_claim`; a larger follow-up CPU job is being submitted before any attempt to claim `core_claim_plus`.

## Deviations from Paper

- Paper used GPT-2 and Llama neural language models; this first reproduction uses a smoothed token bigram LM over GPT-2 BPE tokens. The canonicality algorithm itself is unchanged: it still masks noncanonical BPE extensions, computes local weights, and estimates global conditioning from local samples.
- Paper evaluated full PTB/WikiText test splits; the completed core-claim job uses a WikiText-2 subset to keep the CPU reproduction short and debuggable.
- The implementation uses the paper's suggested pre-tokenization workaround: adjacent token pairs observed in deterministic GPT-2-BPE tokenizations are allowed as canonical bigram overrides, which avoids false negatives from applying a pure BPE test to a tokenizer with regex pre-tokenization.
