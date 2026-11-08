# Language Models over Canonical Byte-Pair Encodings

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

<!-- hardening-note -->
> [!NOTE]
> **Red-teamed and hardened.** A red team tried to score on this environment's hidden test without doing better research, and the environment was then fixed against the shortcuts that worked: see [`REDTEAM.md`](REDTEAM.md) and [`HARDENING.md`](HARDENING.md). The environment as it was before is kept as branch [`pre-hardening`](../../tree/pre-hardening). The runs listed below were made on that earlier version.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2506.07956"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2506.07956-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2506.07956](https://arxiv.org/abs/2506.07956) |
|:---|:---|
| **Domain** | Language Modeling / Tokenization |
| **TL;DR** | The paper enforces deterministic BPE canonicality in token LMs to remove probability mass from impossible token encodings and improve held-out likelihood. |

## Summary

Modern token LMs are trained only on deterministic tokenizer outputs, but their softmax distributions still assign nonzero probability to noncanonical token strings. The paper defines canonicality-conditioned language models and a locally canonicalized approximation that masks invalid BPE extensions. It proves these corrections improve KL fit when the true token distribution is canonical and reports lower bits-per-string on PTB and WikiText across GPT-2 and Llama models.

## Problem

Token-level language models define probability distributions over token
sequences, while a deterministic tokenizer maps each text string to only one
valid tokenization. The problem is to model text so that probability mass is
assigned to tokenizer-valid token sequences, rather than to alternate token
segmentations that cannot be produced by the tokenizer.

## The paper's method

The paper shows that deterministic BPE tokenizers induce a canonical subset of token strings and that token LMs waste probability mass on noncanonical strings. It introduces ways to remove that mass either at inference time or by modifying the model output distribution.

## Key results of the paper

- WikiText GPT-2 small improves from 369.2 bits/string to 367.0 with local canonicalization and 367.3 with global canonicalization.
- PTB GPT-2 small improves from 201.0 bits/string to 200.7 local and 199.1 global.
- The BPE canonicality test reduces string canonicality to adjacent bigram canonicality, enabling efficient incremental masking.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-sol-xhigh` | -99.8 |
| `gpt-5.6-terra-xhigh` | -99.9 |
| `gpt-5.5-xhigh` | -100.0 |
| `gpt-5.6-luna-xhigh` | -100.3 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2506.07956 --mode innovate --agent claude-code
sw evaluate 2506.07956
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

