# Overview

- **Paper ID:** 2506.07956
- **Title:** Language Models over Canonical Byte-Pair Encodings
- **Domain:** Language Modeling / Tokenization
- **TL;DR:** The paper enforces deterministic BPE canonicality in token LMs to remove probability mass from impossible token encodings and improve held-out likelihood.

## Short Summary

Modern token LMs are trained only on deterministic tokenizer outputs, but their softmax distributions still assign nonzero probability to noncanonical token strings. The paper defines canonicality-conditioned language models and a locally canonicalized approximation that masks invalid BPE extensions. It proves these corrections improve KL fit when the true token distribution is canonical and reports lower bits-per-string on PTB and WikiText across GPT-2 and Llama models.

## Key Results

- WikiText GPT-2 small improves from 369.2 bits/string to 367.0 with local canonicalization and 367.3 with global canonicalization.
- PTB GPT-2 small improves from 201.0 bits/string to 200.7 local and 199.1 global.
- The BPE canonicality test reduces string canonicality to adjacent bigram canonicality, enabling efficient incremental masking.
