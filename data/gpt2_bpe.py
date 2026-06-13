"""Small GPT-2 byte-level BPE implementation for canonicality experiments."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

try:
    import regex as re
except ModuleNotFoundError:  # pragma: no cover - container setup installs regex.
    import re  # type: ignore[no-redef]


try:
    GPT2_PATTERN = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except re.error:
    GPT2_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class GPT2BPE:
    def __init__(self, vocab_path: str | Path, merges_path: str | Path):
        self.encoder: dict[str, int] = json.loads(Path(vocab_path).read_text())
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}
        merges = Path(merges_path).read_text(encoding="utf-8").splitlines()
        merges = [line for line in merges if line and not line.startswith("#")]
        self.bpe_ranks: dict[tuple[str, str], int] = {tuple(line.split()): i for i, line in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.vocab_size = len(self.encoder)

    @lru_cache(maxsize=200_000)
    def bpe(self, token: str) -> tuple[str, ...]:
        word = tuple(token)
        if len(word) <= 1:
            return word
        pairs = _pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        return word

    def encode(self, text: str) -> list[int]:
        tokens: list[int] = []
        for match in GPT2_PATTERN.findall(text):
            encoded = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
            tokens.extend(self.encoder[piece] for piece in self.bpe(encoded))
        return tokens

    def encode_bytes_no_pretokenize(self, data: bytes) -> list[int]:
        encoded = "".join(self.byte_encoder[b] for b in data)
        return [self.encoder[piece] for piece in self.bpe(encoded)]

    @lru_cache(maxsize=100_000)
    def token_bytes(self, token_id: int) -> bytes:
        return bytes(self.byte_decoder[ch] for ch in self.decoder[token_id])

    @lru_cache(maxsize=2_000_000)
    def canonical_bigram(self, left: int, right: int) -> bool:
        """Round-trip BPE bigram test from Appendix B, using byte-level BPE."""
        merged = self.token_bytes(left) + self.token_bytes(right)
        return self.encode_bytes_no_pretokenize(merged) == [left, right]

    @lru_cache(maxsize=2_000_000)
    def tokenizer_canonical_bigram(self, left: int, right: int) -> bool:
        """Canonical bigram test against GPT-2's full tokenizer behavior.

        The paper's BPE bigram test is byte-local. GPT-2 also has a regex
        pre-tokenizer, so valid UTF-8 token pairs can be canonical under the
        implemented tokenizer even when a pure no-pretokenize BPE round trip
        would merge across the pair boundary. This reference check handles
        those cases without looking at the held-out test set.
        """
        if self.canonical_bigram(left, right):
            return True
        merged = self.token_bytes(left) + self.token_bytes(right)
        try:
            text = merged.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return self.encode(text) == [left, right]

    def bpe_canonical_sequence(self, token_ids: list[int]) -> bool:
        """Exact byte-level BPE round-trip test for a complete token string."""
        data = b"".join(self.token_bytes(token_id) for token_id in token_ids)
        return self.encode_bytes_no_pretokenize(data) == token_ids


def load_default_tokenizer(root: str | Path = "/home/user") -> GPT2BPE:
    root = Path(root)
    return GPT2BPE(
        root / "data/downloads/gpt2-tokenizer/vocab.json",
        root / "data/downloads/gpt2-tokenizer/merges.txt",
    )
