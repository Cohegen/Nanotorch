"""Unit tests for the tokenization module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from tokenization.tokenization import (
    CharTokenizer,
    BPETokenizer,
    create_tokenizer,
    tokenize_dataset,
    analyze_tokenization,
)


class TestCharTokenizer:
    def test_build_vocab(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello")
        assert len(tokenizer.char_to_id) > 0
        assert len(tokenizer.id_to_char) > 0

    def test_encode_decode_roundtrip(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello world")
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        assert decoded == "hello"

    def test_vocab_size(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("abc")
        # Should include unique chars + special tokens
        assert tokenizer.vocab_size >= 3

    def test_encode_returns_list(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("test")
        result = tokenizer.encode("test")
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)

    def test_unk_token(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hi")
        # Should have UNK token
        assert tokenizer.unk_id is not None

    def test_unknown_char_handling(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("abc")
        # Encoding a char not in vocab should use UNK token
        encoded = tokenizer.encode("z")
        assert tokenizer.unk_id in encoded


class TestBPETokenizer:
    def test_train_and_encode(self):
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train("hello hello world world")
        encoded = tokenizer.encode("hello")
        assert isinstance(encoded, list)
        assert len(encoded) > 0

    def test_decode(self):
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train("hello hello world world")
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_vocab_size_constraint(self):
        tokenizer = BPETokenizer(vocab_size=30)
        tokenizer.train("the quick brown fox jumps over the lazy dog")
        assert len(tokenizer.vocab) <= 30 + 10  # Allow some margin for base vocab

    def test_empty_input(self):
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.train("hello world")
        encoded = tokenizer.encode("")
        assert isinstance(encoded, list)


class TestCreateTokenizer:
    def test_create_char_tokenizer(self):
        tokenizer = create_tokenizer("char", "hello world")
        assert isinstance(tokenizer, CharTokenizer)

    def test_create_bpe_tokenizer(self):
        tokenizer = create_tokenizer("bpe", "hello world hello world")
        assert isinstance(tokenizer, BPETokenizer)


class TestTokenizeDataset:
    def test_tokenize_dataset(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello world")
        texts = ["hello", "world"]
        result = tokenize_dataset(texts, tokenizer)
        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)


class TestAnalyzeTokenization:
    def test_analyze(self):
        tokenizer = CharTokenizer()
        tokenizer.build_vocab("hello world")
        texts = ["hello", "world"]
        # Should run without error
        stats = analyze_tokenization(texts, tokenizer)
        assert isinstance(stats, dict)
