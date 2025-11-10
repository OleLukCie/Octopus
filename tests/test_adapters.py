# tests/test_adapters.py

import unittest
import torch
from src.registry import global_registry


class TestChineseAdapter(unittest.TestCase):
    """Test cases for ChineseAdapter."""

    def setUp(self):
        self.adapter = global_registry.get_adapter(
            "chinese_adapter_v1",
            model_name="bert-base-chinese"
        )
        self.test_text = "心脏病"  # "heart disease"

    def test_tokenize(self):
        tokens = self.adapter.tokenize(self.test_text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_detokenize(self):
        tokens = self.adapter.tokenize(self.test_text)
        reconstructed = self.adapter.detokenize(tokens)
        self.assertEqual(reconstructed, self.test_text)

    def test_embed_shape(self):
        embed = self.adapter.embed(self.test_text)
        self.assertEqual(embed.shape, (1, 128, 768))  # [batch, seq_len, dim]

    def test_parse_syntax(self):
        syntax = self.adapter.parse_syntax(self.test_text)
        self.assertIn("tokens", syntax)
        self.assertIn("token_count", syntax)


class TestEnglishAdapter(unittest.TestCase):
    """Test cases for EnglishAdapter."""

    def setUp(self):
        self.adapter = global_registry.get_adapter(
            "english_adapter_v1",
            model_name="bert-base-uncased"
        )
        self.test_text = "heart disease"

    def test_tokenize(self):
        tokens = self.adapter.tokenize(self.test_text)
        self.assertEqual(tokens, ["heart", "disease"])

    def test_detokenize(self):
        tokens = ["heart", "disease"]
        reconstructed = self.adapter.detokenize(tokens)
        self.assertEqual(reconstructed, self.test_text)

    def test_embed_shape(self):
        embed = self.adapter.embed(self.test_text)
        self.assertEqual(embed.shape, (1, 128, 768))


if __name__ == "__main__":
    unittest.main()