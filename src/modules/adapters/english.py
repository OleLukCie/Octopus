# src/modules/adapters/english.py

from typing import List, Dict
import torch
from transformers import BertTokenizer, BertModel
from src.interfaces.adapter import BaseLanguageAdapter
from src.registry import global_registry


@global_registry.register_adapter("english_adapter_v1")
class EnglishAdapter(BaseLanguageAdapter):
    """English language adapter using BERT for robust text processing.
    
    Handles tokenization, embedding, and syntax parsing for English.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        model_name: str = "bert-base-uncased",
        max_seq_len: int = 128
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Load pre-trained model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        
        # Freeze pre-trained layers by default
        for param in self.model.parameters():
            param.requires_grad = False

    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text into subwords (lowercase by default)."""
        return self.tokenizer.tokenize(text.lower())

    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct English text from tokens (removes special markers)."""
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text.replace("[cls]", "").replace("[sep]", "").strip()

    def embed(self, text: str) -> torch.Tensor:
        """Generate BERT embeddings for text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len
        )
        
        with torch.no_grad():
            outputs = self.model(** inputs)
        
        return outputs.last_hidden_state  # Shape: [1, max_seq_len, embed_dim]

    def parse_syntax(self, text: str) -> Dict:
        """Extract basic syntax features (extend with spaCy for deep parsing)."""
        tokens = self.tokenize(text)
        return {
            "tokens": tokens,
            "token_count": len(tokens),
            "word_count": len(text.split()),
            "structure": "basic"  # Placeholder for dependency parsing
        }