# src/interfaces/adapter.py

from abc import ABC, abstractmethod
from typing import List, Dict
import torch


class BaseLanguageAdapter(ABC, torch.nn.Module):
    """Abstract base class for language adapters.
    
    Defines core language processing operations (tokenization, embedding, etc.)
    that must be implemented for each language.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split text into atomic tokens (words/subwords)."""
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: str) -> torch.Tensor:
        """Generate dense semantic embeddings for text.
        
        Returns:
            Tensor of shape [1, seq_len, embed_dim]
        """
        raise NotImplementedError

    @abstractmethod
    def parse_syntax(self, text: str) -> Dict:
        """Extract syntactic structure (e.g., dependencies, phrase boundaries)."""
        raise NotImplementedError