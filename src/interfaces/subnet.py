# src/interfaces/subnet.py

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import torch
from src.interfaces.adapter import BaseLanguageAdapter
from src.modules.knowledge import DomainKnowledge


class BaseSubnet(ABC, torch.nn.Module):
    """Abstract base class for task-specific subnets.
    
    Subnets specialize in narrow tasks (lexical alignment, syntax transformation)
    and collaborate via the coordinator.
    """

    @abstractmethod
    def __init__(
        self,
        src_adapter: BaseLanguageAdapter,
        tgt_adapter: BaseLanguageAdapter,
        domain_knowledge: DomainKnowledge
    ):
        super().__init__()
        self.src_adapter = src_adapter
        self.tgt_adapter = tgt_adapter
        self.domain_knowledge = domain_knowledge

    @abstractmethod
    def forward(self, input_text: str, context: str = "") -> Tuple[str, torch.Tensor]:
        """Process input text and return intermediate result + features.
        
        Args:
            input_text: Source language text to process
            context: Optional context for disambiguation
        
        Returns:
            Intermediate translation (target language)
            Feature vector (shape [1, embed_dim]) for coordinator
        """
        raise NotImplementedError

    @abstractmethod
    def update_memory(self, samples: List[Dict]) -> None:
        """Update subnet memory with new training samples.
        
        Args:
            samples: List of {"src":..., "tgt":..., "context":...}
        """
        raise NotImplementedError