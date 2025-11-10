# src/modules/subnets/syntax.py

from typing import List, Dict, Tuple, Optional
import torch
from src.interfaces.subnet import BaseSubnet
from src.utils.memory import GenericMemoryBank
from src.registry import global_registry


@global_registry.register_subnet("syntax_subnet_v1")
class SyntaxSubnet(BaseSubnet):
    """Syntax transformation subnet: Handles sentence structure conversion.
    
    Applies domain-specific syntax rules to transform source language structure
    into target language conventions.
    """

    def __init__(self, *args, memory_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = GenericMemoryBank(
            max_size=1000,
            save_path=memory_path or f"memory/syntax_{self.domain_knowledge.domain}.json"
        )

    def forward(self, input_text: str, context: str = "") -> Tuple[str, torch.Tensor]:
        # 1. Expand abbreviations and parse syntax
        expanded_text = self.domain_knowledge.expand_abbreviations(input_text)
        syntax = self.src_adapter.parse_syntax(expanded_text)
        
        # 2. Apply domain-specific syntax rules
        transformed_text = self.domain_knowledge.apply_transformation_rules(expanded_text)
        
        # 3. Generate feature vector (mean of transformed text embeddings)
        tgt_embed = self.tgt_adapter.embed(transformed_text)
        feature_vector = torch.mean(tgt_embed, dim=1)  # Shape: [1, embed_dim]

        return transformed_text, feature_vector

    def update_memory(self, samples: List[Dict]) -> None:
        """Update memory with syntax transformation examples."""
        self.memory.add_samples(samples)