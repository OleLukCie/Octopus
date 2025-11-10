# src/modules/subnets/domain.py

from typing import List, Dict, Tuple, Optional
import torch
from src.interfaces.subnet import BaseSubnet
from src.utils.memory import GenericMemoryBank
from src.registry import global_registry


@global_registry.register_subnet("domain_subnet_v1")
class DomainSubnet(BaseSubnet):
    """Domain adaptation subnet: Specializes in domain-specific expressions.
    
    Combines abbreviation expansion, term translation, and domain conventions
    to handle specialized language (e.g., medical jargon, legal terms).
    """

    def __init__(self, *args, memory_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = GenericMemoryBank(
            max_size=1000,
            save_path=memory_path or f"memory/domain_{self.domain_knowledge.domain}.json"
        )

    def forward(self, input_text: str, context: str = "") -> Tuple[str, torch.Tensor]:
        # 1. Expand domain abbreviations (critical for domain understanding)
        expanded_text = self.domain_knowledge.expand_abbreviations(input_text)
        
        # 2. Apply domain transformation rules
        rule_transformed = self.domain_knowledge.apply_transformation_rules(expanded_text)
        
        # 3. Translate domain-specific terms
        translated_text = self.domain_knowledge.translate_term(rule_transformed)
        
        # 4. Generate feature vector (domain-specific embeddings)
        domain_embed = self.src_adapter.embed(expanded_text)  # Use expanded text for embedding
        feature_vector = torch.mean(domain_embed, dim=1)  # Shape: [1, embed_dim]

        return translated_text, feature_vector

    def update_memory(self, samples: List[Dict]) -> None:
        """Update memory with domain-specific examples."""
        self.memory.add_samples(samples)