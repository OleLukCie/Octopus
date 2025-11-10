# src/modules/subnets/context.py

from typing import List, Dict, Tuple, Optional
import torch
from src.interfaces.subnet import BaseSubnet
from src.utils.memory import GenericMemoryBank
from src.registry import global_registry


@global_registry.register_subnet("context_subnet_v1")
class ContextSubnet(BaseSubnet):
    """Context understanding subnet: Resolves ambiguities using context.
    
    Handles pronoun reference, elliptical structures, and cross-sentence dependencies.
    """

    def __init__(self, *args, memory_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = GenericMemoryBank(
            max_size=1000,
            save_path=memory_path or f"memory/context_{self.domain_knowledge.domain}.json"
        )

    def forward(self, input_text: str, context: str = "") -> Tuple[str, torch.Tensor]:
        # 1. Combine context and input text
        full_context = f"{context}. {input_text}" if context else input_text
        
        # 2. Expand abbreviations in combined text
        expanded_context = self.domain_knowledge.expand_abbreviations(full_context)
        
        # 3. Simple pronoun resolution (extend with coreference models for production)
        resolved_text = self._resolve_pronouns(expanded_context)
        
        # 4. Generate feature vector (fused context + input embeddings)
        context_embed = self.src_adapter.embed(context) if context else torch.zeros(1, 1, self.src_adapter.embed_dim)
        input_embed = self.src_adapter.embed(input_text)
        combined_embed = torch.cat([context_embed, input_embed], dim=1)
        feature_vector = torch.mean(combined_embed, dim=1)  # Shape: [1, embed_dim]

        return resolved_text, feature_vector

    def _resolve_pronouns(self, text: str) -> str:
        """Resolve common pronouns using domain context (simplified example)."""
        # Replace Chinese pronouns with domain-appropriate nouns (e.g., "patient" in medical)
        return text.replace("他", "the patient").replace("她", "the patient").replace("它", "the object")

    def update_memory(self, samples: List[Dict]) -> None:
        """Update memory with context-aware examples."""
        self.memory.add_samples(samples)