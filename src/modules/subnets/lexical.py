# src/modules/subnets/lexical.py

from typing import List, Dict, Tuple, Optional
import torch
from src.interfaces.subnet import BaseSubnet
from src.utils.memory import GenericMemoryBank
from src.registry import global_registry


@global_registry.register_subnet("lexical_subnet_v1")
class LexicalSubnet(BaseSubnet):
    """Lexical alignment subnet: Focuses on term translation and word-level mapping.
    
    Uses domain terms and historical memory to align source/target vocabulary.
    """

    def __init__(self, *args, memory_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = GenericMemoryBank(
            max_size=1000,
            save_path=memory_path or f"memory/lexical_{self.domain_knowledge.domain}.json"
        )

    def forward(self, input_text: str, context: str = "") -> Tuple[str, torch.Tensor]:
        # 1. Expand abbreviations first (critical for accurate term matching)
        expanded_text = self.domain_knowledge.expand_abbreviations(input_text)
        
        # 2. Tokenize and translate domain terms
        src_tokens = self.src_adapter.tokenize(expanded_text)
        translated_tokens = [self.domain_knowledge.translate_term(token) for token in src_tokens]
        translated_text = self.tgt_adapter.detokenize(translated_tokens)
        
        # 3. Generate feature vector (mean of source embeddings)
        src_embed = self.src_adapter.embed(expanded_text)
        feature_vector = torch.mean(src_embed, dim=1)  # Shape: [1, embed_dim]

        return translated_text, feature_vector

    def update_memory(self, samples: List[Dict]) -> None:
        """Update memory with new lexical pairs for future reference."""
        self.memory.add_samples(samples)