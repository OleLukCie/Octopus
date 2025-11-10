from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.interfaces.coordinator import BaseCoordinator
from src.registry import global_registry


@global_registry.register_coordinator("attention_coordinator_v1")
class AttentionCoordinator(BaseCoordinator):
    """Attention-based coordinator: Dynamically weights subnet contributions.
    
    Uses input text embeddings to compute attention weights for subnets,
    prioritizing those most relevant to the input content.
    """

    def __init__(self, subnet_count: int, embed_dim: int, hidden_dim: int = 256):
        super().__init__(subnet_count, embed_dim)
        
        # Attention network to compute subnet weights
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, subnet_count),
            nn.Softmax(dim=1)  # Normalize to weights
        )

    def forward(
        self,
        subnet_outputs: List[str],
        subnet_features: List[torch.Tensor],
        input_embed: torch.Tensor
    ) -> str:
        # 1. Compute global input embedding (mean over sequence length)
        input_global = torch.mean(input_embed, dim=1)  # Shape: [1, embed_dim]
        
        # 2. Compute attention weights for subnets
        weights = self.attention(input_global)  # Shape: [1, subnet_count]
        
        # 3. Select top-weighted subnet output (simplified fusion; extend with text fusion for production)
        top_idx = torch.argmax(weights, dim=1).item()
        return subnet_outputs[top_idx]