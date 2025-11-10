# src/interfaces/coordinator.py

from abc import ABC, abstractmethod
from typing import List
import torch


class BaseCoordinator(ABC, torch.nn.Module):
    """Abstract base class for global coordinators.
    
    Coordinators fuse results from multiple subnets to produce the final translation,
    using attention or other mechanisms to weight subnet contributions.
    """

    @abstractmethod
    def __init__(self, subnet_count: int, embed_dim: int):
        super().__init__()
        self.subnet_count = subnet_count  # Number of subnets to coordinate
        self.embed_dim = embed_dim        # Dimension of feature vectors

    @abstractmethod
    def forward(
        self,
        subnet_outputs: List[str],
        subnet_features: List[torch.Tensor],
        input_embed: torch.Tensor
    ) -> str:
        """Fuse subnet outputs into a final translation.
        
        Args:
            subnet_outputs: Text results from each subnet
            subnet_features: Feature vectors from each subnet
            input_embed: Embedding of the original input text
        
        Returns:
            Final translated text
        """
        raise NotImplementedError