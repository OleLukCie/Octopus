# src/translator.py

from typing import List, Dict, Optional
import torch
from src.interfaces.adapter import BaseLanguageAdapter
from src.interfaces.subnet import BaseSubnet
from src.interfaces.coordinator import BaseCoordinator


class OctopusTranslator:
    """Main translation class: Orchestrates adapters, subnets, and coordinator.
    
    Provides high-level interfaces for translation and model management.
    """

    def __init__(
        self,
        src_adapter: BaseLanguageAdapter,
        tgt_adapter: BaseLanguageAdapter,
        subnets: List[BaseSubnet],
        coordinator: BaseCoordinator
    ):
        self.src_adapter = src_adapter
        self.tgt_adapter = tgt_adapter
        self.subnets = subnets
        self.coordinator = coordinator

    def translate(self, text: str, context: str = "") -> str:
        """Translate text from source to target language.
        
        Args:
            text: Source language text to translate
            context: Optional context for disambiguation
        
        Returns:
            Final translated text
        """
        # Run subnets in parallel (simulated; use torch.multiprocessing for true parallelism)
        subnet_outputs = []
        subnet_features = []
        for subnet in self.subnets:
            output, feature = subnet.forward(text, context)
            subnet_outputs.append(output)
            subnet_features.append(feature)

        # Generate input embedding for coordinator
        input_embed = self.src_adapter.embed(text)

        # Coordinate to get final result
        return self.coordinator.forward(subnet_outputs, subnet_features, input_embed)

    def update_memory(self, samples: List[Dict]) -> None:
        """Update all subnets with new training samples.
        
        Args:
            samples: List of {"src":..., "tgt":..., "context":...}
        """
        for subnet in self.subnets:
            subnet.update_memory(samples)

    def save(self, path: str) -> None:
        """Save model state to disk.
        
        Args:
            path: Path to save checkpoint (.pth file)
        """
        torch.save({
            "src_adapter": self.src_adapter.state_dict(),
            "tgt_adapter": self.tgt_adapter.state_dict(),
            "subnets": [s.state_dict() for s in self.subnets],
            "coordinator": self.coordinator.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """Load model state from disk.
        
        Args:
            path: Path to checkpoint (.pth file)
        """
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.src_adapter.load_state_dict(checkpoint["src_adapter"])
        self.tgt_adapter.load_state_dict(checkpoint["tgt_adapter"])
        
        for i, subnet in enumerate(self.subnets):
            subnet.load_state_dict(checkpoint["subnets"][i])
        
        self.coordinator.load_state_dict(checkpoint["coordinator"])

    def train(self) -> None:
        """Set all modules to training mode."""
        self.src_adapter.train()
        self.tgt_adapter.train()
        for subnet in self.subnets:
            subnet.train()
        self.coordinator.train()

    def eval(self) -> None:
        """Set all modules to evaluation mode."""
        self.src_adapter.eval()
        self.tgt_adapter.eval()
        for subnet in self.subnets:
            subnet.eval()
        self.coordinator.eval()