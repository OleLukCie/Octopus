# src/registry.py

from typing import Dict, Type, Any
from src.interfaces.adapter import BaseLanguageAdapter
from src.interfaces.subnet import BaseSubnet
from src.interfaces.coordinator import BaseCoordinator


class GlobalRegistry:
    """Central registry for modules (adapters, subnets, coordinators).
    
    Enables dynamic module loading via configuration files, critical for
    collaborative development and flexible task assembly.
    """
    def __init__(self):
        self.adapters: Dict[str, Type[BaseLanguageAdapter]] = {}
        self.subnets: Dict[str, Type[BaseSubnet]] = {}
        self.coordinators: Dict[str, Type[BaseCoordinator]] = {}

    # Registration decorators
    def register_adapter(self, name: str):
        def decorator(cls: Type[BaseLanguageAdapter]) -> Type[BaseLanguageAdapter]:
            self.adapters[name] = cls
            return cls
        return decorator

    def register_subnet(self, name: str):
        def decorator(cls: Type[BaseSubnet]) -> Type[BaseSubnet]:
            self.subnets[name] = cls
            return cls
        return decorator

    def register_coordinator(self, name: str):
        def decorator(cls: Type[BaseCoordinator]) -> Type[BaseCoordinator]:
            self.coordinators[name] = cls
            return cls
        return decorator

    # Module retrieval
    def get_adapter(self, name: str, **kwargs) -> BaseLanguageAdapter:
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not registered. Available: {list(self.adapters.keys())}")
        return self.adapters[name](** kwargs)

    def get_subnet(self, name: str, **kwargs) -> BaseSubnet:
        if name not in self.subnets:
            raise ValueError(f"Subnet '{name}' not registered. Available: {list(self.subnets.keys())}")
        return self.subnets[name](** kwargs)

    def get_coordinator(self, name: str, **kwargs) -> BaseCoordinator:
        if name not in self.coordinators:
            raise ValueError(f"Coordinator '{name}' not registered. Available: {list(self.coordinators.keys())}")
        return self.coordinators[name](** kwargs)


# Singleton registry instance
global_registry = GlobalRegistry()