# src/factory.py

import yaml
from typing import Dict, List, Any
from src.modules.knowledge import DomainKnowledge
from src.interfaces.adapter import BaseLanguageAdapter
from src.interfaces.subnet import BaseSubnet
from src.interfaces.coordinator import BaseCoordinator
from src.registry import global_registry
from src.translator import OctopusTranslator


class OctopusTranslatorFactory:
    """Factory for assembling OctopusTranslator instances from config files.
    
    Dynamically loads modules (adapters, subnets, coordinator) based on YAML config,
    enabling flexible task configuration without code changes.
    """

    @staticmethod
    def create_from_config(config_path: str) -> OctopusTranslator:
        """Create a translator instance from a YAML configuration file."""
        # Load and validate config
        config = OctopusTranslatorFactory._load_config(config_path)
        OctopusTranslatorFactory._validate_config(config)

        # Initialize domain knowledge (centralizes all domain data)
        domain_knowledge = DomainKnowledge(
            domain=config["domain"],
            data_dir=config.get("data_dir", "data")
        )

        # Load source and target language adapters
        src_adapter = OctopusTranslatorFactory._load_adapter(
            config["adapters"]["source"],
            config["adapters"]["source_params"]
        )
        tgt_adapter = OctopusTranslatorFactory._load_adapter(
            config["adapters"]["target"],
            config["adapters"]["target_params"]
        )

        # Load subnets (inject adapters and domain knowledge)
        subnets = OctopusTranslatorFactory._load_subnets(
            config["subnets"],
            src_adapter,
            tgt_adapter,
            domain_knowledge
        )

        # Load coordinator
        coordinator = OctopusTranslatorFactory._load_coordinator(
            config["coordinator"]["name"],
            config["coordinator"]["params"],
            len(subnets),
            src_adapter.embed_dim
        )

        # Assemble and return translator
        return OctopusTranslator(
            src_adapter=src_adapter,
            tgt_adapter=tgt_adapter,
            subnets=subnets,
            coordinator=coordinator
        )

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise ValueError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config: {str(e)}")

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate required fields in config."""
        required = ["domain", "adapters", "subnets", "coordinator"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

    @staticmethod
    def _load_adapter(adapter_name: str, params: Dict[str, Any]) -> BaseLanguageAdapter:
        """Load a language adapter from the registry."""
        return global_registry.get_adapter(adapter_name,** params)

    @staticmethod
    def _load_subnets(
        subnet_configs: List[Dict[str, Any]],
        src_adapter: BaseLanguageAdapter,
        tgt_adapter: BaseLanguageAdapter,
        domain_knowledge: DomainKnowledge
    ) -> List[BaseSubnet]:
        """Load subnets from the registry, injecting dependencies."""
        subnets = []
        for cfg in subnet_configs:
            subnet = global_registry.get_subnet(
                cfg["name"],
                src_adapter=src_adapter,
                tgt_adapter=tgt_adapter,
                domain_knowledge=domain_knowledge,
                **cfg.get("params", {})
            )
            subnets.append(subnet)
        return subnets

    @staticmethod
    def _load_coordinator(
        coord_name: str,
        params: Dict[str, Any],
        subnet_count: int,
        embed_dim: int
    ) -> BaseCoordinator:
        """Load a coordinator from the registry."""
        return global_registry.get_coordinator(
            coord_name,
            subnet_count=subnet_count,
            embed_dim=embed_dim,** params
        )