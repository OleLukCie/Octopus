from typing import Dict, List, Optional, Any
import json
import os


class DomainKnowledge:
    """Domain-specific knowledge manager.
    
    Centralizes loading and accessing domain data (terms, rules, abbreviations)
    from external files, ensuring core logic remains domain-agnostic.
    """

    def __init__(self, domain: str, data_dir: str = "data"):
        self.domain = domain
        self.data_dir = data_dir
        
        # Load domain resources (all optional; fall back to empty)
        self.terms: Dict[str, str] = self._load_resource("terms")
        self.rules: List[Dict] = self._load_resource("rules")
        self.abbreviations: Dict[str, str] = self._load_resource("abbreviations")

    def _load_resource(self, resource_type: str) -> Any:
        """Generic loader for domain resources (terms/rules/abbreviations)."""
        filename = f"domain_{self.domain}_{resource_type}.json"
        path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(path):
            return {} if resource_type in ["terms", "abbreviations"] else []
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def translate_term(self, term: str) -> str:
        """Translate a domain-specific term using loaded term mappings."""
        return self.terms.get(term, term)  # Fall back to original term

    def apply_transformation_rules(self, text: str) -> str:
        """Apply domain-specific syntax transformation rules."""
        for rule in self.rules:
            text = text.replace(rule["source_pattern"], rule["target_pattern"])
        return text

    def expand_abbreviations(self, text: str) -> str:
        """Expand domain-specific abbreviations (e.g., "心梗" → "心肌梗死")."""
        for abbr, full_form in self.abbreviations.items():
            text = text.replace(abbr, full_form)
        return text