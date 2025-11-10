from typing import List, Dict, Optional
import json
import os
from datetime import datetime


class GenericMemoryBank:
    """Generic memory bank for storing and managing training samples.
    
    Used by subnets to retain recent data for incremental learning and
    context-aware processing.
    """

    def __init__(
        self,
        max_size: int = 1000,
        save_path: Optional[str] = None,
        auto_save: bool = True
    ):
        self.max_size = max_size      # Max samples to store
        self.save_path = save_path    # Path for persistence (optional)
        self.auto_save = auto_save    # Auto-save on update
        self.memory: List[Dict] = []  # Stores {"src":..., "tgt":..., "context":..., "timestamp":...}

        # Load existing memory if available
        if self.save_path and os.path.exists(self.save_path):
            self.load()

    def add_samples(self, samples: List[Dict]) -> None:
        """Add new samples with timestamps, truncating old data if needed."""
        timestamped = [
            {**sample, "timestamp": datetime.utcnow().isoformat()}
            for sample in samples
        ]
        self.memory.extend(timestamped)
        
        # Truncate to max size (keep most recent)
        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size:]
        
        # Auto-save if enabled
        if self.auto_save and self.save_path:
            self.save()

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Retrieve n most recent samples (excluding timestamps)."""
        recent = self.memory[-n:] if len(self.memory) >= n else self.memory
        return [{k: v for k, v in s.items() if k != "timestamp"} for s in recent]

    def save(self) -> None:
        """Persist memory to disk."""
        if not self.save_path:
            raise ValueError("No save path specified for memory bank.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """Load memory from disk."""
        if not self.save_path or not os.path.exists(self.save_path):
            return
        
        with open(self.save_path, "r", encoding="utf-8") as f:
            self.memory = json.load(f)