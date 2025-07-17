"""
Configuration management utilities for AmpSearcher.
"""

import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config_data[key] = value

    def save(self) -> None:
        """Save the current configuration to the YAML file."""
        with open(self.config_path, 'w') as config_file:
            yaml.dump(self.config_data, config_file)
