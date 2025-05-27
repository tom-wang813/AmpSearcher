import yaml
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
