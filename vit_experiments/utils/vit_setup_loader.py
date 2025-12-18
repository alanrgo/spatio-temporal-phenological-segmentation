import yaml
from pathlib import Path
from typing import Dict, Any
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config with base inheritance support."""
    path = os.path.join(os.getcwd(), config_path)
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base inheritance
    if 'base' in config:
        base_path = Path(config_path).parent / config.pop('base')
        base_config = load_config(str(base_path))
        config = deep_merge(base_config, config)
        
    # Load extends if specified
    if 'extends' in config:
        extends_path = Path(config_path).parent / config.pop('extends')
        extends_config = load_config(extends_path)
        config = deep_merge(extends_config, config)
    
    return config

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result