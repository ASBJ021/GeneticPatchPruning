import os
import yaml
import torch
from typing import Any, Dict


def load_config(cfg_path: str) -> Dict[str, Any]:
    """Load YAML configuration from a path.

    Parameters
    - cfg_path: path to YAML config file
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_device(cfg: Dict[str, Any]) -> str:
    """Resolve device from config with CUDA availability fallback."""
    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available():
        return "cpu"
    return device


def default_config_path() -> str:
    """Return default config path relative to this module file."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

