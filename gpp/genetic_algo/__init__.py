"""Genetic algorithm utilities for patch pruning with CLIP.

Modules:
- config: configuration loading and device selection
- io: JSONL save/restore helpers
- clip_model: CLIP loading and low-level forwards
- scoring: fitness components and combined objective
- ga: genetic algorithm implementation
- runner: end-to-end evaluation loop using GA-based patch selection
"""

__all__ = [
    "config",
    "io",
    "clip_model",
    "scoring",
    "ga",
    "runner",
]

