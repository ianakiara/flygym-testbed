"""Minimal embodied cognition research substrate for FlyGym."""

from .body_reflex import BodyLayerConfig, BodylessBodyLayer, FlyBodyLayer
from .envs import FlyAvatarEnv, FlyBodyWorldEnv, FlyDualWorldEnv
from .experiments import BenchmarkResult, run_baseline_suite, run_episode

__all__ = [
    "BodyLayerConfig",
    "BodylessBodyLayer",
    "FlyAvatarEnv",
    "FlyBodyLayer",
    "FlyBodyWorldEnv",
    "FlyDualWorldEnv",
    "BenchmarkResult",
    "run_baseline_suite",
    "run_episode",
]
