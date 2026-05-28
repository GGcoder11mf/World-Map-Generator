"""Core module for the CUDA World Generator."""

from .config import WorldConfig
from .noise import NoiseGenerator
from .tectonics import TectonicSimulator
from .erosion import ErosionSimulator
from .climate import ClimateSimulator
from .biome import BiomeClassifier
from .hydrology import HydrologySimulator
from .ecosystem import EcosystemSimulator
from .lod import LODManager
from .backend import xp, gpu_available, to_cpu, to_gpu

__all__ = [
    "WorldConfig",
    "NoiseGenerator",
    "TectonicSimulator",
    "ErosionSimulator",
    "ClimateSimulator",
    "BiomeClassifier",
    "HydrologySimulator",
    "EcosystemSimulator",
    "LODManager",
    "xp",
    "gpu_available",
    "to_cpu",
    "to_gpu",
]
