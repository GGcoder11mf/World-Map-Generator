"""
CUDA Physically Accurate Procedural World Generator
====================================================

A GPU-accelerated procedural world generation engine that creates
Earth-like planets with realistic terrain, climate, biomes, and ecosystems.

Architecture:
- CuPy backend for CUDA GPU acceleration (automatic fallback to NumPy)
- Deterministic generation via seed
- Physically-based simulation (tectonics, erosion, climate, hydrology)
- LOD chunking system for planet-scale worlds

Usage:
    from cuda_world_generator import World, WorldConfig

    config = WorldConfig(seed=42, size=1024)
    world = World(config)
    world.generate()
    world.export_terrain("output/")
    world.preview()
"""

__version__ = "1.0.0"
__author__ = "CUDA World Generator Team"

from .world import World
from .core.config import WorldConfig

__all__ = ["World", "WorldConfig"]
