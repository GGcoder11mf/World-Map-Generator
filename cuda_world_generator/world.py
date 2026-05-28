"""
World Generator - Main API
============================

The World class ties together all simulation subsystems into a
coherent, easy-to-use API for generating and interacting with
procedural worlds.

Pipeline:
1. generate_world(seed) - Full world generation
2. simulate_step(delta_time) - Advance simulation
3. export_terrain() - Export all maps
4. preview() - Visualize the world
"""

import numpy as np
import os
from .core.config import WorldConfig
from .core.backend import xp, to_cpu, to_gpu, gpu_available
from .core.noise import NoiseGenerator
from .core.tectonics import TectonicSimulator
from .core.erosion import ErosionSimulator
from .core.climate import ClimateSimulator
from .core.biome import BiomeClassifier
from .core.hydrology import HydrologySimulator
from .core.ecosystem import EcosystemSimulator
from .core.lod import LODManager


class World:
    """
    Procedural world generator with GPU acceleration.

    Creates a physically accurate Earth-like world with terrain,
    climate, biomes, rivers, and ecosystems.

    Parameters
    ----------
    config : WorldConfig or dict
        World configuration. If dict, passed to WorldConfig constructor.

    Examples
    --------
    >>> from cuda_world_generator import World, WorldConfig
    >>> config = WorldConfig(seed=42, size=1024)
    >>> world = World(config)
    >>> world.generate()
    >>> world.preview()
    >>> world.export_terrain("output/")
    """

    def __init__(self, config=None):
        if config is None:
            config = WorldConfig()
        elif isinstance(config, dict):
            config = WorldConfig(**config)

        config.validate()
        self.config = config

        # Simulation subsystems
        self.noise = NoiseGenerator(config.seed)
        self.tectonics = TectonicSimulator(config)
        self.erosion = ErosionSimulator(config)
        self.climate = ClimateSimulator(config)
        self.biomes = BiomeClassifier(config)
        self.hydrology = HydrologySimulator(config)
        self.ecosystem = EcosystemSimulator(config)
        self.lod = LODManager(config)

        # World data arrays (None until generated)
        self.heightmap = None
        self.plate_map = None
        self.boundary_map = None
        self.temperature = None
        self.humidity = None
        self.rainfall = None
        self.wind_u = None
        self.wind_v = None
        self.ocean_temp = None
        self.rain_shadow = None
        self.biome_map = None
        self.soil_fertility = None
        self.river_map = None
        self.flow_accumulation = None
        self.lake_map = None
        self.river_graph = None
        self.vegetation_density = None
        self.tree_density = None
        self.animal_probability = None
        self.biodiversity = None
        self.sediment_map = None
        self.water_map = None

        # Metadata
        self.generation_time = 0
        self.sea_level = config.sea_level
        self._generated = False

    def generate(self):
        """
        Run the full world generation pipeline.

        Steps:
        1. Generate base terrain from multi-octave noise
        2. Apply tectonic plate simulation
        3. Apply hydraulic and thermal erosion
        4. Simulate climate (temperature, humidity, wind)
        5. Compute hydrology (rivers, lakes)
        6. Classify biomes
        7. Generate ecosystem maps

        All steps use GPU-parallel array operations.
        """
        import time
        start = time.time()

        config = self.config
        size = config.size
        w, h = size, size

        print(f"[World Generator] Initializing world {w}x{h} with seed={config.seed}")
        print(f"[World Generator] GPU acceleration: {'ACTIVE (CuPy)' if gpu_available else 'OFF (NumPy fallback)'}")

        # ── Step 1: Base Terrain ────────────────────────────────────────
        print("[1/7] Generating base terrain (multi-octave fBm + domain warp)...")
        self._generate_base_terrain(w, h)

        # ── Step 2: Tectonic Plates ────────────────────────────────────
        print("[2/7] Simulating tectonic plates...")
        self._apply_tectonics(w, h)

        # ── Step 3: Erosion ────────────────────────────────────────────
        print("[3/7] Applying erosion (hydraulic + thermal)...")
        self._apply_erosion(w, h)

        # ── Step 4: Climate ────────────────────────────────────────────
        print("[4/7] Simulating climate system...")
        self._simulate_climate(w, h)

        # ── Step 5: Hydrology ──────────────────────────────────────────
        print("[5/7] Computing hydrology (rivers, lakes)...")
        self._compute_hydrology(w, h)

        # ── Step 6: Biomes ─────────────────────────────────────────────
        print("[6/7] Classifying biomes...")
        self._classify_biomes(w, h)

        # ── Step 7: Ecosystems ─────────────────────────────────────────
        print("[7/7] Generating ecosystem maps...")
        self._generate_ecosystems(w, h)

        self._generated = True
        self.generation_time = time.time() - start
        print(f"[World Generator] Generation complete in {self.generation_time:.2f}s")

    def _generate_base_terrain(self, w, h):
        """Generate base heightmap using layered noise with smooth blending."""
        config = self.config

        # Layer 1: Large-scale continent shapes (domain-warped fBm)
        continent_noise = self.noise.domain_warp(
            w, h,
            warp_strength=0.5,
            octaves=config.terrain_octaves,
            persistence=config.terrain_persistence,
            lacunarity=config.terrain_lacunarity,
            scale=config.terrain_scale * 0.7  # Larger scale = smoother continents
        )

        # Layer 2: Medium-scale terrain variation
        terrain_noise = self.noise.fbm(
            w, h,
            octaves=config.terrain_octaves,
            persistence=config.terrain_persistence,
            lacunarity=config.terrain_lacunarity,
            scale=config.terrain_scale * 2
        )

        # Layer 3: Fine detail (subtle)
        detail_noise = self.noise.fbm(
            w, h,
            octaves=max(config.terrain_octaves - 2, 2),
            persistence=config.terrain_persistence * 0.8,
            lacunarity=config.terrain_lacunarity,
            scale=config.terrain_scale * 6
        )

        # Combine layers with decreasing amplitude
        # Continent noise dominates for smooth landmasses
        heightmap = (
            continent_noise * 0.70 +
            terrain_noise * 0.22 +
            detail_noise * 0.08
        )

        # Normalize to [0, 1]
        hmin = heightmap.min()
        hmax = heightmap.max()
        if hmax > hmin:
            heightmap = (heightmap - hmin) / (hmax - hmin)

        # Apply smooth sea level transition
        # Use a smoothstep instead of hard cutoff for natural coastlines
        sea = config.sea_level
        # Smooth transition zone around sea level
        transition = 0.05
        below_sea = xp.where(
            heightmap < sea - transition,
            heightmap * 0.4,  # Compress ocean depths
            xp.where(
                heightmap < sea + transition,
                # Smooth interpolation in transition zone
                heightmap * 0.4 + (heightmap - (sea - transition)) * 3.0 * 0.6,
                sea + (heightmap - sea) * 1.3  # Boost land
            )
        )
        heightmap = below_sea

        # Re-normalize
        hmin = heightmap.min()
        hmax = heightmap.max()
        if hmax > hmin:
            heightmap = (heightmap - hmin) / (hmax - hmin)

        # Final smoothing pass — gentle Gaussian-like blur to remove artifacts
        # 3x3 average filter using padded operations (no wrapping)
        padded = xp.pad(heightmap, 1, mode='edge')
        heightmap = (
            padded[:-2, 1:-1] * 0.0625 +
            padded[2:, 1:-1] * 0.0625 +
            padded[1:-1, :-2] * 0.0625 +
            padded[1:-1, 2:] * 0.0625 +
            padded[1:-1, 1:-1] * 0.75
        )

        self.heightmap = heightmap

    def _apply_tectonics(self, w, h):
        """Apply tectonic plate simulation to modify terrain."""
        # Generate plates
        self.plate_map = self.tectonics.generate_plates(w, h)

        # Compute boundaries
        self.boundary_map, self.boundary_type_map = self.tectonics.compute_boundaries(w, h)

        # Apply tectonic modifications to heightmap
        self.heightmap = self.tectonics.apply_tectonics(self.heightmap, w, h)

        # Re-normalize
        hmin = self.heightmap.min()
        hmax = self.heightmap.max()
        if hmax > hmin:
            self.heightmap = (self.heightmap - hmin) / (hmax - hmin)

    def _apply_erosion(self, w, h):
        """Apply erosion simulation."""
        # Generate rainfall-weighted erosion
        rainfall_simple = self.noise.fbm(
            w, h, octaves=4, scale=3.0,
            offset_x=self.config.seed * 0.1,
            offset_y=self.config.seed * 0.2
        )
        rainfall_simple = (rainfall_simple + 1) / 2  # Normalize to [0, 1]

        self.heightmap, self.water_map, self.sediment_map = self.erosion.erode(
            self.heightmap, rainfall_simple
        )

        # Apply coastal erosion
        self.heightmap = self.erosion.coastal_erosion(
            self.heightmap, self.sea_level
        )

        # Final normalization
        hmin = self.heightmap.min()
        hmax = self.heightmap.max()
        if hmax > hmin:
            self.heightmap = (self.heightmap - hmin) / (hmax - hmin)

    def _simulate_climate(self, w, h):
        """Simulate climate system."""
        self.temperature, self.humidity, self.wind_u, self.wind_v, \
            self.rainfall, self.ocean_temp, self.rain_shadow = self.climate.simulate(
                self.heightmap, self.sea_level, w, h
            )

    def _compute_hydrology(self, w, h):
        """Compute river networks and lakes."""
        self.river_map, self.flow_accumulation, self.lake_map, \
            self.river_graph = self.hydrology.simulate(
                self.heightmap, self.rainfall, self.sea_level, w, h
            )

    def _classify_biomes(self, w, h):
        """Classify biomes."""
        self.biome_map, self.soil_fertility = self.biomes.classify(
            self.heightmap, self.temperature, self.humidity,
            self.rainfall, self.sea_level, w, h
        )

    def _generate_ecosystems(self, w, h):
        """Generate ecosystem maps."""
        self.vegetation_density, self.tree_density, \
            self.animal_probability, self.biodiversity = self.ecosystem.simulate(
                self.biome_map, self.temperature, self.humidity,
                self.rainfall, self.soil_fertility,
                self.heightmap, self.sea_level, w, h
            )

    def simulate_step(self, delta_time=1.0):
        """
        Advance the world simulation by one time step.

        Simulates:
        - Plate tectonic motion (continental drift)
        - Erosion (continued weathering)
        - Climate evolution (seasonal changes)

        Parameters
        ----------
        delta_time : float
            Time step in simulation units (millions of years for tectonics).
        """
        if not self._generated:
            raise RuntimeError("World must be generated first. Call generate().")

        w, h = self.config.size, self.config.size

        # Tectonic drift
        self.tectonics.simulate_drift(delta_time, w, h)
        self.heightmap = self.tectonics.apply_tectonics(self.heightmap, w, h)

        # Erosion (reduced iterations for step updates)
        saved_hyd_iter = self.config.hydraulic_erosion_iterations
        saved_therm_iter = self.config.thermal_erosion_iterations
        self.config.hydraulic_erosion_iterations = max(saved_hyd_iter // 4, 5)
        self.config.thermal_erosion_iterations = max(saved_therm_iter // 4, 3)
        self.heightmap, self.water_map, self.sediment_map = self.erosion.erode(
            self.heightmap
        )
        self.config.hydraulic_erosion_iterations = saved_hyd_iter
        self.config.thermal_erosion_iterations = saved_therm_iter

        # Re-simulate climate
        self._simulate_climate(w, h)

        # Re-compute hydrology
        self._compute_hydrology(w, h)

        # Re-classify biomes
        self._classify_biomes(w, h)

        # Re-generate ecosystems
        self._generate_ecosystems(w, h)

        # Normalize
        hmin = self.heightmap.min()
        hmax = self.heightmap.max()
        if hmax > hmin:
            self.heightmap = (self.heightmap - hmin) / (hmax - hmin)

    def export_terrain(self, output_dir=None):
        """
        Export all world data to files.

        Parameters
        ----------
        output_dir : str, optional
            Output directory. Defaults to config.output_dir.

        Returns
        -------
        files : dict
            Dictionary of exported file paths.
        """
        if not self._generated:
            raise RuntimeError("World must be generated first. Call generate().")

        if output_dir is None:
            output_dir = self.config.output_dir

        os.makedirs(output_dir, exist_ok=True)
        files = {}

        # Export NumPy compressed archive with all maps
        npz_path = os.path.join(output_dir, "world_data.npz")
        np.savez_compressed(
            npz_path,
            heightmap=to_cpu(self.heightmap),
            plate_map=to_cpu(self.plate_map),
            temperature=to_cpu(self.temperature),
            humidity=to_cpu(self.humidity),
            rainfall=to_cpu(self.rainfall),
            rain_shadow=to_cpu(self.rain_shadow),
            biome_map=to_cpu(self.biome_map),
            soil_fertility=to_cpu(self.soil_fertility),
            river_map=to_cpu(self.river_map),
            flow_accumulation=to_cpu(self.flow_accumulation),
            lake_map=to_cpu(self.lake_map),
            vegetation_density=to_cpu(self.vegetation_density),
            tree_density=to_cpu(self.tree_density),
            animal_probability=to_cpu(self.animal_probability),
            biodiversity=to_cpu(self.biodiversity),
            wind_u=to_cpu(self.wind_u),
            wind_v=to_cpu(self.wind_v),
            sediment_map=to_cpu(self.sediment_map),
            sea_level=np.array([self.sea_level]),
            config_seed=np.array([self.config.seed]),
        )
        files['npz'] = npz_path
        print(f"[Export] Saved world data to {npz_path}")

        return files

    def get_heightmap(self):
        """Return heightmap as NumPy array (CPU)."""
        return to_cpu(self.heightmap) if self.heightmap is not None else None

    def get_biome_map(self):
        """Return biome map as NumPy array (CPU)."""
        return to_cpu(self.biome_map) if self.biome_map is not None else None

    def get_temperature_map(self):
        """Return temperature map as NumPy array (CPU)."""
        return to_cpu(self.temperature) if self.temperature is not None else None

    def get_humidity_map(self):
        """Return humidity map as NumPy array (CPU)."""
        return to_cpu(self.humidity) if self.humidity is not None else None

    def get_river_map(self):
        """Return river map as NumPy array (CPU)."""
        return to_cpu(self.river_map) if self.river_map is not None else None

    def get_river_graph(self):
        """Return river network graph."""
        return self.river_graph

    def get_all_maps(self):
        """
        Return all generated maps as a dictionary of NumPy arrays.

        Returns
        -------
        maps : dict
            All world maps transferred to CPU memory.
        """
        if not self._generated:
            raise RuntimeError("World must be generated first.")

        return {
            'heightmap': to_cpu(self.heightmap),
            'plate_map': to_cpu(self.plate_map),
            'temperature': to_cpu(self.temperature),
            'humidity': to_cpu(self.humidity),
            'rainfall': to_cpu(self.rainfall),
            'biome_map': to_cpu(self.biome_map),
            'soil_fertility': to_cpu(self.soil_fertility),
            'river_map': to_cpu(self.river_map),
            'flow_accumulation': to_cpu(self.flow_accumulation),
            'lake_map': to_cpu(self.lake_map),
            'vegetation_density': to_cpu(self.vegetation_density),
            'tree_density': to_cpu(self.tree_density),
            'animal_probability': to_cpu(self.animal_probability),
            'biodiversity': to_cpu(self.biodiversity),
            'wind_u': to_cpu(self.wind_u),
            'wind_v': to_cpu(self.wind_v),
            'rain_shadow': to_cpu(self.rain_shadow),
        }

    def get_world_stats(self):
        """
        Compute and return statistics about the generated world.

        Returns
        -------
        stats : dict
            World statistics.
        """
        if not self._generated:
            raise RuntimeError("World must be generated first.")

        h = to_cpu(self.heightmap)
        t = to_cpu(self.temperature)
        b = to_cpu(self.biome_map)

        is_ocean = h < self.sea_level
        land_fraction = 1.0 - np.mean(is_ocean)

        unique_biomes = np.unique(b[~is_ocean].astype(int))
        biome_names = [self.biomes.get_biome_name(bid) for bid in unique_biomes]

        stats = {
            'seed': self.config.seed,
            'resolution': self.config.size,
            'generation_time_s': self.generation_time,
            'gpu_accelerated': gpu_available,
            'land_fraction': float(land_fraction),
            'ocean_fraction': float(1 - land_fraction),
            'max_elevation': float(h.max() * self.config.max_elevation_km),
            'mean_temperature': float(t[~is_ocean].mean()) if land_fraction > 0 else 0,
            'min_temperature': float(t[~is_ocean].min()) if land_fraction > 0 else 0,
            'max_temperature': float(t[~is_ocean].max()) if land_fraction > 0 else 0,
            'num_biomes': len(unique_biomes),
            'biome_list': biome_names,
            'num_river_nodes': len(self.river_graph['nodes']) if self.river_graph else 0,
            'num_river_edges': len(self.river_graph['edges']) if self.river_graph else 0,
        }

        return stats


def generate_world(seed=42, size=1024, **kwargs):
    """
    Convenience function to generate a world.

    Parameters
    ----------
    seed : int
        Random seed for deterministic generation.
    size : int
        Heightmap resolution (power of 2).
    **kwargs
        Additional WorldConfig parameters.

    Returns
    -------
    world : World
        Generated world object.
    """
    config = WorldConfig(seed=seed, size=size, **kwargs)
    world = World(config)
    world.generate()
    return world
