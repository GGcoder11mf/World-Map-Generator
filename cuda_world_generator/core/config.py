"""
World Configuration
====================

Defines all configurable parameters for world generation,
with physically-grounded defaults based on Earth-like planets.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorldConfig:
    """Master configuration for the world generator."""

    # ── Seed & Dimensions ──────────────────────────────────────────────
    seed: int = 42
    size: int = 1024          # Heightmap resolution (size x size)

    # ── Planet Physics ─────────────────────────────────────────────────
    planet_radius_km: float = 6371.0       # Earth-like radius
    gravity: float = 9.81                  # m/s^2
    rotation_period_hours: float = 24.0    # Hours per rotation
    axial_tilt_deg: float = 23.44          # Degrees
    sea_level: float = 0.35                # Fraction of max elevation that is sea level
    ocean_fraction: float = 0.71           # Target ocean coverage (Earth-like)

    # ── Terrain Generation ─────────────────────────────────────────────
    terrain_octaves: int = 8
    terrain_persistence: float = 0.5
    terrain_lacunarity: float = 2.0
    terrain_scale: float = 1.0
    max_elevation_km: float = 8.8          # Max peak elevation (km, Everest-scale)
    continental_shelf_depth: float = 0.2   # Fraction below sea level for shelf

    # ── Tectonic Simulation ────────────────────────────────────────────
    num_plates: int = 12
    tectonic_iterations: int = 50
    plate_speed_factor: float = 1.0
    mountain_height_factor: float = 1.5
    ridge_height_factor: float = 0.8
    subduction_depth_factor: float = 1.2

    # ── Erosion Simulation ─────────────────────────────────────────────
    hydraulic_erosion_iterations: int = 40
    thermal_erosion_iterations: int = 20
    rainfall_rate: float = 0.012
    water_evaporation_rate: float = 0.005
    sediment_capacity_factor: float = 4.0
    sediment_deposition_rate: float = 0.3
    thermal_erosion_rate: float = 0.001
    thermal_talus_angle: float = 0.6       # Radians (~34 degrees)

    # ── Climate ────────────────────────────────────────────────────────
    solar_constant: float = 1361.0         # W/m^2 (Earth-like)
    albedo_land: float = 0.3
    albedo_ocean: float = 0.06
    greenhouse_factor: float = 0.4         # Fraction of outgoing radiation re-absorbed
    lapse_rate: float = 6.5                # Degrees C per km altitude
    base_temperature: float = 15.0         # Global mean surface temp (C)
    humidity_diffusion_rate: float = 0.1
    wind_coriolis_factor: float = 0.5      # Simplified Coriolis scaling

    # ── Hydrology ──────────────────────────────────────────────────────
    river_iterations: int = 100
    river_flow_threshold: float = 0.01
    lake_fill_iterations: int = 50

    # ── Biome ──────────────────────────────────────────────────────────
    biome_temp_tropical: float = 22.0      # C
    biome_temp_temperate: float = 10.0
    biome_temp_cold: float = 0.0
    biome_humid_arid: float = 0.25
    biome_humid_dry: float = 0.45

    # ── Ecosystem ──────────────────────────────────────────────────────
    tree_density_scale: float = 1.0
    animal_density_scale: float = 1.0

    # ── LOD ────────────────────────────────────────────────────────────
    lod_levels: int = 5
    chunk_size: int = 128

    # ── Output ─────────────────────────────────────────────────────────
    output_dir: str = "./world_output"
    export_formats: list = field(default_factory=lambda: ["png", "npz"])

    def validate(self):
        """Validate configuration parameters."""
        assert self.size >= 64, "Size must be at least 64"
        assert self.size & (self.size - 1) == 0, "Size must be power of 2"
        assert self.num_plates >= 2, "Need at least 2 tectonic plates"
        assert 0 < self.sea_level < 1, "Sea level must be in (0, 1)"
        assert 0 < self.ocean_fraction < 1, "Ocean fraction must be in (0, 1)"
        assert self.terrain_octaves >= 1, "Need at least 1 noise octave"
        return True
