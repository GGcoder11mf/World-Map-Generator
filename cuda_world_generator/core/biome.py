"""
Biome Classification
=====================

GPU-accelerated biome classification based on the Whittaker
biome system and Koppen climate classification.

Biomes are determined by the intersection of:
- Temperature (mean annual, seasonal range)
- Precipitation (annual total, seasonality)
- Altitude (altitudinal zonation)
- Latitude (day length, solar angle)
- Soil type (simplified)
- Distance from coast

Classification follows physically-grounded rules that mirror
real-world biome distribution patterns.
"""

import numpy as np
from .backend import xp, to_cpu


# Biome ID definitions (consistent with visualization coloring)
BIOMES = {
    0:  "Deep Ocean",
    1:  "Ocean",
    2:  "Shallow Ocean",
    3:  "Beach",
    4:  "Subtropical Desert",
    5:  "Tropical Seasonal Forest",
    6:  "Tropical Rainforest",
    7:  "Savanna",
    8:  "Grassland",
    9:  "Temperate Deciduous Forest",
    10: "Temperate Rainforest",
    11: "Mediterranean Scrub",
    12: "Boreal Forest (Taiga)",
    13: "Tundra",
    14: "Ice Sheet",
    15: "Alpine Tundra",
    16: "Mountain Forest",
    17: "Cold Desert",
    18: "Wetland",
}

NUM_BIOMES = len(BIOMES)

# RGB colors for each biome
BIOME_COLORS = {
    0:  (10, 30, 80),       # Deep Ocean
    1:  (20, 50, 120),      # Ocean
    2:  (30, 80, 160),      # Shallow Ocean
    3:  (210, 200, 160),    # Beach
    4:  (200, 180, 100),    # Subtropical Desert
    5:  (80, 160, 50),      # Tropical Seasonal Forest
    6:  (30, 130, 30),      # Tropical Rainforest
    7:  (160, 180, 60),     # Savanna
    8:  (140, 180, 80),     # Grassland
    9:  (60, 140, 40),      # Temperate Deciduous Forest
    10: (20, 100, 50),      # Temperate Rainforest
    11: (150, 170, 80),     # Mediterranean Scrub
    12: (40, 80, 50),       # Boreal Forest (Taiga)
    13: (170, 190, 200),    # Tundra
    14: (230, 240, 250),    # Ice Sheet
    15: (150, 170, 180),    # Alpine Tundra
    16: (50, 100, 60),      # Mountain Forest
    17: (150, 140, 120),    # Cold Desert
    18: (60, 130, 130),     # Wetland
}


class BiomeClassifier:
    """
    GPU-accelerated biome classification engine.

    Classifies each cell into a biome based on temperature,
    precipitation, altitude, and other environmental factors.
    Uses vectorized conditional operations for GPU parallelism.
    """

    def __init__(self, config):
        self.config = config
        self._rng = np.random.RandomState(config.seed + 400)

    def classify(self, heightmap, temperature, humidity, rainfall,
                 sea_level, width, height):
        """
        Classify biomes based on environmental conditions.

        Parameters
        ----------
        heightmap : array (H, W)
        temperature : array (H, W) - Celsius
        humidity : array (H, W) - [0, 1]
        rainfall : array (H, W) - [0, 1]
        sea_level : float
        width, height : int

        Returns
        -------
        biome_map : array (H, W) - biome ID per cell
        soil_fertility : array (H, W) - soil quality [0, 1]
        """
        config = self.config
        is_ocean = heightmap < sea_level
        elevation = (heightmap - sea_level).clip(0)  # Above sea level only

        # Elevation in km
        elev_km = elevation * config.max_elevation_km

        # ── Ocean biomes ───────────────────────────────────────────────
        # Deep ocean: far from coast, Shallow: near coast
        ocean_depth = (sea_level - heightmap).clip(0)
        biome_map = xp.where(
            is_ocean,
            xp.where(ocean_depth > 0.2, 0,      # Deep Ocean
                     xp.where(ocean_depth > 0.05, 1,  # Ocean
                              2)),                    # Shallow Ocean
            0  # Placeholder for land (will be classified below)
        )

        # ── Beach ──────────────────────────────────────────────────────
        # Low elevation, near sea level
        beach_mask = (~is_ocean) & (elev_km < 0.05)
        biome_map = xp.where(beach_mask, 3, biome_map)

        # ── Land classification ────────────────────────────────────────
        land_mask = (~is_ocean) & (~beach_mask)

        # Temperature thresholds
        T_tropical = config.biome_temp_tropical
        T_temperate = config.biome_temp_temperate
        T_cold = config.biome_temp_cold

        # Humidity thresholds
        H_arid = config.biome_humid_arid
        H_dry = config.biome_humid_dry

        # ── Tropical biomes (temperature > T_tropical) ─────────────────
        tropical = land_mask & (temperature > T_tropical)
        biome_map = xp.where(
            tropical,
            xp.where(
                humidity > H_dry, 6,       # Tropical Rainforest
                xp.where(
                    humidity > H_arid, 5,   # Tropical Seasonal Forest
                    xp.where(
                        humidity > 0.15, 7,  # Savanna
                        4                   # Subtropical Desert
                    )
                )
            ),
            biome_map
        )

        # ── Temperate biomes (T_cold < temperature < T_tropical) ──────
        temperate = land_mask & (temperature <= T_tropical) & (temperature > T_temperate)
        biome_map = xp.where(
            temperate,
            xp.where(
                humidity > H_dry, 10,      # Temperate Rainforest
                xp.where(
                    humidity > 0.35, 9,     # Temperate Deciduous Forest
                    xp.where(
                        humidity > H_arid, 8,  # Grassland
                        xp.where(
                            humidity > 0.15, 11,  # Mediterranean Scrub
                            4                   # Subtropical Desert
                        )
                    )
                )
            ),
            biome_map
        )

        # ── Cold biomes (T_cold < temperature < T_temperate) ──────────
        cold = land_mask & (temperature <= T_temperate) & (temperature > T_cold)
        biome_map = xp.where(
            cold,
            xp.where(
                humidity > 0.3, 12,         # Boreal Forest (Taiga)
                xp.where(
                    humidity > 0.15, 8,      # Grassland
                    17                       # Cold Desert
                )
            ),
            biome_map
        )

        # ── Polar biomes (temperature < T_cold) ────────────────────────
        polar = land_mask & (temperature <= T_cold)
        biome_map = xp.where(
            polar,
            xp.where(
                temperature < -15, 14,      # Ice Sheet
                13,                          # Tundra
            ),
            biome_map
        )

        # ── Altitudinal zonation ───────────────────────────────────────
        # High elevation overrides: Alpine tundra, Mountain forest
        alpine = land_mask & (elev_km > 3.5)
        biome_map = xp.where(alpine, 15, biome_map)  # Alpine Tundra

        mountain_forest = land_mask & (elev_km > 2.0) & (elev_km <= 3.5) & (temperature > T_cold)
        biome_map = xp.where(mountain_forest, 16, biome_map)  # Mountain Forest

        # ── Ice: very cold at high altitude ─────────────────────────────
        ice = land_mask & (temperature < -10) & (elev_km > 2.0)
        biome_map = xp.where(ice, 14, biome_map)

        # ── Wetland: flat, low elevation, high humidity ─────────────────
        wetland = land_mask & (elev_km < 0.2) & (humidity > 0.7) & (rainfall > 0.5)
        biome_map = xp.where(wetland, 18, biome_map)

        # ── Soil fertility ──────────────────────────────────────────────
        soil_fertility = self._compute_soil_fertility(
            biome_map, temperature, rainfall, humidity, elevation
        )

        return biome_map, soil_fertility

    def _compute_soil_fertility(self, biome_map, temperature, rainfall, humidity, elevation):
        """
        Compute soil fertility based on biome and environmental factors.

        Soil fertility is influenced by:
        - Vegetation cover (more biomass = more organic matter)
        - Temperature (moderate temps favor decomposition)
        - Rainfall (moderate rainfall favors leaching balance)
        - Slope (steep slopes have thin soils)
        - Parent material (simplified)
        """
        # Base fertility from biome
        fertility = xp.ones_like(biome_map, dtype=xp.float32) * 0.3

        # Tropical rainforest: high biomass but nutrients locked in vegetation
        fertility = xp.where(biome_map == 6, 0.4, fertility)
        # Temperate forests: best agricultural soils
        fertility = xp.where(biome_map == 9, 0.8, fertility)
        fertility = xp.where(biome_map == 10, 0.7, fertility)
        # Grasslands: good soils (Mollisols)
        fertility = xp.where(biome_map == 8, 0.75, fertility)
        # Savanna: moderate
        fertility = xp.where(biome_map == 7, 0.5, fertility)
        # Deserts: very poor
        fertility = xp.where((biome_map == 4) | (biome_map == 17), 0.1, fertility)
        # Boreal: moderate, acidic
        fertility = xp.where(biome_map == 12, 0.4, fertility)
        # Tundra: poor, permafrost
        fertility = xp.where(biome_map == 13, 0.15, fertility)
        # Wetland: organic-rich but waterlogged
        fertility = xp.where(biome_map == 18, 0.6, fertility)
        # Mediterranean: moderate
        fertility = xp.where(biome_map == 11, 0.55, fertility)
        # Mountain: thin soils
        fertility = xp.where((biome_map == 15) | (biome_map == 16), 0.25, fertility)

        # Temperature modifier: moderate temps (10-25°C) are best
        temp_factor = 1.0 - xp.abs(temperature - 17) / 40.0
        fertility *= xp.clip(temp_factor, 0.3, 1.0)

        # Rainfall modifier: moderate rainfall (0.3-0.6) is best for soil
        rain_factor = 1.0 - xp.abs(rainfall - 0.4) * 2
        fertility *= xp.clip(rain_factor, 0.5, 1.0)

        return xp.clip(fertility, 0, 1)

    @staticmethod
    def get_biome_name(biome_id):
        """Get human-readable biome name from ID."""
        return BIOMES.get(int(biome_id), "Unknown")

    @staticmethod
    def get_biome_color(biome_id):
        """Get RGB color tuple for a biome."""
        return BIOME_COLORS.get(int(biome_id), (128, 128, 128))

    @staticmethod
    def create_biome_colormap():
        """
        Create a matplotlib-compatible colormap for biome visualization.
        """
        colors = [BIOME_COLORS[i] for i in range(NUM_BIOMES)]
        return colors
