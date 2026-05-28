"""
Ecosystem Simulation
=====================

GPU-accelerated procedural ecosystem generation:

- Tree growth probability based on biome, soil, water
- Vegetation density maps (NDVI-like)
- Animal spawn probability maps (abstract habitat suitability)
- Species diversity estimation

Physical constraints:
- Trees can't grow in deserts, ice, or deep water
- Vegetation density correlates with rainfall and temperature
- Animal presence requires food (vegetation) and water
- Biodiversity peaks in tropical rainforests
"""

import numpy as np
from .backend import xp, to_cpu
from .noise import NoiseGenerator


class EcosystemSimulator:
    """
    GPU-accelerated ecosystem generation engine.

    Creates vegetation and wildlife distribution maps based
    on environmental conditions. All computations are vectorized.
    """

    def __init__(self, config):
        self.config = config
        self.noise = NoiseGenerator(config.seed + 600)
        self._rng = np.random.RandomState(config.seed + 600)

    def simulate(self, biome_map, temperature, humidity, rainfall,
                 soil_fertility, heightmap, sea_level, width, height):
        """
        Generate ecosystem maps.

        Parameters
        ----------
        biome_map : array (H, W)
        temperature : array (H, W) - Celsius
        humidity : array (H, W) - [0, 1]
        rainfall : array (H, W) - [0, 1]
        soil_fertility : array (H, W) - [0, 1]
        heightmap : array (H, W)
        sea_level : float
        width, height : int

        Returns
        -------
        vegetation_density : array (H, W) - [0, 1]
        tree_density : array (H, W) - trees per unit area
        animal_probability : array (H, W) - habitat suitability
        biodiversity : array (H, W) - species diversity index
        """
        is_ocean = heightmap < sea_level

        # ── Vegetation Density ──────────────────────────────────────────
        vegetation_density = self._compute_vegetation_density(
            biome_map, temperature, humidity, rainfall,
            soil_fertility, is_ocean, width, height
        )

        # ── Tree Density ────────────────────────────────────────────────
        tree_density = self._compute_tree_density(
            biome_map, temperature, humidity, soil_fertility,
            vegetation_density, is_ocean, width, height
        )

        # ── Animal Probability ──────────────────────────────────────────
        animal_probability = self._compute_animal_probability(
            biome_map, vegetation_density, tree_density,
            temperature, humidity, is_ocean, width, height
        )

        # ── Biodiversity Index ──────────────────────────────────────────
        biodiversity = self._compute_biodiversity(
            biome_map, temperature, humidity, vegetation_density, is_ocean
        )

        return vegetation_density, tree_density, animal_probability, biodiversity

    def _compute_vegetation_density(self, biome_map, temperature, humidity,
                                     rainfall, soil_fertility, is_ocean,
                                     width, height):
        """
        Compute vegetation density (NDVI-like, 0-1 scale).

        Based on:
        - Biome type (fundamental vegetation potential)
        - Water availability (rainfall + humidity)
        - Temperature (growing season length)
        - Soil fertility (nutrient availability)
        - Slope (steep slopes have less vegetation)
        """
        # Base vegetation from biome
        veg = xp.ones_like(biome_map, dtype=xp.float32) * 0.1

        # Dense vegetation biomes
        veg = xp.where(biome_map == 6, 0.95, veg)   # Tropical Rainforest
        veg = xp.where(biome_map == 5, 0.75, veg)   # Tropical Seasonal
        veg = xp.where(biome_map == 10, 0.85, veg)  # Temperate Rainforest
        veg = xp.where(biome_map == 9, 0.70, veg)   # Temperate Deciduous
        veg = xp.where(biome_map == 12, 0.55, veg)  # Boreal Forest
        veg = xp.where(biome_map == 16, 0.40, veg)  # Mountain Forest

        # Moderate vegetation
        veg = xp.where(biome_map == 7, 0.50, veg)   # Savanna
        veg = xp.where(biome_map == 8, 0.45, veg)   # Grassland
        veg = xp.where(biome_map == 11, 0.40, veg)  # Mediterranean
        veg = xp.where(biome_map == 18, 0.65, veg)  # Wetland

        # Low vegetation
        veg = xp.where(biome_map == 13, 0.10, veg)  # Tundra
        veg = xp.where(biome_map == 15, 0.08, veg)  # Alpine Tundra
        veg = xp.where(biome_map == 4, 0.05, veg)   # Subtropical Desert
        veg = xp.where(biome_map == 17, 0.03, veg)  # Cold Desert
        veg = xp.where(biome_map == 3, 0.15, veg)   # Beach
        veg = xp.where(biome_map == 14, 0.01, veg)  # Ice Sheet

        # Ocean: no land vegetation
        veg = xp.where(is_ocean, 0, veg)

        # Modulate by environmental factors
        # Water availability
        water_factor = (humidity * 0.6 + rainfall * 0.4)
        veg *= (0.5 + 0.5 * water_factor)

        # Temperature (growing season)
        # Optimal: 15-30°C, too cold: <5°C, too hot: >40°C
        temp_factor = xp.where(
            temperature < 0, 0.1,
            xp.where(temperature < 5, 0.3,
                     xp.where(temperature > 40, 0.2,
                              xp.where(temperature > 30, 0.8, 1.0)))
        )
        veg *= temp_factor

        # Soil fertility
        veg *= (0.5 + 0.5 * soil_fertility)

        # Add natural noise for patchiness
        veg_noise = self.noise.fbm(width, height, octaves=4, scale=6.0,
                                    offset_x=self._rng.uniform(-100, 100),
                                    offset_y=self._rng.uniform(-100, 100))
        veg *= (0.8 + 0.2 * veg_noise)

        return xp.clip(veg * self.config.tree_density_scale, 0, 1)

    def _compute_tree_density(self, biome_map, temperature, humidity,
                               soil_fertility, vegetation_density,
                               is_ocean, width, height):
        """
        Compute tree density (trees per unit area).

        Trees require:
        - Sufficient water (annual rainfall > 250mm equivalent)
        - Non-freezing temperatures (growing season)
        - Deep enough soil
        - Not too steep slopes
        """
        # Base tree density from biome
        trees = xp.zeros_like(biome_map, dtype=xp.float32)

        # Forest biomes
        trees = xp.where(biome_map == 6, 0.90, trees)   # Tropical Rainforest
        trees = xp.where(biome_map == 5, 0.60, trees)   # Tropical Seasonal
        trees = xp.where(biome_map == 10, 0.80, trees)  # Temperate Rainforest
        trees = xp.where(biome_map == 9, 0.55, trees)   # Temperate Deciduous
        trees = xp.where(biome_map == 12, 0.45, trees)  # Boreal Forest
        trees = xp.where(biome_map == 16, 0.30, trees)  # Mountain Forest

        # Sparse tree biomes
        trees = xp.where(biome_map == 7, 0.15, trees)   # Savanna
        trees = xp.where(biome_map == 11, 0.10, trees)  # Mediterranean
        trees = xp.where(biome_map == 8, 0.05, trees)   # Grassland
        trees = xp.where(biome_map == 18, 0.20, trees)  # Wetland

        # No trees
        trees = xp.where(is_ocean, 0, trees)
        trees = xp.where((biome_map == 4) | (biome_map == 17), 0, trees)  # Deserts
        trees = xp.where((biome_map == 13) | (biome_map == 15), 0, trees)  # Tundra
        trees = xp.where(biome_map == 14, 0, trees)  # Ice

        # Water stress reduction
        trees *= xp.where(humidity > 0.3, 1.0, humidity / 0.3)

        # Temperature constraint (trees need growing season)
        trees *= xp.where(temperature > 5, 1.0, xp.maximum(0, temperature / 5))

        # Soil depth requirement
        trees *= (0.3 + 0.7 * soil_fertility)

        # Modulate by overall vegetation
        trees *= vegetation_density

        # Tree clustering noise (forests have natural clearings)
        tree_noise = self.noise.fbm(width, height, octaves=5, scale=8.0,
                                     offset_x=self._rng.uniform(-100, 100),
                                     offset_y=self._rng.uniform(-100, 100))
        trees *= (0.6 + 0.4 * tree_noise)

        return xp.clip(trees, 0, 1)

    def _compute_animal_probability(self, biome_map, vegetation_density,
                                     tree_density, temperature, humidity,
                                     is_ocean, width, height):
        """
        Compute animal habitat suitability (abstract probability map).

        Animals require:
        - Food (vegetation/other prey)
        - Water (rivers, lakes, or high humidity)
        - Shelter (trees, terrain complexity)
        - Suitable temperature range
        - Not ocean (for terrestrial animals)
        """
        # Ocean marine life (simplified)
        marine_prob = xp.where(is_ocean, 0.3, 0)

        # Terrestrial habitat suitability
        food = vegetation_density * 0.5 + tree_density * 0.3
        water = humidity * 0.7

        # Temperature suitability: most animals prefer 5-35°C
        temp_suit = xp.where(
            temperature < -10, 0.05,
            xp.where(temperature < 5, 0.3,
                     xp.where(temperature > 40, 0.1,
                              xp.where(temperature > 35, 0.5, 1.0)))
        )

        # Shelter from terrain complexity (slope variation)
        shelter = vegetation_density * 0.4 + tree_density * 0.3

        # Combined habitat score
        habitat = (food * 0.35 + water * 0.25 + temp_suit * 0.25 + shelter * 0.15)

        # Biome-specific modifiers
        habitat = xp.where(biome_map == 6, habitat * 1.2, habitat)    # Rainforest: high diversity
        habitat = xp.where(biome_map == 8, habitat * 1.0, habitat)    # Grassland: grazers
        habitat = xp.where(biome_map == 9, habitat * 1.1, habitat)    # Temperate forest
        habitat = xp.where(biome_map == 4, habitat * 0.2, habitat)    # Desert: sparse
        habitat = xp.where(biome_map == 14, habitat * 0.02, habitat)  # Ice: nearly none
        habitat = xp.where(biome_map == 13, habitat * 0.15, habitat)  # Tundra: sparse
        habitat = xp.where(biome_map == 12, habitat * 0.8, habitat)   # Taiga: moderate

        # No terrestrial animals in ocean
        habitat = xp.where(is_ocean, 0, habitat)

        # Add spatial noise for natural distribution patterns
        animal_noise = self.noise.fbm(width, height, octaves=3, scale=7.0,
                                       offset_x=self._rng.uniform(-100, 100),
                                       offset_y=self._rng.uniform(-100, 100))
        habitat *= (0.7 + 0.3 * animal_noise)

        # Combine with marine
        result = xp.where(is_ocean, marine_prob, habitat)

        return xp.clip(result * self.config.animal_density_scale, 0, 1)

    def _compute_biodiversity(self, biome_map, temperature, humidity,
                               vegetation_density, is_ocean):
        """
        Compute biodiversity index (species richness proxy).

        Biodiversity patterns follow real-world observations:
        - Highest in tropical rainforests
        - Decreases with latitude (latitudinal diversity gradient)
        - Higher in heterogeneous habitats
        - Lower in extreme environments (deserts, ice)
        - Ocean biodiversity peaks in coral reef areas
        """
        # Base biodiversity from biome
        bio = xp.ones_like(biome_map, dtype=xp.float32) * 0.2

        bio = xp.where(biome_map == 6, 1.0, bio)    # Tropical Rainforest: highest
        bio = xp.where(biome_map == 5, 0.8, bio)    # Tropical Seasonal
        bio = xp.where(biome_map == 7, 0.7, bio)    # Savanna
        bio = xp.where(biome_map == 10, 0.75, bio)  # Temperate Rainforest
        bio = xp.where(biome_map == 9, 0.6, bio)    # Temperate Deciduous
        bio = xp.where(biome_map == 8, 0.5, bio)    # Grassland
        bio = xp.where(biome_map == 12, 0.45, bio)  # Boreal Forest
        bio = xp.where(biome_map == 11, 0.55, bio)  # Mediterranean
        bio = xp.where(biome_map == 18, 0.65, bio)  # Wetland
        bio = xp.where(biome_map == 16, 0.35, bio)  # Mountain Forest
        bio = xp.where(biome_map == 13, 0.15, bio)  # Tundra
        bio = xp.where(biome_map == 15, 0.10, bio)  # Alpine Tundra
        bio = xp.where(biome_map == 4, 0.10, bio)   # Desert
        bio = xp.where(biome_map == 17, 0.05, bio)  # Cold Desert
        bio = xp.where(biome_map == 14, 0.02, bio)  # Ice Sheet

        # Ocean biodiversity: moderate, with hotspots
        bio = xp.where(is_ocean & (temperature > 15) & (temperature < 30), 0.5, bio)
        bio = xp.where(is_ocean & (temperature > 25), 0.7, bio)  # Coral reef areas

        # Habitat heterogeneity increases biodiversity
        bio *= (0.5 + 0.5 * vegetation_density)

        return xp.clip(bio, 0, 1)
