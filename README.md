# CUDA Physically Accurate Procedural World Generator

A GPU-accelerated procedural world generation engine that creates Earth-like planets with realistic terrain, tectonics, climate, hydrology, biomes, and ecosystems — powered by CUDA via CuPy with automatic NumPy CPU fallback.

---

## Features

- **GPU-Accelerated Pipeline** — All simulation stages use CuPy for CUDA GPU parallelism with seamless NumPy fallback, so the code runs on any machine
- **Physically-Grounded Simulation** — Tectonic plates, hydraulic/thermal erosion, orographic precipitation, rain shadows, and adiabatic lapse rates are modeled after real Earth physics
- **Deterministic Generation** — Every world is reproducible from a single seed, enabling consistent exploration and iteration
- **19 Distinct Biomes** — Whittaker/Koppen-style classification produces Tropical Rainforests, Savannas, Boreal Forests, Tundra, Deserts, Wetlands, and more
- **River Networks & Lakes** — D8 flow-direction hydrology simulation carves realistic river systems and fills enclosed basins
- **Orographic Rain Shadows** — Windward precipitation and leeward drying follow the Clausius-Clapeyron equation and adiabatic lapse physics
- **Ecosystem Maps** — Vegetation density, tree density, animal habitat suitability, and biodiversity index per cell
- **LOD Chunking System** — Quadtree-based level-of-detail management for planet-scale worlds with on-demand chunk generation
- **Rich Visualization** — Multi-panel matplotlib previews, 3D terrain rendering, Tkinter GUI with zoom/pan and 13 switchable map layers
- **Multiple Export Formats** — PNG images, compressed NPZ archives, Wavefront OBJ meshes, and JSON metadata

---

## Gallery

Below are the output maps generated from a single seed. Each image represents one data layer from the 7-stage pipeline.

### Terrain & Geology

| Heightmap | Biome Map |
|:---------:|:---------:|
| ![Heightmap](.Outputs/heightmap.png) | ![Biome Map](.Outputs/biome_map.png) |

### Climate & Atmosphere

| Temperature | Humidity | Rainfall |
|:-----------:|:--------:|:--------:|
| ![Temperature](.Outputs/temperature_c.png) | ![Humidity](.Outputs/humidity.png) | ![Rainfall](.Outputs/rainfall.png) |

### Hydrology & Ecosystems

| Rivers | Vegetation | Animals | Biodiversity |
|:------:|:----------:|:-------:|:------------:|
| ![Rivers](.Outputs/rivers.png) | ![Vegetation](.Outputs/vegetation.png) | ![Animals](.Outputs/animals.png) | ![Biodiversity](.Outputs/biodiversity.png) |

---

## Architecture

```
cuda_world_generator/
├── __init__.py              # Package entry point (World, WorldConfig)
├── world.py                 # Main API — generation pipeline & data access
├── gui.py                   # Tkinter GUI with map viewer & controls
├── visualization.py         # Matplotlib multi-panel & 3D preview
├── export.py                # PNG, NPZ, OBJ, JSON exporters
├── requirements.txt         # Core & optional dependencies
├── core/
│   ├── backend.py           # CuPy/NumPy dual backend abstraction
│   ├── config.py            # WorldConfig dataclass (40+ parameters)
│   ├── noise.py             # GPU-parallel Perlin/fBm/ridged/domain-warp noise
│   ├── tectonics.py         # Voronoi plate generation & boundary simulation
│   ├── erosion.py           # Hydraulic + thermal erosion (padded, no wrapping)
│   ├── climate.py           # Temperature, wind, humidity, rain shadow
│   ├── biome.py             # 19-biome classifier & soil fertility
│   ├── hydrology.py         # D8 flow accumulation, rivers, lakes
│   ├── ecosystem.py         # Vegetation, trees, animals, biodiversity
│   └── lod.py               # Quadtree LOD chunk manager
└── examples/
    └── demo.py              # CLI demo with argparse options
```

---

## Generation Pipeline

The world generation follows a 7-stage physically-based pipeline:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `noise` | Base terrain from multi-octave fBm + domain warping |
| 2 | `tectonics` | Voronoi plate tessellation, convergent/divergent/transform boundaries |
| 3 | `erosion` | Hydraulic erosion (rainfall → flow → sediment) + thermal erosion (talus angle) |
| 4 | `climate` | Latitude temperature, lapse rate, Hadley/Ferrel/Polar wind cells, ocean currents |
| 5 | `hydrology` | D8 flow direction, flow accumulation, river extraction, lake filling |
| 6 | `biome` | Temperature × humidity classification into 19 biomes + soil fertility |
| 7 | `ecosystem` | Vegetation density, tree density, animal probability, biodiversity index |

---

## Installation

### Prerequisites

- Python 3.8+
- NumPy >= 1.24, SciPy >= 1.10, Matplotlib >= 3.7

### Install Core Dependencies

```bash
pip install numpy>=1.24.0 scipy>=1.10.0 matplotlib>=3.7.0
```

### GPU Acceleration (Optional)

For CUDA GPU acceleration, install the CuPy package matching your CUDA version:

```bash
# CUDA 12.x
pip install cupy-cuda12x

# CUDA 11.x
pip install cupy-cuda11x
```

CuPy is auto-detected at runtime. If not installed, the engine transparently falls back to NumPy (CPU mode) with no code changes required.

### Optional Visualization & Export

```bash
pip install Pillow>=9.0.0      # GUI image processing
pip install vispy>=0.12.0      # OpenGL-accelerated visualization
pip install open3d>=0.17.0     # 3D mesh export and rendering
```

---

## Quick Start

### Python API

```python
from cuda_world_generator import World, WorldConfig

# Create a world with default Earth-like parameters
config = WorldConfig(seed=42, size=1024)
world = World(config)

# Run the full generation pipeline
world.generate()

# Access generated data
heightmap = world.get_heightmap()        # NumPy array (CPU)
biome_map = world.get_biome_map()
all_maps = world.get_all_maps()          # Dict of all 17+ maps

# Export data
world.export_terrain("output/")

# Preview in matplotlib
from cuda_world_generator.visualization import preview_world
preview_world(world, output_path="preview.png")
```

### Custom Configuration

```python
config = WorldConfig(
    seed=12345,
    size=512,
    sea_level=0.35,               # Ocean coverage fraction
    num_plates=12,                 # Tectonic plate count
    hydraulic_erosion_iterations=40,
    thermal_erosion_iterations=20,
    max_elevation_km=8.8,         # Everest-scale peaks
    solar_constant=1361.0,        # W/m² (Earth-like)
    axial_tilt_deg=23.44,         # Seasons
    lapse_rate=6.5,               # °C/km altitude cooling
)
```

### CLI Demo

```bash
# Default world (seed=42, 512×512)
python examples/demo.py

# Custom seed and resolution
python examples/demo.py --seed 12345 --size 1024

# Quick mode (reduced iterations for fast preview)
python examples/demo.py --quick

# Include 3D terrain rendering
python examples/demo.py --3d

# Export as OBJ mesh
python examples/demo.py --obj --output ./my_world
```

### GUI

```bash
python gui.py
```

The Tkinter GUI provides:
- Seed, size, erosion, plates, and sea level controls
- One-click generation with progress indication
- 13 switchable map layers (heightmap, biome, temperature, humidity, rainfall, rain shadow, rivers, vegetation, trees, animals, biodiversity, soil fertility, flow accumulation)
- Zoom (scroll), pan (drag), and reset (double-click)
- World statistics panel
- Export all maps as PNG + NPZ

---

## Simulation Details

### Tectonic Plates

Plates are generated via Voronoi tessellation on a sphere with clustered continental seeds. Each plate has physical properties (density, thickness, velocity) and is classified as oceanic or continental. Boundaries are detected and classified as convergent (mountains/subduction), divergent (ridges/rifts), or transform (faults). Isostatic adjustment raises thicker continental crust and sinks thinner oceanic crust.

### Erosion

**Hydraulic erosion** simulates rainfall deposition, downhill water flow, sediment pickup based on slope and flow speed, sediment deposition when capacity is exceeded, and water evaporation. All neighbor access uses padded arrays — no `xp.roll()` wrapping artifacts.

**Thermal erosion** transfers material downhill when slopes exceed the talus angle (~34°), simulating freeze-thaw weathering and rockfall.

**Coastal erosion** wears away land cells adjacent to ocean, creating natural coastlines with bays and peninsulas.

### Climate

Temperature is computed from latitude (solar insolation), altitude (lapse rate cooling), continentality (interior extremes), and ocean proximity (maritime moderation). Wind follows the three-cell atmospheric circulation model (Hadley, Ferrel, Polar) with Coriolis deflection and topographic steering. The **orographic precipitation** model uses the Clausius-Clapeyron equation to determine saturation vapor pressure, computes vertical velocity from wind-terrain interaction, and precipitates excess moisture on windward slopes. **Rain shadows** form on leeward slopes where adiabatic warming increases moisture capacity beyond available moisture.

### Hydrology

D8 flow direction assigns each cell to its steepest downhill neighbor. Flow accumulation iteratively propagates rainfall downstream. Rivers are extracted where accumulation exceeds a threshold, with morphological filtering to remove isolated pixels. Lakes form in enclosed basins (pits with no outflow) and fill iteratively until overflow.

### Biomes

19 biomes classified by temperature × humidity intersection, following Whittaker/Koppen principles. Altitudinal zonation overrides lowland classification above 2 km (mountain forest) and 3.5 km (alpine tundra). Soil fertility is derived from biome type, temperature, and rainfall.

### Ecosystems

Vegetation density integrates biome potential, water availability, growing-season temperature, and soil fertility. Tree density further constrains by water stress, temperature thresholds, and soil depth. Animal habitat suitability combines food (vegetation), water, shelter, and temperature factors. Biodiversity follows the latitudinal diversity gradient — highest in tropical rainforests, lowest in ice sheets.

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | 42 | Random seed for deterministic generation |
| `size` | 1024 | Heightmap resolution (size × size, must be power of 2) |
| `planet_radius_km` | 6371.0 | Planet radius in km |
| `gravity` | 9.81 | Surface gravity (m/s²) |
| `rotation_period_hours` | 24.0 | Rotation period |
| `axial_tilt_deg` | 23.44 | Axial tilt (controls seasons) |
| `sea_level` | 0.35 | Sea level as fraction of max elevation |
| `ocean_fraction` | 0.71 | Target ocean coverage |
| `terrain_octaves` | 8 | Noise layers for terrain |
| `terrain_persistence` | 0.5 | Amplitude decay per octave |
| `terrain_lacunarity` | 2.0 | Frequency growth per octave |
| `max_elevation_km` | 8.8 | Maximum peak elevation |
| `num_plates` | 12 | Number of tectonic plates |
| `tectonic_iterations` | 50 | Plate simulation iterations |
| `hydraulic_erosion_iterations` | 40 | Hydraulic erosion passes |
| `thermal_erosion_iterations` | 20 | Thermal erosion passes |
| `rainfall_rate` | 0.012 | Base rainfall per iteration |
| `lapse_rate` | 6.5 | Temperature decrease per km altitude (°C/km) |
| `solar_constant` | 1361.0 | Solar irradiance (W/m²) |
| `greenhouse_factor` | 0.4 | Greenhouse warming fraction |
| `river_iterations` | 100 | Flow propagation iterations |
| `river_flow_threshold` | 0.01 | Minimum flow for river presence |
| `lake_fill_iterations` | 50 | Lake basin filling passes |
| `lod_levels` | 5 | Quadtree LOD depth |
| `chunk_size` | 128 | Chunk resolution at LOD 0 |

---

## Output Maps

The generator produces 17+ data layers, all available as NumPy arrays:

| Map | Range | Description |
|-----|-------|-------------|
| `heightmap` | [0, 1] | Terrain elevation |
| `plate_map` | int | Tectonic plate ID per cell |
| `boundary_map` | [0, 1] | Plate boundary strength |
| `temperature` | °C | Surface temperature |
| `humidity` | [0, 1] | Relative humidity |
| `rainfall` | [0, 1] | Annual precipitation |
| `rain_shadow` | [0, 1] | Rain shadow intensity (0=wet, 1=dry) |
| `wind_u` | m/s | East-west wind component |
| `wind_v` | m/s | North-south wind component |
| `ocean_temp` | °C | Sea surface temperature |
| `biome_map` | 0–18 | Biome classification ID |
| `soil_fertility` | [0, 1] | Soil quality index |
| `river_map` | [0, 1] | River flow magnitude |
| `flow_accumulation` | [0, 1] | Accumulated water flow |
| `lake_map` | [0, +] | Lake depth above terrain |
| `vegetation_density` | [0, 1] | NDVI-like vegetation cover |
| `tree_density` | [0, 1] | Tree density per unit area |
| `animal_probability` | [0, 1] | Habitat suitability |
| `biodiversity` | [0, 1] | Species diversity index |
| `sediment_map` | [0, +] | Deposited sediment thickness |
| `water_map` | [0, +] | Water flow accumulation |

---

## Biome Reference

| ID | Biome | Key Conditions |
|----|-------|---------------|
| 0 | Deep Ocean | > 0.2 below sea level |
| 1 | Ocean | 0.05–0.2 below sea level |
| 2 | Shallow Ocean | < 0.05 below sea level |
| 3 | Beach | < 0.05 km above sea level |
| 4 | Subtropical Desert | Hot, very dry |
| 5 | Tropical Seasonal Forest | Hot, moderate moisture |
| 6 | Tropical Rainforest | Hot, humid |
| 7 | Savanna | Hot, low moisture |
| 8 | Grassland | Temperate, dry-moderate |
| 9 | Temperate Deciduous Forest | Temperate, moderate-humid |
| 10 | Temperate Rainforest | Temperate, very humid |
| 11 | Mediterranean Scrub | Temperate, dry |
| 12 | Boreal Forest (Taiga) | Cold, moderate moisture |
| 13 | Tundra | Very cold |
| 14 | Ice Sheet | Extremely cold, high altitude |
| 15 | Alpine Tundra | > 3.5 km elevation |
| 16 | Mountain Forest | 2–3.5 km elevation |
| 17 | Cold Desert | Cold, very dry |
| 18 | Wetland | Low elevation, very humid, high rainfall |

---

## Performance

Performance depends heavily on resolution, erosion iterations, and GPU availability:

| Resolution | GPU (CuPy) | CPU (NumPy) |
|-----------|------------|-------------|
| 256 × 256 | ~3s | ~8s |
| 512 × 512 | ~8s | ~30s |
| 1024 × 1024 | ~25s | ~120s |

*Benchmarks with default iterations on a modern system. Actual times vary by hardware and configuration.*

---

## Time-Stepping Simulation

After initial generation, worlds can be advanced in time:

```python
world.generate()

# Advance by 1 million years
world.simulate_step(delta_time=1.0)

# Advance again
world.simulate_step(delta_time=0.5)
```

Each step simulates:
- Continental drift (plate motion updates)
- Continued erosion (reduced iterations)
- Climate re-simulation
- Hydrology re-computation
- Biome and ecosystem re-classification

---

## License

This project is provided as-is for educational and creative purposes. See the source code for individual module documentation and license information.
