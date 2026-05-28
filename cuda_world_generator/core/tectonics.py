"""
Tectonic Plate Simulation
==========================

Simulates tectonic plate dynamics including:
- Plate generation via Voronoi tessellation
- Continental drift (plate motion vectors)
- Convergent boundary mountain formation
- Divergent boundary rift/ridge formation
- Transform boundary fault lines
- Subduction zones and ocean trenches
- Isostatic adjustment (buoyancy equilibrium)

All computations use GPU-parallel array operations.
"""

import numpy as np
from .backend import xp, to_cpu, to_gpu, gpu_available
from .noise import NoiseGenerator


class TectonicPlate:
    """Represents a single tectonic plate with physical properties."""

    def __init__(self, plate_id, center, velocity, is_oceanic, density, thickness):
        self.plate_id = plate_id
        self.center = center          # (lon, lat) in radians
        self.velocity = velocity      # (vx, vy) movement vector
        self.is_oceanic = is_oceanic  # Oceanic vs continental
        self.density = density        # g/cm^3 (oceanic ~3.0, continental ~2.7)
        self.thickness = thickness    # km (oceanic ~7, continental ~35)
        self.angular_velocity = 0.0   # Rotation rate around Euler pole
        self.euler_pole = (0, 0)     # Euler pole position


class TectonicSimulator:
    """
    GPU-accelerated tectonic plate simulation.

    Generates realistic plate boundaries and their associated
    geological features using physical principles of plate tectonics.

    The simulation creates:
    - Continent/ocean distribution via clustered Voronoi cells
    - Mountain ranges at convergent boundaries
    - Mid-ocean ridges at divergent boundaries
    - Trenches at subduction zones
    - Strike-slip features at transform boundaries
    """

    def __init__(self, config):
        self.config = config
        self.noise = NoiseGenerator(config.seed)
        self.plates = []
        self.plate_map = None
        self.boundary_map = None
        self.boundary_type_map = None
        self._rng = np.random.RandomState(config.seed + 100)

    def generate_plates(self, width, height):
        """
        Generate tectonic plates using Voronoi tessellation with
        clustered continent seeds for realistic land distribution.

        Parameters
        ----------
        width, height : int
            Map dimensions.

        Returns
        -------
        plate_map : array (height, width) - plate ID per cell
        """
        config = self.config
        num_plates = config.num_plates

        # Generate plate seed points with clustering for continents
        # Use 2-4 continental clusters, each with multiple plates
        num_continental_clusters = self._rng.randint(2, 5)
        cluster_centers_lon = self._rng.uniform(0, 2 * np.pi, num_continental_clusters)
        cluster_centers_lat = self._rng.uniform(-np.pi / 4, np.pi / 4, num_continental_clusters)

        plate_seeds_lon = []
        plate_seeds_lat = []
        plate_types = []  # True = continental

        # Distribute plates among continental clusters
        plates_per_cluster = max(1, num_plates // (num_continental_clusters + 1))
        remaining = num_plates

        for i in range(num_continental_clusters):
            n = min(plates_per_cluster, remaining)
            for j in range(n):
                lon = cluster_centers_lon[i] + self._rng.normal(0, 0.3)
                lat = cluster_centers_lat[i] + self._rng.normal(0, 0.2)
                plate_seeds_lon.append(lon % (2 * np.pi))
                plate_seeds_lat.append(lat)
                plate_types.append(True)  # Continental
                remaining -= 1

        # Remaining plates are oceanic
        for _ in range(remaining):
            lon = self._rng.uniform(0, 2 * np.pi)
            lat = self._rng.uniform(-np.pi / 2, np.pi / 2)
            plate_seeds_lon.append(lon)
            plate_seeds_lat.append(lat)
            plate_types.append(False)  # Oceanic

        num_plates = len(plate_seeds_lon)

        # Create plate objects with velocities
        self.plates = []
        for i in range(num_plates):
            center = (plate_seeds_lon[i], plate_seeds_lat[i])
            # Plate velocity: continental plates move slower
            speed = self._rng.uniform(1, 5) if plate_types[i] else self._rng.uniform(3, 8)
            angle = self._rng.uniform(0, 2 * np.pi)
            velocity = (speed * np.cos(angle) * config.plate_speed_factor,
                       speed * np.sin(angle) * config.plate_speed_factor)

            is_oceanic = not plate_types[i]
            density = 3.0 if is_oceanic else 2.7
            thickness = 7.0 if is_oceanic else 35.0

            plate = TectonicPlate(i, center, velocity, is_oceanic, density, thickness)
            # Assign angular velocity for Euler pole rotation
            plate.angular_velocity = self._rng.uniform(-2, 2) * 1e-7
            plate.euler_pole = (self._rng.uniform(0, 2 * np.pi),
                               self._rng.uniform(-np.pi / 2, np.pi / 2))
            self.plates.append(plate)

        # Build plate map via Voronoi tessellation on sphere
        self._build_plate_map(width, height)

        return self.plate_map

    def _build_plate_map(self, width, height):
        """
        Assign each cell to its nearest plate using spherical distance.
        GPU-parallel computation of all distances simultaneously.
        """
        num_plates = len(self.plates)

        # Get spherical coordinates for each pixel
        lon = xp.linspace(0, 2 * xp.pi, width, dtype=xp.float32)
        lat = xp.linspace(-xp.pi / 2, xp.pi / 2, height, dtype=xp.float32)
        LON, LAT = xp.meshgrid(lon, lat)

        # Plate centers on GPU
        centers_lon = xp.array([p.center[0] for p in self.plates], dtype=xp.float32)
        centers_lat = xp.array([p.center[1] for p in self.plates], dtype=xp.float32)

        # Compute geodesic distance from each pixel to each plate center
        # Using Haversine-like formula (simplified for speed)
        min_dist = xp.full((height, width), 1e10, dtype=xp.float32)
        plate_map = xp.zeros((height, width), dtype=xp.int32)

        for i in range(num_plates):
            # Angular distance approximation
            dlat = LAT - centers_lat[i]
            dlon = LON - centers_lon[i]
            # Great circle distance (simplified)
            dist = xp.sqrt(dlat * dlat + (dlon * xp.cos((LAT + centers_lat[i]) / 2)) ** 2)

            # Add some noise to plate boundaries for natural shapes
            boundary_noise = self.noise.perlin_2d(width, height, 3.0, 
                                                   centers_lon[i] * 10, 
                                                   centers_lat[i] * 10)
            dist = dist + boundary_noise * 0.1

            mask = dist < min_dist
            plate_map = xp.where(mask, i, plate_map)
            min_dist = xp.where(mask, dist, min_dist)

        self.plate_map = plate_map

    def compute_boundaries(self, width, height):
        """
        Detect plate boundaries and classify their type.

        Boundary types:
        0 = interior (not a boundary)
        1 = convergent (mountains / subduction)
        2 = divergent (ridges / rifts)
        3 = transform (faults)

        Returns
        -------
        boundary_map : array (height, width) - boundary strength
        boundary_type_map : array (height, width) - boundary type
        """
        plate_map = self.plate_map
        boundary_map = xp.zeros((height, width), dtype=xp.float32)
        boundary_type_map = xp.zeros((height, width), dtype=xp.int32)

        # Check neighbors for plate changes (boundary detection)
        # Use padded shifts — no edge wrapping artifacts
        padded_pm = xp.pad(plate_map, 1, mode='edge')
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Shifted plate map via padded indexing (no wrapping)
            oy, ox = 1 + dy, 1 + dx
            shifted = padded_pm[oy:oy + height, ox:ox + width]

            # Find boundary cells (where plate ID changes)
            boundary_mask = plate_map != shifted
            boundary_strength = boundary_mask.astype(xp.float32)

            # Determine boundary type from relative plate velocities
            # For each boundary cell, get the two plate IDs
            plate_a = plate_map
            plate_b = shifted

            # Compute relative velocity (convergent vs divergent)
            rel_vel = self._compute_relative_velocity(plate_a, plate_b, width, height)

            # Classify: positive = convergent, negative = divergent, near-zero = transform
            btype = xp.where(
                rel_vel > 0.1, 1,  # Convergent
                xp.where(rel_vel < -0.1, 2, 3)  # Divergent or Transform
            )

            boundary_map = xp.maximum(boundary_map, boundary_strength)
            boundary_type_map = xp.where(boundary_mask, btype, boundary_type_map)

        # Smooth boundary map
        boundary_map = self._gaussian_blur(boundary_map, 3)

        self.boundary_map = boundary_map
        self.boundary_type_map = boundary_type_map

        return boundary_map, boundary_type_map

    def _compute_relative_velocity(self, plate_a_map, plate_b_map, width, height):
        """
        Compute relative velocity between adjacent plates.
        Positive = convergent, Negative = divergent.

        Uses the velocity vectors of each plate to determine
        whether neighboring plates are moving toward or away from each other.
        """
        # Build velocity fields for each plate
        vx_field = xp.zeros((height, width), dtype=xp.float32)
        vy_field = xp.zeros((height, width), dtype=xp.float32)

        for plate in self.plates:
            mask = plate_a_map == plate.plate_id
            vx_field = xp.where(mask, plate.velocity[0], vx_field)
            vy_field = xp.where(mask, plate.velocity[1], vy_field)

        vx_field_b = xp.zeros((height, width), dtype=xp.float32)
        vy_field_b = xp.zeros((height, width), dtype=xp.float32)

        for plate in self.plates:
            mask = plate_b_map == plate.plate_id
            vx_field_b = xp.where(mask, plate.velocity[0], vx_field_b)
            vy_field_b = xp.where(mask, plate.velocity[1], vy_field_b)

        # Relative velocity dot product with boundary normal
        # Simplified: use magnitude of relative velocity with sign
        rel_vx = vx_field - vx_field_b
        rel_vy = vy_field - vy_field_b

        # Convergence = negative dot product (plates moving toward each other)
        # We approximate boundary normal as the gradient of the plate map
        rel_vel = -(rel_vx + rel_vy)  # Simplified convergence measure

        return rel_vel

    def apply_tectonics(self, heightmap, width, height):
        """
        Modify the heightmap based on tectonic plate interactions.

        - Convergent boundaries: raise elevation (mountains / subduction)
        - Divergent boundaries: lower elevation (rifts / ridges)
        - Transform boundaries: slight elevation changes (faults)

        Parameters
        ----------
        heightmap : array (height, width)
            Current heightmap to modify.

        Returns
        -------
        heightmap : array (height, width)
            Modified heightmap.
        """
        config = self.config

        # Generate boundaries if not yet computed
        if self.boundary_map is None:
            self.compute_boundaries(width, height)

        boundary_map = self.boundary_map
        boundary_type = self.boundary_type_map

        # Smooth the boundary map further for gradual transitions (no hard lines)
        boundary_map = self._gaussian_blur(boundary_map, 5)

        # ── Convergent boundaries: Mountain formation ───────────────────
        # Use soft masks instead of hard binary — gradual falloff
        convergent = xp.where(boundary_type == 1, 1.0, 0.0)
        convergent = self._gaussian_blur(convergent, 4)  # Smooth transition
        # Mountain height depends on boundary strength and collision force
        mountain_height = convergent * boundary_map * config.mountain_height_factor

        # Add mountain ridgeline noise for realistic mountain shapes
        mountain_noise = self.noise.ridged_multifractal(
            width, height, octaves=6, scale=4.0,
            offset_x=self._rng.uniform(-100, 100),
            offset_y=self._rng.uniform(-100, 100)
        )
        # Modulate mountain height with ridged noise
        mountain_height *= (0.5 + 0.5 * mountain_noise)

        # Continental-continental collision = highest mountains (Himalayas-like)
        # Oceanic-continental = medium mountains with trench
        # Oceanic-oceanic = island arcs
        for plate in self.plates:
            if not plate.is_oceanic:
                # Continental plate: stronger uplift
                mask = (self.plate_map == plate.plate_id).astype(xp.float32)
                mountain_height += convergent * mask * 0.3

        # ── Divergent boundaries: Rifts and ridges ──────────────────────
        divergent = xp.where(boundary_type == 2, 1.0, 0.0)
        divergent = self._gaussian_blur(divergent, 4)

        # Mid-ocean ridges: elevated but below land
        ridge_height = divergent * boundary_map * config.ridge_height_factor
        ridge_noise = self.noise.fbm(width, height, octaves=4, scale=5.0)
        ridge_height *= (0.5 + 0.5 * ridge_noise)

        # Continental rifts: lowered elevation
        rift_depth = divergent * boundary_map * 0.3

        # ── Transform boundaries: Fault scarps ──────────────────────────
        transform = xp.where(boundary_type == 3, 1.0, 0.0)
        transform = self._gaussian_blur(transform, 3)
        fault_height = transform * boundary_map * 0.2
        fault_noise = self.noise.perlin_2d(width, height, 8.0)
        fault_height *= fault_noise

        # ── Apply all tectonic modifications ────────────────────────────
        heightmap = heightmap + mountain_height * 0.15
        heightmap = heightmap + ridge_height * 0.08
        heightmap = heightmap - rift_depth * 0.1
        heightmap = heightmap + fault_height * 0.05

        # ── Isostatic adjustment ────────────────────────────────────────
        # Thicker (continental) crust floats higher, thinner (oceanic) sinks
        isostasy = self._compute_isostasy(width, height)
        heightmap += isostasy * 0.1

        return heightmap

    def _compute_isostasy(self, width, height):
        """
        Compute isostatic adjustment based on crustal thickness and density.
        Thicker, less dense crust (continental) floats higher.
        Thinner, denser crust (oceanic) sits lower.
        """
        thickness_map = xp.zeros((height, width), dtype=xp.float32)
        density_map = xp.zeros((height, width), dtype=xp.float32)

        for plate in self.plates:
            mask = (self.plate_map == plate.plate_id).astype(xp.float32)
            thickness_map += mask * plate.thickness
            density_map += mask * plate.density

        # Isostatic elevation ~ thickness * (mantle_density - crust_density) / mantle_density
        # Simplified: thicker + less dense = higher elevation
        isostasy = (thickness_map / 35.0) * (3.3 - density_map) / 0.6

        return isostasy

    def _gaussian_blur(self, arr, radius):
        """
        Gaussian blur using padded convolution — no edge wrapping.
        Separable 1D kernel applied horizontally then vertically.
        """
        size = 2 * radius + 1
        x = xp.arange(size, dtype=xp.float32) - radius
        kernel_1d = xp.exp(-x * x / (2 * (radius / 2) ** 2))
        kernel_1d /= kernel_1d.sum()

        h, w = arr.shape

        # Horizontal pass — pad array to avoid wrapping
        padded = xp.pad(arr, ((0, 0), (radius, radius)), mode='edge')
        result = xp.zeros_like(arr)
        for i in range(size):
            result += padded[:, i:i + w] * kernel_1d[i]

        # Vertical pass — pad result to avoid wrapping
        padded2 = xp.pad(result, ((radius, radius), (0, 0)), mode='edge')
        temp = xp.zeros_like(arr)
        for i in range(size):
            temp += padded2[i:i + h, :] * kernel_1d[i]

        return temp

    def simulate_drift(self, delta_time, width, height):
        """
        Simulate one step of plate motion (continental drift).

        Updates plate positions based on their velocity vectors,
        then recomputes the plate map and boundaries.

        Parameters
        ----------
        delta_time : float
            Time step in simulation units (millions of years).
        """
        for plate in self.plates:
            # Move plate center
            new_lon = (plate.center[0] + plate.velocity[0] * delta_time * 1e-8) % (2 * np.pi)
            new_lat = np.clip(
                plate.center[1] + plate.velocity[1] * delta_time * 1e-8,
                -np.pi / 2, np.pi / 2
            )
            plate.center = (new_lon, new_lat)

        # Rebuild plate map with new positions
        self._build_plate_map(width, height)
        self.compute_boundaries(width, height)
