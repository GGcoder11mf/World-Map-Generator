"""
Erosion Simulation
===================

GPU-accelerated hydraulic and thermal erosion that reshapes
the heightmap according to physically-grounded processes.

Hydraulic Erosion:
- Rainfall deposits water on the terrain
- Water flows downhill, picking up sediment based on slope and speed
- Sediment is deposited when water slows or exceeds capacity
- Water evaporates over time
- Creates realistic river valleys, gullies, alluvial plains

Thermal Erosion:
- Material slides down slopes exceeding the talus angle
- Simulates freeze-thaw weathering and rockfall
- Smooths steep cliffs while maintaining realistic angles
- Creates talus slopes and scree fields

All operations are fully vectorized for GPU parallelism.
NO xp.roll is used — all neighbor access uses padded arrays
to avoid edge-wrapping artifacts (seam lines).
"""

import numpy as np
from .backend import xp, to_cpu, to_gpu


def _shift(arr, dy, dx):
    """
    Shift an array by (dy, dx) cells with edge padding (no wrapping).
    Replaces xp.roll() which wraps data around edges causing seam lines.

    Parameters
    ----------
    arr : array (H, W)
    dy : int - vertical shift (positive = down)
    dx : int - horizontal shift (positive = right)

    Returns
    -------
    shifted : array (H, W)
    """
    h, w = arr.shape
    padded = xp.pad(arr, max(abs(dy), abs(dx)) + 1, mode='edge')
    # Compute the top-left corner in the padded array
    oy = max(abs(dy), abs(dx)) + 1 + dy
    ox = max(abs(dy), abs(dx)) + 1 + dx
    return padded[oy:oy + h, ox:ox + w]


def _padded_neighbors(arr):
    """
    Return the 4 cardinal neighbors using edge-padded array.
    Avoids xp.roll wrapping artifacts.

    Returns
    -------
    left, right, up, down : arrays same shape as arr
    """
    padded = xp.pad(arr, 1, mode='edge')
    left  = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    up    = padded[:-2, 1:-1]
    down  = padded[2:, 1:-1]
    return left, right, up, down


def _padded_gradient(arr):
    """
    Compute central-difference gradient using edge-padded array.
    No wrapping artifacts.

    Returns
    -------
    grad_x, grad_y : arrays same shape as arr
    """
    padded = xp.pad(arr, 1, mode='edge')
    grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2.0
    grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2.0
    return grad_x, grad_y


class ErosionSimulator:
    """
    GPU-accelerated erosion engine.

    Implements iterative hydraulic and thermal erosion using
    batched array operations. Each iteration processes all
    cells simultaneously on the GPU, avoiding per-cell loops.
    """

    def __init__(self, config):
        self.config = config
        self._rng = np.random.RandomState(config.seed + 200)

    def erode(self, heightmap, rainfall_map=None):
        """
        Apply full erosion pipeline to the heightmap.

        Parameters
        ----------
        heightmap : array (H, W)
            Terrain elevation data.
        rainfall_map : array (H, W), optional
            Spatially varying rainfall. Uniform if not provided.

        Returns
        -------
        heightmap : array (H, W)
            Eroded heightmap.
        water_map : array (H, W)
            Final water flow accumulation.
        sediment_map : array (H, W)
            Deposited sediment thickness.
        """
        config = self.config

        # Apply hydraulic erosion
        heightmap, water_map, sediment_map = self.hydraulic_erosion(
            heightmap, rainfall_map,
            iterations=config.hydraulic_erosion_iterations
        )

        # Apply thermal erosion
        heightmap = self.thermal_erosion(
            heightmap,
            iterations=config.thermal_erosion_iterations
        )

        return heightmap, water_map, sediment_map

    def hydraulic_erosion(self, heightmap, rainfall_map=None, iterations=40):
        """
        GPU-parallel hydraulic erosion simulation.
        Uses edge-padded neighbor access — NO seam-line artifacts.
        """
        config = self.config
        h, w = heightmap.shape

        # Initialize water and sediment maps
        water = xp.zeros_like(heightmap)
        sediment = xp.zeros_like(heightmap)

        # Default uniform rainfall
        if rainfall_map is None:
            rainfall_map = xp.full_like(heightmap, config.rainfall_rate)
        else:
            rainfall_map = xp.asarray(rainfall_map)

        # Rainfall noise for spatial variation
        rain_noise = self._simple_noise(h, w)
        rainfall_map = rainfall_map * (0.5 + 0.5 * rain_noise)

        for iteration in range(iterations):
            # ── 1. Add rainfall ────────────────────────────────────────
            water += rainfall_map * 0.1

            # ── 2. Compute water surface height ────────────────────────
            water_surface = heightmap + water

            # ── 3. Compute flow directions and amounts ─────────────────
            # Use padded neighbors (no wrapping)
            ws_left, ws_right, ws_up, ws_down = _padded_neighbors(water_surface)

            # Height differences (positive = current is higher = flow out)
            dh_left  = water_surface - ws_left
            dh_right = water_surface - ws_right
            dh_up    = water_surface - ws_up
            dh_down  = water_surface - ws_down

            # Only flow downhill
            flow_left  = xp.maximum(dh_left, 0)
            flow_right = xp.maximum(dh_right, 0)
            flow_up    = xp.maximum(dh_up, 0)
            flow_down  = xp.maximum(dh_down, 0)

            # Total outflow
            total_flow = flow_left + flow_right + flow_up + flow_down + 1e-10

            # Normalize: can't flow more water than available
            flow_fraction = xp.minimum(water / (total_flow + 1e-10), 1.0)
            total_flow *= flow_fraction

            # ── 4. Distribute water to neighbors ───────────────────────
            # Fraction of flow in each direction
            fl = flow_left / (total_flow + 1e-10)
            fr = flow_right / (total_flow + 1e-10)
            fu = flow_up / (total_flow + 1e-10)
            fd = flow_down / (total_flow + 1e-10)

            # Remove outflowing water
            water -= total_flow

            # Add inflowing water from neighbors (using padded shift — no wrapping)
            water += _shift(total_flow * fl, 0, 1)    # from left neighbor
            water += _shift(total_flow * fr, 0, -1)   # from right neighbor
            water += _shift(total_flow * fu, 1, 0)    # from above neighbor
            water += _shift(total_flow * fd, -1, 0)   # from below neighbor

            # ── 5. Compute slope for erosion (padded gradient) ─────────
            grad_x, grad_y = _padded_gradient(heightmap)
            slope = xp.sqrt(grad_x ** 2 + grad_y ** 2)

            # ── 6. Erode terrain ───────────────────────────────────────
            erosion_rate = config.sediment_capacity_factor
            sediment_capacity = water * slope * erosion_rate

            # If carrying less than capacity, erode more
            erosion_amount = xp.minimum(
                xp.maximum(sediment_capacity - sediment, 0) * 0.1,
                slope * 0.05
            )

            heightmap -= erosion_amount * 0.01
            sediment += erosion_amount * 0.01

            # ── 7. Deposit sediment ────────────────────────────────────
            excess = xp.maximum(sediment - sediment_capacity, 0)
            deposit = excess * config.sediment_deposition_rate
            deposit = xp.minimum(deposit, sediment)

            heightmap += deposit
            sediment -= deposit

            # ── 8. Evaporate water ─────────────────────────────────────
            water *= (1.0 - config.water_evaporation_rate)
            water = xp.maximum(water, 0)

            # ── 9. Deposit remaining sediment with evaporation ─────────
            deposit_remaining = sediment * config.water_evaporation_rate
            heightmap += deposit_remaining
            sediment -= deposit_remaining
            sediment = xp.maximum(sediment, 0)

        return heightmap, water, sediment

    def thermal_erosion(self, heightmap, iterations=20):
        """
        GPU-parallel thermal erosion simulation.
        Uses padded neighbors — no wrapping artifacts.
        """
        config = self.config
        talus = config.thermal_talus_angle
        rate = config.thermal_erosion_rate

        for iteration in range(iterations):
            # Padded neighbors (no wrapping)
            h_left, h_right, h_up, h_down = _padded_neighbors(heightmap)

            # Height differences to 4 neighbors
            dh_left  = heightmap - h_left
            dh_right = heightmap - h_right
            dh_up    = heightmap - h_up
            dh_down  = heightmap - h_down

            # Positive differences only (downhill)
            max_slope = xp.maximum(dh_left, 0)
            max_slope = xp.maximum(max_slope, dh_right)
            max_slope = xp.maximum(max_slope, dh_up)
            max_slope = xp.maximum(max_slope, dh_down)

            # Material to transfer = excess above talus angle
            excess = xp.maximum(max_slope - talus, 0) * rate

            # Distribute excess to each neighbor proportionally
            for dh, dy, dx in [
                (dh_left,  0, -1),   # transfer to left neighbor
                (dh_right, 0,  1),   # transfer to right neighbor
                (dh_up,   -1,  0),   # transfer to upper neighbor
                (dh_down,  1,  0),   # transfer to lower neighbor
            ]:
                transfer = xp.where(
                    dh > talus,
                    excess * dh / (max_slope + 1e-10) * 0.25,
                    0
                )
                heightmap -= transfer
                heightmap += _shift(transfer, dy, dx)  # No wrapping

        return heightmap

    def coastal_erosion(self, heightmap, sea_level, iterations=10):
        """
        Erode coastal areas where land meets sea.
        Creates realistic coastlines with bays, peninsulas, and beaches.
        """
        for _ in range(iterations):
            is_land = heightmap > sea_level
            padded = xp.pad(is_land.astype(xp.float32), 1, mode='constant', constant_values=0)

            sea_neighbors = (
                (1 - padded[:-2, 1:-1]) +
                (1 - padded[2:, 1:-1]) +
                (1 - padded[1:-1, :-2]) +
                (1 - padded[1:-1, 2:])
            )

            coastal_mask = is_land & (sea_neighbors > 0)
            erosion = coastal_mask.astype(xp.float32) * sea_neighbors * 0.002
            heightmap -= erosion

            inland_mask = is_land & (sea_neighbors > 0) & (sea_neighbors < 3)
            heightmap += inland_mask.astype(xp.float32) * 0.0005

        return heightmap

    def _simple_noise(self, h, w):
        """Generate simple noise for rainfall variation."""
        xs = xp.linspace(0, 5, w, dtype=xp.float32)
        ys = xp.linspace(0, 5, h, dtype=xp.float32)
        X, Y = xp.meshgrid(xs, ys)

        noise = (
            xp.sin(X * 1.7 + Y * 2.3) * 0.5 +
            xp.sin(X * 3.1 - Y * 1.1) * 0.25 +
            xp.sin(X * 5.3 + Y * 4.7) * 0.125
        )

        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)
        return noise

    def compute_slope_map(self, heightmap):
        """
        Compute terrain slope at each point.
        Uses padded gradient — no wrapping.
        """
        grad_x, grad_y = _padded_gradient(heightmap)
        slope = xp.sqrt(grad_x ** 2 + grad_y ** 2)
        aspect = xp.arctan2(grad_y, grad_x)
        return slope, aspect
