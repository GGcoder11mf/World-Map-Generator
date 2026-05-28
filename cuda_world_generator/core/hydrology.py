"""
Hydrology Simulation
=====================

GPU-accelerated hydrological simulation that generates:

- River networks (water flowing downhill from sources to sea)
- Lake/pond formation in enclosed basins
- Flow accumulation maps (watershed analysis)
- Drainage basins and watershed boundaries

Physical principles:
- Water always flows downhill (steepest descent)
- Flow accumulates downstream (rivers grow larger)
- Enclosed basins form lakes until they overflow
- River sources: high altitude, high rainfall areas
- Meandering: rivers curve due to Coriolis and terrain
"""

import numpy as np
from .backend import xp, to_cpu, to_gpu


class HydrologySimulator:
    """
    GPU-accelerated hydrology engine.

    Generates realistic river networks and water features
    using physically-based flow simulation.
    """

    def __init__(self, config):
        self.config = config
        self._rng = np.random.RandomState(config.seed + 500)
        self.river_map = None
        self.flow_accumulation = None
        self.flow_direction = None
        self.lake_map = None

    def simulate(self, heightmap, rainfall, sea_level, width, height):
        """
        Run full hydrological simulation.

        Parameters
        ----------
        heightmap : array (H, W)
        rainfall : array (H, W) - precipitation
        sea_level : float
        width, height : int

        Returns
        -------
        river_map : array (H, W) - river flow magnitude
        flow_accumulation : array (H, W) - accumulated flow
        lake_map : array (H, W) - lake depth
        river_graph : dict - river network graph
        """
        # ── Step 1: Compute flow directions ─────────────────────────────
        flow_dir = self._compute_flow_directions(heightmap, width, height)
        self.flow_direction = flow_dir

        # ── Step 2: Compute flow accumulation ───────────────────────────
        flow_acc = self._compute_flow_accumulation(heightmap, flow_dir, 
                                                     rainfall, sea_level,
                                                     width, height)
        self.flow_accumulation = flow_acc

        # ── Step 3: Extract rivers from flow accumulation ───────────────
        river_map = self._extract_rivers(flow_acc, sea_level, heightmap, width, height)
        self.river_map = river_map

        # ── Step 4: Find and fill lakes ─────────────────────────────────
        lake_map = self._find_lakes(heightmap, flow_acc, sea_level, width, height)
        self.lake_map = lake_map

        # ── Step 5: Build river network graph ───────────────────────────
        river_graph = self._build_river_graph(river_map, flow_dir, width, height)

        return river_map, flow_acc, lake_map, river_graph

    def _compute_flow_directions(self, heightmap, width, height):
        """
        Compute D8 flow direction: each cell flows to its steepest
        downhill neighbor among 8 adjacent cells.

        Direction encoding (D8):
        0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE

        Returns
        -------
        flow_dir : array (H, W) - D8 direction code per cell
        """
        # Pad for boundary handling
        padded = xp.pad(heightmap, 1, mode='edge')

        # Height of 8 neighbors
        neighbors = xp.stack([
            padded[1:-1, 2:],    # 0: East
            padded[2:, 2:],      # 1: Southeast
            padded[2:, 1:-1],    # 2: South
            padded[2:, :-2],     # 3: Southwest
            padded[1:-1, :-2],   # 4: West
            padded[:-2, :-2],    # 5: Northwest
            padded[:-2, 1:-1],   # 6: North
            padded[:-2, 2:],     # 7: Northeast
        ], axis=0)  # Shape: (8, H, W)

        # Distance factors (cardinal=1, diagonal=sqrt(2))
        distances = xp.array([1, 1.414, 1, 1.414, 1, 1.414, 1, 1.414],
                             dtype=xp.float32).reshape(8, 1, 1)

        # Slope to each neighbor (positive = downhill)
        slopes = (heightmap[xp.newaxis, :, :] - neighbors) / distances

        # Find steepest downhill neighbor
        # Replace negative (uphill) slopes with large negative number
        slopes = xp.where(slopes > 0, slopes, -1e6)
        flow_dir = xp.argmax(slopes, axis=0).astype(xp.int32)

        # Flat areas or pits: mark as -1 (no outflow)
        max_slope = xp.max(slopes, axis=0)
        flow_dir = xp.where(max_slope <= 0, -1, flow_dir)

        return flow_dir

    def _compute_flow_accumulation(self, heightmap, flow_dir, rainfall,
                                    sea_level, width, height):
        """
        Compute flow accumulation: how much water flows through each cell.

        Uses an iterative approach suitable for GPU parallelism:
        1. Initialize each cell with its local rainfall contribution
        2. Iteratively propagate flow downhill
        3. Converges to stable flow accumulation

        Parameters
        ----------
        heightmap, flow_dir, rainfall, sea_level

        Returns
        -------
        flow_acc : array (H, W)
        """
        h, w = heightmap.shape
        is_ocean = heightmap < sea_level

        # Initialize with rainfall as source
        flow_acc = rainfall.copy()
        # Ocean cells get zero accumulation (water already reached sea)
        flow_acc = xp.where(is_ocean, 0, flow_acc)

        # D8 direction offsets (dy, dx)
        dy_dx = xp.array([
            [0, 1],    # 0: East
            [1, 1],    # 1: Southeast
            [1, 0],    # 2: South
            [1, -1],   # 3: Southwest
            [0, -1],   # 4: West
            [-1, -1],  # 5: Northwest
            [-1, 0],   # 6: North
            [-1, 1],   # 7: Northeast
        ], dtype=xp.int32)

        # Iterative flow propagation
        # Process from highest to lowest elevation for correct accumulation
        iterations = self.config.river_iterations

        for iteration in range(iterations):
            # For each cell, add its flow to the downhill neighbor
            outflow = xp.zeros_like(flow_acc)

            for d in range(8):
                # Mask of cells flowing in direction d
                flowing = (flow_dir == d) & (~is_ocean)

                # Shift flow to the target cell (padded — no wrapping)
                dy, dx_val = int(dy_dx[d, 0]), int(dy_dx[d, 1])
                flow_contribution = flow_acc * flowing
                # Pad and extract shifted region
                padded_fc = xp.pad(flow_contribution, 1, mode='constant', constant_values=0)
                # Source cell at (y, x) needs to go to (y+dy, x+dx)
                # So to get what arrives at (y, x), we look at (y-dy, x-dx) in padded
                sy, sx = 1 - dy, 1 - dx_val
                shifted = padded_fc[sy:sy + h, sx:sx + w]
                outflow += shifted * 0.1  # Fraction propagated per iteration

            # Accumulate inflow
            flow_acc += outflow

            # Prevent ocean accumulation from growing
            flow_acc = xp.where(is_ocean, 0, flow_acc)

        # Normalize flow accumulation
        max_flow = flow_acc.max()
        if max_flow > 0:
            flow_acc = flow_acc / max_flow

        return flow_acc

    def _extract_rivers(self, flow_acc, sea_level, heightmap, width, height):
        """
        Extract river map from flow accumulation.
        Rivers exist where flow exceeds a threshold, creating
        a network from headwaters to the sea.
        """
        is_ocean = heightmap < sea_level
        threshold = self.config.river_flow_threshold

        # River presence based on flow accumulation
        river = xp.where(
            (flow_acc > threshold) & (~is_ocean),
            flow_acc,
            0
        )

        # Apply minimum length filter (remove isolated pixels)
        # Simple morphological opening
        for _ in range(3):
            padded = xp.pad(river, 1, mode='constant', constant_values=0)
            # Erosion: keep only cells with 2+ neighbors
            neighbor_count = (
                (padded[:-2, 1:-1] > 0).astype(xp.float32) +
                (padded[2:, 1:-1] > 0).astype(xp.float32) +
                (padded[1:-1, :-2] > 0).astype(xp.float32) +
                (padded[1:-1, 2:] > 0).astype(xp.float32)
            )
            river = xp.where(neighbor_count >= 2, river, 0)

        return river

    def _find_lakes(self, heightmap, flow_acc, sea_level, width, height):
        """
        Find enclosed basins (depressions) and fill them as lakes.

        A lake forms when water accumulates in a topographic low
        with no outflow. The lake fills until it overflows at
        the lowest point on the basin rim.

        Parameters
        ----------
        heightmap, flow_acc, sea_level

        Returns
        -------
        lake_map : array (H, W) - lake depth (>0 where lake exists)
        """
        is_ocean = heightmap < sea_level
        lake_map = xp.zeros_like(heightmap)

        # Find pits (cells with no outflow that aren't ocean)
        pit_mask = (self.flow_direction == -1) & (~is_ocean)

        # Fill pits iteratively (simplified lake formation)
        water_level = heightmap.copy()

        for iteration in range(self.config.lake_fill_iterations):
            # Add water to pits
            water_level = xp.where(pit_mask, water_level + 0.001, water_level)

            # Water flows to lowest neighbor
            padded = xp.pad(water_level, 1, mode='edge')
            min_neighbor = xp.minimum(
                xp.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
                xp.minimum(padded[1:-1, :-2], padded[1:-1, 2:])
            )

            # Level water: water_surface flows toward neighbors
            water_level = xp.where(
                pit_mask,
                xp.maximum(water_level * 0.99, min_neighbor * 1.001),
                water_level
            )

            # If water level reaches sea level, it drains
            water_level = xp.where(
                pit_mask & (water_level <= sea_level + 0.01),
                sea_level,
                water_level
            )

        # Lake depth = water level - terrain height
        lake_map = xp.maximum(water_level - heightmap, 0)
        lake_map = xp.where(is_ocean, 0, lake_map)

        return lake_map

    def _build_river_graph(self, river_map, flow_dir, width, height):
        """
        Build a river network graph for navigation and analysis.

        The graph represents the connectivity of the river system:
        - Nodes: significant river points (confluences, sources, mouths)
        - Edges: river segments between nodes
        - Attributes: flow volume, width, direction

        Returns
        -------
        graph : dict with keys:
            'nodes': list of (x, y, flow) tuples
            'edges': list of (node_a, node_b) tuples
            'node_attrs': dict of node attributes
            'edge_attrs': dict of edge attributes
        """
        river_cpu = to_cpu(river_map)
        flow_dir_cpu = to_cpu(flow_dir)

        nodes = []
        edges = []
        node_id_map = {}
        next_id = 0

        # D8 direction offsets
        dy_dx = [(0, 1), (1, 1), (1, 0), (1, -1),
                 (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        # Find significant river points (confluences, sources, mouths)
        step = max(1, min(width, height) // 64)

        for y in range(0, height, step):
            for x in range(0, width, step):
                flow = river_cpu[y, x]
                if flow > 0.02:
                    node_id_map[(x, y)] = next_id
                    nodes.append((x, y, float(flow)))
                    next_id += 1

        # Connect adjacent nodes following flow direction
        for (x, y), nid in node_id_map.items():
            d = int(flow_dir_cpu[y, x])
            if 0 <= d < 8:
                # Follow flow direction to find next node
                dy, dx = dy_dx[d]
                nx, ny = x + dx * step, y + dy * step
                if (nx, ny) in node_id_map:
                    edges.append((nid, node_id_map[(nx, ny)]))

        return {
            'nodes': nodes,
            'edges': edges,
            'node_attrs': {
                i: {'flow': nodes[i][2], 'x': nodes[i][0], 'y': nodes[i][1]}
                for i in range(len(nodes))
            },
            'edge_attrs': {
                i: {'flow': min(nodes[e[0]][2], nodes[e[1]][2])}
                for i, e in enumerate(edges)
            }
        }
