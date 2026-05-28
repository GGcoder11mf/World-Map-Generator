"""
LOD (Level of Detail) Chunking System
=======================================

Implements a hierarchical chunking system for managing
planet-scale or continent-scale worlds efficiently.

Features:
- Quadtree-based LOD: higher detail near viewer, lower far away
- Chunk-based loading/unloading for memory efficiency
- Seamless stitching between LOD levels
- Deterministic generation per chunk (same seed = same content)
- Frustum culling for rendering optimization

Architecture:
- World is divided into a grid of chunks
- Each chunk can be subdivided into 4 children (quadtree)
- LOD level determines chunk resolution
- Chunks are generated on-demand and cached
"""

import numpy as np
from .backend import xp, to_cpu, to_gpu
from .noise import NoiseGenerator


class Chunk:
    """
    Represents a rectangular region of the world at a specific LOD level.

    Attributes
    ----------
    x, y : int
        Grid coordinates of this chunk.
    lod : int
        Level of detail (0 = highest detail).
    resolution : int
        Heightmap resolution of this chunk.
    bounds : tuple
        (x_min, y_min, x_max, y_max) in world coordinates.
    """

    def __init__(self, chunk_x, chunk_y, lod, chunk_size, world_size):
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.lod = lod
        self.chunk_size = chunk_size

        # World-space bounds
        scale = 2 ** lod
        self.x_min = chunk_x * chunk_size * scale
        self.y_min = chunk_y * chunk_size * scale
        self.x_max = self.x_min + chunk_size * scale
        self.y_max = self.y_min + chunk_size * scale

        # Resolution at this LOD
        self.resolution = max(chunk_size // (2 ** lod), 8)

        # Data (generated on demand)
        self.heightmap = None
        self.biome_map = None
        self.generated = False

    def __repr__(self):
        return (f"Chunk(x={self.chunk_x}, y={self.chunk_y}, "
                f"lod={self.lod}, res={self.resolution})")


class LODManager:
    """
    Level-of-detail chunk management system.

    Manages world subdivision into chunks with varying detail
    levels based on distance from a viewer position.
    """

    def __init__(self, config):
        self.config = config
        self.chunk_size = config.chunk_size
        self.lod_levels = config.lod_levels
        self.noise = NoiseGenerator(config.seed)
        self.chunks = {}  # (x, y, lod) -> Chunk
        self._rng = np.random.RandomState(config.seed + 700)

    def get_chunk(self, chunk_x, chunk_y, lod=0):
        """
        Get or create a chunk at the specified position and LOD.

        Parameters
        ----------
        chunk_x, chunk_y : int
            Grid coordinates.
        lod : int
            Level of detail (0 = highest).

        Returns
        -------
        Chunk
        """
        key = (chunk_x, chunk_y, lod)
        if key not in self.chunks:
            self.chunks[key] = Chunk(
                chunk_x, chunk_y, lod,
                self.chunk_size, self.config.size
            )
        return self.chunks[key]

    def get_visible_chunks(self, viewer_x, viewer_y, view_distance,
                           world_width, world_height):
        """
        Determine which chunks should be visible based on viewer position.

        Chunks near the viewer get higher LOD; distant chunks get lower LOD.

        Parameters
        ----------
        viewer_x, viewer_y : float
            Viewer position in world coordinates.
        view_distance : float
            Maximum view distance.
        world_width, world_height : int

        Returns
        -------
        chunks : list of Chunk
            Visible chunks at appropriate LOD levels.
        """
        visible = []

        # Determine LOD for each region based on distance
        for lod in range(self.lod_levels):
            scale = 2 ** lod
            chunk_world_size = self.chunk_size * scale

            # How many chunks at this LOD cover the view distance
            chunks_in_range = int(view_distance / chunk_world_size) + 1

            # Viewer's chunk position at this LOD
            vcx = int(viewer_x / chunk_world_size)
            vcy = int(viewer_y / chunk_world_size)

            # Only add chunks at this LOD if they're beyond the
            # higher LOD's range
            if lod > 0:
                inner_range = view_distance / (2 ** lod)

            for dx in range(-chunks_in_range, chunks_in_range + 1):
                for dy in range(-chunks_in_range, chunks_in_range + 1):
                    cx = vcx + dx
                    cy = vcy + dy

                    # Check if within view distance
                    chunk_center_x = (cx + 0.5) * chunk_world_size
                    chunk_center_y = (cy + 0.5) * chunk_world_size
                    dist = np.sqrt(
                        (chunk_center_x - viewer_x) ** 2 +
                        (chunk_center_y - viewer_y) ** 2
                    )

                    # LOD selection based on distance
                    min_dist = view_distance / (2 ** (self.lod_levels - lod))
                    max_dist = view_distance / (2 ** max(self.lod_levels - lod - 1, 0))

                    if lod == 0:
                        if dist < min_dist:
                            visible.append(self.get_chunk(cx, cy, lod))
                    else:
                        if min_dist <= dist < max_dist:
                            visible.append(self.get_chunk(cx, cy, lod))

        return visible

    def generate_chunk(self, chunk, noise_gen):
        """
        Generate terrain data for a single chunk.

        Uses the noise generator with chunk-specific offsets
        for deterministic, seamless generation.

        Parameters
        ----------
        chunk : Chunk
            The chunk to generate.
        noise_gen : NoiseGenerator
            Noise generator instance.

        Returns
        -------
        heightmap : array (resolution, resolution)
        """
        if chunk.generated:
            return chunk.heightmap

        res = chunk.resolution
        scale = 2 ** chunk.lod

        # Compute noise offset from chunk position
        offset_x = chunk.chunk_x * self.chunk_size * scale / self.config.size
        offset_y = chunk.chunk_y * self.chunk_size * scale / self.config.size

        # Generate heightmap for this chunk
        heightmap = noise_gen.fbm(
            res, res,
            octaves=max(self.config.terrain_octaves - chunk.lod, 2),
            persistence=self.config.terrain_persistence,
            lacunarity=self.config.terrain_lacunarity,
            scale=self.config.terrain_scale,
            offset_x=offset_x,
            offset_y=offset_y
        )

        chunk.heightmap = heightmap
        chunk.generated = True

        return heightmap

    def stitch_chunk_boundaries(self, chunk, neighbors):
        """
        Smoothly stitch a chunk with its neighbors to avoid
        visible seams between LOD levels.

        Uses weighted blending at chunk boundaries where
        different LOD levels meet.

        Parameters
        ----------
        chunk : Chunk
            Chunk to stitch.
        neighbors : dict
            {(dx, dy): Chunk} neighboring chunks.
        """
        if chunk.heightmap is None:
            return

        blend_width = max(4, chunk.resolution // 16)

        for (dx, dy), neighbor in neighbors.items():
            if neighbor.heightmap is None:
                continue

            # Resample neighbor to match this chunk's resolution
            # (if different LOD)
            neighbor_resampled = self._resample(
                neighbor.heightmap, chunk.resolution
            )

            if dx == 1:  # Right neighbor
                for i in range(blend_width):
                    weight = i / blend_width
                    col = chunk.resolution - blend_width + i
                    chunk.heightmap[:, col] = (
                        chunk.heightmap[:, col] * (1 - weight) +
                        neighbor_resampled[:, i] * weight
                    )
            elif dx == -1:  # Left neighbor
                for i in range(blend_width):
                    weight = (blend_width - i) / blend_width
                    chunk.heightmap[:, i] = (
                        chunk.heightmap[:, i] * (1 - weight) +
                        neighbor_resampled[:, chunk.resolution - blend_width + i] * weight
                    )
            elif dy == 1:  # Bottom neighbor
                for i in range(blend_width):
                    weight = i / blend_width
                    row = chunk.resolution - blend_width + i
                    chunk.heightmap[row, :] = (
                        chunk.heightmap[row, :] * (1 - weight) +
                        neighbor_resampled[i, :] * weight
                    )
            elif dy == -1:  # Top neighbor
                for i in range(blend_width):
                    weight = (blend_width - i) / blend_width
                    chunk.heightmap[i, :] = (
                        chunk.heightmap[i, :] * (1 - weight) +
                        neighbor_resampled[chunk.resolution - blend_width + i, :] * weight
                    )

    def _resample(self, data, target_size):
        """
        Resample a 2D array to a target size using bilinear interpolation.
        """
        if data.shape[0] == target_size and data.shape[1] == target_size:
            return data

        src_h, src_w = data.shape
        y_indices = xp.linspace(0, src_h - 1, target_size)
        x_indices = xp.linspace(0, src_w - 1, target_size)

        x_grid, y_grid = xp.meshgrid(x_indices, y_indices)

        x0 = xp.floor(x_grid).astype(xp.int32)
        y0 = xp.floor(y_grid).astype(xp.int32)
        x1 = xp.minimum(x0 + 1, src_w - 1)
        y1 = xp.minimum(y0 + 1, src_h - 1)

        xf = x_grid - x0
        yf = y_grid - y0

        result = (
            data[y0, x0] * (1 - xf) * (1 - yf) +
            data[y0, x1] * xf * (1 - yf) +
            data[y1, x0] * (1 - xf) * yf +
            data[y1, x1] * xf * yf
        )

        return result

    def unload_distant_chunks(self, viewer_x, viewer_y, unload_distance):
        """
        Remove chunks that are far from the viewer to save memory.

        Parameters
        ----------
        viewer_x, viewer_y : float
        unload_distance : float
            Distance beyond which chunks are unloaded.
        """
        to_remove = []
        for key, chunk in self.chunks.items():
            cx = (chunk.chunk_x + 0.5) * chunk.chunk_size * (2 ** chunk.lod)
            cy = (chunk.chunk_y + 0.5) * chunk.chunk_size * (2 ** chunk.lod)
            dist = np.sqrt((cx - viewer_x) ** 2 + (cy - viewer_y) ** 2)
            if dist > unload_distance:
                to_remove.append(key)

        for key in to_remove:
            del self.chunks[key]
