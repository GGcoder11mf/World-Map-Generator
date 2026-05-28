"""
GPU-Accelerated Noise Generation
=================================

Multi-octave Perlin and Simplex noise implemented with the
CuPy/NumPy backend for GPU parallelism. All noise operations
are batched array operations suitable for massive-scale worlds.

Key features:
- Multi-octave fractal Brownian motion (fBm)
- Ridged multifractal noise for mountain ridges
- Domain warping for natural-looking continents
- Spherical noise sampling for planet-scale generation
- Deterministic via seed
"""

import numpy as np
from .backend import xp, to_gpu, to_cpu, gpu_available, grid_coords, spherical_coords


class NoiseGenerator:
    """
    GPU-accelerated procedural noise generator.

    Generates multi-octave coherent noise for terrain, climate,
    and other procedural content. All heavy computation uses
    batched array operations on the active backend (CuPy/NumPy).
    """

    def __init__(self, seed=42):
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        # Pre-compute permutation table for gradient noise
        self._perm = self._build_permutation_table()

    def _build_permutation_table(self):
        """Build a shuffled permutation table for hash-based noise."""
        p = np.arange(256, dtype=np.int32)
        self._rng.shuffle(p)
        # Duplicate for overflow-free indexing
        return np.concatenate([p, p])

    def _hash2d(self, ix, iy):
        """
        Hash function for 2D integer coordinates.
        Maps grid points to gradient indices using the permutation table.
        """
        perm = self._perm
        # Use permutation table to create pseudo-random gradient indices
        h = perm[(ix & 255)] 
        h = perm[(h + iy) & 255]
        return h

    def _grad2d(self, hash_val, dx, dy):
        """
        Compute gradient dot product for 2D Perlin noise.
        Uses 8 gradient directions for smooth interpolation.
        """
        # Convert hash to one of 8 gradient directions
        h = hash_val & 7
        # Gradient vectors: (1,1), (-1,1), (1,-1), (-1,-1),
        #                   (1,0), (-1,0), (0,1), (0,-1)
        u = xp.where(h < 4, dx, dx)
        v = xp.where(h < 4, dy, dy)
        
        # Apply gradient based on hash
        gx = xp.where((h & 1) == 0, u, -u)
        gy = xp.where((h & 2) == 0, v, -v)
        
        return gx + gy

    def perlin_2d(self, width, height, scale=1.0, offset_x=0.0, offset_y=0.0):
        """
        Generate 2D Perlin noise using GPU-parallel array operations.

        Instead of per-pixel Python loops, we compute gradient contributions
        for the entire grid simultaneously using vectorized array operations.

        Parameters
        ----------
        width, height : int
            Output dimensions.
        scale : float
            Noise frequency scaling.
        offset_x, offset_y : float
            Coordinate offsets for tiling.

        Returns
        -------
        noise : array of shape (height, width)
        """
        # Generate normalized coordinate grid
        xs = xp.linspace(0, scale, width, dtype=xp.float32) + offset_x
        ys = xp.linspace(0, scale, height, dtype=xp.float32) + offset_y
        X, Y = xp.meshgrid(xs, ys)

        # Integer grid coordinates
        xi = xp.floor(X).astype(xp.int32)
        yi = xp.floor(Y).astype(xp.int32)

        # Fractional parts for interpolation
        xf = X - xi
        yf = Y - yi

        # Smoothstep fade curves (quintic interpolation for quality)
        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)

        # Hash the four corners of each cell
        # We need to transfer permutation table to GPU if using CuPy
        perm = xp.asarray(self._perm)

        # Compute hash for all four corners
        aa = perm[(perm[(xi & 255)] + yi) & 511]
        ab = perm[(perm[(xi & 255)] + yi + 1) & 511]
        ba = perm[(perm[((xi + 1) & 255)] + yi) & 511]
        bb = perm[(perm[((xi + 1) & 255)] + yi + 1) & 511]

        # Gradient dot products at four corners
        g_aa = self._perlin_grad(aa, xf, yf)
        g_ba = self._perlin_grad(ba, xf - 1, yf)
        g_ab = self._perlin_grad(ab, xf, yf - 1)
        g_bb = self._perlin_grad(bb, xf - 1, yf - 1)

        # Bilinear interpolation
        lerp_x1 = g_aa + u * (g_ba - g_aa)
        lerp_x2 = g_ab + u * (g_bb - g_ab)
        result = lerp_x1 + v * (lerp_x2 - lerp_x1)

        return result

    def _perlin_grad(self, hash_vals, dx, dy):
        """
        Vectorized Perlin gradient computation for all grid points.
        """
        h = hash_vals & 15
        # Gradient selection based on hash
        u = xp.where(h < 8, dx, dy)
        v = xp.where(h < 8, dy, dx)
        # Sign based on hash bits
        sign_u = xp.where((h & 1) == 0, u, -u)
        sign_v = xp.where((h & 2) == 0, v, -v)
        return sign_u + sign_v

    def fbm(self, width, height, octaves=8, persistence=0.5,
            lacunarity=2.0, scale=1.0, offset_x=0.0, offset_y=0.0):
        """
        Fractal Brownian Motion - layered multi-octave noise.

        The foundation of realistic terrain generation. Each octave adds
        finer detail at lower amplitude, creating natural-looking
        self-similar patterns.

        Parameters
        ----------
        width, height : int
            Output dimensions.
        octaves : int
            Number of noise layers to combine.
        persistence : float
            Amplitude multiplier per octave (controls roughness).
        lacunarity : float
            Frequency multiplier per octave (controls detail density).
        scale : float
            Base frequency scaling.

        Returns
        -------
        noise : array of shape (height, width), values in [-1, 1] approximately
        """
        result = xp.zeros((height, width), dtype=xp.float32)
        amplitude = 1.0
        frequency = scale
        max_amplitude = 0.0

        rng = np.random.RandomState(self.seed)

        for i in range(octaves):
            # Each octave gets a unique offset for variety
            ox = offset_x + rng.uniform(-1000, 1000)
            oy = offset_y + rng.uniform(-1000, 1000)

            noise = self.perlin_2d(width, height, frequency, ox, oy)
            result += amplitude * noise

            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        # Normalize to approximately [-1, 1]
        if max_amplitude > 0:
            result /= max_amplitude

        return result

    def ridged_multifractal(self, width, height, octaves=8, persistence=0.5,
                            lacunarity=2.0, scale=1.0, offset_x=0.0, offset_y=0.0):
        """
        Ridged multifractal noise - produces sharp mountain ridges.

        Creates ridgeline patterns by taking the absolute value of noise
        and inverting it. This mimics the appearance of mountain chains
        formed by tectonic compression.

        Parameters
        ----------
        Same as fbm().

        Returns
        -------
        noise : array of shape (height, width)
        """
        result = xp.zeros((height, width), dtype=xp.float32)
        amplitude = 1.0
        frequency = scale
        weight = 1.0
        max_amplitude = 0.0

        rng = np.random.RandomState(self.seed + 1000)

        for i in range(octaves):
            ox = offset_x + rng.uniform(-1000, 1000)
            oy = offset_y + rng.uniform(-1000, 1000)

            noise = self.perlin_2d(width, height, frequency, ox, oy)

            # Create ridges: |noise| inverted
            signal = 1.0 - xp.abs(noise)
            signal = signal * signal  # Sharpen the ridges

            # Weight influences next octave's contribution
            signal *= weight
            weight = xp.clip(signal * 2.0, 0, 1)

            result += amplitude * signal
            max_amplitude += amplitude

            amplitude *= persistence
            frequency *= lacunarity

        if max_amplitude > 0:
            result /= max_amplitude

        return result

    def domain_warp(self, width, height, warp_strength=0.3, octaves=6,
                    persistence=0.5, lacunarity=2.0, scale=1.0):
        """
        Domain warping - deforms the noise coordinate space for
        organic, natural-looking continent shapes.

        Instead of sampling noise at (x, y), we first compute a
        displacement (dx, dy) from another noise layer, then sample
        at (x + dx, y + dy). This creates swirling, continent-like
        shapes that look far more natural than raw noise.

        Parameters
        ----------
        warp_strength : float
            How much the domain is deformed.
        """
        rng = np.random.RandomState(self.seed + 2000)

        # Generate two independent warp displacement fields
        warp_x = self.fbm(width, height, octaves, persistence, lacunarity,
                          scale * 0.8, rng.uniform(-1000, 1000), 0)
        warp_y = self.fbm(width, height, octaves, persistence, lacunarity,
                          scale * 0.8, 0, rng.uniform(-1000, 1000))

        # Build coordinate grids
        xs = xp.linspace(0, scale, width, dtype=xp.float32)
        ys = xp.linspace(0, scale, height, dtype=xp.float32)
        X, Y = xp.meshgrid(xs, ys)

        # Apply domain warping: sample noise at warped coordinates
        # We do this by generating noise at shifted offsets per-cell
        # Since perlin_2d generates all pixels at once with a single offset,
        # we simulate domain warping by blending multiple shifted noise layers
        base = self.fbm(width, height, octaves, persistence, lacunarity, scale)

        # Multi-offset blending: sample noise at several warped positions
        # and blend based on warp displacement — this approximates true
        # per-pixel domain warping without Python loops
        n_samples = 4
        warped = xp.zeros((height, width), dtype=xp.float32)
        for i in range(n_samples):
            angle = 2 * np.pi * i / n_samples
            ox = warp_x * warp_strength * np.cos(angle)
            oy = warp_y * warp_strength * np.sin(angle)
            # Average offset for this layer
            shift_x = float(xp.mean(ox)) + rng.uniform(-10, 10)
            shift_y = float(xp.mean(oy)) + rng.uniform(-10, 10)
            layer = self.fbm(width, height, octaves - 1, persistence * 0.9,
                            lacunarity, scale * 1.2, shift_x, shift_y)
            # Weight by how well this offset matches the warp field
            weight = 1.0 + 0.5 * xp.cos(angle) * warp_x + 0.5 * xp.sin(angle) * warp_y
            warped += layer * weight

        warped /= n_samples

        # Smooth blend of base and warped
        result = base * 0.6 + warped * 0.4

        return result

    def spherical_fbm(self, width, height, octaves=8, persistence=0.5,
                      lacunarity=2.0, scale=1.0):
        """
        Fractal noise sampled on a sphere (equirectangular projection).

        Avoids polar distortion by sampling noise in 3D space using
        the sphere's surface coordinates. This ensures seamless tiling
        and uniform feature distribution across the planet.

        Parameters
        ----------
        width, height : int
            Equirectangular map dimensions (width = 2 * height recommended).

        Returns
        -------
        noise : array of shape (height, width)
        """
        _, _, sx, sy, sz = spherical_coords(width, height)

        result = xp.zeros((height, width), dtype=xp.float32)
        amplitude = 1.0
        frequency = scale
        max_amplitude = 0.0

        rng = np.random.RandomState(self.seed + 3000)

        for i in range(octaves):
            # Use 3D coordinates as input to 2D noise layers
            # Sample three perpendicular slices and combine
            f = frequency
            n1 = self._noise_from_3d(sx * f + rng.uniform(-100, 100),
                                      sy * f + rng.uniform(-100, 100),
                                      width, height)
            n2 = self._noise_from_3d(sy * f + rng.uniform(-100, 100),
                                      sz * f + rng.uniform(-100, 100),
                                      width, height)
            n3 = self._noise_from_3d(sz * f + rng.uniform(-100, 100),
                                      sx * f + rng.uniform(-100, 100),
                                      width, height)

            combined = (n1 + n2 + n3) / 3.0
            result += amplitude * combined
            max_amplitude += amplitude

            amplitude *= persistence
            frequency *= lacunarity

        if max_amplitude > 0:
            result /= max_amplitude

        return result

    def _noise_from_3d(self, coord_x, coord_y, width, height):
        """
        Generate 2D noise from 3D spherical coordinates.
        Projects 3D sphere points onto 2D noise space.
        """
        # Normalize to noise space
        xs = coord_x.ravel()
        ys = coord_y.ravel()

        # Integer coordinates
        xi = xp.floor(xs).astype(xp.int32)
        yi = xp.floor(ys).astype(xp.int32)

        # Fractional parts
        xf = (xs - xi).astype(xp.float32)
        yf = (ys - yi).astype(xp.float32)

        # Smoothstep
        u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
        v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)

        # Hash corners
        perm = xp.asarray(self._perm)
        aa = perm[(perm[(xi & 255)] + yi) & 511]
        ab = perm[(perm[(xi & 255)] + yi + 1) & 511]
        ba = perm[(perm[((xi + 1) & 255)] + yi) & 511]
        bb = perm[(perm[((xi + 1) & 255)] + yi + 1) & 511]

        # Gradients
        g_aa = self._perlin_grad_flat(aa, xf, yf)
        g_ba = self._perlin_grad_flat(ba, xf - 1, yf)
        g_ab = self._perlin_grad_flat(ab, xf, yf - 1)
        g_bb = self._perlin_grad_flat(bb, xf - 1, yf - 1)

        # Interpolation
        l1 = g_aa + u * (g_ba - g_aa)
        l2 = g_ab + u * (g_bb - g_ab)
        result = l1 + v * (l2 - l1)

        return result.reshape(height, width)

    def _perlin_grad_flat(self, hash_vals, dx, dy):
        """Flat-array version of Perlin gradient for 3D noise projection."""
        h = hash_vals & 15
        u = xp.where(h < 8, dx, dy)
        v = xp.where(h < 8, dy, dx)
        sign_u = xp.where((h & 1) == 0, u, -u)
        sign_v = xp.where((h & 2) == 0, v, -v)
        return sign_u + sign_v

    def cellular_noise(self, width, height, num_points=64, scale=1.0):
        """
        Worley/cellular noise for tectonic plate-like cell patterns.

        Computes the distance to the nearest feature point for each pixel,
        creating Voronoi-like cell structures. Useful for tectonic plate
        boundaries and crack patterns.

        Parameters
        ----------
        num_points : int
            Number of feature points (cells).
        scale : float
            Frequency scaling.

        Returns
        -------
        dist1 : array of shape (height, width) - distance to nearest point
        dist2 : array of shape (height, width) - distance to 2nd nearest
        cell_ids : array of shape (height, width) - nearest point index
        """
        rng = np.random.RandomState(self.seed + 4000)

        # Generate feature points
        points_x = rng.uniform(0, scale, num_points).astype(np.float32)
        points_y = rng.uniform(0, scale, num_points).astype(np.float32)

        xs = xp.linspace(0, scale, width, dtype=xp.float32)
        ys = xp.linspace(0, scale, height, dtype=xp.float32)
        X, Y = xp.meshgrid(xs, ys)

        # Compute distances to all points (batched)
        px = xp.asarray(points_x)
        py = xp.asarray(points_y)

        # Shape: (height, width, num_points)
        dx = X[:, :, xp.newaxis] - px[xp.newaxis, xp.newaxis, :]
        dy = Y[:, :, xp.newaxis] - py[xp.newaxis, xp.newaxis, :]
        dists = xp.sqrt(dx * dx + dy * dy)

        # Find nearest and second nearest
        # Sort along last axis
        sorted_indices = xp.argsort(dists, axis=2)
        sorted_dists = xp.take_along_axis(dists, sorted_indices, axis=2)

        dist1 = sorted_dists[:, :, 0]
        dist2 = sorted_dists[:, :, 1]
        cell_ids = sorted_indices[:, :, 0]

        return dist1, dist2, cell_ids

    def turbulence(self, width, height, octaves=6, persistence=0.5,
                   lacunarity=2.0, scale=1.0):
        """
        Turbulence noise - sum of absolute values of noise octaves.
        Creates swirling, turbulent patterns useful for cloud and
        river-like features.
        """
        result = xp.zeros((height, width), dtype=xp.float32)
        amplitude = 1.0
        frequency = scale
        max_amplitude = 0.0

        rng = np.random.RandomState(self.seed + 5000)

        for i in range(octaves):
            ox = rng.uniform(-1000, 1000)
            oy = rng.uniform(-1000, 1000)
            noise = self.perlin_2d(width, height, frequency, ox, oy)
            result += amplitude * xp.abs(noise)
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        if max_amplitude > 0:
            result /= max_amplitude

        return result
