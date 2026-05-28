"""
GPU Backend Abstraction Layer
=============================

Provides a seamless CuPy/NumPy dual backend. All array operations
automatically use GPU when CuPy + CUDA are available, falling back
to NumPy otherwise. This ensures the code is GPU-ready without
requiring a GPU to run.
"""

import numpy as np

# Attempt CuPy import for GPU acceleration
try:
    import cupy as cp
    _gpu_available = True
    xp = cp  # GPU-accelerated array operations
except ImportError:
    _gpu_available = False
    xp = np  # CPU fallback

gpu_available = _gpu_available


def to_gpu(array):
    """Transfer a NumPy array to GPU memory."""
    if gpu_available:
        return cp.asarray(array)
    return array


def to_cpu(array):
    """Transfer an array from GPU to CPU (NumPy)."""
    if gpu_available and hasattr(array, 'get'):
        return array.get()
    return np.asarray(array)


def is_on_gpu(array):
    """Check if an array resides on GPU memory."""
    if gpu_available:
        return isinstance(array, cp.ndarray)
    return False


def ensure_2d(array):
    """Ensure array is 2D, adding dimensions if needed."""
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def grid_coords(width, height, normalized=True):
    """
    Generate a 2D coordinate grid using the active backend.

    Parameters
    ----------
    width, height : int
        Grid dimensions.
    normalized : bool
        If True, coordinates are in [0, 1]. If False, integer indices.

    Returns
    -------
    X, Y : arrays of shape (height, width)
    """
    if normalized:
        xs = xp.linspace(0, 1, width, dtype=xp.float32)
        ys = xp.linspace(0, 1, height, dtype=xp.float32)
    else:
        xs = xp.arange(width, dtype=xp.float32)
        ys = xp.arange(height, dtype=xp.float32)
    X, Y = xp.meshgrid(xs, ys)
    return X, Y


def spherical_coords(width, height):
    """
    Generate spherical coordinate mapping for planet-scale generation.
    Maps a 2D equirectangular grid to 3D sphere coordinates.

    Returns
    -------
    lon, lat : arrays of shape (height, width) in radians
    x, y, z : arrays of shape (height, width) unit sphere coordinates
    """
    lon = xp.linspace(0, 2 * xp.pi, width, dtype=xp.float32)
    lat = xp.linspace(-xp.pi / 2, xp.pi / 2, height, dtype=xp.float32)
    LON, LAT = xp.meshgrid(lon, lat)

    x = xp.cos(LAT) * xp.cos(LON)
    y = xp.cos(LAT) * xp.sin(LON)
    z = xp.sin(LAT)

    return LON, LAT, x, y, z
