"""
Export Utilities
=================

Functions for exporting world data in various formats:
- PNG images (heightmap, biome map, etc.)
- NPZ compressed archive (all maps)
- OBJ mesh (3D terrain)
- JSON metadata
"""

import numpy as np
import os
import json
from .core.backend import to_cpu, gpu_available
from .core.biome import BIOMES, BIOME_COLORS, NUM_BIOMES


def export_heightmap_png(heightmap, output_path, sea_level=0.35):
    """Export heightmap as a PNG image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h = to_cpu(heightmap)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(h, cmap='terrain', vmin=0, vmax=1)
    ax.contour(h, levels=[sea_level], colors='blue', linewidths=1)
    ax.set_title('Heightmap')
    plt.colorbar(im, ax=ax, label='Elevation')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def export_biome_map_png(biome_map, output_path):
    """Export biome map as a colored PNG image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    b = to_cpu(biome_map).astype(int)
    colors = [tuple(c / 255 for c in BIOME_COLORS[i]) for i in range(NUM_BIOMES)]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(b, cmap=cmap, vmin=0, vmax=NUM_BIOMES - 1)
    ax.set_title('Biome Map')

    # Add legend
    unique_biomes = np.unique(b)
    patches = []
    import matplotlib.patches as mpatches
    for bid in unique_biomes:
        bid = int(bid)
        color = tuple(c / 255 for c in BIOME_COLORS[bid])
        patches.append(mpatches.Patch(color=color, label=BIOMES.get(bid, '?')))
    ax.legend(handles=patches, loc='lower right', fontsize=7, ncol=2)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def export_obj_mesh(heightmap, output_path, scale=1.0, sea_level=0.35):
    """
    Export terrain as Wavefront OBJ mesh.

    Parameters
    ----------
    heightmap : array
        2D heightmap data.
    output_path : str
        Path for .obj file.
    scale : float
        Vertical scale factor.
    sea_level : float
        Sea level for reference.
    """
    h = to_cpu(heightmap)
    rows, cols = h.shape

    # Downsample if too large for OBJ
    step = max(1, max(rows, cols) // 512)
    h = h[::step, ::step]
    rows, cols = h.shape

    vertices = []
    faces = []
    normals = []

    # Generate vertices
    for y in range(rows):
        for x in range(cols):
            z = float(h[y, x]) * scale
            vertices.append(f"v {x / cols:.6f} {z:.6f} {y / rows:.6f}")

    # Generate faces (two triangles per grid cell)
    for y in range(rows - 1):
        for x in range(cols - 1):
            v1 = y * cols + x + 1      # 1-indexed
            v2 = y * cols + x + 2
            v3 = (y + 1) * cols + x + 1
            v4 = (y + 1) * cols + x + 2
            faces.append(f"f {v1} {v2} {v4}")
            faces.append(f"f {v1} {v4} {v3}")

    with open(output_path, 'w') as f:
        f.write("# Procedural World Generator - OBJ Export\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        f.write('\n'.join(vertices))
        f.write('\n\n')
        f.write('\n'.join(faces))

    print(f"[Export] OBJ mesh saved to {output_path} ({len(vertices)} vertices)")


def export_metadata(world, output_path):
    """
    Export world metadata as JSON.

    Parameters
    ----------
    world : World
        Generated world object.
    output_path : str
        Path for .json file.
    """
    stats = world.get_world_stats()
    stats['gpu_accelerated'] = bool(stats['gpu_accelerated'])
    stats['biome_list'] = list(stats['biome_list'])

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"[Export] Metadata saved to {output_path}")
