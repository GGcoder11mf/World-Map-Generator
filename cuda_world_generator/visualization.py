"""
Visualization & Preview System
================================

Provides real-time and static visualization of generated worlds:

- matplotlib-based static previews (always available)
- Multi-panel overview with all maps
- 3D terrain rendering
- Biome color maps
- River network overlay
- Climate visualization
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm
from .core.backend import to_cpu, gpu_available
from .core.biome import BIOME_COLORS, NUM_BIOMES, BIOMES


def create_biome_colormap():
    """Create a matplotlib ListedColormap for biome visualization."""
    colors = [tuple(c / 255 for c in BIOME_COLORS[i]) for i in range(NUM_BIOMES)]
    return ListedColormap(colors)


def preview_world(world, output_path=None, show_3d=False):
    """
    Generate a comprehensive multi-panel preview of the world.

    Parameters
    ----------
    world : World
        Generated world object.
    output_path : str, optional
        Path to save the preview image. If None, shows interactively.
    show_3d : bool
        Whether to include 3D terrain view.
    """
    if not world._generated:
        raise RuntimeError("World must be generated first.")

    # Transfer all data to CPU
    maps = world.get_all_maps()
    heightmap = maps['heightmap']
    temperature = maps['temperature']
    humidity = maps['humidity']
    rainfall = maps['rainfall']
    rain_shadow = maps.get('rain_shadow', None)
    biome_map = maps['biome_map'].astype(int)
    river_map = maps['river_map']
    vegetation = maps['vegetation_density']
    tree_density = maps['tree_density']
    animal_prob = maps['animal_probability']
    biodiversity = maps['biodiversity']

    sea_level = world.sea_level

    # Configure matplotlib for high-quality output
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # ── Main overview figure ────────────────────────────────────────────
    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    fig.suptitle(
        f'Procedural World Generator — Seed {world.config.seed} — '
        f'{world.config.size}x{world.config.size} — '
        f'GPU: {"Yes" if gpu_available else "No"}',
        fontsize=14, fontweight='bold'
    )

    # 1. Heightmap with terrain colormap
    ax = axes[0, 0]
    im = ax.imshow(heightmap, cmap='terrain', vmin=0, vmax=1)
    ax.set_title('Heightmap (Elevation)')
    ax.contour(heightmap, levels=[sea_level], colors='blue', linewidths=0.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2. Biome map
    ax = axes[0, 1]
    biome_cmap = create_biome_colormap()
    im = ax.imshow(biome_map, cmap=biome_cmap, vmin=0, vmax=NUM_BIOMES - 1)
    ax.set_title('Biome Map')
    # Add compact legend
    unique_biomes = np.unique(biome_map)
    legend_labels = [BIOMES.get(int(b), '?') for b in unique_biomes[:8]]
    ax.set_xlabel(f'{len(unique_biomes)} biomes')

    # 3. Temperature map
    ax = axes[0, 2]
    im = ax.imshow(temperature, cmap='RdYlBu_r', vmin=-30, vmax=40)
    ax.set_title('Temperature (°C)')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 4. Humidity map
    ax = axes[0, 3]
    im = ax.imshow(humidity, cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_title('Humidity')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 5. Rainfall
    ax = axes[1, 0]
    im = ax.imshow(rainfall, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Rainfall / Precipitation')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 6. Heightmap + Rivers overlay
    ax = axes[1, 1]
    ax.imshow(heightmap, cmap='terrain', vmin=0, vmax=1, alpha=0.7)
    river_overlay = np.where(river_map > 0.01, 1, 0).astype(float)
    ax.imshow(river_overlay, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    ax.set_title('Rivers (overlay on terrain)')

    # 7. Vegetation density
    ax = axes[1, 2]
    im = ax.imshow(vegetation, cmap='YlGn', vmin=0, vmax=1)
    ax.set_title('Vegetation Density')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 8. Tree density
    ax = axes[1, 3]
    im = ax.imshow(tree_density, cmap='Greens', vmin=0, vmax=1)
    ax.set_title('Tree Density')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 9. Animal probability
    ax = axes[2, 0]
    im = ax.imshow(animal_prob, cmap='OrRd', vmin=0, vmax=1)
    ax.set_title('Animal Habitat Suitability')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 10. Biodiversity
    ax = axes[2, 1]
    im = ax.imshow(biodiversity, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Biodiversity Index')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 11. Rain shadow map (orographic effect visualization)
    ax = axes[2, 2]
    if rain_shadow is not None:
        # Custom colormap: green (wet) → yellow → red (dry/desert)
        from matplotlib.colors import LinearSegmentedColormap
        rs_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8b4513']
        rs_cmap = LinearSegmentedColormap.from_list('rain_shadow', rs_colors)
        im = ax.imshow(rain_shadow, cmap=rs_cmap, vmin=0, vmax=1)
        ax.set_title('Rain Shadow (Orographic Drying)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        # Fallback: slope map
        padded = np.pad(heightmap, 1, mode='edge')
        dz_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / 2
        dz_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / 2
        slope = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        im = ax.imshow(slope, cmap='magma', vmin=0)
        ax.set_title('Terrain Slope')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 12. World statistics text
    ax = axes[2, 3]
    stats = world.get_world_stats()
    ax.axis('off')
    stats_text = (
        f"World Statistics\n"
        f"{'─' * 30}\n"
        f"Seed: {stats['seed']}\n"
        f"Resolution: {stats['resolution']}x{stats['resolution']}\n"
        f"Generation: {stats['generation_time_s']:.1f}s\n"
        f"GPU: {'Yes' if stats['gpu_accelerated'] else 'No'}\n"
        f"{'─' * 30}\n"
        f"Land: {stats['land_fraction']:.1%}\n"
        f"Ocean: {stats['ocean_fraction']:.1%}\n"
        f"Max Elevation: {stats['max_elevation']:.1f} km\n"
        f"{'─' * 30}\n"
        f"Mean Temp: {stats['mean_temperature']:.1f}°C\n"
        f"Temp Range: {stats['min_temperature']:.0f} to {stats['max_temperature']:.0f}°C\n"
        f"{'─' * 30}\n"
        f"Biomes: {stats['num_biomes']}\n"
        f"River Nodes: {stats['num_river_nodes']}\n"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title('World Statistics')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"[Preview] Saved to {output_path}")
    plt.close(fig)

    # ── 3D terrain view ────────────────────────────────────────────────
    if show_3d:
        _render_3d_terrain(heightmap, sea_level, world.config.seed, output_path)


def _render_3d_terrain(heightmap, sea_level, seed, output_path=None):
    """Render a 3D perspective view of the terrain."""
    # Downsample for 3D rendering performance
    step = max(1, heightmap.shape[0] // 256)
    h = heightmap[::step, ::step]
    size = h.shape[0]

    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Color by elevation + sea
    colors = cm.terrain((h - h.min()) / (h.max() - h.min() + 1e-10))
    # Ocean areas: blue tint
    ocean_mask = h < sea_level
    colors[ocean_mask] = [0.1, 0.2, 0.5, 1.0]

    surf = ax.plot_surface(X, Y, h, facecolors=colors,
                           rstride=1, cstride=1,
                           antialiased=False, shade=True)

    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title(f'3D Terrain View — Seed {seed}')
    ax.view_init(elev=35, azim=225)

    if output_path:
        base = output_path.rsplit('.', 1)[0]
        path_3d = f"{base}_3d.png"
        plt.savefig(path_3d, dpi=120, bbox_inches='tight')
        print(f"[Preview] 3D view saved to {path_3d}")
    plt.close(fig)


def preview_single_map(data, title="Map", cmap='viridis', output_path=None,
                        vmin=None, vmax=None):
    """
    Preview a single map with colorbar.

    Parameters
    ----------
    data : array
        2D array to visualize.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    output_path : str, optional
        Path to save the image.
    vmin, vmax : float, optional
        Color scale limits.
    """
    data_cpu = to_cpu(data) if hasattr(data, 'get') else np.asarray(data)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_cpu, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Preview] Saved to {output_path}")
    plt.close(fig)
