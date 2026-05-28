#!/usr/bin/env python3
"""
CUDA Physically Accurate Procedural World Generator — Demo
============================================================

This demo script generates a complete Earth-like world and
exports all maps as images and data files.

Usage:
    python demo.py [--seed SEED] [--size SIZE] [--output DIR] [--3d]

Examples:
    python demo.py                      # Default: seed=42, size=512
    python demo.py --seed 12345         # Custom seed
    python demo.py --size 1024          # Higher resolution
    python demo.py --3d                 # Include 3D terrain view
"""

import argparse
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cuda_world_generator import World, WorldConfig
from cuda_world_generator.visualization import preview_world, preview_single_map
from cuda_world_generator.export import (
    export_heightmap_png,
    export_biome_map_png,
    export_obj_mesh,
    export_metadata
)
from cuda_world_generator.core.backend import gpu_available


def main():
    parser = argparse.ArgumentParser(
        description='CUDA Physically Accurate Procedural World Generator'
    )
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic generation')
    parser.add_argument('--size', type=int, default=512,
                        help='Heightmap resolution (power of 2)')
    parser.add_argument('--output', type=str, default='./world_output',
                        help='Output directory')
    parser.add_argument('--3d', action='store_true',
                        help='Include 3D terrain rendering')
    parser.add_argument('--quick', action='store_true',
                        help='Quick generation (reduced iterations)')
    parser.add_argument('--obj', action='store_true',
                        help='Export OBJ mesh')
    args = parser.parse_args()

    # Configure world generation
    config = WorldConfig(
        seed=args.seed,
        size=args.size,
        output_dir=args.output,
    )

    # Quick mode: reduce iterations for faster generation
    if args.quick:
        config.hydraulic_erosion_iterations = 10
        config.thermal_erosion_iterations = 5
        config.river_iterations = 30
        config.tectonic_iterations = 20

    print("=" * 60)
    print("  CUDA Physically Accurate Procedural World Generator")
    print("=" * 60)
    print(f"  Seed:      {args.seed}")
    print(f"  Size:      {args.size}x{args.size}")
    print(f"  GPU:       {'CuPy (CUDA)' if gpu_available else 'NumPy (CPU fallback)'}")
    print(f"  Output:    {args.output}")
    print(f"  Quick:     {args.quick}")
    print("=" * 60)

    # Generate world
    world = World(config)
    world.generate()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # ── Generate preview images ─────────────────────────────────────
    print("\n[Preview] Generating overview...")

    preview_path = os.path.join(args.output, f"world_overview_seed{args.seed}.png")
    preview_world(world, output_path=preview_path, show_3d=args.__dict__['3d'])

    # ── Export individual maps ──────────────────────────────────────
    print("\n[Export] Saving individual maps...")

    export_heightmap_png(
        world.heightmap,
        os.path.join(args.output, f"heightmap_seed{args.seed}.png"),
        sea_level=world.sea_level
    )

    export_biome_map_png(
        world.biome_map,
        os.path.join(args.output, f"biome_map_seed{args.seed}.png")
    )

    # Export climate maps
    for name, cmap in [
        ('temperature', 'RdYlBu_r'),
        ('humidity', 'YlGnBu'),
        ('rainfall', 'Blues'),
        ('vegetation', 'YlGn'),
        ('tree_density', 'Greens'),
        ('biodiversity', 'viridis'),
    ]:
        data = getattr(world, name, None)
        if data is not None:
            preview_single_map(
                data, title=name.replace('_', ' ').title(),
                cmap=cmap,
                output_path=os.path.join(args.output, f"{name}_seed{args.seed}.png")
            )

    # ── Export data files ───────────────────────────────────────────
    print("\n[Export] Saving data files...")

    files = world.export_terrain(args.output)

    # Export metadata
    export_metadata(
        world,
        os.path.join(args.output, f"world_metadata_seed{args.seed}.json")
    )

    # Export OBJ mesh if requested
    if args.obj:
        export_obj_mesh(
            world.heightmap,
            os.path.join(args.output, f"terrain_seed{args.seed}.obj"),
            sea_level=world.sea_level
        )

    # ── Print world statistics ──────────────────────────────────────
    stats = world.get_world_stats()
    print("\n" + "=" * 60)
    print("  World Generation Complete!")
    print("=" * 60)
    print(f"  Land/Ocean:    {stats['land_fraction']:.1%} / {stats['ocean_fraction']:.1%}")
    print(f"  Max Elevation: {stats['max_elevation']:.1f} km")
    print(f"  Temperature:   {stats['min_temperature']:.0f}°C to {stats['max_temperature']:.0f}°C")
    print(f"  Mean Temp:     {stats['mean_temperature']:.1f}°C")
    print(f"  Biomes:        {stats['num_biomes']}")
    print(f"  Rivers:        {stats['num_river_nodes']} nodes, {stats['num_river_edges']} segments")
    print(f"  Gen Time:      {stats['generation_time_s']:.2f}s")
    print(f"  GPU:           {'Yes' if stats['gpu_accelerated'] else 'No (CPU)'}")
    print(f"  Output:        {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
