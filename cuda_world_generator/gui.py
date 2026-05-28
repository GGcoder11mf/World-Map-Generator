#!/usr/bin/env python3
"""
CUDA World Generator — Tkinter GUI
====================================

A simple graphical interface for the procedural world generator.
Features:
- Seed & size controls
- One-click world generation
- Map selector (heightmap, biome, temperature, etc.)
- Zoom & pan
- World statistics panel
- Export to files

Usage:
    python gui.py
"""

import sys
import os
import io
import time
import threading

# Add parent directory to path so the package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

from cuda_world_generator import World, WorldConfig
from cuda_world_generator.core.backend import to_cpu, gpu_available
from cuda_world_generator.core.biome import BIOMES, BIOME_COLORS, NUM_BIOMES


# ── Color maps (implemented in pure NumPy for speed) ──────────────────

def colormap_terrain(data):
    """Terrain colormap: ocean blues → green lowlands → brown highlands → white peaks."""
    h, w = data.shape
    r = np.zeros_like(data, dtype=np.uint8)
    g = np.zeros_like(data, dtype=np.uint8)
    b = np.zeros_like(data, dtype=np.uint8)

    # Deep ocean
    mask = data < 0.15
    r[mask] = 10;  g[mask] = 30;  b[mask] = 80

    # Ocean
    mask = (data >= 0.15) & (data < 0.30)
    t = (data[mask] - 0.15) / 0.15
    r[mask] = (10 + t * 20).astype(np.uint8)
    g[mask] = (30 + t * 40).astype(np.uint8)
    b[mask] = (80 + t * 60).astype(np.uint8)

    # Shallow water / beach
    mask = (data >= 0.30) & (data < 0.36)
    t = (data[mask] - 0.30) / 0.06
    r[mask] = (30 + t * 180).astype(np.uint8)
    g[mask] = (70 + t * 130).astype(np.uint8)
    b[mask] = (140 - t * 0).astype(np.uint8)

    # Lowlands (green)
    mask = (data >= 0.36) & (data < 0.55)
    t = (data[mask] - 0.36) / 0.19
    r[mask] = (50 + t * 40).astype(np.uint8)
    g[mask] = (120 + t * 20).astype(np.uint8)
    b[mask] = (40 + t * 10).astype(np.uint8)

    # Hills (brown-green)
    mask = (data >= 0.55) & (data < 0.72)
    t = (data[mask] - 0.55) / 0.17
    r[mask] = (90 + t * 70).astype(np.uint8)
    g[mask] = (140 - t * 40).astype(np.uint8)
    b[mask] = (50 - t * 10).astype(np.uint8)

    # Mountains (gray-brown)
    mask = (data >= 0.72) & (data < 0.88)
    t = (data[mask] - 0.72) / 0.16
    r[mask] = (160 + t * 40).astype(np.uint8)
    g[mask] = (100 + t * 40).astype(np.uint8)
    b[mask] = (40 + t * 40).astype(np.uint8)

    # Peaks (white)
    mask = data >= 0.88
    t = np.minimum((data[mask] - 0.88) / 0.12, 1.0)
    r[mask] = (200 + t * 55).astype(np.uint8)
    g[mask] = (140 + t * 115).astype(np.uint8)
    b[mask] = (80 + t * 175).astype(np.uint8)

    return np.stack([r, g, b], axis=2)


def colormap_biome(biome_map):
    """Direct biome ID → RGB color lookup."""
    h, w = biome_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for bid in range(NUM_BIOMES):
        mask = biome_map == bid
        if np.any(mask):
            rgb[mask] = BIOME_COLORS.get(bid, (128, 128, 128))
    return rgb


def colormap_heat(data, vmin=None, vmax=None):
    """Blue → Cyan → Green → Yellow → Red heatmap."""
    if vmin is None: vmin = data.min()
    if vmax is None: vmax = data.max()
    if vmax <= vmin: vmax = vmin + 1
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    h, w = data.shape
    r = np.zeros((h, w), dtype=np.uint8)
    g = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)

    # Blue → Cyan (0.0 – 0.25)
    m = norm < 0.25
    t = norm[m] * 4
    r[m] = 0; g[m] = (t * 255).astype(np.uint8); b[m] = 255

    # Cyan → Green (0.25 – 0.5)
    m = (norm >= 0.25) & (norm < 0.5)
    t = (norm[m] - 0.25) * 4
    r[m] = 0; g[m] = 255; b[m] = ((1 - t) * 255).astype(np.uint8)

    # Green → Yellow (0.5 – 0.75)
    m = (norm >= 0.5) & (norm < 0.75)
    t = (norm[m] - 0.5) * 4
    r[m] = (t * 255).astype(np.uint8); g[m] = 255; b[m] = 0

    # Yellow → Red (0.75 – 1.0)
    m = norm >= 0.75
    t = (norm[m] - 0.75) * 4
    r[m] = 255; g[m] = ((1 - t) * 255).astype(np.uint8); b[m] = 0

    return np.stack([r, g, b], axis=2)


def colormap_blues(data, vmin=None, vmax=None):
    """White → Blue sequential colormap."""
    if vmin is None: vmin = data.min()
    if vmax is None: vmax = data.max()
    if vmax <= vmin: vmax = vmin + 1
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    r = ((1 - norm) * 240 + norm * 10).astype(np.uint8)
    g = ((1 - norm) * 240 + norm * 50).astype(np.uint8)
    b = ((1 - norm) * 255 + norm * 180).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def colormap_greens(data, vmin=None, vmax=None):
    """Black → Green sequential colormap."""
    if vmin is None: vmin = data.min()
    if vmax is None: vmax = data.max()
    if vmax <= vmin: vmax = vmin + 1
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    r = (norm * 50).astype(np.uint8)
    g = (norm * 200 + 20).astype(np.uint8)
    b = (norm * 30).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def colormap_rain_shadow(data, vmin=None, vmax=None):
    """
    Rain shadow colormap: green (wet) → yellow → orange → red (dry/desert).

    Shows the contrast between windward (wet, green) and leeward (dry, red)
    sides of mountain ranges.
    """
    if vmin is None: vmin = data.min()
    if vmax is None: vmax = data.max()
    if vmax <= vmin: vmax = vmin + 1
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    h, w = data.shape
    r = np.zeros((h, w), dtype=np.uint8)
    g = np.zeros((h, w), dtype=np.uint8)
    b = np.zeros((h, w), dtype=np.uint8)

    # Green (no shadow / wet) → Yellow (mild shadow)
    m = norm < 0.33
    t = norm[m] / 0.33
    r[m] = (t * 255).astype(np.uint8)
    g[m] = 200
    b[m] = 0

    # Yellow → Orange (moderate shadow)
    m = (norm >= 0.33) & (norm < 0.66)
    t = (norm[m] - 0.33) / 0.33
    r[m] = 255
    g[m] = (200 - t * 120).astype(np.uint8)
    b[m] = 0

    # Orange → Red/Brown (strong shadow / desert)
    m = norm >= 0.66
    t = (norm[m] - 0.66) / 0.34
    r[m] = (255 - t * 55).astype(np.uint8)
    g[m] = (80 - t * 60).astype(np.uint8)
    b[m] = (t * 30).astype(np.uint8)

    return np.stack([r, g, b], axis=2)


# ── Map definitions ────────────────────────────────────────────────────

MAP_DEFS = [
    ("Heightmap",       "heightmap",             colormap_terrain, None),
    ("Biome Map",       "biome_map",             colormap_biome,   None),
    ("Temperature °C",  "temperature",           colormap_heat,    (-30, 35)),
    ("Humidity",        "humidity",              colormap_blues,   (0, 1)),
    ("Rainfall",        "rainfall",              colormap_blues,   (0, 1)),
    ("Rain Shadow",     "rain_shadow",           colormap_rain_shadow, (0, 1)),
    ("Rivers",          "river_map",             colormap_blues,   (0, None)),
    ("Vegetation",      "vegetation_density",    colormap_greens,  (0, 1)),
    ("Tree Density",    "tree_density",          colormap_greens,  (0, 1)),
    ("Animals",         "animal_probability",    colormap_heat,    (0, 1)),
    ("Biodiversity",    "biodiversity",          colormap_greens,  (0, 1)),
    ("Soil Fertility",  "soil_fertility",        colormap_greens,  (0, 1)),
    ("Flow Accum.",     "flow_accumulation",     colormap_blues,   (0, None)),
]


# ── Main GUI class ─────────────────────────────────────────────────────

class WorldGeneratorGUI:
    """Tkinter GUI for the CUDA Procedural World Generator."""

    def __init__(self, root):
        self.root = root
        self.root.title("🌍 CUDA World Generator")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)

        # State
        self.world = None
        self.current_map_name = "Heightmap"
        self.display_image = None       # Current PhotoImage
        self.map_images = {}            # Cached map PhotoImages
        self.map_data_cache = {}        # Cached NumPy arrays
        self.is_generating = False
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._drag_start = None

        self._build_ui()

    # ── UI Construction ────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel",   font=("Segoe UI", 16, "bold"), background="#1a1a2e", foreground="#e0e0ff")
        style.configure("Info.TLabel",    font=("Segoe UI", 10),         background="#1a1a2e", foreground="#b0b0d0")
        style.configure("Stat.TLabel",    font=("Consolas", 9),          background="#1a1a2e", foreground="#80ffb0")
        style.configure("Dark.TFrame",    background="#1a1a2e")
        style.configure("Card.TFrame",    background="#16213e")
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Map.TButton",    font=("Segoe UI", 9),         padding=4)
        style.configure("Dark.TLabelframe",       background="#16213e", foreground="#e0e0ff")
        style.configure("Dark.TLabelframe.Label",  background="#16213e", foreground="#e0e0ff",
                         font=("Segoe UI", 10, "bold"))

        # ── Left panel (controls) ──────────────────────────────────────
        left = ttk.Frame(self.root, style="Dark.TFrame", width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0), pady=8)
        left.pack_propagate(False)

        ttk.Label(left, text="🌍 World Generator", style="Title.TLabel").pack(pady=(12, 4))
        gpu_text = "GPU: CUDA ✓" if gpu_available else "GPU: CPU mode"
        ttk.Label(left, text=gpu_text, style="Info.TLabel").pack(pady=(0, 12))

        # ── Parameters ─────────────────────────────────────────────────
        params = ttk.LabelFrame(left, text="Parameters", style="Dark.TLabelframe", padding=10)
        params.pack(fill=tk.X, padx=8, pady=4)

        # Seed
        f = ttk.Frame(params, style="Dark.TFrame")
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="Seed:", style="Info.TLabel", width=10).pack(side=tk.LEFT)
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(f, from_=0, to=99999, textvariable=self.seed_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(f, text="🎲", width=3, command=self._random_seed).pack(side=tk.LEFT)

        # Size
        f = ttk.Frame(params, style="Dark.TFrame")
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="Size:", style="Info.TLabel", width=10).pack(side=tk.LEFT)
        self.size_var = tk.IntVar(value=512)
        size_combo = ttk.Combobox(f, textvariable=self.size_var, values=[128, 256, 512, 1024],
                                   width=8, state="readonly")
        size_combo.pack(side=tk.LEFT, padx=4)

        # Erosion iterations
        f = ttk.Frame(params, style="Dark.TFrame")
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="Erosion:", style="Info.TLabel", width=10).pack(side=tk.LEFT)
        self.erosion_var = tk.IntVar(value=20)
        ttk.Scale(f, from_=5, to=60, variable=self.erosion_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Tectonic plates
        f = ttk.Frame(params, style="Dark.TFrame")
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="Plates:", style="Info.TLabel", width=10).pack(side=tk.LEFT)
        self.plates_var = tk.IntVar(value=12)
        ttk.Scale(f, from_=4, to=24, variable=self.plates_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Sea level
        f = ttk.Frame(params, style="Dark.TFrame")
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="Sea Lvl:", style="Info.TLabel", width=10).pack(side=tk.LEFT)
        self.sealevel_var = tk.DoubleVar(value=0.35)
        ttk.Scale(f, from_=0.1, to=0.6, variable=self.sealevel_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── Generate button ────────────────────────────────────────────
        self.gen_btn = ttk.Button(left, text="⚡ Generate World", style="Accent.TButton",
                                   command=self._generate_world)
        self.gen_btn.pack(fill=tk.X, padx=12, pady=(16, 4))

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=12, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, style="Info.TLabel").pack(pady=4)

        # ── Map selector ───────────────────────────────────────────────
        maps_frame = ttk.LabelFrame(left, text="Map View", style="Dark.TLabelframe", padding=6)
        maps_frame.pack(fill=tk.X, padx=8, pady=8)

        self.map_var = tk.StringVar(value="Heightmap")
        for name, _, _, _ in MAP_DEFS:
            ttk.Radiobutton(maps_frame, text=name, variable=self.map_var,
                            value=name, command=self._switch_map).pack(anchor=tk.W, pady=1)

        # ── Export button ──────────────────────────────────────────────
        ttk.Button(left, text="💾 Export All Maps", command=self._export).pack(fill=tk.X, padx=12, pady=(8, 4))

        # ── Statistics ─────────────────────────────────────────────────
        stats_frame = ttk.LabelFrame(left, text="Statistics", style="Dark.TLabelframe", padding=6)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self.stats_text = tk.Text(stats_frame, bg="#0f0f23", fg="#80ffb0", font=("Consolas", 9),
                                   relief=tk.FLAT, wrap=tk.WORD, height=10)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.insert("1.0", "No world generated yet.\nClick 'Generate World' to begin.")
        self.stats_text.config(state=tk.DISABLED)

        # ── Right panel (map display) ──────────────────────────────────
        right = ttk.Frame(self.root, style="Card.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Canvas header
        hdr = ttk.Frame(right, style="Card.TFrame")
        hdr.pack(fill=tk.X, padx=4, pady=(4, 0))
        self.map_label_var = tk.StringVar(value="Heightmap")
        ttk.Label(hdr, textvariable=self.map_label_var, style="Title.TLabel",
                  font=("Segoe UI", 13, "bold")).pack(side=tk.LEFT, padx=8)
        ttk.Label(hdr, text="Scroll=Zoom  Drag=Pan  DblClick=Reset", style="Info.TLabel").pack(side=tk.RIGHT, padx=8)

        # Canvas
        self.canvas = tk.Canvas(right, bg="#0a0a1a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Canvas events
        self.canvas.bind("<MouseWheel>", self._on_zoom)        # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_zoom)          # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_zoom)          # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Double-Button-1>", self._on_reset_view)

    # ── Actions ────────────────────────────────────────────────────────

    def _random_seed(self):
        self.seed_var.set(np.random.randint(0, 99999))

    def _generate_world(self):
        if self.is_generating:
            return
        self.is_generating = True
        self.gen_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_var.set("Generating...")

        # Run generation in a background thread to keep UI responsive
        thread = threading.Thread(target=self._do_generate, daemon=True)
        thread.start()

    def _do_generate(self):
        try:
            seed = self.seed_var.get()
            size = self.size_var.get()
            erosion_iter = int(self.erosion_var.get())
            num_plates = int(self.plates_var.get())
            sea_level = self.sealevel_var.get()

            config = WorldConfig(seed=seed, size=size, num_plates=num_plates, sea_level=sea_level)
            config.hydraulic_erosion_iterations = erosion_iter
            config.thermal_erosion_iterations = max(erosion_iter // 3, 3)
            config.river_iterations = max(erosion_iter * 2, 20)
            config.lake_fill_iterations = max(erosion_iter, 10)

            self.world = World(config)

            # Schedule status updates during generation
            self.root.after(0, lambda: self.status_var.set("Generating terrain..."))

            t0 = time.time()
            self.world.generate()
            elapsed = time.time() - t0

            # Cache all map arrays (CPU)
            self.map_data_cache = {
                key: to_cpu(getattr(self.world, key))
                for key in ["heightmap", "biome_map", "temperature", "humidity",
                            "rainfall", "rain_shadow", "river_map", "vegetation_density",
                            "tree_density", "animal_probability", "biodiversity",
                            "soil_fertility", "flow_accumulation"]
                if getattr(self.world, key, None) is not None
            }

            # Render all maps to images
            self.map_images = {}
            for name, key, cmap_fn, crange in MAP_DEFS:
                if key in self.map_data_cache:
                    data = self.map_data_cache[key]
                    if key == "biome_map":
                        rgb = cmap_fn(data.astype(int))
                    elif crange is None:
                        # Colormap takes only data (no vmin/vmax)
                        rgb = cmap_fn(data)
                    else:
                        vmin = crange[0] if crange[0] is not None else None
                        vmax = crange[1] if crange[1] is not None else None
                        rgb = cmap_fn(data, vmin, vmax)
                    pil_img = Image.fromarray(rgb, "RGB")
                    self.map_images[name] = pil_img

            # Update stats
            stats = self.world.get_world_stats()
            stats_lines = [
                f"Seed:         {stats['seed']}",
                f"Resolution:   {stats['resolution']}×{stats['resolution']}",
                f"Gen Time:     {elapsed:.1f}s",
                f"GPU:          {'Yes' if stats['gpu_accelerated'] else 'No (CPU)'}",
                f"{'─' * 28}",
                f"Land:         {stats['land_fraction']:.1%}",
                f"Ocean:        {stats['ocean_fraction']:.1%}",
                f"Max Elev:     {stats['max_elevation']:.1f} km",
                f"{'─' * 28}",
                f"Mean Temp:    {stats['mean_temperature']:.1f}°C",
                f"Temp Range:   {stats['min_temperature']:.0f} to {stats['max_temperature']:.0f}°C",
                f"{'─' * 28}",
                f"Biomes:       {stats['num_biomes']}",
                f"Rivers:       {stats['num_river_nodes']} nodes",
                f"{'─' * 28}",
                f"Biomelist:",
            ]
            for bname in stats['biome_list']:
                stats_lines.append(f"  • {bname}")

            self.root.after(0, self._update_stats, stats_lines, elapsed)

        except Exception as exc:
            err_msg = str(exc)
            self.root.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))
        finally:
            self.root.after(0, self._generation_done)

    def _update_stats(self, lines, elapsed):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert("1.0", "\n".join(lines))
        self.stats_text.config(state=tk.DISABLED)
        self.status_var.set(f"Done in {elapsed:.1f}s")
        # Show the current map
        self._render_map()

    def _generation_done(self):
        self.is_generating = False
        self.gen_btn.config(state=tk.NORMAL)
        self.progress.stop()

    def _switch_map(self):
        self.current_map_name = self.map_var.get()
        self.map_label_var.set(self.current_map_name)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render_map()

    def _render_map(self):
        name = self.current_map_name
        if name not in self.map_images:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 400,
                self.canvas.winfo_height() // 2 or 300,
                text="No map data.\nGenerate a world first.", fill="#606080",
                font=("Segoe UI", 14), justify=tk.CENTER
            )
            return

        pil_img = self.map_images[name]

        # Get canvas size
        self.canvas.update_idletasks()
        cw = max(self.canvas.winfo_width(), 200)
        ch = max(self.canvas.winfo_height(), 200)

        # Fit image to canvas
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih) * self.zoom
        new_w = max(int(iw * scale), 1)
        new_h = max(int(ih * scale), 1)

        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self.display_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x = (cw - new_w) // 2 + self.pan_x
        y = (ch - new_h) // 2 + self.pan_y
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.display_image)

        # Draw biome legend if biome map
        if name == "Biome Map" and self.world:
            self._draw_biome_legend(cw)

    def _draw_biome_legend(self, canvas_width):
        """Draw a compact biome color legend on the canvas."""
        if not self.world:
            return
        biome_map = self.map_data_cache.get("biome_map")
        if biome_map is None:
            return

        unique_ids = np.unique(biome_map.astype(int))
        x0 = canvas_width - 170
        y0 = 10

        # Background
        self.canvas.create_rectangle(x0 - 8, y0 - 4, x0 + 162, y0 + len(unique_ids) * 18 + 8,
                                      fill="#1a1a2e", outline="#404060", stipple="")

        for i, bid in enumerate(unique_ids):
            color = "#{:02x}{:02x}{:02x}".format(*BIOME_COLORS.get(bid, (128, 128, 128)))
            name = BIOMES.get(bid, "?")
            if len(name) > 16:
                name = name[:15] + "…"
            yy = y0 + i * 18
            self.canvas.create_rectangle(x0, yy, x0 + 14, yy + 14, fill=color, outline="")
            self.canvas.create_text(x0 + 20, yy + 7, text=name, fill="#d0d0e0",
                                     anchor=tk.W, font=("Segoe UI", 8))

    # ── Zoom & Pan ─────────────────────────────────────────────────────

    def _on_zoom(self, event):
        if event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom = max(0.25, self.zoom * 0.85)
        else:
            self.zoom = min(8.0, self.zoom * 1.18)
        self._render_map()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_drag(self, event):
        if self._drag_start:
            dx = event.x - self._drag_start[0]
            dy = event.y - self._drag_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self._drag_start = (event.x, event.y)
            self._render_map()

    def _on_reset_view(self, event):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render_map()

    # ── Export ─────────────────────────────────────────────────────────

    def _export(self):
        if not self.world:
            messagebox.showinfo("Export", "Generate a world first!")
            return

        out_dir = filedialog.askdirectory(title="Select Export Folder")
        if not out_dir:
            return

        try:
            self.status_var.set("Exporting...")
            self.root.update_idletasks()

            # Export all map images as PNG
            for name, pil_img in self.map_images.items():
                fname = name.lower().replace(" ", "_").replace("°c", "c").replace(".", "")
                path = os.path.join(out_dir, f"{fname}.png")
                pil_img.save(path, dpi=(150, 150))

            # Export data archive
            self.world.export_terrain(out_dir)

            self.status_var.set(f"Exported to {out_dir}")
            messagebox.showinfo("Export", f"All maps exported to:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


# ── Entry point ────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = WorldGeneratorGUI(root)

    # Handle window resize → re-render map
    def _on_resize(event):
        if app.world and not app.is_generating:
            app._render_map()

    root.bind("<Configure>", _on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
