"""
Climate Simulation
===================

GPU-accelerated climate system that computes physically-grounded:

- Temperature distribution (latitude + altitude + ocean proximity)
- Humidity/precipitation patterns (evaporation + wind transport)
- Global wind circulation (Hadley, Ferrel, Polar cells)
- Ocean currents and their climate influence
- Seasonal variation based on axial tilt
- Orographic precipitation (windward rain from mountain lift)
- Rain shadow effects (leeward drying from adiabatic warming)

All computations follow real physical principles in simplified form,
suitable for procedural generation at continental/planetary scale.

Key Physics
-----------
- Clausius-Clapeyron equation: e_s = 6.11 * 10^(7.5*T/(237.3+T))
  determines how much water vapor air can hold at a given temperature.
- Adiabatic lapse rate: rising air cools ~6.5 C/km (moist) or ~9.8 C/km
  (dry), and descending air warms at the same rate.
- Orographic lift: when wind encounters a mountain slope, the air is
  forced upward. Cooling reduces moisture capacity, forcing precipitation.
- Rain shadow: after losing moisture on the windward side, descending
  air on the leeward side warms adiabatically, dramatically increasing
  its moisture capacity while the actual moisture content is already low,
  creating extremely dry conditions (e.g., Death Valley, Patagonia).
"""

import numpy as np
from .backend import xp, to_cpu, to_gpu, spherical_coords
from .noise import NoiseGenerator


class ClimateSimulator:
    """
    GPU-accelerated climate simulation engine.

    Models the major drivers of Earth's climate system:
    1. Solar radiation (latitude-dependent)
    2. Altitude lapse rate (temperature decrease with height)
    3. Ocean thermal inertia (moderate coastal temperatures)
    4. Atmospheric circulation (trade winds, westerlies, polar easterlies)
    5. Orographic effects (rain shadows from mountains)
    6. Ocean currents (heat transport)
    7. Moisture transport by prevailing winds from ocean to land
    8. Clausius-Clapeyron moisture capacity limits
    """

    def __init__(self, config):
        self.config = config
        self.noise = NoiseGenerator(config.seed + 300)
        self._rng = np.random.RandomState(config.seed + 300)

    def simulate(self, heightmap, sea_level, width, height):
        """
        Run full climate simulation pipeline.

        Parameters
        ----------
        heightmap : array (H, W)
            Terrain elevation data.
        sea_level : float
            Sea level threshold.
        width, height : int
            Map dimensions.

        Returns
        -------
        temperature : array (H, W) - temperature in Celsius
        humidity : array (H, W) - relative humidity [0, 1]
        wind_u : array (H, W) - east-west wind component
        wind_v : array (H, W) - north-south wind component
        rainfall : array (H, W) - annual precipitation
        ocean_temp : array (H, W) - sea surface temperature
        rain_shadow : array (H, W) - rain shadow intensity [0, 1]
            (0 = no shadow, 1 = extreme dryness)
        """
        # Generate coordinate grids
        lon, lat, sx, sy, sz = spherical_coords(width, height)

        # -- Temperature -----------------------------------------------
        temperature = self._compute_temperature(heightmap, lat, sea_level, width, height)

        # -- Wind ------------------------------------------------------
        wind_u, wind_v = self._compute_wind(lat, heightmap, sea_level, width, height)

        # -- Humidity & Rainfall (with orographic + rain shadow) -------
        humidity, rainfall, rain_shadow = self._compute_humidity_rainfall(
            heightmap, temperature, wind_u, wind_v, sea_level,
            lat, width, height
        )

        # -- Ocean Currents --------------------------------------------
        ocean_temp = self._compute_ocean_temperature(lat, sea_level, width, height)

        # -- Ocean influence on coastal climate -------------------------
        temperature = self._apply_ocean_influence(
            temperature, ocean_temp, heightmap, sea_level, width, height
        )

        return temperature, humidity, wind_u, wind_v, rainfall, ocean_temp, rain_shadow

    def _compute_temperature(self, heightmap, lat, sea_level, width, height):
        """
        Compute temperature distribution based on physical principles.

        Temperature is determined by:
        1. Latitude: Solar angle affects insolation
           - Tropics (0-23.5): Direct overhead sun, highest temps
           - Temperate (23.5-66.5): Moderate, seasonal variation
           - Polar (66.5-90): Low angle sun, very cold
        2. Altitude: Temperature decreases ~6.5 C per km (lapse rate)
        3. Continentality: Interior locations have more extreme temps
        4. Ocean proximity: Maritime climates are more moderate
        """
        config = self.config

        # -- Latitude-based temperature --------------------------------
        abs_lat = xp.abs(lat)
        lat_fraction = abs_lat / (xp.pi / 2)
        lat_temp = 27 - 47 * xp.power(xp.clip(lat_fraction, 0, 1), 1.4)

        # -- Altitude lapse rate ---------------------------------------
        is_ocean = heightmap < sea_level
        land_elevation = xp.maximum(heightmap - sea_level, 0)
        elevation_km = land_elevation * config.max_elevation_km * 2.5
        alt_temp = -config.lapse_rate * elevation_km

        # -- Continentality effect -------------------------------------
        is_ocean_f = (heightmap < sea_level).astype(xp.float32)
        dist_from_ocean = self._distance_transform(is_ocean_f, width, height)
        dist_from_ocean = dist_from_ocean / (dist_from_ocean.max() + 1e-10)
        continentality = -3.0 * dist_from_ocean

        # -- Combined temperature --------------------------------------
        temperature = lat_temp + alt_temp + continentality

        # -- Add thermal noise for natural variation -------------------
        thermal_noise = self.noise.fbm(width, height, octaves=4, scale=3.0,
                                        offset_x=self._rng.uniform(-100, 100),
                                        offset_y=self._rng.uniform(-100, 100))
        temperature += thermal_noise * 3.0

        # Ocean temperature: more uniform, rarely below -2 C
        ocean_mask = heightmap < sea_level
        ocean_temp_base = xp.maximum(lat_temp * 0.7 + 10, -2)
        temperature = xp.where(ocean_mask, ocean_temp_base, temperature)

        return temperature

    def _compute_wind(self, lat, heightmap, sea_level, width, height):
        """
        Compute global wind circulation patterns.

        Earth's atmospheric circulation consists of three cells per hemisphere:
        - Hadley Cell (0-30): Trade winds (eastward near surface)
        - Ferrel Cell (30-60): Westerlies
        - Polar Cell (60-90): Polar easterlies

        Wind direction is modified by:
        - Coriolis effect (deflection to right in NH, left in SH)
        - Topographic steering (mountains redirect wind)
        - Pressure gradients (from temperature differences)
        """
        config = self.config
        lat_deg = xp.degrees(lat)

        # -- Zonal wind (east-west) from circulation cells -------------
        abs_lat = xp.abs(lat_deg)

        wind_u = xp.where(
            abs_lat < 30,
            -5.0 * xp.sin(xp.radians(abs_lat)),  # Trade winds
            xp.where(
                abs_lat < 60,
                8.0 * xp.sin(xp.radians(abs_lat - 30) / 3),  # Westerlies
                -4.0 * xp.sin(xp.radians(abs_lat - 60) / 3)  # Polar easterlies
            )
        )

        # -- Meridional wind (north-south) -----------------------------
        wind_v = xp.where(
            abs_lat < 15,
            -2.0 * xp.sign(lat_deg),  # Toward equator in Hadley cell
            xp.where(
                abs_lat < 45,
                2.0 * xp.sign(lat_deg),  # Poleward in Ferrel cell
                -1.5 * xp.sign(lat_deg)  # Equatorward in Polar cell
            )
        )

        # -- Coriolis deflection ---------------------------------------
        coriolis = config.wind_coriolis_factor * xp.sin(lat)
        wind_u_deflected = wind_u + wind_v * coriolis
        wind_v_deflected = wind_v - wind_u * coriolis

        # -- Topographic steering --------------------------------------
        padded_h = xp.pad(heightmap, 1, mode='edge')
        slope_x = padded_h[1:-1, 2:] - padded_h[1:-1, :-2]
        slope_y = padded_h[2:, 1:-1] - padded_h[:-2, 1:-1]

        topo_deflection = 2.0
        wind_u_final = wind_u_deflected - slope_x * topo_deflection
        wind_v_final = wind_v_deflected - slope_y * topo_deflection

        # -- Add turbulence noise --------------------------------------
        turb_u = self.noise.fbm(width, height, octaves=4, scale=5.0,
                                 offset_x=self._rng.uniform(-100, 100),
                                 offset_y=self._rng.uniform(-100, 100))
        turb_v = self.noise.fbm(width, height, octaves=4, scale=5.0,
                                 offset_x=self._rng.uniform(-100, 100),
                                 offset_y=self._rng.uniform(-100, 100))

        wind_u_final += turb_u * 1.5
        wind_v_final += turb_v * 1.5

        return wind_u_final, wind_v_final

    # =====================================================================
    #  CORE: Orographic Precipitation & Rain Shadow
    # =====================================================================

    def _compute_humidity_rainfall(self, heightmap, temperature, wind_u, wind_v,
                                    sea_level, lat, width, height):
        """
        Compute humidity and rainfall with physically-based orographic
        precipitation and rain shadow effects.

        Physical Model
        --------------
        1. **Ocean evaporation**: Oceans continuously supply moisture to the
           atmosphere. The evaporation rate depends on sea surface temperature
           via the Clausius-Clapeyron relation.

        2. **Prevailing wind transport**: Semi-Lagrangian advection carries
           moisture from oceans inland along the prevailing wind direction.
           Trade winds, westerlies, and polar easterlies determine which
           coastlines receive moisture-laden air.

        3. **Orographic lift**: When wind encounters a mountain slope, the air
           is forced upward. The vertical velocity w = V dot grad(h) where V
           is the horizontal wind and h is the terrain height. Rising air
           cools adiabatically at the moist lapse rate (~6.5 C/km).

        4. **Windward precipitation**: As air rises and cools, its saturation
           vapor pressure drops (Clausius-Clapeyron). When the actual moisture
           content exceeds the new (lower) capacity, the excess precipitates
           as rain or snow on the windward side of the mountain.

        5. **Rain shadow**: After crossing the mountain crest, the now-dry air
           descends on the leeward side. Descending air warms adiabatically,
           which INCREASES its moisture capacity. But the actual moisture
           content is already depleted from windward precipitation. The result
           is extremely low relative humidity and minimal rainfall — a rain
           shadow. Classic examples: the Atacama Desert (Andes), Death Valley
           (Sierra Nevada), Patagonia (Andes), Tibetan Plateau (Himalayas).

        6. **Convectional rainfall**: In tropical regions (ITCZ), warm humid
           air rises due to surface heating, causing thunderstorm-type
           precipitation independent of terrain.

        Implementation
        --------------
        The simulation runs 40 iterative steps of wind advection. At each step:
        - Moisture is advected by prevailing winds (semi-Lagrangian scheme)
        - Atmospheric diffusion spreads moisture laterally
        - Ocean cells replenish moisture via evaporation
        - Orographic forcing is computed from wind-terrain interaction
        - Windward slopes: adiabatic cooling → precipitation → moisture loss
        - Leeward slopes: adiabatic warming → capacity increase → drying
        - A cumulative rain shadow tracker records areas of persistent dryness
        """
        config = self.config
        is_ocean = heightmap < sea_level
        is_land = ~is_ocean
        is_land_f = is_land.astype(xp.float32)
        is_ocean_f = is_ocean.astype(xp.float32)

        # ================================================================
        #  Step 1: Saturation Vapor Pressure (Clausius-Clapeyron)
        # ================================================================
        # e_s = 6.11 * 10^(7.5 * T / (237.3 + T))  [hPa]
        # This determines the maximum moisture the air can hold.
        # Warmer air holds exponentially more moisture.
        T_safe = xp.clip(temperature, -40, 60)
        e_sat = 6.11 * xp.power(10.0, 7.5 * T_safe / (237.3 + T_safe))
        e_sat_max = e_sat.max() + 1e-10
        e_sat_norm = e_sat / e_sat_max  # Normalized capacity [0, 1]

        # ================================================================
        #  Step 2: Initialize Moisture Field
        # ================================================================
        # Over ocean: air is near saturation at SST
        # Over land: starts with minimal moisture (will be advected in)
        moisture = xp.where(is_ocean, e_sat_norm * 0.9, 0.03)

        # ================================================================
        #  Step 3: Terrain Gradient (for orographic forcing)
        # ================================================================
        padded_h = xp.pad(heightmap, 1, mode='edge')
        grad_x = (padded_h[1:-1, 2:] - padded_h[1:-1, :-2]) * 0.5
        grad_y = (padded_h[2:, 1:-1] - padded_h[:-2, 1:-1]) * 0.5

        # Height above sea level in km (for lapse rate calculations)
        land_elev_km = xp.maximum(heightmap - sea_level, 0) * config.max_elevation_km * 2.5

        # Wind speed (for scaling orographic effects)
        wind_speed = xp.sqrt(wind_u ** 2 + wind_v ** 2 + 1e-10)

        # Vertical velocity from orographic forcing:
        # w = V · ∇h (positive = ascending air on windward slopes)
        vertical_vel = wind_u * grad_x + wind_v * grad_y

        # Scale vertical velocity by wind speed for proportional effect
        # Stronger winds → more orographic lift → more precipitation
        vert_vel_norm = vertical_vel / (wind_speed + 1e-10)  # Normalized direction

        # ================================================================
        #  Step 4: Iterative Moisture Transport Simulation
        # ================================================================
        # Track cumulative precipitation and rain shadow intensity
        total_orographic_precip = xp.zeros_like(heightmap)
        total_rain_shadow = xp.zeros_like(heightmap)

        # Parameters
        num_iterations = 40
        km_per_heightmap_unit = config.max_elevation_km * 2.5

        for iteration in range(num_iterations):
            # ── 4a. Wind advection (semi-Lagrangian) ──────────────────
            # Transport moisture along the prevailing wind direction.
            # This is how ocean moisture reaches inland areas.
            advected = self._advect(moisture, wind_u, wind_v)
            # Blend: keep some local moisture, add advected moisture
            # (simulates continuous atmospheric flow)
            moisture = moisture * 0.82 + advected * 0.18

            # ── 4b. Atmospheric diffusion ─────────────────────────────
            # Lateral mixing spreads moisture (turbulent diffusion)
            padded_m = xp.pad(moisture, 1, mode='edge')
            neighbors = (
                padded_m[:-2, 1:-1] + padded_m[2:, 1:-1] +
                padded_m[1:-1, :-2] + padded_m[1:-1, 2:]
            ) / 4.0
            moisture += (neighbors - moisture) * 0.04

            # ── 4c. Ocean evaporation (continuous moisture source) ────
            # Oceans constantly replenish atmospheric moisture.
            # Evaporation rate depends on SST (warmer = more evaporation).
            # This is the PRIMARY moisture source that feeds orographic rain.
            ocean_evap_rate = e_sat_norm * 0.88
            moisture = xp.where(is_ocean, ocean_evap_rate, moisture)

            # Ensure land moisture doesn't go below a tiny baseline
            moisture = xp.maximum(moisture, 0.005)

            # ── 4d. OROGRAPHIC LIFT → WINDWARD PRECIPITATION ──────────
            # When wind blows toward a slope (positive vertical velocity),
            # air is forced upward. Rising air cools adiabatically.
            # Cooling reduces saturation vapor pressure → excess moisture
            # precipitates as rain/snow.

            is_ascending = vertical_vel > 0
            ascending_f = is_ascending.astype(xp.float32)

            # Altitude gained from orographic lift per iteration
            # Scale: vertical_vel is in heightmap-gradient units,
            # convert to km of altitude gain
            lift_km = xp.maximum(vertical_vel, 0) * km_per_heightmap_unit * 2.5

            # Adiabatic cooling from this lift (moist adiabatic lapse rate)
            # Use a slightly lower lapse rate for saturated air (~5.5 C/km)
            # since condensation releases latent heat, partially offsetting cooling
            moist_lapse = 5.5  # C/km (moist adiabatic, lower than dry 9.8)
            cooling = lift_km * moist_lapse

            # New temperature after adiabatic cooling
            cooled_temp = temperature - cooling
            T_cooled_safe = xp.clip(cooled_temp, -40, 60)

            # New saturation vapor pressure at the cooled temperature
            e_sat_cooled = 6.11 * xp.power(10.0, 7.5 * T_cooled_safe / (237.3 + T_cooled_safe))
            e_sat_cooled_norm = e_sat_cooled / e_sat_max

            # Moisture capacity is now lower due to cooling.
            # Any moisture above the new capacity must precipitate.
            new_capacity = e_sat_cooled_norm * 0.9
            excess_moisture = xp.maximum(moisture - new_capacity, 0)

            # Only precipitate on windward (ascending) land slopes
            # The stronger the lift, the more moisture is squeezed out
            precip_efficiency = ascending_f * is_land_f
            # Scale by how strong the orographic forcing is
            orographic_strength = xp.minimum(xp.maximum(vertical_vel, 0) * 15.0, 1.0)
            precipitation = excess_moisture * precip_efficiency * orographic_strength

            # Not all excess precipitates immediately — some remains as
            # supersaturation (clouds). Use a precipitation efficiency < 1.
            precip_rate = 0.5  # 50% of excess moisture precipitates per step
            actual_precip = precipitation * precip_rate

            # Accumulate orographic precipitation
            total_orographic_precip += actual_precip

            # Remove precipitated moisture from the atmosphere
            moisture = moisture - actual_precip

            # ── 4e. RAIN SHADOW → LEEWARD DRYING ─────────────────────
            # After crossing the mountain crest, air descends on the
            # leeward side. Descending air warms adiabatically (compression),
            # which INCREASES its moisture capacity dramatically.
            # But the air has already lost its moisture on the windward side,
            # so relative humidity plummets → rain shadow.

            is_descending = vertical_vel < 0
            descending_f = is_descending.astype(xp.float32)

            # Altitude lost from descent per iteration
            descent_km = xp.maximum(-vertical_vel, 0) * km_per_heightmap_unit * 2.0

            # Adiabatic warming from descent (dry adiabatic lapse rate)
            # Descending air is typically unsaturated → dry lapse rate ~9.8 C/km
            dry_lapse = 9.8  # C/km
            warming = descent_km * dry_lapse

            # New temperature after adiabatic warming
            warmed_temp = temperature + warming
            T_warmed_safe = xp.clip(warmed_temp, -40, 60)

            # New (higher) saturation vapor pressure after warming
            e_sat_warmed = 6.11 * xp.power(10.0, 7.5 * T_warmed_safe / (237.3 + T_warmed_safe))
            e_sat_warmed_norm = e_sat_warmed / e_sat_max

            # The capacity increase from warming means the air can hold
            # much more moisture, but it doesn't have any more moisture
            # to give. This dramatically reduces relative humidity.
            capacity_increase = xp.maximum(e_sat_warmed_norm - e_sat_norm, 0)

            # Rain shadow intensity: stronger on steeper leeward slopes
            # and when the air has already been depleted by windward rain
            shadow_strength = descending_f * is_land_f * xp.minimum(-vertical_vel * 10.0, 1.0)

            # Track cumulative rain shadow effect
            total_rain_shadow += capacity_increase * shadow_strength * 0.3

            # Moisture is further depleted on the leeward side
            # because the increased capacity makes the air "thirsty"
            leeward_drying = capacity_increase * shadow_strength * 0.2
            moisture = moisture - leeward_drying

            # Ensure moisture stays non-negative
            moisture = xp.maximum(moisture, 0.005)

        # ================================================================
        #  Step 5: Compute Final Humidity
        # ================================================================
        # Relative humidity = actual moisture / capacity
        humidity = xp.clip(moisture / (e_sat_norm + 1e-10), 0, 1)
        # Over ocean, humidity is always high (constant evaporation)
        humidity = xp.where(is_ocean, 0.85, humidity)
        humidity = xp.clip(humidity, 0, 1)

        # ================================================================
        #  Step 6: Compute Rain Shadow Map [0, 1]
        # ================================================================
        # Normalize rain shadow intensity to [0, 1]
        rs_max = total_rain_shadow.max() + 1e-10
        rain_shadow = total_rain_shadow / rs_max
        rain_shadow = xp.clip(rain_shadow, 0, 1)
        # Only applies on land
        rain_shadow = xp.where(is_ocean, 0, rain_shadow)

        # ================================================================
        #  Step 7: Compute Final Rainfall
        # ================================================================
        rainfall = xp.zeros_like(heightmap)

        # 7a. Orographic precipitation (dominant near mountains)
        # Average over iterations
        rainfall += total_orographic_precip / num_iterations

        # 7b. Convectional rainfall (ITCZ / tropical thunderstorms)
        # Independent of terrain — caused by surface heating
        itcz_factor = xp.exp(-xp.square(lat) / (xp.pi / 6) ** 2)
        temp_factor = xp.clip((temperature + 10) / 40, 0, 1)
        convectional = humidity * temp_factor * itcz_factor * 0.25
        rainfall += convectional * is_land_f

        # 7c. Frontal rainfall (simplified — mid-latitude cyclonic)
        # Ferrel cell zone (30-60 degrees) gets more frontal rain
        abs_lat_deg = xp.abs(xp.degrees(lat))
        frontal_factor = xp.exp(-xp.square(abs_lat_deg - 45) / (15 ** 2))
        frontal = humidity * frontal_factor * 0.15
        rainfall += frontal * is_land_f

        # 7d. Base rainfall (minimal everywhere on land)
        rainfall += 0.02 * is_land_f

        # 7e. Apply rain shadow suppression
        # Rain shadow areas get dramatically less rainfall
        # The shadow factor reduces rain proportionally to shadow intensity
        shadow_factor = 1.0 - rain_shadow * 0.85  # Up to 85% reduction
        shadow_factor = xp.maximum(shadow_factor, 0.05)  # Never zero
        rainfall *= shadow_factor

        # 7f. Noise for natural variation
        rain_noise = self.noise.fbm(width, height, octaves=4, scale=4.0,
                                     offset_x=self._rng.uniform(-100, 100),
                                     offset_y=self._rng.uniform(-100, 100))
        rainfall *= (0.8 + 0.2 * rain_noise)

        rainfall = xp.clip(rainfall, 0, 1)
        rainfall = xp.where(is_ocean, 0, rainfall)

        return humidity, rainfall, rain_shadow

    def _compute_ocean_temperature(self, lat, sea_level, width, height):
        """
        Compute sea surface temperature.

        Ocean temperature follows latitude bands with modifications
        from ocean currents. Warm currents (like Gulf Stream) transport
        heat poleward on western boundaries.
        """
        abs_lat = xp.abs(lat)

        # Base SST: warm at equator (~27 C), cold at poles (~-2 C)
        lat_frac = abs_lat / (xp.pi / 2)
        sst = 25 - 27 * xp.power(xp.clip(lat_frac, 0, 1), 1.2)

        # Warm current noise (western boundary currents)
        warm_current = self.noise.fbm(width, height, octaves=3, scale=2.0)
        sst += warm_current * 5

        # Minimum SST for seawater: -2 C
        sst = xp.maximum(sst, -2)

        return sst

    def _apply_ocean_influence(self, temperature, ocean_temp, heightmap,
                                sea_level, width, height):
        """
        Apply ocean thermal moderation to coastal areas.

        Oceans have high thermal inertia, moderating nearby land
        temperatures (cooler summers, warmer winters). This effect
        diminishes with distance from coast.
        """
        is_ocean = heightmap < sea_level
        ocean_proximity = self._compute_ocean_proximity(is_ocean, width, height)

        # Blend toward ocean temperature based on proximity
        blend = ocean_proximity * 0.3
        temperature = temperature * (1 - blend) + ocean_temp * blend

        return temperature

    def _compute_ocean_proximity(self, is_ocean, width, height):
        """
        Compute how close each cell is to the ocean.
        Returns values in [0, 1] where 1 = ocean, 0 = far inland.
        """
        dist = self._distance_transform(is_ocean.astype(xp.float32), width, height)
        max_dist = dist.max() + 1e-10
        proximity = 1.0 - dist / max_dist
        proximity = xp.where(is_ocean, 1.0, proximity)
        return proximity

    def _distance_transform(self, binary_map, width, height, iterations=30):
        """
        Approximate Euclidean distance transform using iterative relaxation.
        GPU-parallel implementation.
        """
        dist = xp.where(binary_map > 0.5, 0, xp.full_like(binary_map, 1e6))

        cell_size = 1.0 / max(width, height)

        for _ in range(iterations):
            padded = xp.pad(dist, 1, mode='constant', constant_values=1e6)

            new_dist = xp.minimum(
                dist,
                xp.minimum(
                    xp.minimum(padded[:-2, 1:-1] + cell_size, padded[2:, 1:-1] + cell_size),
                    xp.minimum(padded[1:-1, :-2] + cell_size, padded[1:-1, 2:] + cell_size)
                )
            )
            new_dist = xp.minimum(
                new_dist,
                xp.minimum(
                    xp.minimum(padded[:-2, :-2] + 1.414 * cell_size,
                              padded[:-2, 2:] + 1.414 * cell_size),
                    xp.minimum(padded[2:, :-2] + 1.414 * cell_size,
                              padded[2:, 2:] + 1.414 * cell_size)
                )
            )
            dist = new_dist

        return dist

    def _advect(self, field, wind_u, wind_v):
        """
        Semi-Lagrangian advection: transport field by wind.
        For each cell, trace back along wind vector and sample field.
        This is the key mechanism for carrying ocean moisture inland
        along prevailing wind directions.
        """
        h, w = field.shape

        j, i = xp.meshgrid(xp.arange(w, dtype=xp.float32),
                            xp.arange(h, dtype=xp.float32))

        # Trace back along wind to find source location
        src_i = i - wind_v * 0.1
        src_j = j - wind_u * 0.1

        # Clamp to bounds
        src_i = xp.clip(src_i, 0, h - 1.001)
        src_j = xp.clip(src_j, 0, w - 1.001)

        # Bilinear interpolation
        i0 = src_i.astype(xp.int32)
        j0 = src_j.astype(xp.int32)
        i1 = xp.minimum(i0 + 1, h - 1)
        j1 = xp.minimum(j0 + 1, w - 1)

        fi = src_i - i0
        fj = src_j - j0

        result = (
            field[i0, j0] * (1 - fi) * (1 - fj) +
            field[i0, j1] * (1 - fi) * fj +
            field[i1, j0] * fi * (1 - fj) +
            field[i1, j1] * fi * fj
        )

        return result
