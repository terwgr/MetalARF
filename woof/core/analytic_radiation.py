"""Production analytic clear-sky radiation for ``ra_physics=90``.

The flux proxy is the Phase-3 real-data implementation, while
:class:`AnalyticClearSkyRadiation` adapts it to the Phase-4 radiation-slot
call contract.  The adapter supplies zero atmospheric heating because the
proxy only diagnoses surface shortwave and longwave fluxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


SOLAR_CONSTANT_WM2 = 1361.0
CLEAR_SKY_TRANSMISSIVITY = 0.75
STEFAN_BOLTZMANN = 5.670374419e-8


def _array_namespace(*values):
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        import cupy as cp
        return cp
    return np


def analytic_clear_sky_forcing(
        valid_time: datetime, latitude_deg, longitude_deg,
        air_temperature_k, qv_kgkg, pressure_pa, *,
        transmissivity: float = CLEAR_SKY_TRANSMISSIVITY):
    """Return the Phase-3 clear-sky ``(SWDOWN, GLW)`` proxy.

    Solar position uses the fractional-year/equation-of-time approximation
    from NOAA's solar-calculation equations.  Longwave uses a Brunt-family
    vapor-pressure emissivity.  Inputs may be NumPy or CuPy arrays; device
    inputs stay on device.
    """
    if not isinstance(valid_time, datetime):
        raise TypeError("valid_time must be a datetime")
    if not np.isfinite(transmissivity) or not 0.70 <= transmissivity <= 0.80:
        raise ValueError("transmissivity must be within the plan's 0.75+/-0.05")
    xp = _array_namespace(
        latitude_deg, longitude_deg, air_temperature_k, qv_kgkg, pressure_pa)
    lat = xp.asarray(latitude_deg)
    lon = xp.asarray(longitude_deg)
    temperature = xp.asarray(air_temperature_k)
    qv = xp.asarray(qv_kgkg)
    pressure = xp.asarray(pressure_pa)
    if not (lat.shape == lon.shape == temperature.shape == qv.shape
            == pressure.shape):
        raise ValueError("clear-sky forcing inputs must share one grid shape")

    day = valid_time.timetuple().tm_yday
    days_in_year = 366.0 if (valid_time.year % 4 == 0
                             and (valid_time.year % 100 != 0
                                  or valid_time.year % 400 == 0)) else 365.0
    utc_hour = (valid_time.hour + valid_time.minute / 60.0
                + valid_time.second / 3600.0
                + valid_time.microsecond / 3.6e9)
    gamma = 2.0 * np.pi / days_in_year * (day - 1.0
                                           + (utc_hour - 12.0) / 24.0)
    equation_of_time_min = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma))
    declination = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma) + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma) + 0.00148 * np.sin(3.0 * gamma))
    true_solar_minutes = 60.0 * utc_hour + equation_of_time_min + 4.0 * lon
    hour_angle = xp.deg2rad(true_solar_minutes / 4.0 - 180.0)
    latitude = xp.deg2rad(lat)
    cosine_zenith = (xp.sin(latitude) * np.sin(declination)
                      + xp.cos(latitude) * np.cos(declination)
                      * xp.cos(hour_angle))
    swdown = (SOLAR_CONSTANT_WM2 * transmissivity
              * xp.maximum(cosine_zenith, 0.0))

    vapor_pressure_hpa = xp.maximum(
        xp.maximum(qv, 0.0) * pressure / (0.622 + xp.maximum(qv, 0.0))
        / 100.0, 0.01)
    emissivity = 0.70 + 0.09 * xp.log10(vapor_pressure_hpa)
    glw = emissivity * STEFAN_BOLTZMANN * temperature ** 4
    return swdown, glw


@dataclass
class AnalyticClearSkyRadiation:
    """Adapt the clear-sky proxy to the production radiation-slot API."""

    start_time: datetime
    latitude_deg: object
    longitude_deg: object
    longwave: bool = True
    shortwave: bool = True
    update_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if type(self.longwave) is not bool or type(self.shortwave) is not bool:
            raise TypeError("analytic radiation spectrum selectors must be boolean")
        if not (self.longwave or self.shortwave):
            raise ValueError("analytic radiation needs at least one selected spectrum")
        if not isinstance(self.start_time, datetime):
            raise TypeError("radiation_start_time must be a datetime")
        try:
            import cupy as xp
        except ModuleNotFoundError as error:
            if error.name != "cupy":
                raise
            xp = np
        self.latitude_deg = xp.ascontiguousarray(
            xp.asarray(self.latitude_deg, dtype=xp.float32))
        self.longitude_deg = xp.ascontiguousarray(
            xp.asarray(self.longitude_deg, dtype=xp.float32))
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError("radiation latitude/longitude shapes must match")

    def __call__(self, *, atmosphere, fields, state, cfg):
        """Return a radiation-slot result at the state's current time."""
        if getattr(state, "_metal_setup_state", False):
            return self._call_metal(atmosphere=atmosphere, fields=fields,
                                    state=state, cfg=cfg)
        import cupy as cp
        elapsed = float(state.elapsed_seconds)
        theta = atmosphere["theta"]
        pressure = atmosphere["pressure"]
        surface_shape = pressure.shape[1:]
        if self.latitude_deg.shape != surface_shape:
            raise ValueError(
                "radiation latitude/longitude shape must match the state "
                f"surface shape {surface_shape}")
        # Preserve the Phase-3 proxy's pinned 287/1004 FP32 exponent.
        temperature = (theta[0]
                       * (pressure[0] / cp.float32(100000.0))
                       ** cp.float32(287.0 / 1004.0))
        valid_time = self.start_time + timedelta(seconds=elapsed)
        swdown, glw = analytic_clear_sky_forcing(
            valid_time, self.latitude_deg, self.longitude_deg, temperature,
            atmosphere["qv"][0], pressure[0])
        # The analytic flux remains this adapter's approximation, but the
        # surface carrier follows WRF's radiation-clock geometry: only the
        # hour angle is shifted by half the radiation interval.  A strict
        # (GPUWM_WRF_EXACT=1) wrf_461 run takes WRF 4.6.1's radiation-call
        # cosine word for word (woof.core.wrf_radiation_clock: radconst +
        # calc_coszen on WRF's REAL(4) calendar, the namelist radt used raw,
        # graded by tools/analytic_clear_sky_oracle); every other run keeps
        # the solar generation's geometry.
        from woof.wrf_exact import ENABLED as _strict
        radt_minutes = cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes
        generation = getattr(cfg, "solar_generation", "wrf_480")
        if _strict and generation == "wrf_461":
            from woof.core.wrf_radiation_clock import wrf_radiation_coszen
            coszen = wrf_radiation_coszen(
                self.start_time, elapsed, radt_minutes,
                self.latitude_deg, self.longitude_deg)
        else:
            from woof.core.dudhia import wrf_solar_geometry
            from woof.core.physics import _physics_period_seconds
            interval = _physics_period_seconds(radt_minutes, cfg)
            from woof.core.solar import solar_period_seconds
            interval = solar_period_seconds(cfg, interval)
            coszen, _ = wrf_solar_geometry(
                valid_time, self.latitude_deg, self.longitude_deg,
                hour_offset_seconds=0.5 * interval, generation=generation,
                start_time=self.start_time, elapsed_seconds=state.elapsed_seconds)
        gsw = swdown * (cp.float32(1.0) - fields["albedo"])
        swdown, glw, gsw = self._selected_fluxes(
            cp, fields, swdown, glw, gsw)

        # Imported lazily to keep the generic result contract in physics.py
        # without creating a module-import cycle.
        from woof.core.physics import RadiationResult
        zero_heating = cp.zeros_like(theta)
        result = RadiationResult(
            rthratenlw=zero_heating,
            rthratensw=zero_heating,
            swdown=swdown,
            glw=glw,
            gsw=gsw,
            coszen=coszen)
        self.update_count += 1
        return result


    def _selected_fluxes(self, arrays, fields, swdown, glw, gsw):
        """Publish only selected spectra, preserving caller-owned longwave.

        The surface proxy evaluates both fluxes together; the same selection
        applies to the CUDA and Metal results before the driver captures them.
        """
        if not self.longwave:
            glw = fields["glw"]
        if not self.shortwave:
            swdown = arrays.zeros(swdown.shape, dtype=swdown.dtype)
            gsw = swdown
        return swdown, glw, gsw

    def _call_metal(self, *, atmosphere, fields, state, cfg):
        from woof.wrf_exact import ENABLED as _strict
        if _strict:
            raise NotImplementedError(
                "strict Metal analytic radiation surface flux arithmetic is not supported")
        from woof.metal.array import MetalArrayModule
        from woof.metal.analytic_radiation import clear_sky_fluxes
        from woof.core.dudhia import wrf_solar_geometry
        from woof.core.physics import RadiationResult, _physics_period_seconds

        pressure = atmosphere["pressure"]
        arrays = MetalArrayModule(pressure.buffer.context)
        shape = pressure.shape[1:]
        if self.latitude_deg.shape != shape:
            raise ValueError("radiation latitude/longitude must match the surface grid")
        valid_time = self.start_time + timedelta(seconds=float(state.elapsed_seconds))
        swdown, glw, gsw = clear_sky_fluxes(
            valid_time, self.latitude_deg, self.longitude_deg,
            atmosphere["theta"][0], atmosphere["qv"][0], pressure[0],
            fields["albedo"], potential_temperature=True)
        swdown, glw, gsw = self._selected_fluxes(
            arrays, fields, swdown, glw, gsw)
        minutes = cfg.radt if cfg.radt > 0.0 else cfg.radt_minutes
        interval = _physics_period_seconds(minutes, cfg)
        from woof.core.solar import solar_period_seconds
        interval = solar_period_seconds(cfg, interval)
        lat, lon = self.latitude_deg, self.longitude_deg
        if cfg.solar_generation == "wrf_480":
            lat, lon = arrays.asarray(lat, dtype=np.float32), arrays.asarray(lon, dtype=np.float32)
        coszen, _ = wrf_solar_geometry(
            valid_time, lat, lon, hour_offset_seconds=0.5 * interval,
            generation=cfg.solar_generation, start_time=self.start_time,
            elapsed_seconds=state.elapsed_seconds)
        zero = arrays.zeros(pressure.shape, dtype=pressure.dtype)
        self.update_count += 1
        return RadiationResult(rthratenlw=zero, rthratensw=zero,
            swdown=swdown, glw=glw, gsw=gsw, coszen=arrays.asarray(coszen))

    def restart_identity(self):
        from woof.io.restart import _array_setup_identity
        identity = {
            "start_time": self.start_time.isoformat(),
            "latitude": _array_setup_identity(self.latitude_deg),
            "longitude": _array_setup_identity(self.longitude_deg),
            "algorithm": "analytic-clear-sky-surface-flux-v1",
            "above_atmosphere_policy": "surface proxy; zero atmospheric heating",
            "proxy_constants": [CLEAR_SKY_TRANSMISSIVITY,
                                SOLAR_CONSTANT_WM2, STEFAN_BOLTZMANN],
        }
        if not (self.longwave and self.shortwave):
            identity["spectra"] = {"lw": self.longwave, "sw": self.shortwave}
        return identity


__all__ = ["AnalyticClearSkyRadiation", "CLEAR_SKY_TRANSMISSIVITY",
           "SOLAR_CONSTANT_WM2", "STEFAN_BOLTZMANN",
           "analytic_clear_sky_forcing"]
