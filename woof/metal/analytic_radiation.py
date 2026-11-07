"""Device surface fluxes for the upstream analytic clear-sky proxy."""
from __future__ import annotations
from datetime import datetime
import numpy as np
from .array import MetalArray, MetalArrayModule


def clear_sky_fluxes(valid_time, latitude, longitude, temperature, qv, pressure,
                     albedo, *, potential_temperature=False):
    from woof.core.analytic_radiation import (
        CLEAR_SKY_TRANSMISSIVITY, SOLAR_CONSTANT_WM2, STEFAN_BOLTZMANN)
    if not isinstance(valid_time, datetime):
        raise TypeError("valid_time must be a datetime")
    if not isinstance(temperature, MetalArray):
        raise TypeError("analytic Metal radiation requires resident surface inputs")
    context = temperature.buffer.context
    arrays = MetalArrayModule(context)
    shape = temperature.shape
    values = dict(latitude=arrays.asarray(latitude, dtype=np.float32),
                  longitude=arrays.asarray(longitude, dtype=np.float32),
                  temperature=temperature, qv=qv, pressure=pressure, albedo=albedo)
    for name, value in values.items():
        if (not isinstance(value, MetalArray) or value.shape != shape
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(f"analytic radiation {name} must be FP32 {shape} on one context")
    day = valid_time.timetuple().tm_yday
    year = valid_time.year
    days = 366.0 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365.0
    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0 + valid_time.microsecond / 3.6e9)
    gamma = 2.0 * np.pi / days * (day - 1.0 + (hour - 12.0) / 24.0)
    equation = 229.18 * (0.000075 + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma) - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma))
    declination = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2.0 * gamma) + 0.000907 * np.sin(2.0 * gamma)
        - 0.002697 * np.cos(3.0 * gamma) + 0.00148 * np.sin(3.0 * gamma))
    for name in ("swdown", "glw", "gsw"):
        values[name] = arrays.empty(shape, dtype=np.float32)
    context.submit_kernel("analytic_clear_sky_fluxes",
        {name: value.view((temperature.size,)) for name, value in values.items()},
        dict(n=temperature.size, solar_minutes0=float(60.0 * hour + equation),
             sin_declination=float(np.sin(declination)),
             cos_declination=float(np.cos(declination)),
             solar_flux=SOLAR_CONSTANT_WM2 * CLEAR_SKY_TRANSMISSIVITY,
             sigma=STEFAN_BOLTZMANN, potential_temperature=int(potential_temperature)),
        exact_policy=False, grid_size=temperature.size, threads_per_group=128)
    return values["swdown"], values["glw"], values["gsw"]
