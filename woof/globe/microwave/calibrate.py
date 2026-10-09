"""Two-direction calibration of the microwave operator on synthetic columns.

Every number the operator reports against real radiances is preceded by
these readings, each with an analytic or planted answer:

1. ``isothermal``: an isothermal column over a black surface at the same
   temperature reads back that temperature in every channel to numerical
   precision (Planck exact), whatever the absorption does.  With an
   emissivity below one the analytic answer is
   ``B(TB) = B(T) - (1 - e) t_s^2 (B(T) - B(T_cmb))`` and is read back to
   the same precision.
2. ``planted_layer``: a +0.5 K perturbation placed at the level pair where
   a channel's weighting function peaks (per unit ln p) reads back in that
   channel as 0.5 K times the weight the pair carries (to 3e-7 K under
   frozen opacity, 0.03 K with the absorption's own temperature term), no
   channel whose weight at that pair is below 1e-3 moves by more than
   0.02 K, and the planted channel is the loudest sounding channel except
   where a lower channel carries more weight at that very pair (channel 5
   peaks in the surface layer of the tropical column, where channel 4's
   weight is larger); the rank is recorded per channel.
3. ``null``: a column identical to itself moves nothing: O-B is exactly
   zero for every channel (the operator is deterministic and the
   comparison bitwise).
4. ``weights_sum``: the Rayleigh-Jeans weights of every channel sum with
   the surface transmittance to one within 1e-6.
5. ``planck_term``: the Planck-exact minus Rayleigh-Jeans brightness
   temperature per channel on a standard column, the term the design
   asks to be stated.
6. ``absorption_oracle``: the P.676-13 absorption against an independent
   public implementation (recorded from the run of 2026-09-06 on
   pyrtlib 1.2.0, models R98 and R22 for oxygen, R98 and R22SD for water
   vapour, N2 R18) at the pressures the sounding channels weight;
   ``max_abs_percent`` names the largest disagreement per pressure.

``run()`` returns the receipt as a dictionary; the door writes it beside
the scorecard and the tests hold every reading.
"""

from __future__ import annotations

import numpy as np

from .channels import CHANNELS, TEMPERATURE_SOUNDING_CHANNELS
from .emissivity import COSMIC_BACKGROUND_K
from .rte import (
    Column, brightness_temperature, channel_quadrature, planck_radiance,
    planck_temperature, surface_transmittance, weighting_function,
)

STANDARD_LEVELS_PA = np.array([
    1, 2, 4, 7, 10, 20, 40, 70, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000,
    5000, 7000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000,
    60000, 65000, 70000, 75000, 80000, 85000, 90000, 92500, 95000, 97500, 100000,
], dtype=np.float64)


def standard_column(*, surface_k: float = 300.0, ps_pa: float = 101000.0,
                    lapse_k_per_km: float = 6.5, tropopause_km: float = 16.0,
                    stratosphere_k_per_km: float = 2.0, q_surface: float = 0.016,
                    q_scale_km: float = 2.2) -> Column:
    """A smooth tropical-like column on the GDAS level set: a lapse-rate
    troposphere, an isothermal layer to 25 km, warming above."""
    p = STANDARD_LEVELS_PA
    z = -7.5 * np.log(p / ps_pa)
    t = np.where(
        z < tropopause_km,
        surface_k - lapse_k_per_km * z,
        np.where(
            z < 25.0,
            surface_k - lapse_k_per_km * tropopause_km,
            surface_k - lapse_k_per_km * tropopause_km + stratosphere_k_per_km * (z - 25.0),
        ),
    )
    t = np.clip(t, 180.0, 330.0)
    q = q_surface * np.exp(-z / q_scale_km)
    q = np.where(p < 10000.0, 3.0e-6, q)
    return Column(
        pressure_pa=p,
        temperature_k=t[:, None],
        specific_humidity=q[:, None],
        surface_pressure_pa=np.array([ps_pa]),
        skin_temperature_k=np.array([surface_k + 1.0]),
        air_temperature_2m_k=np.array([surface_k - 1.0]),
    )


def isothermal_reading(temperature_k: float = 250.0, emissivity: float = 1.0,
                       zenith_deg: float = 30.0) -> dict:
    p = STANDARD_LEVELS_PA
    column = Column(
        pressure_pa=p,
        temperature_k=np.full((p.size, 1), temperature_k),
        specific_humidity=np.zeros((p.size, 1)),
        surface_pressure_pa=np.array([101325.0]),
        skin_temperature_k=np.array([temperature_k]),
    )
    channels = list(range(1, 23))
    tb = brightness_temperature(column, channels, zenith_deg, 0.0, emissivity=emissivity)[:, 0]
    expected = np.empty(len(channels))
    for k, number in enumerate(channels):
        frequencies, weights = channel_quadrature(CHANNELS[number - 1])
        # The analytic answer per quadrature frequency, with that
        # frequency's own surface-to-space transmittance.
        t_s = surface_transmittance(column, frequencies, zenith_deg)[:, 0]
        f = frequencies
        b_t = planck_radiance(f, temperature_k)
        b_cmb = planck_radiance(f, COSMIC_BACKGROUND_K)
        radiance = b_t - (1.0 - emissivity) * t_s ** 2 * (b_t - b_cmb)
        expected[k] = np.sum(weights * planck_temperature(f, radiance))
    error = tb - expected
    return {
        "temperature_k": temperature_k,
        "emissivity": emissivity,
        "zenith_deg": zenith_deg,
        "max_abs_error_k": float(np.max(np.abs(error))),
        "per_channel_error_k": [float(x) for x in error],
    }


def weight_density(column: Column, number: int, zenith_deg: float):
    """``(p_hpa, weight, weight per unit ln p, surface transmittance)`` of a
    channel's Rayleigh-Jeans weighting function on the refined sub-layers.

    The peak of a weighting function is the layer of largest weight PER
    UNIT ln p.  The largest weight per sub-layer is not it: the sub-layers
    are quarters of the level pairs and the GDAS pairs are 25 to 50 hPa
    wide at the bottom and up to 0.4 in ln p in the upper troposphere, so
    the per-sub-layer maximum lands on the thickest pair near the peak
    (channels 8 and 9 both read 143 hPa that way; per unit ln p they peak
    at 243 and 155 hPa)."""
    from .rte import build_layers

    p_hpa, w, t_s = weighting_function(column, number, zenith_deg)
    thickness = build_layers(column).thickness_lnp
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(thickness > 0.0, w / np.where(thickness > 0.0, thickness, 1.0), 0.0)
    return p_hpa, w, density, t_s


def weights_sum_reading(zenith_deg: float = 30.0) -> dict:
    column = standard_column()
    worst = 0.0
    peaks = {}
    for number in range(1, 23):
        p_hpa, w, density, t_s = weight_density(column, number, zenith_deg)
        total = float(np.sum(w[:, 0]) + t_s[0])
        worst = max(worst, abs(total - 1.0))
        peaks[number] = {
            "peak_hpa": float(p_hpa[np.argmax(density[:, 0]), 0]),
            "peak_per_sublayer_hpa": float(p_hpa[np.argmax(w[:, 0]), 0]),
            "centroid_hpa": float(np.exp(np.sum(w[:, 0] * np.log(p_hpa[:, 0])) / np.sum(w[:, 0]))),
            "surface_transmittance": float(t_s[0]),
        }
    return {"max_abs_deviation_from_one": worst, "channels": peaks}


def planted_layer_reading(delta_k: float = 0.5, zenith_deg: float = 30.0,
                          channels=TEMPERATURE_SOUNDING_CHANNELS) -> dict:
    """For each sounding channel: plant +delta_k at the parent level pair
    where its weight peaks and read the response of every channel.

    The expected response is linear (the weights times the planted
    temperature), so ``delta_k`` is small; a plant ten times larger moves
    the absorption and the layer thickness themselves and reads below the
    linear answer in the upper-tropospheric channels, which is the
    physics, not an error, and is reported as ``nonlinearity_10x_k``."""
    base = standard_column()
    all_channels = list(range(1, 23))
    tb0 = brightness_temperature(base, all_channels, zenith_deg, 0.0, emissivity=1.0)[:, 0]
    readings = []
    from .rte import build_layers

    layers = build_layers(base)
    for number in channels:
        p_hpa, w, density, _ = weight_density(base, number, zenith_deg)
        peak = int(np.argmax(density[:, 0]))
        parent = int(layers.parent_level[peak])
        # A peak inside the surface slab (below the lowest analysis level)
        # is planted at the lowest level pair: the slab follows that pair's
        # slope, so it moves with it.
        peak_in_slab = parent > base.pressure_pa.size - 2
        parent = min(parent, base.pressure_pa.size - 2)
        # The parent level pair (parent, parent+1) in the top-down refined
        # order; map back to the column's own level indices.
        order = np.argsort(base.pressure_pa)  # ascending pressure = top down
        # build_layers prepends an extension level when the top is above 1 Pa;
        # STANDARD_LEVELS_PA starts at 1 Pa so no extension is added here.
        i_top = order[parent]
        i_bot = order[min(parent + 1, base.pressure_pa.size - 1)]
        t = base.temperature_k.copy()
        t[i_top, 0] += delta_k
        t[i_bot, 0] += delta_k
        planted = Column(
            pressure_pa=base.pressure_pa, temperature_k=t,
            specific_humidity=base.specific_humidity,
            surface_pressure_pa=base.surface_pressure_pa,
            skin_temperature_k=base.skin_temperature_k,
            air_temperature_2m_k=base.air_temperature_2m_k,
        )
        # Frozen absorption: the opacity stays the reference column's, so
        # the plant reads back through the weights alone (the linear
        # reading).  Full physics: the warmer layer is also a little more
        # opaque, which moves weight from the layers below it; that term
        # is the difference between the two readings and is reported.
        tb_frozen = brightness_temperature(
            planted, all_channels, zenith_deg, 0.0, emissivity=1.0, absorption_reference=base,
        )[:, 0]
        tb1 = brightness_temperature(planted, all_channels, zenith_deg, 0.0, emissivity=1.0)[:, 0]
        response = tb1 - tb0
        response_frozen = tb_frozen - tb0
        # Expected response in the planted channel: the sum of its weights
        # over the sub-layers whose interpolated temperature moved.  Both
        # levels of the pair move by delta, so every sub-layer between them
        # moves by delta; the neighbouring pairs move by a fraction (the
        # linear interpolation between a moved and an unmoved level).
        moved = np.zeros(layers.parent_level.size)
        first = {}
        for k, par in enumerate(layers.parent_level):
            first.setdefault(int(par), k)
        for k, par in enumerate(layers.parent_level):
            par = int(par)
            n_sub = int(np.sum(layers.parent_level == par))
            frac = (k - first[par] + 0.5) / n_sub
            if par == parent:
                moved[k] = 1.0
            elif par == parent - 1:
                moved[k] = frac
            elif par == parent + 1:
                moved[k] = 1.0 - frac
        # The surface slab below the lowest level follows the lowest pair's
        # slope, so a plant at the lowest pair moves the slab too: both of
        # its parents moved by delta, and the slab's extrapolated bottom
        # moves by the same delta (the slope is unchanged).
        if parent == base.pressure_pa.size - 2 and layers.parent_level.max() > parent:
            moved[layers.parent_level == parent + 1] = 1.0
        expected = delta_k * float(np.sum(w[:, 0] * moved))
        quiet = [c for c in all_channels
                 if np.sum(weighting_function(base, c, zenith_deg)[1][:, 0] * (moved > 0)) < 1.0e-3]
        sounding_response = {c: float(response[c - 1]) for c in TEMPERATURE_SOUNDING_CHANNELS}
        ranked = sorted(sounding_response, key=lambda c: -sounding_response[c])
        # The same plant at ten times the size, to read the nonlinearity.
        t10 = base.temperature_k.copy()
        t10[i_top, 0] += 10.0 * delta_k
        t10[i_bot, 0] += 10.0 * delta_k
        tb10 = brightness_temperature(
            Column(
                pressure_pa=base.pressure_pa, temperature_k=t10,
                specific_humidity=base.specific_humidity,
                surface_pressure_pa=base.surface_pressure_pa,
                skin_temperature_k=base.skin_temperature_k,
                air_temperature_2m_k=base.air_temperature_2m_k,
            ),
            [number], zenith_deg, 0.0, emissivity=1.0,
        )[0, 0]
        readings.append({
            "channel": number,
            "peak_hpa": float(p_hpa[peak, 0]),
            "peak_in_surface_slab": bool(peak_in_slab),
            "planted_levels_hpa": [float(base.pressure_pa[i_top] / 100.0), float(base.pressure_pa[i_bot] / 100.0)],
            "expected_response_k": expected,
            "frozen_absorption_response_k": float(response_frozen[number - 1]),
            "frozen_absorption_error_k": float(response_frozen[number - 1] - expected),
            "response_k": float(response[number - 1]),
            "response_error_k": float(response[number - 1] - expected),
            "absorption_temperature_term_k": float(response[number - 1] - response_frozen[number - 1]),
            "loudest_sounding_channel": int(ranked[0]),
            "rank_among_sounding_channels": int(ranked.index(number) + 1),
            "nonlinearity_10x_k": float((tb10 - tb0[number - 1]) - 10.0 * expected),
            "quiet_channel_max_response_k": float(max((abs(response[c - 1]) for c in quiet), default=0.0)),
            "quiet_channels": quiet,
        })
    return {
        "delta_k": delta_k,
        "max_abs_frozen_error_k": max(abs(r["frozen_absorption_error_k"]) for r in readings),
        "max_abs_response_error_k": max(abs(r["response_error_k"]) for r in readings),
        "max_abs_absorption_temperature_term_k": max(
            abs(r["absorption_temperature_term_k"]) for r in readings
        ),
        "max_quiet_response_k": max(r["quiet_channel_max_response_k"] for r in readings),
        "worst_rank": max(r["rank_among_sounding_channels"] for r in readings),
        "readings": readings,
    }


def null_reading(zenith_deg: float = 30.0) -> dict:
    column = standard_column()
    channels = list(range(1, 23))
    a = brightness_temperature(column, channels, zenith_deg, 10.0)
    b = brightness_temperature(column, channels, zenith_deg, 10.0)
    return {"max_abs_difference_k": float(np.max(np.abs(a - b))), "bitwise_equal": bool(np.array_equal(a, b))}


def planck_term_reading(zenith_deg: float = 30.0) -> dict:
    column = standard_column()
    channels = list(range(1, 23))
    exact = brightness_temperature(column, channels, zenith_deg, 10.0)[:, 0]
    rj = brightness_temperature(column, channels, zenith_deg, 10.0, rayleigh_jeans=True)[:, 0]
    return {
        "planck_minus_rayleigh_jeans_k": {int(c): float(exact[k] - rj[k]) for k, c in enumerate(channels)},
        "brightness_temperature_k": {int(c): float(exact[k]) for k, c in enumerate(channels)},
    }


#: Recorded 2026-09-06 on the 16 GB host, pyrtlib 1.2.0 (GPLv3, run as an oracle
#: only; nothing of it is shipped): largest absolute relative difference
#: (percent) between this module's P.676-13 absorption and each pyrtlib
#: model over the 50.3 to 57.3 GHz sounding frequencies, per pressure.
ABSORPTION_ORACLE_PERCENT = {
    "frequencies_ghz": "50.3, 51.76, 52.8, 53.481, 53.711, 54.4, 54.94, 55.5, 57.290, 57.073, 56.946, 56.964",
    "R98": {1013.0: 2.64, 700.0: 1.74, 500.0: 4.25, 200.0: 9.61, 50.0: 10.87, 10.0: 9.49, 2.0: 16.5},
    "R22": {1013.0: 4.9, 700.0: 4.5, 500.0: 4.06, 200.0: 3.0, 50.0: 2.65, 10.0: 2.87, 2.0: 26.09},
    "reading": (
        "within 5 percent of R22 from the surface to 10 hPa; the 2 hPa row "
        "(channels 14 and 15) diverges by 12 to 26 percent where the pressure "
        "width meets the P.676 width floor and the Zeeman splitting neither "
        "model carries"
    ),
}


def run() -> dict:
    return {
        "schema": "gpuwm-arwen-global-microwave-calibration-v1",
        "isothermal_black": isothermal_reading(250.0, 1.0),
        "isothermal_grey": isothermal_reading(280.0, 0.6),
        "weights_sum": weights_sum_reading(),
        "planted_layer": planted_layer_reading(),
        "null": null_reading(),
        "planck_term": planck_term_reading(),
        "absorption_oracle": ABSORPTION_ORACLE_PERCENT,
    }


def passes(receipt: dict, *, isothermal_tol_k: float = 1.0e-6, frozen_tol_k: float = 1.0e-3,
           planted_tol_k: float = 0.03, quiet_tol_k: float = 0.02,
           weights_tol: float = 1.0e-6) -> dict[str, bool]:
    """The gate on the calibration receipt.  The frozen-absorption plant
    is the exact reading (weights times the plant, to a millikelvin, the
    Planck curvature); the full-physics plant carries the absorption's own
    temperature dependence and is held to 0.03 K at a 0.5 K plant."""
    planted = receipt["planted_layer"]
    return {
        "isothermal_black": receipt["isothermal_black"]["max_abs_error_k"] <= isothermal_tol_k,
        "isothermal_grey": receipt["isothermal_grey"]["max_abs_error_k"] <= isothermal_tol_k,
        "weights_sum": receipt["weights_sum"]["max_abs_deviation_from_one"] <= weights_tol,
        "planted_layer_frozen": planted["max_abs_frozen_error_k"] <= frozen_tol_k,
        "planted_layer_full": planted["max_abs_response_error_k"] <= planted_tol_k,
        "quiet_channels": planted["max_quiet_response_k"] <= quiet_tol_k,
        "null": receipt["null"]["bitwise_equal"],
    }
