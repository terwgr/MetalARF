"""Executable reference physics for the moist global spectral prototype.

This suite is intentionally compact but complete enough to exercise the full
physics exchange: gray radiation, bulk surface fluxes, implicit vertical
mixing, dry/moist convective adjustment, saturation adjustment, warm-rain and
mixed-phase conversion, and mass-conserving fallout to a finite surface-water
reservoir. It is not presented as a replacement for Arwen's WRF-derived CUDA
schemes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import math

import numpy as np

from ..constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EPSILON,
    GRAVITY_M_S2,
    LATENT_HEAT_FUSION,
    LATENT_HEAT_SUBLIMATION,
    LATENT_HEAT_VAPORIZATION,
    REFERENCE_PRESSURE_PA,
    STEFAN_BOLTZMANN,
    WATER_VAPOR_GAS_CONSTANT,
    NUMBER_MOMENTS,
    WATER_SPECIES,
)
from .exchange import PhysicsExchange, PhysicsResult
from .native_batch import staged_physics_state, staged_surface


@dataclass(frozen=True)
class ReferencePhysicsOptions:
    radiation: bool = True
    surface_fluxes: bool = True
    turbulence: bool = True
    convection: bool = True
    moist_convection: bool = True
    saturation_adjustment: bool = True
    microphysics: bool = True
    convective_relaxation_time_s: float = 7200.0
    convective_reference_rh: float = 0.8
    convective_min_depth_pa: float = 20_000.0
    # Volumetric soil water at which land evaporation runs unstressed;
    # roughly field capacity for a loam.
    soil_wetness_capacity: float = 0.30
    # Cold-top relaxation (the missing ozone/residual-circulation warming
    # a gray absorber cannot supply): points above stratospheric_floor_pa
    # relax one-sidedly toward stratospheric_floor_k.  Zero disables.
    # Sizing (measured on the T533 hour-3 crash states, 2026-08-31): the
    # resolved mountain-wave train cools its cold pocket at 0.0134 K/s,
    # so the one-sided floor balances at deficit = cooling * tau below
    # the reference.  tau=1800 s caps the excursion at 24 K -> the pocket
    # holds near 171 K even with the driver unabated; the previous
    # 21,600 s balanced 290 K below reference - unreachable, both runs
    # died at the 140 K research gate first (139.6-140.0 K, hours
    # 3.0-3.5); 3600 s would cap at 48 K -> 147 K, 7 K from that gate.
    # 5000 Pa depth: the train's cold pockets reach levels 2-3 (8-9 K
    # deep at ring-mean p_full 2693/4766 Pa); the old 2000 Pa left them
    # uncovered.
    stratospheric_floor_k: float = 195.0
    stratospheric_floor_pa: float = 5_000.0
    stratospheric_relaxation_time_s: float = 1_800.0
    # The v7 top-of-model graded wave absorber (sponge_base_pa /
    # sponge_lid_relaxation_time_s) moved to the dycore in v8: it
    # compensates the dycore's rigid p_top lid, so it belongs to the
    # model, not to whichever physics suite is loaded.  Configure it in
    # the run config's [sponge] section (dynamics.MoistHybridModel
    # sponge_base_pa / sponge_lid_relaxation_time_s).
    solar_constant_w_m2: float = 1361.0
    atmospheric_shortwave_absorptivity: float = 0.18
    atmospheric_longwave_emissivity: float = 0.72
    bulk_exchange_coefficient: float = 1.5e-3
    background_diffusivity_m2_s: float = 0.25
    pbl_diffusivity_m2_s: float = 30.0
    pbl_depth_m: float = 1800.0
    cloud_autoconversion_threshold: float = 1.0e-3
    cloud_autoconversion_time_s: float = 1200.0
    ice_to_snow_time_s: float = 1800.0
    snow_to_graupel_time_s: float = 3600.0
    rain_fall_speed_m_s: float = 6.0
    snow_fall_speed_m_s: float = 1.0
    graupel_fall_speed_m_s: float = 3.0
    soil_exchange_time_s: float = 43_200.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"reference physics option {name} must be finite")
        if not 0.0 <= self.atmospheric_shortwave_absorptivity <= 1.0:
            raise ValueError("atmospheric_shortwave_absorptivity must lie in [0,1]")
        if not 0.0 <= self.atmospheric_longwave_emissivity <= 1.0:
            raise ValueError("atmospheric_longwave_emissivity must lie in [0,1]")
        if not 0.0 < self.convective_reference_rh <= 1.0:
            raise ValueError("convective_reference_rh must lie in (0,1]")
        for name in (
            "solar_constant_w_m2", "bulk_exchange_coefficient",
            "cloud_autoconversion_time_s", "ice_to_snow_time_s",
            "snow_to_graupel_time_s", "rain_fall_speed_m_s",
            "snow_fall_speed_m_s", "graupel_fall_speed_m_s",
            "soil_exchange_time_s", "convective_relaxation_time_s",
            "convective_min_depth_pa", "soil_wetness_capacity",
            "stratospheric_floor_pa", "stratospheric_relaxation_time_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"reference physics option {name} must be positive")

    @property
    def identity(self) -> dict[str, object]:
        payload = asdict(self)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {
            "mode": "reference",
            "suite": "gray-surface-implicit-pbl-adjustment-betts-miller-soilbeta-coldtop-mixed-phase-v8",
            "options": payload,
            "options_sha256": hashlib.sha256(raw).hexdigest(),
        }


def _saturation_vapor_pressure(temperature, pressure, xp):
    """Guarded Tetens vapor pressure; returns (e, clipped temperature)."""
    t = xp.clip(temperature, 150.0, 350.0)
    exponent = 17.67 * (t - 273.15) / xp.maximum(t - 29.65, 1.0)
    e = 611.2 * xp.exp(exponent)
    return xp.minimum(e, 0.95 * pressure), t


def saturation_mixing_ratio(temperature, pressure, xp):
    """Liquid-water saturation mixing ratio with guarded Tetens vapor pressure."""
    e, _ = _saturation_vapor_pressure(temperature, pressure, xp)
    return EPSILON * e / xp.maximum(pressure - e, 1.0)


def _implicit_diffuse(field, height, dp, diffusivity, dt_s: float, xp):
    """Backward-Euler flux-form vertical diffusion, no-flux top/bottom.

    The interface coupling dt*K*m_int/dz^2 (m_int = mean adjacent layer
    mass) is one antisymmetric flux shared by both layers, each divided by
    its own dp, so sum(field*dp) is invariant to solver roundoff on any
    spacing. A coupling not divided by the layer's own dp conserves the
    unweighted sum(field) instead and manufactures column-integrated mass
    wherever dp varies. On uniform dp this reduces exactly to dt*K/dz^2.
    """
    x = field
    nz = x.shape[0]
    if nz < 2:
        return x.copy()
    dz_interface = xp.maximum(xp.abs(height[:-1] - height[1:]), 1.0)
    k_interface = 0.5 * (diffusivity[:-1] + diffusivity[1:])
    mass_interface = 0.5 * (dp[:-1] + dp[1:])
    coupling = float(dt_s) * k_interface * mass_interface / (
        dz_interface * dz_interface
    )
    a = xp.zeros_like(x)
    b = xp.ones_like(x)
    c = xp.zeros_like(x)
    a[1:] = -coupling / dp[1:]
    c[:-1] = -coupling / dp[:-1]
    b[:-1] += coupling / dp[:-1]
    b[1:] += coupling / dp[1:]

    c_prime = xp.zeros_like(x)
    d_prime = xp.zeros_like(x)
    c_prime[0] = c[0] / b[0]
    d_prime[0] = x[0] / b[0]
    for k in range(1, nz):
        denominator = b[k] - a[k] * c_prime[k - 1]
        c_prime[k] = c[k] / denominator if k < nz - 1 else 0.0
        d_prime[k] = (x[k] - a[k] * d_prime[k - 1]) / denominator
    out = xp.empty_like(x)
    out[-1] = d_prime[-1]
    for k in range(nz - 2, -1, -1):
        out[k] = d_prime[k] - c_prime[k] * out[k + 1]
    return out


def _column_water(q, dp, xp):
    return xp.sum(sum(q[name] for name in WATER_SPECIES) * dp / GRAVITY_M_S2, axis=0)


# ---------------------------------------------------------------------------
# CuPy fused fast paths for the column physics chains.
#
# The numpy code in ReferencePhysics is the specification; every fused
# kernel below computes the same arithmetic in the same per-element
# operation order.  The kernels compile with -fmad=false so each
# multiply-add pair rounds exactly like the two numpy ufuncs it replaces
# (the same choice core/moist.py made for the WRF scalar update); the only
# reassociation is in the per-column level sums of the Betts-Miller kernel,
# a last-bits floating-point reordering of numpy's pairwise xp.sum.
#
# Per-step launch census that motivated each fusion (nlev=20 stack,
# measured 2026-08-30): _saturation_adjust ~970 elementwise launches,
# _microphysics + _fallout ~1,990, _betts_miller a per-level Python ascent
# loop of guarded-Tetens chains, _convective_adjust ~66 per pair
# iteration.  Each becomes one kernel per call site below.
#
# The kernel sources are module-level strings so a host-side harness can
# compile and exercise the same C against the numpy reference without a
# GPU (done at authoring time: bit-exact at float64 across random and
# adversarial suites, clang -ffp-contract=off); the standalone
# gpu_selftest.py harness repeats that comparison on the real CUDA build.
# ---------------------------------------------------------------------------

#: Compiled kernels, one entry per (name, DEVICE).  See
#: :func:`~woof.globe.spectral.backend.device_cache_key`; gate CARD-1.
_CUPY_KERNELS: dict[tuple, object] = {}


def _cupy_kernel(name: str, build):
    """Build-once-per-device cache; each kernel compiles lazily at first
    GPU call, once for every card the process opens."""
    from ..spectral.backend import device_cache_key

    key = (name, *device_cache_key())
    kernel = _CUPY_KERNELS.get(key)
    if kernel is None:
        kernel = build()
        _CUPY_KERNELS[key] = kernel
    return kernel


def _kernel_defines() -> str:
    """C macros for the physical constants at full float64 repr precision.

    Generated from the imported constants so the C and numpy paths cannot
    drift apart; every use site casts to the kernel dtype T, matching how
    numpy casts a Python float scalar into a float32 array operation.
    """
    return (
        f"#define GPUWM_REF_EPSILON {EPSILON!r}\n"
        f"#define GPUWM_REF_GRAV {GRAVITY_M_S2!r}\n"
        f"#define GPUWM_REF_RD {DRY_AIR_GAS_CONSTANT!r}\n"
        f"#define GPUWM_REF_CP {DRY_AIR_CP!r}\n"
        f"#define GPUWM_REF_RV {WATER_VAPOR_GAS_CONSTANT!r}\n"
        f"#define GPUWM_REF_LV {LATENT_HEAT_VAPORIZATION!r}\n"
        f"#define GPUWM_REF_LV2 {LATENT_HEAT_VAPORIZATION ** 2!r}\n"
        f"#define GPUWM_REF_LS {LATENT_HEAT_SUBLIMATION!r}\n"
        f"#define GPUWM_REF_LVCP {LATENT_HEAT_VAPORIZATION / DRY_AIR_CP!r}\n"
        f"#define GPUWM_REF_LFCP {LATENT_HEAT_FUSION / DRY_AIR_CP!r}\n"
        f"#define GPUWM_REF_LSCP {LATENT_HEAT_SUBLIMATION / DRY_AIR_CP!r}\n"
        f"#define GPUWM_REF_RDCP {DRY_AIR_GAS_CONSTANT / DRY_AIR_CP!r}\n"
    )


# saturation_mixing_ratio as a device function: the guarded Tetens chain in
# _saturation_vapor_pressure + the mixing-ratio line, same op order.
_QSAT_PREAMBLE = """
template<typename T>
__device__ T gpuwm_ref_qsat(T temperature, T pressure) {
    T tclip = min(max(temperature, (T)150.0), (T)350.0);
    T denom = max(tclip - (T)29.65, (T)1.0);
    T e = (T)611.2 * exp((T)17.67 * (tclip - (T)273.15) / denom);
    e = min(e, (T)0.95 * pressure);
    return (T)GPUWM_REF_EPSILON * e / max(pressure - e, (T)1.0);
}
"""


# _saturation_adjust: the four Newton iterations are data-independent per
# point (each reads only that point's T/qv/qc/p), so one kernel runs the
# whole loop internally.  ** 2 appears as denom * denom because numpy's
# `** 2` takes the square fast path, which is exactly x * x.
_SATURATION_ADJUST_SRC = """
T t = t_in;
T qv = qv_in;
T qc = qc_in;
T pres = p_in;
T total = (T)0.0;
for (int newton = 0; newton < 4; ++newton) {
    T tclip = min(max(t, (T)150.0), (T)350.0);
    T denom = max(tclip - (T)29.65, (T)1.0);
    T e = (T)611.2 * exp((T)17.67 * (tclip - (T)273.15) / denom);
    e = min(e, (T)0.95 * pres);
    T dry = max(pres - e, (T)1.0);
    T qsat = (T)GPUWM_REF_EPSILON * e / dry;
    T dqsat = qsat * (pres / dry) * (T)17.67 * (T)243.5 / (denom * denom);
    T gamma = (T)GPUWM_REF_LVCP * dqsat;
    T phase = (qv - qsat) / ((T)1.0 + gamma);
    phase = max(phase, -qc);
    qv -= phase;
    qc += phase;
    t += (T)GPUWM_REF_LVCP * phase;
    total += phase;
}
t_out = t;
qv_out = qv;
qc_out = qc;
phase_out = total;
"""


def _build_saturation_adjust_kernel():
    import cupy

    return cupy.ElementwiseKernel(
        "T t_in, T qv_in, T qc_in, T p_in",
        "T t_out, T qv_out, T qc_out, T phase_out",
        _SATURATION_ADJUST_SRC,
        "gpuwm_reference_saturation_adjust",
        preamble=_kernel_defines(),
        options=("-fmad=false",),
    )


# _microphysics conversion chain (everything between the theta*exner entry
# and the fallout calls): rain evaporation, autoconversion, accretion,
# freezing, ice->snow->graupel, melting, and the feedback-corrected
# sublimation ladder, sequenced exactly like the numpy statements.
_MICROPHYSICS_SRC = """
T t = t_in;
T qv = qv_in;
T qc = qc_in;
T qr = qr_in;
T qi = qi_in;
T qs = qs_in;
T qg = qg_in;
T pres = p_in;

T qsat = gpuwm_ref_qsat(t, pres);
T deficit = max(qsat - qv, (T)0.0);
T rain_evap = min(qr, deficit * rain_evap_f);
qr -= rain_evap;
qv += rain_evap;
t -= (T)GPUWM_REF_LVCP * rain_evap;

T autoconvert = max(qc - auto_thresh, (T)0.0) * auto_f;
qc -= autoconvert;
qr += autoconvert;
T accrete = min(qc, accrete_c * qc * sqrt(max(qr, (T)0.0)));
qc -= accrete;
qr += accrete;

T freeze_w = min(max(((T)273.15 - t) / (T)15.0, (T)0.0), (T)1.0);
T freeze_frac = freeze_w * freeze_f;
T freeze_cloud = qc * freeze_frac;
T freeze_rain = qr * freeze_frac * (T)0.5;
qc -= freeze_cloud;
qr -= freeze_rain;
qi += freeze_cloud;
qs += freeze_rain;
t += (T)GPUWM_REF_LFCP * (freeze_cloud + freeze_rain);

T ice_to_snow = qi * ice_snow_f * min(max(qi / (T)1.0e-4, (T)0.0), (T)1.0);
qi -= ice_to_snow;
qs += ice_to_snow;
T snow_to_graupel =
    qs * snow_graupel_f * min(max(qc / (T)5.0e-4, (T)0.0), (T)1.0);
qs -= snow_to_graupel;
qg += snow_to_graupel;

T melt_w = min(max((t - (T)273.15) / (T)5.0, (T)0.0), (T)1.0);
T melt_frac = melt_w * melt_f;
T melt_ice = qi * melt_frac;
T melt_snow = qs * melt_frac;
T melt_graupel = qg * melt_frac;
qi -= melt_ice;
qc += melt_ice;
qs -= melt_snow;
qg -= melt_graupel;
qr += melt_snow + melt_graupel;
t -= (T)GPUWM_REF_LFCP * (melt_ice + melt_snow + melt_graupel);

T qsat2 = gpuwm_ref_qsat(t, pres);
T cold_w = min(max(((T)273.15 - t) / (T)5.0, (T)0.0), (T)1.0);
T subl_frac = cold_w * subl_f;
T latent_feedback = (T)1.0 + (T)GPUWM_REF_LSCP
    * (qsat2 * (T)GPUWM_REF_LS / ((T)GPUWM_REF_RV * t * t));
T remaining = max(qsat2 - qv, (T)0.0) / latent_feedback;
T subl = min(qi * subl_frac, remaining);
qi -= subl;
qv += subl;
remaining -= subl;
t -= (T)GPUWM_REF_LSCP * subl;
subl = min(qs * subl_frac, remaining);
qs -= subl;
qv += subl;
remaining -= subl;
t -= (T)GPUWM_REF_LSCP * subl;
subl = min(qg * subl_frac, remaining);
qg -= subl;
qv += subl;
t -= (T)GPUWM_REF_LSCP * subl;

t_out = t;
qv_out = qv;
qc_out = qc;
qr_out = qr;
qi_out = qi;
qs_out = qs;
qg_out = qg;
"""


def _build_microphysics_kernel():
    import cupy

    return cupy.ElementwiseKernel(
        "T t_in, T qv_in, T qc_in, T qr_in, T qi_in, T qs_in, T qg_in, "
        "T p_in, T rain_evap_f, T auto_thresh, T auto_f, T accrete_c, "
        "T freeze_f, T ice_snow_f, T snow_graupel_f, T melt_f, T subl_f",
        "T t_out, T qv_out, T qc_out, T qr_out, T qi_out, T qs_out, T qg_out",
        _MICROPHYSICS_SRC,
        "gpuwm_reference_microphysics_chain",
        preamble=_kernel_defines() + _QSAT_PREAMBLE,
        options=("-fmad=false",),
    )


# _fallout: sedimentation is sequential top-down (level k+1 receives level
# k's flux before it sheds its own), so levels cannot run as one stacked
# elementwise pass; instead one thread walks each column, turning the
# nlev-deep Python loop of ~7 launches per level into a single launch.
_FALLOUT_SRC = """
const long long col = i;
T acc = (T)0.0;
for (int k = 0; k < nz; ++k) {
    const long long idx = (long long)k * plane + col;
    T z_a;
    T z_b;
    if (k < nz - 1) {
        z_a = phi[idx] / (T)GPUWM_REF_GRAV;
        z_b = phi[idx + plane] / (T)GPUWM_REF_GRAV;
    } else {
        z_a = phi[(long long)(nz - 2) * plane + col] / (T)GPUWM_REF_GRAV;
        z_b = phi[(long long)(nz - 1) * plane + col] / (T)GPUWM_REF_GRAV;
    }
    T dz = max(fabs(z_a - z_b), (T)20.0);
    T fraction = (T)1.0 - exp(-fs_dt / dz);
    fraction = min(max(fraction, (T)0.0), (T)1.0);
    T removed_mixing = species[idx] * fraction;
    T mass_k = dp[idx] / (T)GPUWM_REF_GRAV;
    T removed_mass = removed_mixing * mass_k;
    species[idx] -= removed_mixing;
    if (k + 1 < nz) {
        T mass_next = dp[idx + plane] / (T)GPUWM_REF_GRAV;
        species[idx + plane] += removed_mass / max(mass_next, (T)1.0e-12);
    } else {
        acc += removed_mass;
    }
}
precip = acc;
"""


def _build_fallout_kernel():
    import cupy

    return cupy.ElementwiseKernel(
        "raw T dp, raw T phi, T fs_dt, int32 nz, int64 plane",
        "raw T species, T precip",
        _FALLOUT_SRC,
        "gpuwm_reference_fallout_column",
        preamble=_kernel_defines(),
        options=("-fmad=false",),
    )


# _betts_miller_adjust: the parcel ascent is a level recurrence
# (parcel_t[k] needs parcel_t[k+1]), so one thread walks each column:
# ascent, cloud-layer detection, the closure sums, and the application all
# in one launch.  The level sums reassociate numpy's pairwise xp.sum into
# sequential order - the one place the fused path is not
# operation-for-operation identical.  parcel_virtual deliberately uses the
# final (cloud-top) parcel_q at every level because the numpy
# specification broadcasts the loop-final parcel_q.
_BETTS_MILLER_SRC = """
const long long col = i;
const long long bottom = (long long)(nz - 1) * plane + col;
T pt_bot = temp[bottom];
parcel[bottom] = pt_bot;
T qsat_bot = gpuwm_ref_qsat(pt_bot, p[bottom]);
T pq = min(qv[bottom], qsat_bot);
bool saturated = pq >= qsat_bot - (T)1.0e-12;
int lcl = saturated ? (nz - 1) : -1;
for (int k = nz - 2; k >= 0; --k) {
    const long long idx = (long long)k * plane + col;
    const long long below = idx + plane;
    const T pk = p[idx];
    const T pk1 = p[below];
    const T pt_below = parcel[below];
    const T dlnp = log(pk / pk1);
    const T dry_t = pt_below * pow(pk / pk1, (T)GPUWM_REF_RDCP);
    const T qs_prev = gpuwm_ref_qsat(pt_below, pk1);
    const T moist_rate =
        ((T)GPUWM_REF_RD * pt_below + (T)GPUWM_REF_LV * qs_prev)
        / ((T)GPUWM_REF_CP
           + (T)GPUWM_REF_LV2 * qs_prev * (T)GPUWM_REF_EPSILON
             / ((T)GPUWM_REF_RD * (pt_below * pt_below)));
    const T moist_t = pt_below + moist_rate * dlnp;
    const bool is_sat = saturated || (pq >= gpuwm_ref_qsat(dry_t, pk));
    const T ptk = is_sat ? moist_t : dry_t;
    parcel[idx] = ptk;
    if (is_sat) {
        pq = gpuwm_ref_qsat(ptk, pk);
    }
    if (is_sat && !saturated) {
        lcl = k;
    }
    saturated = is_sat;
}
const T qfac = (T)1.0 + (T)0.61 * pq;
int cloud_top = 0;
bool has_cloud = false;
for (int k = 0; k < nz; ++k) {
    const long long idx = (long long)k * plane + col;
    const bool buoyant =
        (parcel[idx] * qfac > tv[idx]) && (k <= lcl) && (lcl >= 0);
    if (buoyant && !has_cloud) {
        cloud_top = k;
        has_cloud = true;
    }
}
const int base = lcl < 0 ? 0 : (lcl > nz - 1 ? nz - 1 : lcl);
const T p_base = p[(long long)base * plane + col];
const T p_top = p[(long long)cloud_top * plane + col];
const bool deep = has_cloud && ((p_base - p_top) >= min_depth);
T drying_neg = (T)0.0;
T moistening = (T)0.0;
T heating_raw = (T)0.0;
T cloud_mass = (T)0.0;
for (int k = 0; k < nz; ++k) {
    if (!deep || k < cloud_top || k > lcl) {
        continue;
    }
    const long long idx = (long long)k * plane + col;
    const T mass_k = dp[idx] / (T)GPUWM_REF_GRAV;
    const T dtr = (parcel[idx] - temp[idx]) * relax;
    const T refq = rh * gpuwm_ref_qsat(parcel[idx], p[idx]);
    const T dq = (refq - qv[idx]) * relax;
    drying_neg += min(dq, (T)0.0) * mass_k;
    moistening += max(dq, (T)0.0) * mass_k;
    heating_raw += dtr * mass_k;
    cloud_mass += mass_k;
}
const T drying = -drying_neg;
const bool precipitating = deep && (drying > moistening);
const T scale = (deep && !precipitating && (moistening > (T)0.0))
    ? drying / max(moistening, (T)1.0e-30)
    : (T)1.0;
T dq_mass = (T)0.0;
for (int k = 0; k < nz; ++k) {
    if (!deep || k < cloud_top || k > lcl) {
        continue;
    }
    const long long idx = (long long)k * plane + col;
    const T refq = rh * gpuwm_ref_qsat(parcel[idx], p[idx]);
    T dq = (refq - qv[idx]) * relax;
    if (dq > (T)0.0) {
        dq = dq * scale;
    }
    dq_mass += dq * (dp[idx] / (T)GPUWM_REF_GRAV);
    qv[idx] = qv[idx] + dq;
}
const T precip_col = precipitating ? -dq_mass : (T)0.0;
const T correction = deep
    ? ((T)GPUWM_REF_LV * precip_col - (T)GPUWM_REF_CP * heating_raw)
      / ((T)GPUWM_REF_CP * max(cloud_mass, (T)1.0e-6))
    : (T)0.0;
for (int k = 0; k < nz; ++k) {
    if (!deep || k < cloud_top || k > lcl) {
        continue;
    }
    const long long idx = (long long)k * plane + col;
    const T dtr = (parcel[idx] - temp[idx]) * relax;
    temp[idx] = temp[idx] + (dtr + correction);
}
precip = precip_col;
deep_out = deep ? 1 : 0;
"""


def _build_betts_miller_kernel():
    import cupy

    return cupy.ElementwiseKernel(
        "raw T p, raw T dp, raw T tv, T relax, T rh, T min_depth, "
        "int32 nz, int64 plane",
        "raw T temp, raw T qv, raw T parcel, T precip, int32 deep_out",
        _BETTS_MILLER_SRC,
        "gpuwm_reference_betts_miller_column",
        preamble=_kernel_defines() + _QSAT_PREAMBLE,
        options=("-fmad=false",),
    )


# _convective_adjust: the four sweeps over adjacent pairs are sequential
# within a column (pair k feeds pair k+1) and independent across columns,
# so one thread runs all sweeps for its column; the per-pair where/count
# chain (~66 launches per pair iteration) becomes one launch per call.
_CONVECTIVE_ADJUST_SRC = """
const long long col = i;
long long count = 0;
for (int sweep = 0; sweep < 4; ++sweep) {
    for (int k = 0; k < nz - 1; ++k) {
        const long long a = (long long)k * plane + col;
        const long long b = a + plane;
        const T th_a = theta[a];
        const T th_b = theta[b];
        if (th_a < th_b) {
            const T m_a = dp[a] / (T)GPUWM_REF_GRAV;
            const T m_b = dp[b] / (T)GPUWM_REF_GRAV;
            const T tm = max(m_a + m_b, (T)1.0e-12);
            const T mixed_theta = (m_a * th_a + m_b * th_b) / tm;
            theta[a] = mixed_theta;
            theta[b] = mixed_theta;
            T blend;
            blend = (m_a * qv[a] + m_b * qv[b]) / tm;
            qv[a] = blend;
            qv[b] = blend;
            blend = (m_a * qc[a] + m_b * qc[b]) / tm;
            qc[a] = blend;
            qc[b] = blend;
            blend = (m_a * qr[a] + m_b * qr[b]) / tm;
            qr[a] = blend;
            qr[b] = blend;
            blend = (m_a * qi[a] + m_b * qi[b]) / tm;
            qi[a] = blend;
            qi[b] = blend;
            blend = (m_a * qs[a] + m_b * qs[b]) / tm;
            qs[a] = blend;
            qs[b] = blend;
            blend = (m_a * qg[a] + m_b * qg[b]) / tm;
            qg[a] = blend;
            qg[b] = blend;
            ++count;
        }
    }
}
mixed = count;
"""


def _build_convective_adjust_kernel():
    import cupy

    return cupy.ElementwiseKernel(
        "raw T dp, int32 nz, int64 plane",
        "raw T theta, raw T qv, raw T qc, raw T qr, raw T qi, raw T qs, "
        "raw T qg, int64 mixed",
        _CONVECTIVE_ADJUST_SRC,
        "gpuwm_reference_convective_pair_mix",
        preamble=_kernel_defines(),
        options=("-fmad=false",),
    )


class ReferencePhysics:
    # In-situ component capture slot (woof.globe.insitu.capture):
    # marked after every process function in step().  Read-only by contract.
    observer = None

    def __init__(self, backend, options: ReferencePhysicsOptions | None = None):
        self.backend = backend
        self.options = options or ReferencePhysicsOptions()
        self.metal_context = None

    #: How a call's scalar readings are formed from its bands'
    #: (physics.banding): the maxima fold, the two column counts add
    #: exactly, everything else is the same on every band.  The grid
    #: means are planes the bands hand back and :meth:`finish` reduces
    #: once over the globe.
    DIAGNOSTIC_MERGE = {
        "maximum_cosine_zenith": "max",
        "convective_pair_mixes": "count",
        "deep_convective_columns": "count",
        "maximum_column_condensation_kg_kg": "max",
        "maximum_local_water_repair_kg_m2": "max",
        "mean_outgoing_longwave_w_m2": "column_mean",
        "mean_surface_evaporation_kg_m2_s": "column_mean",
    }
    #: The planes and the reading each one's whole-grid mean is.
    _PLANE_MEANS = (
        ("net_surface_radiation", "mean_net_surface_radiation_w_m2"),
        ("evaporation", "mean_surface_evaporation_kg_m2_s"),
        ("rain_rate", "mean_rain_rate_kg_m2_s"),
        ("rain_amount", "mean_rain_rate_kg_m2_s"),
        ("snow_rate", "mean_snow_rate_kg_m2_s"),
        ("snow_amount", "mean_snow_rate_kg_m2_s"),
        ("graupel_rate", "mean_graupel_rate_kg_m2_s"),
        ("graupel_amount", "mean_graupel_rate_kg_m2_s"),
    )

    def _observe(self, name, u, v, theta, q, band=None) -> None:
        if self.observer is not None:
            # ComponentCapture now keeps resident Metal prognostics on the
            # device and computes its small mean/max record there.  The
            # previous per-field to_numpy calls copied four full grid fields
            # at every component mark even though the ledger only persists
            # eight scalars per mark.
            fields = {"theta": theta, "qv": q["qv"], "u": u, "v": v}
            if band is None:
                self.observer.mark(name, fields, self.backend.xp)
            else:
                self.observer.mark(name, fields, self.backend.xp, band=band)

    def finish(self, band_diagnostics, band_metadata, planes, surface,
               physics_state, *, metadata_in=None, columns=None, dt_s=None):
        """The call's diagnostics and namespace metadata from its bands'
        (physics.banding): every grid mean once over the assembled plane,
        in the array dtype the one-call form reduced."""
        from .banding import merge_band_metadata, merge_band_scalars, to_host

        diagnostics = merge_band_scalars(
            band_diagnostics, self.DIAGNOSTIC_MERGE, columns=columns,
            what="reference physics diagnostic")
        for plane, name in self._PLANE_MEANS:
            if plane not in planes:
                if name not in diagnostics:
                    raise KeyError(
                        f"reference physics omitted both plane {plane!r} "
                        f"and diagnostic {name!r}"
                    )
                continue
            values = to_host(planes[plane])
            if plane in ("rain_amount", "snow_amount", "graupel_amount"):
                values = values / max(float(dt_s), 1.0e-12)
            diagnostics[name] = float(np.mean(values))
        diagnostics["mean_convective_rain_rate_kg_m2_s"] = float(
            np.mean(to_host(planes["convective_precip"]))
            / max(float(dt_s), 1.0e-12)
        )
        if self.observer is not None and hasattr(self.observer, "close_call"):
            self.observer.close_call(self.backend.xp)
        return diagnostics, merge_band_metadata(band_metadata, {})

    @property
    def identity(self) -> dict[str, object]:
        return self.options.identity

    def _metal_maximum(self, values) -> float:
        """Reduce a resident FP64 Global plane, materializing only partials."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        if not isinstance(values, MetalArray):
            raise TypeError("Global Metal maximum input must be a MetalArray")
        if values.buffer.context is not self.metal_context:
            raise ValueError("Global Metal maximum input belongs to another context")
        if values.dtype != np.dtype(np.float64):
            raise TypeError("Global Metal maximum input must be FP64")
        source = values.view((values.size,), dtype=np.float64)
        chunk_size = 256
        partial_count = max(1, (source.size + chunk_size - 1) // chunk_size)
        partial_max = MetalArrayModule(self.metal_context).empty(
            (partial_count,), dtype=np.float64
        )
        self.metal_context.submit_kernel(
            "global_semilag_max_fragment",
            {"values": source, "partial_max": partial_max},
            {"count": source.size, "chunk_size": chunk_size,
             "partial_count": partial_count},
            exact_policy=False, grid_size=partial_count,
            threads_per_group=kernel_specs.GLOBAL_SEMILAG_MAX_FRAGMENT_THREADS_PER_GROUP,
        )
        maximum = MetalArrayModule(self.metal_context).empty((1,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_semilag_max_finalize_fragment",
            {"partial_max": partial_max, "maximum": maximum},
            {"partial_count": partial_count}, exact_policy=False,
            grid_size=1,
            threads_per_group=kernel_specs.GLOBAL_SEMILAG_MAX_FINALIZE_FRAGMENT_THREADS_PER_GROUP,
        )
        return float(maximum.to_numpy()[0])

    def _metal_saturation_pair_maximum(self, first, second) -> float:
        """Reduce the per-column sum of two resident condensation planes."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        if (not isinstance(first, MetalArray) or not isinstance(second, MetalArray)
                or first.dtype != np.dtype(np.float64)
                or second.dtype != np.dtype(np.float64)
                or first.shape != second.shape
                or first.buffer.context is not self.metal_context
                or second.buffer.context is not self.metal_context):
            raise TypeError("Global saturation reductions require matching resident FP64 planes")
        count = first.size
        chunk_size = 256
        partial_count = max(1, (count + chunk_size - 1) // chunk_size)
        metal = MetalArrayModule(self.metal_context)
        partial_max = metal.empty((partial_count,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_saturation_pair_max_fragment",
            {"first": first.view((count,), dtype=np.float64),
             "second": second.view((count,), dtype=np.float64),
             "partial_max": partial_max},
            {"count": count, "chunk_size": chunk_size,
             "partial_count": partial_count}, exact_policy=False,
            grid_size=partial_count,
            threads_per_group=kernel_specs.GLOBAL_SATURATION_PAIR_MAX_FRAGMENT_THREADS_PER_GROUP,
        )
        maximum = metal.empty((1,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_semilag_max_finalize_fragment",
            {"partial_max": partial_max, "maximum": maximum},
            {"partial_count": partial_count}, exact_policy=False,
            grid_size=1,
            threads_per_group=kernel_specs.GLOBAL_SEMILAG_MAX_FINALIZE_FRAGMENT_THREADS_PER_GROUP,
        )
        return float(maximum.to_numpy()[0])

    def _metal_saturation_closure(self, theta, q, dp, before_water, surface_water):
        """Clamp and repair resident saturation state with scalar readbacks."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(theta.shape)
        nlevel = int(shape[0])
        ncolumn = int(np.prod(shape[1:]))
        metal = MetalArrayModule(self.metal_context)

        def device(value, target_shape):
            if isinstance(value, MetalArray):
                if (value.buffer.context is not self.metal_context
                        or value.dtype != np.dtype(np.float64)
                        or value.size != int(np.prod(target_shape))):
                    raise ValueError("Global saturation closure input has invalid Metal storage")
                resident = metal.asarray(value)
                return resident.view((resident.size,), dtype=np.float64)
            host = np.ascontiguousarray(
                np.broadcast_to(np.asarray(value, dtype=np.float64), target_shape)
            )
            return metal.asarray(host.reshape(-1))

        resources = {
            "theta": device(theta, shape),
            **{name: device(q[name], shape) for name in WATER_SPECIES},
            "dp": device(dp, shape),
            "before_water": device(before_water, shape[1:]),
            "surface_water": device(surface_water, shape[1:]),
            "gravity": metal.asarray(np.asarray([GRAVITY_M_S2], dtype=np.float64)),
        }
        for name in WATER_SPECIES:
            resources[f"{name}_out"] = metal.empty((nlevel * ncolumn,), dtype=np.float64)
        resources["surface_water_out"] = metal.empty((ncolumn,), dtype=np.float64)
        resources["partial_status"] = metal.empty((ncolumn,), dtype=np.uint32)
        resources["partial_repair_max"] = metal.empty((ncolumn,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_saturation_closure_fragment", resources,
            {"nlevel": nlevel, "ncolumn": ncolumn}, exact_policy=False,
            grid_size=ncolumn,
            threads_per_group=kernel_specs.GLOBAL_SATURATION_CLOSURE_FRAGMENT_THREADS_PER_GROUP,
        )
        status = metal.empty((1,), dtype=np.uint32)
        maximum_repair = metal.empty((1,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_saturation_closure_finalize_fragment",
            {"partial_status": resources["partial_status"],
             "partial_repair_max": resources["partial_repair_max"],
             "status": status, "maximum_repair": maximum_repair},
            {"count": ncolumn}, exact_policy=False, grid_size=1,
            threads_per_group=kernel_specs.GLOBAL_SATURATION_CLOSURE_FINALIZE_FRAGMENT_THREADS_PER_GROUP,
        )
        flags = int(status.to_numpy()[0])
        if flags & 0x3F:
            for index, name in enumerate(WATER_SPECIES):
                if flags & (1 << index):
                    raise FloatingPointError(
                        f"reference physics produced negative {name}: min < -1e-10"
                    )
        if flags & 64:
            raise FloatingPointError("reference physics produced non-finite theta")
        if flags & 128:
            raise FloatingPointError("reference physics exhausted the surface-water reservoir")
        if flags & 256:
            raise FloatingPointError(
                "physics water repair requires more surface water than available"
            )
        for name in WATER_SPECIES:
            q[name] = resources[f"{name}_out"].view(shape, dtype=np.float64)
        return (
            q,
            resources["surface_water_out"].view(shape[1:], dtype=np.float64),
            float(maximum_repair.to_numpy()[0]),
        )

    def _metal_column_water(self, q, dp, surface_water, shape):
        """Compute atmosphere + surface water per column without field readback."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(shape)
        nlevel = int(shape[0])
        ncolumn = int(np.prod(shape[1:]))
        metal = MetalArrayModule(self.metal_context)
        resources = {}
        for name in WATER_SPECIES:
            value = q[name]
            if (not isinstance(value, MetalArray)
                    or value.buffer.context is not self.metal_context
                    or value.dtype != np.dtype(np.float64)
                    or tuple(value.shape) != shape):
                raise ValueError(
                    f"Global column-water {name} must be FP64 on this Metal context"
                )
            resources[name] = value.view((nlevel * ncolumn,), dtype=np.float64)

        def upload(value, target_shape):
            if isinstance(value, MetalArray):
                if (value.buffer.context is not self.metal_context
                        or value.dtype != np.dtype(np.float64)
                        or tuple(value.shape) != tuple(target_shape)):
                    raise ValueError(
                        "Global column-water fields must be matching FP64 "
                        "arrays on this Metal context"
                    )
                return value.view((int(np.prod(target_shape)),), dtype=np.float64)
            host = np.ascontiguousarray(np.broadcast_to(
                np.asarray(value, dtype=np.float64), target_shape
            ).reshape(-1))
            return metal.asarray(host)

        if isinstance(surface_water, MetalArray):
            if (surface_water.buffer.context is not self.metal_context
                    or surface_water.dtype != np.dtype(np.float64)
                    or tuple(surface_water.shape) != tuple(shape[1:])):
                raise ValueError(
                    "Global surface water must be an FP64 plane on this Metal context"
                )
            surface_water_device = surface_water.view((ncolumn,), dtype=np.float64)
        else:
            surface_water_device = upload(surface_water, shape[1:])
        resources.update({
            "dp": upload(dp, shape),
            "surface_water": surface_water_device,
            "gravity": upload(np.asarray([GRAVITY_M_S2], dtype=np.float64), (1,)),
            "before_water": metal.empty((ncolumn,), dtype=np.float64),
        })
        self.metal_context.submit_kernel(
            "global_column_water_before_fragment", resources,
            {"nlevel": nlevel, "ncolumn": ncolumn}, exact_policy=False,
            grid_size=ncolumn,
            threads_per_group=(
                kernel_specs.GLOBAL_COLUMN_WATER_BEFORE_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return resources["before_water"].view(shape[1:], dtype=np.float64)

    def _metal_add_surface_water(self, water, precip):
        """Return the resident surface-water plane after one precipitation input."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        if (not isinstance(water, MetalArray)
                or water.buffer.context is not self.metal_context
                or water.dtype != np.dtype(np.float64)
                or water.ndim != 2):
            raise ValueError("Global surface-water reservoir must be a 2-D FP64 MetalArray")
        shape = tuple(water.shape)
        count = int(water.size)
        metal = MetalArrayModule(self.metal_context)
        if isinstance(precip, MetalArray):
            if (precip.buffer.context is not self.metal_context
                    or precip.dtype != np.dtype(np.float64)
                    or tuple(precip.shape) != shape):
                raise ValueError("Global precipitation plane does not match surface water")
            precip_device = precip.view((count,), dtype=np.float64)
        else:
            host = np.ascontiguousarray(np.broadcast_to(
                np.asarray(precip, dtype=np.float64), shape
            ).reshape(-1))
            precip_device = metal.asarray(host)
        updated = metal.empty((count,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_surface_water_add_fragment",
            {"water": water.view((count,), dtype=np.float64),
             "precip": precip_device, "water_out": updated},
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SURFACE_WATER_ADD_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return updated.view(shape, dtype=np.float64)

    def _metal_surface_flux_preparation(self, exchange, surface, u, v, theta, qv):
        """Derive the per-column surface flux inputs from resident state."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(surface.temperature_k.shape)
        count = int(np.prod(shape))
        for name, value in (("u", u), ("v", v), ("theta", theta), ("qv", qv)):
            if (not isinstance(value, MetalArray)
                    or value.buffer.context is not self.metal_context
                    or value.dtype != np.dtype(np.float64)):
                raise ValueError(
                    f"Global surface flux {name} must be FP64 on this Metal context"
                )
        if (tuple(theta.shape[1:]) != shape
                or tuple(qv.shape) != tuple(theta.shape)
                or tuple(u.shape) != tuple(theta.shape)
                or tuple(v.shape) != tuple(theta.shape)):
            raise ValueError("Global surface flux prognostics have incompatible shapes")
        metal = MetalArrayModule(self.metal_context)

        def upload(value):
            if isinstance(value, MetalArray):
                if (value.buffer.context is not self.metal_context
                        or value.dtype != np.dtype(np.float64)
                        or tuple(value.shape) != shape):
                    raise ValueError(
                        "Global surface-flux input must be an FP64 plane on this Metal context"
                    )
                return value.view((count,), dtype=np.float64)
            host = np.ascontiguousarray(np.broadcast_to(
                np.asarray(value, dtype=np.float64), shape
            ).reshape(-1))
            return metal.asarray(host)

        if isinstance(surface.water_kg_m2, MetalArray):
            if (surface.water_kg_m2.buffer.context is not self.metal_context
                    or surface.water_kg_m2.dtype != np.dtype(np.float64)
                    or tuple(surface.water_kg_m2.shape) != shape):
                raise ValueError(
                    "Global surface water must be an FP64 plane on this Metal context"
                )
            surface_water = surface.water_kg_m2.view((count,), dtype=np.float64)
        else:
            surface_water = upload(surface.water_kg_m2)

        resources = {
            "theta": theta[-1].view((count,), dtype=np.float64),
            "exner_bottom": upload(exchange.exner[-1]),
            "qv": qv[-1].view((count,), dtype=np.float64),
            "p_full_bottom": upload(exchange.p_full[-1]),
            "p_surface": upload(exchange.p_half[-1]),
            "u": u[-1].view((count,), dtype=np.float64),
            "v": v[-1].view((count,), dtype=np.float64),
            "surface_temperature": upload(surface.temperature_k),
            "surface_water": surface_water,
            "dt": metal.asarray(np.asarray([exchange.dt_s], dtype=np.float64)),
            "roughness": upload(surface.roughness_m),
            "soil_water_fraction": upload(surface.soil_water_fraction[0]),
            "land_fraction": upload(surface.land_fraction),
            "dry_air_gas_constant": metal.asarray(np.asarray(
                [DRY_AIR_GAS_CONSTANT], dtype=np.float64)),
            "epsilon": metal.asarray(np.asarray([EPSILON], dtype=np.float64)),
            "bulk_exchange_coefficient": metal.asarray(np.asarray(
                [self.options.bulk_exchange_coefficient], dtype=np.float64)),
            "soil_wetness_capacity": metal.asarray(np.asarray(
                [self.options.soil_wetness_capacity], dtype=np.float64)),
        }
        for name in (
            "air_temperature", "rho_bottom", "wind", "cd",
            "saturation_mixing_ratio", "beta",
            "available_water_rate",
        ):
            resources[name] = metal.empty((count,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_surface_flux_prep_resident_fragment", resources,
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SURFACE_FLUX_PREP_RESIDENT_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return {
            name: resources[name].view(shape, dtype=np.float64)
            for name in (
                "air_temperature", "rho_bottom", "wind", "cd",
                "saturation_mixing_ratio", "beta",
                "available_water_rate",
            )
        }

    def _metal_surface_emission(self, temperature, emissivity, shape):
        """Form sigma-epsilon-T^4 from a resident surface-temperature plane."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        count = int(np.prod(shape))
        if (not isinstance(temperature, MetalArray)
                or temperature.buffer.context is not self.metal_context
                or temperature.dtype != np.dtype(np.float64)
                or tuple(temperature.shape) != tuple(shape)):
            raise ValueError("Global skin temperature must be a resident FP64 plane")
        metal = MetalArrayModule(self.metal_context)
        if isinstance(emissivity, MetalArray):
            if (emissivity.buffer.context is not self.metal_context
                    or emissivity.dtype != np.dtype(np.float64)
                    or tuple(emissivity.shape) != tuple(shape)):
                raise ValueError("Global surface emissivity has incompatible Metal residency")
            emissivity_device = emissivity.view((count,), dtype=np.float64)
        else:
            emissivity_device = metal.asarray(np.ascontiguousarray(
                np.broadcast_to(np.asarray(emissivity, dtype=np.float64), shape)
                .reshape(-1)))
        emission = metal.empty((count,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_surface_emission_fragment",
            {"surface_temperature": temperature.view((count,), dtype=np.float64),
             "emissivity": emissivity_device,
             "stefan_boltzmann": metal.asarray(np.asarray(
                 [STEFAN_BOLTZMANN], dtype=np.float64)),
             "surface_emission": emission},
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SURFACE_EMISSION_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return emission.view(shape, dtype=np.float64)

    def _metal_soil_temperature_relax(self, surface_temperature, soil_temperature,
                                      relax):
        """Relax soil layers toward resident skin temperature without a readback."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(surface_temperature.shape)
        count = int(np.prod(shape))
        nlayer = int(soil_temperature.shape[0])
        metal = MetalArrayModule(self.metal_context)
        if (not isinstance(surface_temperature, MetalArray)
                or surface_temperature.buffer.context is not self.metal_context
                or surface_temperature.dtype != np.dtype(np.float64)
                or len(shape) != 2):
            raise TypeError("Global skin temperature must be a resident FP64 plane")
        if isinstance(soil_temperature, MetalArray):
            if (soil_temperature.buffer.context is not self.metal_context
                    or soil_temperature.dtype != np.dtype(np.float64)
                    or tuple(soil_temperature.shape) != (nlayer, *shape)):
                raise ValueError("Global soil temperature has incompatible Metal residency")
            soil_device = soil_temperature.view((nlayer * count,), dtype=np.float64)
        else:
            soil_device = metal.asarray(np.ascontiguousarray(
                np.asarray(soil_temperature, dtype=np.float64).reshape(-1)))
        updated = metal.empty((nlayer * count,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_soil_temperature_relax_fragment",
            {"surface_temperature": surface_temperature.view((count,), dtype=np.float64),
             "soil_temperature": soil_device,
             "relax": metal.asarray(np.asarray([relax], dtype=np.float64)),
             "updated_soil_temperature": updated},
            {"count": count, "nlayer": nlayer}, exact_policy=False,
            grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SOIL_TEMPERATURE_RELAX_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return updated.view((nlayer, *shape), dtype=np.float64)

    def _metal_surface_fallout_cooling(self, surface_temperature, snow, graupel,
                                       heat_capacity):
        """Apply the fusion-energy debit while keeping skin temperature resident."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(surface_temperature.shape)
        count = int(np.prod(shape))
        metal = MetalArrayModule(self.metal_context)

        def input_plane(value):
            if isinstance(value, MetalArray):
                if (value.buffer.context is not self.metal_context
                        or value.dtype != np.dtype(np.float64)
                        or tuple(value.shape) != shape):
                    raise ValueError("Global surface-cooling input has incompatible residency")
                return value.view((count,), dtype=np.float64)
            return metal.asarray(np.ascontiguousarray(np.broadcast_to(
                np.asarray(value, dtype=np.float64), shape).reshape(-1)))

        updated = metal.empty((count,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_surface_fallout_cooling_fragment",
            {"surface_temperature": input_plane(surface_temperature),
             "snow": input_plane(snow), "graupel": input_plane(graupel),
             "heat_capacity": input_plane(heat_capacity),
             "latent_heat_fusion": metal.asarray(np.asarray(
                 [LATENT_HEAT_FUSION], dtype=np.float64)),
             "updated_surface_temperature": updated},
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SURFACE_FALLOUT_COOLING_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return updated.view(shape, dtype=np.float64)

    def _metal_surface_energy_update(self, surface_temperature, net_radiation,
                                     sensible, evaporation, heat_capacity, dt):
        """Apply the residual surface energy tendency to resident skin state."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        shape = tuple(surface_temperature.shape)
        count = int(np.prod(shape))
        metal = MetalArrayModule(self.metal_context)

        def input_plane(value):
            if isinstance(value, MetalArray):
                if (value.buffer.context is not self.metal_context
                        or value.dtype != np.dtype(np.float64)
                        or tuple(value.shape) != shape):
                    raise ValueError("Global surface-energy input has incompatible residency")
                return value.view((count,), dtype=np.float64)
            return metal.asarray(np.ascontiguousarray(np.broadcast_to(
                np.asarray(value, dtype=np.float64), shape).reshape(-1)))

        updated = metal.empty((count,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_surface_energy_update_fragment",
            {"surface_temperature": input_plane(surface_temperature),
             "net_surface_radiation": input_plane(net_radiation),
             "sensible": input_plane(sensible),
             "evaporation": input_plane(evaporation),
             "heat_capacity": input_plane(heat_capacity),
             "dt": metal.asarray(np.asarray([dt], dtype=np.float64)),
             "latent_heat_vaporization": metal.asarray(np.asarray(
                 [LATENT_HEAT_VAPORIZATION], dtype=np.float64)),
             "updated_surface_temperature": updated},
            {"count": count}, exact_policy=False, grid_size=count,
            threads_per_group=(
                kernel_specs.GLOBAL_SURFACE_ENERGY_UPDATE_FRAGMENT_THREADS_PER_GROUP
            ),
        )
        return updated.view(shape, dtype=np.float64)

    def _metal_row_mean(self, values):
        """Reduce a 2-D FP64 plane to one resident Metal scalar."""
        from woof.metal.array import MetalArray, MetalArrayModule
        from woof.metal.generated import kernel_specs

        if not isinstance(values, MetalArray):
            raise TypeError("Global Metal row-mean input must be a MetalArray")
        if values.buffer.context is not self.metal_context:
            raise ValueError("Global Metal row-mean input belongs to another context")
        if values.dtype != np.dtype(np.float64) or values.ndim != 2:
            raise TypeError("Global Metal row-mean input must be a 2-D FP64 plane")
        nlat, nlon = map(int, values.shape)
        partial_sums = MetalArrayModule(self.metal_context).empty(
            (nlat,), dtype=np.float64
        )
        self.metal_context.submit_kernel(
            "global_reductions_fragment",
            {"values": values.view((values.size,), dtype=np.float64),
             "partial_sums": partial_sums},
            {"count": nlat, "chunk_size": nlon}, exact_policy=False,
            grid_size=nlat,
            threads_per_group=kernel_specs.GLOBAL_REDUCTIONS_FRAGMENT_THREADS_PER_GROUP,
        )
        total = MetalArrayModule(self.metal_context).empty((1,), dtype=np.float64)
        self.metal_context.submit_kernel(
            "global_sum_finalize_fragment",
            {"partial_sums": partial_sums, "total": total},
            {"count": nlat}, exact_policy=False, grid_size=1,
            threads_per_group=kernel_specs.GLOBAL_SUM_FINALIZE_FRAGMENT_THREADS_PER_GROUP,
        )
        # Keep the reduced sum resident.  The caller applies this scale
        # when the per-call diagnostic scalars are materialized together.
        return total, 1.0 / values.size

    def _saturation_adjust(self, theta, q, exner, pressure, xp,
                           *, keep_resident: bool = False,
                           keep_condensed_resident: bool = False):
        if self.metal_context is not None and xp is np:
            from woof.metal.array import MetalArray, MetalArrayModule
            from woof.metal.generated import kernel_specs

            shape = tuple(theta.shape)
            nlevel = shape[0]
            ncolumn = int(np.prod(shape[1:]))
            metal = MetalArrayModule(self.metal_context)
            resident_input = isinstance(theta, MetalArray)
            if resident_input:
                for name in ("qv", "qc"):
                    value = q[name]
                    if (not isinstance(value, MetalArray)
                            or value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != shape):
                        raise ValueError(
                            "Global saturation-adjust resident theta, qv, and qc "
                            "must be matching FP64 MetalArrays"
                        )
                # A resident prognostic input must not silently take the
                # host-staging return path, even if an older caller omits the
                # retention flags. Condensed water is an output diagnostic;
                # the only host readback is its eventual scalar maximum.
                keep_resident = True
                keep_condensed_resident = True

            def upload(value):
                if isinstance(value, MetalArray):
                    if value.buffer.context is not self.metal_context:
                        raise ValueError(
                            "Global saturation-adjust input belongs to another Metal context"
                        )
                    if value.dtype != np.dtype(np.float64):
                        raise TypeError("Global saturation-adjust inputs must be FP64")
                    resident = metal.asarray(value)
                    return (resident.view((resident.size,), dtype=np.float64)
                            if resident.ndim != 1 else resident)
                return metal.asarray(np.ascontiguousarray(
                    np.asarray(value, dtype=np.float64).reshape(-1)))

            resources = {
                "theta": upload(theta), "exner": upload(exner),
                "pressure": upload(pressure), "qv": upload(q["qv"]),
                "qc": upload(q["qc"]),
                "epsilon": upload(np.asarray([EPSILON], dtype=np.float64)),
                "dry_air_cp": upload(np.asarray([DRY_AIR_CP], dtype=np.float64)),
                "latent_heat": upload(np.asarray([LATENT_HEAT_VAPORIZATION], dtype=np.float64)),
                "theta_out": metal.empty((nlevel * ncolumn,), dtype=np.float64),
                "qv_out": metal.empty((nlevel * ncolumn,), dtype=np.float64),
                "qc_out": metal.empty((nlevel * ncolumn,), dtype=np.float64),
                "condensed": metal.empty((ncolumn,), dtype=np.float64),
            }
            self.metal_context.submit_kernel(
                "global_saturation_adjust_fragment", resources,
                {"nlevel": nlevel, "ncolumn": ncolumn}, exact_policy=False,
                grid_size=ncolumn,
                threads_per_group=kernel_specs.GLOBAL_SATURATION_ADJUST_FRAGMENT_THREADS_PER_GROUP,
            )
            if keep_resident:
                q["qv"] = resources["qv_out"].view(shape, dtype=np.float64)
                q["qc"] = resources["qc_out"].view(shape, dtype=np.float64)
                theta_out = resources["theta_out"].view(shape, dtype=np.float64)
            else:
                q["qv"] = resources["qv_out"].to_numpy().reshape(shape)
                q["qc"] = resources["qc_out"].to_numpy().reshape(shape)
                theta_out = resources["theta_out"].to_numpy().reshape(shape)
            condensed = (
                resources["condensed"].view(shape[1:], dtype=np.float64)
                if keep_condensed_resident else
                resources["condensed"].to_numpy().reshape(shape[1:])
            )
            return theta_out, condensed
        temperature = theta * exner
        if xp is not np:
            return self._saturation_adjust_cupy(temperature, q, exner, pressure, xp)
        total_phase = xp.zeros_like(temperature)
        # Newton iteration on f(T) = qv(T) - qsat(T) along the isobaric moist
        # adiabat qv(T) = qv0 - (cp/L)(T - T0): each step exchanges
        # (qv - qsat) / (1 + (L/cp) dqsat/dT) between vapor and cloud.
        # Condensing the bare excess instead diverges wherever
        # (L/cp) dqsat/dT > 1 (roughly T > 275 K at 1000 hPa). Four steps
        # leave |qv - qsat| below 3e-17 kg/kg at the warm extreme (measured
        # 2026-08-30: T=300 K, p=1e5 Pa, 2e-3 kg/kg initial supersaturation).
        for _ in range(4):
            e, t = _saturation_vapor_pressure(temperature, pressure, xp)
            dry_partial = xp.maximum(pressure - e, 1.0)
            qsat = EPSILON * e / dry_partial
            # Tetens derivative: dln(e)/dT = 17.67 * 243.5 / (T - 29.65)^2.
            dqsat_dt = (
                qsat
                * (pressure / dry_partial)
                * 17.67 * 243.5 / xp.maximum(t - 29.65, 1.0) ** 2
            )
            gamma = (LATENT_HEAT_VAPORIZATION / DRY_AIR_CP) * dqsat_dt
            phase = (q["qv"] - qsat) / (1.0 + gamma)
            # Evaporation stops when the cloud reservoir empties.
            phase = xp.maximum(phase, -q["qc"])
            q["qv"] -= phase
            q["qc"] += phase
            temperature += (LATENT_HEAT_VAPORIZATION / DRY_AIR_CP) * phase
            total_phase += phase
        condensed = xp.sum(xp.maximum(total_phase, 0.0), axis=0)
        return temperature / exner, condensed

    def _saturation_adjust_cupy(self, temperature, q, exner, pressure, xp):
        """Fused Newton loop: all four iterations inside one kernel.

        The iterations are data-independent across points - each reads
        only its own T/qv/qc/p - so the loop runs internally.  Inputs and
        outputs alias so qv/qc mutate in place exactly like the numpy
        path's -=/+=.
        """
        kernel = _cupy_kernel(
            "reference_saturation_adjust", _build_saturation_adjust_kernel
        )
        total_phase = xp.empty_like(temperature)
        kernel(
            temperature, q["qv"], q["qc"], pressure,
            temperature, q["qv"], q["qc"], total_phase,
        )
        condensed = xp.sum(xp.maximum(total_phase, 0.0), axis=0)
        return temperature / exner, condensed

    def _surface_radiation_fluxes(self, exchange, u, v, theta, q, surface, xp):
        dt = float(exchange.dt_s)
        bottom = -1
        p = exchange.p_full
        dp = exchange.dp
        exner = exchange.exner
        from woof.metal.array import MetalArray, MetalArrayModule

        resident_theta = isinstance(theta, MetalArray)
        if resident_theta:
            if (theta.buffer.context is not self.metal_context
                    or theta.dtype != np.dtype(np.float64)):
                raise ValueError(
                    "Global radiation theta must be FP64 on this Metal context"
                )
            # Radiation optics derives temperature inside its Metal kernel.
            # Surface-layer thermodynamics are prepared from bottom-level
            # Metal views immediately before the surface-flux kernel.
            temperature = virtual = rho = None
        else:
            temperature = theta * exner
            virtual = temperature * (1.0 + 0.61 * q["qv"])
            rho = p / (DRY_AIR_GAS_CONSTANT * xp.maximum(virtual, 150.0))

        metal_solar = self.metal_context is not None and xp is np
        if metal_solar:
            from woof.metal.array import MetalArray, MetalArrayModule
            from woof.metal.generated import kernel_specs

            shape = tuple(surface.temperature_k.shape)
            ncolumn = int(np.prod(shape))
            metal = MetalArrayModule(self.metal_context)
            geometry = getattr(self, "_metal_solar_geometry", None)
            latitude_deg = np.ascontiguousarray(np.broadcast_to(
                np.asarray(exchange.latitude_deg, dtype=np.float64), shape
            ))
            longitude_deg = np.ascontiguousarray(np.broadcast_to(
                np.asarray(exchange.longitude_deg, dtype=np.float64), shape
            ))
            geometry_key = (shape, hashlib.sha256(
                latitude_deg.tobytes() + longitude_deg.tobytes()
            ).digest())
            if geometry is None or geometry[0] != geometry_key:
                latitude = latitude_deg * (math.pi / 180.0)
                longitude = longitude_deg * (math.pi / 180.0)
                geometry = (
                    geometry_key,
                    metal.asarray(np.ascontiguousarray(np.cos(latitude).reshape(-1))),
                    metal.asarray(np.ascontiguousarray(np.sin(longitude).reshape(-1))),
                    metal.asarray(np.ascontiguousarray(np.cos(longitude).reshape(-1))),
                )
                self._metal_solar_geometry = geometry
            solar_angle = 2.0 * math.pi * (
                (exchange.time_s + 0.5 * dt) % 86_400.0
            ) / 86_400.0
            solar_angle = (solar_angle + math.pi) % (2.0 * math.pi) - math.pi
            solar = {
                "cos_lat": geometry[1], "sin_lon": geometry[2],
                "cos_lon": geometry[3],
                "albedo": metal.asarray(np.ascontiguousarray(
                    np.broadcast_to(np.asarray(surface.albedo, dtype=np.float64), shape)
                    .reshape(-1))),
                "solar_angle": metal.asarray(np.asarray([solar_angle], dtype=np.float64)),
                "solar_constant": metal.asarray(np.asarray(
                    [self.options.solar_constant_w_m2], dtype=np.float64)),
                "atmospheric_absorptivity": metal.asarray(np.asarray(
                    [self.options.atmospheric_shortwave_absorptivity], dtype=np.float64)),
                "cosine_zenith": metal.empty((ncolumn,), dtype=np.float64),
                "sw_atmosphere": metal.empty((ncolumn,), dtype=np.float64),
                "absorbed_shortwave": metal.empty((ncolumn,), dtype=np.float64),
            }
            self.metal_context.submit_kernel(
                "global_gray_solar_fragment", solar, {"count": ncolumn},
                exact_policy=False, grid_size=ncolumn,
                threads_per_group=kernel_specs.GLOBAL_GRAY_SOLAR_FRAGMENT_THREADS_PER_GROUP,
            )
            coszen = solar["cosine_zenith"].view(shape, dtype=np.float64)
            sw_atmosphere = solar["sw_atmosphere"].view(shape, dtype=np.float64)
            absorbed_surface = solar["absorbed_shortwave"].view(shape, dtype=np.float64)
            # The receipt needs one scalar, not a host copy of every column.
            maximum_cosine_zenith = self._metal_maximum(
                solar["cosine_zenith"]
            )
        else:
            latitude = exchange.latitude_deg * (math.pi / 180.0)
            longitude = exchange.longitude_deg * (math.pi / 180.0)
            solar_angle = 2.0 * math.pi * (
                (exchange.time_s + 0.5 * dt) % 86_400.0
            ) / 86_400.0
            hour_angle = solar_angle + longitude - math.pi
            coszen = xp.maximum(0.0, xp.cos(latitude) * xp.cos(hour_angle))
            sw_toa = self.options.solar_constant_w_m2 * coszen
            sw_atmosphere = self.options.atmospheric_shortwave_absorptivity * sw_toa
            sw_surface = (1.0 - self.options.atmospheric_shortwave_absorptivity) * sw_toa
            absorbed_surface = (1.0 - surface.albedo) * sw_surface
            maximum_cosine_zenith = float(self.backend.to_numpy(coszen).max())

        if isinstance(surface.temperature_k, MetalArray):
            lw_up = (
                self._metal_surface_emission(
                    surface.temperature_k, surface.emissivity,
                    tuple(surface.temperature_k.shape),
                )
                if self.options.radiation else 0.0
            )
            from woof.metal.array import MetalArrayModule
            zero_plane = MetalArrayModule(self.metal_context).zeros(
                tuple(surface.temperature_k.shape), dtype=np.float64
            )
            net_surface_radiation = zero_plane
            outgoing_longwave = zero_plane.copy()
        else:
            lw_up = surface.emissivity * STEFAN_BOLTZMANN * surface.temperature_k ** 4
            net_surface_radiation = xp.zeros_like(surface.temperature_k)
            outgoing_longwave = xp.zeros_like(surface.temperature_k)
        radiation_theta_updated = False
        if self.options.radiation:
            # Gray two-stream longwave without scattering. The configured
            # column emissivity is 1 - exp(-tau_column) for a gray absorber
            # mixed uniformly with mass, so layer k carries optical depth
            # tau_column * dp_k / sum(dp); condensate adds gray optical depth
            # 80 * q_cond (tau 0.8 at the 0.01 kg/kg cap). Each layer emits
            # eps_k * sigma * T_k^4 into both streams and transmits
            # (1 - eps_k), so a layer whose emission exceeds what it absorbs
            # from its neighbors cools, and the column as a whole loses
            # OLR = F_up(top) to space instead of relaxing toward the
            # surface temperature.
            tau_column = -math.log(
                max(1.0 - self.options.atmospheric_longwave_emissivity, 1.0e-12)
            )
            nz = theta.shape[0]
            if self.metal_context is not None and xp is np:
                from woof.metal.array import MetalArray, MetalArrayModule
                from woof.metal.generated import kernel_specs

                shape = tuple(theta.shape[1:])
                field_shape = (nz, *shape)
                ncolumn = int(np.prod(shape))
                metal = MetalArrayModule(self.metal_context)

                def upload(value, expected_shape):
                    if isinstance(value, MetalArray):
                        if value.buffer.context is not self.metal_context:
                            raise ValueError(
                                "Global gray-radiation input belongs to another Metal context"
                            )
                        if value.dtype != np.dtype(np.float64):
                            raise TypeError(
                                "Global gray-radiation inputs must be FP64"
                            )
                        if tuple(value.shape) != tuple(expected_shape):
                            raise ValueError(
                                "Global gray-radiation input has shape "
                                f"{value.shape}, expected {tuple(expected_shape)}"
                            )
                        return (value.view((value.size,), dtype=np.float64)
                                if value.ndim != 1 else value)
                    return metal.asarray(np.ascontiguousarray(
                        np.asarray(value, dtype=np.float64).reshape(-1)))

                optics = {
                    "theta": upload(theta, field_shape),
                    "exner": upload(exner, field_shape),
                    "dp": upload(dp, field_shape),
                    "qc": upload(q["qc"], field_shape),
                    "qi": upload(q["qi"], field_shape),
                    "qs": upload(q["qs"], field_shape),
                    "sw_atmosphere": upload(sw_atmosphere, shape),
                    "tau_column": upload(np.asarray([tau_column], dtype=np.float64), (1,)),
                    "cloud_optical_depth": upload(np.asarray([80.0], dtype=np.float64), (1,)),
                    "stefan_boltzmann": upload(
                        np.asarray([STEFAN_BOLTZMANN], dtype=np.float64), (1,)),
                    "epsilon": metal.empty((nz * ncolumn,), dtype=np.float64),
                    "emission": metal.empty((nz * ncolumn,), dtype=np.float64),
                    "shortwave_heating": metal.empty((nz * ncolumn,), dtype=np.float64),
                }
                self.metal_context.submit_kernel(
                    "global_gray_radiation_optics_fragment", optics,
                    {"count": nz * ncolumn, "nlevel": nz,
                     "ncolumn": ncolumn}, exact_policy=False,
                    grid_size=nz * ncolumn,
                    threads_per_group=(
                        kernel_specs.GLOBAL_GRAY_RADIATION_OPTICS_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                resources = {
                    "epsilon": optics["epsilon"],
                    "emission": optics["emission"],
                    "surface_emission": upload(lw_up, shape),
                    "surface_emissivity": upload(surface.emissivity, shape),
                    "absorbed_shortwave": upload(absorbed_surface, shape),
                    "longwave_heating": metal.empty((nz * ncolumn,), dtype=np.float64),
                    "outgoing_longwave": metal.empty((ncolumn,), dtype=np.float64),
                    "net_surface_radiation": metal.empty((ncolumn,), dtype=np.float64),
                }
                self.metal_context.submit_kernel(
                    "global_gray_radiation_fragment", resources,
                    {"nlevel": nz, "ncolumn": ncolumn}, exact_policy=False,
                    grid_size=nz * ncolumn,
                    threads_per_group=kernel_specs.GLOBAL_GRAY_RADIATION_FRAGMENT_THREADS_PER_GROUP,
                )
                count = nz * ncolumn
                update = {
                    "theta": upload(theta, field_shape),
                    "exner": upload(exner, field_shape),
                    "dp": upload(dp, field_shape),
                    "shortwave_heating": optics["shortwave_heating"],
                    "longwave_heating": resources["longwave_heating"],
                    "dt": upload(np.asarray([dt], dtype=np.float64), (1,)),
                    "gravity": upload(np.asarray([GRAVITY_M_S2], dtype=np.float64), (1,)),
                    "dry_air_cp": upload(np.asarray([DRY_AIR_CP], dtype=np.float64), (1,)),
                    "theta_out": metal.empty((count,), dtype=np.float64),
                }
                self.metal_context.submit_kernel(
                    "global_gray_radiation_update_fragment", update,
                    {"count": count}, exact_policy=False, grid_size=count,
                    threads_per_group=(
                        kernel_specs.GLOBAL_GRAY_RADIATION_UPDATE_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                # The cold-top floor is the next operation and already has
                # a Metal kernel. Keep this FP64 result resident until that
                # kernel runs instead of downloading and uploading theta.
                theta = update["theta_out"].view(field_shape, dtype=np.float64)
                radiation_theta_updated = True
                outgoing_longwave = resources[
                    "outgoing_longwave"
                ].view(shape, dtype=np.float64)
                net_surface_radiation = resources[
                    "net_surface_radiation"
                ].view(shape, dtype=np.float64)
            else:
                weights = dp / xp.maximum(xp.sum(dp, axis=0, keepdims=True), 1.0)
                shortwave_heating = sw_atmosphere[None] * weights
                cloud = xp.clip(q["qc"] + q["qi"] + q["qs"], 0.0, 0.01)
                eps_layer = 1.0 - xp.exp(-(tau_column * weights + 80.0 * cloud))
                emission = STEFAN_BOLTZMANN * temperature ** 4
                flux_down = xp.zeros((nz + 1, *emission.shape[1:]), dtype=emission.dtype)
                flux_up = xp.zeros_like(flux_down)
                for k in range(nz):
                    flux_down[k + 1] = (
                        flux_down[k] * (1.0 - eps_layer[k]) + eps_layer[k] * emission[k]
                    )
                # The surface absorbs eps_s * F_down, emits eps_s * sigma * Ts^4,
                # and reflects the remainder into the upward stream, closing the
                # surface-atmosphere-space budget exactly.
                flux_up[nz] = lw_up + (1.0 - surface.emissivity) * flux_down[nz]
                for k in range(nz - 1, -1, -1):
                    flux_up[k] = (
                        flux_up[k + 1] * (1.0 - eps_layer[k]) + eps_layer[k] * emission[k]
                    )
                outgoing_longwave = flux_up[0]
                net_flux = flux_up - flux_down
                longwave_heating = net_flux[1:] - net_flux[:-1]
                net_surface_radiation = (
                    absorbed_surface + surface.emissivity * flux_down[nz] - lw_up
                )
            if not radiation_theta_updated:
                dtemp = (shortwave_heating + longwave_heating) * GRAVITY_M_S2 / (
                    DRY_AIR_CP * xp.maximum(dp, 1.0)
                )
                theta += dt * dtemp / exner
            # Cold-top relaxation: a gray absorber has no ozone shortwave
            # heating and a hydrostatic run this short cannot supply the
            # residual-circulation warming that holds the real polar-night
            # upper stratosphere near 230 K, so the top levels relax
            # radiatively toward the skin temperature instead - the T63
            # 48 h run drifted to the 140 K research floor and the first
            # T533 run crossed it at hour 3 (139.92 K) and died.  This is
            # the standard simple-GCM parameterization of that missing
            # physics (Held-Suarez-style relaxation), one-sided so it only
            # warms columns colder than the reference and only above the
            # configured pressure.
            theta = self._stratospheric_floor_pull(
                theta, exner, exchange.p_full, dt, xp
            )
            # The surface-flux kernel accepts a resident theta plane. Keep
            # the radiation/floor result on Metal through that adjacent handoff.
            from woof.metal.array import MetalArray

        if isinstance(surface.temperature_k, MetalArray):
            sensible = evaporation = stress = zero_plane
        else:
            sensible = xp.zeros_like(surface.temperature_k)
            evaporation = xp.zeros_like(surface.temperature_k)
            stress = xp.zeros_like(surface.temperature_k)
        surface_temperature_updated = False
        if self.options.surface_fluxes:
            if resident_theta:
                prepared = self._metal_surface_flux_preparation(
                    exchange, surface, u, v, theta, q["qv"]
                )
                wind = prepared["wind"]
                cd = prepared["cd"]
                qsat_surface = prepared["saturation_mixing_ratio"]
                beta = prepared["beta"]
                rho_bottom = prepared["rho_bottom"]
                air_temperature = prepared["air_temperature"]
            else:
                wind = xp.maximum(xp.sqrt(u[bottom] ** 2 + v[bottom] ** 2), 0.5)
                cd = self.options.bulk_exchange_coefficient * (
                    1.0 + 0.08 * xp.log1p(
                        xp.maximum(surface.roughness_m, 1.0e-5) / 1.0e-4
                    )
                )
                qsat_surface = saturation_mixing_ratio(
                    surface.temperature_k, exchange.p_half[-1], xp
                )
                rho_bottom = rho[bottom]
                air_temperature = temperature[bottom]
            # Soil-wetness resistance (bucket beta): land evaporates at a
            # fraction of the potential rate set by the analyzed top-layer
            # soil water; ocean stays at potential.  Without it every
            # continent evaporates like a warm ocean, and 21 hours of
            # August heating over the Rockies with unlimited moisture fed
            # a CISK runaway to a spurious 971 hPa continental cyclone on
            # the first convection-enabled real-data run (2026-08-31).
            if not resident_theta:
                wetness = xp.clip(
                    surface.soil_water_fraction[0]
                    / self.options.soil_wetness_capacity,
                    0.0,
                    1.0,
                )
                beta = (
                    1.0
                    - surface.land_fraction
                    + surface.land_fraction * wetness
                )
            if self.metal_context is not None and xp is np:
                shape = tuple(surface.temperature_k.shape)
                count = int(np.prod(shape))
                values = {
                    "rho_bottom": rho_bottom, "cd": cd, "wind": wind,
                    "surface_temperature": surface.temperature_k,
                    "air_temperature": air_temperature,
                    "theta": theta[bottom], "saturation_mixing_ratio": qsat_surface,
                    "qv": q["qv"][bottom], "beta": beta,
                    "available_water_rate": (
                        prepared["available_water_rate"]
                        if resident_theta else surface.water_kg_m2 / dt
                    ),
                    "surface_water": surface.water_kg_m2, "dp_bottom": dp[bottom],
                    "exner_bottom": exner[bottom], "u": u[bottom], "v": v[bottom],
                    "net_surface_radiation": net_surface_radiation,
                    "surface_heat_capacity": surface.heat_capacity_j_m2_k,
                }
                from woof.metal.array import MetalArray, MetalArrayModule
                from woof.metal.generated import kernel_specs

                metal = MetalArrayModule(self.metal_context)
                resources = {}
                for name, value in values.items():
                    if isinstance(value, MetalArray):
                        if value.buffer.context is not self.metal_context:
                            raise ValueError(
                                "Global surface-flux input belongs to another Metal context"
                            )
                        if value.dtype != np.dtype(np.float64):
                            raise TypeError("Global surface-flux inputs must be FP64")
                        if value.shape != shape:
                            raise ValueError(
                                "Global surface-flux Metal input has shape "
                                f"{value.shape}, expected {shape}"
                            )
                        resources[name] = value.view((count,), dtype=np.float64)
                    else:
                        host = np.ascontiguousarray(np.broadcast_to(
                            np.asarray(value, dtype=np.float64), shape
                        ).reshape(-1))
                        resources[name] = metal.asarray(host)
                for name in ("sensible", "evaporation", "updated_qv", "updated_theta",
                             "updated_u", "updated_v", "updated_surface_water",
                             "updated_surface_temperature"):
                    resources[name] = metal.empty((count,), dtype=np.float64)
                for name, value in (("dt", dt), ("dry_air_cp", DRY_AIR_CP),
                                    ("gravity", GRAVITY_M_S2),
                                    ("latent_heat", LATENT_HEAT_VAPORIZATION)):
                    resources[name] = metal.asarray(np.asarray([value], dtype=np.float64))
                self.metal_context.submit_kernel(
                    "global_surface_flux_fragment", resources, {"count": count},
                    exact_policy=False, grid_size=count,
                    threads_per_group=kernel_specs.GLOBAL_SURFACE_FLUX_FRAGMENT_THREADS_PER_GROUP,
                )
                def result(name):
                    return resources[name].to_numpy().reshape(shape)
                # The kernel has already applied sensible heating to the
                # returned skin temperature; this diagnostic output would
                # otherwise be downloaded only for the skipped NumPy update.
                evaporation = resources["evaporation"].view(
                    shape, dtype=np.float64
                )
                for field, output in (
                    (q["qv"], "updated_qv"),
                    (u, "updated_u"),
                    (v, "updated_v"),
                ):
                    if isinstance(field, MetalArray):
                        field[bottom][...] = resources[output].view(
                            shape, dtype=np.float64
                        )
                    else:
                        field[bottom] = result(output)
                if isinstance(theta, MetalArray):
                    plane_size = int(np.prod(shape))
                    theta_bottom = theta.view(
                        shape, dtype=np.float64,
                        byte_offset=theta.byte_offset
                        + (theta.shape[0] - 1) * plane_size * np.dtype(np.float64).itemsize,
                    )
                    theta_bottom[...] = resources["updated_theta"].view(
                        shape, dtype=np.float64
                    )
                else:
                    theta[bottom] = result("updated_theta")
                updated_surface_water = resources["updated_surface_water"].view(
                    shape, dtype=np.float64
                )
                surface.water_kg_m2 = (
                    updated_surface_water
                    if isinstance(surface.water_kg_m2, MetalArray)
                    else result("updated_surface_water")
                )
                surface.temperature_k = resources[
                    "updated_surface_temperature"
                ].view(shape, dtype=np.float64)
                surface_temperature_updated = True
            else:
                sensible = (
                    rho[bottom] * DRY_AIR_CP * cd * wind
                    * (surface.temperature_k - temperature[bottom])
                )
                evaporation = beta * rho[bottom] * cd * wind * (
                    qsat_surface - q["qv"][bottom]
                )
                available = surface.water_kg_m2 / dt
                evaporation = xp.minimum(xp.maximum(evaporation, 0.0), available)
                dq = evaporation * GRAVITY_M_S2 / xp.maximum(dp[bottom], 1.0)
                dtemp = sensible * GRAVITY_M_S2 / (
                    DRY_AIR_CP * xp.maximum(dp[bottom], 1.0)
                )
                q["qv"][bottom] += dt * dq
                theta[bottom] += dt * dtemp / exner[bottom]
                stress = rho[bottom] * cd * wind
                momentum_factor = dt * stress * GRAVITY_M_S2 / xp.maximum(dp[bottom], 1.0)
                u[bottom] -= momentum_factor * u[bottom]
                v[bottom] -= momentum_factor * v[bottom]
                surface.water_kg_m2 -= dt * evaporation

        from woof.metal.array import MetalArray
        mean_outgoing_longwave = (
            self._metal_row_mean(outgoing_longwave)
            if isinstance(outgoing_longwave, MetalArray)
            else float(np.mean(outgoing_longwave))
        )
        diagnostics = {
            "maximum_cosine_zenith": maximum_cosine_zenith,
            "mean_outgoing_longwave_w_m2": mean_outgoing_longwave,
        }
        planes = {
            "net_surface_radiation": net_surface_radiation,
        }
        if isinstance(evaporation, MetalArray):
            diagnostics["mean_surface_evaporation_kg_m2_s"] = (
                self._metal_row_mean(evaporation)
            )
        else:
            planes["evaporation"] = evaporation
        if not surface_temperature_updated:
            if isinstance(surface.temperature_k, MetalArray):
                surface.temperature_k = self._metal_surface_energy_update(
                    surface.temperature_k, net_surface_radiation,
                    sensible, evaporation, surface.heat_capacity_j_m2_k, dt,
                )
            else:
                net_surface = net_surface_radiation - sensible - LATENT_HEAT_VAPORIZATION * evaporation
                surface.temperature_k += dt * net_surface / xp.maximum(
                    surface.heat_capacity_j_m2_k, 1.0e4
                )
        soil_relax = 1.0 - math.exp(-dt / self.options.soil_exchange_time_s)
        if isinstance(surface.temperature_k, MetalArray):
            surface.soil_temperature_k = self._metal_soil_temperature_relax(
                surface.temperature_k, surface.soil_temperature_k, soil_relax
            )
        else:
            surface.soil_temperature_k[0] += soil_relax * (
                surface.temperature_k - surface.soil_temperature_k[0]
            )
            for k in range(1, surface.soil_temperature_k.shape[0]):
                surface.soil_temperature_k[k] += 0.25 * soil_relax * (
                    surface.soil_temperature_k[k - 1] - surface.soil_temperature_k[k]
                )
        # The three grid means are planes the caller assembles over the
        # globe and finish() reduces once (PhysicsResult.planes); the
        # maximum folds across bands exactly.
        return u, v, theta, q, surface, diagnostics, planes

    def _turbulence(self, exchange, u, v, theta, q, xp):
        if not self.options.turbulence:
            return u, v, theta, q
        from woof.metal.array import MetalArray
        resident_state = isinstance(theta, MetalArray)
        if resident_state:
            z = height_agl = diffusivity = None
        else:
            z = exchange.geopotential / GRAVITY_M_S2
            surface_z = z[-1]
            height_agl = xp.maximum(z - surface_z[None], 0.0)
            diffusivity = self.options.background_diffusivity_m2_s + (
                self.options.pbl_diffusivity_m2_s
                * xp.exp(-height_agl / self.options.pbl_depth_m)
            )
        dt = float(exchange.dt_s)
        dp = exchange.dp
        if (self.metal_context is not None and xp is np
                and np.dtype(theta.dtype) == np.dtype(np.float64)
                and int(theta.shape[0]) <= 128):
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            shape = tuple(theta.shape)
            nlevel = shape[0]
            ncolumn = int(np.prod(shape[1:]))
            metal = MetalArrayModule(self.metal_context)
            resident = isinstance(theta, MetalArray)
            field_values = (u, v, theta, q["qv"], q["qc"], q["qi"])
            if resident:
                for name, value in zip(
                    ("u", "v", "theta", "qv", "qc", "qi"), field_values
                ):
                    if (not isinstance(value, MetalArray)
                            or value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != shape):
                        raise ValueError(
                            f"Global turbulence {name} must match resident theta"
                        )
                fields = metal.empty((6 * nlevel * ncolumn,), dtype=np.float64)
                packed = {
                    name: value.view((nlevel * ncolumn,), dtype=np.float64)
                    for name, value in zip(
                        ("u", "v", "theta", "qv", "qc", "qi"), field_values
                    )
                }
                packed["fields"] = fields
                self.metal_context.submit_kernel(
                    "global_implicit_diffusion_pack_fragment", packed,
                    {"nlevel": nlevel, "ncolumn": ncolumn},
                    exact_policy=False, grid_size=nlevel * ncolumn,
                    threads_per_group=(
                        kernel_specs.GLOBAL_IMPLICIT_DIFFUSION_PACK_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
            else:
                if any(isinstance(value, MetalArray) for value in field_values):
                    raise ValueError(
                        "Global turbulence prognostics must share residency"
                    )
                host_fields = np.ascontiguousarray(
                    np.stack(field_values, axis=0)
                )
                fields = None

            def upload(value, target_shape):
                if isinstance(value, MetalArray):
                    if (value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != tuple(target_shape)):
                        raise ValueError(
                            "Global turbulence resident inputs must be same-context FP64"
                        )
                    return value.view((value.size,), dtype=np.float64)
                host = np.broadcast_to(np.asarray(value, dtype=np.float64), target_shape)
                return metal.asarray(np.ascontiguousarray(host.reshape(-1)))

            if resident:
                height_device = metal.empty(shape, dtype=np.float64)
                diffusivity_device = metal.empty(shape, dtype=np.float64)
                constants = metal.asarray(np.asarray([
                    GRAVITY_M_S2,
                    self.options.background_diffusivity_m2_s,
                    self.options.pbl_diffusivity_m2_s,
                    self.options.pbl_depth_m,
                ], dtype=np.float64))
                geopotential = upload(exchange.geopotential, shape)
                self.metal_context.submit_kernel(
                    "global_turbulence_geometry_fragment",
                    {"geopotential": geopotential, "constants": constants,
                     "height": height_device.view((nlevel * ncolumn,), dtype=np.float64),
                     "diffusivity": diffusivity_device.view(
                         (nlevel * ncolumn,), dtype=np.float64)},
                    {"nlevel": nlevel, "ncolumn": ncolumn},
                    exact_policy=False, grid_size=nlevel * ncolumn,
                    threads_per_group=(
                        kernel_specs.GLOBAL_TURBULENCE_GEOMETRY_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                height_resource = height_device.view(
                    (nlevel * ncolumn,), dtype=np.float64
                )
                diffusivity_resource = diffusivity_device.view(
                    (nlevel * ncolumn,), dtype=np.float64
                )
            else:
                height_resource = upload(height_agl, shape)
                diffusivity_resource = upload(diffusivity, shape)

            resources = {
                "fields": fields if resident else upload(host_fields, (6, *shape)),
                "height": height_resource, "dp": upload(dp, shape),
                "diffusivity": diffusivity_resource,
                "dt": metal.asarray(np.asarray([dt], dtype=np.float64)),
                "updated_fields": metal.empty((6 * nlevel * ncolumn,), dtype=np.float64),
            }
            self.metal_context.submit_kernel(
                "global_implicit_diffusion_fragment", resources,
                {"nlevel": nlevel, "ncolumn": ncolumn, "nfield": 6},
                exact_policy=False, grid_size=ncolumn,
                threads_per_group=kernel_specs.GLOBAL_IMPLICIT_DIFFUSION_FRAGMENT_THREADS_PER_GROUP,
            )
            if resident:
                updated = resources["updated_fields"]
                plane_bytes = nlevel * ncolumn * np.dtype(np.float64).itemsize
                u, v, theta = (
                    updated.view(shape, dtype=np.float64, byte_offset=i * plane_bytes)
                    for i in range(3)
                )
                q["qv"], q["qc"], q["qi"] = (
                    updated.view(shape, dtype=np.float64, byte_offset=i * plane_bytes)
                    for i in range(3, 6)
                )
            else:
                updated = resources["updated_fields"].to_numpy().reshape((6, *shape))
                u, v, theta = updated[:3]
                q["qv"], q["qc"], q["qi"] = updated[3:]
            return u, v, theta, q
        u = _implicit_diffuse(u, z, dp, diffusivity, dt, xp)
        v = _implicit_diffuse(v, z, dp, diffusivity, dt, xp)
        theta = _implicit_diffuse(theta, z, dp, diffusivity, dt, xp)
        for name in ("qv", "qc", "qi"):
            q[name] = _implicit_diffuse(q[name], z, dp, diffusivity, dt, xp)
        return u, v, theta, q

    def _stratospheric_floor_pull(self, theta, exner, p_full, dt, xp):
        """One-sided pull toward the stratospheric floor, in theta.

        Points above stratospheric_floor_pa colder than the floor gain
        max(floor - T, 0) * dt / tau per call; nothing is ever cooled and
        nothing at or above the floor temperature is touched, so the warm
        side of the T533 wave train (+16 K at 223 K, level 1 of the
        hour-3 crash state) stays free.  Conservation consequence: the
        pull adds column enthalpy with no compensating booking - it
        stands in for the ozone/residual-circulation warming the gray
        absorber cannot supply.  Measured on the v6 T533 hour-3 crash
        state over a 30-minute absorber-only window (60 steps of
        dt=30 s): +23.5 K at the 158.05 K pocket point (0.0130 K/s mean,
        against the measured 0.0134 K/s wave cooling; the one-sided pull
        balances that driver at 195 - 0.0134*1800 = 171 K), grid-mean
        booking +2.5e4 J/m2 (14.1 W/m2 over the window, an extreme
        state's bill) confined to points colder than 195 K above
        5000 Pa.
        """
        if self.options.stratospheric_floor_k <= 0.0:
            return theta
        if self.metal_context is not None and xp is np:
            from woof.metal.array import MetalArray, MetalArrayModule
            resident_theta = isinstance(theta, MetalArray)
            if resident_theta:
                if theta.buffer.context is not self.metal_context:
                    raise ValueError(
                        "Global stratospheric-floor input belongs to another Metal context"
                    )
                if theta.dtype != np.dtype(np.float64):
                    raise TypeError("Global stratospheric-floor input must be FP64")
                shape = tuple(theta.shape)
                theta_device = theta.view((theta.size,), dtype=np.float64)
            else:
                theta_host = np.ascontiguousarray(np.asarray(theta, dtype=np.float64))
                shape = tuple(theta_host.shape)
                theta_device = None
            count = int(np.prod(shape))
            from woof.metal.generated import kernel_specs

            metal = MetalArrayModule(self.metal_context)

            def resident_field(value, name):
                if isinstance(value, MetalArray):
                    if (value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != shape):
                        raise ValueError(
                            f"Global stratospheric-floor {name} must be a "
                            "matching FP64 MetalArray"
                        )
                    return value.view((count,), dtype=np.float64)
                host = np.ascontiguousarray(np.broadcast_to(
                    np.asarray(value, dtype=np.float64), shape
                ))
                return metal.asarray(host.reshape(-1))

            resources = {
                "theta": (theta_device if resident_theta else
                          metal.asarray(theta_host.reshape(-1))),
                "exner": resident_field(exner, "exner"),
                "pressure": resident_field(p_full, "pressure"),
                "updated_theta": metal.empty((count,), dtype=np.float64),
                "floor_temperature": metal.asarray(np.asarray(
                    [self.options.stratospheric_floor_k], dtype=np.float64)),
                "floor_pressure": metal.asarray(np.asarray(
                    [self.options.stratospheric_floor_pa], dtype=np.float64)),
                "relax": metal.asarray(np.asarray(
                    [dt / self.options.stratospheric_relaxation_time_s], dtype=np.float64)),
            }
            self.metal_context.submit_kernel(
                "global_stratospheric_floor_fragment", resources,
                {"count": count}, exact_policy=False, grid_size=count,
                threads_per_group=kernel_specs.GLOBAL_STRATOSPHERIC_FLOOR_FRAGMENT_THREADS_PER_GROUP,
            )
            if resident_theta:
                return resources["updated_theta"].view(shape, dtype=np.float64)
            return resources["updated_theta"].to_numpy().reshape(shape)
        temperature_now = theta * exner
        cold = xp.maximum(
            self.options.stratospheric_floor_k - temperature_now, 0.0
        )
        in_stratosphere = p_full < self.options.stratospheric_floor_pa
        relax = dt / self.options.stratospheric_relaxation_time_s
        return theta + xp.where(in_stratosphere, cold * relax, 0.0) / exner

    def _convective_adjust(self, exchange, theta, q, xp):
        if not self.options.convection:
            return theta, q, 0
        if self.metal_context is not None and xp is np:
            from woof.metal.array import MetalArray, MetalArrayModule
            from woof.metal.generated import kernel_specs

            resident = isinstance(theta, MetalArray)
            shape = tuple(theta.shape)
            nlevel = int(shape[0])
            ncolumn = int(np.prod(shape[1:]))
            count = nlevel * ncolumn
            if resident:
                if (theta.buffer.context is not self.metal_context
                        or theta.dtype != np.dtype(np.float64)):
                    raise ValueError(
                        "Global convective theta must be FP64 on this Metal context"
                    )
                for name in WATER_SPECIES:
                    value = q[name]
                    if (not isinstance(value, MetalArray)
                            or value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != shape):
                        raise ValueError(
                            f"Global convective {name} must match resident theta"
                        )
            elif any(isinstance(q[name], MetalArray) for name in WATER_SPECIES):
                raise ValueError(
                    "Global convective theta and water fields must share residency"
                )
            q_names = tuple(WATER_SPECIES)
            metal = MetalArrayModule(self.metal_context)
            if resident:
                dp = exchange.dp
                if isinstance(dp, MetalArray):
                    if (dp.buffer.context is not self.metal_context
                            or dp.dtype != np.dtype(np.float64)
                            or tuple(dp.shape) != shape):
                        raise ValueError(
                            "Global convective dp must match resident theta: "
                            f"{getattr(dp, 'shape', None)} != {shape}"
                        )
                    dp_device = dp.view((count,), dtype=np.float64)
                else:
                    dp_device = metal.asarray(np.ascontiguousarray(
                        np.broadcast_to(
                            np.asarray(dp, dtype=np.float64), shape
                        ).reshape(-1)
                    ))
                mass_device = metal.empty((count,), dtype=np.float64)
                self.metal_context.submit_kernel(
                    "global_convective_mass_fragment",
                    {"dp": dp_device,
                     "gravity": metal.asarray(np.asarray(
                         [GRAVITY_M_S2], dtype=np.float64)),
                     "mass": mass_device},
                    {"count": count}, exact_policy=False, grid_size=count,
                    threads_per_group=(
                        kernel_specs.GLOBAL_CONVECTIVE_MASS_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                theta_device = metal.empty((count,), dtype=np.float64)
                q_device = metal.empty((len(q_names) * count,), dtype=np.float64)
                packed = {
                    "theta": theta.view((count,), dtype=np.float64),
                    **{name: q[name].view((count,), dtype=np.float64)
                       for name in q_names},
                    "theta_packed": theta_device,
                    "q_packed": q_device,
                }
                self.metal_context.submit_kernel(
                    "global_convective_pack_fragment", packed,
                    {"nlevel": nlevel, "ncolumn": ncolumn},
                    exact_policy=False, grid_size=count,
                    threads_per_group=(
                        kernel_specs.GLOBAL_CONVECTIVE_PACK_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
            else:
                mass_host = np.ascontiguousarray(
                    np.broadcast_to(
                        np.asarray(exchange.dp, dtype=np.float64), shape
                    ) / GRAVITY_M_S2
                ).reshape(-1)
                theta_host = np.ascontiguousarray(
                    np.asarray(theta, dtype=np.float64)
                )
                q_host = np.ascontiguousarray(np.stack([
                    np.asarray(q[name], dtype=np.float64) for name in q_names
                ]).reshape(len(q_names), nlevel, ncolumn))
                theta_device = metal.asarray(theta_host.reshape(-1))
                q_device = metal.asarray(q_host.reshape(-1))
            mixed = metal.empty((ncolumn,), dtype=np.uint32)
            resources = {
                "mass": mass_device if resident else metal.asarray(mass_host),
                "theta": theta_device,
                "q": q_device, "mixed_pairs": mixed,
            }
            self.metal_context.submit_kernel(
                "global_convective_adjust_fragment", resources,
                {"nlevel": nlevel, "ncolumn": ncolumn,
                 "nspecies": len(q_names)}, exact_policy=False,
                grid_size=ncolumn,
                threads_per_group=kernel_specs.GLOBAL_CONVECTIVE_ADJUST_FRAGMENT_THREADS_PER_GROUP,
            )
            mixed_total = metal.empty((1,), dtype=np.uint32)
            self.metal_context.submit_kernel(
                "global_convective_count_finalize_fragment",
                {"mixed_pairs": mixed, "mixed_total": mixed_total},
                {"count": ncolumn}, exact_policy=False, grid_size=1,
                threads_per_group=(
                    kernel_specs.GLOBAL_CONVECTIVE_COUNT_FINALIZE_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            if resident:
                theta = theta_device.view(shape, dtype=np.float64)
                for index, name in enumerate(q_names):
                    q[name] = q_device.view(
                        shape, dtype=np.float64,
                        byte_offset=index * count * np.dtype(np.float64).itemsize,
                    )
            else:
                theta[...] = theta_device.to_numpy().reshape(shape)
                q_result = q_device.to_numpy().reshape(len(q_names), *shape)
                for index, name in enumerate(q_names):
                    q[name][...] = q_result[index]
            return theta, q, int(mixed_total.to_numpy()[0])
        if xp is not np:
            return self._convective_adjust_cupy(exchange, theta, q, xp)
        mass = exchange.dp / GRAVITY_M_S2
        # Four deterministic sweeps are enough for the small research columns
        # and avoid a data-dependent host loop on the CuPy path.  The
        # instability count accumulates device-side across every pair of
        # every sweep and crosses to the host once per physics call: the
        # per-pair to_numpy it replaces was one device sync per layer pair
        # per sweep (152 per step at nlev=20).
        mixed_total = xp.zeros((), dtype=xp.int64)
        for _ in range(4):
            for k in range(theta.shape[0] - 1):
                unstable = theta[k] < theta[k + 1]
                total_mass = mass[k] + mass[k + 1]
                mixed_theta = (
                    mass[k] * theta[k] + mass[k + 1] * theta[k + 1]
                ) / xp.maximum(total_mass, 1.0e-12)
                theta[k] = xp.where(unstable, mixed_theta, theta[k])
                theta[k + 1] = xp.where(unstable, mixed_theta, theta[k + 1])
                for name in WATER_SPECIES:
                    mixed = (
                        mass[k] * q[name][k] + mass[k + 1] * q[name][k + 1]
                    ) / xp.maximum(total_mass, 1.0e-12)
                    q[name][k] = xp.where(unstable, mixed, q[name][k])
                    q[name][k + 1] = xp.where(unstable, mixed, q[name][k + 1])
                mixed_total = mixed_total + xp.count_nonzero(unstable)
        return theta, q, int(self.backend.to_numpy(mixed_total))

    def _convective_adjust_cupy(self, exchange, theta, q, xp):
        """One thread per column runs all four sweeps of the pair mix.

        The sweeps are sequential within a column (pair k feeds pair
        k+1) and independent across columns; theta and the six species
        mutate in place like the numpy path, and the instability count
        still crosses to the host exactly once per call.
        """
        kernel = _cupy_kernel(
            "reference_convective_pair_mix", _build_convective_adjust_kernel
        )
        theta = xp.ascontiguousarray(theta)
        for name in WATER_SPECIES:
            q[name] = xp.ascontiguousarray(q[name])
        # The kernel indexes the raw arrays with theta's plane stride, so a
        # broadcast-shaped dp (e.g. a (nlev, 1, 1) column exchange) must be
        # materialized to the full grid first.
        dp = xp.ascontiguousarray(xp.broadcast_to(exchange.dp, theta.shape))
        mixed = xp.empty(theta.shape[-2:], dtype=xp.int64)
        plane = int(theta.shape[-2] * theta.shape[-1])
        kernel(
            dp, np.int32(theta.shape[0]), np.int64(plane),
            theta, q["qv"], q["qc"], q["qr"], q["qi"], q["qs"], q["qg"],
            mixed,
        )
        return theta, q, int(self.backend.to_numpy(xp.sum(mixed)))

    def _betts_miller_adjust(self, exchange, theta, q, surface, xp):
        """Deep moist convective adjustment (Betts-Miller family).

        The suite's only other release of conditional instability is the
        grid-scale saturation adjustment, and on real tropical moisture that
        releases column instability one grid point at a time: the first
        real-data T63 run grew 168 m/s winds at 143 hPa over 17N by hour 18
        (500 hPa at 104 m/s) - textbook grid-point storms.  Columns with a
        deep buoyant cloud layer relax toward the lifted parcel's moist
        adiabat and a subsaturated reference humidity instead, with the
        column enthalpy change closed exactly against the precipitated
        water.
        """
        zeros = xp.zeros(theta.shape[-2:], dtype=theta.dtype)
        if not self.options.moist_convection:
            return theta, q, surface, zeros, 0
        if self.metal_context is not None and xp is np:
            from woof.metal.array import MetalArray, MetalArrayModule
            from woof.metal.generated import kernel_specs

            exner = exchange.exner
            resident = isinstance(theta, MetalArray)
            shape = tuple(theta.shape)
            nlevel = shape[0]
            ncolumn = int(np.prod(shape[1:]))
            count = nlevel * ncolumn
            metal = MetalArrayModule(self.metal_context)

            def upload(value):
                if isinstance(value, MetalArray):
                    if (value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or value.size != count):
                        raise ValueError(
                            "Global Betts-Miller input must be FP64 on this context"
                        )
                    return value.view((count,), dtype=np.float64)
                host = np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
                return metal.asarray(np.ascontiguousarray(host.reshape(-1)))

            if resident:
                if (theta.buffer.context is not self.metal_context
                        or theta.dtype != np.dtype(np.float64)):
                    raise ValueError(
                        "Global Betts-Miller theta must be FP64 on this Metal context"
                    )
                for name in WATER_SPECIES:
                    value = q[name]
                    if (not isinstance(value, MetalArray)
                            or value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != shape):
                        raise ValueError(
                            f"Global Betts-Miller {name} must match resident theta"
                        )
                temperature = metal.empty((count,), dtype=np.float64)
                self.metal_context.submit_kernel(
                    "global_betts_miller_temperature_fragment",
                    {"theta": theta.view((count,), dtype=np.float64),
                     "exner": upload(exner), "temperature": temperature},
                    {"count": count}, exact_policy=False, grid_size=count,
                    threads_per_group=(
                        kernel_specs.GLOBAL_BETTS_MILLER_TEMPERATURE_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
            else:
                temperature = theta * exner

            relax = 1.0 - math.exp(-float(exchange.dt_s) /
                                   self.options.convective_relaxation_time_s)
            scalar_values = {
                "relax": relax, "rh": self.options.convective_reference_rh,
                "minimum_depth": self.options.convective_min_depth_pa,
                "epsilon": EPSILON, "gravity": GRAVITY_M_S2,
                "rd": DRY_AIR_GAS_CONSTANT, "lv": LATENT_HEAT_VAPORIZATION,
                "cp": DRY_AIR_CP, "lv2": LATENT_HEAT_VAPORIZATION ** 2,
                "rd_cp": DRY_AIR_GAS_CONSTANT / DRY_AIR_CP,
            }
            resources = {
                "temperature": upload(temperature), "qv": upload(q["qv"]),
                "pressure": upload(exchange.p_full), "dp": upload(exchange.dp),
                "virtual_temperature": upload(exchange.virtual_temperature),
                **{name: metal.asarray(np.asarray([value], dtype=np.float64))
                   for name, value in scalar_values.items()},
                "temperature_out": metal.empty((nlevel * ncolumn,), dtype=np.float64),
                "qv_out": metal.empty((nlevel * ncolumn,), dtype=np.float64),
                "precip": metal.empty((ncolumn,), dtype=np.float64),
                "deep": metal.empty((ncolumn,), dtype=np.uint32),
            }
            self.metal_context.submit_kernel(
                "global_betts_miller_fragment", resources,
                {"nlevel": nlevel, "ncolumn": ncolumn}, exact_policy=False,
                grid_size=ncolumn,
                threads_per_group=kernel_specs.GLOBAL_BETTS_MILLER_FRAGMENT_THREADS_PER_GROUP,
            )
            deep_count = metal.empty((1,), dtype=np.uint32)
            self.metal_context.submit_kernel(
                "global_betts_miller_deep_finalize_fragment",
                {"deep": resources["deep"], "deep_count": deep_count},
                {"count": ncolumn}, exact_policy=False, grid_size=1,
                threads_per_group=(
                    kernel_specs.GLOBAL_BETTS_MILLER_DEEP_FINALIZE_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            precip_device = resources["precip"].view(shape[1:], dtype=np.float64)
            deep_columns = int(deep_count.to_numpy()[0])
            resident_surface_water = isinstance(surface.water_kg_m2, MetalArray)
            if resident_surface_water:
                surface.water_kg_m2 = self._metal_add_surface_water(
                    surface.water_kg_m2, precip_device
                )
                # Precipitation is an output plane. Keep it resident until
                # the band assembler reaches its output-diagnostic boundary.
                precip = precip_device
            else:
                precip = precip_device.to_numpy()
            if not resident_surface_water:
                surface.water_kg_m2 = surface.water_kg_m2 + precip
            if resident:
                self.metal_context.submit_kernel(
                    "global_betts_miller_theta_fragment",
                    {"temperature": resources["temperature_out"],
                     "exner": upload(exner),
                     "theta": theta.view((count,), dtype=np.float64)},
                    {"count": count}, exact_policy=False, grid_size=count,
                    threads_per_group=(
                        kernel_specs.GLOBAL_BETTS_MILLER_THETA_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                q["qv"] = resources["qv_out"].view(shape, dtype=np.float64)
            else:
                temperature_host = resources["temperature_out"].to_numpy().reshape(shape)
                q["qv"] = resources["qv_out"].to_numpy().reshape(shape)
                theta = temperature_host / exner
            return theta, q, surface, precip, deep_columns
        if xp is not np:
            return self._betts_miller_adjust_cupy(exchange, theta, q, surface, xp)
        dt = float(exchange.dt_s)
        exner = exchange.exner
        p = exchange.p_full
        temperature = theta * exner
        nlev = temperature.shape[0]
        mass = exchange.dp / GRAVITY_M_S2

        # Pseudo-adiabatic ascent of the lowest-level parcel, elementwise
        # over columns (one short loop over levels, matching _fallout).
        parcel_t = xp.empty_like(temperature)
        parcel_t[-1] = temperature[-1]
        parcel_q = xp.minimum(
            q["qv"][-1], saturation_mixing_ratio(temperature[-1], p[-1], xp)
        )
        saturated = parcel_q >= saturation_mixing_ratio(
            temperature[-1], p[-1], xp
        ) - 1.0e-12
        level_index = xp.arange(nlev)[:, None, None]
        lcl_level = xp.where(saturated, nlev - 1, -1)
        for k in range(nlev - 2, -1, -1):
            dlnp = xp.log(p[k] / p[k + 1])
            dry_t = parcel_t[k + 1] * (p[k] / p[k + 1]) ** (
                DRY_AIR_GAS_CONSTANT / DRY_AIR_CP
            )
            qs_prev = saturation_mixing_ratio(parcel_t[k + 1], p[k + 1], xp)
            moist_rate = (
                DRY_AIR_GAS_CONSTANT * parcel_t[k + 1]
                + LATENT_HEAT_VAPORIZATION * qs_prev
            ) / (
                DRY_AIR_CP
                + LATENT_HEAT_VAPORIZATION**2
                * qs_prev
                * EPSILON
                / (DRY_AIR_GAS_CONSTANT * parcel_t[k + 1] ** 2)
            )
            moist_t = parcel_t[k + 1] + moist_rate * dlnp
            is_saturated = saturated | (
                parcel_q >= saturation_mixing_ratio(dry_t, p[k], xp)
            )
            parcel_t[k] = xp.where(is_saturated, moist_t, dry_t)
            parcel_q = xp.where(
                is_saturated,
                saturation_mixing_ratio(parcel_t[k], p[k], xp),
                parcel_q,
            )
            lcl_level = xp.where(is_saturated & ~saturated, k, lcl_level)
            saturated = is_saturated

        # Cloud layer: buoyant (virtual temperature) levels at or above the
        # LCL; deep when the layer spans the configured pressure depth.
        parcel_virtual = parcel_t * (1.0 + 0.61 * parcel_q)
        buoyant = (
            (parcel_virtual > exchange.virtual_temperature)
            & (level_index <= lcl_level[None])
            & (lcl_level[None] >= 0)
        )
        has_cloud = xp.any(buoyant, axis=0)
        cloud_top = xp.argmax(buoyant, axis=0)
        base_gather = xp.clip(lcl_level, 0, nlev - 1)
        p_base = xp.take_along_axis(p, base_gather[None], axis=0)[0]
        p_top = xp.take_along_axis(p, cloud_top[None], axis=0)[0]
        deep = has_cloud & (
            (p_base - p_top) >= self.options.convective_min_depth_pa
        )
        in_cloud = (
            (level_index >= cloud_top[None])
            & (level_index <= lcl_level[None])
            & deep[None]
        )

        # Relax toward the parcel adiabat and a subsaturated reference
        # humidity.  Two regimes close the column budgets exactly:
        # PRECIPITATING (raw relaxation dries the column): the dried water
        # rains to the surface reservoir and a uniform temperature offset
        # over the cloud layer sets cp*sum(dT dm) == L*P.
        # NON-PRECIPITATING (raw relaxation would moisten - there is no
        # water source here): the moistening levels are scaled back until
        # the column's water books to zero, and the temperature offset sets
        # the column enthalpy change to zero.  Skipping this regime instead
        # leaves the instability standing in exactly the deep dry-aloft
        # columns that grid-point storms feed on.
        relax = 1.0 - math.exp(-dt / self.options.convective_relaxation_time_s)
        reference_q = self.options.convective_reference_rh * (
            saturation_mixing_ratio(parcel_t, p, xp)
        )
        delta_t_raw = xp.where(in_cloud, (parcel_t - temperature) * relax, 0.0)
        delta_q = xp.where(in_cloud, (reference_q - q["qv"]) * relax, 0.0)
        drying = -xp.sum(xp.minimum(delta_q, 0.0) * mass, axis=0)
        moistening = xp.sum(xp.maximum(delta_q, 0.0) * mass, axis=0)
        precipitating = deep & (drying > moistening)
        # Non-precipitating: scale the moistening part down to what the
        # drying supplies (never below zero, so vapor stays nonnegative).
        moisten_scale = xp.where(
            deep & ~precipitating & (moistening > 0.0),
            drying / xp.maximum(moistening, 1.0e-30),
            1.0,
        )
        delta_q = xp.where(
            delta_q > 0.0, delta_q * moisten_scale[None], delta_q
        )
        precip = xp.where(
            precipitating, -xp.sum(delta_q * mass, axis=0), 0.0
        )
        # Column enthalpy closure as a uniform offset over the cloud layer
        # (>= the configured depth of mass by the deep gate, so the divisor
        # is never small): cp*sum(dT dm) == L*P, with P == 0 in the
        # non-precipitating regime.
        cloud_mass = xp.sum(xp.where(in_cloud, mass, 0.0), axis=0)
        heating_raw = xp.sum(delta_t_raw * mass, axis=0)
        correction = xp.where(
            deep,
            (LATENT_HEAT_VAPORIZATION * precip - DRY_AIR_CP * heating_raw)
            / (DRY_AIR_CP * xp.maximum(cloud_mass, 1.0e-6)),
            0.0,
        )
        delta_t = xp.where(in_cloud, delta_t_raw + correction[None], 0.0)

        temperature = temperature + delta_t
        q["qv"] = q["qv"] + delta_q
        surface.water_kg_m2 = surface.water_kg_m2 + precip
        deep_columns = int(np.count_nonzero(self.backend.to_numpy(deep)))
        return temperature / exner, q, surface, precip, deep_columns

    def _betts_miller_adjust_cupy(self, exchange, theta, q, surface, xp):
        """One thread per column: ascent recurrence, cloud layer, closure.

        The parcel ascent is a level recurrence (parcel_t[k] needs
        parcel_t[k+1]), so the per-level Python loop becomes an in-kernel
        walk; the buoyancy scan, both closure regimes, and the
        application run in the same launch.  qv is updated on a fresh
        copy and rebound, matching the numpy path's out-of-place
        q["qv"] = q["qv"] + delta_q.
        """
        kernel = _cupy_kernel(
            "reference_betts_miller_column", _build_betts_miller_kernel
        )
        dt = float(exchange.dt_s)
        exner = exchange.exner
        temperature = xp.ascontiguousarray(theta * exner)
        qv_new = q["qv"].copy()
        # Raw-array indexing uses temperature's plane stride: materialize
        # any broadcast-shaped column inputs to the full grid.
        p = xp.ascontiguousarray(xp.broadcast_to(exchange.p_full, temperature.shape))
        dp = xp.ascontiguousarray(xp.broadcast_to(exchange.dp, temperature.shape))
        tv = xp.ascontiguousarray(
            xp.broadcast_to(exchange.virtual_temperature, temperature.shape)
        )
        parcel = xp.empty_like(temperature)
        precip = xp.empty(temperature.shape[-2:], dtype=temperature.dtype)
        deep_flag = xp.empty(temperature.shape[-2:], dtype=xp.int32)
        relax = 1.0 - math.exp(-dt / self.options.convective_relaxation_time_s)
        cast = temperature.dtype.type
        plane = int(temperature.shape[-2] * temperature.shape[-1])
        kernel(
            p, dp, tv, cast(relax),
            cast(self.options.convective_reference_rh),
            cast(self.options.convective_min_depth_pa),
            np.int32(temperature.shape[0]), np.int64(plane),
            temperature, qv_new, parcel, precip, deep_flag,
        )
        q["qv"] = qv_new
        surface.water_kg_m2 = surface.water_kg_m2 + precip
        deep_columns = int(np.count_nonzero(self.backend.to_numpy(deep_flag)))
        return temperature / exner, q, surface, precip, deep_columns

    def _fallout(self, species, fall_speed, exchange, surface, accumulator, xp,
                 *, keep_resident: bool = False):
        dt = float(exchange.dt_s)
        if self.metal_context is not None and xp is np and species.shape[0] >= 2:
            from woof.metal.array import MetalArray, MetalArrayModule
            from woof.metal.generated import kernel_specs

            shape = tuple(species.shape)
            nlevel = shape[0]
            ncolumn = int(np.prod(shape[1:]))
            metal = MetalArrayModule(self.metal_context)

            def upload(value):
                if isinstance(value, MetalArray):
                    if value.buffer.context is not self.metal_context:
                        raise ValueError("Global fallout input belongs to another Metal context")
                    if value.dtype != np.dtype(np.float64):
                        raise TypeError("Global fallout inputs must be FP64")
                    resident = metal.asarray(value)
                    return (resident.view((resident.size,), dtype=np.float64)
                            if resident.ndim != 1 else resident)
                host = np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
                return metal.asarray(np.ascontiguousarray(host.reshape(-1)))

            resources = {
                "species": upload(species),
                "dp": upload(exchange.dp),
                "geopotential": upload(exchange.geopotential),
                "gravity": metal.asarray(np.asarray([GRAVITY_M_S2], dtype=np.float64)),
                "fall_speed_dt": metal.asarray(np.asarray(
                    [float(fall_speed) * dt], dtype=np.float64)),
                "species_out": metal.empty((nlevel * ncolumn,), dtype=np.float64),
                "precip": metal.empty((ncolumn,), dtype=np.float64),
            }
            self.metal_context.submit_kernel(
                "global_fallout_fragment", resources,
                {"nlevel": nlevel, "ncolumn": ncolumn}, exact_policy=False,
                grid_size=ncolumn,
                threads_per_group=kernel_specs.GLOBAL_FALLOUT_FRAGMENT_THREADS_PER_GROUP,
            )
            species = (
                resources["species_out"].view(shape, dtype=np.float64)
                if keep_resident else resources["species_out"].to_numpy().reshape(shape)
            )
            precip_device = resources["precip"].view(shape[1:], dtype=np.float64)
            resident_surface_water = isinstance(surface.water_kg_m2, MetalArray)
            if resident_surface_water:
                surface.water_kg_m2 = self._metal_add_surface_water(
                    surface.water_kg_m2, precip_device
                )
            resident_accumulator = isinstance(accumulator, MetalArray)
            if resident_accumulator:
                updated_accumulator = self._metal_add_surface_water(
                    accumulator, precip_device
                )
                accumulator[...] = updated_accumulator
            if resident_surface_water and resident_accumulator:
                # Precipitation is also an output plane. Keep it on Metal
                # until the grid assembler reaches that diagnostic boundary.
                precip = precip_device
            else:
                precip = precip_device.to_numpy()
            if not resident_surface_water:
                surface.water_kg_m2 += precip
            if not resident_accumulator:
                accumulator += precip
            return species, precip
        if xp is not np and species.shape[0] >= 2:
            # The nz >= 2 gate mirrors the dz[-1] = |z[-2] - z[-1]| read
            # below, which needs two levels to exist; a one-level column
            # takes the numpy path and fails the same way it always did.
            return self._fallout_cupy(
                species, fall_speed, exchange, surface, accumulator, xp
            )
        mass = exchange.dp / GRAVITY_M_S2
        z = exchange.geopotential / GRAVITY_M_S2
        dz = xp.empty_like(species)
        dz[:-1] = xp.maximum(xp.abs(z[:-1] - z[1:]), 20.0)
        dz[-1] = xp.maximum(xp.abs(z[-2] - z[-1]), 20.0)
        precip = xp.zeros(species.shape[-2:], dtype=species.dtype)
        for k in range(species.shape[0]):
            fraction = 1.0 - xp.exp(-float(fall_speed) * dt / dz[k])
            removed_mixing = species[k] * xp.clip(fraction, 0.0, 1.0)
            removed_mass = removed_mixing * mass[k]
            species[k] -= removed_mixing
            if k + 1 < species.shape[0]:
                species[k + 1] += removed_mass / xp.maximum(mass[k + 1], 1.0e-12)
            else:
                precip += removed_mass
        surface.water_kg_m2 += precip
        accumulator += precip
        return species, precip

    def _fallout_cupy(self, species, fall_speed, exchange, surface, accumulator, xp):
        """One thread per column walks the sequential sedimentation loop.

        Level k+1 receives level k's flux before shedding its own, so the
        levels cannot run as one stacked elementwise pass; the walk stays
        a loop, inside the kernel.
        """
        dt = float(exchange.dt_s)
        kernel = _cupy_kernel("reference_fallout_column", _build_fallout_kernel)
        species = xp.ascontiguousarray(species)
        # Raw-array indexing uses the species' plane stride: materialize
        # any broadcast-shaped column inputs to the full grid.
        dp = xp.ascontiguousarray(xp.broadcast_to(exchange.dp, species.shape))
        phi = xp.ascontiguousarray(
            xp.broadcast_to(exchange.geopotential, species.shape)
        )
        precip = xp.empty(species.shape[-2:], dtype=species.dtype)
        cast = species.dtype.type
        plane = int(species.shape[-2] * species.shape[-1])
        kernel(
            dp, phi, cast(float(fall_speed) * dt),
            np.int32(species.shape[0]), np.int64(plane),
            species, precip,
        )
        surface.water_kg_m2 += precip
        accumulator += precip
        return species, precip

    def _microphysics_chain_cupy(self, exchange, temperature, q, xp):
        """Fused conversion chain: one kernel for the ~80-op sequence.

        Covers rain evaporation through the sublimation ladder; the
        dt-dependent fractions are host scalars cast to the field dtype,
        matching how numpy folds a Python float into a float32 ufunc.
        temperature and the six species mutate in place via aliased
        input/output parameters.
        """
        dt = float(exchange.dt_s)
        kernel = _cupy_kernel(
            "reference_microphysics_chain", _build_microphysics_kernel
        )
        cast = temperature.dtype.type
        kernel(
            temperature, q["qv"], q["qc"], q["qr"], q["qi"], q["qs"], q["qg"],
            exchange.p_full,
            cast(1.0 - math.exp(-dt / 600.0)),
            cast(self.options.cloud_autoconversion_threshold),
            cast(1.0 - math.exp(-dt / self.options.cloud_autoconversion_time_s)),
            cast(dt * 35.0),
            cast(1.0 - math.exp(-dt / 900.0)),
            cast(1.0 - math.exp(-dt / self.options.ice_to_snow_time_s)),
            cast(1.0 - math.exp(-dt / self.options.snow_to_graupel_time_s)),
            cast(1.0 - math.exp(-dt / 600.0)),
            cast(1.0 - math.exp(-dt / 1800.0)),
            temperature, q["qv"], q["qc"], q["qr"], q["qi"], q["qs"], q["qg"],
        )

    def _microphysics(self, exchange, theta, q, surface, xp,
                      *, keep_resident: bool = False):
        dt = float(exchange.dt_s)
        exner = exchange.exner
        temperature = None
        if self.metal_context is not None and xp is np:
            from woof.metal.array import MetalArray, MetalArrayModule
            from woof.metal.generated import kernel_specs

            shape = tuple(theta.shape)
            count = int(np.prod(shape))
            metal = MetalArrayModule(self.metal_context)

            def upload(value):
                if isinstance(value, MetalArray):
                    if value.buffer.context is not self.metal_context:
                        raise ValueError(
                            "Global microphysics input belongs to another Metal context"
                        )
                    if value.dtype != np.dtype(np.float64):
                        raise TypeError("Global microphysics inputs must be FP64")
                    resident = metal.asarray(value)
                    return (resident.view((resident.size,), dtype=np.float64)
                            if resident.ndim != 1 else resident)
                return metal.asarray(np.ascontiguousarray(
                    np.asarray(value, dtype=np.float64).reshape(-1)))

            scalar_values = {
                "epsilon": EPSILON, "lv_cp": LATENT_HEAT_VAPORIZATION / DRY_AIR_CP,
                "lf_cp": LATENT_HEAT_FUSION / DRY_AIR_CP,
                "ls_cp": LATENT_HEAT_SUBLIMATION / DRY_AIR_CP,
                "latent_heat_sub": LATENT_HEAT_SUBLIMATION,
                "rv": WATER_VAPOR_GAS_CONSTANT,
                "rain_evap_fraction": 1.0 - math.exp(-dt / 600.0),
                "auto_threshold": self.options.cloud_autoconversion_threshold,
                "auto_fraction": 1.0 - math.exp(-dt / self.options.cloud_autoconversion_time_s),
                "accrete_coefficient": dt * 35.0,
                "freeze_fraction": 1.0 - math.exp(-dt / 900.0),
                "ice_snow_fraction": 1.0 - math.exp(-dt / self.options.ice_to_snow_time_s),
                "snow_graupel_fraction": 1.0 - math.exp(-dt / self.options.snow_to_graupel_time_s),
                "melt_fraction": 1.0 - math.exp(-dt / 600.0),
                "sublimation_fraction": 1.0 - math.exp(-dt / 1800.0),
            }
            resources = {
                "theta": upload(theta), "exner": upload(exner),
                "pressure": upload(
                    exchange.p_full if isinstance(exchange.p_full, MetalArray)
                    else np.broadcast_to(exchange.p_full, shape)
                ),
                **{name: upload(q[name]) for name in ("qv", "qc", "qr", "qi", "qs", "qg")},
                **{name: upload(np.asarray([value], dtype=np.float64))
                   for name, value in scalar_values.items()},
            }
            for name in ("theta_out", "qv_out", "qc_out", "qr_out", "qi_out", "qs_out", "qg_out"):
                resources[name] = metal.empty((count,), dtype=np.float64)
            self.metal_context.submit_kernel(
                "global_microphysics_fragment", resources, {"count": count},
                exact_policy=False, grid_size=count,
                threads_per_group=kernel_specs.GLOBAL_MICROPHYSICS_FRAGMENT_THREADS_PER_GROUP,
            )
            if keep_resident:
                theta = resources["theta_out"].view(shape, dtype=np.float64)
                for key in ("qv", "qc", "qr", "qi", "qs", "qg"):
                    q[key] = resources[f"{key}_out"].view(shape, dtype=np.float64)
            else:
                theta = resources["theta_out"].to_numpy().reshape(shape)
                for key in ("qv", "qc", "qr", "qi", "qs", "qg"):
                    q[key] = resources[f"{key}_out"].to_numpy().reshape(shape)
        elif xp is not np:
            temperature = theta * exner
            self._microphysics_chain_cupy(exchange, temperature, q, xp)
        else:
            temperature = theta * exner
            qsat = saturation_mixing_ratio(temperature, exchange.p_full, xp)

            # Rain evaporation before conversion/fallout.
            deficit = xp.maximum(qsat - q["qv"], 0.0)
            rain_evap = xp.minimum(q["qr"], deficit * (1.0 - math.exp(-dt / 600.0)))
            q["qr"] -= rain_evap
            q["qv"] += rain_evap
            temperature -= (LATENT_HEAT_VAPORIZATION / DRY_AIR_CP) * rain_evap

            auto_fraction = 1.0 - math.exp(-dt / self.options.cloud_autoconversion_time_s)
            autoconvert = xp.maximum(
                q["qc"] - self.options.cloud_autoconversion_threshold, 0.0
            ) * auto_fraction
            q["qc"] -= autoconvert
            q["qr"] += autoconvert
            accrete = xp.minimum(q["qc"], dt * 35.0 * q["qc"] * xp.sqrt(xp.maximum(q["qr"], 0.0)))
            q["qc"] -= accrete
            q["qr"] += accrete

            freeze_weight = xp.clip((273.15 - temperature) / 15.0, 0.0, 1.0)
            freeze_fraction = freeze_weight * (1.0 - math.exp(-dt / 900.0))
            freeze_cloud = q["qc"] * freeze_fraction
            freeze_rain = q["qr"] * freeze_fraction * 0.5
            q["qc"] -= freeze_cloud
            q["qr"] -= freeze_rain
            q["qi"] += freeze_cloud
            q["qs"] += freeze_rain
            temperature += (LATENT_HEAT_FUSION / DRY_AIR_CP) * (freeze_cloud + freeze_rain)

            snow_fraction = 1.0 - math.exp(-dt / self.options.ice_to_snow_time_s)
            ice_to_snow = q["qi"] * snow_fraction * xp.clip(q["qi"] / 1.0e-4, 0.0, 1.0)
            q["qi"] -= ice_to_snow
            q["qs"] += ice_to_snow
            graupel_fraction = 1.0 - math.exp(-dt / self.options.snow_to_graupel_time_s)
            snow_to_graupel = q["qs"] * graupel_fraction * xp.clip(q["qc"] / 5.0e-4, 0.0, 1.0)
            q["qs"] -= snow_to_graupel
            q["qg"] += snow_to_graupel

            # Melting closes the fusion ledger the freezing terms open: without
            # it every vapor->ice->surface cycle leaves +Lf in the column. The
            # 5 K ramp and 600 s timescale melt falling snow within a few
            # hundred meters below the freezing level at the O(1 m/s) configured
            # fall speeds, mirroring the freezing ramp above.
            melt_weight = xp.clip((temperature - 273.15) / 5.0, 0.0, 1.0)
            melt_fraction = melt_weight * (1.0 - math.exp(-dt / 600.0))
            melt_ice = q["qi"] * melt_fraction
            melt_snow = q["qs"] * melt_fraction
            melt_graupel = q["qg"] * melt_fraction
            q["qi"] -= melt_ice
            q["qc"] += melt_ice
            q["qs"] -= melt_snow
            q["qg"] -= melt_graupel
            q["qr"] += melt_snow + melt_graupel
            temperature -= (LATENT_HEAT_FUSION / DRY_AIR_CP) * (
                melt_ice + melt_snow + melt_graupel
            )

            # Sublimation returns frozen condensate to vapor in subsaturated
            # cold air at Ls = Lv + Lf, the exact reverse of the condense-freeze
            # path.  The claimable deficit is divided by the latent feedback
            # 1 + (Ls/cp) dqsat/dT: consuming raw deficit d also cools the layer
            # by (Ls/cp) d and drops qsat by ~0.75 d at 265 K, so an uncorrected
            # cap left layers at RH 1.06..1.11 in one step and the default
            # saturation adjustment then condensed the excess as LIQUID cloud
            # 9 K below freezing.
            qsat = saturation_mixing_ratio(temperature, exchange.p_full, xp)
            cold_weight = xp.clip((273.15 - temperature) / 5.0, 0.0, 1.0)
            sublimation_fraction = cold_weight * (1.0 - math.exp(-dt / 1800.0))
            latent_feedback = 1.0 + (LATENT_HEAT_SUBLIMATION / DRY_AIR_CP) * (
                qsat * LATENT_HEAT_SUBLIMATION
                / (WATER_VAPOR_GAS_CONSTANT * temperature * temperature)
            )
            remaining_deficit = xp.maximum(qsat - q["qv"], 0.0) / latent_feedback
            for name in ("qi", "qs", "qg"):
                sublimated = xp.minimum(
                    q[name] * sublimation_fraction, remaining_deficit
                )
                q[name] -= sublimated
                q["qv"] += sublimated
                remaining_deficit = remaining_deficit - sublimated
                temperature -= (LATENT_HEAT_SUBLIMATION / DRY_AIR_CP) * sublimated

        if temperature is not None:
            theta = temperature / exner
        q["qr"], rain = self._fallout(
            q["qr"], self.options.rain_fall_speed_m_s, exchange, surface,
            surface.accumulated_rain_kg_m2, xp, keep_resident=keep_resident,
        )
        q["qs"], snow = self._fallout(
            q["qs"], self.options.snow_fall_speed_m_s, exchange, surface,
            surface.accumulated_snow_kg_m2, xp, keep_resident=keep_resident,
        )
        q["qg"], graupel = self._fallout(
            q["qg"], self.options.graupel_fall_speed_m_s, exchange, surface,
            surface.accumulated_graupel_kg_m2, xp, keep_resident=keep_resident,
        )
        # Frozen fallout melts into the liquid reservoir only where the
        # surface is above freezing, and that melt is paid from the surface
        # energy budget.  Below freezing no melt is thermodynamically
        # possible and the single-reservoir surface has no frozen store, so
        # the fallout joins the reservoir with its fusion heat unbooked -
        # a stated limit.  An unconditional debit measured -0.42 K/day of
        # compounding one-way cooling on a 250 K surface under 1 mm/h snow.
        from woof.metal.array import MetalArray
        if isinstance(surface.temperature_k, MetalArray):
            surface.temperature_k = self._metal_surface_fallout_cooling(
                surface.temperature_k, snow, graupel,
                surface.heat_capacity_j_m2_k,
            )
        else:
            surface.temperature_k = surface.temperature_k - xp.where(
                surface.temperature_k > 273.15,
                LATENT_HEAT_FUSION * (snow + graupel)
                / xp.maximum(surface.heat_capacity_j_m2_k, 1.0e4),
                0.0,
            )
        from woof.metal.array import MetalArray

        def rate(value):
            # The resident precipitation planes are mass per area for this
            # step; finish() converts them to rates after output assembly.
            return value if isinstance(value, MetalArray) else value / dt

        return theta, q, surface, rate(rain), rate(snow), rate(graupel)

    def step(self, exchange: PhysicsExchange) -> PhysicsResult:
        exchange.validate()
        xp = self.backend.xp
        from woof.metal.array import MetalArray, MetalArrayModule

        prognostic_names = ("u", "v", "theta", *WATER_SPECIES, *NUMBER_MOMENTS)
        prognostic_inputs = {
            name: getattr(exchange, name) for name in prognostic_names
        }
        resident_flags = {
            name: isinstance(value, MetalArray)
            for name, value in prognostic_inputs.items()
        }
        if any(resident_flags.values()) and not all(resident_flags.values()):
            raise ValueError(
                "Global physics prognostics must all be resident together or all be host arrays"
            )
        resident_input = (
            all(resident_flags.values())
            or (self.metal_context is not None and xp is np)
        )
        if resident_input and self.metal_context is None:
            raise ValueError("resident Global physics input has no Metal context")
        metal = MetalArrayModule(self.metal_context) if resident_input else None

        shape = tuple(exchange.theta.shape)

        def workspace(name, value):
            if isinstance(value, MetalArray):
                if self.metal_context is None or value.buffer.context is not self.metal_context:
                    raise ValueError(
                        f"Global physics input {name} belongs to another Metal context"
                    )
                if value.dtype != np.dtype(np.float64):
                    raise TypeError(
                        f"Global physics input {name} must be FP64 on Metal"
                    )
                if tuple(value.shape) != shape:
                    raise ValueError(
                        f"Global physics input {name} has shape {value.shape}, expected {shape}"
                    )
                # The suite mutates its stage. A device copy preserves the
                # caller's state and does not round-trip through NumPy.
                return value.copy()
            if resident_input:
                return metal.asarray(value, dtype=np.float64)
            return xp.asarray(value, dtype=self.backend.float_dtype).copy()

        staged = {
            name: workspace(name, value)
            for name, value in prognostic_inputs.items()
        }
        u, v, theta = staged["u"], staged["v"], staged["theta"]
        q = {name: staged[name] for name in WATER_SPECIES}
        moments = {name: staged[name] for name in NUMBER_MOMENTS}
        # ``staged_surface`` is ``SurfaceState.copy`` for a surface the
        # card holds, and the copy-that-is-the-stage for one the pinned
        # host tier holds.  Either way this suite works on its own copy
        # and never writes the caller's.
        surface = staged_surface(xp, exchange.surface)
        if resident_input:
            # Keep the water reservoirs on the physics context so surface
            # flux and precipitation feedback do not download full planes
            # just to update host-backed state.
            for member in (
                "water_kg_m2", "accumulated_rain_kg_m2",
                "accumulated_snow_kg_m2", "accumulated_graupel_kg_m2",
            ):
                value = getattr(surface, member)
                if isinstance(value, MetalArray):
                    if (value.buffer.context is not self.metal_context
                            or value.dtype != np.dtype(np.float64)
                            or tuple(value.shape) != tuple(shape[-2:])):
                        raise ValueError(
                            f"Global surface field {member} must be a matching "
                            "FP64 plane on this Metal context"
                        )
                    continue
                host = np.asarray(value, dtype=np.float64)
                if tuple(host.shape) != tuple(shape[-2:]):
                    raise ValueError(
                        f"Global surface field {member} has shape {host.shape}, "
                        f"expected {tuple(shape[-2:])}"
                    )
                setattr(surface, member, metal.asarray(
                    np.ascontiguousarray(host),
                    dtype=np.float64,
                ))
        before_local = (
            self._metal_column_water(q, exchange.dp, surface.water_kg_m2, theta.shape)
            if isinstance(q["qv"], MetalArray)
            else _column_water(q, exchange.dp, xp) + surface.water_kg_m2
        )

        band = exchange.band
        self._observe("start", u, v, theta, q, band)
        u, v, theta, q, surface, diagnostics, planes = self._surface_radiation_fluxes(
            exchange, u, v, theta, q, surface, xp
        )
        self._observe("surface_radiation_fluxes", u, v, theta, q, band)
        u, v, theta, q = self._turbulence(exchange, u, v, theta, q, xp)
        self._observe("turbulence", u, v, theta, q, band)
        theta, q, mixed_count = self._convective_adjust(exchange, theta, q, xp)
        self._observe("convective_adjust", u, v, theta, q, band)
        theta, q, surface, convective_precip, deep_columns = (
            self._betts_miller_adjust(exchange, theta, q, surface, xp)
        )
        self._observe("betts_miller", u, v, theta, q, band)
        condensed = xp.zeros(theta.shape[-2:], dtype=theta.dtype)
        condensed_parts = []
        if self.options.saturation_adjustment:
            theta, condensed = self._saturation_adjust(
                theta, q, exchange.exner, exchange.p_full, xp,
                keep_resident=bool(
                    self.metal_context is not None and xp is np
                ),
                keep_condensed_resident=bool(
                    self.metal_context is not None and xp is np
                ),
            )
            if isinstance(condensed, MetalArray):
                condensed_parts.append(condensed)
        self._observe("saturation_adjust", u, v, theta, q, band)

        zeros = xp.zeros(theta.shape[-2:], dtype=theta.dtype)
        rain_rate = zeros.copy()
        snow_rate = zeros.copy()
        graupel_rate = zeros.copy()
        if self.options.microphysics:
            theta, q, surface, rain_rate, snow_rate, graupel_rate = self._microphysics(
                exchange, theta, q, surface, xp,
                keep_resident=bool(self.metal_context is not None and xp is np),
            )
            if self.options.saturation_adjustment:
                theta, extra = self._saturation_adjust(
                    theta, q, exchange.exner, exchange.p_full, xp,
                    keep_resident=bool(self.metal_context is not None and xp is np),
                    keep_condensed_resident=bool(
                        self.metal_context is not None and xp is np
                    ),
                )
                if condensed_parts and isinstance(extra, MetalArray):
                    condensed_parts.append(extra)
                else:
                    condensed += extra
        self._observe("microphysics", u, v, theta, q, band)

        # Reject real defects, then remove only roundoff-scale negatives.
        from woof.metal.array import MetalArray

        if isinstance(theta, MetalArray):
            q, surface.water_kg_m2, maximum_water_repair = (
                self._metal_saturation_closure(
                    theta, q, exchange.dp, before_local, surface.water_kg_m2
                )
            )
            self._observe("negative_clamp", u, v, theta, q, band)
        else:
            for name, value in q.items():
                if isinstance(value, MetalArray):
                    value = value.to_numpy()
                minimum = float(self.backend.to_numpy(value).min())
                if minimum < -1.0e-10:
                    raise FloatingPointError(
                        f"reference physics produced negative {name}: min={minimum:.9g}"
                    )
                q[name] = xp.maximum(value, 0.0)
            self._observe("negative_clamp", u, v, theta, q, band)
            if not bool(xp.all(xp.isfinite(theta))):
                raise FloatingPointError("reference physics produced non-finite theta")
            if bool(xp.any(surface.water_kg_m2 < -1.0e-8)):
                raise FloatingPointError("reference physics exhausted the surface-water reservoir")
            surface.water_kg_m2 = xp.maximum(surface.water_kg_m2, 0.0)

            # Close local atmosphere+surface water after roundoff clipping.
            after_local = _column_water(q, exchange.dp, xp) + surface.water_kg_m2
            water_repair = after_local - before_local
            surface.water_kg_m2 -= water_repair
            if bool(xp.any(surface.water_kg_m2 < -1.0e-8)):
                raise FloatingPointError(
                    "physics water repair requires more surface water than available"
                )
            surface.water_kg_m2 = xp.maximum(surface.water_kg_m2, 0.0)
            maximum_water_repair = float(
                np.max(np.abs(self.backend.to_numpy(water_repair)))
            )

        # The grid means go back as planes for finish() to reduce once
        # over the assembled globe; the counts and maxima fold exactly.
        if len(condensed_parts) == 2:
            maximum_condensation = self._metal_saturation_pair_maximum(
                condensed_parts[0], condensed_parts[1]
            )
        elif condensed_parts:
            maximum_condensation = self._metal_maximum(condensed_parts[0])
        else:
            maximum_condensation = float(self.backend.to_numpy(condensed).max())
        diagnostics.update(
            convective_pair_mixes=float(mixed_count),
            deep_convective_columns=float(deep_columns),
            maximum_column_condensation_kg_kg=maximum_condensation,
            maximum_local_water_repair_kg_m2=maximum_water_repair,
        )
        if any(isinstance(value, MetalArray)
               for value in (rain_rate, snow_rate, graupel_rate)):
            planes.update(
                convective_precip=convective_precip,
                rain_amount=rain_rate, snow_amount=snow_rate,
                graupel_amount=graupel_rate,
            )
        else:
            planes.update(
                convective_precip=convective_precip,
                rain_rate=rain_rate, snow_rate=snow_rate,
                graupel_rate=graupel_rate,
            )
        physics_state = staged_physics_state(xp, exchange.physics_state)
        physics_state.arrays.update(self._screen_level_diagnostics(
            exchange, u, v, theta, q, surface, xp
        ))
        physics_state.metadata[_SURFACE_DIAGNOSTICS_KEY] = _SURFACE_DIAGNOSTICS_SOURCE
        result = PhysicsResult(
            u=u,
            v=v,
            theta=theta,
            qv=q["qv"],
            qc=q["qc"],
            qr=q["qr"],
            qi=q["qi"],
            qs=q["qs"],
            qg=q["qg"],
            nc=moments["nc"],
            nr=moments["nr"],
            ni=moments["ni"],
            ns=moments["ns"],
            ng=moments["ng"],
            surface=surface,
            physics_state=physics_state,
            diagnostics=diagnostics,
            adapter_receipt=self.identity,
            planes=planes,
        )
        if band is None:
            # The whole grid, handed over by a harness or a test rather
            # than by the model's band loop: the one result is the call.
            from .banding import finish_alone

            return finish_alone(self, exchange, result)
        return result

    def _screen_level_diagnostics(self, exchange, u, v, theta, q, surface, xp):
        """2 m T/q and 10 m wind by Monin-Obukhov similarity (task 1b).

        Formulation and stability functions are documented in
        ``physics.surface_diagnostics``; the surface humidity is the same
        bucket-beta effective value the evaporation used, so the profile
        is the one the fluxes were computed against.
        """
        from .surface_diagnostics import (
            effective_surface_humidity,
            similarity_surface_diagnostics,
        )

        bottom = -1
        from woof.metal.array import MetalArray

        if self.metal_context is not None and xp is np:
            from woof.metal.array import MetalArray, MetalArrayModule
            from woof.metal.generated import kernel_specs

            shape = tuple(surface.temperature_k.shape)
            count = int(np.prod(shape))
            metal = MetalArrayModule(self.metal_context)
            values = {
                "u": u[bottom],
                "v": v[bottom],
                "theta": theta[bottom],
                "exner": exchange.exner[bottom],
                "qv": q["qv"][bottom],
                "p_full": exchange.p_full[bottom],
                "p_surface": exchange.p_half[-1],
                "skin_temperature": surface.temperature_k,
                "land_fraction": surface.land_fraction,
                "soil_wetness": surface.soil_water_fraction[0],
                "roughness": surface.roughness_m,
            }

            def device_input(value):
                if isinstance(value, MetalArray):
                    if value.buffer.context is not self.metal_context:
                        raise ValueError(
                            "Global screen-diagnostics input belongs to another Metal context"
                        )
                    if value.dtype != np.dtype(np.float64):
                        raise TypeError(
                            "Global screen-diagnostics inputs must be FP64"
                        )
                    if value.shape != shape:
                        raise ValueError(
                            "Global screen-diagnostics Metal input has shape "
                            f"{value.shape}, expected {shape}"
                        )
                    return value.view((count,), dtype=np.float64)
                host = np.asarray(value, dtype=np.float64)
                return metal.asarray(np.ascontiguousarray(
                    np.broadcast_to(host, shape).reshape(-1)
                ))

            resources = {name: device_input(value)
                         for name, value in values.items()}
            resources["thermodynamic_constants"] = metal.asarray(np.asarray(
                [
                    1.0, 0.95, 611.2, 17.67, 273.15, 150.0, 350.0,
                    EPSILON, self.options.soil_wetness_capacity, 29.65,
                ],
                dtype=np.float64,
            ))
            for name in ("t2", "th2", "q2", "u10", "v10"):
                resources[name] = metal.empty((count,), dtype=np.float64)
            self.metal_context.submit_kernel(
                "global_screen_diagnostics_resident_fragment", resources,
                {"count": count}, exact_policy=False, grid_size=count,
                threads_per_group=(
                    kernel_specs.GLOBAL_SCREEN_DIAGNOSTICS_RESIDENT_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            # A whole-grid result can pass these state planes directly to the
            # resident Global assembler. Split bands still materialize here:
            # the current assembler's band writes accept backend arrays.
            whole_grid = (exchange.band is None or
                          tuple(exchange.band) == (0, int(shape[-2])))
            if whole_grid:
                return {name: resources[name].view(shape)
                        for name in ("t2", "th2", "q2", "u10", "v10")}
            return {name: resources[name].to_numpy().reshape(shape)
                    for name in ("t2", "th2", "q2", "u10", "v10")}
        theta_bottom = theta[bottom]
        qv_bottom = q["qv"][bottom]
        temperature = theta_bottom * exchange.exner[bottom]

        def output_host(value):
            return value.to_numpy() if isinstance(value, MetalArray) else value

        skin_temperature = output_host(surface.temperature_k)
        surface_pressure = output_host(exchange.p_half[-1])
        land_fraction = output_host(surface.land_fraction)
        wetness = output_host(surface.soil_water_fraction[0]) / (
            self.options.soil_wetness_capacity
        )
        humidity = effective_surface_humidity(
            skin_temperature, surface_pressure, land_fraction,
            wetness, output_host(qv_bottom), xp,
        )
        out = similarity_surface_diagnostics(
            u_lowest=u[bottom], v_lowest=v[bottom],
            temperature_lowest=temperature, qv_lowest=output_host(qv_bottom),
            p_full_lowest_pa=exchange.p_full[bottom],
            p_surface_pa=exchange.p_half[-1],
            skin_temperature_k=surface.temperature_k,
            surface_humidity=humidity, roughness_m=surface.roughness_m, xp=xp,
        )
        return {
            name: xp.ascontiguousarray(out[name])
            for name in ("t2", "th2", "q2", "u10", "v10")
        }


# physics_state.metadata key and value the reference suite stamps beside
# its screen-level fields (mirrors surface_diagnostics.SOURCE_METADATA_KEY /
# SIMILARITY_SOURCE; the literal avoids a circular import).
_SURFACE_DIAGNOSTICS_KEY = "surface_diagnostics"
_SURFACE_DIAGNOSTICS_SOURCE = "reference-similarity"


__all__ = [
    "ReferencePhysics",
    "ReferencePhysicsOptions",
    "saturation_mixing_ratio",
]
