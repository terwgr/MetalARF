"""Multilayer dry hydrostatic primitive equations in sigma coordinates.

This is an executable research core, not a production replacement for WOOF's
regional nonhydrostatic solver.  Horizontal dynamics are spectral; nonlinear
products and the sigma-coordinate vertical operators are evaluated on the
Gaussian grid.  Fast modes are integrated explicitly and therefore guarded by
a strict spectral CFL limit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    SECONDS_PER_DAY,
)
from .diffusion import ExponentialHyperdiffusion
from .state import PrimitiveDryState
from .timestep import step_with_scheme
from .transform import SphericalHarmonicTransform
from .vector import VorticityDivergenceOperator

# The one authority for the admitted surface-pressure band.  ``config.py``
# admits an initial ``primitive.surface_pressure_pa`` against the same
# ceiling, so a configuration the schema accepts cannot be refused by
# :meth:`PrimitiveDryModel.enforce` before step 0.  The evolving floor is
# lower than the schema's initial floor because the dynamics may deepen a
# column well below any admitted starting value.
SURFACE_PRESSURE_CEILING_PA = 120_000.0
EVOLVED_SURFACE_PRESSURE_FLOOR_PA = 20_000.0
# The state carries ln(ps), so a constant surface pressure comes back from
# synthesis as exp(log(p)) with a few ulp of round-trip error: measured
# +4.9e-16 relative at T10/T15 and +2.3e-15 at T21 in float64 for a config
# sitting exactly on the ceiling.  64*eps = 1.42e-14 clears that noise by an
# order of magnitude and is 1.7e-6 Pa at the ceiling, far below any physical
# exceedance.
_PRESSURE_ROUNDTRIP_SLACK = 64.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class SigmaCoordinate:
    half_levels: np.ndarray

    def __post_init__(self) -> None:
        half = np.asarray(self.half_levels, dtype=np.float64)
        if half.ndim != 1 or half.size < 2:
            raise ValueError("sigma half_levels must be a one-dimensional array of length >= 2")
        if not np.isfinite(half).all():
            raise ValueError("sigma half_levels contain non-finite values")
        if abs(float(half[0])) > 1.0e-14 or abs(float(half[-1]) - 1.0) > 1.0e-14:
            raise ValueError("sigma half_levels must start at 0 and end at 1")
        if np.any(np.diff(half) <= 0):
            raise ValueError("sigma half_levels must be strictly increasing")
        object.__setattr__(self, "half_levels", half)

    @property
    def nlev(self) -> int:
        return self.half_levels.size - 1

    @property
    def full_levels(self) -> np.ndarray:
        return 0.5 * (self.half_levels[:-1] + self.half_levels[1:])

    @property
    def thickness(self) -> np.ndarray:
        return np.diff(self.half_levels)

    @classmethod
    def equally_spaced(cls, nlev: int) -> "SigmaCoordinate":
        n = int(nlev)
        if n < 2:
            raise ValueError("primitive dry model needs at least two sigma layers")
        return cls(np.linspace(0.0, 1.0, n + 1, dtype=np.float64))


@dataclass(frozen=True)
class HeldSuarezForcing:
    enabled: bool = False
    equator_surface_temperature_k: float = 315.0
    meridional_contrast_k: float = 60.0
    vertical_contrast_k: float = 10.0
    minimum_temperature_k: float = 200.0
    sigma_boundary: float = 0.7
    free_atmosphere_relaxation_days: float = 40.0
    surface_relaxation_days: float = 4.0
    surface_drag_days: float = 1.0

    def tendencies(self, model: "PrimitiveDryModel", temperature, pressure, u, v):
        xp = model.transform.backend.xp
        if not self.enabled:
            return xp.zeros_like(temperature), xp.zeros_like(u), xp.zeros_like(v)
        sinlat = model._sinlat
        coslat = model._coslat
        sigma = model._sigma_full[:, None, None]
        p_ratio = xp.maximum(pressure / REFERENCE_PRESSURE_PA, 1.0e-8)
        equilibrium = (
            self.equator_surface_temperature_k
            - self.meridional_contrast_k * sinlat[None] ** 2
            - self.vertical_contrast_k * xp.log(p_ratio) * coslat[None] ** 2
        ) * p_ratio ** KAPPA
        equilibrium = xp.maximum(self.minimum_temperature_k, equilibrium)
        lower = xp.maximum(0.0, (sigma - self.sigma_boundary) / (1.0 - self.sigma_boundary))
        k_a = 1.0 / (self.free_atmosphere_relaxation_days * SECONDS_PER_DAY)
        k_s = 1.0 / (self.surface_relaxation_days * SECONDS_PER_DAY)
        k_t = k_a + (k_s - k_a) * lower * coslat[None] ** 4
        k_f = lower / (self.surface_drag_days * SECONDS_PER_DAY)
        return -k_t * (temperature - equilibrium), -k_f * u, -k_f * v


@dataclass
class PrimitiveDryModel:
    transform: SphericalHarmonicTransform
    sigma: SigmaCoordinate
    rotation_rate_s: float = EARTH_ROTATION_RATE_S
    gas_constant: float = DRY_AIR_GAS_CONSTANT
    cp: float = DRY_AIR_CP
    integrator: str = "ssprk3"
    diffusion: ExponentialHyperdiffusion | None = None
    divergence_diffusion_strength: float = 1.5
    pressure_diffusion_strength: float = 0.25
    held_suarez: HeldSuarezForcing = HeldSuarezForcing()
    surface_geopotential: object | None = None
    mass_fixer: bool = True
    maximum_cfl: float = 0.65

    def __post_init__(self) -> None:
        self.vector = VorticityDivergenceOperator(self.transform)
        b = self.transform.backend
        self._sigma_full = b.asarray(self.sigma.full_levels, dtype=b.float_dtype)
        self._sigma_half = b.asarray(self.sigma.half_levels, dtype=b.float_dtype)
        self._delta_sigma = b.asarray(self.sigma.thickness, dtype=b.float_dtype)
        self._sinlat = b.asarray(self.transform.grid.sin_lat[:, None], dtype=b.float_dtype)
        self._coslat = b.asarray(self.transform.grid.cos_lat[:, None], dtype=b.float_dtype)
        self.coriolis = 2.0 * float(self.rotation_rate_s) * self._sinlat
        if self.surface_geopotential is None:
            self.surface_geopotential = b.xp.zeros(
                self.transform.grid.shape, dtype=b.float_dtype
            )
        else:
            self.surface_geopotential = b.asarray(
                self.surface_geopotential, dtype=b.float_dtype
            )
            self.transform._validate_grid(self.surface_geopotential)
        self._target_mass_pa: float | None = None

    @property
    def nlev(self) -> int:
        return self.sigma.nlev

    def _validate_state_shape(self, state: PrimitiveDryState) -> None:
        expected = (self.nlev, *self.transform.spectral_shape)
        for name in ("vorticity", "divergence", "temperature"):
            arr = getattr(state, name)
            if tuple(arr.shape) != expected:
                raise ValueError(f"{name} shape {arr.shape} != {expected}")
        if tuple(state.log_surface_pressure.shape) != self.transform.spectral_shape:
            raise ValueError("log_surface_pressure has wrong spectral shape")

    def grid_state(self, state: PrimitiveDryState) -> dict:
        self._validate_state_shape(state)
        logps = self.transform.inverse(state.log_surface_pressure)
        ps = self.transform.backend.xp.exp(logps)
        temperature = self.transform.inverse(state.temperature)
        zeta = self.transform.inverse(state.vorticity)
        divergence = self.transform.inverse(state.divergence)
        u, v = self.vector.wind_from_vordiv(state.vorticity, state.divergence)
        pressure = self._sigma_full[:, None, None] * ps[None]
        return {
            "logps": logps,
            "ps": ps,
            "temperature": temperature,
            "vorticity": zeta,
            "divergence": divergence,
            "u": u,
            "v": v,
            "pressure": pressure,
        }

    def hydrostatic_geopotential(self, temperature):
        xp = self.transform.backend.xp
        t = xp.asarray(temperature, dtype=self.transform.backend.float_dtype)
        if t.shape[0] != self.nlev:
            raise ValueError("temperature level count does not match sigma coordinate")
        phi = xp.empty_like(t)
        lower_phi = self.surface_geopotential
        half = self.sigma.half_levels
        full = self.sigma.full_levels
        for k in range(self.nlev - 1, -1, -1):
            sigma_lower = float(half[k + 1])
            sigma_full = float(full[k])
            phi[k] = lower_phi + self.gas_constant * t[k] * math.log(
                sigma_lower / sigma_full
            )
            if k:
                sigma_upper = float(half[k])
                lower_phi = lower_phi + self.gas_constant * t[k] * math.log(
                    sigma_lower / sigma_upper
                )
        return phi

    def vertical_velocity(self, divergence, pressure_flux):
        """Return (d ln ps/dt, sigma-dot at half levels).

        Continuity is integrated top to bottom.  The bottom half-level is
        forced to exact zero; any residual there is a diagnostic of arithmetic
        closure rather than a boundary condition silently drifting.
        """
        xp = self.transform.backend.xp
        a = divergence + pressure_flux
        mean_a = xp.sum(self._delta_sigma[:, None, None] * a, axis=0)
        logps_t = -mean_a
        dot = xp.zeros(
            (self.nlev + 1, *self.transform.grid.shape),
            dtype=self.transform.backend.float_dtype,
        )
        cumulative = xp.zeros_like(mean_a)
        for k in range(self.nlev):
            cumulative = cumulative + self._delta_sigma[k] * (a[k] - mean_a)
            dot[k + 1] = -cumulative
        dot[-1] = 0.0
        return logps_t, dot

    def centered_vertical_advection(self, field, sigma_dot_half):
        xp = self.transform.backend.xp
        x = xp.asarray(field)
        out = xp.zeros_like(x)
        for k in range(self.nlev):
            if k < self.nlev - 1:
                out[k] += 0.5 * sigma_dot_half[k + 1] * (x[k + 1] - x[k])
            if k > 0:
                out[k] += 0.5 * sigma_dot_half[k] * (x[k] - x[k - 1])
            out[k] /= self._delta_sigma[k]
        return out

    def rhs(self, state: PrimitiveDryState) -> PrimitiveDryState:
        xp = self.transform.backend.xp
        g = self.grid_state(state)
        grad_ps_east, grad_ps_north = self.transform.gradient(
            state.log_surface_pressure
        )
        pressure_flux = g["u"] * grad_ps_east[None] + g["v"] * grad_ps_north[None]
        logps_t_grid, sigma_dot_half = self.vertical_velocity(
            g["divergence"], pressure_flux
        )
        sigma_dot_full = 0.5 * (sigma_dot_half[:-1] + sigma_dot_half[1:])
        material_logp_t = (
            logps_t_grid[None]
            + pressure_flux
            + sigma_dot_full / self._sigma_full[:, None, None]
        )
        geopotential = self.hydrostatic_geopotential(g["temperature"])
        w_u = self.centered_vertical_advection(g["u"], sigma_dot_half)
        w_v = self.centered_vertical_advection(g["v"], sigma_dot_half)
        w_t = self.centered_vertical_advection(g["temperature"], sigma_dot_half)

        temp_grad_east, temp_grad_north = self.transform.gradient(state.temperature)
        forcing_t, drag_u, drag_v = self.held_suarez.tendencies(
            self, g["temperature"], g["pressure"], g["u"], g["v"]
        )

        zeta_t = self.transform.zeros(self.nlev)
        div_t = self.transform.zeros(self.nlev)
        temp_t = self.transform.zeros(self.nlev)
        for k in range(self.nlev):
            absolute_vorticity = g["vorticity"][k] + self.coriolis
            momentum_u = (
                absolute_vorticity * g["v"][k]
                - w_u[k]
                - self.gas_constant * g["temperature"][k] * grad_ps_east
                + drag_u[k]
            )
            momentum_v = (
                -absolute_vorticity * g["u"][k]
                - w_v[k]
                - self.gas_constant * g["temperature"][k] * grad_ps_north
                + drag_v[k]
            )
            z_t, d_t = self.vector.flux_curl_divergence(momentum_u, momentum_v)
            bernoulli = (
                0.5 * (g["u"][k] ** 2 + g["v"][k] ** 2) + geopotential[k]
            )
            d_t = d_t - self.transform.laplacian(self.transform.forward(bernoulli))
            horizontal_advection = -(
                g["u"][k] * temp_grad_east[k]
                + g["v"][k] * temp_grad_north[k]
            )
            adiabatic = (self.gas_constant / self.cp) * g["temperature"][k] * material_logp_t[k]
            t_grid = horizontal_advection - w_t[k] + adiabatic + forcing_t[k]
            zeta_t[k] = z_t
            div_t[k] = d_t
            temp_t[k] = self.transform.forward(t_grid)

        return PrimitiveDryState(
            self.transform.project(zeta_t),
            self.transform.project(div_t),
            self.transform.project(temp_t),
            self.transform.project(self.transform.forward(logps_t_grid)),
            time_s=state.time_s,
            step=state.step,
        )

    def _apply_diffusion(self, state: PrimitiveDryState, dt_s: float) -> PrimitiveDryState:
        if self.diffusion is None:
            return state
        z = self.diffusion.apply(state.vorticity, self.transform, dt_s)
        d = self.diffusion.apply(
            state.divergence,
            self.transform,
            dt_s,
            strength=self.divergence_diffusion_strength,
        )
        t = self.diffusion.apply(state.temperature, self.transform, dt_s)
        p = self.diffusion.apply(
            state.log_surface_pressure,
            self.transform,
            dt_s,
            strength=self.pressure_diffusion_strength,
        )
        return state.with_fields((z, d, t, p))

    def initialize_mass_target(self, state: PrimitiveDryState) -> float:
        ps = self.transform.backend.to_numpy(
            self.transform.backend.xp.exp(
                self.transform.inverse(state.log_surface_pressure)
            )
        )
        self._target_mass_pa = self.transform.grid.global_mean(ps)
        return self._target_mass_pa

    def _fix_mass(self, state: PrimitiveDryState) -> tuple[PrimitiveDryState, float]:
        if not self.mass_fixer:
            return state, 0.0
        if self._target_mass_pa is None:
            self.initialize_mass_target(state)
        logps_grid = self.transform.inverse(state.log_surface_pressure)
        ps = self.transform.backend.to_numpy(
            self.transform.backend.xp.exp(logps_grid)
        )
        current = self.transform.grid.global_mean(ps)
        if current <= 0 or not math.isfinite(current):
            raise FloatingPointError(f"invalid global mean surface pressure {current!r}")
        correction = math.log(float(self._target_mass_pa) / current)
        fixed = self.transform.add_grid_constant(state.log_surface_pressure, correction)
        return state.with_fields(
            (state.vorticity, state.divergence, state.temperature, fixed)
        ), correction

    def cfl(self, state: PrimitiveDryState, dt_s: float) -> float:
        g = self.grid_state(state)
        speed = self.transform.backend.to_numpy(
            self.transform.backend.xp.sqrt(g["u"] ** 2 + g["v"] ** 2)
        )
        tmax = float(np.max(self.transform.backend.to_numpy(g["temperature"])))
        psmax = float(np.max(self.transform.backend.to_numpy(g["ps"])))
        ptop = max(1.0, float(self.sigma.full_levels[0]) * psmax)
        external = math.sqrt(max(1.0, self.gas_constant * tmax * math.log(psmax / ptop)))
        characteristic = float(speed.max()) + external
        return float(dt_s) * characteristic * math.sqrt(
            self.transform.truncation * (self.transform.truncation + 1.0)
        ) / self.transform.grid.radius_m

    def enforce(self, state: PrimitiveDryState) -> None:
        arrays = [self.transform.backend.to_numpy(x) for x in state.fields()]
        if not all(np.isfinite(x).all() for x in arrays):
            raise FloatingPointError("primitive spectral state contains non-finite values")
        g = self.grid_state(state)
        temp = self.transform.backend.to_numpy(g["temperature"])
        ps = self.transform.backend.to_numpy(g["ps"])
        if float(temp.min()) < 120.0 or float(temp.max()) > 450.0:
            raise FloatingPointError(
                f"temperature outside research bounds: {temp.min():g}..{temp.max():g} K"
            )
        floor = EVOLVED_SURFACE_PRESSURE_FLOOR_PA * (1.0 - _PRESSURE_ROUNDTRIP_SLACK)
        ceiling = SURFACE_PRESSURE_CEILING_PA * (1.0 + _PRESSURE_ROUNDTRIP_SLACK)
        if float(ps.min()) < floor or float(ps.max()) > ceiling:
            raise FloatingPointError(
                f"surface pressure outside research bounds: {ps.min():g}..{ps.max():g} Pa "
                f"(admitted {EVOLVED_SURFACE_PRESSURE_FLOOR_PA:g}.."
                f"{SURFACE_PRESSURE_CEILING_PA:g} Pa); the explicit integration has "
                "left the terrestrial column this sigma discretization, its CFL "
                "estimate and its hydrostatic integral were written for, and every "
                "later diagnostic and gate value would be computed from a diverged "
                "state"
            )

    def step(self, state: PrimitiveDryState, dt_s: float) -> tuple[PrimitiveDryState, dict]:
        if self._target_mass_pa is None:
            self.initialize_mass_target(state)
        cfl = self.cfl(state, dt_s)
        if cfl > self.maximum_cfl:
            raise ValueError(
                f"explicit primitive-equation spectral CFL {cfl:.3f} exceeds "
                f"{self.maximum_cfl:.3f}; reduce dt_s or truncation"
            )
        advanced = step_with_scheme(state, float(dt_s), self.rhs, self.integrator)
        advanced = self._apply_diffusion(advanced, float(dt_s))
        advanced, mass_log_correction = self._fix_mass(advanced)
        advanced.step = state.step + 1
        advanced.time_s = state.time_s + float(dt_s)
        self.enforce(advanced)
        return advanced, {
            "spectral_cfl": float(cfl),
            "mass_fixer_log_offset": float(mass_log_correction),
        }

    def diagnostics(self, state: PrimitiveDryState) -> dict:
        g = self.grid_state(state)
        grid = self.transform.grid
        u = self.transform.backend.to_numpy(g["u"])
        v = self.transform.backend.to_numpy(g["v"])
        t = self.transform.backend.to_numpy(g["temperature"])
        ps = self.transform.backend.to_numpy(g["ps"])
        speed2 = u * u + v * v
        ds = self.sigma.thickness[:, None, None]
        mass = grid.global_mean(ps)
        kinetic = grid.global_mean(np.sum(ds * 0.5 * speed2, axis=0))
        internal = grid.global_mean(np.sum(ds * self.cp * t, axis=0))
        return {
            "time_s": float(state.time_s),
            "step": int(state.step),
            "global_mean_surface_pressure_pa": float(mass),
            "column_mean_kinetic_energy_j_kg": float(kinetic),
            "column_mean_internal_energy_j_kg": float(internal),
            "max_wind_m_s": float(np.sqrt(speed2).max()),
            "min_temperature_k": float(t.min()),
            "max_temperature_k": float(t.max()),
            "min_surface_pressure_pa": float(ps.min()),
            "max_surface_pressure_pa": float(ps.max()),
        }
