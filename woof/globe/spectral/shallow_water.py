"""Global rotating shallow-water model in spectral vorticity/divergence form."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import EARTH_ROTATION_RATE_S, GRAVITY_M_S2
from .diffusion import ExponentialHyperdiffusion
from .state import ShallowWaterState
from .timestep import step_with_scheme
from .transform import SphericalHarmonicTransform
from .vector import VorticityDivergenceOperator


@dataclass
class ShallowWaterModel:
    transform: SphericalHarmonicTransform
    rotation_rate_s: float = EARTH_ROTATION_RATE_S
    gravity_m_s2: float = GRAVITY_M_S2
    integrator: str = "ssprk3"
    diffusion: ExponentialHyperdiffusion | None = None
    divergence_diffusion_strength: float = 1.0
    geopotential_diffusion_strength: float = 1.0
    maximum_cfl: float = 0.95

    def __post_init__(self) -> None:
        self.vector = VorticityDivergenceOperator(self.transform)
        sinlat = self.transform.backend.asarray(
            self.transform.grid.sin_lat[:, None],
            dtype=self.transform.backend.float_dtype,
        )
        self.coriolis = 2.0 * float(self.rotation_rate_s) * sinlat

    def rhs(self, state: ShallowWaterState) -> ShallowWaterState:
        xp = self.transform.backend.xp
        zeta_grid = self.transform.inverse(state.vorticity)
        phi_grid = self.transform.inverse(state.geopotential)
        u, v = self.vector.wind_from_vordiv(state.vorticity, state.divergence)
        absolute_vorticity = zeta_grid + self.coriolis

        # Vector-invariant rotational acceleration: (f+zeta) * (v, -u).
        rot_u = absolute_vorticity * v
        rot_v = -absolute_vorticity * u
        zeta_t, div_rot_t = self.vector.flux_curl_divergence(rot_u, rot_v)

        bernoulli = 0.5 * (u * u + v * v) + phi_grid
        bernoulli_spec = self.transform.forward(bernoulli)
        div_t = div_rot_t - self.transform.laplacian(bernoulli_spec)

        # Geopotential is g*h, so its continuity equation is conservative.
        _, volume_div = self.vector.flux_curl_divergence(phi_grid * u, phi_grid * v)
        phi_t = -volume_div
        return ShallowWaterState(
            self.transform.project(zeta_t),
            self.transform.project(div_t),
            self.transform.project(phi_t),
            time_s=state.time_s,
            step=state.step,
        )

    def _apply_diffusion(self, state: ShallowWaterState, dt_s: float) -> ShallowWaterState:
        if self.diffusion is None:
            return state
        z = self.diffusion.apply(state.vorticity, self.transform, dt_s)
        d = self.diffusion.apply(
            state.divergence,
            self.transform,
            dt_s,
            strength=self.divergence_diffusion_strength,
        )
        # Diffuse only the anomaly; the global-mean layer mass is exact.
        mean = state.geopotential[..., 0, 0].copy()
        p = self.diffusion.apply(
            state.geopotential,
            self.transform,
            dt_s,
            strength=self.geopotential_diffusion_strength,
        )
        p[..., 0, 0] = mean
        return state.with_fields((z, d, p))

    def cfl(self, state: ShallowWaterState, dt_s: float) -> float:
        u, v = self.vector.wind_from_vordiv(state.vorticity, state.divergence)
        phi = self.transform.inverse(state.geopotential)
        speed = self.transform.backend.to_numpy(
            self.transform.backend.xp.sqrt(u * u + v * v)
        )
        gravity_wave = math.sqrt(max(0.0, float(np.max(self.transform.backend.to_numpy(phi)))))
        characteristic = float(np.max(speed)) + gravity_wave
        return float(dt_s) * characteristic * math.sqrt(
            self.transform.truncation * (self.transform.truncation + 1.0)
        ) / self.transform.grid.radius_m

    def enforce(self, state: ShallowWaterState) -> None:
        arrays = [self.transform.backend.to_numpy(x) for x in state.fields()]
        if not all(np.isfinite(x).all() for x in arrays):
            raise FloatingPointError("shallow-water state contains non-finite values")
        phi = self.transform.backend.to_numpy(self.transform.inverse(state.geopotential))
        if float(np.min(phi)) <= 0.0:
            raise FloatingPointError(
                f"shallow-water geopotential became nonpositive: min={np.min(phi):g}"
            )

    def step(self, state: ShallowWaterState, dt_s: float) -> ShallowWaterState:
        cfl = self.cfl(state, dt_s)
        if cfl > self.maximum_cfl:
            raise ValueError(
                f"explicit shallow-water spectral CFL {cfl:.3f} exceeds {self.maximum_cfl:.3f}; "
                "reduce dt_s or truncation"
            )
        advanced = step_with_scheme(state, float(dt_s), self.rhs, self.integrator)
        advanced = self._apply_diffusion(advanced, float(dt_s))
        advanced.step = state.step + 1
        advanced.time_s = state.time_s + float(dt_s)
        self.enforce(advanced)
        return advanced

    def diagnostics(self, state: ShallowWaterState) -> dict:
        grid = self.transform.grid
        u, v = self.vector.wind_from_vordiv(state.vorticity, state.divergence)
        phi = self.transform.inverse(state.geopotential)
        zeta = self.transform.inverse(state.vorticity)
        un = self.transform.backend.to_numpy(u)
        vn = self.transform.backend.to_numpy(v)
        pn = self.transform.backend.to_numpy(phi)
        zn = self.transform.backend.to_numpy(zeta)
        speed2 = un * un + vn * vn
        mass = grid.global_mean(pn) / self.gravity_m_s2
        energy = grid.global_mean(
            0.5 * (pn * speed2 + pn * pn) / self.gravity_m_s2
        )
        potential_enstrophy = grid.global_mean(
            0.5 * (zn + 2.0 * self.rotation_rate_s * grid.sin_lat[:, None]) ** 2
            / np.maximum(pn / self.gravity_m_s2, 1.0e-12)
        )
        return {
            "time_s": float(state.time_s),
            "step": int(state.step),
            "mass_kg_m2_mean": float(mass),
            "total_energy_j_m2_mean": float(energy),
            "potential_enstrophy": float(potential_enstrophy),
            "max_wind_m_s": float(np.sqrt(speed2).max()),
            "min_geopotential_m2_s2": float(pn.min()),
            "max_geopotential_m2_s2": float(pn.max()),
        }
