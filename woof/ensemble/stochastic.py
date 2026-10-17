"""THIRD-PARTY NOTICE

The stochastic equations are transcribed from WRF v4.6.1
``dyn_em/module_stoch.F``. WRF's public-domain notice is reproduced below.
Philox4x32-10 follows Random123; its BSD notice is in
``licenses/LICENSE-Random123.txt`` and the root NOTICE.

WRF was developed at the National Center for Atmospheric Research (NCAR)
which is operated by the University Corporation for Atmospheric Research
(UCAR). NCAR and UCAR make no proprietary claims, either statutory or
otherwise, to this version and release of WRF and consider WRF to be in the
public domain for use by any person or entity for any purpose without any
fee or charge. UCAR requests that any WRF user include this notice on any
partial or full copies of WRF. WRF is provided on an "AS IS" basis and any
warranties, either express or implied, including but not limited to implied
warranties of non-infringement, originality, merchantability and fitness for
a particular purpose, are disclaimed. In no event shall UCAR be liable for
any damages, whatsoever, whether direct, indirect, consequential or special,
that arise out of or in connection with the access, use or performance of
WRF, including infringement actions.

GPU pattern and tendency primitives, not an integrated physics option.
The port covers WRF's vertically uniform spectral AR(1) patterns, SKEBS
rotational wind and temperature forcing, and SPPT multiplication of supplied
nonmicrophysics tendencies. SPP returns its three independent patterns;
physics-specific parameter consumers receive explicitly shaped patterns.

Explicit differences from the Fortran implementation:
* Counter-based Philox replaces compiler-dependent RANDOM_NUMBER streams.
* cuFFT replaces FFTPACK. No WRF bitwise-equivalence claim is made.
* Spectrum normalization uses binary64 and a shifted exponential to avoid
  zero divided by zero when correlation lengths exceed the domain size.
* Only vertical structure option 0 is supported. No vertical phase pattern
  is silently substituted.

WRF namelist defaults here are reference parameters, not calibrated defaults.
No source adapter, timestep or executor calls these primitives implicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

WRF_VERSION = "v4.6.1"
WRF_SOURCE = "https://github.com/wrf-model/WRF/blob/v4.6.1/dyn_em/module_stoch.F"
WRF_SOURCE_SHA256 = "e28c912fa69aebdfd9d27fef5e75bd2ab84018767f5a0e7c642aeeae31190ae2"
RNG_VERSION = "gpuwm-stochastic-philox4x32-10.v1"
STATE_VERSION = "gpuwm-wrf-stochastic-pattern.v1"
_UINT32_MAX = (1 << 32) - 1
_STREAMS = {
    "sppt": 1, "skebs_psi": 2, "skebs_theta": 3,
    "spp_conv": 4, "spp_pbl": 5, "spp_lsm": 6,
}


def _uint(value: int, bits: int, name: str) -> int:
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


def philox4x32_10(counter: tuple[int, int, int, int],
                  key: tuple[int, int]) -> tuple[int, int, int, int]:
    """Scalar integer oracle for the GPU RNG, not an atmospheric data path.

    The counter is (twice the global spectral index plus component,
    absolute update step, physical stream, rejection attempt). The key is
    the low and high word of the member seed. No batch position appears.
    """
    if len(counter) != 4 or len(key) != 2:
        raise ValueError("Philox requires four counter words and two key words")
    c0, c1, c2, c3 = (_uint(v, 32, "counter word") for v in counter)
    k0, k1 = (_uint(v, 32, "key word") for v in key)
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & _UINT32_MAX,
            p1 & _UINT32_MAX,
            ((p0 >> 32) ^ c3 ^ k1) & _UINT32_MAX,
            p0 & _UINT32_MAX,
        )
        k0 = (k0 + 0x9E3779B9) & _UINT32_MAX
        k1 = (k1 + 0xBB67AE85) & _UINT32_MAX
    return c0, c1, c2, c3


@dataclass(frozen=True)
class StochasticConfig:
    """Reference parameters from WRF Registry/registry.stoch:170-246.

    ``backscatter`` is m2 s-3, ``timescale_s`` seconds, ``lengthscale_m``
    metres. ``stddev`` and ``cutoff_sigma`` apply to gridpoint patterns.
    WRF SKEBS also clips using the SPPT threshold, 2 * 0.5 by default.
    """

    kind: str = "sppt"
    stddev: float = 0.5
    cutoff_sigma: float = 2.0
    lengthscale_m: float = 150000.0
    timescale_s: float = 21600.0
    backscatter: float = 0.0
    spectral_exponent: float = -1.83
    min_wavenumber: int = 1
    max_wavenumber: int = 1000000
    vertical_structure: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _STREAMS:
            raise ValueError(f"unknown stochastic scheme {self.kind!r}")
        for name in ("stddev", "cutoff_sigma", "lengthscale_m", "timescale_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.backscatter) or self.backscatter < 0:
            raise ValueError("backscatter must be finite and non-negative")
        if not math.isfinite(self.spectral_exponent):
            raise ValueError("spectral_exponent must be finite")
        if (type(self.min_wavenumber) is not int or self.min_wavenumber < 1
                or type(self.max_wavenumber) is not int
                or self.max_wavenumber < self.min_wavenumber):
            raise ValueError("wavenumber limits must be positive ordered integers")
        if type(self.vertical_structure) is not int or self.vertical_structure != 0:
            raise ValueError(
                "vertical_structure must be 0: vertical phase rotation is not "
                "implemented and substituting a uniform field changes forcing")
        if self.kind == "sppt" and self.stddev * self.cutoff_sigma > 1.0:
            raise ValueError("SPPT cutoff above 1 can reverse physical tendencies")

    @classmethod
    def wrf_reference(cls, kind: str) -> StochasticConfig:
        """WRF reference settings; no observational calibration is implied."""
        rows = {
            "sppt": {},
            "skebs_psi": {"timescale_s": 10800.0, "backscatter": 1.0e-5},
            "skebs_theta": {"timescale_s": 10800.0, "backscatter": 1.0e-6},
            "spp_conv": {"stddev": 0.3, "cutoff_sigma": 3.0},
            "spp_pbl": {"stddev": 0.15, "lengthscale_m": 700000.0},
            "spp_lsm": {"stddev": 0.3, "cutoff_sigma": 3.0,
                        "lengthscale_m": 50000.0, "timescale_s": 86400.0},
        }
        if kind not in rows:
            raise ValueError(f"unknown stochastic scheme {kind!r}")
        return cls(kind=kind, **rows[kind])


def _kernel(name: str):
    from woof.core.kernels import load_module
    return load_module("ensemble_stochastic").get_function(name)


def _launch(name: str, size: int, args: tuple, *, block_size: int = 128) -> None:
    _kernel(name)(((size + block_size - 1) // block_size,), (block_size,), args)


# Host transcriptions of ensemble_stochastic.cu for states without CUDA (the
# Metal route keeps its patterns in host memory, which it shares with the
# GPU).  Same counters, conjugacy and unfused FP32 AR(1) arithmetic; libm and
# the FFT may differ from CUDA's by rounding, as cuFFT already does from WRF.

def _host_philox(c0, c1, c2, c3, seed: int):
    mask = np.uint64(_UINT32_MAX)
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) for c in (c0, c1, c2, c3))
    k0, k1 = seed & _UINT32_MAX, seed >> 32
    for _ in range(10):
        p0 = np.uint64(0xD2511F53) * c0
        p1 = np.uint64(0xCD9E8D57) * c2
        c0, c1, c2, c3 = ((p1 >> np.uint64(32)) ^ c1 ^ np.uint64(k0),
                          p1 & mask,
                          (p0 >> np.uint64(32)) ^ c3 ^ np.uint64(k1),
                          p0 & mask)
        k0 = (k0 + 0x9E3779B9) & _UINT32_MAX
        k1 = (k1 + 0xBB67AE85) & _UINT32_MAX
    return c0, c1, c2, c3


def _host_gaussian(index, step: int, stream: int, seed: int) -> np.ndarray:
    """WRF gauss_noise with |z| < 3 rejection, one counter per attempt."""
    index = np.asarray(index, dtype=np.uint64)
    out = np.empty(index.shape, dtype=np.float32)
    pending = np.arange(index.size)
    scale = np.float32(2.0 ** -24)
    one, two = np.float32(1.0), np.float32(2.0)
    attempt = 0
    while pending.size:
        q = index.ravel()[pending]
        r = _host_philox(q, np.full_like(q, step), np.full_like(q, stream),
                         np.full_like(q, attempt), seed)
        u = (r[0] >> np.uint64(8)).astype(np.float32) * scale
        v = (r[1] >> np.uint64(8)).astype(np.float32) * scale
        x = two * u - one
        y = two * v - one
        radius = x * x + y * y
        inside = (radius > 0) & (radius < 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.sqrt((np.float32(-2.0) * np.log(radius)) / radius) * x
        accept = inside & (np.abs(z) < np.float32(3.0))
        out.ravel()[pending[accept]] = z[accept]
        pending = pending[~accept]
        attempt += 1
    return out


def _host_spectrum_weights(nx, ny, dx, dy, kind, length, exponent, minimum, maximum):
    y, x = np.mgrid[0:ny, 0:nx]
    rx, ry = nx * dx, ny * dy
    kx, ky = x / rx, y / ry
    rho2 = kx * kx + ky * ky
    rho = np.sqrt(rho2)
    band = (((rho < (maximum + 0.5) / rx) & (rho >= (minimum - 0.5) / rx))
            | ((rho < (maximum + 0.5) / ry) & (rho >= (minimum - 0.5) / ry)))
    off = ~band | ((x == 0) & (y == 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        if kind == 0:
            log_weights = -2.0 * math.pi * math.pi * length * length * rho2
            gamma = log_weights.copy()
        else:
            log_weights = 0.5 * exponent * np.log(rho2)
            gamma = rho2 ** (exponent + (1.0 if kind == 1 else 0.0))
    log_weights[off] = -np.inf
    gamma[off] = -np.inf if kind == 0 else 0.0
    return log_weights, gamma


def _host_folded(nx: int, ny: int):
    y, x = np.mgrid[0:ny, 0:nx]
    return (np.where(y <= ny // 2, y, ny - y), np.where(x <= nx // 2, x, nx - x))


def _host_noise_index(nx: int, ny: int):
    y, x = np.mgrid[0:ny, 0:nx]
    mirror = ((y == 0) & (x > nx // 2)) | (y > ny // 2)
    qx = np.where(mirror, np.where(x == 0, 0, nx - x), x)
    qy = np.where(mirror, np.where(y == 0, 0, ny - y), y)
    sign = np.where(mirror, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
    return (qy * nx + qx).astype(np.uint64), sign


def _host_derivative(spectrum, nx, ny, dx, dy, component):
    y, x = np.mgrid[0:ny, 0:nx]
    if component == 1:
        k = np.where(y <= ny // 2, y, y - ny)
        size = np.float32(ny) * np.float32(dy)
    else:
        k = np.where(x <= nx // 2, x, x - nx)
        size = np.float32(nx) * np.float32(dx)
    factor = (np.float32(6.2831853071795864769) / size) * k.astype(np.float32)
    re, im = spectrum.real, spectrum.imag
    out = np.empty_like(spectrum)
    if component == 1:
        out.real, out.imag = factor * im, -(factor * re)
    else:
        out.real, out.imag = -(factor * im), factor * re
    return out


class WrfStochasticPattern:
    """One member's spectral process, kept entirely on its current CUDA device.

    ``shape_yx`` includes the complete stochastic domain. ``advance(0)`` is
    the first draw from WRF's zero initial spectral state. Subsequent steps
    must be consecutive, so a skipped call cannot silently change its AR(1)
    history. Each member owns its object even when other members are batched.
    """

    def __init__(self, config: StochasticConfig, shape_yx: tuple[int, int], *,
                 dx: float, dy: float, dt: float, member_seed: int,
                 stream_id: int | None = None, host: bool = False):
        if (len(shape_yx) != 2 or any(type(n) is not int or n < 12 for n in shape_yx)
                or math.prod(shape_yx) >= (1 << 31)):
            raise ValueError(
                "stochastic domain needs at least 12 points per direction and "
                "fewer than 2**31 cells to retain WRF's spectral band and counters")
        for name, value in (("dx", dx), ("dy", dy), ("dt", dt)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        _uint(member_seed, 64, "member_seed")
        self.config = config
        self.shape = tuple(shape_yx)
        self.dx, self.dy, self.dt = float(dx), float(dy), float(dt)
        self.seed = member_seed
        self.stream = _uint(_STREAMS[config.kind] if stream_id is None else stream_id,
                            32, "stream_id")
        self.completed_step = -1
        skebs = config.kind.startswith("skebs_")
        if skebs and dt > config.timescale_s:
            raise ValueError("SKEBS dt exceeds decorrelation time, making WRF AR(1) unstable")
        self.alpha = (dt / config.timescale_s if skebs else
                      -math.expm1(-dt / config.timescale_s))
        self.host = bool(host)
        if self.host:
            self.device = None
            self.spectrum = np.zeros(self.shape, dtype=np.complex64)
        else:
            import cupy as cp
            self.device = cp.cuda.Device().id
            self.spectrum = cp.zeros(self.shape, dtype=cp.complex64)
        self.amplitude = self._amplitude()

    def _amplitude(self):
        if self.host:
            cp = np
        else:
            import cupy as cp
        cfg = self.config
        ny, nx = self.shape
        n = ny * nx
        mode_max = min(min(nx // 2, ny // 2) - 5, cfg.max_wavenumber)
        if mode_max < cfg.min_wavenumber:
            raise ValueError("WRF spectral limits contain no forced wavenumbers")
        kind = {"skebs_psi": 1, "skebs_theta": 2}.get(cfg.kind, 0)
        if self.host:
            log_weights, gamma_terms = _host_spectrum_weights(
                nx, ny, self.dx, self.dy, kind, cfg.lengthscale_m,
                cfg.spectral_exponent, cfg.min_wavenumber, mode_max)
        else:
            log_weights = cp.empty(self.shape, dtype=cp.float64)
            gamma_terms = cp.empty_like(log_weights)
            _launch("stoch_spectrum_weights", n, (
                log_weights, gamma_terms, np.int32(nx), np.int32(ny),
                np.float64(self.dx), np.float64(self.dy), np.int32(kind),
                np.float64(cfg.lengthscale_m), np.float64(cfg.spectral_exponent),
                np.int32(cfg.min_wavenumber), np.int32(mode_max)))
        shift = float(cp.max(log_weights).item()) if kind == 0 else 0.0
        if not math.isfinite(shift):
            raise ValueError("WRF spectral limits contain no finite forcing amplitudes")
        if kind == 0:
            # A common log shift cancels algebraically between ZCHI and
            # sqrt(ZGAMMAN), preserving the WRF spectrum on small domains.
            gamma = float(cp.sum(cp.exp(2.0 * (gamma_terms - shift)),
                                 dtype=cp.float64).item()) * 4.0
            if not math.isfinite(gamma) or gamma <= 0:
                raise ValueError("WRF spectral normalization contains no finite forced modes")
            f0 = cfg.stddev * math.sqrt(
                (-math.expm1(-2.0 * self.dt / cfg.timescale_s)) / (2.0 * gamma))
        else:
            gamma = float(cp.sum(gamma_terms, dtype=cp.float64).item()) * 4.0
            if not math.isfinite(gamma) or gamma <= 0:
                raise ValueError("WRF spectral normalization contains no finite forced modes")
            sigma2 = 1.0 / (12.0 * self.alpha)
            energy = self.alpha * cfg.backscatter / (self.dt * sigma2 * gamma)
            # module_model_constants.F: T0=300, cp=7*287/2.
            f0 = (math.sqrt(energy) / (2.0 * math.pi) if kind == 1 else
                  math.sqrt(300.0 * energy / 1004.5))
        if not math.isfinite(gamma) or gamma <= 0 or not math.isfinite(f0):
            raise ValueError("spectral normalization is not finite and positive")
        if self.host:
            qy, qx = _host_folded(nx, ny)
            return (f0 * np.exp(log_weights[qy, qx] - shift)).astype(np.float32)
        amplitude = cp.empty(self.shape, dtype=cp.float32)
        _launch("stoch_spectrum_amplitude", n, (
            amplitude, log_weights, np.int32(nx), np.int32(ny),
            np.float64(shift), np.float64(f0)))
        return amplitude

    def metadata(self) -> dict:
        return {"version": STATE_VERSION, "rng_version": RNG_VERSION,
                "wrf_version": WRF_VERSION, "wrf_source_sha256": WRF_SOURCE_SHA256,
                "config": asdict(self.config), "shape": list(self.shape),
                "dx": self.dx, "dy": self.dy, "dt": self.dt,
                "member_seed": self.seed, "stream_id": self.stream}

    def set_time_step(self, dt: float) -> None:
        """Use the accepted model step without resetting the spectral history."""
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("stochastic dt must be finite and positive")
        skebs = self.config.kind.startswith("skebs_")
        if skebs and dt > self.config.timescale_s:
            raise ValueError("SKEBS dt exceeds decorrelation time, making WRF AR(1) unstable")
        if dt == self.dt:
            return
        self._check_device()
        self.dt = dt
        self.alpha = (dt / self.config.timescale_s if skebs else
                      -math.expm1(-dt / self.config.timescale_s))
        self.amplitude = self._amplitude()

    def _check_device(self) -> None:
        if self.host:
            return
        import cupy as cp
        if cp.cuda.Device().id != self.device:
            raise ValueError("stochastic state belongs to a different CUDA device")

    def advance(self, step: int, *, block_size: int = 128):
        """Advance once; return a capped float32 pattern, constant in height."""
        self._check_device()
        _uint(step, 32, "step")
        if step != self.completed_step + 1:
            raise ValueError("stochastic steps must be consecutive; restore the matching state")
        if type(block_size) is not int or not 1 <= block_size <= 1024:
            raise ValueError("block_size must be an integer from 1 through 1024")
        ny, nx = self.shape
        if self.host:
            q, sign = _host_noise_index(nx, ny)
            cosine = _host_gaussian(2 * q, step, self.stream, self.seed)
            sine = _host_gaussian(2 * q + np.uint64(1), step, self.stream, self.seed)
            phi = np.float32(1.0) - np.float32(self.alpha)
            old = self.spectrum
            updated = np.empty_like(old)
            updated.real = phi * old.real + self.amplitude * cosine
            updated.imag = phi * old.imag + self.amplitude * (sign * sine)
            self.spectrum = updated
        else:
            _launch("stoch_update", ny * nx, (
                self.spectrum, self.amplitude, np.int32(nx), np.int32(ny),
                np.float32(self.alpha), np.uint64(self.seed), np.uint32(step),
                np.uint32(self.stream)), block_size=block_size)
        self.completed_step = step
        return self.gridpoint()

    def gridpoint(self, component: str = "scalar"):
        """WRF RAND_PERT_UPDATE:1200-1345, using cuFFT's unscaled inverse."""
        self._check_device()
        if component not in ("scalar", "u", "v"):
            raise ValueError("component must be scalar, u or v")
        if component != "scalar" and self.config.kind != "skebs_psi":
            raise ValueError("rotational derivatives require the SKEBS streamfunction")
        cutoff = np.float32(self.config.stddev * self.config.cutoff_sigma)
        if self.host:
            spectral = self.spectrum
            if component != "scalar":
                ny, nx = self.shape
                spectral = _host_derivative(spectral, nx, ny, self.dx, self.dy,
                                            1 if component == "u" else 2)
            field = np.fft.ifft2(spectral, norm="forward").real.astype(np.float32)
            return np.clip(field, -cutoff, cutoff, out=field)
        import cupy as cp
        spectral = self.spectrum
        if component != "scalar":
            spectral = cp.empty_like(self.spectrum)
            ny, nx = self.shape
            _launch("stoch_derivative", ny * nx, (
                spectral, self.spectrum, np.int32(nx), np.int32(ny),
                np.float32(self.dx), np.float32(self.dy),
                np.int32(1 if component == "u" else 2)))
        # Independent 2D FFT calls keep transform plans independent of N.
        field = cp.fft.ifft2(spectral, norm="forward").real.copy()
        return cp.clip(field, -cutoff, cutoff, out=field)

    def snapshot(self) -> dict:
        """GPU-resident restart payload; caller owns persistence and transfer."""
        self._check_device()
        return {"metadata": self.metadata(), "completed_step": self.completed_step,
                "spectrum": self.spectrum.copy()}

    def _validate_state(self, state: Mapping) -> None:
        self._check_device()
        if self.host:
            cp = np
        else:
            import cupy as cp
        metadata = state.get("metadata")
        expected = self.metadata()
        if not isinstance(metadata, Mapping):
            raise ValueError("stochastic restart metadata is missing")
        dt = metadata.get("dt")
        if (isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not math.isfinite(dt) or dt <= 0
                or (self.config.kind.startswith("skebs_") and dt > self.config.timescale_s)):
            raise ValueError("stochastic restart has an invalid accepted time step")
        expected["dt"] = dt
        if metadata != expected:
            raise ValueError("stochastic restart metadata does not match this member and grid")
        step = state.get("completed_step")
        if type(step) is not int or not -1 <= step <= _UINT32_MAX:
            raise ValueError("stochastic restart has an invalid completed_step")
        spectrum = state.get("spectrum")
        if (not isinstance(spectrum, cp.ndarray) or spectrum.shape != self.shape
                or spectrum.dtype != cp.complex64
                or (not self.host and spectrum.device.id != self.device)):
            raise ValueError("stochastic restart spectrum has wrong shape, dtype or CUDA device")
        if not bool(cp.all(cp.isfinite(spectrum)).item()):
            raise ValueError("stochastic restart spectrum contains nonfinite coefficients")
    def restore(self, state: Mapping) -> None:
        """Reject a restart that would splice another seed/grid into this AR(1)."""
        self._validate_state(state)
        self.set_time_step(state["metadata"]["dt"])
        self.spectrum[...] = state["spectrum"]
        self.completed_step = state["completed_step"]


class WrfSkebs:
    """SKEBS U/V share one streamfunction; theta has an independent stream."""

    def __init__(self, shape_yx: tuple[int, int], *, dx: float, dy: float,
                 dt: float, member_seed: int,
                 psi_config: StochasticConfig | None = None,
                 theta_config: StochasticConfig | None = None,
                 wrf_seed_labels: Mapping | None = None, host: bool = False):
        psi_config = psi_config or StochasticConfig.wrf_reference("skebs_psi")
        theta_config = theta_config or StochasticConfig.wrf_reference("skebs_theta")
        if psi_config.kind != "skebs_psi" or theta_config.kind != "skebs_theta":
            raise ValueError("SKEBS requires streamfunction and temperature configurations")
        kwargs = dict(dx=dx, dy=dy, dt=dt, member_seed=member_seed,
                      **({"host": True} if host else {}))
        from woof.ensemble.stochastic_seeds import process_seed
        self.psi = WrfStochasticPattern(psi_config, shape_yx, **dict(kwargs,
            member_seed=process_seed(member_seed, "skebs_psi", wrf_seed_labels)))
        self.theta = WrfStochasticPattern(theta_config, shape_yx, **dict(kwargs,
            member_seed=process_seed(member_seed, "skebs_theta", wrf_seed_labels)))

    def advance(self, step: int) -> dict:
        self.psi.advance(step)
        theta = self.theta.advance(step)
        return {"u": self.psi.gridpoint("u"), "v": self.psi.gridpoint("v"),
                "theta": theta}

    def snapshot(self) -> dict:
        return {"psi": self.psi.snapshot(), "theta": self.theta.snapshot()}

    def restore(self, state: Mapping) -> None:
        if set(state) != {"psi", "theta"}:
            raise ValueError("SKEBS restart must contain both streamfunction and temperature")
        if state["psi"].get("completed_step") != state["theta"].get("completed_step"):
            raise ValueError("SKEBS restart components have different update histories")
        self.psi._validate_state(state["psi"])
        self.theta._validate_state(state["theta"])
        self.psi.restore(state["psi"])
        self.theta.restore(state["theta"])


def _module_of(array):
    """``(module, device)`` owning ``array``: NumPy (device None) or CuPy."""
    if isinstance(array, np.ndarray):
        return np, None
    import cupy as cp
    return cp, getattr(getattr(array, "device", None), "id", None)


def apply_nonmicrophysics_sppt(pattern, tendencies: Mapping, *,
                              tendency_scope: str) -> dict:
    """WRF perturb_physics_tend:1013-1087 on active tendency views.

    Inputs are physical tendencies, before microphysics, for keys u, v,
    theta and qv. Horizontal shape can include staggered edges and must fit
    the full pattern. All vertical levels use the same factor. Return new
    arrays, leaving callers' deterministic and microphysics terms intact.
    """
    if tendency_scope != "nonmicrophysics":
        raise ValueError("SPPT requires nonmicrophysics tendencies to avoid perturbing the microphysics increment")
    cp, device = _module_of(pattern)
    if (not isinstance(pattern, cp.ndarray) or pattern.ndim != 2
            or pattern.dtype != cp.float32):
        raise ValueError("SPPT pattern must be a 2D float32 CUDA array")
    if set(tendencies) != {"u", "v", "theta", "qv"}:
        raise ValueError("SPPT requires the same pattern for u, v, theta and qv")
    if not bool(cp.all(cp.isfinite(pattern) & (cp.abs(pattern) <= 1.0)).item()):
        raise ValueError("SPPT pattern must be finite and bounded by 1 to preserve tendency sign")
    result = {}
    for name, values in tendencies.items():
        if (not isinstance(values, cp.ndarray) or values.dtype != cp.float32
                or values.ndim != 3 or _module_of(values)[1] != device
                or any(v > p for v, p in zip(values.shape[-2:], pattern.shape))):
            raise ValueError(f"{name} tendency must be a float32 CUDA volume within the pattern grid")
        ny, nx = values.shape[-2:]
        result[name] = values * (np.float32(1.0) + pattern[:ny, :nx])
    return result


def apply_nonmicrophysics_sppt_components(patterns: Mapping, tendencies: Mapping, *,
                                         tendency_scope: str) -> dict:
    """Apply windowed samples of one full-domain pattern at each stagger."""
    if tendency_scope != "nonmicrophysics" or set(patterns) != {"u", "v", "theta", "qv"} or set(tendencies) != set(patterns):
        raise ValueError("windowed SPPT needs every original nonmicrophysics component")
    import cupy as cp
    result = {}
    for name, values in tendencies.items():
        pattern = patterns[name]
        if (not isinstance(pattern, cp.ndarray) or pattern.ndim != 2 or pattern.dtype != cp.float32
                or not isinstance(values, cp.ndarray) or values.dtype != cp.float32
                or values.ndim != 3 or values.device.id != pattern.device.id
                or values.shape[-2:] != pattern.shape):
            raise ValueError("windowed SPPT component does not match its actual tendency shape/device")
        if not bool(cp.all(cp.isfinite(pattern) & (cp.abs(pattern) <= 1.0)).item()):
            raise ValueError("windowed SPPT pattern must retain the full-domain finite bound")
        result[name] = values * (np.float32(1.0) + pattern)
    return result


def spp_pattern_generators(shape_yx: tuple[int, int], *, dx: float, dy: float,
                           dt: float, member_seed: int) -> dict:
    """Independent WRF SPP patterns; this does not enable parameter consumers.

    Full SPP requires GF, MYNN and RUC parameter-level wiring. Applying a
    pattern to an arbitrary final tendency would be SPPT, not WRF SPP.
    """
    return {name: WrfStochasticPattern(StochasticConfig.wrf_reference("spp_" + name),
                                      shape_yx, dx=dx, dy=dy, dt=dt,
                                      member_seed=member_seed)
            for name in ("conv", "pbl", "lsm")}


class StochasticTimestepHook:
    """Explicit seam around the first-RK nonmicrophysics tendency evaluation.

    A caller invokes ``before_timestep(step)`` once, then
    ``after_nonmicrophysics(tendencies)`` once, holding the returned rates
    through all RK stages. Never call after microphysics or once per RK
    substep. WRF first_rk_step_part2.F:822-857 adds SKEBS before SPPT.

    This adapter does not monkeypatch a driver or register itself with an
    executor. It is inert when disabled, including no device allocation.
    The host must include the spectra in its checkpoint and device inventory.
    The full WRF pattern grid is (mass_ny + 1, mass_nx + 1); its two outer
    edges serve the staggered momentum tendencies.
    """

    def __init__(self, shape_yx: tuple[int, int], *, dx: float, dy: float,
                 dt: float, member_seed: int, enabled: bool = True,
                 sppt: StochasticConfig | None = None,
                 skebs_psi: StochasticConfig | None = None,
                 skebs_theta: StochasticConfig | None = None,
                 spp: bool = False, spp_levels: Mapping[str, int] | None = None,
                 spp_configs: Mapping[str, StochasticConfig] | None = None,
                 wrf_seed_labels: Mapping | None = None, host: bool = False):
        self.host = bool(host)
        self.wrf_seed_labels = None
        self.spp_levels = dict(spp_levels or {}) if enabled and spp else {}
        self.spp_configs = {}
        if enabled and spp:
            if (not self.spp_levels or set(self.spp_levels) - {"conv", "pbl", "lsm"}
                    or any(type(value) is not int or value < 1 for value in self.spp_levels.values())
                    or ("conv" in self.spp_levels and self.spp_levels["conv"] != 4)):
                raise ValueError(
                    "SPP requires positive spp_levels for selected conv/pbl/lsm consumers; "
                    "GF conv requires four closure channels to avoid dropping parameter families")
            if spp_configs is None:
                self.spp_configs = {name: StochasticConfig.wrf_reference("spp_"+name)
                                    for name in self.spp_levels}
            else:
                if not isinstance(spp_configs, Mapping) or set(spp_configs) != set(self.spp_levels):
                    raise ValueError("spp_configs must exactly match spp_levels; keys and StochasticConfig kinds must identify every enabled consumer")
                for name, config in spp_configs.items():
                    if not isinstance(config, StochasticConfig) or config.kind != "spp_" + name:
                        raise ValueError(
                            "spp_configs must exactly match spp_levels: "
                            f"spp_configs[{name!r}] must be a StochasticConfig of kind spp_{name}, "
                            "or its pattern would perturb another consumer")
                self.spp_configs = dict(spp_configs)
        if (skebs_psi is None) != (skebs_theta is None):
            raise ValueError("SKEBS requires both wind and temperature configurations")
        if sppt is not None and sppt.kind != "sppt":
            raise ValueError("SPPT hook requires a sppt configuration")
        self.enabled = bool(enabled and (sppt is not None or skebs_psi is not None or self.spp_levels))
        self.pending_step = None
        self.completed_step = -1
        self.sppt, self.skebs = None, None
        self.spp = {}
        self.parameter_patterns = {}
        self._pattern = self._forcing = None
        if self.enabled:
            from woof.ensemble.stochastic_seeds import normalize_wrf_seed_labels, process_seed
            self.wrf_seed_labels = normalize_wrf_seed_labels(wrf_seed_labels)
            kwargs = dict(dx=dx, dy=dy, dt=dt, member_seed=member_seed,
                          **({"host": True} if self.host else {}))
            if sppt is not None:
                self.sppt = WrfStochasticPattern(sppt, shape_yx, **dict(kwargs,
                    member_seed=process_seed(member_seed, "sppt", self.wrf_seed_labels)))
            if skebs_psi is not None:
                self.skebs = WrfSkebs(shape_yx, **kwargs,
                    psi_config=skebs_psi, theta_config=skebs_theta,
                    **({} if self.wrf_seed_labels is None else {"wrf_seed_labels": self.wrf_seed_labels}))
            for scheme in self.spp_levels:
                self.spp[scheme] = WrfStochasticPattern(
                    self.spp_configs[scheme], shape_yx, **dict(kwargs,
                        member_seed=process_seed(member_seed, "spp_" + scheme, self.wrf_seed_labels)))

    def set_time_step(self, dt: float) -> None:
        """Bind the accepted adaptive or fixed step before generating patterns."""
        if not self.enabled:
            return
        if self.pending_step is not None:
            raise ValueError("cannot change stochastic dt during a pending timestep")
        processes = list(self.spp.values())
        if self.sppt is not None:
            processes.append(self.sppt)
        if self.skebs is not None:
            processes.extend((self.skebs.psi, self.skebs.theta))
        # Validate the complete collection before replacing any coefficients.
        if (isinstance(dt, bool) or not isinstance(dt, (int, float))
                or not math.isfinite(dt) or dt <= 0
                or any(p.config.kind.startswith("skebs_") and dt > p.config.timescale_s
                       for p in processes)):
            raise ValueError("stochastic accepted time step is invalid for its processes")
        for process in processes:
            process.set_time_step(dt)

    def before_timestep(self, step: int) -> None:
        if not self.enabled:
            return
        if self.pending_step is not None:
            raise ValueError("previous stochastic pattern has not been applied to the timestep")
        _uint(step, 32, "step")
        if step != self.completed_step + 1:
            raise ValueError("stochastic timestep hook requires consecutive absolute steps")
        if self.skebs is not None:
            self._forcing = self.skebs.advance(step)
        if self.sppt is not None:
            self._pattern = self.sppt.advance(step)
        if self.spp:
            if self.host:
                cp = np
            else:
                import cupy as cp
            # WRF vertstruc_spp_*=0 repeats the same spectrum at every
            # physical level. Parameter arrays address mass cells only.
            self.parameter_patterns = {
                scheme: cp.broadcast_to(cp.ascontiguousarray(process.advance(step)[:-1, :-1]),
                                        (self.spp_levels[scheme], process.shape[0]-1, process.shape[1]-1))
                for scheme, process in self.spp.items()}
        self.pending_step = step

    def after_nonmicrophysics(self, tendencies: Mapping, *,
                             tendency_scope: str,
                             mass_factors: Mapping | None = None) -> Mapping:
        """Transform physical rates, or dry-mass-coupled rates with factors.

        For mass-coupled tendencies the caller supplies u/v/theta factors
        in exactly those arrays' staggered representation, including any
        map-factor divisions already folded into its slow-tendency convention.
        Omitting factors means the supplied rates are in physical units.
        """
        result = self.transform_nonmicrophysics(tendencies, tendency_scope=tendency_scope,
                                                mass_factors=mass_factors)
        self.complete_timestep()
        return result

    def transform_nonmicrophysics(self, tendencies: Mapping, *, tendency_scope: str,
                                 mass_factors: Mapping | None = None,
                                 sppt_patterns: Mapping | None = None,
                                 skebs_forcing: Mapping | None = None) -> Mapping:
        """Transform rates without consuming the full-domain pending step.

        Windowed execution supplies each component's exact global samples.
        Ordinary calls retain their original pattern slices and arithmetic.
        """
        if not self.enabled:
            return tendencies
        if self.pending_step is None:
            raise ValueError("before_timestep must generate the pattern before tendency application")
        if tendency_scope != "nonmicrophysics":
            raise ValueError("stochastic hook requires nonmicrophysics tendencies")
        if set(tendencies) != {"u", "v", "theta", "qv"}:
            raise ValueError("stochastic hook requires u, v, theta and qv tendencies")
        result = dict(tendencies)
        forcing_fields = self._forcing if skebs_forcing is None else skebs_forcing
        if forcing_fields is not None:
            if mass_factors is not None and set(mass_factors) != {"u", "v", "theta"}:
                raise ValueError("SKEBS mass factors require u, v and theta")
            for name in ("u", "v", "theta"):
                values = result[name]
                field = forcing_fields[name]
                cp, device = _module_of(field)
                if (not isinstance(values, cp.ndarray) or values.ndim != 3
                        or values.dtype != cp.float32 or _module_of(values)[1] != device
                        or any(v > p for v, p in zip(values.shape[-2:], field.shape))):
                    raise ValueError(f"{name} must be a float32 CUDA tendency within the forcing grid")
                ny, nx = values.shape[-2:]
                forcing = field[:ny, :nx]
                if mass_factors is not None:
                    factor = mass_factors[name]
                    if (not isinstance(factor, cp.ndarray) or factor.dtype != cp.float32
                            or factor.shape != values.shape or _module_of(factor)[1] != device):
                        raise ValueError(f"{name} mass factor must match its CUDA tendency")
                    forcing = forcing * factor
                result[name] = values + forcing
        if sppt_patterns is not None:
            result = apply_nonmicrophysics_sppt_components(sppt_patterns, result,
                                                          tendency_scope=tendency_scope)
        elif self._pattern is not None:
            result = apply_nonmicrophysics_sppt(self._pattern, result,
                                               tendency_scope=tendency_scope)
        return result

    def complete_timestep(self) -> None:
        """Commit exactly once after original rates or every owned window."""
        if not self.enabled:
            return
        if self.pending_step is None:
            raise ValueError("stochastic timestep has no pending complete-domain application")
        self.completed_step = self.pending_step
        self.pending_step = None
        self._pattern = self._forcing = None

    def snapshot(self) -> dict:
        if self.pending_step is not None:
            raise ValueError("checkpoint stochastic state after applying the timestep tendencies")
        result = {"enabled": self.enabled, "completed_step": self.completed_step,
                "sppt": self.sppt.snapshot() if self.sppt else None,
                "skebs": self.skebs.snapshot() if self.skebs else None,
                "spp_levels": dict(self.spp_levels),
                "spp": {name: process.snapshot() for name, process in self.spp.items()}}
        if self.wrf_seed_labels is not None:
            from woof.ensemble.stochastic_seeds import seed_label_receipt
            result["wrf_seed_labels"] = seed_label_receipt(self.wrf_seed_labels)
        return result

    def validate_snapshot(self, state: Mapping) -> None:
        """Validate a complete step snapshot without changing any spectrum."""
        if self.pending_step is not None:
            raise ValueError("cannot restore stochastic state during a pending timestep")
        labels = None
        if self.wrf_seed_labels is not None:
            from woof.ensemble.stochastic_seeds import seed_label_receipt
            labels = seed_label_receipt(self.wrf_seed_labels)
        if state.get("wrf_seed_labels") != labels:
            raise ValueError("stochastic restart WRF seed labels or process key derivation differ")
        if state.get("enabled") is not self.enabled:
            raise ValueError("stochastic restart enabling differs from this hook")
        if (state.get("sppt") is None) != (self.sppt is None) or (
                state.get("skebs") is None) != (self.skebs is None):
            raise ValueError("stochastic restart schemes differ from this hook")
        if (state.get("spp_levels", {}) != self.spp_levels
                or set(state.get("spp", {})) != set(self.spp)):
            raise ValueError("stochastic restart SPP consumer levels differ from this hook")
        expected = state.get("completed_step")
        components = []
        if self.sppt is not None:
            components.append((self.sppt, state["sppt"]))
        if self.skebs is not None:
            components.extend(((self.skebs.psi, state["skebs"]["psi"]),
                               (self.skebs.theta, state["skebs"]["theta"])))
        components.extend((process, state["spp"][name]) for name, process in self.spp.items())
        for process, snapshot in components:
            process._validate_state(snapshot)
            if snapshot["completed_step"] != expected:
                raise ValueError("stochastic restart processes have inconsistent histories")
        if len({snapshot["metadata"]["dt"] for _, snapshot in components}) > 1:
            raise ValueError("stochastic restart processes have inconsistent time steps")
    def restore(self, state: Mapping) -> None:
        self.validate_snapshot(state)
        if self.sppt is not None:
            self.sppt.restore(state["sppt"])
        if self.skebs is not None:
            self.skebs.restore(state["skebs"])
        for name, process in self.spp.items():
            process.restore(state["spp"][name])
        self.completed_step = state["completed_step"]
        if self.spp:
            if self.host:
                cp = np
            else:
                import cupy as cp
            self.parameter_patterns = {
                name: cp.broadcast_to(cp.ascontiguousarray(process.gridpoint()[:-1, :-1]),
                                      (self.spp_levels[name], process.shape[0]-1, process.shape[1]-1))
                for name, process in self.spp.items()}
