"""The static background-error covariance of the WOOF global hybrid filter.

What it is
----------
A climatological background-error covariance ``B_s`` in the model's own
spectral variables (vorticity, divergence, theta, vapor, ln surface
pressure on the hybrid levels), estimated from a lagged-forecast ensemble
of differences (the NMC method: the 24 h forecast minus the 12 h forecast
valid at the same instant, one pair per analysis time), stored as a
versioned table with its receipt, and SAMPLED each analysis: ``K`` draws
from ``B_s`` join the ``N`` dynamic members inside the localised solve as
one augmented perturbation matrix, so the control's gain is that of ONE
positive-semidefinite covariance ``beta L o P_ens + (1 - beta) L o P_s``
(Kretschmer, Hunt and Ott 2015, the climatologically augmented LETKF;
Wang, Snyder and Hamill 2007 on the equivalence with the hybrid gain),
never several band analyses summed.

The representation is the spectral-space form of Parrish and Derber
(1992) and Derber and Bouttier (1999): homogeneous and isotropic on the
sphere (the variance of a coefficient depends on the total wavenumber
``n`` and the level, not on the order ``m``), vertically correlated per
wavenumber band, and MULTIVARIATE through the linear balance:

* ``psi`` is the streamfunction of the vorticity; the balanced
  geopotential ``Phi_b = nabla^-2 [nabla . (f nabla psi)]`` is formed with
  the transform's own gradient, the vector analysis and the inverse
  Laplacian (the exact linear balance on the sphere, whose ``f = 2 Omega
  sin(lat)`` couples degree ``n`` to ``n +- 1``; a same-degree regression
  of temperature on vorticity would read zero balance, because the
  balanced mass at degree n comes from the streamfunction at n - 1 and
  n + 1);
* the balanced theta and ln ps are regressions on ``Phi_b`` per
  wavenumber band across levels (``theta_b = N Phi_b``, ``lnps_b = P
  Phi_b``); the balanced divergence a regression on ``psi`` (``D_b = Q
  psi``); the unbalanced residuals ``theta_u``, ``lnps_u``, ``D_u`` and
  the vapor are the univariate control variables whose per-degree
  variances and per-band vertical correlations the table holds.

A draw is therefore: white complex coefficients on the triangle, the
band's vertical correlation applied across levels, the per-degree
standard deviation applied, the balance applied (``theta = theta_u + N
Phi_b(psi)`` and so on), synthesised onto the ensemble grid as the five
analysis fields (``u, v, theta, qv, lnps``).  The sample covariance of
many draws is ``B_s`` by construction, positive-semidefinite whatever the
estimate's sampling noise.

What the sample of record is: the table's receipt names every pair
(the analyses, the forecast checkpoints and their hashes, the instants),
the count, the truncation, the bands and the balanced share of each
variable's variance per band.  A table is refused by name where its
truncation is below the ensemble's (the variances above it are unknown)
or its level count differs from the model's.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import EARTH_ROTATION_RATE_S, SPECTRAL_FIELDS

__all__ = [
    "BALANCE_BANDS_T255",
    "CONTROL_VARIABLES",
    "PACKAGED_TABLE",
    "packaged_table_path",
    "STATIC_COVARIANCE_SCHEMA",
    "StaticCovariance",
    "band_index_by_degree",
    "default_bands",
    "draw_static_perturbations",
    "estimate_static_covariance",
    "lagged_pair_differences",
    "linear_balance_geopotential",
    "load_static_covariance",
    "resolve_static_covariance",
    "spectral_variance_by_degree",
]

STATIC_COVARIANCE_SCHEMA = "gpuwm.arwen-global-static-covariance/v1"

#: The packaged table's filename under this package's own ``data``
#: directory; ``resolve_static_covariance`` turns the word ``packaged`` into
#: the path to it.
#:
#: THE BREAKAGE THIS PREVENTS: in the source tree this table lives under
#: ``woof/data`` and is reached through the engine's ``data_assets``.  No
#: published engine carries it -- the global model is not in the engine's
#: wheel and neither is its data -- so on any clean install the word
#: ``packaged`` resolved to a path that does not exist, and the hybrid
#: covariance, which is ON by default at beta 0.75, refused at the first
#: analysis.  The table is this package's, so this package ships it.
PACKAGED_TABLE = "static-covariance-v1.npz"

#: The univariate control variables the table holds statistics for, in
#: the order the arrays are stored.  ``vorticity`` is the balanced master
#: variable; the three ``*_unbalanced`` variables are the residuals after
#: the balance regressions; the vapor is univariate (Derber and Bouttier).
CONTROL_VARIABLES = (
    "vorticity", "divergence_unbalanced", "theta_unbalanced", "qv", "log_surface_pressure_unbalanced",
)
THREE_D_CONTROL = ("vorticity", "divergence_unbalanced", "theta_unbalanced", "qv")

#: Wavenumber bands of the vertical correlations and the balance
#: regressions: roughly octaves from the planetary scales to the
#: truncation, so each band pools enough (n, m) samples for a levels by
#: levels matrix while the balance still weakens with scale the way the
#: statistics say it does.  ``default_bands(T)`` cuts the last band at
#: the table's truncation.
BALANCE_BANDS_T255 = ((1, 2), (3, 5), (6, 11), (12, 23), (24, 47), (48, 95), (96, 191), (192, 255))


def default_bands(truncation: int) -> tuple[tuple[int, int], ...]:
    """The bands of :data:`BALANCE_BANDS_T255` cut at ``truncation`` (the
    last band ends at the truncation; a band starting above it is
    dropped; a truncation above 255 extends the last band)."""
    t = int(truncation)
    out = []
    for lo, hi in BALANCE_BANDS_T255:
        if lo > t:
            break
        out.append((lo, min(hi, t)))
    if out and out[-1][1] < t:
        out[-1] = (out[-1][0], t)
    if not out:
        out = [(1, max(1, t))]
    return tuple(out)


def band_index_by_degree(bands, truncation: int) -> np.ndarray:
    """``(T+1,)`` the band each degree belongs to (-1 for degree 0 and
    for a degree no band covers)."""
    index = np.full(int(truncation) + 1, -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(bands):
        lo_i = max(int(lo), 0)
        hi_i = min(int(hi), int(truncation))
        if hi_i >= lo_i:
            index[lo_i:hi_i + 1] = b
    return index


def _to_host_complex(coeff) -> np.ndarray:
    host = coeff.get() if hasattr(coeff, "get") else np.asarray(coeff)
    return np.asarray(host, dtype=np.complex128)


def spectral_variance_by_degree(coeff) -> np.ndarray:
    """``(..., T+1)`` the estimate of ``E|c_nm|^2`` per total degree from
    one coefficient set, ``sum_m mult_m |c_nm|^2 / (2n + 1)`` (mult 1 at
    m = 0, where the coefficient is real, 2 for the mirrored m > 0): the
    convention of :func:`woof.globe.da.perturbations.draw_coefficients`,
    under which a real field's grid mean square is ``(1 / 4 pi) sum_n
    (2n + 1) sigma_n^2``."""
    c = _to_host_complex(coeff)
    t = c.shape[-1]
    mult = np.where(np.arange(t) == 0, 1.0, 2.0)
    n = np.arange(c.shape[-2], dtype=np.float64)
    power = np.sum(mult * (c.real ** 2 + c.imag ** 2), axis=-1)
    return power / (2.0 * n + 1.0)


# ---------------------------------------------------------------------------
# The linear balance on the sphere through the transform's own operators
# ---------------------------------------------------------------------------

def linear_balance_geopotential(transform, vector, vorticity, *, rotation_rate_s: float | None = None):
    """The balanced geopotential of a vorticity coefficient stack ``(...,
    n, m)``: ``Phi_b = nabla^-2 [nabla . (f nabla psi)]`` with ``psi =
    nabla^-2 zeta`` and ``f = 2 Omega sin(lat)``, on the transform's
    namespace.  The divergence of the vector ``f nabla psi`` is taken by
    the vector analysis on the dealiased grid (the product with ``f`` is
    quadratic, exact at dealias factor 1.5), so the operator is the
    transform's own arithmetic and not a recurrence written here."""
    xp = transform.backend.xp
    omega = EARTH_ROTATION_RATE_S if rotation_rate_s is None else float(rotation_rate_s)
    zeta = xp.asarray(vorticity, dtype=transform.backend.complex_dtype)
    lead = zeta.shape[:-2]
    psi = transform.inverse_laplacian(zeta.reshape((-1,) + zeta.shape[-2:]))
    east, north = transform.gradient(psi)
    f = xp.asarray(2.0 * omega * np.asarray(transform.grid.sin_lat, dtype=np.float64),
                   dtype=transform.backend.float_dtype)[:, None]
    _zeta_f, div_f = vector.vordiv_from_wind(east * f, north * f)
    phi = transform.inverse_laplacian(div_f)
    return phi.reshape(lead + phi.shape[-2:])


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

@dataclass
class StaticCovariance:
    """The static covariance table.

    truncation, nlev
        The table's triangle and level count.
    bands
        ``(B, 2)`` the wavenumber bands (inclusive degrees).
    reference_p_full_hpa
        ``(nlev,)`` the full-level pressures the levels were at for the
        global-mean surface pressure of the sample (the receipt's reading
        of which level is which).
    variance
        ``{variable: (T+1, nlev)}`` for the 3-D control variables and
        ``(T+1,)`` for ln ps: ``E|c_nm|^2`` per degree and level.
    vertical_correlation
        ``{variable: (B, nlev, nlev)}`` per band, unit diagonal.
    theta_on_phi, lnps_on_phi, divergence_on_psi
        The balance regressions per band: ``(B, nlev, nlev)``, ``(B,
        nlev)`` and ``(B, nlev, nlev)``.
    receipt
        What the table was made from.
    """

    truncation: int
    nlev: int
    bands: np.ndarray
    reference_p_full_hpa: np.ndarray
    variance: dict[str, np.ndarray]
    vertical_correlation: dict[str, np.ndarray]
    theta_on_phi: np.ndarray
    lnps_on_phi: np.ndarray
    divergence_on_psi: np.ndarray
    receipt: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = int(self.truncation)
        nlev = int(self.nlev)
        self.bands = np.asarray(self.bands, dtype=np.int64).reshape(-1, 2)
        nb = int(self.bands.shape[0])
        for name in CONTROL_VARIABLES:
            if name not in self.variance or name not in self.vertical_correlation:
                raise ValueError(f"static covariance table lacks {name!r}")
            v = np.asarray(self.variance[name], dtype=np.float64)
            want = (t + 1,) if name == "log_surface_pressure_unbalanced" else (t + 1, nlev)
            if v.shape != want:
                raise ValueError(f"variance[{name!r}] has shape {v.shape}, expected {want}")
            if not np.all(np.isfinite(v)) or np.any(v < 0.0):
                raise ValueError(f"variance[{name!r}] is not finite and nonnegative")
            self.variance[name] = v
            c = np.asarray(self.vertical_correlation[name], dtype=np.float64)
            want_c = (nb, 1, 1) if name == "log_surface_pressure_unbalanced" else (nb, nlev, nlev)
            if c.shape != want_c:
                raise ValueError(f"vertical_correlation[{name!r}] has shape {c.shape}, expected {want_c}")
            self.vertical_correlation[name] = c
        self.theta_on_phi = np.asarray(self.theta_on_phi, dtype=np.float64).reshape(nb, nlev, nlev)
        self.lnps_on_phi = np.asarray(self.lnps_on_phi, dtype=np.float64).reshape(nb, nlev)
        self.divergence_on_psi = np.asarray(self.divergence_on_psi, dtype=np.float64).reshape(nb, nlev, nlev)
        self.reference_p_full_hpa = np.asarray(self.reference_p_full_hpa, dtype=np.float64).reshape(nlev)

    # -- identity and IO ---------------------------------------------------

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"bands": self.bands, "reference_p_full_hpa": self.reference_p_full_hpa,
               "theta_on_phi": self.theta_on_phi, "lnps_on_phi": self.lnps_on_phi,
               "divergence_on_psi": self.divergence_on_psi}
        for name in CONTROL_VARIABLES:
            out[f"variance__{name}"] = self.variance[name]
            out[f"vertical_correlation__{name}"] = self.vertical_correlation[name]
        return out

    def sha256(self) -> str:
        h = hashlib.sha256()
        h.update(f"{STATIC_COVARIANCE_SCHEMA}|T{self.truncation}|L{self.nlev}".encode("utf-8"))
        for name in sorted(self.arrays()):
            arr = np.ascontiguousarray(self.arrays()[name])
            h.update(name.encode("utf-8"))
            h.update(str(arr.shape).encode("utf-8"))
            h.update(arr.tobytes())
        return h.hexdigest()

    def identity(self) -> dict[str, object]:
        return {
            "schema": STATIC_COVARIANCE_SCHEMA,
            "truncation": int(self.truncation),
            "nlev": int(self.nlev),
            "bands": [[int(a), int(b)] for a, b in self.bands],
            "sha256": self.sha256(),
            "samples": self.receipt.get("samples"),
            "version": self.receipt.get("version"),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": STATIC_COVARIANCE_SCHEMA, "truncation": int(self.truncation), "nlev": int(self.nlev),
            "receipt": self.receipt, "sha256": self.sha256(),
        }
        np.savez_compressed(path, __metadata__=np.array(json.dumps(metadata)), **self.arrays())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StaticCovariance":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            if "__metadata__" not in archive:
                raise ValueError(f"{path} is not a static covariance table (no metadata)")
            metadata = json.loads(str(archive["__metadata__"].item()))
            if metadata.get("schema") != STATIC_COVARIANCE_SCHEMA:
                raise ValueError(f"{path} carries schema {metadata.get('schema')!r}, not {STATIC_COVARIANCE_SCHEMA}")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files if name != "__metadata__"}
        table = cls(
            truncation=int(metadata["truncation"]), nlev=int(metadata["nlev"]),
            bands=arrays["bands"], reference_p_full_hpa=arrays["reference_p_full_hpa"],
            variance={name: arrays[f"variance__{name}"] for name in CONTROL_VARIABLES},
            vertical_correlation={name: arrays[f"vertical_correlation__{name}"] for name in CONTROL_VARIABLES},
            theta_on_phi=arrays["theta_on_phi"], lnps_on_phi=arrays["lnps_on_phi"],
            divergence_on_psi=arrays["divergence_on_psi"], receipt=dict(metadata.get("receipt", {})),
        )
        if metadata.get("sha256") != table.sha256():
            raise ValueError(f"{path} does not carry its own hash: the table was altered after it was written")
        table.receipt.setdefault("path", str(path))
        return table

    # -- what the sampler needs ---------------------------------------------

    def band_index(self, truncation: int | None = None) -> np.ndarray:
        return band_index_by_degree(self.bands, self.truncation if truncation is None else int(truncation))

    def cholesky_factors(self, name: str) -> np.ndarray:
        """``(B, nlev, nlev)`` lower Cholesky factors of the band
        correlations (a nudge on the diagonal where a band's sample made
        the matrix indefinite to rounding)."""
        corr = self.vertical_correlation[name]
        out = np.zeros_like(corr)
        for b in range(corr.shape[0]):
            c = 0.5 * (corr[b] + corr[b].T)
            eps = 0.0
            for _ in range(8):
                try:
                    out[b] = np.linalg.cholesky(c + eps * np.eye(c.shape[0]))
                    break
                except np.linalg.LinAlgError:
                    eps = 1.0e-10 if eps == 0.0 else eps * 10.0
            else:
                raise ValueError(f"vertical correlation of {name!r} band {b} is not positive-semidefinite")
        return out

    def check_against(self, ensemble_truncation: int, nlev: int) -> None:
        if int(ensemble_truncation) > int(self.truncation):
            raise ValueError(
                f"the static covariance table covers degrees up to T{self.truncation} and the ensemble "
                f"is T{ensemble_truncation}: the variances above the table's truncation are unknown, "
                "so the hybrid refuses rather than draw zero power there"
            )
        if int(nlev) != int(self.nlev):
            raise ValueError(
                f"the static covariance table has {self.nlev} levels and the model {nlev}: the "
                "vertical correlations and the balance regressions are per level and do not "
                "interpolate; estimate a table on the model's own ladder"
            )


def packaged_table_path() -> Path:
    """The shipped table's path inside the installed package."""
    return Path(__file__).resolve().parent.parent / "data" / PACKAGED_TABLE


def resolve_static_covariance(spec: str | Path | None) -> Path | None:
    """The table path a ``FilterOptions.static_covariance`` names:
    ``None`` stays None, ``packaged`` is the table this package ships,
    anything else is a path."""
    if spec is None:
        return None
    text = str(spec)
    if text == "packaged":
        return packaged_table_path()
    return Path(text)


def load_static_covariance(spec: str | Path | None) -> StaticCovariance | None:
    path = resolve_static_covariance(spec)
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(
            f"static covariance table {path} does not exist"
            + (" (the table this package ships is missing from the install, which "
               "means the wheel was built without its package data; reinstall, or "
               "estimate one with `woof global da static-covariance` and name its "
               "path)" if str(spec) == "packaged" else "")
        )
    return StaticCovariance.load(path)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def lagged_pair_differences(pairs, *, read=None):
    """The spectral differences of checkpoint pairs ``(later_forecast,
    earlier_forecast)`` valid at the same instant (the 24 h forecast and
    the 12 h forecast): a generator of ``(record, {field: (nlev, T+1,
    T+1) complex})`` on the host, ``record`` naming both files, their
    hashes, steps and times.  The two must agree on their instant to the
    second and on their shapes."""
    from ..checkpoint import read_checkpoint

    read = read or read_checkpoint
    for later, earlier in pairs:
        later = Path(later)
        earlier = Path(earlier)
        meta_l, arrays_l = read(later)
        meta_e, arrays_e = read(earlier)
        diff = {}
        for name in SPECTRAL_FIELDS:
            a = np.asarray(arrays_l[f"atmosphere__{name}"], dtype=np.complex128)
            b = np.asarray(arrays_e[f"atmosphere__{name}"], dtype=np.complex128)
            if a.shape != b.shape:
                raise ValueError(f"{later} and {earlier} disagree on the shape of {name}: {a.shape} against {b.shape}")
            diff[name] = a - b
        record = {
            "later": {"path": str(later), "self_sha256": meta_l.get("self_sha256"), "step": meta_l.get("step"),
                      "time_s": meta_l.get("time_s")},
            "earlier": {"path": str(earlier), "self_sha256": meta_e.get("self_sha256"), "step": meta_e.get("step"),
                        "time_s": meta_e.get("time_s")},
        }
        yield record, diff


def _band_slices(bands, truncation: int):
    for b, (lo, hi) in enumerate(bands):
        lo_i = max(int(lo), 0)
        hi_i = min(int(hi), int(truncation))
        if hi_i >= lo_i:
            yield b, slice(lo_i, hi_i + 1)


def _cross(x, y):
    """``Re(sum x y^H)`` over the trailing (degree, order) axes of two
    ``(rows, n, m)`` stacks: ``(rows_x, rows_y)`` real; the real and the
    imaginary part each count as one sample, which is what a real linear
    operator between real fields regresses on."""
    xf = x.reshape(x.shape[0], -1)
    yf = y.reshape(y.shape[0], -1)
    return np.real(xf @ np.conj(yf).T)


def _ridge_solve(sxy, sxx, ridge: float):
    """``N = S_xy (S_xx + ridge tr(S_xx)/n I)^-1``."""
    n = sxx.shape[0]
    reg = sxx + float(ridge) * (np.trace(sxx) / n) * np.eye(n)
    return np.linalg.solve(reg.T, sxy.T).T


def estimate_static_covariance(differences, transform, vector, *, bands=None, ridge: float = 1.0e-3,
                               reference_p_full_hpa=None, progress=None) -> StaticCovariance:
    """The table from an iterable of ``(record, {field: coefficients})``
    lagged differences (:func:`lagged_pair_differences`; the iterable is
    consumed twice, so hand over a list or a re-iterable).

    Pass one forms the balance regressions per band (theta and ln ps on
    the balanced geopotential of the difference's vorticity, the
    divergence on its streamfunction); pass two removes the balanced part
    and accumulates the per-degree variances and the per-band vertical
    correlations of the residuals."""
    differences = list(differences)
    if len(differences) < 2:
        raise ValueError("a static covariance needs at least two lagged differences")
    xp = transform.backend.xp
    t = int(transform.truncation)
    first = differences[0][1]
    nlev = int(first["theta"].shape[0])
    bands = tuple(default_bands(t) if bands is None else tuple((int(a), int(b)) for a, b in bands))
    nb = len(bands)
    clock = time.perf_counter()

    # The sample mean of the differences is removed before anything is
    # regressed or squared: a lagged difference carries the model's own
    # systematic drift between the two forecast ranges (the same sign in
    # every pair: on the T255 imex_ssp3 sample of record the global-mean
    # theta at the 1.2 hPa lid read -114.6 K in all eleven pairs, 13,100 K^2
    # of "variance" that was a bias, and 20 to 50 percent of the theta
    # second moment at every level was the mean's), and a background-error
    # covariance is the covariance about that mean, not the second moment
    # about zero.  The drift is recorded in the receipt per variable and
    # level as the share of the raw second moment the sample mean held,
    # with the global-mean theta drift per level in kelvin.
    count = float(len(differences))
    mult = np.where(np.arange(t + 1) == 0, 1.0, 2.0)
    drift_mean = {name: sum(diff[name] for _, diff in differences) / count for name in SPECTRAL_FIELDS}
    drift_share = {}
    for name in SPECTRAL_FIELDS:
        raw = sum(np.sum(mult * np.abs(diff[name]) ** 2, axis=(-2, -1)) for _, diff in differences) / count
        held = np.sum(mult * np.abs(drift_mean[name]) ** 2, axis=(-2, -1))
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(raw > 0.0, held / raw, 0.0)
        drift_share[name] = [float(s) for s in np.atleast_1d(share)]
    for _, diff in differences:
        for name in SPECTRAL_FIELDS:
            diff[name] -= drift_mean[name]
    drift_record = {
        "removed": True,
        "rule": "the sample mean of the lagged differences over the pairs is subtracted per spectral coefficient "
                "before the regressions, variances and correlations are formed; the variances are the unbiased "
                "sample variances about that mean (denominator samples - 1)",
        "share_of_raw_second_moment_by_level": drift_share,
        "expected_share_if_no_drift": 1.0 / count,
        "theta_global_mean_drift_k_by_level": [
            float(np.real(drift_mean["theta"][k, 0, 0]) / math.sqrt(4.0 * math.pi)) for k in range(nlev)],
    }
    del drift_mean

    def balance_fields(diff):
        zeta = xp.asarray(diff["vorticity"], dtype=transform.backend.complex_dtype)
        psi = _to_host_complex(transform.inverse_laplacian(zeta))
        phi = _to_host_complex(linear_balance_geopotential(transform, vector, zeta))
        return psi, phi

    # Pass one: the regression sums per band.
    s_pp = np.zeros((nb, nlev, nlev)); s_tp = np.zeros((nb, nlev, nlev)); s_lp = np.zeros((nb, 1, nlev))
    s_ss = np.zeros((nb, nlev, nlev)); s_ds = np.zeros((nb, nlev, nlev))
    cache = []
    for i, (record, diff) in enumerate(differences):
        psi, phi = balance_fields(diff)
        cache.append((psi, phi))
        theta = diff["theta"]; lnps = diff["log_surface_pressure"].reshape(1, t + 1, t + 1); div = diff["divergence"]
        for b, sl in _band_slices(bands, t):
            s_pp[b] += _cross(phi[:, sl, :], phi[:, sl, :])
            s_tp[b] += _cross(theta[:, sl, :], phi[:, sl, :])
            s_lp[b] += _cross(lnps[:, sl, :], phi[:, sl, :])
            s_ss[b] += _cross(psi[:, sl, :], psi[:, sl, :])
            s_ds[b] += _cross(div[:, sl, :], psi[:, sl, :])
        if progress is not None:
            progress(f"static covariance pass 1: sample {i + 1} of {len(differences)}")
    theta_on_phi = np.zeros((nb, nlev, nlev)); lnps_on_phi = np.zeros((nb, nlev)); div_on_psi = np.zeros((nb, nlev, nlev))
    for b in range(nb):
        if np.trace(s_pp[b]) > 0.0:
            theta_on_phi[b] = _ridge_solve(s_tp[b], s_pp[b], ridge)
            lnps_on_phi[b] = _ridge_solve(s_lp[b], s_pp[b], ridge)[0]
        if np.trace(s_ss[b]) > 0.0:
            div_on_psi[b] = _ridge_solve(s_ds[b], s_ss[b], ridge)

    # Pass two: the residuals' variances and vertical correlations.
    variance = {name: np.zeros((t + 1, nlev)) for name in THREE_D_CONTROL}
    variance["log_surface_pressure_unbalanced"] = np.zeros(t + 1)
    total_variance = {"theta": np.zeros((t + 1, nlev)), "divergence": np.zeros((t + 1, nlev)),
                      "log_surface_pressure": np.zeros(t + 1)}
    corr_sum = {name: np.zeros((nb, nlev, nlev)) for name in THREE_D_CONTROL}
    corr_sum["log_surface_pressure_unbalanced"] = np.zeros((nb, 1, 1))
    for i, (record, diff) in enumerate(differences):
        psi, phi = cache[i]
        theta_u = diff["theta"].copy(); lnps_u = diff["log_surface_pressure"].copy(); div_u = diff["divergence"].copy()
        for b, sl in _band_slices(bands, t):
            theta_u[:, sl, :] -= np.einsum("ij,jnm->inm", theta_on_phi[b], phi[:, sl, :])
            lnps_u[sl, :] -= np.einsum("j,jnm->nm", lnps_on_phi[b], phi[:, sl, :])
            div_u[:, sl, :] -= np.einsum("ij,jnm->inm", div_on_psi[b], psi[:, sl, :])
        fields = {"vorticity": diff["vorticity"], "divergence_unbalanced": div_u, "theta_unbalanced": theta_u,
                  "qv": diff["qv"], "log_surface_pressure_unbalanced": lnps_u[None]}
        for name, coeff in fields.items():
            v = spectral_variance_by_degree(coeff)            # (rows, T+1)
            if name == "log_surface_pressure_unbalanced":
                variance[name] += v[0]
            else:
                variance[name] += v.T
            for b, sl in _band_slices(bands, t):
                corr_sum[name][b] += _cross(coeff[:, sl, :], coeff[:, sl, :])
        total_variance["theta"] += spectral_variance_by_degree(diff["theta"]).T
        total_variance["divergence"] += spectral_variance_by_degree(diff["divergence"]).T
        total_variance["log_surface_pressure"] += spectral_variance_by_degree(diff["log_surface_pressure"][None])[0]
        if progress is not None:
            progress(f"static covariance pass 2: sample {i + 1} of {len(differences)}")
    # Unbiased about the removed mean: one degree of freedom went to it.
    dof = max(count - 1.0, 1.0)
    for name in variance:
        variance[name] /= dof
    for name in total_variance:
        total_variance[name] /= dof
    # Degree 0 carries no vorticity, divergence or ln ps (the mass rule
    # owns the global-mean pressure); theta's and the vapor's global-mean
    # variances are kept.
    for name in ("vorticity", "divergence_unbalanced"):
        variance[name][0] = 0.0
    variance["log_surface_pressure_unbalanced"][0] = 0.0
    vertical_correlation = {}
    for name, s in corr_sum.items():
        out = np.zeros_like(s)
        for b in range(nb):
            d = np.sqrt(np.clip(np.diag(s[b]), 0.0, None))
            with np.errstate(divide="ignore", invalid="ignore"):
                c = s[b] / np.outer(d, d)
            c[~np.isfinite(c)] = 0.0
            np.fill_diagonal(c, 1.0)
            out[b] = 0.5 * (c + c.T)
        vertical_correlation[name] = out

    def balanced_share(total, residual):
        share = []
        for b, sl in _band_slices(bands, t):
            tot = float(np.sum(total[sl]))
            res = float(np.sum(residual[sl]))
            share.append(None if tot <= 0.0 else float(1.0 - res / tot))
        return share

    receipt = {
        "schema": STATIC_COVARIANCE_SCHEMA,
        "method": "lagged-forecast differences (NMC): later forecast minus earlier forecast valid at one instant",
        "samples": len(differences),
        "pairs": [record for record, _ in differences],
        "truncation": t, "nlev": nlev,
        "bands": [[int(a), int(b)] for a, b in bands],
        "ridge": float(ridge),
        "balance": {
            "form": "Phi_b = nabla^-2[nabla.(f nabla psi)] through the transform's gradient, vector analysis and "
                    "inverse Laplacian; theta_b = N(band) Phi_b, lnps_b = P(band) Phi_b, D_b = Q(band) psi; "
                    "N, P, Q least squares per band over levels with a relative ridge",
            "balanced_share_of_variance": {
                "theta": balanced_share(total_variance["theta"].sum(axis=1), variance["theta_unbalanced"].sum(axis=1)),
                "divergence": balanced_share(total_variance["divergence"].sum(axis=1),
                                             variance["divergence_unbalanced"].sum(axis=1)),
                "log_surface_pressure": balanced_share(total_variance["log_surface_pressure"],
                                                       variance["log_surface_pressure_unbalanced"]),
            },
        },
        "degree_zero": "vorticity, divergence and ln ps carry no degree-0 variance; theta and qv keep theirs",
        "drift": drift_record,
        "estimation_wall_s": float(time.perf_counter() - clock),
    }
    p_ref = (np.full(nlev, np.nan) if reference_p_full_hpa is None
             else np.asarray(reference_p_full_hpa, dtype=np.float64).reshape(nlev))
    return StaticCovariance(
        truncation=t, nlev=nlev, bands=np.asarray(bands), reference_p_full_hpa=p_ref,
        variance=variance, vertical_correlation=vertical_correlation,
        theta_on_phi=theta_on_phi, lnps_on_phi=lnps_on_phi, divergence_on_psi=div_on_psi, receipt=receipt,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _white_triangle(rng, count: int, rows: int, truncation: int) -> np.ndarray:
    """``(count, rows, T+1, T+1)`` complex white coefficients on the
    triangle with ``E|c|^2 = 1`` (real at m = 0, nothing above the
    triangle)."""
    t = truncation + 1
    real = rng.standard_normal((count, rows, t, t))
    imag = rng.standard_normal((count, rows, t, t))
    unit = (real + 1j * imag) / math.sqrt(2.0)
    unit[..., 0] = real[..., 0]
    tri = np.tri(t, dtype=bool)
    unit[..., ~tri] = 0.0
    return unit


def draw_static_spectral(table: StaticCovariance, truncation: int, count: int, rng) -> dict[str, np.ndarray]:
    """``count`` draws of the five spectral fields at ``truncation`` from
    the table, on the host: ``{field: (count, nlev, T+1, T+1)}`` (ln ps
    ``(count, 1, T+1, T+1)``), the balance NOT yet applied (the balance
    needs the transform; :func:`draw_static_perturbations` applies it)."""
    t = int(truncation)
    table.check_against(t, table.nlev)
    nlev = int(table.nlev)
    out = {}
    for name in CONTROL_VARIABLES:
        rows = 1 if name == "log_surface_pressure_unbalanced" else nlev
        w = _white_triangle(rng, int(count), rows, t)
        if rows > 1:
            factors = table.cholesky_factors(name)
            for b, sl in _band_slices(table.bands, t):
                w[:, :, sl, :] = np.einsum("ij,kjnm->kinm", factors[b], w[:, :, sl, :])
        var = table.variance[name][:t + 1]
        sigma = np.sqrt(var if rows == 1 else var.T)         # (nlev, T+1) or (T+1,)
        if rows == 1:
            w *= sigma[None, None, :, None]
        else:
            w *= sigma[None, :, :, None]
        out[name] = w
    return out


def draw_static_perturbations(table: StaticCovariance, transform, vector, count: int, rng, *,
                              include_spectral: bool = True, balance_chunk: int = 8) -> dict[str, object]:
    """``count`` static perturbations on the transform's grid: ``{u, v,
    theta, qv, lnps}`` each ``(count, ...)`` on the transform's namespace
    (and, with ``include_spectral``, ``spectral`` as the five balanced
    spectral fields ``{field: (count, nlev, T+1, T+1)}`` on the host for
    the observation operators).  The balance is applied per band: theta
    takes ``N Phi_b``, ln ps ``P Phi_b``, the divergence ``Q psi``."""
    xp = transform.backend.xp
    backend = transform.backend
    t = int(transform.truncation)
    draws = draw_static_spectral(table, t, count, rng)
    zeta_all = draws["vorticity"]
    theta = draws["theta_unbalanced"]
    lnps = draws["log_surface_pressure_unbalanced"]
    div = draws["divergence_unbalanced"]
    # The balance, a chunk of draws at a time: the balanced geopotential's
    # gradient synthesis is (draws x levels) grid fields on the device, so
    # the chunk keeps the transient to a few hundred megabytes at T127.
    chunk = max(1, int(balance_chunk))
    for start in range(0, int(count), chunk):
        stop = min(int(count), start + chunk)
        zeta_dev = xp.asarray(zeta_all[start:stop], dtype=backend.complex_dtype)
        psi = _to_host_complex(transform.inverse_laplacian(zeta_dev.reshape(-1, t + 1, t + 1))).reshape(
            zeta_all[start:stop].shape)
        phi = _to_host_complex(linear_balance_geopotential(transform, vector, zeta_dev))
        del zeta_dev
        for b, sl in _band_slices(table.bands, t):
            theta[start:stop, :, sl, :] += np.einsum("ij,kjnm->kinm", table.theta_on_phi[b], phi[:, :, sl, :])
            lnps[start:stop, 0, sl, :] += np.einsum("j,kjnm->knm", table.lnps_on_phi[b], phi[:, :, sl, :])
            div[start:stop, :, sl, :] += np.einsum("ij,kjnm->kinm", table.divergence_on_psi[b], psi[:, :, sl, :])
    spectral = {"vorticity": zeta_all, "divergence": div, "theta": theta, "qv": draws["qv"],
                "log_surface_pressure": lnps[:, 0]}
    grid = {name: [] for name in ("u", "v", "theta", "qv", "lnps")}
    for k in range(int(count)):
        z = xp.asarray(spectral["vorticity"][k], dtype=backend.complex_dtype)
        d = xp.asarray(spectral["divergence"][k], dtype=backend.complex_dtype)
        u, v = vector.wind_from_vordiv(z, d)
        grid["u"].append(u)
        grid["v"].append(v)
        grid["theta"].append(transform.inverse(xp.asarray(spectral["theta"][k], dtype=backend.complex_dtype)))
        grid["qv"].append(transform.inverse(xp.asarray(spectral["qv"][k], dtype=backend.complex_dtype)))
        grid["lnps"].append(transform.inverse(xp.asarray(spectral["log_surface_pressure"][k], dtype=backend.complex_dtype)))
    out = {name: xp.stack(values) for name, values in grid.items()}
    if include_spectral:
        out["spectral"] = spectral
    return out
