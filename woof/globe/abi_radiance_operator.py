"""The ABI clear-sky brightness-temperature stream for the ensemble filter.

Two halves, both built on the interfaces the ensemble lane committed
(``woof.globe.da.observations.PointObs``: one batch per stream and
variable, ``operator(members, batch) -> (R, n)`` re-evaluated on the
analysed members for O-A):

* **observations**: the block table ``rw_goes colocate`` writes (the
  superobservation vector: the mean observed brightness temperature of
  the both-clear pixels of each 24-pixel block, the block's mean latitude,
  longitude and satellite zenith) becomes a :class:`PointObs` batch of
  ``brightness_temperature_k`` for one band, under the stream's block QC
  (:data:`abi_reference.STREAM_QC`: enough both-clear pixels, at least
  half the block both-clear so a cloud edge is not an observation, the
  admitted zenith band), restricted to the surface classes the operator
  entry admits (the entry is graded on exactly this population) with the
  entry's per-class observation error and linear correction, and with the
  block's time inside the scan (the ABI full disk scans north to south; a
  block's row places it in the scan window, an approximation the record
  names);
* **the operator**: every member's columns at the blocks are sampled from
  its spectral state (theta and vapor synthesised at the points, the
  surface pressure for the coordinate, the member's own skin and the
  shared surface classes), written as a ``gpuwm-da.abi-columns.v2``
  stream, evaluated by the Rust operator ``rw_goes forward`` with the
  band's coefficient table, and read back as ``(R, n)``.

Vertical placement (amendment H of the design: a radiance is not a
one-level observation): a band's rows carry a localisation PROFILE built
from the table's reference Jacobians (:func:`band_profile`: the
temperature Jacobian per unit ln p plus the vapor Jacobian for a ten
percent vapor change, convolved with the Gaspari-Cohn kernel of the
band's vertical length measured on the ensemble each window), read by
the filter at every analysis level
(:mod:`woof.globe.da.localisation`); ``ln_pressure`` names the
Jacobian's centroid for thinning and the receipt.

The operator answers for the members, the control and the analysed
copies alike: it holds one transform per truncation (:meth:`bind`) and
chooses by the states' own (``evaluates_states``); the per-row viewing
geometry and the entry's linear correction are keyed by row identity so
any subset evaluates in the corrected frame the entry was graded in.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

from . import abi_fast_model as fm
from . import abi_reference as ref
from .abi_operator import AbiOperatorError, find_rw_goes
from .abi_reference import STREAM_QC

STREAM = "goes-abi-l1b-bt"
VARIABLE = "brightness_temperature_k"
FORWARD_RECORD_SCHEMA = "gpuwm-da.abi-forward.v1"


def run_forward(columns: str | os.PathLike, table: str | os.PathLike, out: str | os.PathLike, *,
                rw_goes: str | os.PathLike | None = None, emis_mode: int = 0, bands: tuple[int, ...] | None = None,
                threads: int | None = None) -> dict:
    """``rw_goes forward`` on a columns stream; returns its receipt with
    the wall and the command."""
    exe = find_rw_goes(rw_goes)
    command = [str(exe), "forward", "--columns", str(columns), "--table", str(table), "--out", str(out),
               "--emis-mode", str(int(emis_mode))]
    if bands:
        command += ["--bands", ",".join(str(int(b)) for b in bands)]
    if threads:
        command += ["--threads", str(int(threads))]
    started = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise AbiOperatorError(f"rw_goes forward failed (rc {done.returncode}): {done.stderr.strip() or done.stdout.strip()}")
    record = json.loads(done.stdout)
    if record.get("schema") != FORWARD_RECORD_SCHEMA:
        raise AbiOperatorError(f"rw_goes forward answered with schema {record.get('schema')!r}")
    record["_wall_s"] = round(time.perf_counter() - started, 3)
    record["_command"] = command
    return record


# ---------------------------------------------------------------------------
# observations: the block table as a PointObs batch
# ---------------------------------------------------------------------------

def scan_row_time(scan_start: dt.datetime, scan_end: dt.datetime, block_y: np.ndarray, block_pixels: int,
                  axis_pixels: int = 5424) -> list[dt.datetime]:
    """A block's time inside the scan: the ABI full-disk scan runs north
    to south in swaths, so a row's fraction of the disk from the north
    places it linearly in the scan window (the swath structure is not
    modelled; a full-disk swath is about 26 s, the stated uncertainty)."""
    span = (scan_end - scan_start).total_seconds()
    centre = (np.asarray(block_y, dtype=np.float64) + 0.5) * block_pixels
    fraction = np.clip(1.0 - centre / axis_pixels, 0.0, 1.0)       # block_y counts from the south
    return [scan_start + dt.timedelta(seconds=float(f) * span) for f in fraction]


def superobs_from_blocks(block_csv: str | os.PathLike, *, band: int, scan_start: dt.datetime, scan_end: dt.datetime,
                         observation_error_k: float | None = None, entry: dict | None = None, land_fraction=None,
                         zenith_max_deg: float = STREAM_QC["zenith_max_deg"], minimum_pixels: int = STREAM_QC["minimum_pixels"],
                         minimum_clear_fraction: float = STREAM_QC["minimum_clear_fraction"], ln_pressure_hpa: float | None = None,
                         vertical_cutoff_lnp: float | None = None, horizontal_cutoff_km: float | None = None,
                         block_pixels: int = 24, identity_prefix: str = ""):
    """The both-clear blocks of one band's colocation table as one
    :class:`PointObs` batch under the stream's QC: a block with fewer than
    ``minimum_pixels`` both-clear pixels, or with less than
    ``minimum_clear_fraction`` of its paired pixels both-clear (a cloud
    edge: on the case such blocks read the cloud the mask missed, 8 to
    12 K, not the operator), or beyond ``zenith_max_deg`` is dropped and
    counted.  With an operator ``entry`` (one band of
    ``gpuwm-da.abi-operator-entries.v1``) the batch keeps only the
    surface classes the entry admits, classed by ``land_fraction`` (an
    array aligned with the table's rows, or a callable of latitude and
    longitude returning one), carries the entry's per-class observation
    error, and returns the entry's per-row linear correction in the
    extras for the operator to apply; the entry's QC must be this batch's.
    Without an entry ``observation_error_k`` is the error of every row.
    ``ln_pressure_hpa`` is the release-1 vertical placement (the reference
    weighting function's median peak)."""
    from .da.observations import PointObs

    table = np.genfromtxt(block_csv, delimiter=",", names=True)
    table = np.atleast_1d(table)
    n_both = table["n_both_clear"].astype(np.float64)
    pairs = table["n_pair"].astype(np.float64)
    zen_all = table["zenith_mean_deg"].astype(np.float64)
    clear_fraction = np.where(pairs > 0, n_both / np.maximum(pairs, 1.0), 0.0)
    enough = n_both >= minimum_pixels
    clear = enough & (clear_fraction >= minimum_clear_fraction)
    inside = clear & (zen_all <= zenith_max_deg)
    rejections = {"thinned_below_minimum_pixels": int(np.sum(~enough)),
                  "below_minimum_clear_fraction": int(np.sum(enough & ~clear)),
                  "beyond_zenith": int(np.sum(clear & ~inside))}
    keep = inside
    qc = {"minimum_pixels": int(minimum_pixels), "minimum_clear_fraction": float(minimum_clear_fraction),
          "zenith_max_deg": float(zenith_max_deg)}
    surface_class = np.full(table.size, "any", dtype=object)
    intercept = np.zeros(table.size)
    slope = np.ones(table.size)
    error = np.full(table.size, np.nan)
    if entry is not None:
        if int(entry.get("band", band)) != int(band):
            raise AbiOperatorError(f"the entry is band {entry.get('band')}, the batch band {band}")
        if land_fraction is None:
            raise AbiOperatorError(
                "the operator entry admits blocks by surface class, so the batch needs land_fraction (an array aligned with "
                "the block table's rows, or a callable of latitude and longitude); without it a land block would be handed "
                "to the filter under the water class's error")
        lf = (land_fraction(table["lat_mean_deg"].astype(np.float64), table["lon_mean_deg"].astype(np.float64))
              if callable(land_fraction) else np.asarray(land_fraction, dtype=np.float64))
        lf = np.atleast_1d(lf)
        if lf.shape != (table.size,):
            raise AbiOperatorError(f"land_fraction has shape {lf.shape}, the block table {table.size} rows")
        surface_class = np.where(lf >= 0.5, "land", "water").astype(object)
        classes = entry["classes"]
        admitted = [c for c in entry.get("admitted_classes", []) if c in classes]
        for c in admitted:
            row_qc = classes[c].get("qc") or (classes[c].get("filter_facing") or {}).get("qc")
            if row_qc:
                for key in ("minimum_pixels", "minimum_clear_fraction", "zenith_max_deg"):
                    if key in row_qc and abs(float(row_qc[key]) - qc[key]) > 1e-9:
                        raise AbiOperatorError(
                            f"the entry's {c} class was graded with {key} = {row_qc[key]}, this batch asks {qc[key]}; the "
                            "entry's error and correction hold only for the population it was graded on")
        ok = np.array([c in admitted for c in surface_class], dtype=bool)
        rejections["class_not_admitted"] = int(np.sum(keep & ~ok))
        keep &= ok
        for c in admitted:
            m = surface_class == c
            corr = classes[c].get("bias_correction") or {}
            intercept[m] = float(corr.get("intercept_k", 0.0))
            slope[m] = float(corr.get("slope", 1.0))
            error[m] = float(classes[c]["observation_error_k"])
    else:
        if observation_error_k is None:
            raise AbiOperatorError("without an operator entry the batch needs observation_error_k for every row")
        error[:] = float(observation_error_k)
    rows = table[keep]
    n = int(rows.size)
    lat = rows["lat_mean_deg"].astype(np.float64)
    lon = rows["lon_mean_deg"].astype(np.float64)
    value = rows["obs_mean_both_clear_k"].astype(np.float64)
    ln_p = np.full(n, np.log(ln_pressure_hpa * 100.0) if ln_pressure_hpa else np.nan)
    identity = np.array([f"{identity_prefix}abi{band:02d}:{int(x)}:{int(y)}" for x, y in zip(rows["block_x"], rows["block_y"])],
                        dtype=object)
    batch = PointObs(
        stream=STREAM, variable=VARIABLE, latitude_deg=lat, longitude_deg=lon, ln_pressure=ln_p,
        surface=np.zeros(n, dtype=bool), value=value, error=error[keep],
        identity=identity, valid_time=scan_row_time(scan_start, scan_end, rows["block_y"], block_pixels),
        horizontal_cutoff_km=horizontal_cutoff_km, vertical_cutoff_lnp=vertical_cutoff_lnp,
    )
    batch.rejections.update(rejections)
    extras = {"band": int(band), "zenith_deg": rows["zenith_mean_deg"].astype(np.float64),
              "n_pixels": rows["n_both_clear"].astype(int), "clear_fraction": clear_fraction[keep],
              "block_x": rows["block_x"].astype(int), "block_y": rows["block_y"].astype(int),
              "surface_class": surface_class[keep], "qc": qc,
              "correction_intercept_k": intercept[keep], "correction_slope": slope[keep],
              "admitted_classes": (list(entry.get("admitted_classes", [])) if entry is not None else None)}
    return batch, extras


# ---------------------------------------------------------------------------
# the operator: members to columns to rw_goes forward
# ---------------------------------------------------------------------------

#: The vapor floor of the tangent-linear form's logarithm (g/kg): the
#: model's upper layers carry vapor of the forward's Jacobian step floor
#: (1e-9 g/kg, a spectral synthesis of a near-zero field), so the ratio of
#: a member's or a draw's vapor to the reference's is taken above this
#: floor; the weight itself carries the reference's own vapor, unfloored.
VAPOR_LINEARISATION_FLOOR_GKG = 1.0e-6

#: The linear-regime guard of :meth:`AbiRadianceOperator.__call__`: a pair
#: whose tangent-linear departure from its reference exceeds this many
#: assigned errors (never under the floor) takes the full forward.
LINEAR_REGIME_ERROR_MULTIPLE = 3.0
LINEAR_REGIME_FLOOR_K = 1.0


def vapor_tangent_linear(jac_q: np.ndarray, q_ref: np.ndarray, q: np.ndarray,
                         *, floor_gkg: float = VAPOR_LINEARISATION_FLOOR_GKG) -> np.ndarray:
    """The vapor term of the tangent-linear forward in the relative form
    ``dTb/dln q = jac_q * q_ref`` times ``ln(q / q_ref)`` (both floored),
    summed over layers: ``(rows, nlay)`` in, ``(rows,)`` out.  To first
    order it is ``jac_q * (q - q_ref)``; for a departure large against the
    reference it grows with the logarithm of the ratio, as the layer's
    optical depth does (the weight ``jac_q * q_ref`` is the forward's own
    per-ln-q sensitivity, at most about 1 K on the packaged table, and
    zero on a reference layer with no vapor), where the per-g/kg form grows
    with the ratio itself
    (the reference door's own note on ``jacobian_agreement``: the per-g/kg
    Jacobian explodes where the air is dry, so a static draw or an analysis
    increment of 1e-4 g/kg over a 1e-7 g/kg reference read as millions of
    kelvin on the water-vapor band)."""
    f = float(floor_gkg)
    ref = np.maximum(np.asarray(q_ref, dtype=np.float64), 0.0)
    weight = np.asarray(jac_q, dtype=np.float64) * ref          # K per unit ln q, zero on a dry reference layer
    ratio = np.log(np.maximum(np.asarray(q, dtype=np.float64), f)) - np.log(np.maximum(ref, f))
    return np.sum(weight * ratio, axis=-1)


class AbiRadianceOperator:
    """``operator(members, batch) -> (R, n)`` for the brightness-temperature
    stream, through the Rust forward operator.

    ``transform`` and ``vertical`` are the ensemble's (the members'
    truncation); ``table`` the coefficient table (its coordinate must be
    the ensemble's, which the Rust proves); ``zenith_deg`` the per-row
    satellite zenith the batch's extras carry (the operator keys it by
    the row identity so a subset keeps its angles); ``month`` sets the
    climatology code the columns carry for the reference (unused by the
    fast model, carried for the stream's contract).
    """

    #: The flag :func:`woof.globe.da.operators.evaluate_batches` reads.
    evaluates_states = True

    def __init__(self, table: str | os.PathLike, transform, vertical, *, band: int, zenith_by_identity: dict[str, float],
                 rw_goes: str | os.PathLike | None = None, work_dir: str | os.PathLike | None = None,
                 month: int = 1, threads: int | None = None, member_chunk: int = 8, keep_files: bool = False,
                 transforms=()):
        self.table_path = Path(table)
        self.table = fm.read_table(self.table_path)
        self.band = int(band)
        if str(self.band) not in self.table["bands"]:
            raise AbiOperatorError(f"band {band} is not in {self.table_path} (bands {sorted(self.table['bands'])})")
        self.transforms: dict[int, object] = {}
        for t in ((transform,) if transform is not None else ()) + tuple(transforms):
            self.bind(t)
        self.transform = transform
        self.vertical = vertical
        self.a = np.asarray(vertical.a_half_pa, dtype=np.float64)
        self.b = np.asarray(vertical.b_half, dtype=np.float64)
        identity = fm.coordinate_identity(self.a, self.b)
        stated = (self.table.get("vertical") or {}).get("sha256")
        if stated and stated != identity["sha256"]:
            raise AbiOperatorError(
                f"the coefficient table {self.table_path} was fitted on vertical coordinate {stated[:12]}, the "
                f"states carry {identity['sha256'][:12]}; a table on another coordinate is a table for another model"
            )
        self.zenith_by_identity = dict(zenith_by_identity)
        #: the entry's linear correction per row identity, ``(intercept, slope)``:
        #: the operator's output is returned as ``intercept + slope * bt`` so
        #: the filter's O-B is in the corrected frame the entry was graded in
        self.correction_by_identity: dict[str, tuple[float, float]] = {}
        self.rw_goes = find_rw_goes(rw_goes)
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="abi-forward-"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.month = int(month)
        self.threads = threads
        self.member_chunk = max(1, int(member_chunk))
        self.keep_files = keep_files
        self.receipts: list[dict] = []
        #: (truncation, identity) -> (reference temperature (nlay,), reference
        #: vapor g/kg (nlay,), jac_t (nlay,), jac_q (nlay,), reference bt):
        #: the forward's reading a later evaluation of the row linearises
        #: about (the members about their mean column, the analysed states
        #: about their backgrounds), the absorption held.
        self.references: dict = {}
        self.linearise_members = True
        self.check_rows = 64
        self.check_members = 2
        #: The linear-regime guard: a (state, row) pair whose tangent-linear
        #: departure from its reference exceeds ``multiple`` assigned errors
        #: (never under ``floor_k``) is evaluated with the full forward
        #: instead.  What it prevents: on the case day's run of record the
        #: band-8 residual of the linear form against the full forward read
        #: 1.22 K rms (largest 10.2 K) at 23Z and 2.02 K rms (largest 17.5 K)
        #: at 00Z on the checked subsample, above the band's 1.21 K error,
        #: where a few members' upper-tropospheric vapor sat e-folds from the
        #: reference and the logarithm of the ratio, summed over the layers,
        #: read what the forward itself never does.
        self.linear_regime_multiple = LINEAR_REGIME_ERROR_MULTIPLE
        self.linear_regime_floor_k = LINEAR_REGIME_FLOOR_K
        #: The window's linearisation record over every call (see
        #: :meth:`_window_record`); the stream reads it after the reset.
        self.last_linearisation_check: dict | None = None
        self._window_checks: list[dict] = []

    def reset(self) -> None:
        self.references.clear()
        self.receipts.clear()
        self._window_checks = []

    def bind(self, transform) -> "AbiRadianceOperator":
        """Register the transform of one resolution (the ensemble's, the
        control's); the states pick theirs by truncation."""
        self.transforms[int(transform.truncation)] = transform
        return self

    def transform_for(self, states):
        from .microwave.entry import truncation_of

        t = truncation_of(states)
        found = self.transforms.get(t)
        if found is None:
            raise AbiOperatorError(
                f"the ABI operator holds no transform at T{t} (bound: {sorted(self.transforms)}); bind the "
                "ensemble's and the control's transforms before the window opens"
            )
        return found

    def _host(self, transform, atmosphere, name):
        from .assimilate import _to_numpy_spectral
        return _to_numpy_spectral(transform.backend, getattr(atmosphere, name))

    def _grid_plane(self, transform, plane, lat, lon):
        from .assimilate import _sample_grid
        return _sample_grid(np.asarray(transform.backend.to_numpy(plane), dtype=np.float64), transform.grid, lat, lon)

    def member_columns(self, members, lat: np.ndarray, lon: np.ndarray, zenith: np.ndarray, *, transform=None) -> dict:
        """Every member's columns at the points, stacked ``(R * n)`` in
        member-major order, as the columns dict :func:`abi_reference.write_columns`
        takes; ``transform`` the members' own (chosen by their truncation
        when None).  On a device backend the coefficient stack of every
        member is one device array and the synthesis runs there."""
        from woof.globe.spectral.sampling import sample_scalar
        from .constants import KAPPA, REFERENCE_PRESSURE_PA

        transform = self.transform_for(members) if transform is None else transform
        xp = transform.backend.xp
        device = xp is not np

        def coeff(atmosphere, name):
            return getattr(atmosphere, name) if device else self._host(transform, atmosphere, name)

        chunk_size = len(members) if device else self.member_chunk
        nlev = self.a.size - 1
        n = lat.size
        r = len(members)
        temperature = np.zeros((r, n, nlev))
        q_gkg = np.zeros((r, n, nlev))
        p_half = np.zeros((r, n, nlev + 1))
        p_full = np.zeros((r, n, nlev))
        skin = np.zeros((r, n))
        land = np.zeros((r, n))
        cls = np.zeros((r, n), dtype=np.int32)
        seaice = np.zeros((r, n))
        snowh = np.zeros((r, n))
        for start in range(0, r, max(1, chunk_size)):
            chunk = members[start:start + max(1, chunk_size)]
            stack = xp.stack([
                xp.concatenate([coeff(m.atmosphere, "theta"), coeff(m.atmosphere, "qv"),
                                coeff(m.atmosphere, "log_surface_pressure")[None]])
                for m in chunk
            ])                                                        # (Rc, 2 nlev + 1, n, m)
            sampled = np.asarray(sample_scalar(transform, stack, lat, lon))   # (Rc, 2 nlev + 1, n)
            del stack
            for c, member in enumerate(chunk):
                k = start + c
                ps = np.exp(sampled[c, -1])
                ph = self.a[None, :] + self.b[None, :] * ps[:, None]  # (n, nlev + 1), top first
                pf = np.sqrt(ph[:, :-1] * ph[:, 1:])
                theta = sampled[c, :nlev].T                            # (n, nlev)
                temperature[k] = theta * (pf / REFERENCE_PRESSURE_PA) ** KAPPA
                q_gkg[k] = np.maximum(sampled[c, nlev:2 * nlev].T, 0.0) * 1000.0
                p_half[k] = ph / 100.0
                p_full[k] = pf / 100.0
                surface = member.surface
                skin[k] = self._grid_plane(transform, surface.temperature_k, lat, lon)
                land[k] = np.clip(self._grid_plane(transform, surface.land_fraction, lat, lon), 0.0, 1.0)
                cls[k] = np.rint(self._grid_plane(transform, surface.landuse_category, lat, lon)).astype(np.int32) \
                    if getattr(surface, "landuse_category", None) is not None else 10
                seaice[k] = np.clip(self._grid_plane(transform, surface.sea_ice_fraction, lat, lon), 0.0, 1.0) \
                    if getattr(surface, "sea_ice_fraction", None) is not None else 0.0
                if getattr(surface, "snow_depth_m", None) is not None:
                    snowh[k] = np.maximum(self._grid_plane(transform, surface.snow_depth_m, lat, lon), 0.0)
        flat = lambda arr: arr.reshape(r * n, *arr.shape[2:])  # noqa: E731
        return {
            "lat": np.tile(lat, r), "lon": np.tile(lon, r), "zenith": np.tile(zenith, r),
            "land_fraction": flat(land), "landuse_category": flat(cls), "sea_ice_fraction": flat(seaice),
            "skin_k": flat(skin), "psfc_hpa": p_half[:, :, -1].reshape(r * n), "wind10_ms": np.zeros(r * n),
            "snowh_m": flat(snowh), "p_half_hpa": flat(p_half), "p_full_hpa": flat(p_full),
            "temperature_k": flat(temperature), "q_gkg": flat(q_gkg),
            "o3_ppmv": np.stack([ref.ozone_ppmv(p_full.reshape(r * n, nlev)[i]) for i in range(r * n)]),
            "climatology": ref.climatology_for(np.tile(lat, r), self.month),
            "a_half_pa": self.a, "b_half": self.b,
            "emissivity_channels": [], "emissivity_user": np.zeros((r * n, 0)),
            "checks": {"members": r, "points": n},
        }

    def _forward(self, columns: dict, members: int, n: int, truncation: int) -> dict:
        """``rw_goes forward`` on a columns dict; the output stream read back."""
        stamp = f"{int(time.time() * 1000)}-{os.getpid()}-{truncation}"
        columns_path = self.work_dir / f"columns-{stamp}.bin"
        out_path = self.work_dir / f"forward-{stamp}.bin"
        ref.write_columns(columns_path, columns, provenance={"members": members, "points": n, "band": self.band})
        record = run_forward(columns_path, self.table_path, out_path, rw_goes=self.rw_goes, emis_mode=0, bands=(self.band,),
                             threads=self.threads)
        result = ref.read_crtm_output(out_path)
        result["_record"] = record
        if not self.keep_files:
            for path in (columns_path, Path(f"{columns_path}.json"), out_path):
                path.unlink(missing_ok=True)
        return result

    @staticmethod
    def _select(columns: dict, r: int, n: int, states, rows) -> dict:
        """The columns dict restricted to ``states`` x ``rows`` (member-major)."""
        out = {}
        for key, value in columns.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == r * n:
                arr = value.reshape(r, n, *value.shape[1:])[np.ix_(states, rows)]
                out[key] = arr.reshape(len(states) * len(rows), *value.shape[1:])
            else:
                out[key] = value
        out["checks"] = {"members": len(states), "points": len(rows)}
        return out

    @staticmethod
    def _mean(columns: dict, r: int, n: int) -> dict:
        """The member-mean columns dict (the surface classes of member 0)."""
        out = {}
        for key, value in columns.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == r * n:
                arr = value.reshape(r, n, *value.shape[1:])
                if np.issubdtype(arr.dtype, np.floating):
                    out[key] = arr.mean(axis=0)
                else:
                    out[key] = arr[0]
            else:
                out[key] = value
        out["checks"] = {"members": 1, "points": n}
        return out

    def __call__(self, members, batch) -> np.ndarray:
        """The batch contract: ``(R, batch.count)`` brightness temperatures:
        the full forward on the reference column of a row's first
        evaluation (the members' mean column, or a single state's own),
        the tangent-linear forward through the reference's temperature and
        vapor Jacobians after it (:attr:`references`; the vapor term in the
        relative form of :func:`vapor_tangent_linear`), the entry's linear
        correction applied to every value."""
        n = batch.count
        zenith = np.array([self.zenith_by_identity.get(str(i), np.nan) for i in batch.identity], dtype=np.float64)
        if np.any(~np.isfinite(zenith)):
            raise AbiOperatorError("a row of the brightness-temperature batch carries no satellite zenith; the batch "
                                   "must come from superobs_from_blocks with its extras registered")
        r = len(members)
        columns = self.member_columns(members, np.asarray(batch.latitude_deg), np.asarray(batch.longitude_deg), zenith)
        identities = [str(i) for i in batch.identity]
        bt = np.empty((r, n))
        wall = 0.0
        linearise = bool(getattr(self, "linearise_members", True)) and "temperature_k" in columns
        t = int(self.transform_for(members).truncation) if linearise else 0
        if not linearise:
            result = self._forward(columns, r, n, t)
            bt[:] = np.asarray(result["bt"][self.band], dtype=np.float64).reshape(r, n)
            wall = result["_record"]["_wall_s"]
            mode = "full"
        else:
            temperature = columns["temperature_k"].reshape(r, n, -1)
            q_gkg = columns["q_gkg"].reshape(r, n, -1)
            cached = np.array([(t, i) in self.references for i in identities], dtype=bool)
            reference_bt = np.full(n, np.nan)
            if (~cached).any():
                rows = np.flatnonzero(~cached)
                if r > 1:
                    reference = self._mean(self._select(columns, r, n, list(range(r)), rows), r, rows.size)
                else:
                    reference = self._select(columns, r, n, [0], rows)
                result = self._forward(reference, 1, rows.size, t)
                wall += result["_record"]["_wall_s"]
                bt_ref = np.asarray(result["bt"][self.band], dtype=np.float64).reshape(-1)
                jac_t = np.asarray(result["jac_t"][self.band], dtype=np.float64).reshape(rows.size, -1)
                jac_q = np.asarray(result["jac_q"][self.band], dtype=np.float64).reshape(rows.size, -1)
                t_ref = reference["temperature_k"].reshape(rows.size, -1)
                q_ref = reference["q_gkg"].reshape(rows.size, -1)
                for j, row in enumerate(rows):
                    self.references[(t, identities[row])] = (t_ref[j].copy(), q_ref[j].copy(), jac_t[j].copy(), jac_q[j].copy(),
                                                             float(bt_ref[j]))
                reference_bt[rows] = bt_ref
                if r == 1:
                    bt[0, rows] = bt_ref
                else:
                    for k in range(r):
                        bt[k, rows] = bt_ref + np.sum(jac_t * (temperature[k, rows] - t_ref), axis=1) \
                            + vapor_tangent_linear(jac_q, q_ref, q_gkg[k, rows])
            if cached.any():
                rows = np.flatnonzero(cached)
                refs = [self.references[(t, identities[row])] for row in rows]
                t_ref = np.stack([v[0] for v in refs])
                q_ref = np.stack([v[1] for v in refs])
                jac_t = np.stack([v[2] for v in refs])
                jac_q = np.stack([v[3] for v in refs])
                bt_ref = np.array([v[4] for v in refs])
                reference_bt[rows] = bt_ref
                for k in range(r):
                    bt[k, rows] = bt_ref + np.sum(jac_t * (temperature[k, rows] - t_ref), axis=1) \
                        + vapor_tangent_linear(jac_q, q_ref, q_gkg[k, rows])
            mode = "full" if (r == 1 and not cached.any()) else "linearised"
            pairs_full = 0
            max_departure = 0.0
            if mode == "linearised":
                # The linear-regime guard: a pair whose linear form has left
                # its reference by more than the guard's limit takes the full
                # forward (one call per state over that state's rows).
                departure = bt - reference_bt[None, :]
                multiple = float(getattr(self, "linear_regime_multiple", LINEAR_REGIME_ERROR_MULTIPLE))
                floor = float(getattr(self, "linear_regime_floor_k", LINEAR_REGIME_FLOOR_K))
                error = np.asarray(getattr(batch, "error", np.full(n, np.nan)), dtype=np.float64)
                limit = np.where(np.isfinite(error), np.maximum(multiple * error, floor), floor)
                finite = np.isfinite(departure)
                max_departure = float(np.max(np.abs(departure[finite]))) if finite.any() else 0.0
                beyond = finite & (np.abs(departure) > limit[None, :])
                for k in np.flatnonzero(beyond.any(axis=1)):
                    rows_k = np.flatnonzero(beyond[k])
                    full = self._forward(self._select(columns, r, n, [int(k)], rows_k), 1, rows_k.size, t)
                    wall += full["_record"]["_wall_s"]
                    bt[k, rows_k] = np.asarray(full["bt"][self.band], dtype=np.float64).reshape(-1)
                    pairs_full += int(rows_k.size)
            if r > 1:
                rng = np.random.default_rng(n * 7919 + r)
                check_rows = np.sort(rng.choice(n, size=min(int(self.check_rows), n), replace=False))
                check_members = np.sort(rng.choice(r, size=min(int(self.check_members), r), replace=False))
                full = self._forward(self._select(columns, r, n, list(check_members), check_rows), len(check_members),
                                     check_rows.size, t)
                wall += full["_record"]["_wall_s"]
                full_bt = np.asarray(full["bt"][self.band], dtype=np.float64).reshape(len(check_members), check_rows.size)
                diff = (full_bt - bt[np.ix_(check_members, check_rows)]).ravel()
                if not hasattr(self, "_window_checks"):
                    self._window_checks = []
                self._window_checks.append({
                    "members_checked": [int(k) for k in check_members], "rows_checked": int(check_rows.size),
                    "sum_sq_k2": float(np.sum(diff ** 2)), "count": int(diff.size), "max_abs_k": float(np.max(np.abs(diff))),
                    "pairs_full_forward": int(pairs_full), "max_abs_departure_k": float(max_departure),
                    "states": int(r), "rows": int(n),
                })
                self.last_linearisation_check = self._window_record()
        corrected = 0
        if self.correction_by_identity:
            a = np.zeros(n)
            b = np.ones(n)
            for j, identity in enumerate(batch.identity):
                pair = self.correction_by_identity.get(str(identity))
                if pair is not None:
                    a[j], b[j] = pair
                    corrected += 1
            bt = a[None, :] + b[None, :] * bt
        self.receipts.append({"members": len(members), "points": n, "rows_corrected": corrected, "wall_s": wall,
                              "mode": mode, "truncation": t,
                              "pairs_full_forward": int(pairs_full) if linearise else 0})
        return bt

    def _window_record(self) -> dict:
        """The window's linearisation record over every member evaluation
        of the band so far (the members' background, then their analysis):
        the rms and the largest residual of the delivered values against the
        full forward on the checked subsamples, the pairs the linear-regime
        guard sent to the full forward and the largest linear departure it
        saw."""
        checks = getattr(self, "_window_checks", [])
        total_sq = sum(float(c["sum_sq_k2"]) for c in checks)
        total_n = sum(int(c["count"]) for c in checks)
        members: set[int] = set()
        for c in checks:
            members.update(c["members_checked"])
        return {
            "rule": "the members through the tangent-linear forward about the reference column (the forward's "
                    "temperature and vapor Jacobians per layer, the vapor term in the relative form); a pair whose "
                    "linear form leaves its reference by more than the linear-regime limit (assigned errors times "
                    "the multiple, never under the floor) takes the full forward; the residual of the delivered "
                    "values against the full forward on a subsample of members and rows, over every call of the "
                    "window",
            "calls": len(checks),
            "members_checked": sorted(members), "rows_checked": int(sum(int(c["rows_checked"]) for c in checks)),
            "rms_k": float(np.sqrt(total_sq / total_n)) if total_n else None,
            "max_abs_k": max((float(c["max_abs_k"]) for c in checks), default=None),
            "pairs_full_forward": int(sum(int(c["pairs_full_forward"]) for c in checks)),
            "pairs_evaluated": int(sum(int(c["states"]) * int(c["rows"]) for c in checks)),
            "max_abs_departure_k": max((float(c["max_abs_departure_k"]) for c in checks), default=None),
            "linear_regime": {"multiple_of_error": float(getattr(self, "linear_regime_multiple", LINEAR_REGIME_ERROR_MULTIPLE)),
                              "floor_k": float(getattr(self, "linear_regime_floor_k", LINEAR_REGIME_FLOOR_K))},
        }


def band_weighting_layers(table: dict, band: int, *, vapor_fraction: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """``(layer_lnp, weights)`` of a band's sensitivity per unit ln p from
    the table's reference Jacobian summary: the mean temperature Jacobian
    per unit ln p (K per K) plus the mean vapor Jacobian (K per g/kg)
    times ``vapor_fraction`` of the reference column's vapor per unit ln p
    (the brightness temperature's response to a one kelvin and a ten
    percent change of a layer, the currency the two share)."""
    row = table["bands"][str(int(band))]["reference_jacobians"]
    p_hpa = np.asarray(row["mean_layer_pressure_hpa"], dtype=np.float64)
    lnp = np.log(p_hpa * 100.0)
    jt = np.abs(np.asarray(row["mean_temperature_jacobian_per_lnp"], dtype=np.float64))
    jq = np.abs(np.asarray(row["mean_vapor_jacobian_k_per_gkg"], dtype=np.float64))
    # A reference vapor profile (g/kg) per layer: the tropical standard
    # column of the microwave calibration, the same currency both operators use.
    from .microwave.calibrate import standard_column

    column = standard_column()
    q_gkg = np.interp(lnp, np.log(np.asarray(column.pressure_pa, dtype=np.float64))[::-1],
                      (np.asarray(column.specific_humidity, dtype=np.float64)[:, 0] * 1000.0)[::-1])
    edges = np.concatenate([[lnp[0] - 0.5 * (lnp[1] - lnp[0])], 0.5 * (lnp[1:] + lnp[:-1]), [lnp[-1] + 0.5 * (lnp[-1] - lnp[-2])]])
    thickness = np.maximum(np.diff(edges), 1e-6)
    weights = jt + jq * q_gkg * float(vapor_fraction) / thickness
    return lnp, weights


def band_profile(table: dict, band: int, cutoff_lnp: float) -> np.ndarray:
    """``(M,)`` the band's localisation profile on
    :data:`~woof.globe.da.observations.LOCALISATION_AXIS_LNP`."""
    from .da.localisation import profile_from_weighting_function

    lnp, weights = band_weighting_layers(table, band)
    return profile_from_weighting_function(lnp, weights, cutoff_lnp)


def band_centroid_lnp(table: dict, band: int) -> float:
    """The ln p centroid of the band's sensitivity (the row's
    ``ln_pressure`` for thinning and the receipt)."""
    from .da.localisation import profile_centroid_lnp

    lnp, weights = band_weighting_layers(table, band)
    return profile_centroid_lnp(lnp, weights)


def stream_name(satellite: str, band: int) -> str:
    """One stream name per satellite and band (``goes-abi-l1b-bt:G19:band13``)."""
    return f"{STREAM}:{satellite}:band{int(band):02d}"


def register_batch(operator: AbiRadianceOperator, batch, extras: dict) -> None:
    """Bind a superobs batch to the operator: its zenith angles by row
    identity (so subsets keep their angles), the entry's linear correction
    by row identity where the batch carries one, and the operator itself."""
    for identity, zenith in zip(batch.identity, extras["zenith_deg"]):
        operator.zenith_by_identity[str(identity)] = float(zenith)
    if "correction_intercept_k" in extras:
        for identity, a, b in zip(batch.identity, extras["correction_intercept_k"], extras["correction_slope"]):
            if a != 0.0 or b != 1.0:
                operator.correction_by_identity[str(identity)] = (float(a), float(b))
    batch.operator = operator


__all__ = ["STREAM", "VARIABLE", "AbiRadianceOperator", "band_centroid_lnp", "band_profile", "band_weighting_layers",
           "register_batch", "run_forward", "scan_row_time", "stream_name", "superobs_from_blocks"]


def superobs_from_pack(pack_path: str | os.PathLike, block_csv: str | os.PathLike, *, observation_error_k: float | None = None,
                       **kwargs):
    """The batch of :func:`superobs_from_blocks` with the scan window, the
    band and the bookkeeping read from the ``gpuwm-obs.goes-bt.v1`` pack
    the blocks were colocated against: scan start and end, and the pack's
    provenance row (publication time, producer identity, first receipt)
    returned in the extras so the receipt can carry them."""
    from .obs_pack import read_goes_pack

    pack = read_goes_pack(pack_path)
    meta = pack.meta if hasattr(pack, "meta") else pack
    start = dt.datetime.fromisoformat(str(meta["scan_start"]).replace("Z", "+00:00"))
    end = dt.datetime.fromisoformat(str(meta["scan_end"]).replace("Z", "+00:00"))
    batch, extras = superobs_from_blocks(block_csv, band=int(meta["band"]), observation_error_k=observation_error_k,
                                         scan_start=start, scan_end=end, **kwargs)
    provenance = dict(meta.get("provenance") or {})
    extras["provenance"] = provenance
    extras["latency_class"] = ("latency unverified" if not provenance.get("received_utc")
                               else "first receipt recorded")
    extras["pack"] = str(pack_path)
    return batch, extras


__all__ += ["superobs_from_pack"]
