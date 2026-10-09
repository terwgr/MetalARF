"""The radiance streams of the DA door: ATMS and ABI clear-sky brightness
temperatures as assimilated streams of the ensemble filter.

Two :data:`~woof.globe.da_streams.STREAM_TABLE` entries, ``atms``
and ``goes-abi``, each with the two halves a radiance stream needs and a
point stream does not:

``fetch(window_start, window_end, out_dir)``
    Land the window's granules with a manifest (URL, bytes, SHA-256, the
    producer's publication instant, the latency class), decode them
    through the Rust doors (``rw_atms decode`` and ``rw_atms thin`` onto
    the stream's thinning grid and the observation bin; ``rw_goes bt`` and
    ``rw_goes superobs`` onto 24-pixel blocks), and hand back the
    :class:`~woof.globe.da_streams.FetchRecord` list the cycle
    books.  Nothing here reads a granule: Python fetches, books and calls.

``batches(context)``
    The window's :class:`~woof.globe.da.observations.PointObs`
    batches (one per satellite and channel or band) from the thinned cells
    or blocks, screened against the CONTROL background at the window's
    opening (open ocean by its land and sea-ice fractions, clear by its
    column cloud water) and against the observations themselves (the
    Grody 23.8 / 31.4 GHz liquid-water retrieval for ATMS, the ACM clear
    mask and the cloud-edge gate for ABI, the within-cell spread), with the
    operator entry's error and bias correction, the row's localisation
    profile (the channel's weighting function convolved with the
    Gaspari-Cohn kernel of the vertical length MEASURED on the ensemble at
    this window, :mod:`woof.globe.da.localisation`) and the
    operator bound to the ensemble's and the control's transforms.  The
    receipt of every window (screen stages, rows per channel, the
    correlation half widths and cutoffs, the bias coefficients before and
    after the day's update, the linearisation check) rides in the
    analysis report under ``observation_times.radiance``.

The bias correction's day update: the entry's coefficients (fitted on a
day against GDAS columns) are the start; after every window the stream
reads its previous batches' departures against the control background
(``value - control_simulated``, the corrected frame), refits the
residual on the entry's predictors over the rows the gross check kept,
and moves the slope and scan terms toward the window's fit by ``n / (n +
memory)`` (``memory`` 2,000 rows: a 250-row window moves them a ninth
of the way).  The constant term (``a`` for ATMS, ``intercept_shift_k``
for ABI, :data:`ANCHORED_TERMS`) never moves: it is the anchor to the
reference columns the entry was fitted against, and a constant refitted
on the control's own departures would absorb the control's drift as
instrument bias, so the radiances could no longer correct it (the grade
of record's ledger read that constant moving 0.10 to 0.46 K on the ATMS
channels and 0.10 K on ABI band 8 in six windows, the ABI intercept at
0.6 of the way to each window's own mean).  The window's fit of the
constant is still recorded in every move (``fit``) so the drift is
readable; ``applied`` says which terms moved.  The ledger is written
beside the cycle (``radiance-bias-<stream>.json``) so a resumed cycle
carries it, and every move is in the receipt.

Whether the streams are in the door's default set is the scorecard's
verdict (the door page records it); the streams exist here whatever the
verdict, reachable by ``--stream atms`` and ``--stream goes-abi``.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .da_streams import FetchRecord, _digest, _stamp, _utc

ATMS_STREAM = "atms"
ABI_STREAM = "goes-abi"
TABLES_DIR = Path(__file__).resolve().parent / "radiance_tables"
RECEIPT_SCHEMA = "gpuwm.arwen-global-radiance-window/v1"
BIAS_LEDGER_SCHEMA = "gpuwm.arwen-global-radiance-bias/v1"

#: The day update's memory in rows (see the module doc): a window of 250
#: cells per satellite (the case's clear-sky ocean yield at one row per
#: ensemble cell) moves the coefficients an ninth of the way to its fit.
DEFAULT_BIAS_MEMORY_ROWS = 2000

#: The correction terms the day update never moves: the constant of the
#: ATMS geometry model (``a``) and the ABI intercept shift.  A constant
#: refitted on the control's own departures fits the model to itself (the
#: control's drift becomes instrument bias); the entry's constant, fitted
#: against reference columns, is the anchor.  The window's fit of the
#: constant is recorded, not applied.
ANCHORED_TERMS = ("a", "intercept_shift_k")

#: The model-side clear-sky gate: the control background's column cloud
#: water (kg m-2) at the cell, the microwave scorecard's analysis gate.
DEFAULT_MODEL_CLOUD_WATER_KG_M2 = 0.01

ABI_BUCKETS = {"G16": "noaa-goes16", "G18": "noaa-goes18", "G19": "noaa-goes19"}


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------------------
# the context the cycle hands a radiance stream
# ---------------------------------------------------------------------------

@dataclass
class RadianceContext:
    """What a radiance stream needs at a window's opening."""

    window_start: dt.datetime
    window_end: dt.datetime
    epoch: dt.datetime
    control_state: object
    control_model: object
    control_transform: object
    cfg: object
    ensemble: object
    out_dir: Path
    observation_bin_s: float | None = None

    @property
    def bin_s(self) -> float:
        span = (self.window_end - self.window_start).total_seconds()
        return float(span if self.observation_bin_s is None else self.observation_bin_s)

    @property
    def ensemble_transform(self):
        return self.ensemble.transform

    @property
    def vertical(self):
        return self.ensemble.model.vertical


def control_surface_at(context: RadianceContext, latitude_deg, longitude_deg) -> dict[str, np.ndarray]:
    """The control background's surface reads at points (bilinear): land
    and sea-ice fractions, skin temperature and the column cloud water
    (every condensate species integrated over the column, kg m-2)."""
    from .assimilate import _sample_grid
    from .constants import CONDENSATE_SPECIES, GRAVITY_M_S2

    model = context.control_model
    transform = context.control_transform
    backend = transform.backend
    grid = transform.grid
    state = context.control_state
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.asarray(longitude_deg, dtype=np.float64)
    g = model.grid_state(state.atmosphere, only=(*CONDENSATE_SPECIES, "dp"))
    xp = backend.xp
    condensate = sum(xp.maximum(g[name], 0.0) for name in CONDENSATE_SPECIES)
    cwp = np.asarray(backend.to_numpy((condensate * g["dp"]).sum(axis=0) / GRAVITY_M_S2), dtype=np.float64)
    model.release_syntheses()
    surface = state.surface
    out = {
        "land_fraction": np.clip(_sample_grid(np.asarray(backend.to_numpy(surface.land_fraction), dtype=np.float64), grid, lat, lon), 0.0, 1.0),
        "skin_temperature_k": _sample_grid(np.asarray(backend.to_numpy(surface.temperature_k), dtype=np.float64), grid, lat, lon),
        "cloud_water_path": np.maximum(_sample_grid(cwp, grid, lat, lon), 0.0),
    }
    ice = getattr(surface, "sea_ice_fraction", None)
    out["sea_ice_fraction"] = (np.zeros(lat.size) if ice is None
                               else np.clip(_sample_grid(np.asarray(backend.to_numpy(ice), dtype=np.float64), grid, lat, lon), 0.0, 1.0))
    return out


def thin_to_grid(latitude_deg, longitude_deg, grid, *, score=None) -> np.ndarray:
    """The indices of one row per grid cell of ``grid`` (the ensemble's
    Gaussian grid): the row nearest the cell's centre, or the row with the
    largest ``score`` when given.  What it prevents: the filter thins to one
    report per ensemble cell per stream anyway, so every row beyond it was
    evaluated on every member for nothing (a window's blocks and cells were
    four to five times the cells they landed in)."""
    lat = np.asarray(latitude_deg, dtype=np.float64)
    lon = np.mod(np.asarray(longitude_deg, dtype=np.float64), 360.0)
    lat_nodes = np.asarray(grid.latitude_deg, dtype=np.float64)
    order = np.argsort(lat_nodes)
    j = order[np.abs(lat_nodes[order][:, None] - lat[None, :]).argmin(axis=0)]
    dlon = 360.0 / int(grid.nlon)
    i = np.mod(np.round(lon / dlon).astype(int), int(grid.nlon))
    cell_lat = lat_nodes[j]
    cell_lon = i * dlon
    distance = np.hypot(lat - cell_lat, (np.mod(lon - cell_lon + 180.0, 360.0) - 180.0) * np.cos(np.deg2rad(cell_lat)))
    key = j.astype(np.int64) * int(grid.nlon) + i
    best: dict[int, int] = {}
    for idx in range(lat.size):
        held = best.get(int(key[idx]))
        better = (held is None or (distance[idx] < distance[held] if score is None else score[idx] > score[held]))
        if better:
            best[int(key[idx])] = idx
    return np.array(sorted(best.values()), dtype=int)


def ensemble_vertical_lengths(context: RadianceContext, targets_lnp: dict, *, ocean_only: bool = True) -> dict:
    """The ensemble's vertical correlation half width and the cutoff it
    implies, per target (``{key: target_lnp}``), measured once on the
    members' potential temperature at the window's opening over the
    ocean columns (:func:`woof.globe.da.localisation.vertical_correlation_length`)."""
    from .da.localisation import vertical_correlation_length

    ensemble = context.ensemble
    fields, ln_p_full, _ln_ps = ensemble.grid_fields(("theta",))
    theta = fields["theta"]
    grid = ensemble.transform.grid
    mask = None
    if ocean_only:
        backend = ensemble.transform.backend
        land = np.asarray(backend.to_numpy(ensemble.members[0].surface.land_fraction), dtype=np.float64)
        mask = land < 0.5
    out = {}
    for key, target in targets_lnp.items():
        out[key] = vertical_correlation_length(theta, ln_p_full, float(target), latitude_deg=grid.latitude_deg,
                                               column_mask=mask)
    del fields, theta
    return out


# ---------------------------------------------------------------------------
# the bias ledger (the day update)
# ---------------------------------------------------------------------------

@dataclass
class BiasLedger:
    """The live bias coefficients of one stream and their history."""

    stream: str
    path: Path
    memory_rows: int = DEFAULT_BIAS_MEMORY_ROWS
    coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, stream: str, path: Path, initial: dict[str, dict[str, float]], *, memory_rows: int) -> "BiasLedger":
        ledger = cls(stream=stream, path=Path(path), memory_rows=int(memory_rows),
                     coefficients={k: dict(v) for k, v in initial.items()})
        if ledger.path.is_file():
            payload = json.loads(ledger.path.read_text(encoding="utf-8"))
            if payload.get("schema") == BIAS_LEDGER_SCHEMA and payload.get("stream") == stream:
                for key, coefficients in payload.get("coefficients", {}).items():
                    if key in ledger.coefficients:
                        ledger.coefficients[key].update({k: float(v) for k, v in coefficients.items()})
                ledger.history = list(payload.get("history", []))
        return ledger

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": BIAS_LEDGER_SCHEMA, "stream": self.stream, "memory_rows": int(self.memory_rows),
                   "coefficients": self.coefficients, "history": self.history[-200:],
                   "written_utc": _stamp(_now())}
        tmp = self.path.with_name(f".{self.path.name}.partial-{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def move(self, key: str, fit: dict[str, float], rows: int, *, label: str, before: dict, after: dict) -> dict:
        """Move ``key``'s slope and scan terms toward ``fit`` (the window's
        own residual fit) by ``rows / (rows + memory)``; the constant
        (:data:`ANCHORED_TERMS`) is recorded in the fit and not moved. The
        record returned and appended to the history names what moved
        (``applied``) and what stayed (``anchored``)."""
        weight = float(rows) / float(rows + self.memory_rows) if rows > 0 else 0.0
        old = dict(self.coefficients[key])
        applied: dict[str, float] = {}
        anchored: list[str] = []
        for name, value in fit.items():
            if name not in self.coefficients[key]:
                continue
            if name in ANCHORED_TERMS:
                anchored.append(name)
                continue
            self.coefficients[key][name] = float(self.coefficients[key][name] + weight * float(value))
            applied[name] = float(weight * float(value))
        record = {"key": key, "label": label, "rows": int(rows), "weight": weight, "fit": dict(fit),
                  "applied": applied, "anchored": anchored,
                  "before": old, "after": dict(self.coefficients[key]),
                  "residual_before": before, "residual_after": after}
        self.history.append(record)
        return record


def _fit_residual(residual: np.ndarray, background_k: np.ndarray, sec_minus_one: np.ndarray, mean_background_k: float,
                  *, with_slope: bool = True) -> tuple[dict[str, float], np.ndarray]:
    """Least squares of the residual on the entry's predictors (constant,
    ``B - mean_B``, ``sec z - 1``); the fit and the residual after it."""
    n = residual.size
    if with_slope:
        design = np.stack([np.ones(n), background_k - mean_background_k, sec_minus_one], axis=1)
        names = ("a", "b", "c")
    else:
        design = np.ones((n, 1))
        names = ("a",)
    if n < design.shape[1] + 2:
        return {}, residual
    coefficients, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
    if rank < design.shape[1]:
        coefficients = np.zeros(design.shape[1])
        coefficients[0] = float(np.mean(residual))
    return {name: float(v) for name, v in zip(names, coefficients)}, residual - design @ coefficients


def _statistics(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"count": 0}
    return {"count": int(v.size), "mean": float(np.mean(v)), "rms": float(np.sqrt(np.mean(v ** 2)))}


# ---------------------------------------------------------------------------
# ATMS
# ---------------------------------------------------------------------------

def atms_entry_path(satellite: str, tables_dir: Path = TABLES_DIR) -> Path | None:
    """The shipped operator entry of a satellite (``atms-<satellite>-<day>.entry.json``)."""
    found = sorted(Path(tables_dir).glob(f"atms-{satellite}-*.entry.json"))
    return found[-1] if found else None


@dataclass
class AtmsStream:
    """ATMS clear-sky over-ocean brightness temperatures (channels 4 to 14)
    through ``rw_atms``; one batch per satellite and channel."""

    satellites: tuple[str, ...] = ("noaa-20", "noaa-21")
    cache: str | None = None
    entry_paths: dict[str, str] = field(default_factory=dict)
    channels: tuple[int, ...] | None = None
    thin_deg: float = 0.5
    max_zenith_deg: float = 60.0
    max_abs_latitude_deg: float = 60.0
    min_beams_per_cell: int = 3
    max_retrieved_lwp_kg_m2: float = 0.02
    max_model_cloud_water_kg_m2: float = DEFAULT_MODEL_CLOUD_WATER_KG_M2
    max_land_fraction: float = 0.01
    max_sea_ice_fraction: float = 0.01
    max_sounding_cell_std_k: float = 1.5
    bias_memory_rows: int = DEFAULT_BIAS_MEMORY_ROWS
    day_update: bool = True
    workers: int = 8
    name: str = ATMS_STREAM
    description: str = ("ATMS clear-sky open-ocean brightness temperatures (NOAA-20 and NOAA-21, channels 4 to 14) "
                        "from the NOAA JPSS open-data buckets through rw_atms, thinned to the observation bin")
    _entries: dict = field(default_factory=dict, repr=False)
    _operators: dict = field(default_factory=dict, repr=False)
    _ledgers: dict = field(default_factory=dict, repr=False)
    _previous: dict = field(default_factory=dict, repr=False)
    _windows: dict = field(default_factory=dict, repr=False)
    last_receipt: dict | None = field(default=None, repr=False)

    # -- entries -----------------------------------------------------------

    def entry(self, satellite: str):
        from .microwave.entry import read_entry

        if satellite not in self._entries:
            path = self.entry_paths.get(satellite)
            path = Path(path) if path else atms_entry_path(satellite)
            if path is None or not Path(path).is_file():
                raise FileNotFoundError(
                    f"no measured ATMS operator entry for {satellite}: none under {TABLES_DIR} and none named "
                    f"(entry_paths); the entry is the microwave score door's output for a day of that satellite, "
                    "and a channel error and bias correction measured on another satellite are not this one's"
                )
            entry = read_entry(path)
            if entry is None:
                raise ValueError(f"{path} refuses every channel; nothing to assimilate for {satellite}")
            self._entries[satellite] = (entry, Path(path))
        return self._entries[satellite]

    # -- fetch ---------------------------------------------------------------

    def _cache_dir(self, out_dir: Path) -> Path:
        return Path(self.cache) if self.cache else Path(out_dir) / "atms"

    def fetch(self, window_start, window_end, out_dir: Path) -> list[FetchRecord]:
        from .microwave.atms_fetch import fetch_window

        out_dir = Path(out_dir)
        records = []
        for satellite in self.satellites:
            start = time.perf_counter()
            self.entry(satellite)
            manifest = fetch_window(satellite, _utc(window_start), _utc(window_end),
                                    self._cache_dir(out_dir) / satellite, workers=int(self.workers))
            path = Path(manifest["manifest_path"])
            size, sha = _digest(path)
            downloaded = [f for f in manifest["files"] if f.get("download_seconds", 0.0) > 0.0]
            latency = None
            if downloaded:
                latency = (_now() - _utc(window_end)).total_seconds()
            publications = [f.get("s3_last_modified") for f in manifest["files"] if f.get("s3_last_modified")]
            records.append(FetchRecord(
                stream=self.name, location=f"https://{manifest['bucket']}.s3.amazonaws.com/{manifest['prefixes'][0]}",
                path=str(path), bytes=int(manifest["total_bytes"]), sha256=sha,
                window_start_utc=_stamp(window_start), window_end_utc=_stamp(window_end),
                fetched_utc=_stamp(_now()), wall_s=time.perf_counter() - start,
                latency_behind_real_time_s=latency, decoder="rw_atms",
                publication_utc=max(publications) if publications else None,
                extra={"satellite": satellite, "granule_pairs": int(manifest["granule_pairs"]),
                       "downloaded_now": len(downloaded), "manifest_sha256": sha,
                       "publication_latency_s": manifest.get("publication_latency_s")},
            ))
            self._windows[(satellite, _stamp(window_end))] = manifest
        return records

    # -- batches -------------------------------------------------------------

    def _thinned(self, satellite: str, manifest: dict, context: RadianceContext):
        """Decode the window's granule pairs and thin them onto the
        stream's rings and the observation bins, cached per window."""
        from .microwave.atms_bridge import decode, read_thinned, thin

        stamp = _utc(context.window_end).strftime("%Y%m%dT%H%M%SZ")
        root = self._cache_dir(context.out_dir) / satellite
        decoded_dir = root / "decoded" / stamp
        thinned_dir = root / "thinned" / f"{stamp}-{self.thin_deg:g}deg-{int(context.bin_s)}s"
        pairs = [(Path(f["sdr"]), Path(f["geo"])) for f in manifest["pairs"]]
        if not pairs:
            return None, {"granule_pairs": 0}
        if not (decoded_dir / "metadata.json").is_file():
            decode(pairs, decoded_dir)
        if not (thinned_dir / "metadata.json").is_file():
            step = float(self.thin_deg)
            rings = 90.0 - step * np.arange(int(round(180.0 / step)) + 1)
            thin(decoded_dir, thinned_dir, latitudes_deg=rings, nlon=int(round(360.0 / step)),
                 origin_unix_s=_utc(context.window_start).timestamp(), bin_s=float(context.bin_s),
                 max_zenith_deg=float(self.max_zenith_deg))
        thinned = read_thinned(thinned_dir)
        return thinned, {"granule_pairs": len(pairs), "decoded": str(decoded_dir), "thinned": str(thinned_dir),
                         "cells": int(thinned.ncell), "rings": int(thinned.metadata["ring_count"]),
                         "nlon": int(thinned.metadata["nlon"]), "bin_s": float(thinned.metadata["bin_s"])}

    def batches(self, context: RadianceContext) -> tuple[list, dict]:
        from types import SimpleNamespace

        from .microwave.channels import TEMPERATURE_SOUNDING_CHANNELS
        from .microwave.entry import AtmsBatchOperator, point_obs_from_cells, stream_name
        from .microwave.score import grody_cloud_liquid_water

        out_dir = Path(context.out_dir)
        receipt = {"schema": RECEIPT_SCHEMA, "stream": self.name, "satellites": {},
                   "window_start_utc": _stamp(context.window_start), "window_end_utc": _stamp(context.window_end)}
        batches = []
        lengths_cache: dict[int, dict] = {}
        for satellite in self.satellites:
            entry, entry_path = self.entry(satellite)
            manifest = self._windows.get((satellite, _stamp(context.window_end)))
            if manifest is None:
                raise ValueError(f"the atms stream was asked for batches of a window it never fetched ({satellite})")
            thinned, thin_record = self._thinned(satellite, manifest, context)
            sat_record = {"entry": str(entry_path), "entry_name": entry.name, "thinning": thin_record, "screen": {}}
            receipt["satellites"][satellite] = sat_record
            if thinned is None:
                sat_record["screen"] = {"cells": 0}
                continue
            # The screen, every stage counted.
            t = np.asarray(thinned.time_mean_unix_s, dtype=np.float64)
            t0 = _utc(context.window_start).timestamp()
            t1 = _utc(context.window_end).timestamp()
            lat = np.asarray(thinned.lat_mean_deg, dtype=np.float64)
            stages: dict[str, int] = {"cells": int(thinned.ncell)}
            keep = (t > t0) & (t <= t1)
            stages["in_window"] = int(keep.sum())
            keep &= (np.abs(lat) <= self.max_abs_latitude_deg)
            keep &= np.asarray(thinned.count) >= int(self.min_beams_per_cell)
            keep &= np.asarray(thinned.zenith_mean_deg, dtype=np.float64) <= float(self.max_zenith_deg)
            stages["candidate"] = int(keep.sum())
            idx = np.flatnonzero(keep)
            if idx.size:
                surface = control_surface_at(context, lat[idx], np.asarray(thinned.lon_mean_deg, dtype=np.float64)[idx])
                ocean = (surface["land_fraction"] <= self.max_land_fraction) & (surface["sea_ice_fraction"] <= self.max_sea_ice_fraction)
                idx = idx[ocean]
                stages["ocean"] = int(idx.size)
                clear = surface["cloud_water_path"][ocean] <= float(self.max_model_cloud_water_kg_m2)
                idx = idx[clear]
                stages["model_clear"] = int(idx.size)
            else:
                stages["ocean"] = 0
                stages["model_clear"] = 0
            tb = np.asarray(thinned.tb_mean_k, dtype=np.float64)
            if idx.size:
                lwp = grody_cloud_liquid_water(tb[idx, 0], tb[idx, 1], np.asarray(thinned.zenith_mean_deg, dtype=np.float64)[idx])
                obs_clear = np.isfinite(lwp) & (lwp <= float(self.max_retrieved_lwp_kg_m2)) & (tb[idx, 0] > tb[idx, 1])
                idx = idx[obs_clear]
            stages["obs_clear"] = int(idx.size)
            if idx.size:
                keep = thin_to_grid(lat[idx], np.asarray(thinned.lon_mean_deg, dtype=np.float64)[idx],
                                    context.ensemble_transform.grid, score=np.asarray(thinned.count)[idx])
                idx = idx[keep]
            stages["one_per_ensemble_cell"] = int(idx.size)
            sat_record["screen"] = stages
            if idx.size == 0:
                continue
            channels = [int(c) for c in (self.channels or entry.admitted_channels) if int(c) in entry.admitted_channels]
            # The vertical lengths, measured on the ensemble at this window per
            # channel centroid (one synthesis of the members' theta for every
            # centroid not yet measured this window).
            targets = {c: math.log(entry.channel_entry(c).peak_pressure_pa) for c in channels}
            missing = {c: target for c, target in targets.items() if int(round(target * 100)) not in lengths_cache}
            if missing:
                for c, read in ensemble_vertical_lengths(context, missing).items():
                    lengths_cache[int(round(targets[c] * 100))] = read
            lengths = {c: lengths_cache[int(round(target * 100))] for c, target in targets.items()}
            cutoffs = {c: float(lengths[c]["cutoff_lnp"]) for c in channels}
            sat_record["vertical_localisation"] = {
                str(c): {"centroid_pa": float(entry.channel_entry(c).peak_pressure_pa), **lengths[c]} for c in channels}
            # The day update from the previous window's departures (read
            # through the previous window's operator geometry), then the live
            # coefficients this window's batches carry: the ledger's.
            ledger = self._ledger(satellite, entry, out_dir)
            operator = self._operators.get(satellite)
            self._previous.setdefault(satellite, [])
            day_update_record = None
            if self.day_update and operator is not None and self._previous[satellite]:
                day_update_record = self._update_bias(satellite, entry, ledger, operator, self._previous[satellite])
            coefficients = {int(c): dict(ledger.coefficients[str(c)]) for c in channels}
            if operator is None:
                operator = AtmsBatchOperator(entry=entry, vertical=context.vertical, geometry={}, coefficients=coefficients)
                self._operators[satellite] = operator
            else:
                operator.coefficients.update(coefficients)
                operator.reset()
                operator.geometry.clear()
            operator.bind(context.ensemble_transform)
            operator.bind(context.control_transform)
            # Per-channel spread gate: a cell straddling a front or a cloud edge is not clear.
            cells = SimpleNamespace(
                lat_mean_deg=lat[idx], lon_mean_deg=np.asarray(thinned.lon_mean_deg, dtype=np.float64)[idx],
                zenith_mean_deg=np.asarray(thinned.zenith_mean_deg, dtype=np.float64)[idx],
                scan_angle_abs_mean_deg=np.asarray(thinned.scan_angle_abs_mean_deg, dtype=np.float64)[idx],
                time_mean_unix_s=t[idx], tb_mean_k=tb[idx].copy(),
                cell_bin=np.floor(t[idx] / float(context.bin_s)).astype(np.int64),
                cell_j=np.asarray(thinned.cell_j)[idx], cell_i=np.asarray(thinned.cell_i)[idx],
                count=np.asarray(thinned.count)[idx],
            )
            std = np.asarray(thinned.tb_std_k, dtype=np.float64)[idx]
            spread_refused = {}
            for c in channels:
                limit = self.max_sounding_cell_std_k if c in TEMPERATURE_SOUNDING_CHANNELS else 3.0
                bad = std[:, c - 1] > limit
                spread_refused[str(c)] = int(bad.sum())
                cells.tb_mean_k[bad, c - 1] = np.nan
            sat_record["spread_refused"] = spread_refused
            new = point_obs_from_cells(entry, cells, transforms=[context.ensemble_transform, context.control_transform],
                                       vertical=context.vertical, channels=channels, satellite=satellite,
                                       cutoffs_lnp=cutoffs, coefficients=coefficients, operator=operator)
            sat_record["rows_per_channel"] = {b.stream.rsplit(":", 1)[-1]: int(b.count) for b in new}
            sat_record["rows"] = int(sum(b.count for b in new))
            sat_record["bias_coefficients"] = {str(c): dict(operator.coefficients[c]) for c in channels}
            sat_record["linearisation_previous_window"] = operator.last_linearisation_check
            if day_update_record is not None:
                sat_record["day_update"] = day_update_record
            self._previous[satellite] = new
            batches.extend(new)
        self.last_receipt = receipt
        return batches, receipt

    def _ledger(self, satellite: str, entry, out_dir: Path) -> BiasLedger:
        if satellite not in self._ledgers:
            initial = {str(c.channel): dict(c.bias_coefficients) for c in entry.channels}
            self._ledgers[satellite] = BiasLedger.load(
                f"{self.name}:{satellite}", Path(out_dir) / f"radiance-bias-{self.name}-{satellite}.json", initial,
                memory_rows=int(self.bias_memory_rows))
        return self._ledgers[satellite]

    def _update_bias(self, satellite: str, entry, ledger: BiasLedger, operator, previous: list) -> dict:
        """The day update: refit the previous window's residuals against
        the control background per channel and move the coefficients."""
        moves = {}
        for batch in previous:
            if batch.control_simulated is None or batch.count == 0:
                continue
            channel = int(batch.stream.rsplit("ch", 1)[-1])
            c = ledger.coefficients[str(channel)]
            hx = np.asarray(batch.control_simulated, dtype=np.float64)[0]
            finite = np.isfinite(hx)
            if finite.sum() < 30:
                continue
            value = batch.value[finite]
            hx = hx[finite]
            zenith = np.array([operator.geometry[str(i)][1] for i in batch.identity[finite]])
            sec = 1.0 / np.cos(np.deg2rad(zenith))
            # The raw background from the corrected one: corrected = raw (1 + b) + a - b mean_B + c (sec - 1).
            raw = (hx - c["a"] + c["b"] * c["mean_background_k"] - c["c"] * (sec - 1.0)) / (1.0 + c["b"])
            residual = value - hx
            gross = np.abs(residual) <= 3.0 * float(entry.channel_entry(channel).error_k)
            if gross.sum() < 30:
                continue
            fit, after = _fit_residual(residual[gross], raw[gross], sec[gross] - 1.0, c["mean_background_k"])
            if not fit:
                continue
            moves[str(channel)] = ledger.move(str(channel), fit, int(gross.sum()), label=batch.stream,
                                              before=_statistics(residual[gross]), after=_statistics(after))
        if moves:
            ledger.write()
        return {"rule": "the slope and scan terms moved toward the previous window's residual fit by rows / (rows + memory); "
                        "the constant is anchored to the entry's fit and its window fit is recorded, not applied",
                "memory_rows": int(ledger.memory_rows), "anchored": list(ANCHORED_TERMS), "channels": moves}


# ---------------------------------------------------------------------------
# ABI
# ---------------------------------------------------------------------------

#: A bucket request is retried this many times with a growing pause: the
#: public buckets reset a connection now and then (the offline reading of
#: the case died on one at its third window), and a cycle that has run an
#: hour must not die on a transient the next attempt would not see.
FETCH_ATTEMPTS = 4


def _s3_list(bucket: str, prefix: str, *, timeout_s: float = 60.0, attempts: int = FETCH_ATTEMPTS) -> list[dict]:
    import urllib.parse
    import xml.etree.ElementTree as ET

    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "max-keys": "1000"})
    request = urllib.request.Request(f"https://{bucket}.s3.amazonaws.com/?{query}", headers={"User-Agent": "gpuwm-abi-fetch/1"})
    last_error: Exception | None = None
    for attempt in range(int(attempts)):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                root = ET.fromstring(response.read())
            break
        except Exception as error:  # noqa: BLE001 - retried, then raised with the cause
            last_error = error
            time.sleep(2.0 * (attempt + 1))
    else:
        raise ConnectionError(f"listing {bucket}/{prefix} failed after {attempts} attempts: {last_error}")
    out = []
    for contents in root.iter(f"{ns}Contents"):
        out.append({"key": contents.findtext(f"{ns}Key"), "size": int(contents.findtext(f"{ns}Size") or 0),
                    "last_modified": contents.findtext(f"{ns}LastModified"),
                    "etag": (contents.findtext(f"{ns}ETag") or "").strip('"')})
    return out


def _download(url: str, destination: Path, expected: int, *, timeout_s: float = 300.0,
              attempts: int = FETCH_ATTEMPTS) -> tuple[str, float]:
    if destination.exists() and destination.stat().st_size == expected:
        return _digest(destination)[1], 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "gpuwm-abi-fetch/1"})
    last_error: Exception | None = None
    for attempt in range(int(attempts)):
        digest = hashlib.sha256()
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(1 << 18), b""):
                    digest.update(block)
                    handle.write(block)
            if partial.stat().st_size != expected:
                raise ValueError(f"{url} delivered {partial.stat().st_size} bytes, the listing said {expected}")
            partial.replace(destination)
            return digest.hexdigest(), time.monotonic() - started
        except Exception as error:  # noqa: BLE001 - retried, then raised with the cause
            last_error = error
            time.sleep(2.0 * (attempt + 1))
    raise ConnectionError(f"{url} failed after {attempts} attempts: {last_error}")


def abi_scan_prefix(product: str, satellite: str, moment: dt.datetime, band: int | None) -> str:
    """The bucket prefix of the full-disk scan starting at ``moment``'s
    hour and ten-minute slot (mode 6: a full disk every ten minutes)."""
    moment = _utc(moment)
    day = moment.timetuple().tm_yday
    token = f"OR_{product}-M6" + (f"C{band:02d}" if band else "") + f"_{satellite}_s{moment:%Y}{day:03d}{moment:%H}{moment.minute // 10}"
    return f"{product}/{moment:%Y}/{day:03d}/{moment:%H}/{token}"


@dataclass
class AbiStream:
    """GOES ABI clear-sky brightness temperatures over water (bands 13 and
    8) through ``rw_goes bt`` and ``rw_goes superobs``; one batch per
    satellite and band, from the full-disk scan at the top of the window."""

    satellites: tuple[str, ...] = ("G19",)
    bands: tuple[int, ...] = (13, 8)
    cache: str | None = None
    entries_path: str | None = None
    table_path: str | None = None
    rw_goes: str | None = None
    block_pixels: int = 24
    scan_minute: int = 0
    max_model_cloud_water_kg_m2: float = DEFAULT_MODEL_CLOUD_WATER_KG_M2
    max_sea_ice_fraction: float = 0.01
    bias_memory_rows: int = DEFAULT_BIAS_MEMORY_ROWS
    day_update: bool = True
    threads: int | None = None
    name: str = ABI_STREAM
    description: str = ("GOES ABI band 13 and band 8 clear-sky brightness temperatures over water, the full-disk scan "
                        "at the top of the window through rw_goes bt and rw_goes superobs (24-pixel blocks)")
    _entries: dict | None = field(default=None, repr=False)
    _table: dict | None = field(default=None, repr=False)
    _windows: dict = field(default_factory=dict, repr=False)
    _operators: dict = field(default_factory=dict, repr=False)
    _ledgers: dict = field(default_factory=dict, repr=False)
    _previous: dict = field(default_factory=dict, repr=False)
    last_receipt: dict | None = field(default=None, repr=False)

    def _paths(self) -> tuple[Path, Path]:
        entries = Path(self.entries_path) if self.entries_path else TABLES_DIR / "abi-operator-entries.json"
        table = Path(self.table_path) if self.table_path else TABLES_DIR / "abi-fast-model.json"
        for path, what in ((entries, "operator entries"), (table, "coefficient table")):
            if not path.is_file():
                raise FileNotFoundError(f"the ABI {what} {path} does not exist")
        return entries, table

    def entries(self) -> dict:
        if self._entries is None:
            entries, _ = self._paths()
            payload = json.loads(entries.read_text(encoding="utf-8"))
            if payload.get("schema") != "gpuwm-da.abi-operator-entries.v1":
                raise ValueError(f"{entries} is not a gpuwm-da.abi-operator-entries.v1 file")
            self._entries = payload
        return self._entries

    def table(self) -> dict:
        from . import abi_fast_model as fm

        if self._table is None:
            _, table = self._paths()
            self._table = fm.read_table(table)
        return self._table

    def _cache_dir(self, out_dir: Path) -> Path:
        return Path(self.cache) if self.cache else Path(out_dir) / "goes"

    def fetch(self, window_start, window_end, out_dir: Path) -> list[FetchRecord]:
        from .abi_operator import find_rw_goes

        rw_goes = find_rw_goes(self.rw_goes)
        out_dir = Path(out_dir)
        records = []
        scan_moment = _utc(window_start) + dt.timedelta(minutes=int(self.scan_minute))
        for satellite in self.satellites:
            bucket = ABI_BUCKETS.get(satellite)
            if bucket is None:
                raise ValueError(f"no bucket for satellite {satellite!r}; the table carries {sorted(ABI_BUCKETS)}")
            start = time.perf_counter()
            cache = self._cache_dir(out_dir) / satellite
            cache.mkdir(parents=True, exist_ok=True)
            files: dict[str, dict] = {}
            wanted = [("acm", "ABI-L2-ACMF", None)] + [(f"rad{b:02d}", "ABI-L1b-RadF", int(b)) for b in self.bands]
            downloaded_now = 0
            for role, product, band in wanted:
                prefix = abi_scan_prefix(product, satellite, scan_moment, band)
                listing = _s3_list(bucket, prefix)
                if not listing:
                    raise FileNotFoundError(
                        f"no {product} granule of {satellite} starts at {scan_moment:%Y-%m-%dT%H:%M}Z "
                        f"(prefix {prefix} in {bucket}); the window has no full-disk scan to read"
                    )
                obj = sorted(listing, key=lambda o: o["key"])[0]
                url = f"https://{bucket}.s3.amazonaws.com/{obj['key']}"
                filename = obj["key"].rsplit("/", 1)[-1]
                destination = cache / filename
                sha, seconds = _download(url, destination, int(obj["size"]))
                if seconds > 0.0:
                    downloaded_now += 1
                files[role] = {"url": url, "path": str(destination), "bytes": int(obj["size"]), "sha256": sha,
                               "s3_last_modified": obj["last_modified"], "etag": obj["etag"],
                               "download_seconds": seconds, "band": band}
            # rw_goes bt per band (with the clear mask), then the observation-only superobs.
            stamp = scan_moment.strftime("%Y%m%dT%H%M")
            products = {}
            for band in self.bands:
                pack = cache / f"{satellite}-band{band:02d}-{stamp}.goespack"
                table = cache / f"{satellite}-band{band:02d}-{stamp}-superobs.csv"
                stats = cache / f"{satellite}-band{band:02d}-{stamp}-superobs.json"
                if not pack.is_file():
                    _run(rw_goes, ["bt", "--rad", files[f"rad{band:02d}"]["path"], "--acm", files["acm"]["path"],
                                   "--out", str(pack), "--received-utc", _stamp(_now())])
                if not (table.is_file() and stats.is_file()):
                    _run(rw_goes, ["superobs", "--pack", str(pack), "--out", str(table), "--stats", str(stats),
                                   "--block", str(int(self.block_pixels))])
                products[int(band)] = {"pack": str(pack), "superobs": str(table), "stats": str(stats)}
            manifest = {"schema": "gpuwm.arwen-global-abi-window/v1", "satellite": satellite, "bucket": bucket,
                        "scan_moment_utc": _stamp(scan_moment), "files": files, "products": products,
                        "window_start_utc": _stamp(window_start), "window_end_utc": _stamp(window_end),
                        "fetched_utc": _stamp(_now())}
            manifest_path = cache / f"window-{stamp}.json"
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
            size, sha = _digest(manifest_path)
            publications = [f["s3_last_modified"] for f in files.values() if f.get("s3_last_modified")]
            records.append(FetchRecord(
                stream=self.name, location=f"https://{bucket}.s3.amazonaws.com/{abi_scan_prefix('ABI-L1b-RadF', satellite, scan_moment, None)}",
                path=str(manifest_path), bytes=int(sum(f["bytes"] for f in files.values())), sha256=sha,
                window_start_utc=_stamp(window_start), window_end_utc=_stamp(window_end),
                fetched_utc=_stamp(_now()), wall_s=time.perf_counter() - start,
                latency_behind_real_time_s=(_now() - _utc(window_end)).total_seconds() if downloaded_now else None,
                decoder="rw_goes", publication_utc=max(publications) if publications else None,
                extra={"satellite": satellite, "bands": list(self.bands), "downloaded_now": downloaded_now,
                       "scan_moment_utc": _stamp(scan_moment)},
            ))
            self._windows[(satellite, _stamp(window_end))] = manifest
        return records

    def batches(self, context: RadianceContext) -> tuple[list, dict]:
        from .abi_operator import find_rw_goes
        from .abi_radiance_operator import (
            AbiRadianceOperator, band_centroid_lnp, band_profile, register_batch, stream_name, superobs_from_blocks,
        )

        out_dir = Path(context.out_dir)
        entries = self.entries()
        table = self.table()
        rw_goes = find_rw_goes(self.rw_goes)
        receipt = {"schema": RECEIPT_SCHEMA, "stream": self.name, "satellites": {},
                   "window_start_utc": _stamp(context.window_start), "window_end_utc": _stamp(context.window_end)}
        batches = []
        lengths_cache: dict[int, dict] = {}
        for satellite in self.satellites:
            manifest = self._windows.get((satellite, _stamp(context.window_end)))
            if manifest is None:
                raise ValueError(f"the goes-abi stream was asked for batches of a window it never fetched ({satellite})")
            sat_record = {"scan_moment_utc": manifest["scan_moment_utc"], "bands": {}}
            receipt["satellites"][satellite] = sat_record
            ledger = self._ledger(satellite, entries, out_dir)
            for band in self.bands:
                entry = entries["bands"].get(str(int(band)))
                if entry is None or not entry.get("admitted_classes"):
                    sat_record["bands"][str(band)] = {"refused": "no admitted class in the operator entries"}
                    continue
                product = manifest["products"][str(band)] if str(band) in manifest["products"] else manifest["products"][int(band)]
                stats = json.loads(Path(product["stats"]).read_text(encoding="utf-8"))
                scan_start = dt.datetime.fromisoformat(stats["scan_start"].replace("Z", "+00:00"))
                scan_end = dt.datetime.fromisoformat(stats["scan_end"].replace("Z", "+00:00"))
                surface_cache: dict = {}

                def land_fraction(lat, lon):
                    surface_cache.update(control_surface_at(context, lat, lon))
                    return surface_cache["land_fraction"]

                batch, extras = superobs_from_blocks(
                    product["superobs"], band=int(band), scan_start=scan_start, scan_end=scan_end, entry=entry,
                    land_fraction=land_fraction, block_pixels=int(self.block_pixels),
                    identity_prefix=f"{satellite}:{scan_start:%Y%m%dT%H%M}:")
                stages = {"blocks": int(stats.get("block_rows", 0)), **{k: int(v) for k, v in batch.rejections.items()},
                          "admitted_by_stream_qc": int(batch.count)}
                # The control's cloud and sea ice at the admitted blocks.
                if batch.count:
                    surface = control_surface_at(context, batch.latitude_deg, batch.longitude_deg)
                    keep = (surface["cloud_water_path"] <= float(self.max_model_cloud_water_kg_m2)) \
                        & (surface["sea_ice_fraction"] <= float(self.max_sea_ice_fraction))
                    stages["model_cloud_or_ice_refused"] = int((~keep).sum())
                    idx = np.flatnonzero(keep)
                    batch = batch.subset(idx)
                    extras = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) and np.asarray(v).shape[:1] == keep.shape else v)
                              for k, v in extras.items()}
                    # Rows inside the window only (the scan's rows are timed along the scan).
                    t0, t1 = _utc(context.window_start), _utc(context.window_end)
                    inside = np.array([t0 < v <= t1 for v in batch.valid_time], dtype=bool)
                    stages["outside_window"] = int((~inside).sum())
                    idx = np.flatnonzero(inside)
                    batch = batch.subset(idx)
                    extras = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) and np.asarray(v).shape[:1] == inside.shape else v)
                              for k, v in extras.items()}
                if batch.count:
                    keep = thin_to_grid(batch.latitude_deg, batch.longitude_deg, context.ensemble_transform.grid,
                                        score=np.asarray(extras["n_pixels"], dtype=np.float64))
                    stages["one_per_ensemble_cell"] = int(keep.size)
                    batch = batch.subset(keep)
                    rows_before = int(np.asarray(extras["zenith_deg"]).shape[0])
                    extras = {k: (np.asarray(v)[keep] if isinstance(v, np.ndarray) and np.asarray(v).ndim >= 1
                                  and np.asarray(v).shape[0] == rows_before else v)
                              for k, v in extras.items()}
                stages["offered"] = int(batch.count)
                band_record = {"screen": stages, "entry_class_error_k": entry["classes"]["water"].get("observation_error_k")}
                sat_record["bands"][str(band)] = band_record
                if batch.count == 0:
                    continue
                centroid = band_centroid_lnp(table, int(band))
                key = int(round(centroid * 100))
                if key not in lengths_cache:
                    lengths_cache[key] = ensemble_vertical_lengths(context, {"t": centroid})["t"]
                lengths = lengths_cache[key]
                cutoff = float(lengths["cutoff_lnp"])
                band_record["vertical_localisation"] = {"centroid_pa": float(math.exp(centroid)), **lengths}
                profile = band_profile(table, int(band), cutoff)
                batch.ln_pressure[:] = centroid
                batch.vertical_cutoff_lnp = cutoff
                batch.localisation_profile = np.broadcast_to(profile, (batch.count, profile.size)).copy()
                batch.stream = stream_name(satellite, int(band))
                operator = self._operators.get((satellite, int(band)))
                if operator is None:
                    operator = AbiRadianceOperator(
                        self._paths()[1], None, context.vertical, band=int(band), zenith_by_identity={},
                        rw_goes=rw_goes, work_dir=self._cache_dir(out_dir) / satellite / "forward",
                        month=int(_utc(context.window_end).month), threads=self.threads,
                        transforms=[context.ensemble_transform, context.control_transform])
                    self._operators[(satellite, int(band))] = operator
                else:
                    operator.bind(context.ensemble_transform)
                    operator.bind(context.control_transform)
                    operator.reset()
                    operator.zenith_by_identity.clear()
                    operator.correction_by_identity.clear()
                # The day update from the previous window's departures, then
                # the intercept move it left rides on the entry's correction.
                self._previous.setdefault((satellite, int(band)), None)
                previous = self._previous[(satellite, int(band))]
                if self.day_update and previous is not None:
                    band_record["day_update"] = self._update_bias(satellite, int(band), entry, ledger, previous)
                shift = float(ledger.coefficients[str(band)].get("intercept_shift_k", 0.0))
                extras["correction_intercept_k"] = np.asarray(extras["correction_intercept_k"], dtype=np.float64) + shift
                register_batch(operator, batch, extras)
                band_record["rows"] = int(batch.count)
                band_record["bias_correction"] = {**(entry["classes"]["water"].get("bias_correction") or {}),
                                                  "intercept_shift_k": shift}
                band_record["linearisation_previous_window"] = operator.last_linearisation_check
                self._previous[(satellite, int(band))] = batch
                batches.append(batch)
        self.last_receipt = receipt
        return batches, receipt

    def _ledger(self, satellite: str, entries: dict, out_dir: Path) -> BiasLedger:
        if satellite not in self._ledgers:
            initial = {str(b): {"intercept_shift_k": 0.0} for b in self.bands}
            self._ledgers[satellite] = BiasLedger.load(
                f"{self.name}:{satellite}", Path(out_dir) / f"radiance-bias-{self.name}-{satellite}.json", initial,
                memory_rows=int(self.bias_memory_rows))
        return self._ledgers[satellite]

    def _update_bias(self, satellite: str, band: int, entry: dict, ledger: BiasLedger, previous) -> dict:
        if previous.control_simulated is None or previous.count == 0:
            return {"rows": 0}
        hx = np.asarray(previous.control_simulated, dtype=np.float64)[0]
        finite = np.isfinite(hx)
        if finite.sum() < 30:
            return {"rows": int(finite.sum())}
        residual = previous.value[finite] - hx[finite]
        error = float(entry["classes"]["water"]["observation_error_k"])
        gross = np.abs(residual) <= 3.0 * error
        if gross.sum() < 30:
            return {"rows": int(gross.sum())}
        fit = {"intercept_shift_k": float(np.mean(residual[gross]))}
        move = ledger.move(str(band), fit, int(gross.sum()), label=previous.stream,
                           before=_statistics(residual[gross]), after=_statistics(residual[gross] - fit["intercept_shift_k"]))
        ledger.write()
        return {"rule": "the previous window's mean residual is recorded against the entry's intercept and not applied: "
                        "the intercept is anchored to the entry's fit (ANCHORED_TERMS), so the control's drift in the band "
                        "stays an innovation instead of becoming instrument bias",
                "memory_rows": int(ledger.memory_rows), "anchored": list(ANCHORED_TERMS), "move": move}


def _run(executable, arguments: list[str]) -> dict:
    import subprocess

    done = subprocess.run([str(executable), *arguments], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{Path(str(executable)).name} {arguments[0]} failed (rc {done.returncode}): "
                           f"{done.stderr.strip() or done.stdout.strip()}")
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError:
        return {"stdout": done.stdout}


# ---------------------------------------------------------------------------
# the stream-table factories
# ---------------------------------------------------------------------------

def _tuple(text: str | None, default: tuple, *, kind=str) -> tuple:
    if not text:
        return default
    return tuple(kind(v.strip()) for v in text.split(",") if v.strip())


def atms_factory(options: dict[str, str]) -> AtmsStream:
    entry_paths = {}
    for key, value in options.items():
        if key.startswith("entry."):
            entry_paths[key.split(".", 1)[1]] = value
    kwargs = dict(
        satellites=_tuple(options.get("satellites"), ("noaa-20", "noaa-21")),
        cache=options.get("cache"), entry_paths=entry_paths,
        channels=_tuple(options.get("channels"), None, kind=int) if options.get("channels") else None,
    )
    for key, kind in (("thin_deg", float), ("max_zenith_deg", float), ("max_model_cloud_water_kg_m2", float),
                      ("max_retrieved_lwp_kg_m2", float), ("bias_memory_rows", int), ("workers", int)):
        if key in options:
            kwargs[key] = kind(options[key])
    if "day_update" in options:
        kwargs["day_update"] = options["day_update"].lower() not in ("0", "false", "off", "no")
    return AtmsStream(**kwargs)


def abi_factory(options: dict[str, str]) -> AbiStream:
    kwargs = dict(
        satellites=_tuple(options.get("satellites"), ("G19",)),
        bands=_tuple(options.get("bands"), (13, 8), kind=int),
        cache=options.get("cache"), entries_path=options.get("entries"), table_path=options.get("table"),
        rw_goes=options.get("rw_goes"),
    )
    for key, kind in (("block_pixels", int), ("scan_minute", int), ("max_model_cloud_water_kg_m2", float),
                      ("bias_memory_rows", int), ("threads", int)):
        if key in options:
            kwargs[key] = kind(options[key])
    if "day_update" in options:
        kwargs["day_update"] = options["day_update"].lower() not in ("0", "false", "off", "no")
    return AbiStream(**kwargs)


__all__ = [
    "ABI_BUCKETS", "ABI_STREAM", "ATMS_STREAM", "BIAS_LEDGER_SCHEMA", "DEFAULT_BIAS_MEMORY_ROWS",
    "DEFAULT_MODEL_CLOUD_WATER_KG_M2", "RECEIPT_SCHEMA", "TABLES_DIR",
    "AbiStream", "AtmsStream", "BiasLedger", "RadianceContext",
    "abi_factory", "abi_scan_prefix", "atms_entry_path", "atms_factory", "control_surface_at",
    "ensemble_vertical_lengths", "thin_to_grid",
]
