"""Column water-budget instrument for WOOF global.

What it measures
----------------
Per interval between two checkpoints, per column and then area-averaged
over regions, the closure of

    d(W_v + W_c)  =  E - P  +  T  +  R

where, all in kg/m2 over the interval,

* ``W_v``  total precipitable water, the column integral ``int qv dp / g``;
* ``W_c``  column condensate, ``int (qc + qr + qi + qs + qg) dp / g``,
           reported per species;
* ``P``    surface precipitation from the model's own accumulators: the
           convective bucket ``physics__rainc`` (P_conv) plus the three
           grid-scale surface buckets rain + snow + graupel (P_grid, the
           same split the render tape's RAINC / RAINNC carry);
* ``E``    surface evaporation, read from the model's reservoir books
           (below); a second, coarser estimate ``E_flux`` is the trapezoid
           of the endpoint instantaneous ``physics__qfx`` (kg/m2/s) and is
           reported beside it as a cross-check, never as the budget's E;
* ``T``    horizontal transport convergence: zero globally, and per
           latitude band the trapezoid over the interval of the northward
           column water flux ``int v (qv + condensate) dp / g`` crossing
           the band's edges (edges are midpoints between the Gaussian rows
           nearest the nominal latitudes; the edge flux is the mean of the
           two adjacent rows' zonal means times ``2 pi a cos(edge)``);
* ``R``    the residual: the atmosphere's column water change that neither
           the physics exchanges nor the transport account for.  Reported
           in kg/m2 and as a fraction of ``max(|E|, |P|)``.  The global
           residual is exact to the reads (no transport enters it); a band
           residual carries the transport term's time sampling, the
           trapezoid of the flux at the two checkpoints, which the
           calibration proves only for a flux that is steady over the
           interval, so a band residual on 3-hourly checkpoints is a
           transport-limited reading and every output says so
           (``residual_status``).  Band E, P and dW carry no such limit.

Where E comes from.  Every physics exchange in the native runtime books the
column water it moved against the explicit surface reservoir, measured in
the model's own metric (``physics/native_runtime.py``: YSU's surface flux
debits it, Grell-Freitas and Morrison fallout credit it) and the land step
moves reservoir water into Noah's stores and the runoff outflow account
without changing their sum.  So with the reservoir system

    S = surface reservoir + soil water + canopy + snow + booked outflow

priced exactly as ``water.total_water_column`` and the in-situ ledger price
it, the physics net exchange over an interval is

    N = E - P = F - dS

where ``F`` is the global water fixer's uniform per-step correction to the
surface reservoir summed over the interval (``insitu.ndjson`` metric
``global_water_fixer_kg_m2``; zero, and stated, when no ledger is given).
The instrument reports ``E = N + P`` with ``P`` from the accumulators and
states the one approximation that carries: the accumulators are the
microphysics kernel's own kg/m2 (rho dz mixing ratios) while N is measured
in the dycore's metric (q dp / g), which the runtime's audit puts at O(q),
one to two percent of a burst.  The residual itself needs no accumulator:
``R = dW + dS - F - T`` is the per-column non-physics change of total water,
and globally (T = 0, the fixer pinning the total) it equals ``-F``, the
water the dynamics created or destroyed that the fixer replaced.

Sign convention: E > 0 moistens the column (evaporation), P > 0 removes
water, dW > 0 is a column gain, R > 0 is water the column gained from no
booked source.

Sampling.  Checkpoints are the intervals' endpoints; the first checkpoint
of a cold start carries no physics state (no flux field, no stores, no
convective bucket), which the record lists: its interval's ``E_flux`` is
the rectangle on the ending value alone and its stores start at zero.  The
in-situ ledger, when present, gives the same global budget at every step:
its rows at the checkpoint steps are compared with the checkpoint reads
(``ledger.checks``) and its per-step condensate series is written at a
coarse cadence so accumulation aloft can be seen between checkpoints.

Model input: ``arwen_global_step*.npz`` checkpoints read through
``checkpoint.read_checkpoint`` (hash-verified) and synthesized in float64
on the run's own Gaussian grid from the receipt's config (truncation,
grid shape, hybrid A/B tables); no model or initial state is built.

Calibration (``python -m woof.globe.water_budget --calibrate``,
tests/test_arwen_global_water_budget.py): the rows in CALIBRATION below
are recorded verbatim from the CPU test host.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import GRAVITY_M_S2, LIQUID_WATER_DENSITY, WATER_SPECIES
from .water import (
    NATIVE_CANOPY_STORE_NAMES,
    NATIVE_SNOW_STORE_NAMES,
    SOIL_LAYER_THICKNESS_M,
    WATER_OUTFLOW_NAMES,
)

SCHEMA = "gpuwm.arwen-global-water-budget/v1"
CONDENSATE_SPECIES = ("qc", "qr", "qi", "qs", "qg")
SECONDS_PER_DAY = 86400.0

#: Latitude bands (name, south edge, north edge); rows are assigned by
#: ``south <= latitude < north`` (the last band takes the pole).
BANDS: tuple[tuple[str, float, float], ...] = (
    ("90S-60S", -90.0, -60.0),
    ("60S-30S", -60.0, -30.0),
    ("30S-0", -30.0, 0.0),
    ("0-30N", 0.0, 30.0),
    ("30N-60N", 30.0, 60.0),
    ("60N-90N", 60.0, 90.0),
)
GLOBAL = "global"
LAND = "land"
OCEAN = "ocean"

CHECKPOINT_CONVECTIVE = "physics__rainc"
CHECKPOINT_GRID_SCALE_BUCKETS = (
    "surface__accumulated_rain_kg_m2",
    "surface__accumulated_snow_kg_m2",
    "surface__accumulated_graupel_kg_m2",
)
CHECKPOINT_GRID_SCALE_TOTAL = "physics__rainnc"
CHECKPOINT_FLUX = "physics__qfx"
CHECKPOINT_SURFACE_WATER = "surface__surface_water_kg_m2"
CHECKPOINT_SOIL = "surface__soil_water_fraction"
CHECKPOINT_LAND_FRACTION = "surface__land_fraction"
RECEIPT_NAME = "arwen-global-receipt.json"
LEDGER_NAME = "insitu.ndjson"

CALIBRATION = """
Recorded 2026-09-02 on the CPU test host (Python 3.14.4, numpy 2.5.2,
float64) by ``--calibrate``; the test file holds every bar.  Synthetic
Gaussian grid T21 (33 x 66) unless stated; one 3 h interval; backgrounds
W_v 25 kg/m2, W_c 0.2 kg/m2 per species, reservoir 500 kg/m2 (+600 on
land); every planted field carries the 0.5 + cos(lat)(1 + 0.2 cos(lon))
pattern, so the numbers below are area means of that pattern.

Family A, planted P sink, no E (P split 0.3 convective / 0.7 grid-scale):

    planted P  read P     read P_conv  read E     read E_flux  residual   bar
    0.128541   0.128541   0.038562     -9.4e-16   0.0          9.4e-16    1e-12 relative
    1.285409   1.285409   0.385623      1.5e-16   0.0         -4.4e-16
    12.854091  12.854091  3.856227     -4.7e-15   0.0          5.3e-15

Family A', an unbooked leak beside P 1.285 (both directions):

    planted leak   read residual   read fraction   relative error
    -0.128541      -0.128541       -0.1000         1.3e-15
    -1.285409      -1.285409       -1.0000         3.5e-16
    -12.854091     -12.854091      -10.0000        1.4e-16
    +0.128541      +0.128541       +0.1000         2.2e-15
    +1.285409      +1.285409       +1.0000         3.5e-16
    +12.854091     +12.854091      +10.0000        0.0

Family B, planted E source, no P (constant flux; E_flux is the trapezoid):

    planted E   read E_books   read E_flux   read P   residual   bar
    0.128541    0.128541       0.128541      0.0      -8.9e-16   1e-12 relative
    1.285409    1.285409       1.285409      0.0       2.2e-16
    12.854091   12.854091      12.854091     0.0      -3.6e-15
    dew -0.642705  -0.642705   -0.642705     0.0       7.8e-16
    E 1.285409 with a 0.05 fixer credit on the reservoir: read E 1.285409
    (the fixer stays out of E), residual -3.7e-14

Family C, P north of the equator only, E south only (per band):

    band      planted P  read P     planted E  read E     residual
    90S-60S   0.0        0.0        1.635162   1.635162   2.7e-15
    60S-30S   0.0        0.0        2.417510   2.417510   4.4e-15
    30S-0     0.0        0.0        2.907460   2.907460   4.4e-16
    0-30N     2.923522   2.923522   0.0        0.0        0.0
    30N-60N   2.417510   2.417510   0.0        0.0       -4.0e-15
    60N-90N   1.635162   1.635162   0.0        0.0       -2.7e-15

Family D, a planted uniform northward flux (v +-1 m/s of W 25 kg/m2) with
the analytic band convergence at the instrument's own edges applied to the
vapor; without the transport term the band residual reads the planted
convergence exactly (1e-12), with it the residual is the quadrature error
of the band area (T21 polar bands 2.7e-3, other bands 3.7e-4; T63 2.8e-4
and 4.4e-5; the error falls with the square of the row spacing):

    grid  band      planted T   read T      relative error   bar
    T21   90S-60S   -0.168584   -0.169044   2.7e-3           3e-3
    T21   60S-30S   -0.043372   -0.043356   3.7e-4
    T21   30S-0     -0.012246   -0.012241   3.7e-4
    T21   0-30N      0.010120    0.010117   3.7e-4
    T21   30N-60N    0.043372    0.043356   3.7e-4
    T21   60N-90N    0.168584    0.169044   2.7e-3
    T63   90S-60S   -0.156451   -0.156495   2.8e-4           4e-4
    T63   60S-30S   -0.042033   -0.042031   4.4e-5
    T63   30S-0     -0.011293   -0.011293   4.4e-5
    T63   0-30N      0.011293    0.011293   4.4e-5
    T63   30N-60N    0.042033    0.042031   4.4e-5
    T63   60N-90N    0.156451    0.156495   2.8e-4
    v -1 m/s: every sign flips and every error repeats to the digit.

Both directions on every family: sink and source, evaporation and dew,
northward and southward flux read back with their sign.
"""


# --------------------------------------------------------------------------
# grid and samples
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetGrid:
    """The Gaussian rows the columns live on: latitudes, quadrature weights
    (sum 2, the transform's own), zonal count and planetary radius."""

    latitude_deg: np.ndarray
    quadrature_weights: np.ndarray
    nlon: int
    radius_m: float

    def __post_init__(self) -> None:
        lat = np.asarray(self.latitude_deg, dtype=np.float64)
        w = np.asarray(self.quadrature_weights, dtype=np.float64)
        if lat.ndim != 1 or w.shape != lat.shape:
            raise ValueError("latitude_deg and quadrature_weights must be matching 1-D arrays")
        if abs(float(np.sum(w)) - 2.0) > 1.0e-9:
            raise ValueError(f"quadrature weights sum to {float(np.sum(w)):.12g}, expected 2")
        if int(self.nlon) < 1 or not math.isfinite(float(self.radius_m)) or self.radius_m <= 0:
            raise ValueError("nlon must be positive and radius_m finite and positive")
        object.__setattr__(self, "latitude_deg", lat)
        object.__setattr__(self, "quadrature_weights", w)
        object.__setattr__(self, "nlon", int(self.nlon))
        object.__setattr__(self, "radius_m", float(self.radius_m))

    @classmethod
    def from_gaussian(cls, grid) -> "BudgetGrid":
        return cls(grid.latitude_deg, grid.quadrature_weights, grid.nlon, grid.radius_m)

    @classmethod
    def for_shape(cls, nlat: int, nlon: int) -> "BudgetGrid":
        from woof.globe.spectral.grid import GaussianGrid

        return cls.from_gaussian(GaussianGrid.for_shape(int(nlat), int(nlon)))

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.latitude_deg.size), int(self.nlon)

    def cell_weights(self) -> np.ndarray:
        """Area fraction per cell, ``(ny, nx)``, summing to 1."""
        return np.repeat(
            (self.quadrature_weights / (2.0 * self.nlon))[:, None], self.nlon, axis=1
        )

    def area_mean(self, values: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Area-weighted mean over the masked cells (over all cells when
        ``mask`` is None).  Raises on an empty region."""
        w = self.cell_weights()
        if mask is not None:
            w = np.where(mask, w, 0.0)
        total = float(np.sum(w))
        if total <= 0.0:
            raise ValueError("region selects no cells")
        return float(np.sum(w * values) / total)

    def area_fraction(self, mask: np.ndarray) -> float:
        return float(np.sum(np.where(mask, self.cell_weights(), 0.0)))

    def row_order(self) -> np.ndarray:
        """Row indices from south to north."""
        return np.argsort(self.latitude_deg, kind="stable")


@dataclass(frozen=True)
class ColumnSample:
    """One checkpoint's per-column fields (``(ny, nx)`` unless stated).

    ``vapor`` / ``condensate[species]`` are column integrals in kg/m2;
    ``reservoir`` is the reservoir system S; ``convective`` /
    ``grid_scale`` are the accumulators (kg/m2 since the run start);
    ``flux_evaporation`` is the instantaneous surface vapor flux
    (kg/m2/s) or None; ``northward_transport`` is the zonal mean of the
    northward column water flux ``int v W dp/g`` per row (kg/m/s, ``(ny,)``)
    or None; ``land`` is the land mask or None.
    """

    time_s: float
    step: int
    vapor: np.ndarray
    condensate: dict[str, np.ndarray]
    reservoir: np.ndarray
    convective: np.ndarray
    grid_scale: np.ndarray
    flux_evaporation: np.ndarray | None = None
    northward_transport: np.ndarray | None = None
    land: np.ndarray | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shape = np.asarray(self.vapor).shape
        if len(shape) != 2:
            raise ValueError("column fields must be (ny, nx)")
        for name in ("vapor", "reservoir", "convective", "grid_scale"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite (ny, nx) = {shape}")
            object.__setattr__(self, name, value)
        if tuple(self.condensate) != CONDENSATE_SPECIES:
            raise ValueError(f"condensate must carry exactly {CONDENSATE_SPECIES} in order")
        condensate = {}
        for species, value in self.condensate.items():
            value = np.asarray(value, dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"condensate {species} must be finite (ny, nx) = {shape}")
            condensate[species] = value
        object.__setattr__(self, "condensate", condensate)
        if self.flux_evaporation is not None:
            value = np.asarray(self.flux_evaporation, dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("flux_evaporation must be finite (ny, nx)")
            object.__setattr__(self, "flux_evaporation", value)
        if self.northward_transport is not None:
            value = np.asarray(self.northward_transport, dtype=np.float64)
            if value.shape != (shape[0],) or not np.isfinite(value).all():
                raise ValueError("northward_transport must be finite (ny,)")
            object.__setattr__(self, "northward_transport", value)
        if self.land is not None:
            value = np.asarray(self.land, dtype=bool)
            if value.shape != shape:
                raise ValueError("land must be (ny, nx)")
            object.__setattr__(self, "land", value)

    @property
    def condensate_total(self) -> np.ndarray:
        return sum(self.condensate.values())

    @property
    def atmosphere(self) -> np.ndarray:
        return self.vapor + self.condensate_total


# --------------------------------------------------------------------------
# regions and transport
# --------------------------------------------------------------------------


def band_rows(grid: BudgetGrid, bands=BANDS) -> dict[str, np.ndarray]:
    """Row mask per band; every row lands in exactly one band."""
    lat = grid.latitude_deg
    out: dict[str, np.ndarray] = {}
    taken = np.zeros(lat.shape, dtype=bool)
    for index, (name, south, north) in enumerate(bands):
        mask = (lat >= south) & ((lat < north) if index < len(bands) - 1 else (lat <= north))
        out[name] = mask
        taken |= mask
    if not taken.all():
        raise ValueError("the latitude bands do not cover every row")
    return out


def band_edges(grid: BudgetGrid, bands=BANDS) -> dict[str, tuple[float, float]]:
    """The instrument's own edge latitudes per band: midpoints between the
    last row of one band and the first row of the next (poles at +-90)."""
    rows = band_rows(grid, bands)
    order = grid.row_order()
    lat_sorted = grid.latitude_deg[order]
    edges: dict[str, tuple[float, float]] = {}
    for name, _south, _north in bands:
        positions = np.flatnonzero(rows[name][order])
        if positions.size == 0:
            edges[name] = (float("nan"), float("nan"))
            continue
        lo, hi = int(positions[0]), int(positions[-1])
        south = -90.0 if lo == 0 else 0.5 * (lat_sorted[lo - 1] + lat_sorted[lo])
        north = 90.0 if hi == lat_sorted.size - 1 else 0.5 * (lat_sorted[hi] + lat_sorted[hi + 1])
        edges[name] = (float(south), float(north))
    return edges


def edge_flux_kg_s(grid: BudgetGrid, northward_transport: np.ndarray, edge_lat_deg: float) -> float:
    """Northward flux (kg/s) across the latitude circle at an edge: the
    mean of the two adjacent rows' zonal-mean column flux times the
    circle's length ``2 pi a cos(edge)``; zero at the poles."""
    if abs(edge_lat_deg) >= 90.0:
        return 0.0
    order = grid.row_order()
    lat_sorted = grid.latitude_deg[order]
    flux_sorted = np.asarray(northward_transport, dtype=np.float64)[order]
    above = int(np.searchsorted(lat_sorted, edge_lat_deg))
    if above <= 0 or above >= lat_sorted.size:
        raise ValueError(f"edge {edge_lat_deg} is outside the rows")
    mean_flux = 0.5 * (flux_sorted[above - 1] + flux_sorted[above])
    return float(2.0 * math.pi * grid.radius_m * math.cos(math.radians(edge_lat_deg)) * mean_flux)


def band_convergence_kg_m2_s(
    grid: BudgetGrid, northward_transport: np.ndarray, bands=BANDS
) -> dict[str, float]:
    """Net inflow per unit band area (kg/m2/s) from the edge fluxes."""
    rows = band_rows(grid, bands)
    edges = band_edges(grid, bands)
    out: dict[str, float] = {}
    for name, _south, _north in bands:
        if not rows[name].any():
            out[name] = float("nan")
            continue
        area = 4.0 * math.pi * grid.radius_m ** 2 * (
            float(np.sum(grid.quadrature_weights[rows[name]])) / 2.0
        )
        south, north = edges[name]
        inflow = edge_flux_kg_s(grid, northward_transport, south) - edge_flux_kg_s(
            grid, northward_transport, north
        )
        out[name] = inflow / area
    return out


def region_masks(grid: BudgetGrid, land: np.ndarray | None, bands=BANDS) -> dict[str, np.ndarray]:
    ny, nx = grid.shape
    masks = {GLOBAL: np.ones((ny, nx), dtype=bool)}
    for name, rows in band_rows(grid, bands).items():
        masks[name] = np.repeat(rows[:, None], nx, axis=1)
    if land is not None:
        masks[LAND] = np.asarray(land, dtype=bool)
        masks[OCEAN] = ~masks[LAND]
    return masks


# --------------------------------------------------------------------------
# the budget
# --------------------------------------------------------------------------

INTERVAL_TERMS = (
    "dW_vapor", "dW_condensate", "dW_atmosphere", "dS_reservoir", "fixer",
    "physics_net_e_minus_p", "E_books", "E_flux", "P_conv", "P_grid", "P",
    "transport_convergence", "residual",
)
RATE_TERMS = (
    "E_books", "E_flux", "P_conv", "P_grid", "P", "dW_vapor", "dW_condensate",
    "dW_atmosphere", "transport_convergence", "residual",
)


def _fraction(residual: float, e: float, p: float) -> float | None:
    reference = max(abs(e), abs(p))
    return residual / reference if reference > 0.0 else None


def interval_budget(
    before: ColumnSample,
    after: ColumnSample,
    grid: BudgetGrid,
    *,
    fixer_kg_m2: float = 0.0,
    bands=BANDS,
) -> dict[str, object]:
    """The budget of one interval: per-region area means of the per-column
    terms, the residual and its fraction, the condensate loads."""
    if after.time_s <= before.time_s:
        raise ValueError("intervals must have after.time_s > before.time_s")
    if before.vapor.shape != after.vapor.shape or before.vapor.shape != grid.shape:
        raise ValueError("samples and grid disagree on the column shape")
    dt = float(after.time_s - before.time_s)
    d_vapor = after.vapor - before.vapor
    d_species = {s: after.condensate[s] - before.condensate[s] for s in CONDENSATE_SPECIES}
    d_cond = sum(d_species.values())
    d_atm = d_vapor + d_cond
    d_res = after.reservoir - before.reservoir
    p_conv = after.convective - before.convective
    p_grid = after.grid_scale - before.grid_scale
    p = p_conv + p_grid
    net = float(fixer_kg_m2) - d_res
    e_books = net + p
    flux_note = None
    if before.flux_evaporation is not None and after.flux_evaporation is not None:
        e_flux = 0.5 * (before.flux_evaporation + after.flux_evaporation) * dt
    elif after.flux_evaporation is not None:
        e_flux = after.flux_evaporation * dt
        flux_note = "rectangle on the ending flux: the starting checkpoint carries no flux field"
    elif before.flux_evaporation is not None:
        e_flux = before.flux_evaporation * dt
        flux_note = "rectangle on the starting flux: the ending checkpoint carries no flux field"
    else:
        e_flux = None
        flux_note = "no flux field in either checkpoint"
    transport_available = (
        before.northward_transport is not None and after.northward_transport is not None
    )
    convergence: dict[str, float] = {}
    if transport_available:
        c0 = band_convergence_kg_m2_s(grid, before.northward_transport, bands)
        c1 = band_convergence_kg_m2_s(grid, after.northward_transport, bands)
        convergence = {name: 0.5 * (c0[name] + c1[name]) * dt for name in c0}
    land = after.land if after.land is not None else before.land
    masks = region_masks(grid, land, bands)
    regions: dict[str, object] = {}
    for name, mask in masks.items():
        if not mask.any():
            regions[name] = {"status": "empty", "reason": "region selects no cells"}
            continue

        def mean(values, mask=mask):
            return grid.area_mean(values, mask)

        row: dict[str, object] = {
            "area_fraction": grid.area_fraction(mask),
            "dW_vapor": mean(d_vapor),
            "dW_condensate": mean(d_cond),
            "dW_condensate_by_species": {s: mean(d_species[s]) for s in CONDENSATE_SPECIES},
            "dW_atmosphere": mean(d_atm),
            "dS_reservoir": mean(d_res),
            "fixer": float(fixer_kg_m2),
            "physics_net_e_minus_p": mean(net),
            "E_books": mean(e_books),
            "E_flux": mean(e_flux) if e_flux is not None else None,
            "P_conv": mean(p_conv),
            "P_grid": mean(p_grid),
            "P": mean(p),
            "condensate_end_by_species": {s: mean(after.condensate[s]) for s in CONDENSATE_SPECIES},
            "vapor_end": mean(after.vapor),
        }
        if name == GLOBAL:
            transport = 0.0
            transport_status = "zero by construction (closed sphere)"
        elif name in convergence:
            transport = convergence[name]
            transport_status = (
                "trapezoid of the edge fluxes at the two checkpoints (steady-flux "
                "calibration only; a fluctuating flux is sampled, not integrated)"
            )
        else:
            transport = None
            transport_status = (
                "not separable: land/ocean masks have no closed edges"
                if name in (LAND, OCEAN)
                else "no transport field in the samples"
            )
        row["transport_convergence"] = transport
        row["transport_status"] = transport_status
        if transport is None:
            row["residual"] = None
            row["residual_fraction"] = None
            row["residual_status"] = "not separable from transport"
        else:
            residual = row["dW_atmosphere"] - row["physics_net_e_minus_p"] - transport
            row["residual"] = residual
            row["residual_fraction"] = _fraction(residual, row["E_books"], row["P"])
            row["residual_status"] = (
                "measured" if name == GLOBAL
                else "measured (transport-limited: the band flux is sampled at the checkpoints)"
            )
        row["rates_mm_day"] = {
            key: (row[key] * SECONDS_PER_DAY / dt if row[key] is not None else None)
            for key in RATE_TERMS
        }
        regions[name] = row
    return {
        "start_s": float(before.time_s),
        "end_s": float(after.time_s),
        "start_step": int(before.step),
        "end_step": int(after.step),
        "hours": dt / 3600.0,
        "fixer_kg_m2": float(fixer_kg_m2),
        "E_flux_note": flux_note,
        "negative_accumulator_increments": int(
            np.count_nonzero(p_conv < 0.0) + np.count_nonzero(p_grid < 0.0)
        ),
        "most_negative_accumulator_increment_kg_m2": float(min(p_conv.min(), p_grid.min())),
        "max_condensate_column_end_kg_m2": {
            s: float(after.condensate[s].max()) for s in CONDENSATE_SPECIES
        },
        "max_vapor_column_end_kg_m2": float(after.vapor.max()),
        "regions": regions,
    }


def _accumulate(intervals: list[dict], region: str) -> dict[str, object]:
    rows = [
        iv["regions"][region]
        for iv in intervals
        if iv["regions"][region].get("residual_status") is not None
    ]
    if not rows:
        return {"status": "empty"}
    hours = float(sum(iv["hours"] for iv in intervals))
    out: dict[str, object] = {"hours": hours}
    for key in INTERVAL_TERMS:
        values = [row[key] for row in rows]
        out[key] = None if any(v is None for v in values) else float(sum(values))
    out["dW_condensate_by_species"] = {
        s: float(sum(row["dW_condensate_by_species"][s] for row in rows))
        for s in CONDENSATE_SPECIES
    }
    if out["residual"] is None:
        out["residual_fraction"] = None
    else:
        out["residual_fraction"] = _fraction(out["residual"], out["E_books"], out["P"])
    out["residual_status"] = rows[0]["residual_status"]
    out["rates_mm_day"] = {
        key: (out[key] * 24.0 / hours if out[key] is not None else None) for key in RATE_TERMS
    }
    out["condensate_end_by_species"] = rows[-1]["condensate_end_by_species"]
    out["vapor_end"] = rows[-1]["vapor_end"]
    return out


def budget(
    samples: list[ColumnSample],
    grid: BudgetGrid,
    *,
    fixer_kg_m2: list[float] | None = None,
    bands=BANDS,
) -> dict[str, object]:
    """The instrument over a checkpoint series: per-interval budgets and
    their cumulative sums per region."""
    if len(samples) < 2:
        raise ValueError(f"{len(samples)} samples give no interval; at least 2 are needed")
    times = np.asarray([s.time_s for s in samples], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("samples must be strictly increasing in time")
    n = len(samples) - 1
    if fixer_kg_m2 is None:
        fixer = [0.0] * n
        fixer_status = (
            "no ledger: the fixer is taken as zero, so E_books carries any global fixer correction"
        )
    else:
        fixer = [float(v) for v in fixer_kg_m2]
        if len(fixer) != n:
            raise ValueError(f"fixer_kg_m2 must carry one value per interval ({n})")
        fixer_status = "per-interval sum of the ledger's global_water_fixer_kg_m2"
    intervals = [
        interval_budget(samples[i], samples[i + 1], grid, fixer_kg_m2=fixer[i], bands=bands)
        for i in range(n)
    ]
    regions = list(intervals[0]["regions"])
    cumulative = {name: _accumulate(intervals, name) for name in regions}
    return {
        "schema": SCHEMA,
        "measures": (
            "per interval between checkpoints, area means of d(W_v + W_c) = E - P + T + R "
            "per column; E from the reservoir books (F - dS + P), P from the accumulators "
            "(convective + rain + snow + graupel), T the band-edge transport, R the residual"
        ),
        "sign_convention": (
            "E > 0 evaporation into the column, P > 0 out of it, dW > 0 column gain, "
            "R > 0 unbooked gain"
        ),
        "units": "kg/m2 over the interval; rates_mm_day are the same numbers per day",
        "residual_fraction": "residual / max(|E_books|, |P|) of the region",
        "fixer": fixer_status,
        "bands": {name: {"south_deg": s, "north_deg": nn} for name, s, nn in bands},
        "band_edges_deg": {k: list(v) for k, v in band_edges(grid, bands).items()},
        "sampling": {
            "n_intervals": n,
            "interval_hours": [iv["hours"] for iv in intervals],
            "steps": [int(s.step) for s in samples],
            "time_s": [float(s.time_s) for s in samples],
            "notes": sorted({note for s in samples for note in s.notes}),
        },
        "intervals": intervals,
        "cumulative": cumulative,
    }


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------


class CheckpointReader:
    """Synthesizes the column fields of a run's checkpoints on the run's
    own grid, in float64, from the receipt's config."""

    def __init__(self, receipt: dict, *, transport: bool = True):
        from woof.globe.spectral.transform import SphericalHarmonicTransform
        from woof.globe.spectral.vector import VorticityDivergenceOperator

        from .vertical import HybridCoordinate

        cfg = receipt["config"]
        block = receipt["transform"]
        self.transform = SphericalHarmonicTransform.create(
            int(cfg["truncation"]),
            nlat=int(block["nlat"]),
            nlon=int(block["nlon"]),
            dealias_factor=float(cfg["dealias_factor"]),
            radius_m=float(block["radius_m"]),
            backend="numpy",
            precision="float64",
        )
        self.vertical = HybridCoordinate(
            np.asarray(cfg["a_half_pa"], dtype=np.float64),
            np.asarray(cfg["b_half"], dtype=np.float64),
        )
        self.vector = VorticityDivergenceOperator(self.transform)
        self.grid = BudgetGrid.from_gaussian(self.transform.grid)
        self.transport = bool(transport)
        self.config_hash = receipt.get("config_hash")

    def _grid_field(self, value: np.ndarray) -> np.ndarray:
        """A checkpointed atmosphere field on the grid in float64: spectral
        coefficients (vapor, and every tracer of a spectral-tracer-era
        checkpoint) are synthesized; a schema-v3 grid tracer is read as
        it is.  Both eras of the arms read through this one door."""
        if np.iscomplexobj(value):
            return np.asarray(
                self.transform.inverse(value.astype(np.complex128)),
                dtype=np.float64,
            )
        return np.asarray(value, dtype=np.float64)

    def sample(self, path: str | Path) -> ColumnSample:
        from .checkpoint import read_checkpoint

        metadata, arrays = read_checkpoint(path)
        notes: list[str] = []
        backend = self.transform.backend
        logps = self.transform.inverse(
            arrays["atmosphere__log_surface_pressure"].astype(np.complex128)
        )
        ps = np.exp(logps)
        dp_g = self.vertical.pressure(ps, backend)["dp"] / GRAVITY_M_S2
        columns: dict[str, np.ndarray] = {}
        total = np.zeros(dp_g.shape, dtype=np.float64)
        for species in WATER_SPECIES:
            q = self._grid_field(arrays[f"atmosphere__{species}"])
            columns[species] = np.sum(q * dp_g, axis=0)
            if self.transport:
                total += q
        del q
        transport = None
        if self.transport:
            _u, v = self.vector.wind_from_vordiv(
                arrays["atmosphere__vorticity"].astype(np.complex128),
                arrays["atmosphere__divergence"].astype(np.complex128),
            )
            transport = np.mean(np.sum(v * total * dp_g, axis=0), axis=-1)
            del _u, v, total
        land_fraction = np.asarray(arrays[CHECKPOINT_LAND_FRACTION], dtype=np.float64)
        soil = np.sum(
            np.maximum(np.asarray(arrays[CHECKPOINT_SOIL], dtype=np.float64), 0.0)
            * np.asarray(SOIL_LAYER_THICKNESS_M, dtype=np.float64)[:, None, None]
            * LIQUID_WATER_DENSITY
            * land_fraction[None],
            axis=0,
        )
        reservoir = np.asarray(arrays[CHECKPOINT_SURFACE_WATER], dtype=np.float64) + soil
        physics = {
            k.removeprefix("physics__"): v for k, v in arrays.items() if k.startswith("physics__")
        }
        step = int(metadata["step"])
        for choices, label in (
            (NATIVE_CANOPY_STORE_NAMES, "canopy"), (NATIVE_SNOW_STORE_NAMES, "snow"),
        ):
            name = next((n for n in choices if n in physics), None)
            if name is None:
                notes.append(f"step {step}: no {label} store (counted zero)")
            else:
                reservoir = reservoir + np.maximum(np.asarray(physics[name], dtype=np.float64), 0.0)
        name = next((n for n in WATER_OUTFLOW_NAMES if n in physics), None)
        if name is None:
            notes.append(f"step {step}: no outflow account (counted zero)")
        else:
            reservoir = reservoir + np.asarray(physics[name], dtype=np.float64)
        convective_name = CHECKPOINT_CONVECTIVE.removeprefix("physics__")
        if convective_name in physics:
            convective = np.asarray(physics[convective_name], dtype=np.float64)
        else:
            convective = np.zeros(reservoir.shape, dtype=np.float64)
            notes.append(f"step {step}: no convective accumulator (zeros)")
        grid_scale = sum(
            np.asarray(arrays[n], dtype=np.float64) for n in CHECKPOINT_GRID_SCALE_BUCKETS
        )
        total_name = CHECKPOINT_GRID_SCALE_TOTAL.removeprefix("physics__")
        if total_name in physics:
            gap = float(np.max(np.abs(
                np.asarray(physics[total_name], dtype=np.float64) - grid_scale
            )))
            scale = max(float(np.max(grid_scale)), 1.0)
            if gap > 1.0e-5 * scale:
                raise ValueError(
                    f"{path}: {CHECKPOINT_GRID_SCALE_TOTAL} differs from the surface buckets by "
                    f"{gap:.3g} kg/m2; the grid-scale split would be ambiguous"
                )
        flux_name = CHECKPOINT_FLUX.removeprefix("physics__")
        flux = np.asarray(physics[flux_name], dtype=np.float64) if flux_name in physics else None
        if flux is None:
            notes.append(f"step {step}: no surface flux field (E_flux unavailable)")
        return ColumnSample(
            time_s=float(metadata["time_s"]),
            step=step,
            vapor=columns["qv"],
            condensate={s: columns[s] for s in CONDENSATE_SPECIES},
            reservoir=reservoir,
            convective=convective,
            grid_scale=grid_scale,
            flux_evaporation=flux,
            northward_transport=transport,
            land=land_fraction >= 0.5,
            notes=tuple(notes),
        )


def read_receipt(run_dir: Path) -> dict:
    path = Path(run_dir) / RECEIPT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path}: the run receipt is needed for the grid and the vertical tables"
        )
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


# --------------------------------------------------------------------------
# the in-situ ledger
# --------------------------------------------------------------------------

LEDGER_RESERVOIR_TERMS = (
    "water_surface_kg_m2", "water_soil_kg_m2", "water_native_kg_m2", "water_outflow_kg_m2",
)


def read_ledger(path: str | Path) -> dict[str, np.ndarray]:
    """Per-step global series from ``insitu.ndjson``: steps, times, the
    six species columns, the atmosphere and reservoir totals and the
    global water fixer correction."""
    steps: list[int] = []
    times: list[float] = []
    species: dict[str, list[float]] = {s: [] for s in WATER_SPECIES}
    atmosphere: list[float] = []
    reservoir: list[float] = []
    total: list[float] = []
    fixer: list[float] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "step":
                continue
            terms = row["terms"]
            steps.append(int(row["step"]))
            times.append(float(row["time_s"]))
            for s in WATER_SPECIES:
                species[s].append(float(terms[f"water_{s}_kg_m2"]))
            atmosphere.append(float(terms["water_atmosphere_kg_m2"]))
            reservoir.append(float(sum(terms[t] for t in LEDGER_RESERVOIR_TERMS)))
            total.append(float(terms["water_total_kg_m2"]))
            fixer.append(float(row.get("metrics", {}).get("global_water_fixer_kg_m2", 0.0)))
    if not steps:
        raise ValueError(f"{path}: no step rows")
    order = np.argsort(steps, kind="stable")

    def take(values):
        return np.asarray(values, dtype=np.float64)[order]

    return {
        "step": np.asarray(steps, dtype=np.int64)[order],
        "time_s": take(times),
        **{f"water_{s}": take(species[s]) for s in WATER_SPECIES},
        "atmosphere": take(atmosphere),
        "reservoir": take(reservoir),
        "total": take(total),
        "fixer": take(fixer),
    }


def ledger_intervals(ledger: dict[str, np.ndarray], steps: list[int]) -> dict[str, object]:
    """The global budget per checkpoint interval from the ledger alone,
    the per-interval fixer sums, and the ledger rows at the checkpoint
    steps (for the checkpoint cross-check)."""
    index = {int(s): i for i, s in enumerate(ledger["step"])}
    rows: list[dict | None] = []
    for step in steps:
        if int(step) == 0 and 0 not in index:
            rows.append(None)  # the cold start precedes the first ledger row
        elif int(step) not in index:
            raise ValueError(f"ledger has no row at checkpoint step {step}")
        else:
            i = index[int(step)]
            rows.append({
                "step": int(step),
                "atmosphere": float(ledger["atmosphere"][i]),
                "reservoir": float(ledger["reservoir"][i]),
                "total": float(ledger["total"][i]),
                **{f"water_{s}": float(ledger[f"water_{s}"][i]) for s in WATER_SPECIES},
            })
    intervals = []
    for a, b in zip(steps[:-1], steps[1:]):
        inside = (ledger["step"] > int(a)) & (ledger["step"] <= int(b))
        fixer = float(np.sum(ledger["fixer"][inside]))
        row_a, row_b = rows[steps.index(a)], rows[steps.index(b)]
        entry: dict[str, object] = {
            "start_step": int(a), "end_step": int(b),
            "steps_summed": int(np.count_nonzero(inside)),
            "fixer_kg_m2": fixer,
        }
        if row_a is not None and row_b is not None:
            d_atm = row_b["atmosphere"] - row_a["atmosphere"]
            d_res = row_b["reservoir"] - row_a["reservoir"]
            entry.update({
                "dW_atmosphere": d_atm,
                "dS_reservoir": d_res,
                "physics_net_e_minus_p": fixer - d_res,
                "residual": d_atm + d_res - fixer,
                "dW_condensate_by_species": {
                    s: row_b[f"water_{s}"] - row_a[f"water_{s}"] for s in CONDENSATE_SPECIES
                },
            })
        else:
            entry["note"] = (
                "the cold-start checkpoint has no ledger row; "
                "its interval's ledger budget is not formed"
            )
        intervals.append(entry)
    return {"rows": rows, "intervals": intervals}


def ledger_condensate_series(
    ledger: dict[str, np.ndarray], every_s: float = 1800.0
) -> dict[str, object]:
    """Global-mean condensate by species at a coarse cadence (kg/m2)."""
    times = ledger["time_s"]
    keep = [0]
    for i in range(1, times.size):
        if times[i] - times[keep[-1]] >= every_s - 1.0e-6:
            keep.append(i)
    if keep[-1] != times.size - 1:
        keep.append(times.size - 1)
    keep_arr = np.asarray(keep)
    return {
        "every_s": float(every_s),
        "time_s": [float(v) for v in times[keep_arr]],
        **{s: [float(v) for v in ledger[f"water_{s}"][keep_arr]] for s in WATER_SPECIES},
        "atmosphere": [float(v) for v in ledger["atmosphere"][keep_arr]],
        "max_abs_fixer_step_kg_m2": float(np.max(np.abs(ledger["fixer"]))),
        "fixer_total_kg_m2": float(np.sum(ledger["fixer"])),
    }


# --------------------------------------------------------------------------
# a run
# --------------------------------------------------------------------------


def measure_run(
    paths: list[str | Path],
    *,
    receipt: dict,
    ledger_path: str | Path | None = None,
    transport: bool = True,
    bands=BANDS,
    progress=None,
) -> dict[str, object]:
    reader = CheckpointReader(receipt, transport=transport)
    samples: list[ColumnSample] = []
    for path in sorted(Path(p) for p in paths):
        samples.append(reader.sample(path))
        if progress is not None:
            progress(f"read {path.name} step {samples[-1].step}")
    samples.sort(key=lambda s: s.time_s)
    steps = [int(s.step) for s in samples]
    ledger_block: dict[str, object] | None = None
    fixer = None
    if ledger_path is not None:
        ledger = read_ledger(ledger_path)
        block = ledger_intervals(ledger, steps)
        fixer = [iv["fixer_kg_m2"] for iv in block["intervals"]]
        checks = []
        for sample, row in zip(samples, block["rows"]):
            if row is None:
                continue
            atm = reader.grid.area_mean(sample.atmosphere)
            res = reader.grid.area_mean(sample.reservoir)
            checks.append({
                "step": int(sample.step),
                "atmosphere_checkpoint": atm,
                "atmosphere_ledger": row["atmosphere"],
                "atmosphere_relative_gap": (
                    (atm - row["atmosphere"]) / max(abs(row["atmosphere"]), 1.0e-30)
                ),
                "reservoir_checkpoint": res,
                "reservoir_ledger": row["reservoir"],
                "reservoir_relative_gap": (
                    (res - row["reservoir"]) / max(abs(row["reservoir"]), 1.0e-30)
                ),
            })
        ledger_block = {
            "path": str(ledger_path),
            "intervals": block["intervals"],
            "checks": checks,
            "condensate_series": ledger_condensate_series(ledger),
        }
    result = budget(samples, reader.grid, fixer_kg_m2=fixer, bands=bands)
    result["model_source"] = {
        "checkpoints": [str(Path(p)) for p in sorted(Path(p) for p in paths)],
        "config_hash": reader.config_hash,
        "grid": {"kind": "gaussian", "nlat": reader.grid.shape[0], "nlon": reader.grid.shape[1]},
        "synthesis": "numpy float64 from the checkpoint's complex64 coefficients",
        "reservoir": (
            "surface + soil (max(smois,0) x thickness x 1000 x land_fraction) + "
            "max(canopy,0) + max(snow,0) + outflow"
        ),
        "convective": CHECKPOINT_CONVECTIVE,
        "grid_scale": " + ".join(CHECKPOINT_GRID_SCALE_BUCKETS),
        "flux": CHECKPOINT_FLUX,
        "transport": (
            "int v (qv + condensate) dp/g, zonal mean per row" if transport else "not computed"
        ),
        "land_mask": f"{CHECKPOINT_LAND_FRACTION} >= 0.5",
    }
    result["ledger"] = ledger_block if ledger_block is not None else {"status": "absent"}
    return result


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def summary_line(result: dict[str, object], label: str = "") -> str:
    g = result["cumulative"][GLOBAL]
    rates = g["rates_mm_day"]
    head = f"water budget {label}: " if label else "water budget: "
    frac = g["residual_fraction"]
    return (
        head
        + f"{g['hours']:.0f} h global mm/day: E_books {_fmt(rates['E_books'])} "
        f"(E_flux {_fmt(rates['E_flux'])}), P_conv {_fmt(rates['P_conv'])}, "
        f"P_grid {_fmt(rates['P_grid'])}, dTPW {_fmt(rates['dW_vapor'])}, "
        f"dCondensate {_fmt(rates['dW_condensate'])}, residual {_fmt(rates['residual'], 4)} "
        f"({'n/a' if frac is None else f'{100.0 * frac:.2f}%'} of max(E, P))"
    )


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def synthetic_grid(truncation: int = 21) -> BudgetGrid:
    from woof.globe.spectral.grid import GaussianGrid

    return BudgetGrid.from_gaussian(GaussianGrid.create(int(truncation)))


def _pattern(grid: BudgetGrid) -> np.ndarray:
    """A smooth, strictly positive column pattern: 0.5 + cos(latitude)
    with a weak zonal wave."""
    lat = np.deg2rad(grid.latitude_deg)[:, None]
    lon = np.arange(grid.nlon)[None, :] * (2.0 * np.pi / grid.nlon)
    return 0.5 + np.cos(lat) * (1.0 + 0.2 * np.cos(lon))


def synthetic_pair(
    grid: BudgetGrid,
    *,
    p_conv_kg_m2: float = 0.0,
    p_grid_kg_m2: float = 0.0,
    e_kg_m2: float = 0.0,
    leak_kg_m2: float = 0.0,
    fixer_kg_m2: float = 0.0,
    interval_s: float = 10800.0,
    p_rows: np.ndarray | None = None,
    e_rows: np.ndarray | None = None,
    transport_v_m_s: float | None = None,
    vapor_kg_m2: float = 25.0,
    condensate_kg_m2: float = 0.2,
) -> tuple[ColumnSample, ColumnSample, dict[str, np.ndarray]]:
    """Two samples one interval apart with planted exchanges.

    Precipitation removes ``p`` from the condensate (the reservoir is
    credited and the accumulators advanced); evaporation adds ``e`` to the
    vapor (the reservoir debited, the flux field set to the constant rate);
    ``leak`` changes the atmosphere with no booking anywhere; ``fixer`` is
    added to the reservoir uniformly (the fixer's own move); planted rows
    restrict P or E to those rows; ``transport_v_m_s`` plants a uniform
    northward flux of a uniform column ``vapor_kg_m2`` whose analytic band
    convergence (at the instrument's own edges) is applied to the vapor.
    Returns the planted per-column fields as the third element.
    """
    ny, nx = grid.shape
    pattern = _pattern(grid)
    if transport_v_m_s is None:
        vapor0 = vapor_kg_m2 * pattern
    else:
        vapor0 = np.full((ny, nx), vapor_kg_m2)
    cond0 = {
        s: condensate_kg_m2 * pattern * (1.0 + 0.1 * i)
        for i, s in enumerate(CONDENSATE_SPECIES)
    }
    reservoir0 = np.full((ny, nx), 500.0) + 600.0 * (pattern > 1.0)
    conv0 = np.zeros((ny, nx))
    gridacc0 = np.zeros((ny, nx))
    p_mask = (
        np.ones((ny, nx), dtype=bool) if p_rows is None
        else np.repeat(np.asarray(p_rows, bool)[:, None], nx, axis=1)
    )
    e_mask = (
        np.ones((ny, nx), dtype=bool) if e_rows is None
        else np.repeat(np.asarray(e_rows, bool)[:, None], nx, axis=1)
    )
    p_conv = np.where(p_mask, p_conv_kg_m2 * pattern, 0.0)
    p_grid = np.where(p_mask, p_grid_kg_m2 * pattern, 0.0)
    p = p_conv + p_grid
    e = np.where(e_mask, e_kg_m2 * pattern, 0.0)
    leak = leak_kg_m2 * pattern
    # Precipitation leaves through the condensate (rain), evaporation
    # enters the vapor; the leak is split between vapor and condensate so
    # both change.
    cond1 = dict(cond0)
    cond1["qr"] = cond0["qr"] - p + 0.5 * leak
    vapor1 = vapor0 + e + 0.5 * leak
    transport_planted = np.zeros((ny, nx))
    # A transport field is always carried (zero unless planted) so the
    # band residuals form; a sample WITHOUT the field is the reader's
    # --no-transport case, exercised by the CLI test.
    northward0 = np.zeros((ny,))
    northward1 = np.zeros((ny,))
    if transport_v_m_s is not None:
        northward0 = np.full((ny,), transport_v_m_s * vapor_kg_m2)
        northward1 = northward0.copy()
        edges = band_edges(grid)
        rows = band_rows(grid)
        for name, (south, north) in edges.items():
            s, n = math.radians(south), math.radians(north)
            if not rows[name].any():
                continue
            # Inflow per unit band area: (F(south) - F(north)) / area with
            # F = 2 pi a cos(phi) v W and area = 2 pi a^2 (sin n - sin s).
            convergence = (
                transport_v_m_s * vapor_kg_m2 * (math.cos(s) - math.cos(n))
                / (grid.radius_m * (math.sin(n) - math.sin(s)))
            )
            transport_planted[rows[name], :] = convergence * interval_s
        vapor1 = vapor1 + transport_planted
    reservoir1 = reservoir0 + p - e + fixer_kg_m2
    flux0 = flux1 = e / interval_s
    land = pattern > 1.0
    before = ColumnSample(
        time_s=0.0, step=0, vapor=vapor0, condensate=cond0, reservoir=reservoir0,
        convective=conv0, grid_scale=gridacc0, flux_evaporation=flux0,
        northward_transport=northward0, land=land,
    )
    after = ColumnSample(
        time_s=interval_s, step=1, vapor=vapor1, condensate=cond1, reservoir=reservoir1,
        convective=conv0 + p_conv, grid_scale=gridacc0 + p_grid, flux_evaporation=flux1,
        northward_transport=northward1, land=land,
    )
    planted = {
        "P_conv": p_conv, "P_grid": p_grid, "E": e, "leak": leak,
        "transport": transport_planted,
    }
    return before, after, planted


def _read_regions(
    grid: BudgetGrid, before: ColumnSample, after: ColumnSample, fixer: float = 0.0
) -> dict:
    return interval_budget(before, after, grid, fixer_kg_m2=fixer)["regions"]


def calibrate() -> dict[str, object]:
    """Run the calibration families and return their rows."""
    grid = synthetic_grid(21)
    rows: list[dict[str, object]] = []
    for magnitude in (0.1, 1.0, 10.0):
        before, after, planted = synthetic_pair(
            grid, p_conv_kg_m2=0.3 * magnitude, p_grid_kg_m2=0.7 * magnitude
        )
        g = _read_regions(grid, before, after)[GLOBAL]
        p_true = grid.area_mean(planted["P_conv"] + planted["P_grid"])
        conv_true = grid.area_mean(planted["P_conv"])
        rows.append({
            "family": "A", "planted": "P sink, no E", "magnitude_kg_m2": magnitude,
            "P_planted": p_true, "P_read": g["P"], "P_conv_planted": conv_true,
            "P_conv_read": g["P_conv"], "P_grid_read": g["P_grid"],
            "E_read": g["E_books"], "E_flux_read": g["E_flux"], "residual": g["residual"],
            "P_relative_error": abs(g["P"] - p_true) / p_true,
            "P_conv_relative_error": abs(g["P_conv"] - conv_true) / conv_true,
            "E_over_P": abs(g["E_books"]) / p_true,
            "residual_over_P": abs(g["residual"]) / p_true,
        })
    for sign in (-1.0, 1.0):
        for magnitude in (0.1, 1.0, 10.0):
            before, after, planted = synthetic_pair(
                grid, p_conv_kg_m2=0.3, p_grid_kg_m2=0.7, leak_kg_m2=sign * magnitude
            )
            g = _read_regions(grid, before, after)[GLOBAL]
            leak = grid.area_mean(planted["leak"])
            rows.append({
                "family": "A'",
                "planted": "unbooked " + ("source" if sign > 0 else "sink") + " with P 1",
                "leak_planted": leak, "residual_read": g["residual"],
                "fraction_read": g["residual_fraction"],
                "leak_relative_error": abs(g["residual"] - leak) / abs(leak),
            })
    for magnitude in (0.1, 1.0, 10.0):
        before, after, planted = synthetic_pair(grid, e_kg_m2=magnitude)
        g = _read_regions(grid, before, after)[GLOBAL]
        e_true = grid.area_mean(planted["E"])
        rows.append({
            "family": "B", "planted": "E source, no P", "magnitude_kg_m2": magnitude,
            "E_planted": e_true, "E_read": g["E_books"], "E_flux_read": g["E_flux"],
            "P_read": g["P"], "residual": g["residual"],
            "E_relative_error": abs(g["E_books"] - e_true) / e_true,
            "E_flux_relative_error": abs(g["E_flux"] - e_true) / e_true,
            "residual_over_E": abs(g["residual"]) / e_true,
        })
    before, after, planted = synthetic_pair(grid, e_kg_m2=-0.5)
    g = _read_regions(grid, before, after)[GLOBAL]
    e_true = grid.area_mean(planted["E"])
    rows.append({
        "family": "B'", "planted": "dew (E < 0)", "E_planted": e_true, "E_read": g["E_books"],
        "E_flux_read": g["E_flux"], "residual": g["residual"],
        "E_relative_error": abs(g["E_books"] - e_true) / abs(e_true),
    })
    before, after, planted = synthetic_pair(grid, e_kg_m2=1.0, fixer_kg_m2=0.05)
    g = _read_regions(grid, before, after, fixer=0.05)[GLOBAL]
    e_true = grid.area_mean(planted["E"])
    rows.append({
        "family": "B''", "planted": "E 1 with fixer 0.05 on the reservoir",
        "E_planted": e_true, "E_read": g["E_books"], "residual": g["residual"],
        "E_relative_error": abs(g["E_books"] - e_true) / e_true,
    })
    north = grid.latitude_deg >= 0.0  # an equatorial row belongs to 0-30N
    before, after, planted = synthetic_pair(
        grid, p_conv_kg_m2=0.6, p_grid_kg_m2=1.4, e_kg_m2=2.0, p_rows=north, e_rows=~north
    )
    regions = _read_regions(grid, before, after)
    masks = region_masks(grid, None)
    for name, _s, _n in BANDS:
        r = regions[name]
        rows.append({
            "family": "C", "planted": "P north only, E south only", "band": name,
            "P_planted": grid.area_mean(planted["P_conv"] + planted["P_grid"], masks[name]),
            "P_read": r["P"],
            "E_planted": grid.area_mean(planted["E"], masks[name]), "E_read": r["E_books"],
            "residual": r["residual"],
            "residual_over_max": abs(r["residual"]) / max(abs(r["E_books"]), abs(r["P"])),
            "transport_read": r["transport_convergence"],
        })
    for truncation in (21, 63):
        tgrid = synthetic_grid(truncation)
        tmasks = region_masks(tgrid, None)
        for v in (1.0, -1.0):
            before, after, planted = synthetic_pair(tgrid, transport_v_m_s=v)
            regions = _read_regions(tgrid, before, after)
            for name, _s, _n in BANDS:
                r = regions[name]
                t_true = tgrid.area_mean(planted["transport"], tmasks[name])
                without = r["dW_atmosphere"] - r["physics_net_e_minus_p"]
                rows.append({
                    "family": "D",
                    "planted": f"uniform northward flux v {v:+.0f} m/s, W 25 kg/m2",
                    "truncation": truncation, "band": name,
                    "transport_planted": t_true, "transport_read": r["transport_convergence"],
                    "residual_without_transport": without,
                    "residual_with_transport": r["residual"],
                    "transport_relative_error": (
                        abs(r["transport_convergence"] - t_true) / abs(t_true)
                    ),
                })
    return {
        "schema": SCHEMA + "/calibration",
        "grid": "gaussian T21 (32x64) unless stated",
        "rows": rows,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="column water-budget instrument for WOOF global"
    )
    parser.add_argument(
        "--run-dir", help="directory of arwen_global_step*.npz checkpoints with the run receipt"
    )
    parser.add_argument(
        "--checkpoints", nargs="*",
        help="explicit checkpoint paths (else every step file in --run-dir)",
    )
    parser.add_argument("--receipt", help="run receipt JSON (else <run-dir>/arwen-global-receipt.json)")
    parser.add_argument(
        "--ledger",
        help="insitu.ndjson (else <run-dir>/insitu.ndjson when present; 'none' to skip)",
    )
    parser.add_argument(
        "--no-transport", action="store_true",
        help="skip the wind synthesis and the band transport term",
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--out", help="JSON output path")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="run the synthetic families and print their rows",
    )
    parser.add_argument("--quiet", action="store_true", help="no per-checkpoint progress on stderr")
    args = parser.parse_args(argv)

    if args.calibrate:
        payload = calibrate()
        for row in payload["rows"]:
            print(json.dumps(_finite(row), sort_keys=True))
        if args.out:
            _write_json(Path(args.out), _finite(payload))
        return 0

    if not args.run_dir and not args.checkpoints:
        parser.error("--run-dir or --checkpoints is required")
    paths = list(args.checkpoints or [])
    if args.run_dir:
        paths.extend(sorted(glob.glob(str(Path(args.run_dir) / "arwen_global_step*.npz"))))
    if not paths:
        parser.error("no checkpoints found")
    if args.receipt:
        with open(args.receipt, "r", encoding="utf-8") as stream:
            receipt = json.load(stream)
    elif args.run_dir:
        receipt = read_receipt(Path(args.run_dir))
    else:
        parser.error("--receipt is required with --checkpoints")
    ledger_path = None
    if args.ledger and args.ledger != "none":
        ledger_path = args.ledger
    elif not args.ledger and args.run_dir and (Path(args.run_dir) / LEDGER_NAME).exists():
        ledger_path = Path(args.run_dir) / LEDGER_NAME

    def progress(text: str) -> None:
        if not args.quiet:
            print(text, file=sys.stderr, flush=True)

    result = measure_run(
        paths, receipt=receipt, ledger_path=ledger_path,
        transport=not args.no_transport, progress=progress,
    )
    result["label"] = args.label
    result["summary"] = summary_line(result, args.label)
    result = _finite(result)
    if args.out:
        _write_json(Path(args.out), result)
    print(result["summary"])
    return 0


__all__ = [
    "BANDS", "BudgetGrid", "CALIBRATION", "CheckpointReader", "ColumnSample",
    "CONDENSATE_SPECIES", "SCHEMA", "band_convergence_kg_m2_s", "band_edges", "band_rows",
    "budget", "calibrate", "edge_flux_kg_s", "interval_budget", "ledger_condensate_series",
    "ledger_intervals", "measure_run", "read_ledger", "read_receipt", "region_masks",
    "summary_line", "synthetic_grid", "synthetic_pair",
]


if __name__ == "__main__":
    sys.exit(main())
