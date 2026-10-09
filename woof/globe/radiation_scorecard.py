"""Radiation scorecard: the radiation budget of a WOOF global run.

Reads a run's checkpoints and reports, per checkpoint and per region
(global, land, ocean, six latitude bands), the area-weighted broadband
fluxes the native suite holds between radiation buckets: surface net
shortwave (``gsw``), surface downward and upward longwave (``glw``,
``lwupb``), top-of-column outgoing longwave (``olr``), reflected and
incoming shortwave (``swupt``, ``swdnt``), the derived surface net
longwave, surface net radiation, top-of-column net radiation and the
atmospheric column's net radiation (top net less surface net, negative
where the column cools), the column cloud cover the scheme's overlap
implies (``cldfra_total``) and the daylit fraction; plus, from the
checkpointed atmosphere and Morrison size diagnostics, what the cloud
optics saw: cloud fractions (WRF's Xu-Randall ``cal_cldfra1`` in the
float64 mirror), liquid and ice water paths, the effective sizes and how
many cloudy cells and columns had sizes outside the cloud-optics tables'
domain (``woof.globe.core.npref.np_rrtmgp_hydrometeor_paths``, the same
arithmetic the device coupling runs), beside the run's own counters
when the checkpoint carries them.

Interval means.  When the run carries the radiation time integrals
(``physics__acc_*``, native runtime 2026-09-04 and later) the mean flux
between two checkpoints is their difference over the accumulated
seconds: exactly what the model applied.  A run without them (the
12e9aaf9f control) gets the trapezoid over the checkpoint samples of the
held planes, and every such row says ``sampling = "held planes at
checkpoint times"``; the two are not the same measurement and the JSON
never mixes them.

Reference.  The GFS/GDAS pgrb2 files the arms read carry, at every lead,
an instantaneous total, low, middle and high cloud cover (TCDC, LCDC,
MCDC, HCDC; product definition template 4.0) which the mapped engine
binds through ``woof/authorities/rw-wps-gfs-pgrb2-0p25-cloud-cover
.mapping.json``; the model's cover at the matching checkpoint is scored
against it (bias, rmse, area-weighted, on the model's grid).  Their
radiative fluxes (DSWRF, USWRF, DLWRF, ULWRF at the surface, USWRF and
ULWRF at the top; 18-24 h means in f024) are GRIB2 interval products
(template 4.8) which rw-wps.mapping.v1 cannot bind (the engine's
``assemble.rs:266`` refuses interval time semantics, and the analysis
file f000 carries no flux at all), so NO flux reference is read from
the case files and the JSON says so in ``reference.fluxes``.

Climatological ranges, the fallback the fluxes are held against, are
annual global means from the satellite and reanalysis literature
(CERES EBAF Ed4 2005-2015 and ERA5 1979-2018 order of magnitude, W/m2):
outgoing longwave 238-242, reflected shortwave 97-101, surface downward
longwave 342-348, surface net longwave -50 to -58 (up exceeds down),
surface net shortwave 160-168, net radiation of the atmospheric column
-95 to -115 (the top-of-atmosphere net near +1 less the surface net near
+105: the column's shortwave absorption of about 80 less its longwave
cooling of about 185, the deficit that latent and sensible heating
close), total cloud cover 0.62-0.68.  CAVEAT, carried on every range row: they
are annual, all-sky, multi-year means; a single September day of a
24 h forecast differs from them by the season (incoming 340 W/m2 is the
annual mean), the day's weather and the model's own spin-up, so a
reading inside the range is a sanity check, not a verification, and a
reading outside it names a candidate, not a defect.  The obs-skill
verification of the fluxes needs an interval-capable reference decode.

Calibration (``--calibrate``, ``CALIBRATION`` below; every family in
both directions): planted plane fields with closed-form area means
(constants, ``a + b sin^2 lat`` whose global mean is ``a + b/3`` under
the Gaussian quadrature exactly, hemispheres and bands to the quadrature
error the module records; zonal waves ``a + b cos lon`` and ``b sin lon``
against the land mask on half the longitudes, whose land and ocean means
are closed-form sums); planted accumulator differences over planted
seconds, and a zero-order-held diurnal signal on the model's own
radiation schedule where the integrals read the applied mean exactly and
the held-plane trapezoid's error is recorded; planted reference offsets
of both signs and planted noise (bias reads the offset, rmse its
magnitude), and the same through the reference regrid from a GFS-shaped
0.25 degree grid with descending latitude and under both longitude
conventions; planted layer fractions through the overlap (adjacent
layers read the larger, separated layers the random product); planted
sizes above and below the table bounds in known cell and column counts,
with planted cloud fractions weighting the radiative fractions and
counting the unsampled cells.

CLI::

    python -m woof.globe.radiation_scorecard --run-dir DIR --out JSON
        [--reference-grib FILE[,FILE...] --reference-step STEP ...]
        [--window-hours 18 24] [--no-optics] [--label TEXT]
    python -m woof.globe.radiation_scorecard --calibrate [--out JSON]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys

import numpy as np

from .constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    REFERENCE_PRESSURE_PA,
    STEFAN_BOLTZMANN,
)
from .water_budget import (
    BANDS,
    GLOBAL,
    LAND,
    OCEAN,
    BudgetGrid,
    read_receipt,
    region_masks,
    synthetic_grid,
)

CHECKPOINT_LAND_FRACTION = "surface__land_fraction"
CHECKPOINT_SKIN = "surface__surface_temperature_k"
CHECKPOINT_EMISSIVITY = "surface__surface_emissivity"
CHECKPOINT_ALBEDO = "surface__surface_albedo"

#: Held radiation planes, in the checkpoint's ``physics__`` namespace.
HELD_PLANES = ("swdown", "gsw", "glw", "olr", "coszen",
               "swupt", "swdnt", "lwupb", "cldfra_total")
#: The four planes every native run since the level5 tip carries.
REQUIRED_PLANES = ("swdown", "gsw", "glw", "olr")
#: Time integrals (physics__acc_*): name -> the flux they integrate.
ACCUMULATORS = {
    "acc_sw_down_surface_j_m2": "sw_down_surface",
    "acc_sw_up_surface_j_m2": "sw_up_surface",
    "acc_lw_down_surface_j_m2": "lw_down_surface",
    "acc_lw_up_surface_j_m2": "lw_up_surface",
    "acc_sw_up_top_j_m2": "sw_up_top",
    "acc_sw_down_top_j_m2": "sw_down_top",
    "acc_lw_up_top_j_m2": "lw_up_top",
    "acc_total_cloud_cover_s": "total_cloud_cover",
}
ACCUMULATED_SECONDS_KEY = "radiation_accumulated_s"

#: Flux names of the scorecard (W/m2 unless stated), positive as named.
FLUXES = (
    "sw_down_surface", "sw_up_surface", "sw_net_surface",
    "lw_down_surface", "lw_up_surface", "lw_net_surface",
    "net_surface",
    "sw_down_top", "sw_up_top", "lw_up_top", "net_top",
    "atmosphere_net",
    "total_cloud_cover", "daylit_fraction",
)

#: Annual global-mean ranges (see the module docstring for the caveat).
CLIMATOLOGICAL_RANGES = {
    "lw_up_top": (238.0, 242.0),
    "sw_up_top": (97.0, 101.0),
    "lw_down_surface": (342.0, 348.0),
    "lw_net_surface": (-58.0, -50.0),
    "sw_net_surface": (160.0, 168.0),
    "atmosphere_net": (-115.0, -95.0),
    "total_cloud_cover": (0.62, 0.68),
}
CLIMATOLOGY_CAVEAT = (
    "annual, all-sky, multi-year global means (CERES EBAF Ed4 2005-2015, "
    "ERA5 1979-2018 order); a single September forecast day differs by "
    "season, weather and spin-up, so inside the range is a sanity check, "
    "not a verification, and outside it is a candidate, not a defect")

#: Table domain of the shipped RRTMGP cloud optics (liquid radius, ice
#: diameter, microns), read by the device driver off the netCDF.
SIZE_BOUNDS = (2.5, 21.5, 10.0, 180.0)
#: SW band 11 (16000-22650 cm-1, the visible), zero-based, in the shipped
#: 14-band SW cloud table.
ICE_VISIBLE_BAND_INDEX = 10
SIZE_BOUNDING_FIELDS = (
    "liquid_cells", "liquid_below_cells", "liquid_above_cells",
    "liquid_below_columns", "liquid_above_columns",
    "liquid_above_path_fraction",
    "ice_cells", "ice_below_cells", "ice_above_cells",
    "ice_below_columns", "ice_above_columns",
    "ice_above_path_fraction",
    "liquid_sentinel_cells", "ice_sentinel_cells",
    "liquid_above_radiative_fraction", "ice_above_radiative_fraction",
    "liquid_unsampled_cells", "ice_unsampled_cells",
    "liquid_sentinel_radiative_fraction", "ice_sentinel_radiative_fraction")

REFERENCE_MAPPING = "rw-wps-gfs-pgrb2-0p25-cloud-cover.mapping.json"
REFERENCE_COVER_FIELDS = {
    "total_cloud_cover": "total_cloud_cover",
    "low_cloud_cover": "low_cloud_cover",
    "middle_cloud_cover": "middle_cloud_cover",
    "high_cloud_cover": "high_cloud_cover",
}

HELD_SAMPLING = "held planes at checkpoint times"
ACCUMULATED_SAMPLING = "time integrals over every physics call"


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------


def reference_mapping_path() -> Path:
    """The reference cloud-cover mapping, from whichever table has it.

    One resolver answers every mapping this package names
    (:func:`woof.globe.analysis_initial.resolve_analysis_mapping`): the
    engine's table first, this package's carried copies second.
    """

    from .analysis_initial import resolve_analysis_mapping

    return resolve_analysis_mapping(REFERENCE_MAPPING)


@dataclass
class RadiationSample:
    """One checkpoint's radiation view on the run's grid, float64."""

    time_s: float
    step: int
    planes: dict[str, np.ndarray]
    accumulators: dict[str, np.ndarray]
    accumulated_s: float | None
    skin_k: np.ndarray
    emissivity: np.ndarray
    land: np.ndarray
    metadata: dict = field(default_factory=dict)
    optics: dict | None = None
    notes: tuple[str, ...] = ()

    def flux(self, name: str) -> np.ndarray | None:
        """A scorecard flux plane from the held planes, None if the
        checkpoint lacks a carrier it needs."""
        p = self.planes
        if name == "sw_down_surface":
            return p.get("swdown")
        if name == "sw_net_surface":
            return p.get("gsw")
        if name == "sw_up_surface":
            return None if "swdown" not in p or "gsw" not in p else p["swdown"] - p["gsw"]
        if name == "lw_down_surface":
            return p.get("glw")
        if name == "lw_up_surface":
            return p.get("lwupb")
        if name == "lw_net_surface":
            return None if "glw" not in p or "lwupb" not in p else p["glw"] - p["lwupb"]
        if name == "net_surface":
            if any(k not in p for k in ("gsw", "glw", "lwupb")):
                return None
            return p["gsw"] + p["glw"] - p["lwupb"]
        if name == "sw_down_top":
            return p.get("swdnt")
        if name == "sw_up_top":
            return p.get("swupt")
        if name == "lw_up_top":
            return p.get("olr")
        if name == "net_top":
            if any(k not in p for k in ("swdnt", "swupt", "olr")):
                return None
            return p["swdnt"] - p["swupt"] - p["olr"]
        if name == "atmosphere_net":
            top, sfc = self.flux("net_top"), self.flux("net_surface")
            return None if top is None or sfc is None else top - sfc
        if name == "total_cloud_cover":
            return p.get("cldfra_total")
        if name == "daylit_fraction":
            return None if "coszen" not in p else (p["coszen"] > 0.0).astype(np.float64)
        raise KeyError(name)


class ColdStartCheckpoint(ValueError):
    """The step-0 checkpoint of a cold start: no physics call has run, so
    the physics namespace is empty and there is nothing to read."""


class RadiationReader:
    """Reads a run's checkpoints into :class:`RadiationSample` on the run's
    own Gaussian grid (from the receipt's transform block)."""

    def __init__(self, grid: BudgetGrid, *, vertical=None, loader=None, optics: bool = True):
        self.grid = grid
        self.vertical = vertical
        self.optics = bool(optics)
        if loader is None:
            from .checkpoint import read_checkpoint

            loader = read_checkpoint
        self._load = loader

    @classmethod
    def from_receipt(cls, receipt: dict, **kwargs) -> "RadiationReader":
        from .vertical import HybridCoordinate

        cfg = receipt["config"]
        block = receipt["transform"]
        grid = BudgetGrid.for_shape(int(block["nlat"]), int(block["nlon"]))
        if abs(float(grid.radius_m) - float(block["radius_m"])) > 1.0:
            grid = BudgetGrid(grid.latitude_deg, grid.quadrature_weights, grid.nlon,
                              float(block["radius_m"]))
        vertical = HybridCoordinate(
            np.asarray(cfg["a_half_pa"], dtype=np.float64),
            np.asarray(cfg["b_half"], dtype=np.float64),
        )
        return cls(grid, vertical=vertical, **kwargs)

    def sample(self, path: str | Path) -> RadiationSample:
        metadata, arrays = self._load(path)
        notes: list[str] = []
        physics = {
            k.removeprefix("physics__"): v for k, v in arrays.items() if k.startswith("physics__")
        }
        planes = {}
        for name in HELD_PLANES:
            if name in physics:
                planes[name] = np.asarray(physics[name], dtype=np.float64)
        step = int(metadata["step"])
        if step == 0 and not any(name in physics for name in REQUIRED_PLANES):
            # No physics call has run.  The namespace may still carry the
            # cold start's seeded surface store (Noah's snow planes, written
            # at step 0 since the cold-start seeding), which is not a
            # radiation call; the held planes are what the reader needs.
            raise ColdStartCheckpoint(
                f"{path}: the step-0 checkpoint of a cold start carries no held "
                "radiation plane (no physics call has run); the scorecard starts "
                "at the first checkpoint after it")
        missing = [name for name in REQUIRED_PLANES if name not in planes]
        if missing:
            raise ValueError(
                f"{path}: checkpoint step {step} carries no held radiation "
                f"plane(s) {missing}; the scorecard reads native-suite runs")
        skin = np.asarray(arrays[CHECKPOINT_SKIN], dtype=np.float64)
        emissivity = np.asarray(arrays[CHECKPOINT_EMISSIVITY], dtype=np.float64)
        physics_metadata = dict(metadata.get("physics_metadata") or {})
        if "lwupb" not in planes:
            # The control era held no surface upward longwave: the grey-body
            # formula on the checkpointed skin temperature and emissivity,
            # labelled as such (the runtime applies the same formula only
            # for a scheme that publishes none).
            planes["lwupb"] = (
                emissivity * STEFAN_BOLTZMANN * skin ** 4 + (1.0 - emissivity) * planes["glw"]
            )
            notes.append(
                f"step {step}: lw_up_surface from the surface emission formula "
                "(no held plane in this checkpoint)")
            physics_metadata.setdefault("lwupb_source", "surface_emission_formula (scorecard)")
        for name in ("swupt", "swdnt", "cldfra_total"):
            if name not in planes:
                notes.append(f"step {step}: no held {name} plane (rows needing it are INCOMPLETE)")
        accumulators = {
            name: np.asarray(physics[name], dtype=np.float64)
            for name in ACCUMULATORS if name in physics
        }
        accumulated_s = physics_metadata.get(ACCUMULATED_SECONDS_KEY)
        accumulated_s = None if accumulated_s is None else float(accumulated_s)
        land = np.asarray(arrays[CHECKPOINT_LAND_FRACTION], dtype=np.float64) >= 0.5
        optics = None
        if self.optics:
            try:
                optics = self.cloud_optics_view(arrays, physics, physics_metadata)
            except KeyError as exc:
                notes.append(f"step {step}: no cloud-optics view ({exc})")
            except ValueError as exc:
                # the pairing refusal: named in the record, never a count
                notes.append(f"step {step}: cloud-optics view REFUSED ({exc})")
        if "cldfra_total" not in planes and optics is not None:
            planes["cldfra_total"] = optics.pop("_total_cloud_cover_plane")
            notes.append(
                f"step {step}: total_cloud_cover recomputed from the checkpointed "
                "state (cal_cldfra1 + maximum-random overlap), not a held plane")
        elif optics is not None:
            optics.pop("_total_cloud_cover_plane", None)
        return RadiationSample(
            time_s=float(metadata["time_s"]), step=step, planes=planes,
            accumulators=accumulators, accumulated_s=accumulated_s,
            skin_k=skin, emissivity=emissivity, land=land,
            metadata=physics_metadata, optics=optics, notes=tuple(notes),
        )

    # -- the cloud optics view ----------------------------------------------

    def cloud_optics_view(self, arrays: dict, physics: dict, physics_metadata: dict | None = None) -> dict:
        """What the radiation coupling sees in this checkpoint: cloud
        fractions, in-cloud paths, sizes and the size bounding, through the
        float64 mirror of the device coupling."""
        from woof.globe.core.npref import (
            np_cal_cldfra1, np_rrtmgp_hydrometeor_paths)

        from .core.npref import np_max_random_total_cloud_cover

        if self.vertical is None:
            raise KeyError("no vertical coordinate (reader built without a receipt)")
        grid_field = self._grid_field
        logps = grid_field(arrays["atmosphere__log_surface_pressure"])
        ps = np.exp(logps)
        pressure = self.vertical.pressure(ps, _NumpyBackend())
        p_full = np.asarray(pressure["p_full"], dtype=np.float64)
        p_half = np.asarray(pressure["p_half"], dtype=np.float64)
        theta = grid_field(arrays["atmosphere__theta"])
        exner = (p_full / REFERENCE_PRESSURE_PA) ** (DRY_AIR_GAS_CONSTANT / DRY_AIR_CP)
        temperature = theta * exner
        q = {name: grid_field(arrays[f"atmosphere__{name}"]) for name in ("qv", "qc", "qi", "qs")}
        numbers = {name: grid_field(arrays[f"atmosphere__{name}"]) for name in ("nc", "nr", "ni", "ns")}
        nz, ny, nx = theta.shape

        def columns(volume):
            return np.ascontiguousarray(volume.transpose(1, 2, 0).reshape(ny * nx, nz))

        cldfra = np_cal_cldfra1(
            columns(q["qv"]), columns(q["qc"]), columns(q["qi"]), columns(q["qs"]),
            columns(temperature), columns(p_full), f_qc=True, f_qi=True, f_qs=True)
        plev = np.ascontiguousarray(p_half.transpose(1, 2, 0).reshape(ny * nx, nz + 1))
        effective = {}
        for name in ("effc", "effr", "effi", "effs"):
            if name in physics:
                effective[name] = columns(physics_volume_in_atmosphere_order(physics[name]))
        updates = int(dict(physics_metadata or {}).get("microphysics_updates", 1))
        kwargs = dict(play=columns(p_full), tlay=columns(temperature),
                      nc=columns(numbers["nc"]), nr=columns(numbers["nr"]),
                      ni=columns(numbers["ni"]), ns=columns(numbers["ns"]))
        radii_source = "number-moment reconstruction"
        radii_default_share = None
        if len(effective) == 4 and updates > 0:
            radii_default_share = radii_pairing_check(
                effective["effc"], columns(q["qc"]), effective["effs"], columns(q["qs"]))
            kwargs.update(effective)
            radii_source = "Morrison effc/effi/effs diagnostics"
        # The shipped coupling (area-conserving ice and snow merge, the
        # geometric carry above the table's upper bound) and the coupling
        # of every tree before 2026-09-04 (number-weighted merge, bare
        # clip), on the same state, so the before stands beside the after.
        paths = np_rrtmgp_hydrometeor_paths(
            plev, columns(q["qc"]), None, columns(q["qi"]), columns(q["qs"]),
            microphysics="morrison", cldfra=cldfra, size_bounds=SIZE_BOUNDS, **kwargs)
        before = np_rrtmgp_hydrometeor_paths(
            plev, columns(q["qc"]), None, columns(q["qi"]), columns(q["qs"]),
            microphysics="morrison", cldfra=cldfra, size_bounds=SIZE_BOUNDS,
            ice_merge="number", size_treatment="clip", sentinel_fallback=False,
            **kwargs)
        total_cover = np_max_random_total_cloud_cover(cldfra).reshape(ny, nx)
        weights = self.grid.cell_weights().reshape(-1)
        liquid = paths.clwp > 0.0
        ice = paths.ciwp > 0.0
        # In-cloud ice optical depth in the visible band, per column and
        # area-weighted, under both couplings: the radiative weight of the
        # change the counts describe.  (Column sums of the grid-mean tau,
        # in-cloud tau times the layer's cloud fraction.)
        ice_tau_after = np.sum(self._ice_visible_extinction(paths.dgice) * paths.ciwp * cldfra, axis=1)
        ice_tau_before = np.sum(self._ice_visible_extinction(before.dgice) * before.ciwp * cldfra, axis=1)
        grid_lwp = np.sum(columns(q["qc"]) * np.abs(np.diff(plev, axis=1)) * (1000.0 / GRAVITY_M_S2), axis=1)
        grid_iwp = np.sum((columns(q["qi"]) + columns(q["qs"])) * np.abs(np.diff(plev, axis=1))
                          * (1000.0 / GRAVITY_M_S2), axis=1)

        def wmean(values):
            return float(np.sum(weights * values) / np.sum(weights))

        def percentiles(values, mask):
            picked = values[mask]
            if picked.size == 0:
                return None
            return {f"p{p}": float(v) for p, v in zip((5, 50, 95), np.percentile(picked, (5, 50, 95)))}

        view = {
            "radii_source": radii_source,
            "radii_default_share_where_condensate": radii_default_share,
            "size_bounds": {"radliq_lwr": SIZE_BOUNDS[0], "radliq_upr": SIZE_BOUNDS[1],
                            "diamice_lwr": SIZE_BOUNDS[2], "diamice_upr": SIZE_BOUNDS[3]},
            "size_bounding": dict(paths.size_bounding),
            "size_bounding_before": dict(before.size_bounding),
            "coupling": {
                "after": ("solid-ice-equivalent sizes (2 re rho / 917), area-conserving ice+snow merge, "
                          "moment radii where the scheme left its no-mass sentinel beside mass, "
                          "geometric carry above the upper bound, clip below"),
                "before": ("number-weighted ice+snow merge at the sphere-equivalent sizes, the sentinel "
                           "taken at face value, clip to the bounds (every tree before 2026-09-04)"),
            },
            "ice_visible_optical_depth": {
                "band": "RRTMGP SW band 11, 16000-22650 cm-1 (442-625 nm), medium roughness",
                "grid_mean_column_after": wmean(ice_tau_after),
                "grid_mean_column_before": wmean(ice_tau_before),
                "grid_mean_column_change_fraction": (
                    (wmean(ice_tau_after) - wmean(ice_tau_before)) / wmean(ice_tau_before)
                    if wmean(ice_tau_before) > 0.0 else None),
            },
            "ice_diameter_before_bounding_um": percentiles(
                2.0 * self._unbounded_ice_radius(paths, before), ice),
            "columns": int(ny * nx),
            "cells": int(ny * nx * nz),
            "cloudy_layer_fraction": float(np.mean(cldfra > 0.0)),
            "mean_layer_cloud_fraction_where_cloudy": (
                float(np.mean(cldfra[cldfra > 0.0])) if np.any(cldfra > 0.0) else 0.0),
            "grid_mean_liquid_water_path_g_m2": wmean(grid_lwp),
            "grid_mean_ice_plus_snow_water_path_g_m2": wmean(grid_iwp),
            "in_cloud_liquid_path_g_m2_per_layer": percentiles(paths.clwp, liquid),
            "in_cloud_ice_path_g_m2_per_layer": percentiles(paths.ciwp, ice),
            "liquid_radius_um": percentiles(paths.reliq, liquid),
            "ice_diameter_um": percentiles(paths.dgice, ice),
            "_total_cloud_cover_plane": total_cover,
        }
        if len(effective) == 4:
            frozen = (columns(q["qi"]) + columns(q["qs"])) > 0.0
            snow = columns(q["qs"]) > 0.0
            view["morrison_effc_um_where_liquid"] = percentiles(effective["effc"], liquid)
            view["morrison_effi_um_where_ice"] = percentiles(effective["effi"], columns(q["qi"]) > 0.0)
            view["morrison_effs_um_where_snow"] = percentiles(effective["effs"], snow)
            view["snow_share_of_frozen_mass"] = float(
                np.sum(columns(q["qs"])[frozen]) / max(np.sum((columns(q["qi"]) + columns(q["qs"]))[frozen]), 1e-30))
        return view

    _sw_cloud_tables = None

    def _ice_visible_extinction(self, dgice: np.ndarray) -> np.ndarray:
        """Mass extinction (m2/g) of the shipped SW cloud table's ice at
        ``dgice`` (already inside the table domain) in the visible band,
        the table's own linear interpolation in diameter."""
        if RadiationReader._sw_cloud_tables is None:
            from woof.globe.core.rrtmgp import load_cloud_tables

            RadiationReader._sw_cloud_tables = load_cloud_tables("sw")
        tables = RadiationReader._sw_cloud_tables
        ext = np.asarray(tables.extice[:, ICE_VISIBLE_BAND_INDEX, 1], dtype=np.float64)
        pos = (np.asarray(dgice, dtype=np.float64) - tables.diamice_lwr) / tables.ice_step_size
        index = np.clip(np.floor(pos).astype(np.int64), 0, ext.shape[0] - 2)
        fraction = pos - index
        return ext[index] + fraction * (ext[index + 1] - ext[index])

    @staticmethod
    def _unbounded_ice_radius(after, before) -> np.ndarray:
        """The area-conserving merged ice DIAMETER before bounding, recovered
        from the carried path: the carry scales the in-cloud path by
        bound/size, so size = bound * (path before) / (path after) wherever
        the bound was crossed, and the bounded size elsewhere.  ``before``
        is the clip-only coupling whose path is the unscaled one."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(after.ciwp > 0.0, before.ciwp / after.ciwp, 1.0)
        return 0.5 * np.where(ratio > 1.0, after.dgice * ratio, after.dgice)

    def _grid_field(self, value: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(value):
            if not hasattr(self, "_transform"):
                self._transform = self._build_transform()
            return np.asarray(self._transform.inverse(value.astype(np.complex128)), dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    def _build_transform(self):
        from woof.globe.spectral.transform import SphericalHarmonicTransform

        nlat, nlon = self.grid.shape
        truncation = int(round((2 * nlat - 1) / 3.0)) - 1
        # The receipt's truncation is authoritative when a receipt built us.
        truncation = getattr(self, "_truncation", truncation)
        return SphericalHarmonicTransform.create(
            truncation, nlat=nlat, nlon=nlon, dealias_factor=1.5,
            radius_m=self.grid.radius_m, backend="numpy", precision="float64")


#: The Morrison kernel's radius where a species has no mass (microns):
#: the value every cell without that species carries in effc/effr/effi/effs.
MORRISON_NO_MASS_RADIUS_UM = 25.0


def physics_volume_in_atmosphere_order(volume: np.ndarray) -> np.ndarray:
    """A checkpointed ``physics__`` volume in the atmosphere's level order.

    The native suite works on the column batch, which reverses the
    model's vertical order (NativeColumnBatch.from_exchange: index 0 is
    the surface), and the physics namespace is checkpointed in that order,
    while the atmosphere fields are checkpointed in the model's order
    (index 0 at the top).  Pairing the two without this flip put every
    cloud layer beside a clear layer's no-mass radius.
    """
    return np.asarray(volume, dtype=np.float64)[::-1]


def radii_pairing_check(effc, qc, effs, qs) -> float:
    """The share of cells with condensate whose paired radius is the
    kernel's no-mass value.  A right pairing reads it near zero (the
    kernel computes a radius wherever the species has mass); above one
    half the radii volumes and the tracers do not describe the same
    levels, and the view refuses rather than count the clip on ghosts."""
    shares = []
    for radius, mass in ((effc, qc), (effs, qs)):
        with_mass = mass > 0.0
        if np.any(with_mass):
            shares.append(float(np.mean(radius[with_mass] == MORRISON_NO_MASS_RADIUS_UM)))
    share = max(shares) if shares else 0.0
    if share > 0.5:
        raise ValueError(
            f"{100 * share:.0f} percent of the cells with condensate carry the "
            "kernel's no-mass radius: the checkpointed radii volumes and the "
            "tracers are not paired level for level (the physics namespace is "
            "checkpointed in the column batch's bottom-to-top order, the "
            "atmosphere top-to-bottom); the cloud-optics view refuses to count "
            "the size bounding on a mispaired state")
    return share


class _NumpyBackend:
    """The minimal backend HybridCoordinate.pressure needs, on numpy."""

    xp = np
    float_dtype = np.float64

    def asarray(self, value, dtype=None):
        return np.asarray(value, dtype=dtype or self.float_dtype)


# --------------------------------------------------------------------------
# the scorecard
# --------------------------------------------------------------------------


def region_means(grid: BudgetGrid, plane: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float | None]:
    """Area-weighted mean of ``plane`` per region; a region with no cells
    (an all-ocean grid's land) reads None, never a number."""
    return {name: (grid.area_mean(plane, mask) if bool(np.any(mask)) else None)
            for name, mask in masks.items()}


def instantaneous_rows(sample: RadiationSample, grid: BudgetGrid, masks: dict[str, np.ndarray]) -> dict:
    """Every scorecard flux of one checkpoint, area-weighted per region;
    a flux the checkpoint cannot form is INCOMPLETE, never a number."""
    rows = {}
    for name in FLUXES:
        plane = sample.flux(name)
        if plane is None:
            rows[name] = {"status": "INCOMPLETE", "reason": "carrier plane absent from the checkpoint"}
        else:
            rows[name] = {"status": "measured", **region_means(grid, plane, masks)}
    return rows


def accumulated_interval(before: RadiationSample, after: RadiationSample,
                         grid: BudgetGrid, masks: dict[str, np.ndarray]) -> dict | None:
    """Interval-mean fluxes from the time integrals; None when either
    checkpoint lacks them."""
    if not before.accumulators or not after.accumulators:
        return None
    if before.accumulated_s is None or after.accumulated_s is None:
        return None
    seconds = float(after.accumulated_s) - float(before.accumulated_s)
    if seconds <= 0.0:
        return None
    means = {}
    for accumulator, flux in ACCUMULATORS.items():
        if accumulator in before.accumulators and accumulator in after.accumulators:
            plane = (after.accumulators[accumulator] - before.accumulators[accumulator]) / seconds
            means[flux] = plane
    derived = {}
    if "sw_down_surface" in means and "sw_up_surface" in means:
        derived["sw_net_surface"] = means["sw_down_surface"] - means["sw_up_surface"]
    if "lw_down_surface" in means and "lw_up_surface" in means:
        derived["lw_net_surface"] = means["lw_down_surface"] - means["lw_up_surface"]
    if "sw_net_surface" in derived and "lw_net_surface" in derived:
        derived["net_surface"] = derived["sw_net_surface"] + derived["lw_net_surface"]
    if all(k in means for k in ("sw_down_top", "sw_up_top", "lw_up_top")):
        derived["net_top"] = means["sw_down_top"] - means["sw_up_top"] - means["lw_up_top"]
    if "net_top" in derived and "net_surface" in derived:
        derived["atmosphere_net"] = derived["net_top"] - derived["net_surface"]
    means.update(derived)
    rows = {}
    for name in FLUXES:
        if name in means:
            rows[name] = {"status": "measured", **region_means(grid, means[name], masks)}
        elif name == "daylit_fraction":
            continue
        else:
            rows[name] = {"status": "INCOMPLETE", "reason": "time integral absent"}
    return {
        "sampling": ACCUMULATED_SAMPLING,
        "from_step": before.step, "to_step": after.step,
        "from_time_s": before.time_s, "to_time_s": after.time_s,
        "accumulated_s": seconds,
        "fluxes": rows,
    }


def held_interval(samples: list[RadiationSample], grid: BudgetGrid,
                  masks: dict[str, np.ndarray]) -> dict:
    """Trapezoid time mean of the held planes over the samples' span: the
    control-era fallback, labelled as a sampling of held planes."""
    if len(samples) < 2:
        raise ValueError("a held-plane interval needs at least two checkpoints")
    times = np.array([s.time_s for s in samples])
    span = float(times[-1] - times[0])
    rows = {}
    for name in FLUXES:
        planes = [s.flux(name) for s in samples]
        if any(p is None for p in planes):
            rows[name] = {"status": "INCOMPLETE", "reason": "carrier plane absent from a checkpoint"}
            continue
        stack = np.stack(planes)
        mean = np.zeros_like(stack[0])
        for i in range(len(samples) - 1):
            mean += 0.5 * (stack[i] + stack[i + 1]) * (times[i + 1] - times[i])
        mean /= span
        rows[name] = {"status": "measured", **region_means(grid, mean, masks)}
    return {
        "sampling": HELD_SAMPLING,
        "from_step": samples[0].step, "to_step": samples[-1].step,
        "from_time_s": float(times[0]), "to_time_s": float(times[-1]),
        "span_s": span, "samples": len(samples),
        "fluxes": rows,
    }


def compare_to_ranges(rows: dict) -> dict:
    out = {"caveat": CLIMATOLOGY_CAVEAT, "readings": {}}
    for name, (lower, upper) in CLIMATOLOGICAL_RANGES.items():
        row = rows.get(name)
        if row is None or row.get("status") != "measured" or row.get(GLOBAL) is None:
            out["readings"][name] = {"status": "INCOMPLETE", "range": [lower, upper]}
            continue
        value = float(row[GLOBAL])
        out["readings"][name] = {
            "status": "measured", "global": value, "range": [lower, upper],
            "inside": bool(lower <= value <= upper),
            "distance": 0.0 if lower <= value <= upper else (value - upper if value > upper else value - lower),
        }
    return out


def score_against_reference(model: np.ndarray, reference: np.ndarray, grid: BudgetGrid,
                            masks: dict[str, np.ndarray]) -> dict:
    """Area-weighted bias and rmse of ``model - reference`` per region."""
    diff = np.asarray(model, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    out = {}
    for name, mask in masks.items():
        if not bool(np.any(mask)):
            out[name] = {"status": "INCOMPLETE", "reason": "region selects no cells"}
            continue
        bias = grid.area_mean(diff, mask)
        rmse = math.sqrt(grid.area_mean(diff * diff, mask))
        out[name] = {"bias": bias, "rmse": rmse, "model_mean": grid.area_mean(model, mask),
                     "reference_mean": grid.area_mean(reference, mask)}
    return out


def regrid_reference(field: np.ndarray, ref_lat: np.ndarray, ref_lon: np.ndarray,
                     lat2d: np.ndarray, lon2d: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a regular lat-lon field onto the model's
    grid, periodic in longitude (the same construction the bias tools use)."""
    from scipy.interpolate import RegularGridInterpolator

    lat = np.asarray(ref_lat, dtype=np.float64)
    lon = np.asarray(ref_lon, dtype=np.float64) % 360.0
    if lat.ndim == 2:
        lat, lon = lat[:, 0], lon[0, :]
    order = np.argsort(lat)
    lon_order = np.argsort(lon)
    f = np.asarray(field, dtype=np.float64)[order][:, lon_order]
    f = np.concatenate([f, f[:, :1]], axis=1)
    lons = np.concatenate([lon[lon_order], [lon[lon_order][0] + 360.0]])
    interpolator = RegularGridInterpolator((lat[order], lons), f, bounds_error=False, fill_value=None)
    points = np.stack([lat2d.ravel(), lon2d.ravel() % 360.0], axis=1)
    return interpolator(points).reshape(lat2d.shape)


def model_lat_lon(grid: BudgetGrid) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = grid.shape
    lon = np.arange(nx) * (360.0 / nx)
    return (np.repeat(grid.latitude_deg[:, None], nx, axis=1),
            np.repeat(lon[None, :], ny, axis=0))


def decode_reference_cover(grib_paths: list[str], mapping: str | Path | None = None) -> list[dict]:
    """The instantaneous cloud covers of every product in ``grib_paths``
    through the mapped engine, one product at a time: one dict per
    product, either a frame (valid_time, lat, lon, fields) or, when the
    engine refuses that product, ``{"status": "INCOMPLETE", "reason":
    ...}`` naming the refusal, so one product that does not decode does
    not take the others down and the refusal is stated where its
    checkpoint is scored."""
    from .mapped_source_compat import decode_through_engine

    if mapping is None:
        mapping = reference_mapping_path()
    out = []
    for path in grib_paths:
        try:
            decoded = decode_through_engine(str(mapping), [str(path)])
            frames = decoded.frames
        except (ValueError, RuntimeError, OSError) as exc:
            out.append({"status": "INCOMPLETE", "product": str(path),
                        "reason": f"the mapped engine refused {path}: {exc}"})
            continue
        if not frames:
            out.append({"status": "INCOMPLETE", "product": str(path),
                        "reason": f"the mapped engine decoded no frame from {path}"})
            continue
        frame = frames[0]
        fields = {}
        for name in REFERENCE_COVER_FIELDS:
            values = np.asarray(frame.fields[name].values, dtype=np.float64)
            fields[name] = values[-1] if values.ndim == 3 else values
        out.append({
            "status": "measured",
            "product": str(path),
            "valid_time": str(frame.valid_time),
            "latitude": np.asarray(frame.latitude, dtype=np.float64),
            "longitude": np.asarray(frame.longitude, dtype=np.float64),
            "fields": fields,
            # The cover mapping declares two masked surface records, so on
            # a published engine this decode runs the validator adaptation;
            # the receipt travels to the checkpoint entry that is scored
            # against it rather than being dropped here.
            "decode": decoded.receipt,
        })
    return out


def scorecard(samples: list[RadiationSample], grid: BudgetGrid, *, label: str = "",
              window_hours: tuple[float, float] | None = None,
              reference_frames: dict[int, dict] | None = None) -> dict:
    """The scorecard of a run from its samples (sorted by time).

    ``reference_frames`` maps a checkpoint step to a decoded reference
    frame (see :func:`decode_reference_cover`) valid at that step.
    """
    samples = sorted(samples, key=lambda s: s.time_s)
    land = samples[-1].land
    masks = region_masks(grid, land)
    per_checkpoint = []
    for sample in samples:
        row = {
            "step": sample.step, "time_s": sample.time_s, "hour": sample.time_s / 3600.0,
            "fluxes": instantaneous_rows(sample, grid, masks),
            "notes": list(sample.notes),
            "radiation_calls": sample.metadata.get("radiation_calls"),
            "lwupb_source": sample.metadata.get("lwupb_source"),
        }
        if sample.optics is not None:
            row["cloud_optics"] = sample.optics
        run_bounding = sample.metadata.get("radiation_size_bounding_sum")
        if isinstance(run_bounding, dict):
            row["run_size_bounding_sum"] = run_bounding
            row["run_size_bounding_last"] = sample.metadata.get("radiation_size_bounding_last")
        per_checkpoint.append(row)

    intervals = []
    for before, after in zip(samples, samples[1:]):
        accumulated = accumulated_interval(before, after, grid, masks)
        intervals.append(accumulated if accumulated is not None
                         else held_interval([before, after], grid, masks))

    def window(t0_h, t1_h):
        picked = [s for s in samples if t0_h * 3600.0 - 1.0e-6 <= s.time_s <= t1_h * 3600.0 + 1.0e-6]
        if len(picked) < 2:
            return {"status": "INCOMPLETE", "reason": f"fewer than two checkpoints in {t0_h}-{t1_h} h"}
        accumulated = accumulated_interval(picked[0], picked[-1], grid, masks)
        if accumulated is not None:
            return accumulated
        return held_interval(picked, grid, masks)

    whole = window(samples[0].time_s / 3600.0, samples[-1].time_s / 3600.0)
    result = {
        "label": label,
        "instrument": "woof.globe.radiation_scorecard",
        "regions": list(masks),
        "checkpoints": per_checkpoint,
        "intervals": intervals,
        "run_mean": whole,
        "climatology": compare_to_ranges(whole.get("fluxes", {})),
        "reference": {
            "fluxes": ("not read: the case files' radiative fluxes are GRIB2 interval "
                       "products (template 4.8) rw-wps.mapping.v1 cannot bind "
                       "(mapped-engine assemble.rs:266); the analysis f000 carries none"),
            "cloud_cover": {},
        },
    }
    if window_hours is not None:
        result["window"] = {"hours": list(window_hours), **window(*window_hours)}
    if reference_frames:
        lat2d, lon2d = model_lat_lon(grid)
        for sample in samples:
            frame = reference_frames.get(sample.step)
            if frame is None:
                continue
            if frame.get("status") != "measured":
                result["reference"]["cloud_cover"][str(sample.step)] = {
                    "status": "INCOMPLETE", "product": frame.get("product"),
                    "reason": frame.get("reason")}
                continue
            model_cover = sample.flux("total_cloud_cover")
            if model_cover is None:
                result["reference"]["cloud_cover"][str(sample.step)] = {
                    "status": "INCOMPLETE", "reason": "no model cloud cover at this checkpoint"}
                continue
            reference = regrid_reference(
                frame["fields"]["total_cloud_cover"], frame["latitude"], frame["longitude"],
                lat2d, lon2d)
            entry = {
                "status": "measured", "valid_time": frame["valid_time"],
                "decode": frame.get("decode"),
                "model_source": ("held cldfra_total plane" if "recomputed" not in " ".join(sample.notes)
                                 else "recomputed from the checkpointed state"),
                "total_cloud_cover": score_against_reference(model_cover, reference, grid, masks),
            }
            for name in ("low_cloud_cover", "middle_cloud_cover", "high_cloud_cover"):
                ref = regrid_reference(frame["fields"][name], frame["latitude"], frame["longitude"], lat2d, lon2d)
                entry[f"reference_{name}"] = region_means(grid, ref, masks)
            result["reference"]["cloud_cover"][str(sample.step)] = entry
    return result


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

CALIBRATION = {
    "date": "2026-09-04",
    "families": [
        "planted planes: constants (global and every region to roundoff); a + b sin^2(lat) with both "
        "signs of b: the global mean reads a + b/3 to roundoff (the Gaussian quadrature is exact for "
        "it), every region reads the row quadrature over its own rows to roundoff (the 2-D masked "
        "mean against an independent 1-D row sum), and the region's departure from the closed-form "
        "area mean is the grid's own discretization of the region edges, recorded per truncation "
        "(1.6 percent of b for a hemisphere at T21's odd 33 rows where the equator row belongs to "
        "neither, under 0.2 percent of b for every band and hemisphere on the T255 grid of 384 rows)",
        "planted time integrals: accumulator differences over planted seconds read back the planted "
        "constant flux with its sign; a checkpoint pair without integrals falls back to the held "
        "sampling and says so",
        "planted reference offsets: +d and -d read bias d and -d with rmse |d|; planted zero-mean "
        "noise reads bias 0 and rmse sigma",
        "planted overlap: two adjacent layers c1 < c2 read c2, separated by a clear layer read "
        "1 - (1 - c1)(1 - c2), a full layer reads 1",
        "planted sizes: N cells above and M below each bound in K columns read exactly N, M, K "
        "and the planted path fraction; planted cloud fractions on the counted cells read the "
        "cloud-fraction-weighted (radiative) fraction and the count of unsampled cells exactly",
        "planted zonal waves: a + b cos(lon) and a + b sin(lon) with the land mask on the first half "
        "of the longitudes read the closed-form land, ocean and global means to roundoff, both signs "
        "of b (the longitude weighting and the land/ocean partition)",
        "planted zero-order hold: a daytime shortwave bump and a nearly constant longwave held on the "
        "model's own radiation schedule (3120 s buckets crossed at dt 50 steps); the time-integral "
        "path reads the applied mean to roundoff, and the held-plane trapezoid's departure from it "
        "over hours 1-24 and 12-24 is recorded (the size of the fallback sampling's error on a "
        "single-longitude diurnal signal, both signs of the departure appear across the two signals)",
        "planted reference regrid: a smooth cover field on a GFS-shaped 0.25 degree grid with "
        "DESCENDING latitude, regridded to the model grid and scored against the same field there, "
        "reads bias 0 to the bilinear error and a planted model offset of either sign to roundoff; "
        "an ascending copy and the -180..180 longitude convention regrid identically",
    ],
    "fooling_modes_checked": [
        "an absent carrier plane is INCOMPLETE, never 0 or a mean of the others",
        "the time-integral path is refused when the accumulated seconds do not advance",
        "a reference offset planted on the reference reads with the opposite sign of one planted on the model",
        "a counted cell whose cloud fraction is 0 contributes nothing to the radiative fractions",
        "a reference with descending latitude or the -180..180 longitude convention is not mis-oriented",
    ],
}


def _row_regions(grid: BudgetGrid) -> dict[str, np.ndarray]:
    """The latitude rows of every band and both hemispheres, from the
    1-D latitudes alone (independent of the 2-D masks the scorecard uses)."""
    lat = grid.latitude_deg
    out = {}
    for name, lo, hi in BANDS:
        out[name] = (lat >= lo) & ((lat < hi) if hi < 90.0 else (lat <= hi))
    out["northern_hemisphere"] = lat > 0.0
    out["southern_hemisphere"] = lat < 0.0
    return out


def _sin2_region_readings(grid: BudgetGrid) -> dict[str, dict[str, float]]:
    """For each region: the row quadrature of sin^2 lat over its rows and
    the closed-form area mean over its analytic edges; their difference is
    the grid's discretization of the region edges, not the instrument's."""
    mu = np.sin(np.deg2rad(grid.latitude_deg))
    w = grid.quadrature_weights / 2.0
    edges = {name: (lo, hi) for name, lo, hi in BANDS}
    edges["northern_hemisphere"] = (0.0, 90.0)
    edges["southern_hemisphere"] = (-90.0, 0.0)
    out = {}
    for name, rows in _row_regions(grid).items():
        discrete = float(np.sum(w[rows] * mu[rows] ** 2) / np.sum(w[rows]))
        lo, hi = edges[name]
        s_lo, s_hi = math.sin(math.radians(lo)), math.sin(math.radians(hi))
        exact = (s_hi ** 3 - s_lo ** 3) / (3.0 * (s_hi - s_lo))
        out[name] = {"row_quadrature": discrete, "closed_form": exact,
                     "grid_discretization_error": discrete - exact}
    return out


def synthetic_sample(grid: BudgetGrid, *, time_s: float, step: int, planes: dict[str, float | np.ndarray],
                     accumulators: dict[str, np.ndarray] | None = None,
                     accumulated_s: float | None = None, land=None) -> RadiationSample:
    ny, nx = grid.shape
    full = {}
    for name, value in planes.items():
        full[name] = np.full((ny, nx), float(value)) if np.ndim(value) == 0 else np.asarray(value, dtype=np.float64)
    if land is None:
        land = np.zeros((ny, nx), dtype=bool)
        land[:, : nx // 2] = True
    return RadiationSample(
        time_s=float(time_s), step=int(step), planes=full,
        accumulators=dict(accumulators or {}), accumulated_s=accumulated_s,
        skin_k=np.full((ny, nx), 288.0), emissivity=np.full((ny, nx), 0.95),
        land=np.asarray(land, dtype=bool), metadata={},
    )


def calibrate() -> dict:
    from .core.npref import (
        np_bound_cloud_sizes, np_max_random_total_cloud_cover)

    report = {"calibration": CALIBRATION, "readings": {}}
    grid = synthetic_grid(21)
    ny, nx = grid.shape
    masks = region_masks(grid, np.zeros((ny, nx), dtype=bool) | (np.arange(nx)[None, :] < nx // 2))
    lat2d, _ = model_lat_lon(grid)
    mu2 = np.sin(np.deg2rad(lat2d)) ** 2

    # family 1: planted planes
    planes = {}
    sin2 = _sin2_region_readings(grid)
    for sign in (1.0, -1.0):
        a, b = 200.0, sign * 90.0
        field_ = a + b * mu2
        got = region_means(grid, field_, masks)
        nh_mask = np.repeat((grid.latitude_deg > 0.0)[:, None], nx, axis=1)
        nh = grid.area_mean(field_, nh_mask)
        band_names = [name for name, _lo, _hi in BANDS]
        planes[f"b={b:+g}"] = {
            "global_error": got[GLOBAL] - (a + b / 3.0),
            # the 2-D masked means against the independent 1-D row sums
            "band_errors_vs_row_quadrature": {
                name: got[name] - (a + b * sin2[name]["row_quadrature"]) for name in band_names},
            "northern_hemisphere_error_vs_row_quadrature": (
                nh - (a + b * sin2["northern_hemisphere"]["row_quadrature"])),
            # the same readings against the closed-form area means: the
            # grid's discretization of the region edges, in units of b
            "region_discretization_error_over_b": {
                **{name: (got[name] - (a + b * sin2[name]["closed_form"])) / b for name in band_names},
                "northern_hemisphere": (nh - (a + b * sin2["northern_hemisphere"]["closed_form"])) / b,
            },
        }
    constant = region_means(grid, np.full((ny, nx), 123.456), masks)
    planes["constant_max_error"] = max(abs(v - 123.456) for v in constant.values())
    planes["grid_discretization_error_of_sin2_by_truncation"] = {
        "T21 (33 rows)": {name: row["grid_discretization_error"] for name, row in sin2.items()},
        "T255 (384 rows, the control grid)": {
            name: row["grid_discretization_error"]
            for name, row in _sin2_region_readings(BudgetGrid.for_shape(384, 768)).items()},
    }
    report["readings"]["planted_planes"] = planes

    # family 2: planted time integrals, both signs
    integrals = {}
    for flux in (+250.0, -40.0):
        before = synthetic_sample(grid, time_s=0.0, step=0, planes={"swdown": 0, "gsw": 0, "glw": 0, "olr": 0},
                                  accumulators={k: np.zeros((ny, nx)) for k in ACCUMULATORS},
                                  accumulated_s=0.0)
        after = synthetic_sample(grid, time_s=3600.0, step=72, planes={"swdown": 0, "gsw": 0, "glw": 0, "olr": 0},
                                 accumulators={k: np.full((ny, nx), flux * 3600.0) for k in ACCUMULATORS},
                                 accumulated_s=3600.0)
        row = accumulated_interval(before, after, grid, masks)
        integrals[f"flux={flux:+g}"] = {
            name: row["fluxes"][name][GLOBAL] - flux for name in ("sw_down_surface", "lw_up_top", "total_cloud_cover")
        }
        integrals[f"flux={flux:+g}"]["sw_net_surface_reads_zero"] = row["fluxes"]["sw_net_surface"][GLOBAL]
    stalled = accumulated_interval(before, synthetic_sample(
        grid, time_s=3600.0, step=72, planes={"swdown": 0, "gsw": 0, "glw": 0, "olr": 0},
        accumulators={k: np.zeros((ny, nx)) for k in ACCUMULATORS}, accumulated_s=0.0), grid, masks)
    integrals["stalled_seconds_refused"] = stalled is None
    held = held_interval([
        synthetic_sample(grid, time_s=0.0, step=0, planes={"swdown": 100, "gsw": 90, "glw": 300, "olr": 240}),
        synthetic_sample(grid, time_s=3600.0, step=72, planes={"swdown": 300, "gsw": 270, "glw": 300, "olr": 240}),
    ], grid, masks)
    integrals["held_fallback"] = {"sampling": held["sampling"],
                                  "sw_down_surface_minus_200": held["fluxes"]["sw_down_surface"][GLOBAL] - 200.0,
                                  "sw_up_top_incomplete": held["fluxes"]["sw_up_top"]["status"]}
    report["readings"]["planted_time_integrals"] = integrals

    # family 3: planted reference offsets and noise
    rng = np.random.default_rng(7)
    model = 0.5 + 0.2 * np.sin(np.deg2rad(lat2d))
    offsets = {}
    for delta in (+0.05, -0.05, +0.2, -0.2):
        got = score_against_reference(model, model - delta, grid, masks)
        offsets[f"delta={delta:+g}"] = {"bias_error": got[GLOBAL]["bias"] - delta,
                                        "rmse_error": got[GLOBAL]["rmse"] - abs(delta),
                                        "land_bias_error": got[LAND]["bias"] - delta}
    noise = rng.normal(0.0, 0.1, size=model.shape)
    got = score_against_reference(model, model - noise, grid, masks)
    offsets["noise sigma 0.1"] = {"bias": got[GLOBAL]["bias"], "rmse": got[GLOBAL]["rmse"]}
    got_reference_side = score_against_reference(model - 0.05, model, grid, masks)
    offsets["offset_on_reference_reads_opposite"] = got_reference_side[GLOBAL]["bias"]
    report["readings"]["planted_reference_offsets"] = offsets

    # family 4: planted overlap
    overlap = {}
    adjacent = np.zeros((1, 6))
    adjacent[0, 2], adjacent[0, 3] = 0.3, 0.6
    overlap["adjacent_0.3_0.6_reads_0.6_error"] = float(np_max_random_total_cloud_cover(adjacent)[0] - 0.6)
    separated = np.zeros((1, 6))
    separated[0, 1], separated[0, 4] = 0.3, 0.6
    overlap["separated_0.3_0.6_reads_random_error"] = float(
        np_max_random_total_cloud_cover(separated)[0] - (1.0 - 0.7 * 0.4))
    full = np.zeros((1, 6))
    full[0, 3] = 1.0
    overlap["full_layer_reads_1_error"] = float(np_max_random_total_cloud_cover(full)[0] - 1.0)
    overlap["clear_reads_0"] = float(np_max_random_total_cloud_cover(np.zeros((1, 6)))[0])
    report["readings"]["planted_overlap"] = overlap

    # family 5: planted sizes
    sizes = {}
    ncol, nlay = 12, 5
    clwp = np.ones((ncol, nlay))
    ciwp = np.ones((ncol, nlay))
    reliq = np.full((ncol, nlay), 10.0)
    dgice = np.full((ncol, nlay), 50.0)
    reliq[0, :3] = 43.0          # 3 cells above (carried by 21.5/43 = 0.5), 1 column
    reliq[1, 0] = 1.0            # 1 cell below, 1 column
    dgice[2:5, 4] = 360.0        # 3 cells above (carried by 0.5), 3 columns
    dgice[5, 1:3] = 4.0          # 2 cells below, 1 column
    got = np_bound_cloud_sizes(clwp, ciwp, reliq, dgice, SIZE_BOUNDS)
    sizes["counts"] = dict(got.size_bounding)
    sizes["expected"] = {"liquid_above_cells": 3, "liquid_above_columns": 1, "liquid_below_cells": 1,
                         "liquid_below_columns": 1, "ice_above_cells": 3, "ice_above_columns": 3,
                         "ice_below_cells": 2, "ice_below_columns": 1,
                         "liquid_above_path_fraction": 3.0 / 60.0, "ice_above_path_fraction": 3.0 / 60.0}
    sizes["carried_path_error"] = float(np.max(np.abs(got.clwp[0, :3] - 0.5)))
    sizes["untouched_path_error"] = float(np.max(np.abs(got.clwp[1:, :] - 1.0)))
    sizes["sizes_inside_bounds"] = bool(np.all((got.reliq >= 2.5) & (got.reliq <= 21.5)
                                               & (got.dgice >= 10.0) & (got.dgice <= 180.0)))
    # the radiative weight: planted cloud fractions on the counted cells
    cldfra = np.ones((ncol, nlay))
    cldfra[0, 0], cldfra[0, 1], cldfra[3, 4] = 0.5, 0.0, 0.0
    weighted = np_bound_cloud_sizes(clwp, ciwp, reliq, dgice, SIZE_BOUNDS, cldfra=cldfra)
    total = 60.0 - 0.5 - 1.0 - 1.0
    sizes["weighted_expected"] = {
        "liquid_above_radiative_fraction": (0.5 + 0.0 + 1.0) / total,
        "ice_above_radiative_fraction": (1.0 + 0.0 + 1.0) / total,
        "liquid_unsampled_cells": 2, "ice_unsampled_cells": 2,
        "liquid_above_path_fraction": 3.0 / 60.0, "ice_above_path_fraction": 3.0 / 60.0,
    }
    sizes["weighted_counts"] = {name: weighted.size_bounding[name] for name in sizes["weighted_expected"]}
    sizes["unweighted_radiative_equals_in_cloud"] = (
        got.size_bounding["liquid_above_radiative_fraction"] == got.size_bounding["liquid_above_path_fraction"]
        and got.size_bounding["ice_above_radiative_fraction"] == got.size_bounding["ice_above_path_fraction"])
    report["readings"]["planted_sizes"] = sizes

    # family 6: zonal waves against the land mask on the first half of the longitudes
    zonal = {}
    lon_rad = np.deg2rad(model_lat_lon(grid)[1])
    for b in (+37.0, -37.0):
        a = 200.0
        got_c = region_means(grid, a + b * np.cos(lon_rad), masks)
        got_s = region_means(grid, a + b * np.sin(lon_rad), masks)
        # sum_{j < nx/2} cos(2 pi j / nx) = 1 and sum_{j < nx/2} sin(2 pi j / nx) = cot(pi / nx)
        land_cos = a + b * (2.0 / nx)
        land_sin = a + b * (2.0 / nx) / math.tan(math.pi / nx)
        zonal[f"b={b:+g}"] = {
            "cos_global_error": got_c[GLOBAL] - a,
            "cos_land_error": got_c[LAND] - land_cos,
            "cos_ocean_error": got_c[OCEAN] - (2.0 * a - land_cos),
            "sin_global_error": got_s[GLOBAL] - a,
            "sin_land_error": got_s[LAND] - land_sin,
            "sin_ocean_error": got_s[OCEAN] - (2.0 * a - land_sin),
        }
    report["readings"]["planted_zonal_waves"] = zonal

    # family 7: a zero-order-held signal on the model's radiation schedule
    report["readings"]["planted_zero_order_hold"] = _zero_order_hold_readings(grid, masks)

    # family 8: the reference regrid end to end, descending latitude as the GFS grid
    ref_lat = np.linspace(90.0, -90.0, 721)
    ref_lon = np.arange(0.0, 360.0, 0.25)
    ref_lat2d, ref_lon2d = np.meshgrid(ref_lat, ref_lon, indexing="ij")

    def cover_field(lat, lon):
        return (0.5 + 0.2 * np.sin(np.deg2rad(lat))
                + 0.1 * np.cos(np.deg2rad(lon)) * np.cos(np.deg2rad(lat)))

    reference = cover_field(ref_lat2d, ref_lon2d)
    model_plane = cover_field(lat2d, model_lat_lon(grid)[1])
    regridded = regrid_reference(reference, ref_lat, ref_lon, lat2d, model_lat_lon(grid)[1])
    regrid = {"descending_latitude_max_abs_error": float(np.max(np.abs(regridded - model_plane))),
              "descending_latitude_bias": score_against_reference(model_plane, regridded, grid, masks)[GLOBAL]["bias"]}
    for delta in (+0.05, -0.05):
        scored = score_against_reference(model_plane + delta, regridded, grid, masks)
        regrid[f"planted_model_offset_{delta:+g}_bias_error"] = scored[GLOBAL]["bias"] - delta
        regrid[f"planted_model_offset_{delta:+g}_land_bias_error"] = scored[LAND]["bias"] - delta
    ascending = regrid_reference(reference[::-1], ref_lat[::-1], ref_lon, lat2d, model_lat_lon(grid)[1])
    regrid["ascending_minus_descending_max_abs"] = float(np.max(np.abs(ascending - regridded)))
    half = ref_lon.size // 2
    pm_lon = np.concatenate([ref_lon[half:] - 360.0, ref_lon[:half]])
    pm_ref = np.concatenate([reference[:, half:], reference[:, :half]], axis=1)
    pm = regrid_reference(pm_ref, ref_lat, pm_lon, lat2d, model_lat_lon(grid)[1])
    regrid["minus180_convention_minus_descending_max_abs"] = float(np.max(np.abs(pm - regridded)))
    report["readings"]["planted_reference_regrid"] = regrid
    return report


#: The control configuration's radiation schedule (configs/verify dt50):
#: the bucket length and the model step the held planes are integrated over.
ZERO_ORDER_HOLD_INTERVAL_S = 3120.0
ZERO_ORDER_HOLD_DT_S = 50.0


def _zero_order_hold_readings(grid: BudgetGrid, masks: dict[str, np.ndarray]) -> dict:
    """A signal held on the radiation schedule: the time-integral path reads
    the applied mean exactly; the held-plane trapezoid over hourly
    checkpoints does not, and its departure is recorded (W/m2)."""
    ny, nx = grid.shape
    dt, interval, day = ZERO_ORDER_HOLD_DT_S, ZERO_ORDER_HOLD_INTERVAL_S, 86400.0
    steps = int(round(day / dt))

    def held_series(signal):
        held = np.zeros(steps)
        last_bucket, value = -1, 0.0
        for i in range(steps):
            t_call = dt * i
            bucket = int(math.floor(t_call / interval))
            if bucket != last_bucket:
                value, last_bucket = float(signal(t_call)), bucket
            held[i] = value
        return held

    signals = {
        "sw_daytime_bump_1000": lambda t: 1000.0 * max(0.0, math.cos(2.0 * math.pi * (t - 12.0 * 3600.0) / day)),
        "lw_240_plus_10_diurnal": lambda t: 240.0 + 10.0 * math.cos(2.0 * math.pi * (t - 15.0 * 3600.0) / day),
    }
    out = {}
    for name, signal in signals.items():
        held = held_series(signal)
        exact_1_24 = float(np.sum(held[72:] * dt) / (23.0 * 3600.0))
        exact_12_24 = float(np.sum(held[72 * 12:] * dt) / (12.0 * 3600.0))
        samples = []
        for hour in range(1, 25):
            value = held[72 * hour - 1]
            acc = float(np.sum(held[: 72 * hour] * dt))
            samples.append(synthetic_sample(
                grid, time_s=3600.0 * hour, step=72 * hour,
                planes={"swdown": value, "gsw": value, "glw": value, "olr": value},
                accumulators={k: np.full((ny, nx), acc) for k in ACCUMULATORS},
                accumulated_s=3600.0 * hour))
        integrated = accumulated_interval(samples[0], samples[-1], grid, masks)
        integrated_12_24 = accumulated_interval(samples[11], samples[-1], grid, masks)
        held_rows = held_interval(samples, grid, masks)
        held_rows_12_24 = held_interval(samples[11:], grid, masks)
        out[name] = {
            "integral_error_1_24": integrated["fluxes"]["lw_up_top"][GLOBAL] - exact_1_24,
            "integral_error_12_24": integrated_12_24["fluxes"]["lw_up_top"][GLOBAL] - exact_12_24,
            "held_trapezoid_error_1_24": held_rows["fluxes"]["lw_up_top"][GLOBAL] - exact_1_24,
            "held_trapezoid_error_12_24": held_rows_12_24["fluxes"]["lw_up_top"][GLOBAL] - exact_12_24,
            "applied_mean_1_24": exact_1_24,
        }
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_finite(payload), stream, indent=1, sort_keys=True)


def summary_lines(result: dict) -> list[str]:
    lines = [f"radiation scorecard {result.get('label', '')}".rstrip()]
    whole = result.get("run_mean", {})
    lines.append(f"run mean ({whole.get('sampling', 'INCOMPLETE')}):")
    for name in FLUXES:
        row = whole.get("fluxes", {}).get(name)
        if row is None:
            continue
        if row.get("status") != "measured":
            lines.append(f"  {name:22s} INCOMPLETE ({row.get('reason')})")
        else:
            lines.append(f"  {name:22s} global {row[GLOBAL]:9.3f}  land {row.get(LAND, float('nan')):9.3f}"
                         f"  ocean {row.get(OCEAN, float('nan')):9.3f}")
    clim = result.get("climatology", {}).get("readings", {})
    for name, row in clim.items():
        if row.get("status") == "measured":
            lines.append(f"  climatology {name:20s} {row['global']:8.2f} range {row['range']} "
                         f"{'inside' if row['inside'] else 'OUTSIDE by %+.2f' % row['distance']}")
    for step, entry in result.get("reference", {}).get("cloud_cover", {}).items():
        if entry.get("status") == "measured":
            g = entry["total_cloud_cover"][GLOBAL]
            lines.append(f"  cloud cover vs reference at step {step} ({entry['valid_time']}): "
                         f"model {g['model_mean']:.3f} ref {g['reference_mean']:.3f} "
                         f"bias {g['bias']:+.3f} rmse {g['rmse']:.3f}")
        else:
            lines.append(f"  cloud cover vs reference at step {step}: INCOMPLETE ({entry.get('reason')})")
    last = result.get("checkpoints", [{}])[-1]
    optics = last.get("cloud_optics")
    if optics:
        b = optics["size_bounding"]
        lines.append(
            f"  cloud optics at step {last.get('step')}: liquid cells {b['liquid_cells']} "
            f"(above {b['liquid_above_cells']}, below {b['liquid_below_cells']}), ice cells {b['ice_cells']} "
            f"(above {b['ice_above_cells']} in {b['ice_above_columns']} columns carrying "
            f"{100 * b['ice_above_path_fraction']:.1f}% of the in-cloud ice path and "
            f"{100 * b.get('ice_above_radiative_fraction', float('nan')):.1f}% of the cloud-fraction-weighted one, "
            f"below {b['ice_below_cells']}; unsampled cells liquid {b.get('liquid_unsampled_cells')} "
            f"ice {b.get('ice_unsampled_cells')})")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default="")
    parser.add_argument("--reference-grib", default=None,
                        help="comma-separated pgrb2 products of the cycle, ascending lead")
    parser.add_argument("--reference-step", type=int, action="append", default=None,
                        help="checkpoint step each reference product is valid at, in the same order")
    parser.add_argument("--reference-mapping", default=None)
    parser.add_argument("--window-hours", type=float, nargs=2, default=None)
    parser.add_argument("--no-optics", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    args = parser.parse_args(argv)

    if args.calibrate:
        report = calibrate()
        if args.out:
            _write_json(args.out, report)
        print(json.dumps(_finite(report["readings"]), indent=1))
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required unless --calibrate")
    receipt = read_receipt(args.run_dir)
    # The reference decode first: it is cheap beside the checkpoint reads
    # and a refusal there is recorded per product, never fatal.
    reference_frames = None
    if args.reference_grib:
        products = args.reference_grib.split(",")
        steps = args.reference_step or []
        if len(steps) != len(products):
            parser.error("--reference-step must be given once per reference product")
        frames = decode_reference_cover(products, args.reference_mapping)
        reference_frames = {int(step): frame for step, frame in zip(steps, frames)}
        for step, frame in reference_frames.items():
            if frame.get("status") != "measured":
                print(f"reference at step {step} INCOMPLETE: {frame.get('reason')}", flush=True)
            else:
                print(f"reference at step {step}: {frame.get('product')} valid {frame.get('valid_time')}", flush=True)
    reader = RadiationReader.from_receipt(receipt, optics=not args.no_optics)
    reader._truncation = int(receipt["config"]["truncation"])
    paths = sorted(Path(args.run_dir).glob("arwen_global_step*.npz"))
    if args.max_checkpoints:
        paths = paths[: args.max_checkpoints]
    samples = []
    skipped = []
    for path in paths:
        try:
            sample = reader.sample(path)
        except ColdStartCheckpoint as exc:
            skipped.append(str(exc))
            print(f"skipped: {exc}", flush=True)
            continue
        samples.append(sample)
        print(f"read step {sample.step} hour {sample.time_s / 3600.0:.2f}"
              + (f" ({len(sample.notes)} notes)" if sample.notes else ""), flush=True)
    if not samples:
        raise SystemExit(
            f"{args.run_dir}: no checkpoint after the cold start; nothing to score")
    result = scorecard(samples, reader.grid, label=args.label,
                       window_hours=tuple(args.window_hours) if args.window_hours else None,
                       reference_frames=reference_frames)
    result["run_dir"] = str(args.run_dir)
    result["config_hash"] = receipt.get("config_hash")
    result["skipped_checkpoints"] = skipped
    if args.out:
        _write_json(args.out, result)
    print("\n".join(summary_lines(result)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
