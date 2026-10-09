"""Existing-Arwen CUDA scheme calls for the Level-5 native global adapter."""
from __future__ import annotations

from datetime import timedelta
import importlib
import math
from types import SimpleNamespace

import numpy as np

from ..profile import profiler_of
from ..constants import (
    CONVECTIVE_RAIN_ACCUMULATOR,
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EARTH_RADIUS_M,
    GRAVITY_M_S2,
    LATENT_HEAT_VAPORIZATION,
    NUMBER_MOMENTS,
    STEFAN_BOLTZMANN,
    WATER_SPECIES,
)
from ..statics import (
    SURFACE_STATICS_METADATA_KEY, CategoryConvention, frozen_water_columns,
    lake_columns, xland_plane,
)
from ..water import SOIL_LAYER_THICKNESS_M
from . import frozen_surface
from .native_state import (
    CUMULUS_DYNAMICS_QV_LANE, CUMULUS_DYNAMICS_THETA_LANE, CUMULUS_EXIT_QV,
    CUMULUS_EXIT_THETA, CUMULUS_EXIT_TIME_KEY, PersistentNativeState,
    RADIATION_ACCUMULATORS,
)
from . import provenance
from .surface_diagnostics import SOURCE_METADATA_KEY


# The cumulus scheme's per-level rates, in the units woof.globe.core.gf returns
# them (WRF's RTHCUTEN/RQVCUTEN/RQCCUTEN/RQICUTEN: theta K/s and dry mixing
# ratio kg/kg/s) and the batch field each one integrates into.
CUMULUS_RATE_FIELDS = (
    ("rthcuten", "theta"), ("rqvcuten", "qv"),
    ("rqccuten", "qc"), ("rqicuten", "qi"),
)
REQUIRED_CUMULUS_RATES = ("rthcuten", "rqvcuten")
# Module and class of each cumulus option's scheme callable; all share the
# GrellFreitas seam's interface (constructor column_chunk=, bind_driver,
# __call__(atmosphere=, fields=, state=, cfg=)).
CUMULUS_SCHEME_MODULES = {
    "gf": ("woof.globe.core.gf", "GrellFreitas"),
    "own": ("woof.globe.physics.arwen_massflux", "ArwenMassFluxV1"),
    "ntiedtke": ("woof.globe.core.ntiedtke", "NewTiedtke"),
}
# Convective momentum tendencies (WRF's RUCUTEN/RVCUTEN, m/s/s on the
# A-grid wind) and the batch field each integrates into, applied once,
# below the four rates above and with the same integration.  New Tiedtke
# produces the pair (cu_ntiedtke.F90 lmfdudv is a PARAMETER, so its
# momentum update is not optional) and arwen-massflux-v1 transports u
# and v with its plume; Grell-Freitas as adapted produces none (woof.
# core.gf deviation 2: its dudt/dvdt are not coupled).  A result carrying
# one and not the other is refused, never half-applied.
CUMULUS_MOMENTUM_FIELDS = (("rucuten", "u"), ("rvcuten", "v"))


# Fields SFClayResult computes that the land surface consumes.  chs/chs2/cqs2
# are the exchange coefficients Noah's surface balance is defined against and
# qgh the ground saturation humidity; substituting them decouples Noah from the
# surface layer that is supposed to drive it.
REQUIRED_SFCLAY_FIELDS = (
    "znt", "ust", "mol", "hfx", "qfx", "qsfc", "zol", "wspd",
    "br", "fm", "fh", "u10", "v10", "chs", "chs2", "cqs2", "qgh",
)
# SFClayResult carries no PBL height; YSU supplies it when it runs.
OPTIONAL_SFCLAY_FIELDS = ("pblh",)
# Screen-level diagnostics sfclay computes (WRF's T2/TH2/Q2): persisted so
# the render tape exports a 2 m value rather than the lowest full level
# (audit 2026-09-01, task 1b).  Kept when present: a surface layer without
# them breaks nothing in the step, and the export then falls back to the
# similarity diagnostic and labels the tape so.
DIAGNOSTIC_SFCLAY_FIELDS = ("t2", "th2", "q2")
# Water columns carry the surface layer's own T2/TH2/Q2; land columns
# carry WRF's SFCDIAGS after Noah (_screen_level_step), so the label the
# export stamps on a tape names both writers.
NATIVE_SURFACE_DIAGNOSTICS_SOURCE = "native-surface-layer-water+sfcdiags-land"
# Surface fluxes the land surface owns on the columns it integrates.  WRF's
# surface_driver runs sfclay then the LSM every step and the LSM overwrites
# HFX/QFX/QSFC before the PBL reads them, so sfclay's bulk land fluxes never
# reach the PBL.  The bridge persists Noah's values (noah_hfx/noah_qfx/
# noah_qsfc) and restores them onto land columns on every call where Noah
# is not due (audit 2026-09-01 NB-2).
NOAH_HELD_FLUX_FIELDS = ("hfx", "qfx", "qsfc")


#: The floor of an inland lake's skin under its energy budget
#: (NativePhysicsRuntime._lake_surface_step): freshwater freezing.
LAKE_FREEZING_K = 273.15


class _BandCache:
    """What the runtime builds once from the rows it is handed and keeps
    for the run: one of these per band (:attr:`NativePhysicsRuntime._bands`)."""

    __slots__ = ("radiation", "frozen", "frozen_count", "partial",
                 "partial_count", "lake", "lake_count", "dx_column",
                 "statics_checked")

    def __init__(self) -> None:
        self.radiation = None
        self.frozen = None
        self.frozen_count = None
        self.partial = None
        self.partial_count = None
        self.lake = None
        self.lake_count = None
        self.dx_column = None
        self.statics_checked = False


class NativePhysicsRuntime:
    """Runs RRTMGP -> SFCLAY -> Noah -> YSU -> cumulus -> Morrison on a copied batch.

    WRF's order (radiation, surface layer, land surface, PBL, cumulus,
    microphysics); the cumulus slot is Grell-Freitas by default, New
    Tiedtke with ``cumulus="ntiedtke"``, arwen-massflux-v1 with
    ``cumulus="own"`` and empty with ``cumulus="none"``.
    """

    # In-situ component capture slot (woof.globe.insitu.capture):
    # marked after every scheme call in run().  Read-only by contract.
    observer = None
    # Step profiler slot (woof.globe.profile): every scheme call
    # in run() is one named section.  None is the no-op.
    profiler = None

    def __init__(self, options, *, modules: dict[str, object] | None = None):
        self.options = options
        self.modules = dict(modules or {})
        # THE PER-GRID STATE IS PER BAND.  The suite runs a latitude band
        # at a time (dynamics.apply_physics), and everything the runtime
        # builds once from the grid it is handed -- the radiation driver
        # with its solar geometry, the frozen-column and partial-pack
        # masks and their counts, the cumulus grid spacing, the statics
        # check -- is a function of WHICH rows it was handed.  Keyed on
        # the batch's band (None for a whole-grid batch), so a band reads
        # the state built from its own rows and a whole-grid batch reads
        # what it always read.  The attributes below are properties over
        # the current band's cache, so every scheme step reads them as it
        # always did.
        self._bands: dict[tuple[int, int] | None, _BandCache] = {}
        self._band_key: tuple[int, int] | None = None
        self._noah_params = None
        self._cumulus = None
        self._cumulus_diagnostics = {}
        # Per-column readings the scheme holds beside its result (the GF
        # seam's deep exit code); read by the storm reader's tendency trace.
        # Of the LAST band run.
        self._cumulus_column_diagnostics = {}
        # The LANDUSE.TBL moisture availability of the ice class, fixed for
        # the run and built on first use (a table value, not a grid one).
        self._ice_mavail = None
        # The radiation's size-bounding path sums per column of the last
        # call, as (ny, nx) planes keyed by rrtmgp.SIZE_BOUNDING_SUM_NAMES,
        # or None when the radiation was not due.  The suite hands them
        # back as result planes so the record is assembled over the globe.
        self.last_radiation_bounding_columns = None

    @property
    def _band(self) -> "_BandCache":
        cache = self._bands.get(self._band_key)
        if cache is None:
            cache = self._bands[self._band_key] = _BandCache()
        return cache

    @property
    def _radiation(self):
        return self._band.radiation

    @_radiation.setter
    def _radiation(self, value) -> None:
        self._band.radiation = value

    @property
    def _frozen(self):
        return self._band.frozen

    @_frozen.setter
    def _frozen(self, value) -> None:
        self._band.frozen = value

    @property
    def _frozen_count(self):
        return self._band.frozen_count

    @_frozen_count.setter
    def _frozen_count(self, value) -> None:
        self._band.frozen_count = value

    @property
    def _partial(self):
        return self._band.partial

    @_partial.setter
    def _partial(self, value) -> None:
        self._band.partial = value

    @property
    def _partial_count(self):
        return self._band.partial_count

    @_partial_count.setter
    def _partial_count(self, value) -> None:
        self._band.partial_count = value

    @property
    def _lake(self):
        return self._band.lake

    @_lake.setter
    def _lake(self, value) -> None:
        self._band.lake = value

    @property
    def _lake_count(self):
        return self._band.lake_count

    @_lake_count.setter
    def _lake_count(self, value) -> None:
        self._band.lake_count = value

    @property
    def _dx_column(self):
        return self._band.dx_column

    @_dx_column.setter
    def _dx_column(self, value) -> None:
        self._band.dx_column = value

    @property
    def _statics_checked(self) -> bool:
        return self._band.statics_checked

    @_statics_checked.setter
    def _statics_checked(self, value: bool) -> None:
        self._band.statics_checked = bool(value)

    def _observe(self, name, batch) -> None:
        if self.observer is not None:
            fields = {key: batch.arrays[key] for key in ("theta", "qv", "u", "v")}
            if batch.band is None:
                self.observer.mark(name, fields, batch.xp)
            else:
                self.observer.mark(name, fields, batch.xp, band=batch.band)

    def _module(self, name: str):
        """The physics module for ``name``, recorded as it is resolved.

        Recorded rather than inferred.  Both copies of several of these
        modules exist in one process -- the carried ones this model calls and
        the engine's, which the engine's own regional drivers import -- so
        which one integrated is a fact about this run, not about a version
        number.  `provenance.note` reads the module that was actually
        returned, including one a caller substituted through ``modules``.
        """

        if name in self.modules:
            return provenance.note(self.modules[name])
        return provenance.note(importlib.import_module(name))

    @staticmethod
    def _bucket(time_s: float, interval_s: float) -> int:
        return int(math.floor((float(time_s) + 1.0e-9) / float(interval_s)))

    @staticmethod
    def _dz(batch):
        xp = batch.xp
        tv = batch.arrays["virtual_temperature"]
        p_half = batch.arrays["p_half"]
        return xp.ascontiguousarray(
            DRY_AIR_GAS_CONSTANT * tv / GRAVITY_M_S2
            * xp.log(p_half[:-1] / p_half[1:])
        )

    def _radiation_step(self, batch, persistent, cfg):
        xp = batch.xp
        bucket = self._bucket(batch.time_s, self.options.radiation_interval_s)
        due = bucket != int(persistent.metadata["last_radiation_bucket"])
        if due:
            module = self._module("woof.globe.core.rrtmgp")
            if self._radiation is None:
                # One driver per band: its solar geometry is the band's
                # rows.  The k-distribution and cloud tables behind it are
                # loaded once per process and uploaded once per card
                # (rrtmgp.load_gas_tables, GasTables.to_device), so a
                # second band's driver costs its geometry planes and the
                # trace-gas read, not a second set of tables.
                self._radiation = module.RRTMGPRadiation(
                    start_time=self.options.start_time,
                    latitude_deg=batch.arrays["latitude_deg"],
                    longitude_deg=batch.arrays["longitude_deg"],
                    column_chunk=self.options.radiation_column_chunk,
                    validation_mode=self.options.radiation_validation_mode,
                    trace_gas_overrides={"co2": self.options.trace_co2_ppm * 1.0e-6},
                    column_size_bounding=True,
                )
            atmosphere = {
                "pressure": batch.arrays["p_full"],
                "p_interface": batch.arrays["p_half"],
                "temperature": batch.arrays["temperature"],
                "exner": batch.arrays["exner"],
                "qv": batch.arrays["qv"],
                "qc": batch.arrays["qc"],
                "qi": batch.arrays["qi"],
            }
            fields = {
                "tsk": batch.surface.temperature_k,
                "emiss": batch.surface.emissivity,
                "albedo": batch.surface.albedo,
            }
            result = self._radiation(
                atmosphere=atmosphere,
                fields=fields,
                state=persistent.fake_radiation_state(),
                cfg=cfg,
            )
            persistent.arrays["rad_rthratenlw"][...] = result.rthratenlw
            persistent.arrays["rad_rthratensw"][...] = result.rthratensw
            for name in ("swdown", "glw", "olr", "gsw", "coszen",
                         "swupt", "swdnt", "lwupb", "cldfra_total"):
                value = getattr(result, name, None)
                if value is not None:
                    persistent.arrays[name][...] = value
            if getattr(result, "lwupb", None) is not None:
                persistent.metadata["lwupb_source"] = "scheme"
            else:
                # A scheme without a surface upward longwave plane: the
                # grey-body formula on the skin temperature and broadband
                # emissivity, with the reflected downward part, is the
                # surface's own emission and is labelled as such.
                emissivity = batch.surface.emissivity
                persistent.arrays["lwupb"][...] = (
                    emissivity * xp.float32(STEFAN_BOLTZMANN)
                    * batch.surface.temperature_k ** 4
                    + (xp.float32(1.0) - emissivity) * persistent.arrays["glw"]
                )
                persistent.metadata["lwupb_source"] = "surface_emission_formula"
            # The record's path sums per column, as planes of this batch's
            # rows: the suite hands them back unreduced and the record's
            # fractions are formed once over the assembled globe
            # (native_suite.ArwenCudaColumnSuite.finish).  The per-batch
            # record written into the metadata below carries THIS batch's
            # counts, which add across bands exactly, and this batch's
            # fractions, which the finish replaces.
            columns = getattr(self._radiation, "last_size_bounding_columns", None)
            if isinstance(columns, dict):
                ny, nx = batch.surface_shape
                self.last_radiation_bounding_columns = {
                    name: xp.ascontiguousarray(
                        xp.asarray(value).reshape(ny, nx))
                    for name, value in columns.items()
                }
            bounding = getattr(self._radiation, "last_size_bounding", None)
            if isinstance(bounding, dict):
                persistent.metadata["radiation_size_bounding_last"] = dict(bounding)
                running = persistent.metadata.get("radiation_size_bounding_sum")
                persistent.metadata["radiation_size_bounding_sum"] = {
                    key: (0 if running is None else running.get(key, 0)) + value
                    for key, value in bounding.items()
                }
            persistent.metadata["radiation_calls"] = int(
                persistent.metadata.get("radiation_calls", 0)
            ) + 1
            persistent.metadata["last_radiation_bucket"] = bucket
        heating = persistent.arrays["rad_rthratenlw"] + persistent.arrays[
            "rad_rthratensw"
        ]
        batch.arrays["theta"] += xp.float32(batch.dt_s) * heating
        batch.arrays["temperature"] = xp.ascontiguousarray(
            batch.arrays["theta"] * batch.arrays["exner"]
        )
        self._accumulate_radiation(batch, persistent)
        return due

    @staticmethod
    def _accumulate_radiation(batch, persistent) -> None:
        """Advance the radiation time integrals by this call's slab.

        The held planes are what the model applies until the next
        radiation bucket, so integrating them over every call's ``dt_s``
        gives the exact time-mean flux between two checkpoints.  Upward
        surface shortwave is swdown minus gsw (GSW is the absorbed part).
        """
        xp = batch.xp
        dt = xp.float32(batch.dt_s)
        arrays = persistent.arrays
        planes = {
            "swdown": arrays["swdown"],
            "swupb": arrays["swdown"] - arrays["gsw"],
            "glw": arrays["glw"],
            "lwupb": arrays["lwupb"],
            "swupt": arrays["swupt"],
            "swdnt": arrays["swdnt"],
            "olr": arrays["olr"],
            "cldfra_total": arrays["cldfra_total"],
        }
        for accumulator, plane in RADIATION_ACCUMULATORS.items():
            arrays[accumulator] += dt * planes[plane]
        persistent.metadata["radiation_accumulated_s"] = float(
            persistent.metadata.get("radiation_accumulated_s", 0.0)
        ) + float(batch.dt_s)

    @staticmethod
    def _xland(batch):
        """WRF land/water flag (1 land, 2 water) from the land fraction and
        the analysed sea ice (statics.xland_plane: 1 + (1 - land_fraction),
        exactly 1 on every sea-ice column).

        One construction for every kernel call and mask in the runtime, so
        the columns sfclay treats as water, the columns Noah skips as water
        (noah.cu:968, xland >= 1.5) and the columns whose state the bridge
        holds across calls are bit-for-bit the same set, and the sea-ice
        columns are land to sfclay, YSU and the cumulus scheme while Noah
        skips them as ice (noah.cu:969) and the frozen-surface step
        integrates their skin.  The static fields are seeded with the same
        rule (statics.water_columns), so the water category sits on exactly
        these columns and the land and ice classes on the rest;
        _verify_statics refuses a state where it does not.
        """
        xp = batch.xp
        return xp.ascontiguousarray(
            xland_plane(batch.surface.land_fraction, batch.surface.sea_ice_fraction, xp)
        )

    def _frozen_columns(self, batch, persistent):
        """The columns Noah skips as frozen (sea ice by the kernel's
        threshold, land ice by category), fixed for the run."""
        if self._frozen is None:
            xp = batch.xp
            self._frozen = frozen_surface.frozen_columns(
                self._xland(batch), batch.surface.sea_ice_fraction,
                batch.surface.landuse_category, self.options.land_ice_category, xp,
            )
            self._frozen_count = int(xp.count_nonzero(self._frozen))
            fraction = xp.asarray(batch.surface.sea_ice_fraction, dtype=xp.float32)
            self._partial = self._frozen & frozen_water_columns(fraction, xp) & (fraction < xp.float32(1.0))
            self._partial_count = int(xp.count_nonzero(self._partial))
        return self._frozen

    def _lake_columns(self, batch):
        """The open-water columns that are inland lakes (statics.
        lake_columns on the surface state's planes), fixed for the run."""
        if self._lake is None:
            xp = batch.xp
            self._lake = xp.ascontiguousarray(lake_columns(
                batch.surface.land_fraction, batch.surface.lake_fraction, xp,
                sea_ice_fraction=batch.surface.sea_ice_fraction,
            ))
            self._lake_count = int(xp.count_nonzero(self._lake))
        return self._lake

    def _ice_moisture_availability(self, persistent) -> float:
        """LANDUSE.TBL's SLMO for the ice class of the state's convention
        (0.95 for the MODIS snow-and-ice row): what WRF's landuse_init
        hands sfclay on every sea-ice and land-ice column."""
        if self._ice_mavail is None:
            import woof.globe.core.landuse as landuse_module

            load_landuse_table = provenance.note(landuse_module).load_landuse_table

            convention = self._statics_convention(persistent)
            table = load_landuse_table(convention.landuse_dataset)
            rows = table.values[:, convention.ice_category - 1, 1]
            if not all(float(value) == float(rows[0]) for value in rows):
                raise ValueError(
                    f"LANDUSE.TBL {convention.landuse_dataset} gives the ice "
                    f"class {convention.ice_category} a moisture availability "
                    "that changes with season; the frozen-surface columns "
                    "are handed one value for the run"
                )
            self._ice_mavail = float(rows[0])
        return self._ice_mavail

    def _surface_layer_step(self, batch, persistent):
        module = self._module("woof.globe.core.sfclay")
        xp = batch.xp
        dz = self._dz(batch)
        f = persistent.arrays
        xland = self._xland(batch)
        water = xland >= xp.float32(1.5)
        # MAVAIL is WRF's surface moisture availability: LANDUSE.TBL SLMO,
        # 1.0 for every water category, and the regional driver hands its
        # open-water sfclay call ones (core/physics.py).  The kernel scales
        # the water-branch qfx by it linearly (sfclay.cu:494,497) with
        # nothing compensating, and Noah never integrates or rewrites a
        # water column's soil, so handing sfclay the top-layer volumetric
        # soil moisture on water columns ran the ocean -- the model's only
        # moisture source there -- at the 0.25 fill for the whole run
        # (np_sfclay authority, tropical column: LH 33.82 vs 135.28 W/m2,
        # ratio exactly 0.25; audit 2026-09-01 NB-1).  Land keeps the
        # soil-derived value, which is what its slab-style bulk flux means.
        mavail = xp.ascontiguousarray(
            xp.where(
                water, xp.float32(1.0),
                xp.asarray(batch.surface.soil_water_fraction[0], dtype=xp.float32),
            )
        )
        # Sea ice and land ice are land to sfclay (xland 1) but Noah never
        # writes their soil, so the top-layer moisture there is the source
        # fill; WRF hands these columns LANDUSE.TBL's value for the ice
        # class instead (landuse_init MAVAIL), and so does this.
        frozen = self._frozen_columns(batch, persistent)
        # The skin the surface layer exchanges with.  On a sea-ice column
        # it is the ice's own skin (the column's top node), not the
        # fraction-weighted blend the state carries for the radiation and
        # the 2 m diagnostic: the ice tile's fluxes are the ice's, with the
        # ice's own stability, and the open water of a partial pack runs as
        # its own tile below.  Land ice carries the column's skin already.
        tsk = batch.surface.temperature_k
        if self._frozen_count:
            mavail = xp.ascontiguousarray(xp.where(
                frozen, xp.float32(self._ice_moisture_availability(persistent)), mavail
            ))
            seaice = frozen & frozen_water_columns(batch.surface.sea_ice_fraction, xp)
            tsk = xp.ascontiguousarray(xp.where(
                seaice, xp.asarray(batch.surface.soil_temperature_k[0], dtype=xp.float32),
                xp.asarray(tsk, dtype=xp.float32),
            ))
        vegfra = xp.ascontiguousarray(
            xp.asarray(batch.surface.vegetation_fraction, dtype=xp.float32)
        )
        lakemask = batch.xp.zeros(batch.surface_shape, dtype=batch.xp.float32)
        result = module.sfclay(
            batch.arrays["u"][0], batch.arrays["v"][0],
            batch.arrays["temperature"][0], batch.arrays["qv"][0],
            batch.arrays["p_full"][0], dz[0], batch.arrays["p_half"][0],
            tsk, batch.surface.roughness_m,
            f["pblh"], mavail, xland,
            option=self.options.sfclay_option,
            qsfc=f["qsfc"], zol=f["zol"], ust=f["ust"], mol=f["mol"],
            hfx=f["hfx"], qfx=f["qfx"],
            lakemask=lakemask,
            dx=self.options.dx_m,
            iz0tlnd=self.options.sfclay_iz0tlnd,
            # The statics' vegetation fraction (0..1), read by the kernel
            # only under sfclay_iz0tlnd = 3.
            vegfra=vegfra,
        )
        missing = [
            name for name in REQUIRED_SFCLAY_FIELDS if not hasattr(result, name)
        ]
        if missing:
            raise ValueError(
                "surface-layer result is missing "
                f"{', '.join(missing)}; Noah's surface energy and moisture "
                "balance would have to be driven with substituted exchange "
                "coefficients instead of the ones sfclay computed"
            )
        for name in REQUIRED_SFCLAY_FIELDS:
            f[name][...] = getattr(result, name)
        if self._frozen_count:
            # The ice tile's fluxes, kept for the frozen column whatever
            # the atmosphere receives on a partial pack.  ON THE FROZEN
            # COLUMNS ONLY: a tile that does not exist on a column writes
            # nothing there.  Written whole, these planes held the
            # surface layer's fluxes on every column of whatever batch
            # happened to carry a frozen one, so a band without one kept
            # its seed where the resident run wrote a value nothing reads
            # (MEASURED 2026-09-07, the ten-step T255 gate: ice_hfx and
            # ice_qfx differed between one band and eight on exactly those
            # columns), and the checkpoint depended on the band count.
            f["ice_hfx"][...] = xp.where(frozen, f["hfx"], f["ice_hfx"])
            f["ice_qfx"][...] = xp.where(frozen, f["qfx"], f["ice_qfx"])
        if self._partial_count:
            self._lead_tile_step(batch, persistent, module, dz, lakemask, vegfra)
        # znt is WRF inout: over water the kernel returns the Charnock
        # roughness of this call's friction velocity (sfclay.cu:479-480,
        # :504) and the next call starts from it.  The bridge stored that
        # output only in the persistent 'znt' YSU reads, while the next
        # sfclay input was surface.roughness_m, which only Noah writes and
        # only on the land columns it integrates -- so every ocean call
        # restarted from the 1e-4 m initial value.  Measured against the
        # converged authority: ust 0.78 vs 1.00 m/s at 25 m/s, surface
        # stress 13-33% low from 10 to 20 m/s over the whole ocean (audit
        # 2026-09-01 NB-4).  Water columns now carry sfclay's znt in the
        # surface state; land keeps Noah's.
        batch.surface.roughness_m[...] = xp.where(
            water, xp.asarray(result.znt, dtype=xp.float32),
            batch.surface.roughness_m,
        )
        for name in OPTIONAL_SFCLAY_FIELDS:
            if hasattr(result, name):
                f[name][...] = getattr(result, name)
        if all(hasattr(result, name) for name in DIAGNOSTIC_SFCLAY_FIELDS):
            for name in DIAGNOSTIC_SFCLAY_FIELDS:
                value = getattr(result, name)
                if name == "q2":
                    # The kernel's Q2 is a dry mixing ratio like every
                    # humidity it touches; the physics state and the render
                    # tape carry the model's specific humidity, so the
                    # diagnostic converts on its way out exactly as the
                    # species do (native_batch; audit 2026-09-01 NB-3).
                    value = value / (1.0 + value)
                f[name][...] = value
            persistent.metadata[SOURCE_METADATA_KEY] = NATIVE_SURFACE_DIAGNOSTICS_SOURCE
        else:
            for name in DIAGNOSTIC_SFCLAY_FIELDS:
                persistent.arrays.pop(name, None)
            persistent.metadata.pop(SOURCE_METADATA_KEY, None)
        return result

    @staticmethod
    def _category(xp, value):
        """A one-based category plane, int32, from its float carrier."""
        return xp.ascontiguousarray(xp.asarray(xp.rint(value), dtype=xp.int32))

    @staticmethod
    def _statics_convention(persistent) -> CategoryConvention:
        """The convention the surface state's categories index, from the
        row the cold start / migration / checkpoint carries in the physics
        metadata; a state without one is refused, because the categories
        would then index whatever section the tables happened to load."""
        row = persistent.metadata.get(SURFACE_STATICS_METADATA_KEY)
        if row is None:
            raise ValueError(
                "the surface state carries static land-use and soil "
                f"categories but no physics_state.metadata[{SURFACE_STATICS_METADATA_KEY!r}] "
                "row naming the VEGPARM/SOILPARM sections they index, so "
                "Noah cannot check them against the tables it loads; a "
                "state cold-started, migrated or checkpointed by "
                "woof.globe.statics carries that row -- cold-start "
                "this configuration again or migrate the checkpoint"
            )
        try:
            return CategoryConvention.from_metadata(row)
        except ValueError as exc:
            raise ValueError(f"surface statics convention is malformed: {exc}") from exc

    def _verify_statics(self, batch, persistent, params, convention) -> None:
        """Refuse a surface state Noah would misread.

        The kernel indexes VEGPARM with ``vegtyp - 1`` and SOILPARM with
        ``soiltyp - 1`` without a bounds check (noah.cu:1056-1057), skips
        ``vegtyp == isice`` (noah.cu:976) and treats ``isurban`` specially
        (noah.cu:1106): a category from another dataset, out of range, or
        an isice/isurban option from another dataset silently runs a
        column with another class's parameters.  A land column carrying
        the water category would be integrated with VEGPARM's water row
        and a water column carrying a land class would be skipped while
        sfclay runs it as land; both are refused by name.  The categories
        are static, so this runs once per runtime.
        """
        xp = batch.xp
        surface = batch.surface
        if (str(params.lutype) != convention.landuse_dataset
                or str(params.sltype) != convention.soil_dataset):
            raise ValueError(
                "the surface state's categories index the "
                f"{convention.landuse_dataset}/{convention.soil_dataset} "
                f"table sections but Noah loaded {params.lutype}/"
                f"{params.sltype}; the statics and the tables must name "
                "the same land-use and soil datasets"
            )
        for name, actual, expected in (
            ("land_ice_category", self.options.land_ice_category, convention.ice_category),
            ("urban_category", self.options.urban_category, convention.urban_category),
        ):
            if int(actual) != int(expected):
                raise ValueError(
                    f"native adapter option {name}={actual} is not the "
                    f"{convention.landuse_dataset} value {expected} the "
                    "surface state's categories were built with; Noah "
                    "would skip or urbanise the wrong class"
                )
        water = self._xland(batch) >= xp.float32(1.5)
        ivgtyp = self._category(xp, surface.landuse_category)
        isltyp = self._category(xp, surface.soil_category_top)
        is_water_class = ivgtyp == int(convention.water_category)
        land_with_water_class = int(xp.count_nonzero(~water & is_water_class))
        water_with_land_class = int(xp.count_nonzero(water & ~is_water_class))
        if land_with_water_class or water_with_land_class:
            raise ValueError(
                "the surface state's land/water split disagrees with its "
                f"land-use categories: {land_with_water_class} land column(s) "
                f"(land_fraction > 0.5) carry the water category "
                f"{convention.water_category} and {water_with_land_class} "
                "water column(s) carry a land class; Noah decides by "
                "xland = 1 + (1 - land_fraction) >= 1.5 and would integrate "
                "the former with VEGPARM's water row and skip the latter "
                "while sfclay runs them as land -- rebuild the statics "
                "(woof global statics <config>) or "
                "cold-start the state again"
            )
        land = ~water
        veg_bad = int(xp.count_nonzero(
            land & ((ivgtyp < 1) | (ivgtyp > int(params.lucats)))
        ))
        soil_bad = int(xp.count_nonzero(
            land & ((isltyp < 1) | (isltyp > int(params.slcats)))
        ))
        if veg_bad or soil_bad:
            raise ValueError(
                f"{veg_bad} land column(s) carry a land-use category outside "
                f"VEGPARM {params.lutype} 1..{params.lucats} and {soil_bad} a "
                f"soil category outside SOILPARM {params.sltype} 1..{params.slcats}; "
                "the kernel indexes its tables with the category and has no "
                "bounds check"
            )

    def _noah_fields(self, batch, persistent):
        xp = batch.xp
        noah = self._module("woof.globe.core.noah")
        f = persistent.arrays
        s = batch.surface_shape
        dz = self._dz(batch)
        surface = batch.surface
        xland = self._xland(batch)
        # The static surface fields travel in the surface state (real
        # WPS_GEOG statics, or the declared synthetic planet), seeded once
        # by woof.globe.statics; the runtime reads, never invents.
        dev = {
            "ivgtyp": self._category(xp, surface.landuse_category),
            "isltyp": self._category(xp, surface.soil_category_top),
        }
        inputs = {
            "psfc": batch.arrays["p_half"][0],
            "sfcprs": 0.5 * (batch.arrays["p_half"][0] + batch.arrays["p_half"][1]),
            "sfctmp": batch.arrays["temperature"][0],
            "qv1": batch.arrays["qv"][0], "qgh": f["qgh"],
            "dz8w1": dz[0], "glw": f["glw"], "swdown": f["swdown"],
            # land_rainbl is already the total precipitation (Morrison writes
            # rainncv as the total and snowncv/graupelncv as parts of it), and
            # sr is the frozen fraction of that total.
            "rainbl": f["land_rainbl"],
            "sr": (f["land_snowbl"] + f["land_graupelbl"])
            / xp.maximum(f["land_rainbl"], xp.float32(1.0e-12)),
            "chs": f["chs"], "cqs2": f["cqs2"], "chs2": f["chs2"],
            "rib": f["br"],
            # Percent, as VEGFRA/SHDMIN/SHDMAX are in WRF (noah.cu:994-998
            # divides by 100).
            "vegfra": 100.0 * surface.vegetation_fraction,
            "shdmin": 100.0 * surface.vegetation_fraction_min,
            "shdmax": 100.0 * surface.vegetation_fraction_max,
            "tmn": surface.deep_soil_temperature_k, "xland": xland,
            # The analysed sea-ice fraction (surface_seeding): the kernel
            # skips every column at or above its threshold as sea ice
            # (noah.cu:969) and the frozen-surface step integrates the skin.
            "xice": surface.sea_ice_fraction,
            "snoalb": surface.snow_albedo, "embck": surface.emissivity,
            "tsk": surface.temperature_k, "hfx": f["hfx"],
            "qfx": f["qfx"], "lh": f["noah_lh"],
            "grdflx": f["noah_grdflx"], "qsfc": f["qsfc"],
            "canwat": f["noah_canwat"], "snow": f["noah_snow"],
            "snowc": f["noah_snowc"], "snowh": f["noah_snowh"],
            "albedo": surface.albedo,
            "albbck": surface.background_albedo,
            "emiss": surface.emissivity,
            "znt": surface.roughness_m, "z0": surface.roughness_m.copy(),
            "snotime": xp.zeros(s, dtype=xp.float32),
            "lai": surface.leaf_area_index,
            "smstav": xp.zeros(s, dtype=xp.float32),
            "smstot": xp.zeros(s, dtype=xp.float32),
            "sfcrunoff": f["noah_sfcrunoff"], "udrunoff": f["noah_udrunoff"],
            "acsnow": xp.zeros(s, dtype=xp.float32),
            "acsnom": xp.zeros(s, dtype=xp.float32),
            "snopcx": xp.zeros(s, dtype=xp.float32),
            "potevp": xp.zeros(s, dtype=xp.float32),
            "noahres": xp.zeros(s, dtype=xp.float32),
            "reslin": xp.zeros(s, dtype=xp.float32),
            "chklowq": xp.zeros(s, dtype=xp.float32),
        }
        for name in noah._F2D:
            value = inputs.get(name)
            if value is None:
                value = xp.zeros(s, dtype=xp.float32)
            dev[name] = xp.ascontiguousarray(xp.asarray(value, dtype=xp.float32))
        dev.update(
            smois=f["noah_smois"], tslb=f["noah_tslb"],
            sh2o=f["noah_sh2o"], smcrel=f["noah_smcrel"],
            ebal=xp.zeros(s, dtype=xp.int32),
        )
        return noah, dev

    def _priced_land_stores(self, batch, persistent):
        """HELD land-store water priced exactly as both water ledgers price it.

        Mirrors native_suite._water_column / water.py: soil is water per
        unit land area and takes land_fraction; canwat and snow count at
        full weight.  The runoff accumulators are priced separately by
        _priced_runoff_stores because their increments are booked exits,
        not held water.  The ledgers and these bookings must stay at
        matching weights or the closure sees the difference as created or
        destroyed water.
        """
        xp = batch.xp
        f = persistent.arrays
        lf = batch.surface.land_fraction
        total = (
            batch.surface.soil_water_fraction
            * xp.asarray(SOIL_LAYER_THICKNESS_M, dtype=xp.float32)[:, None, None]
            * xp.float32(1000.0)
            * lf[None]
        ).sum(axis=0)
        total = total + xp.maximum(f["noah_canwat"], 0.0)
        total = total + xp.maximum(f["noah_snow"], 0.0)
        return total

    def _priced_runoff_stores(self, batch, persistent):
        """Runoff-accumulator water at ledger pricing (per unit land area).

        The kernel's sfcrunoff/udrunoff are monotone accumulators of water
        that has left the column system for the rivers; the pricing here is
        the v3 ledger's exactly (max(store, 0) * land_fraction), but the
        increment between two calls is now booked to the cumulative outflow
        account instead of being counted as held water.
        """
        xp = batch.xp
        f = persistent.arrays
        lf = batch.surface.land_fraction
        return (
            xp.maximum(f["noah_sfcrunoff"], 0.0) * lf
            + xp.maximum(f["noah_udrunoff"], 0.0) * lf
        )

    def _hold_land_fluxes(self, batch, persistent):
        """Land columns keep the land surface's fluxes across non-due calls.

        The surface layer overwrites the persistent hfx/qfx/qsfc on EVERY
        call, and YSU reads them.  With land_surface_interval_s = dt (the
        shipping T255 setting) the dycore's Strang halves put two physics
        calls in every land bucket -- time_s t and t+dt, the second half
        of step n and the first half of step n+1 share a time_s -- so Noah
        was due on only one of them and YSU's land columns received
        sfclay's own bulk-aerodynamic fluxes on the other: 3 of 8 calls in
        the four-step reproduction, asymptotically 1 in 2, bypassing Noah's
        surface energy balance for half of all PBL forcing on land (audit
        2026-09-01 NB-2).  WRF's surface_driver overwrites HFX/QFX/QSFC with
        the LSM's before the PBL sees them, every step; the bridge holds
        the same rule at physics-call granularity.  Water columns keep
        sfclay's fluxes, which are their only source.  Running Noah on
        every call instead would integrate the soil twice per model second
        on the shared-time_s pairs (or with an elapsed of zero), so the
        held-flux form is the one that matches WRF's cadence.
        """
        if persistent.metadata.get("last_land_time_s") is None:
            return  # Noah has never run; there is nothing to hold yet.
        xp = batch.xp
        f = persistent.arrays
        land = self._xland(batch) < xp.float32(1.5)
        for name in NOAH_HELD_FLUX_FIELDS:
            f[name][...] = xp.where(land, f["noah_" + name], f[name])

    def _land_step(self, batch, persistent):
        bucket = self._bucket(batch.time_s, self.options.land_surface_interval_s)
        due = bucket != int(persistent.metadata["last_land_bucket"])
        if not due:
            self._hold_land_fluxes(batch, persistent)
            return False
        # The soil must integrate the time that actually elapsed since the last
        # due call, not the configured interval: the dycore Strang-splits
        # physics, so the gap between two due land calls is the model time
        # step, which need not equal land_surface_interval_s.
        previous = persistent.metadata.get("last_land_time_s")
        elapsed = self.options.land_surface_interval_s
        if previous is not None:
            measured = float(batch.time_s) - float(previous)
            if measured > 0.0:
                elapsed = measured
        noah, dev = self._noah_fields(batch, persistent)
        if self._noah_params is None:
            # The table sections are the ones the surface state's
            # categories were built against, never a default that happens
            # to agree.
            convention = self._statics_convention(persistent)
            self._noah_params = noah.pack_params(noah.load_tables(
                mminlu=convention.landuse_dataset,
                mminsl=convention.soil_dataset,
            ))
        if not self._statics_checked:
            self._verify_statics(
                batch, persistent, self._noah_params,
                self._statics_convention(persistent),
            )
            self._statics_checked = True
        xp = batch.xp
        # The kernel skips every xland >= 1.5 column before SFLX (open water;
        # noah.cu:968), but its first call also writes the WRF driver's
        # diagnostic fill into exactly those columns first (noah.cu:940-948,
        # module_sf_noahdrv.F:749-788 convention): smois = 1, tslb = 273.16,
        # smcrel = 1.  That fill is a constant stamped on columns the scheme
        # never integrates, not water that arrived from anywhere; if it
        # reaches surface.soil_water_fraction, the suite's closure prices it
        # at (1 - smois) * 2 m * 1000 kg/m3 * land_fraction of created water
        # per column, which on coastal columns (land_fraction in (0, 0.5])
        # out-prices the 500 kg/m2 surface reservoir and kills step 0 with
        # "native physics water closure exceeds the explicit surface
        # reservoir".  The T255 analysis-init device run raised exactly that
        # on its first physics call (a fractional coastline band exists on
        # every analysis-regridded land mask); the CPU reproduction of the
        # call (T21, idealized init) measured the worst-column booking at
        # +717.9 kg/m2.  Never-integrated
        # columns therefore carry their soil state across the call
        # unchanged.  The mask repeats _noah_fields' xland construction
        # bit-for-bit so it selects exactly the columns the kernel skipped:
        # open water (noah.cu:968), sea ice (noah.cu:969, whose first-call
        # fill stamps smois = smcrel = 1 and every call sh2o = 1) and land
        # ice (noah.cu:976).
        skipped = (self._xland(batch) >= xp.float32(1.5)) | self._frozen_columns(
            batch, persistent
        )
        preserved = None
        if bool(xp.any(skipped)):
            preserved = {
                name: dev[name].copy() for name in ("smois", "tslb", "smcrel", "sh2o")
            }
        stores_before = self._priced_land_stores(batch, persistent)
        runoff_before = self._priced_runoff_stores(batch, persistent)
        noah.launch_noah(
            dev, self._noah_params, elapsed,
            SOIL_LAYER_THICKNESS_M,
            isurban=self.options.urban_category,
            isice=self.options.land_ice_category,
            frpcpn=True,
            usemonalb=self.options.use_monthly_albedo,
            rdlai2d=self.options.read_lai_2d,
            opt_thcnd=self.options.noah_thermal_conductivity_option,
            itimestep=int(persistent.metadata["native_calls"]) + 1,
        )
        if preserved is not None:
            # The kernel wrote dev's soil arrays in place (they are the
            # persistent noah_* buffers), so the skipped columns are restored
            # there before anything downstream reads them.
            for name, held in preserved.items():
                dev[name][...] = xp.where(skipped[None], held, dev[name])
        batch.surface.temperature_k[...] = dev["tsk"]
        batch.surface.albedo[...] = dev["albedo"]
        batch.surface.emissivity[...] = dev["emiss"]
        batch.surface.roughness_m[...] = dev["znt"]
        batch.surface.soil_temperature_k[...] = dev["tslb"]
        batch.surface.soil_water_fraction[...] = dev["smois"]
        # WRF's surface_driver runs the LSM after sfclay and the PBL
        # driver reads ZNT afterwards, so on a land column YSU sees the
        # roughness Noah just set (VEGPARM Z0MIN/Z0MAX by green fraction,
        # snow-buried; noah.cu:1486) on the same step.  The persistent
        # 'znt' YSU reads was sfclay's output, which on land is its input,
        # the surface state's roughness BEFORE this call: the statics'
        # LANDUSE.TBL value on the first due call and Noah's previous one
        # after.  Water keeps sfclay's Charnock value.
        persistent.arrays["znt"][...] = xp.where(
            ~skipped, dev["znt"], persistent.arrays["znt"]
        )
        for name, key in (
            ("hfx", "hfx"), ("qfx", "qfx"), ("qsfc", "qsfc"),
            ("noah_hfx", "hfx"), ("noah_qfx", "qfx"), ("noah_qsfc", "qsfc"),
            ("noah_lh", "lh"), ("noah_grdflx", "grdflx"),
            ("noah_canwat", "canwat"), ("noah_snow", "snow"),
            ("noah_snowc", "snowc"), ("noah_snowh", "snowh"),
            ("noah_smois", "smois"), ("noah_tslb", "tslb"),
            ("noah_sh2o", "sh2o"), ("noah_smcrel", "smcrel"),
            ("noah_sfcrunoff", "sfcrunoff"), ("noah_udrunoff", "udrunoff"),
        ):
            persistent.arrays[name][...] = dev[key]
        # Every kg/m2 Noah moved into its stores this call is water the
        # surface reservoir already holds: _microphysics_step credits the
        # full precipitation total to the reservoir AND queues the same
        # water in land_rainbl, and the write-backs above put the consumed
        # rainbl into canwat/snow/smois/sfcrunoff/udrunoff while the
        # reservoir kept its copy.  Without this debit the closure measured
        # that duplicate as the repair, and both T255 six-hour
        # qualification receipts failed the
        # physics_water_repair_max_step_kg_m2 gate at 1.0943603515625 kg/m2
        # (one land-interval convective burst) against 5e-4 (2026-08-31).
        # Debiting the priced store delta also settles Noah's own
        # evaporation and dew (store losses/gains matched by the qfx flux
        # the PBL books below), so the reservoir clears every land-store
        # move at ledger pricing instead of the closure repairing it.
        #
        # The runoff part of the debit is an EXIT, not a move to a held
        # store: the reservoir pays it out of the precipitation credit and
        # the cumulative outflow account records it, so the closure ledger
        # (held + outflow) stays exact while the monotone kernel
        # accumulators stop sequestering held water inside the pinned
        # conservation total (they squeezed the global atmosphere+reservoir
        # mean through the fixer target by every kg they grew, forever).
        runoff_delta = (
            self._priced_runoff_stores(batch, persistent) - runoff_before
        )
        batch.surface.water_kg_m2 -= (
            self._priced_land_stores(batch, persistent) - stores_before
        ) + runoff_delta
        persistent.arrays["water_outflow_kg_m2"] += runoff_delta
        persistent.arrays["land_rainbl"].fill(0.0)
        persistent.arrays["land_snowbl"].fill(0.0)
        persistent.arrays["land_graupelbl"].fill(0.0)
        self._frozen_surface_step(batch, persistent, elapsed)
        self._lake_surface_step(batch, persistent, elapsed)
        persistent.metadata["last_land_bucket"] = bucket
        persistent.metadata["last_land_time_s"] = float(batch.time_s)
        return True

    def _lead_tile_step(self, batch, persistent, module, dz, lakemask, vegfra) -> None:
        """The open water of a partial pack as its own surface-layer tile.

        A cell whose analysed sea-ice fraction lies between one half and
        one is ice to Noah and the frozen column and open water for the
        rest of its area, 15 to 20 K warmer than a winter pack's air.  One
        column at the blended skin gives the whole cell one stability: the
        merged tip's Antarctic marginal pack sent the air 10 W/m2 where the
        reference's tiled composite sent 27, and with the ice column
        charged its own flux the same single column sent 40.  The leads
        run here as WRF's water surface layer would (xland 2, moisture
        availability one, the skin at the freezing point of sea water, the
        Charnock roughness and the friction velocity, Monin-Obukhov scale,
        saturation humidity and fluxes of this tile carried across calls in
        their own persistent planes), and the fluxes the atmosphere
        receives on a partial cell are the area composite of the two
        tiles, the friction velocity the stress-weighted composite; the
        ice tile keeps every other diagnostic (it is the larger tile by
        the threshold rule).  Cells outside the partial pack are untouched.
        """
        xp = batch.xp
        f = persistent.arrays
        partial = self._partial
        shape = batch.surface_shape
        lead = module.sfclay(
            batch.arrays["u"][0], batch.arrays["v"][0],
            batch.arrays["temperature"][0], batch.arrays["qv"][0],
            batch.arrays["p_full"][0], dz[0], batch.arrays["p_half"][0],
            xp.full(shape, xp.float32(frozen_surface.SEA_ICE_BOTTOM_K), dtype=xp.float32),
            f["lead_znt"], f["pblh"],
            xp.ones(shape, dtype=xp.float32),
            xp.full(shape, xp.float32(2.0), dtype=xp.float32),
            option=self.options.sfclay_option,
            qsfc=f["lead_qsfc"], zol=f["lead_zol"], ust=f["lead_ust"], mol=f["lead_mol"],
            hfx=f["lead_hfx"], qfx=f["lead_qfx"],
            lakemask=lakemask, dx=self.options.dx_m,
            iz0tlnd=self.options.sfclay_iz0tlnd, vegfra=vegfra,
        )
        # The lead tile's own state and the composite it makes are written
        # ON THE PARTIAL COLUMNS ONLY: a tile that does not exist on a
        # column writes nothing there.  The kernel runs on the whole batch
        # (column-local, so the partial columns' values are the same
        # whichever batch carries them), and outside the pack its output
        # was a value nothing reads -- but it was written, and an
        # identity composite (1 x hfx + 0 x lead, sqrt(ust^2)) is not the
        # identity in floating point.  MEASURED 2026-09-07, the ten-step
        # T255 gate: the nine lead planes differed between one band and
        # eight on exactly the columns outside the pack.
        for name in ("znt", "ust", "mol", "zol", "qsfc", "hfx", "qfx", "chs2", "cqs2"):
            f["lead_" + name][...] = xp.where(
                partial, xp.asarray(getattr(lead, name), dtype=xp.float32),
                f["lead_" + name])
        open_water = xp.where(
            partial,
            xp.float32(1.0) - xp.clip(
                xp.asarray(batch.surface.sea_ice_fraction, dtype=xp.float32),
                xp.float32(0.0), xp.float32(1.0)),
            xp.float32(0.0),
        )
        ice = xp.float32(1.0) - open_water
        f["hfx"][...] = xp.where(partial, ice * f["hfx"] + open_water * f["lead_hfx"], f["hfx"])
        f["qfx"][...] = xp.where(partial, ice * f["qfx"] + open_water * f["lead_qfx"], f["qfx"])
        f["ust"][...] = xp.where(
            partial,
            xp.sqrt(ice * f["ust"] * f["ust"] + open_water * f["lead_ust"] * f["lead_ust"]),
            f["ust"])
        persistent.metadata["lead_tile_calls"] = int(persistent.metadata.get("lead_tile_calls", 0)) + 1

    def _screen_level_step(self, batch, persistent) -> bool:
        """WRF's SFCDIAGS on the land columns, after the land surface.

        The surface layer diagnoses T2/TH2/Q2 from the skin as it stood
        BEFORE the land surface integrates it, by its own bulk profile
        (sfclay.cu th2 = thgb + (thx - thgb) psit2 / psit).  WRF never
        publishes that value on a Noah column: surface_driver calls
        SFCDIAGS after the LSM and before the PBL (module_surface_driver.F:
        2983-3000; module_sf_sfcdiags.F:45-72) and overwrites the three
        fields from the LSM's own skin and fluxes,

            rho = PSFC / (R_d TSK)
            T2  = TSK  - HFX / (rho c_p CHS2)     (TSK if CHS2 < 1e-5)
            Q2  = QSFC - QFX / (rho CQS2)         (QSFC if CQS2 < 1e-5)
            TH2 = T2 (P0 / PSFC)^(R_d/c_p)

        with CHS2 = CQS2 on every ordinary land column Noah integrated
        (module_sf_noahdrv.F:1275).  The regional driver carries the same
        transcription (woof.globe.core.physics PhysicsDriver._refresh_surface_
        diagnostics); the bridge published the surface layer's value
        instead, a 2 m temperature no WRF configuration produces, which
        sat +2.70 K above SFCDIAGS's over CONUS land at 18Z on the T255
        control (area mean of 4,769 cells, rmse 2.91 K; -0.64 K at night),
        the same arrays read both ways.

        Runs on every call once Noah has integrated at least once: on the
        calls where Noah is not due the held Noah fluxes and the held skin
        are the ones YSU is about to read (_hold_land_fluxes), so the
        published 2 m value describes the same surface the PBL is forced
        with.  Two documented divergences.  The regional driver's: a Q2
        the flux inversion drives out of the physical range (downward
        moisture flux over very cold snow; WRF's own remedy is commented
        out at module_sf_noahdrv.F:1276-1282) publishes the lowest level's
        vapor instead of a negative mixing ratio, and only there.  And the
        water columns: WRF's surface_driver calls SFCDIAGS on the whole
        tile, water included (module_surface_driver.F:2994-2998, with the
        surface layer's own CHS2 there), where this bridge keeps the surface
        layer's T2/TH2/Q2.  The two forms differ by the cpm/cp factor in
        the flux and the theta-to-temperature conversion of the skin:
        measured on the T255 control at 18Z, +0.009 K mean and 0.033 K rmse
        over water; the largest differences (2.5 K, 238 cells of 195,153)
        sit on high-altitude lake cells whose skin the cold start holds at
        the analysis water temperature, a statics matter.  The kernel-side
        mixing ratios are converted to the model's specific humidity on the
        way out exactly as the surface layer's q2 is.
        """
        if persistent.metadata.get("last_land_time_s") is None:
            return False   # Noah has never integrated; the skin is the analysis's.
        f = persistent.arrays
        if not all(name in f for name in DIAGNOSTIC_SFCLAY_FIELDS):
            return False   # a surface layer without screen fields leaves none behind
        from woof.core import constants as c

        xp = batch.xp
        one = xp.float32(1.0)
        land = self._xland(batch) < xp.float32(1.5)
        tsk = batch.surface.temperature_k
        psfc = batch.arrays["p_half"][0]
        hfx, qfx = f["hfx"], f["qfx"]
        if self._frozen_count:
            # The ice tile's own skin and fluxes on sea ice (the state's
            # skin is the blend the radiation sees); the leads' 2 m values
            # are composited in below on a partial pack.
            frozen = self._frozen_columns(batch, persistent)
            seaice = frozen & frozen_water_columns(batch.surface.sea_ice_fraction, xp)
            tsk = xp.where(seaice, xp.asarray(batch.surface.soil_temperature_k[0], dtype=xp.float32), tsk)
            hfx = xp.where(frozen, f["ice_hfx"], hfx)
            qfx = xp.where(frozen, f["ice_qfx"], qfx)
        rho = psfc / (xp.float32(c.RD) * tsk)
        cqs2 = f["cqs2"]
        active = cqs2 >= xp.float32(1.0e-5)
        safe = xp.where(active, cqs2, one)
        t2 = xp.where(
            active, tsk - hfx / (rho * xp.float32(c.CP) * safe), tsk
        )
        q_mix = xp.where(active, f["qsfc"] - qfx / (rho * safe), f["qsfc"])
        if self._partial_count:
            partial = self._partial
            fraction = xp.clip(xp.asarray(batch.surface.sea_ice_fraction, dtype=xp.float32),
                               xp.float32(0.0), xp.float32(1.0))
            water_k = xp.float32(frozen_surface.SEA_ICE_BOTTOM_K)
            rho_lead = psfc / (xp.float32(c.RD) * water_k)
            lead_active = f["lead_cqs2"] >= xp.float32(1.0e-5)
            lead_safe = xp.where(lead_active, f["lead_cqs2"], one)
            t2_lead = xp.where(
                lead_active, water_k - f["lead_hfx"] / (rho_lead * xp.float32(c.CP) * lead_safe), water_k
            )
            q_lead = xp.where(lead_active, f["lead_qsfc"] - f["lead_qfx"] / (rho_lead * lead_safe), f["lead_qsfc"])
            t2 = xp.where(partial, fraction * t2 + (one - fraction) * t2_lead, t2)
            q_mix = xp.where(partial, fraction * q_mix + (one - fraction) * q_lead, q_mix)
        representable = q_mix > xp.float32(0.0)
        # batch.arrays["qv"] is the kernel-side mixing ratio (native_batch);
        # both branches leave here as the model's specific humidity.
        qv_lowest = batch.arrays["qv"][0]
        q2 = xp.where(
            representable, q_mix / (one + q_mix), qv_lowest / (one + qv_lowest)
        )
        th2 = t2 * xp.power(xp.float32(c.P0) / psfc, xp.float32(c.RCP))
        f["t2"][...] = xp.where(land, xp.asarray(t2, dtype=xp.float32), f["t2"])
        f["th2"][...] = xp.where(land, xp.asarray(th2, dtype=xp.float32), f["th2"])
        f["q2"][...] = xp.where(land, xp.asarray(q2, dtype=xp.float32), f["q2"])
        return True

    def _frozen_surface_step(self, batch, persistent, elapsed) -> None:
        """Heat conduction column of the columns Noah skipped as frozen.

        Sea ice and land ice (frozen_surface.py) get one implicit step of
        the four-node column that lives in their soil-temperature layers,
        on the land cadence: the radiation planes the suite holds, the
        surface layer's fluxes of this call corrected to the ice skin's
        own by the surface layer's exchange coefficient (rho c_p C_h and
        rho m C_h from this call's chs, implicit in the new skin; the
        open water of a partial pack keeps what it exchanged with the
        air), conduction through the snow and the ice under it (each
        node takes the medium at its depth) to the freezing point of sea
        water at the analysed ice bottom or to the deep-soil climatology
        at 8 m through firn, capped at the melting point.  The skin the
        surface state carries is the column's top node, blended with open
        water at the freezing point by the analysed fraction on sea ice,
        and the albedo and emissivity the radiation sees are WRF's
        fractional blend of the sea-ice values (0.65, 0.98) with open
        water (0.08, 0.98); the column itself absorbs with the ice's own.
        The surface layer's saturation humidity on those columns is
        refreshed to the value over ice at the blended skin, in both the
        live and the held copy the land columns hand YSU.  A run without
        frozen columns never enters the arithmetic.
        """
        if not self._frozen_count:
            return
        xp = batch.xp
        f = persistent.arrays
        surface = batch.surface
        frozen = self._frozen_columns(batch, persistent)
        seaice = frozen & frozen_water_columns(surface.sea_ice_fraction, xp)
        layers = xp.asarray(surface.soil_temperature_k, dtype=xp.float32)
        dz = xp.where(
            seaice[None],
            frozen_surface.sea_ice_layer_thickness(
                surface.sea_ice_thickness_m, f["noah_snowh"], xp
            ),
            frozen_surface.land_ice_layer_thickness(surface.temperature_k, xp),
        )
        bottom = xp.where(
            seaice, xp.float32(frozen_surface.SEA_ICE_BOTTOM_K),
            xp.asarray(surface.deep_soil_temperature_k, dtype=xp.float32),
        )
        firn = xp.float32(
            (frozen_surface.LAND_ICE_BOTTOM_DEPTH_M - sum(frozen_surface.LAND_ICE_LAYER_THICKNESS_M))
            / frozen_surface.ICE_CONDUCTIVITY_W_M_K
        )
        extra = xp.where(seaice, xp.float32(0.0), firn)
        # The ice's own optical surface: WRF's sea-ice albedo and the
        # emissivity SFLX_SEAICE sets on the pack; on a land-ice column the
        # statics' maximum snow albedo when snow-covered, else its class
        # value, with the class emissivity.
        ice_albedo = xp.where(
            seaice, xp.float32(frozen_surface.SEA_ICE_ALBEDO),
            xp.where(
                f["noah_snow"] >= xp.float32(frozen_surface.SNOW_COVERED_KG_M2),
                xp.asarray(surface.snow_albedo, dtype=xp.float32),
                xp.asarray(surface.background_albedo, dtype=xp.float32),
            ),
        )
        ice_emissivity = xp.where(
            seaice, xp.float32(frozen_surface.SEA_ICE_EMISSIVITY),
            xp.asarray(surface.emissivity, dtype=xp.float32),
        )
        # The surface layer evaluated this call's ice-tile fluxes at the
        # column's own skin (the state's skin is the blend, for the
        # radiation and the 2 m diagnostic); its exchange coefficient is
        # the slope that makes the step's own change of the skin implicit.
        from woof.core import constants as c

        psfc = batch.arrays["p_half"][0]
        skin_seen = xp.asarray(layers[0], dtype=xp.float32)
        rho = xp.asarray(psfc, dtype=xp.float32) / (xp.float32(c.RD) * skin_seen)
        exner = xp.power(xp.float32(c.P0) / xp.asarray(psfc, dtype=xp.float32), xp.float32(c.RCP))
        chs = xp.asarray(f["chs"], dtype=xp.float32)
        exchange_heat = rho * xp.float32(c.CP) * chs * exner
        exchange_moisture = rho * xp.float32(self._ice_moisture_availability(persistent)) * chs
        column = frozen_surface.column_step(
            layers=layers, dz=dz, snow_depth_m=f["noah_snowh"], bottom_k=bottom,
            bottom_extra_resistance=extra, swdown=f["swdown"], albedo=ice_albedo,
            glw=f["glw"], emissivity=ice_emissivity, hfx=f["ice_hfx"], qfx=f["ice_qfx"],
            dt_s=elapsed, xp=xp,
            exchange_heat_w_m2_k=exchange_heat, exchange_moisture_kg_m2_s=exchange_moisture,
            skin_reference_k=skin_seen, psfc_pa=psfc,
        )
        surface.soil_temperature_k[...] = xp.where(frozen[None], column, surface.soil_temperature_k)
        f["noah_tslb"][...] = xp.where(frozen[None], column, f["noah_tslb"])
        skin = xp.where(
            seaice,
            frozen_surface.blended_surface_temperature(column[0], surface.sea_ice_fraction, xp),
            column[0],
        )
        surface.temperature_k[...] = xp.where(frozen, skin, surface.temperature_k)
        # What the radiation sees: on a partial pack WRF's fractional blend
        # of the ice surface with open water; elsewhere the ice's own.
        albedo = xp.where(
            seaice, frozen_surface.composite_albedo(ice_albedo, surface.sea_ice_fraction, xp),
            ice_albedo,
        )
        emissivity = xp.where(
            seaice, frozen_surface.composite_emissivity(ice_emissivity, surface.sea_ice_fraction, xp),
            ice_emissivity,
        )
        surface.albedo[...] = xp.where(frozen, albedo, surface.albedo)
        surface.emissivity[...] = xp.where(frozen, emissivity, surface.emissivity)
        # The ice tile's saturation humidity at its own skin (the leads
        # carry theirs in the lead tile's plane).
        qsat = frozen_surface.saturation_specific_humidity_over_ice(
            column[0], batch.arrays["p_half"][0], xp
        )
        for name in ("qsfc", "noah_qsfc"):
            f[name][...] = xp.where(frozen, qsat, f[name])
        persistent.metadata["frozen_surface_calls"] = int(
            persistent.metadata.get("frozen_surface_calls", 0)
        ) + 1

    def _lake_surface_step(self, batch, persistent, elapsed) -> None:
        """The skin of an inland lake follows its own surface energy budget.

        Open water has no ocean or lake model: Noah skips it and the surface
        layer reads its skin as the water temperature.  On the ocean the
        skin is held at the analysis SST for the run, the usual
        medium-range choice for a sea surface the analysis observes.  An
        inland lake (statics.lake_columns, WRF's LAKEMASK rule) is not that
        surface: the analysis's lake temperature is weakly observed, and
        held, a lake the analysis carries too warm evaporates at that rate
        for the whole forecast.  GDAS carried Lake Tana (11.9 N, 37.5 E) at
        304.8 to 305.1 K at all eight cycles of 2026-09-29 and 2026-09-30,
        9.2 K above the 2 m air over it; held for a 240 h T255 forecast,
        the column evaporated about 22 kg/m2 a day (published estimates of
        the lake's open-water evaporation are about 4 to 5 mm a day) and
        drew its surface reservoir toward the refusal "native physics
        water closure exceeds the explicit surface reservoir", about 23
        days out.

        So a lake column integrates its skin on the land cadence from the
        energy it exchanges, with the heat capacity the state already
        carries for water (4e7 J/m2/K, a 10 m mixed layer, blended with the
        column's land share as everywhere else): the shortwave the water
        absorbs (``swdown`` and the surface albedo the radiation saw, as
        Noah and the frozen-surface step take it), the downward longwave
        it absorbs less its own emission, and this call's surface-layer
        sensible and latent fluxes (``hfx``, and ``qfx`` times the latent
        heat of vaporization: the evaporation the PBL books against the
        surface reservoir).  The step is explicit: the skin's flux
        sensitivity is of order 100 W/m2/K, which over a 2e7 J/m2/K
        capacity and a 300 s land interval moves the skin by 0.0015 of its
        own departure per step.  A lake is not cooled below freshwater
        freezing (273.15 K) by this step: there is no lake ice here, and a
        lake the analysis covers with ice is a frozen column, not this
        one.  Nothing else changes: the surface layer reads the new skin
        as the lake's water temperature on the next call, and the ocean,
        land and frozen columns are untouched.
        """
        lake = self._lake_columns(batch)
        if not self._lake_count:
            return
        xp = batch.xp
        f = persistent.arrays
        surface = batch.surface
        skin = xp.asarray(surface.temperature_k, dtype=xp.float32)
        emissivity = xp.asarray(surface.emissivity, dtype=xp.float32)
        albedo = xp.asarray(surface.albedo, dtype=xp.float32)
        net = (
            f["swdown"] * (xp.float32(1.0) - albedo)
            + emissivity * f["glw"]
            - emissivity * xp.float32(STEFAN_BOLTZMANN) * skin ** 4
            - f["hfx"]
            - xp.float32(LATENT_HEAT_VAPORIZATION) * f["qfx"]
        )
        capacity = xp.asarray(surface.heat_capacity_j_m2_k, dtype=xp.float32)
        stepped = skin + xp.float32(elapsed) * net / capacity
        stepped = xp.maximum(
            stepped, xp.minimum(skin, xp.float32(LAKE_FREEZING_K))
        )
        surface.temperature_k[...] = xp.where(lake, stepped, surface.temperature_k)
        persistent.metadata["lake_surface_calls"] = int(
            persistent.metadata.get("lake_surface_calls", 0)
        ) + 1

    def _pbl_step(self, batch, persistent):
        module = self._module("woof.globe.core.ysu")
        f = persistent.arrays
        dz = self._dz(batch)
        out = module.launch_ysu(
            batch.arrays["u"], batch.arrays["v"], batch.arrays["theta"],
            batch.arrays["qv"], batch.arrays["qc"], batch.arrays["qi"],
            batch.arrays["p_full"], batch.arrays["p_half"],
            batch.arrays["exner"], dz,
            rthraten=f["rad_rthratenlw"] + f["rad_rthratensw"],
            psfc=batch.arrays["p_half"][0], znt=f["znt"], ust=f["ust"],
            hfx=f["hfx"], qfx=f["qfx"], wspd=f["wspd"], br=f["br"],
            psim=f["fm"], psih=f["fh"],
            xland=self._xland(batch),
            u10=f["u10"], v10=f["v10"], dt=batch.dt_s,
            ysu_topdown_pblmix=self.options.ysu_topdown_pblmix,
            free_atmosphere_mixing_length=(
                self.options.ysu_free_atmosphere_mixing_length),
        )
        step = batch.xp.float32(batch.dt_s)
        column_before = batch.atmospheric_water_kg_m2()
        for target, source in (
            ("u", "du"), ("v", "dv"), ("theta", "dtheta"),
            ("qv", "dqv"), ("qc", "dqc"), ("qi", "dqi"),
        ):
            batch.arrays[target] += step * out[source]
        # YSU's vapor bottom boundary injects the surface moisture flux into
        # the column (ysu.cu:617: rhs[0] = qv0 + qf*G/delp[0]*dt2); its
        # interior mixing conserves column water, so the applied tendencies'
        # column integral is the water that crossed the surface.  Booking it
        # against the reservoir pairs the atmosphere's gain with a surface
        # debit (and dew, qfx < 0, with a credit).  Unbooked, this flux was
        # the residual floor beneath the precipitation double-entry: measured
        # 1.5869e-3 kg/m2 per 30 s half-call at 125 W/m2 latent -- above the
        # physics_water_repair_max_step_kg_m2 limit of 5e-4 on essentially
        # every moist column (2026-08-31).  The amount is the column change
        # MEASURED in the model's metric (native_batch
        # atmospheric_water_kg_m2), not the kernel's tendency integral in
        # mixing-ratio units: the two differ by O(q) and only the measured
        # one is what the dycore's ledger will see (audit 2026-09-01 NB-3).
        self._credit_reservoir(
            batch, column_before - batch.atmospheric_water_kg_m2()
        )
        for target, source in (
            ("pblh", "hpbl"), ("u10", "u10"), ("v10", "v10"),
        ):
            if source in out:
                f[target][...] = out[source]
        return out

    @staticmethod
    def _credit_reservoir(batch, amount_kg_m2):
        """Move ``amount_kg_m2`` (float64 plane) into the surface reservoir."""
        reservoir = batch.surface.water_kg_m2
        reservoir += batch.xp.asarray(amount_kg_m2, dtype=reservoir.dtype)

    def _cumulus_inputs(self, batch, persistent, entry, pbl_out):
        """The column state woof.globe.core.gf reads, in the regional driver's shape.

        The kernel takes exactly what WRF's cu_gf_driver takes: the current
        column (``t``/``q``), the forcing it is about to receive from the
        other tendencies (RTHRATEN and RTHBLTEN/RQVBLTEN, from which it
        forms the forced state tn = t + dt*(rad + pbl)*pi), the vertical
        velocity, terrain height, the PBL top index, surface fluxes and
        the land flag.  This runtime applies tendencies sequentially, so
        the batch at this point already holds t + dt*(rad + pbl); handing
        the kernel the ENTRY state plus the two held rates reproduces
        WRF's construction exactly (tn equals the batch's current state)
        and the rates it returns are then integrated into that current
        state, which is the same end state WRF's simultaneous application
        reaches.  The advective lanes (RTHFTEN/RQVFTEN) are the dynamics'
        own theta and vapor tendencies over the last dynamics interval,
        measured by this runtime between its calls
        (_measure_dynamics_forcing) and held in the persistent namespace:
        WRF hands GF the RK stage-1 advective tendencies of the current
        step and MPAS-A its own; here the lane lags the dynamics by one
        interval (the previous step's), the same first-order reading.

        ``entry`` None (arwen-massflux-v1) hands the CURRENT column: that
        scheme triggers on the column as forced, and reads the same two
        rate lanes for its quasi-equilibrium closure.  The fields also
        carry the held PBL depth (``pblh``) and the state the exchange's
        omega (``omega_half``), which GF's seam ignores.
        """
        xp = batch.xp
        f = persistent.arrays
        scheme = self.options.cumulus_scheme
        omega = batch.arrays.get("omega_half")
        if omega is None:
            raise ValueError(
                f"cumulus={scheme!r} needs the exchange's omega_half_pa_s: the "
                "Grell-Freitas closure integrates omega directly (gf.cu "
                "omeg = -g rho w feeds the Brown vertical-velocity ensemble "
                "members and the moisture-convergence term mconv), New "
                "Tiedtke's trigger and closure read the resolved w "
                "(ntiedtke.cu cutypen and the CAPE adjustment) and "
                "arwen-massflux-v1's moisture-convergence closure integrates "
                "it too, so an exchange without it would run every column at "
                "w = 0 and silently remove those terms; "
                "dynamics._physics_exchange_with_closure supplies the field, "
                "or set cumulus='none'"
            )
        if entry is None:
            entry = {
                "temperature": batch.arrays["temperature"],
                "qv": batch.arrays["qv"],
            }
        p_full = batch.arrays["p_full"]
        p_half = batch.arrays["p_half"]
        dz = self._dz(batch)
        rho_full = xp.ascontiguousarray(
            p_full / (xp.float32(DRY_AIR_GAS_CONSTANT) * batch.arrays["virtual_temperature"])
        )
        # w on the half levels; woof.globe.core.gf averages the staggered field
        # to full levels and the kernel forms omeg = -g rho w back from it.
        rho_half = xp.empty_like(omega)
        rho_half[0] = rho_full[0]
        rho_half[-1] = rho_full[-1]
        rho_half[1:-1] = xp.float32(0.5) * (rho_full[:-1] + rho_full[1:])
        w_half = xp.ascontiguousarray(
            -omega / xp.maximum(rho_half * xp.float32(GRAVITY_M_S2), xp.float32(1.0e-12))
        )
        kpbl = pbl_out.get("kpbl")
        if kpbl is None:
            # YSU's own one-based index of the level holding the PBL top
            # (ysu.cu) when the launcher returns it; otherwise the same
            # definition rebuilt from the held pblh: the highest full level
            # whose height above ground lies below it, never below 1.
            height = xp.cumsum(dz, axis=0) - xp.float32(0.5) * dz
            kpbl = xp.maximum(
                xp.sum(height < f["pblh"][None], axis=0), 1
            )
        kpbl = xp.ascontiguousarray(xp.asarray(kpbl, dtype=xp.int32))
        atmosphere = {
            "u": batch.arrays["u"], "v": batch.arrays["v"],
            "temperature": entry["temperature"], "qv": entry["qv"],
            # Cloud condensate: New Tiedtke reads it (its cuinin plume
            # starts from the resolved qc/qi) and arwen-massflux-v1
            # entrains and loads it; Grell-Freitas does not read these
            # two keys.
            "qc": batch.arrays["qc"], "qi": batch.arrays["qi"],
            "pressure": p_full, "exner": batch.arrays["exner"],
            "rho": rho_full, "dz": dz, "p_interface": p_half,
        }
        # gf.cu decides land/water by xland > 1.5 (kernel line 2258); the
        # flag is binarised on the bridge's own >= 1.5 water threshold so
        # the columns GF treats as water are exactly the ones sfclay ran
        # its water branch on and Noah skipped (the _xland contract).
        xland = xp.ascontiguousarray(
            xp.where(
                self._xland(batch) >= xp.float32(1.5),
                xp.float32(2.0), xp.float32(1.0),
            )
        )
        fields = {
            "hfx": f["hfx"], "qfx": f["qfx"], "xland": xland, "kpbl": kpbl,
            "pblh": f["pblh"],
        }
        state = SimpleNamespace(
            p=p_full, w=w_half, ht=batch.arrays["terrain_height_m"],
            qi=batch.arrays["qi"], omega_half=omega,
        )
        driver = SimpleNamespace(
            rthratenlw=f["rad_rthratenlw"], rthratensw=f["rad_rthratensw"],
            gf_rthblten=xp.ascontiguousarray(pbl_out["dtheta"]),
            gf_rqvblten=xp.ascontiguousarray(pbl_out["dqv"]),
            gf_rthdynten=f[CUMULUS_DYNAMICS_THETA_LANE],
            gf_rqvdynten=f[CUMULUS_DYNAMICS_QV_LANE],
            # THE ONE PLACE THE TWO SCHEMES ARE FED DIFFERENTLY.  Both
            # adapters read the same optional per-column lane; the
            # Grell-Freitas slot keeps the scalar dx_m option (its measured
            # baselines and its bit-identity anchor were recorded with
            # it) and the New Tiedtke slot is handed each column's own
            # Gaussian spacing (_cumulus_dx_column).  New Tiedtke's deep
            # closure is scale-aware through scale_fac = f(log(dxref/dx))
            # (ntiedtke.cu nt_scale_factors reads dx[i] per column): on a
            # Gaussian grid the zonal spacing at 60 deg latitude is half
            # the equatorial value, so a scalar would hand a polar column
            # the equator's adjustment time.
            gf_dx_column=(
                self._cumulus_dx_column(batch) if scheme == "ntiedtke" else None
            ),
        )
        return atmosphere, fields, state, driver

    def _cumulus_dx_column(self, batch):
        """Each column's grid spacing sqrt(dx * dy), float32 (ny, nx).

        Built once per runtime from the batch's latitude plane (the grid
        never moves): dx is the zonal spacing R cos(lat) 2 pi / nlon of
        the column's ring, dy the ring's meridional extent -- half the
        distance to each neighbouring ring, and for the two polar rings
        the half-distance to the neighbour plus the reach to the pole.
        The geometric mean is what a scheme that asks for one grid
        length per column should see on an anisotropic cell.  A plane
        whose rows are not latitude rings, or a ring of zero zonal
        extent, is refused by name: the scale factors take log(dxref/dx)
        and a zero spacing would feed them an infinity.
        """
        if self._dx_column is not None:
            return self._dx_column
        xp = batch.xp
        lat = xp.asarray(batch.arrays["latitude_deg"], dtype=xp.float64)
        ny, nx = lat.shape
        rows = lat[:, :1]
        if ny < 2 or not bool(xp.all(xp.abs(lat - rows) < 1.0e-6)):
            raise ValueError(
                "the cumulus per-column spacing is defined on latitude "
                "rings (every column of a row at one latitude, rows "
                "ordered along the first axis) and the batch's latitude "
                "plane is not laid out that way; the native suite's grid "
                "is the Gaussian grid, so this is a malformed exchange"
            )
        lat_rad = xp.deg2rad(rows[:, 0])
        if bool(xp.any(xp.cos(lat_rad) < 1.0e-6)):
            raise ValueError(
                "a latitude ring of the exchange sits at a pole "
                "(|latitude| = 90 deg), where the zonal grid spacing is "
                "zero; the cumulus scale factors take log(dxref/dx) and "
                "would read an infinity there -- a Gaussian grid never "
                "carries a polar ring, so this is a malformed exchange"
            )
        gap = xp.abs(lat_rad[1:] - lat_rad[:-1])
        extent = xp.empty(ny, dtype=xp.float64)
        extent[1:-1] = 0.5 * (gap[:-1] + gap[1:])
        extent[0] = 0.5 * gap[0] + (0.5 * math.pi - float(abs(lat_rad[0])))
        extent[-1] = 0.5 * gap[-1] + (0.5 * math.pi - float(abs(lat_rad[-1])))
        dy = EARTH_RADIUS_M * extent
        dx = EARTH_RADIUS_M * xp.cos(lat_rad) * (2.0 * math.pi / nx)
        spacing = xp.sqrt(dx * dy)
        if not bool(xp.all(spacing > 0.0)):
            raise ValueError(
                "a latitude ring of the exchange has zero grid spacing "
                "(two rows at one latitude); the cumulus scale factors "
                "take log(dxref/dx) and would read an infinity there"
            )
        self._dx_column = xp.ascontiguousarray(
            xp.broadcast_to(spacing[:, None], (ny, nx)).astype(xp.float32)
        )
        return self._dx_column

    def _measure_dynamics_forcing(self, batch, persistent):
        """The dynamics' theta and vapor tendencies since the previous call.

        The runner splits every step as physics (half dt), dynamics (dt),
        physics (half dt), so the state this call enters differs from the
        state the previous call left by exactly what the dynamics (the
        spectral step, the diffusion, the mass fixer and the tracer
        transport) did in between, and the two calls of one step are the
        two sides of one dynamics interval.  When the batch's time_s has
        advanced past the held exit the lanes are (entry - exit) over the
        elapsed time, per level and column; when it has not (the first
        half of a step follows the second half of the previous one at the
        same time_s, with only the positivity repair and the water fixer
        between them) nothing ran that the cumulus schemes should read as
        advection and the lanes of the last interval stand.  The exit
        state and its time_s live in the persistent namespace beside the
        lanes, so a restart measures the same lane the continuous run
        did whatever the call cadence; before any exit is held (a cold
        start) the zero lanes stand.
        """
        held_time = persistent.metadata.get(CUMULUS_EXIT_TIME_KEY)
        if held_time is None:
            return
        elapsed = float(batch.time_s) - float(held_time)
        if elapsed <= 0.0:
            return
        xp = batch.xp
        f = persistent.arrays
        inverse = xp.float32(1.0 / elapsed)
        for lane, name, held in (
            (CUMULUS_DYNAMICS_THETA_LANE, "theta", CUMULUS_EXIT_THETA),
            (CUMULUS_DYNAMICS_QV_LANE, "qv", CUMULUS_EXIT_QV),
        ):
            xp.subtract(batch.arrays[name], f[held], out=f[lane])
            f[lane] *= inverse

    def _hold_exit_state(self, batch, persistent):
        """Keep this call's final theta and vapor with its time_s."""
        f = persistent.arrays
        f[CUMULUS_EXIT_THETA][...] = batch.arrays["theta"]
        f[CUMULUS_EXIT_QV][...] = batch.arrays["qv"]
        persistent.metadata[CUMULUS_EXIT_TIME_KEY] = float(batch.time_s)

    def _cumulus_step(self, batch, persistent, cfg, entry, pbl_out):
        """One cumulus call: rates into the column, rain into the books.

        Grell-Freitas (``cumulus="gf"``), New Tiedtke
        (``cumulus="ntiedtke"``) and arwen-massflux-v1 (``cumulus="own"``)
        share this step: the same column state and forcing lanes go in
        (_cumulus_inputs), the same four rates are integrated the same
        way, the same rain reaches the same accumulators.  New Tiedtke
        and arwen-massflux-v1 additionally return convective momentum
        tendencies, applied once to the A-grid wind on the lane the PBL
        step already uses for YSU's du/dv; Grell-Freitas returns none and
        a result carrying half a pair is refused.  No scheme keeps a held
        state between calls (no NCA hold: RAINCV is a per-call increment
        consumed once), so the checkpointed cumulus state is the
        accumulator and the call count below.

        Water: the reservoir is credited with the column water the call
        removed, MEASURED in the model's metric exactly as the Morrison
        and YSU bookings are (audit 2026-09-01 NB-3), so the suite's
        closure sees the atmosphere and the reservoir move together.  The
        rain that reaches the surface is ONE number: the scheme's RAINCV
        (kg/m2 this call) up to the water the call removed.  It feeds both
        the physics-state RAINC the render tape exports and the land
        bucket Noah is forced with (WRF RAINBL = RAINCV + RAINNCV,
        module_surface_driver.F:1566), so a scheme that reports more rain
        than its rates take out of the column neither hands the land water
        the atmosphere did not lose nor draws it on the maps; what it
        reported beyond that is a diagnostic of the call
        (``cumulus_rain_withheld_kg_m2``).  RAINNC stays the microphysics
        accumulators alone, WRF's own split.  Momentum rates, when the
        scheme carries them, integrate into u and v like the others.
        """
        scheme = self.options.cumulus_scheme
        if scheme is None:
            return False
        if self._cumulus is None:
            # cumulus_column_chunk bounds every scheme's per-pass column
            # packing (woof.globe.core.gf GF_COLUMN_CHUNK; woof.globe.core.ntiedtke
            # caps its SM-derived tile with it), so one option sizes the
            # cumulus workspace whichever scheme fills the slot and the
            # device peak the receipt measures at the allocator holds it.
            module_name, class_name = CUMULUS_SCHEME_MODULES[scheme]
            module = self._module(module_name)
            extra = {}
            if scheme == "gf":
                extra["updraft_only_when_downdraft_dry"] = (
                    self.options.gf_updraft_only_when_downdraft_dry
                )
                extra["resolved_convergence_closure"] = (
                    self.options.gf_resolved_convergence_closure
                )
            self._cumulus = getattr(module, class_name)(
                column_chunk=self.options.cumulus_column_chunk, **extra
            )
        xp = batch.xp
        f = persistent.arrays
        shape = batch.shape
        atmosphere, fields, state, driver = self._cumulus_inputs(
            batch, persistent, entry, pbl_out
        )
        self._cumulus.bind_driver(driver)
        result = self._cumulus(
            atmosphere=atmosphere, fields=fields, state=state, cfg=cfg
        )
        for name in REQUIRED_CUMULUS_RATES:
            if getattr(result, name, None) is None:
                raise ValueError(
                    f"cumulus result is missing {name}; the column would "
                    "receive convective rain with no heating or drying"
                )
        step = xp.float32(batch.dt_s)
        column_before = batch.atmospheric_water_kg_m2()
        for name, target in CUMULUS_RATE_FIELDS:
            rate = getattr(result, name, None)
            if rate is None:
                continue
            if tuple(rate.shape) != shape:
                raise ValueError(
                    f"cumulus {name} shape {tuple(rate.shape)} != {shape}"
                )
            batch.arrays[target] += step * xp.asarray(rate, dtype=xp.float32)
        batch.arrays["temperature"] = xp.ascontiguousarray(
            batch.arrays["theta"] * batch.arrays["exner"]
        )
        # What the scheme's applied rates took out of the column, in the
        # model's metric: the reservoir is credited with it, and it is also
        # the most rain the land surface can receive from this call (below).
        removed = column_before - batch.atmospheric_water_kg_m2()
        self._credit_reservoir(batch, removed)
        momentum = {
            name: getattr(result, name, None) for name, _ in CUMULUS_MOMENTUM_FIELDS
        }
        present = [name for name, rate in momentum.items() if rate is not None]
        if len(present) == 1:
            raise ValueError(
                f"cumulus result carries {present[0]} without its partner; "
                "WRF's convective momentum is a pair (cu_ntiedtke.F90 "
                "lmfdudv guards both) and applying half of it would leave "
                "the wind with a one-component acceleration that no scheme "
                "computed"
            )
        for name, target in CUMULUS_MOMENTUM_FIELDS:
            rate = momentum[name]
            if rate is None:
                continue
            if tuple(rate.shape) != shape:
                raise ValueError(
                    f"cumulus {name} shape {tuple(rate.shape)} != {shape}"
                )
            # The same integration the PBL step applies to YSU's du/dv.
            batch.arrays[target] += step * xp.asarray(rate, dtype=xp.float32)
        rainc = getattr(result, "rainc", None)
        if rainc is not None:
            if tuple(rainc.shape) != batch.surface_shape:
                raise ValueError(
                    f"cumulus rainc shape {tuple(rainc.shape)} != {batch.surface_shape}"
                )
            increment = xp.maximum(xp.asarray(rainc, dtype=xp.float32), 0.0)
            # The rain that reached the surface is the scheme's rain only up
            # to the water its rates removed, and it is ONE number for the
            # land bucket and for RAINC.  Grell-Freitas's deep arm reports
            # more rain than its own tendencies take out of the column
            # (WRF v4.6.1's GFDRV does too: 1.27 times over the 84 raining
            # columns of the oracle fixture), and on the T255 GDAS
            # 2026-09-30 12Z forecast the reported rain was 1.77 times the
            # removed water over the globe and 2.28 times over land in the
            # first day.  Forced with the reported number, Noah's stores
            # gained water that never left the atmosphere and the surface
            # reservoir paid for it (a Texas column lost 143 kg/m2 of
            # reservoir in one 72 h event).  Drawn with it, the maps showed
            # convective rain that never left the air (660 kg/m2 against
            # 288 removed on a Colombian column over 240 h), and the water
            # budget's E = net + P read the same excess as evaporation.
            # What the scheme reported beyond the removed water is kept as
            # this call's diagnostic, never as rain.
            landed = xp.minimum(
                increment,
                xp.maximum(xp.asarray(removed, dtype=xp.float32), xp.float32(0.0)),
            )
            f[CONVECTIVE_RAIN_ACCUMULATOR] += landed
            f["land_rainbl"] += landed
            rain_books = {
                "rain_reported_kg_m2": float(xp.mean(increment, dtype=xp.float64)),
                "rain_withheld_kg_m2": float(xp.mean(increment - landed, dtype=xp.float64)),
                "water_removed_kg_m2": float(xp.mean(removed, dtype=xp.float64)),
            }
        else:
            rain_books = {}
        persistent.metadata["cumulus_updates"] = int(
            persistent.metadata["cumulus_updates"]
        ) + 1
        # The scheme's own grid-mean readings, and the call's rain books
        # (flat column means of this batch, kg/m2 this call; merged across
        # bands by column count as every cumulus_ reading is).
        self._cumulus_diagnostics = {
            **dict(getattr(result, "diagnostics", None) or {}), **rain_books,
        }
        self._cumulus_column_diagnostics = dict(
            getattr(self._cumulus, "last_column_diagnostics", None) or {}
        )
        return True

    def _microphysics_step(self, batch, persistent):
        module = self._module("woof.globe.core.morrison")
        xp = batch.xp
        f = persistent.arrays
        shape = batch.shape
        dz = self._dz(batch)
        rho = xp.empty(shape, dtype=xp.float32)
        column_before = batch.atmospheric_water_kg_m2()
        module.launch_morrison(
            batch.arrays["theta"], batch.arrays["qv"], batch.arrays["qc"],
            batch.arrays["qr"], batch.arrays["qi"], batch.arrays["qs"],
            batch.arrays["qg"], batch.arrays["nc"], batch.arrays["nr"],
            batch.arrays["ni"], batch.arrays["ns"], batch.arrays["ng"],
            rho, batch.arrays["exner"], batch.arrays["p_full"], dz,
            f["rainnc"], f["rainncv"], f["snownc"], f["snowncv"],
            f["graupelnc"], f["graupelncv"], f["sr"], batch.dt_s,
            effc=f["effc"], effr=f["effr"], effi=f["effi"], effs=f["effs"],
            morr_rimed_ice=self.options.morr_rimed_ice,
            _rhoa_scratch=rho,
            _ice_to_snow_scratch=xp.empty(shape, dtype=xp.float32),
        )
        # Morrison follows WRF: rainncv is the TOTAL precipitation reaching the
        # ground and snowncv/graupelncv are the frozen parts of that total.
        # The Arwen Global surface accumulators are disjoint species buckets,
        # so the liquid bucket takes the remainder.  The reservoir is
        # credited with the water that LEFT the atmosphere as measured in
        # the model's metric (specific humidity times dp/g), which is what
        # the dycore's ledger will see: the kernel's own kg/m2 total is
        # formed from its mixing ratios and rho*dz and differs from that by
        # O(q) -- 1-2% of a burst, above the repair gate on a 1 kg/m2 event
        # -- so it feeds the accumulators and the land buckets Noah is
        # forced with, not the closure (audit 2026-09-01 NB-3).  Column
        # interior conversions conserve mixing-ratio water, so fallout is
        # the only change measured here.
        total = xp.maximum(f["rainncv"], 0.0)
        snow = xp.minimum(xp.maximum(f["snowncv"], 0.0), total)
        graupel = xp.minimum(xp.maximum(f["graupelncv"], 0.0), total - snow)
        liquid = total - snow - graupel
        self._credit_reservoir(
            batch, column_before - batch.atmospheric_water_kg_m2()
        )
        batch.surface.accumulated_rain_kg_m2 += liquid
        batch.surface.accumulated_snow_kg_m2 += snow
        batch.surface.accumulated_graupel_kg_m2 += graupel
        f["land_rainbl"] += total
        f["land_snowbl"] += snow
        f["land_graupelbl"] += graupel
        persistent.metadata["microphysics_updates"] = int(
            persistent.metadata["microphysics_updates"]
        ) + 1
        batch.arrays["temperature"] = xp.ascontiguousarray(
            batch.arrays["theta"] * batch.arrays["exner"]
        )

    def run(self, batch, cfg, persistent=None):
        """Run the suite on ``batch``; ``persistent`` is the namespace to
        work on (freshly built from the batch when not supplied, which is
        exactly what a supplied one must be)."""
        batch.validate()
        prof = profiler_of(self)
        # Every per-grid cache below reads the state built from THIS
        # batch's rows (the _BandCache of its band).
        self._band_key = batch.band
        self.last_radiation_bounding_columns = None
        if persistent is None:
            persistent = PersistentNativeState(batch, self.options)
        if self.options.cumulus_enabled:
            self._measure_dynamics_forcing(batch, persistent)
        self._observe("start", batch)
        # Grell-Freitas reads the column as it stood BEFORE this call's
        # radiation and PBL forcing (WRF's t/q), with those two rates
        # handed to it as forcing lanes (_cumulus_inputs); arwen-massflux-v1
        # reads the forced column (entry None).
        entry = None
        if self.options.uses_grell_freitas:
            entry = {
                "temperature": batch.arrays["temperature"].copy(),
                "qv": batch.arrays["qv"].copy(),
            }
        with prof.section("rrtmgp"):
            radiation_due = self._radiation_step(batch, persistent, cfg)
        self._observe("rrtmgp", batch)
        with prof.section("sfclay"):
            self._surface_layer_step(batch, persistent)
        self._observe("sfclay", batch)
        with prof.section("noah"):
            land_due = self._land_step(batch, persistent)
        with prof.section("sfcdiags"):
            self._screen_level_step(batch, persistent)
        self._observe("noah", batch)
        with prof.section("ysu"):
            pbl_out = self._pbl_step(batch, persistent)
        self._observe("ysu", batch)
        # Of the PBL rates only the theta and vapor lanes (and the PBL-top
        # index) feed the cumulus scheme; the applied wind and condensate
        # rates are dead once integrated (memory only: four volumes, 0.76
        # GiB at T533 float32, otherwise held through cumulus and
        # microphysics).
        for name in ("du", "dv", "dqc", "dqi"):
            pbl_out.pop(name, None)
        with prof.section("cumulus"):
            cumulus_active = self._cumulus_step(
                batch, persistent, cfg, entry, pbl_out
            )
        del entry, pbl_out
        self._observe(
            self.options.cumulus_component if cumulus_active else "gf", batch
        )
        with prof.section("morrison"):
            self._microphysics_step(batch, persistent)
        self._observe("morrison", batch)
        if self.options.cumulus_enabled:
            self._hold_exit_state(batch, persistent)
        persistent.metadata["native_calls"] = int(
            persistent.metadata["native_calls"]
        ) + 1
        return persistent, {
            "radiation_due": bool(radiation_due),
            "land_surface_due": bool(land_due),
            "cumulus_active": bool(cumulus_active),
            "native_call_count": int(persistent.metadata["native_calls"]),
            # Sea-ice and land-ice columns whose skin the frozen-surface
            # step integrates in place of the unported WRF schemes.
            "frozen_surface_columns": int(self._frozen_count or 0),
            # Inland lake columns whose skin follows their own surface
            # energy budget (_lake_surface_step).
            "lake_surface_columns": int(self._lake_count or 0),
            # The scheme's own grid-mean readings of this call (arwen-
            # massflux-v1: branch fractions, base mass flux, rain, CAPE);
            # the GF seam carries none.
            **{
                f"cumulus_{name}": float(value)
                for name, value in self._cumulus_diagnostics.items()
            },
        }


__all__ = ["NativePhysicsRuntime"]
