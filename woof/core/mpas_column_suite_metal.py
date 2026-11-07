"""Selectable MPAS column physics on the existing resident Metal seam.

Selector and generation validation is shared with ``mpas_column_suite``.
The regional PhysicsDriver owns MYNN, RUC and GSL; this module supplies their
column geometry, persistent state and phase-one rate contract.
"""
from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np

from woof.core.mpas_column_batch import RESTART_SCHEMA
from woof.core.mpas_column_batch_metal import (
    MetalMpasColumnBatchPhysics, _MetalColumnBatchState)
from woof.core.mpas_column_suite import (
    MpasColumnSuiteTendencies, SUITE_RESTART_SCHEMA, _capturing_momentum_cumulus,
    _initialize_suite_driver, _nominal_znu, orographic_statistic_names,
    resolve_generation, resolve_suite_selection)


class _MetalSuiteColumnState(_MetalColumnBatchState):
    def __init__(self, *args, znu, **kwargs):
        super().__init__(*args, **kwargs)
        cfg, xp = args[0], args[-1]
        self.sina = xp.zeros((cfg.ny, cfg.nx), dtype=np.float32)
        self.cosa = xp.ones((cfg.ny, cfg.nx), dtype=np.float32)
        self.znu = xp.asarray(znu, dtype=np.float32)


class MetalMpasColumnSuitePhysics(MetalMpasColumnBatchPhysics):
    """The selectable suite with Metal state and shared physics dispatch."""

    def __init__(self, *, radiation_scheme, surface_layer_scheme,
                 land_surface_scheme, pbl_scheme, drag_scheme,
                 fractional_seaice=None, orographic_statistics=None,
                 generation=None, cell_area_m2=None, array_module=None,
                 **kwargs):
        resolved = resolve_suite_selection(
            radiation_scheme=radiation_scheme,
            surface_layer_scheme=surface_layer_scheme,
            land_surface_scheme=land_surface_scheme,
            pbl_scheme=pbl_scheme, drag_scheme=drag_scheme)
        self._selection = dict(resolved["selection"])
        self._settings = dict(resolved["config"])
        microphysics = kwargs.get("microphysics_scheme", "wsm6")
        self._generation = resolve_generation(
            generation, microphysics_scheme=microphysics,
            selection=self._selection)
        self._settings.update(self._generation)
        if self._settings.pop("ra_rrtmg_variant") != "legacy":
            raise AssertionError("the suite radiation selector changed")
        self._settings["ra_rrtmg_variant"] = "rrtmg_legacy"
        ncol, nlev = int(kwargs["n_columns"]), int(kwargs["n_levels"])
        n_soil = int(self._settings["num_soil_layers"])
        for name in ("soil_temperature", "soil_moisture"):
            value = kwargs.get(name)
            if value is None or tuple(np.shape(value)) != (n_soil, ncol):
                raise ValueError(f"{name} must have shape {(n_soil, ncol)}")
        threshold = kwargs.pop("xice_threshold", None)
        if land_surface_scheme == "ruc":
            if threshold is not None:
                raise ValueError("RUC takes fractional_seaice, not xice_threshold")
            if fractional_seaice not in (0, 1):
                raise ValueError("RUC requires explicit fractional_seaice 0 or 1")
            self._settings["fractional_seaice"] = int(fractional_seaice)
        else:
            if fractional_seaice is not None:
                raise ValueError("fractional_seaice belongs to RUC")
            threshold = 0.5 if threshold is None else threshold
        needed = orographic_statistic_names(drag_scheme)
        supplied = {} if orographic_statistics is None else dict(orographic_statistics)
        if set(supplied) != set(needed):
            raise ValueError(f"orographic_statistics must carry exactly {list(needed)}")
        self._orographic_statistics = {}
        for name in needed:
            value = np.asarray(supplied[name], dtype=np.float32)
            if value.shape != (ncol,) or not np.all(np.isfinite(value)):
                raise ValueError(f"orographic statistic {name} must be finite {(ncol,)}")
            self._orographic_statistics[name] = value.reshape(1, ncol)
        self._terrain_static = np.asarray(
            kwargs["terrain_height_m"], dtype=np.float32).reshape(1, ncol)
        self._znu = _nominal_znu(kwargs["z_interface_nominal_m"], kwargs["p_top_pa"])
        self._momentum_cumulus = (
            _capturing_momentum_cumulus()
            if kwargs.get("cumulus_scheme") == "ntiedtke" else None)
        if cell_area_m2 is not None:
            area = np.asarray(cell_area_m2, dtype=np.float64)
            if area.shape != (ncol,) or not np.all(np.isfinite(area)) or np.any(area <= 0):
                raise ValueError("cell_area_m2 must be finite and positive per column")
        super().__init__(array_module=array_module, xice_threshold=threshold, **kwargs)
        _seed_ruc_fractional_seaice_optics_metal(
            self._driver, self._cfg, kwargs['latitude_deg'], kwargs['start_time'])
        from woof.config import mynn_mixed_number_species, soil_layer_count
        if soil_layer_count(self._cfg) != n_soil:
            raise ValueError("suite soil geometry differs from the selected land surface")
        self._identity.update(
            schema=SUITE_RESTART_SCHEMA, batch_schema=RESTART_SCHEMA,
            selection=dict(self._selection), n_soil_levels=n_soil,
            fractional_seaice=fractional_seaice, backend="metal")
        self._identity["source_generations"].update({
            name: getattr(self._cfg, name) for name in (
                "bl_mynn_version", "bl_mynn_mixlength", "mynn_sfclay_variant",
                "ruc_soilprop", "ruc_snow", "ruc_irrigation",
                "ruc_qvg_cold_start", "ruc_2m_diagnostic",
                "rrtmg_cloud_optics_form", "rrtmg_aer3_type_source")})
        if self._generation:
            self._identity["generation"] = dict(self._generation)
        mixed = mynn_mixed_number_species(self._cfg)
        if any(name not in self._species for name in mixed):
            raise ValueError("MYNN requested a number species absent from this seam")
        self._scalar_out = {name: self._xp.zeros((nlev, ncol), dtype=np.float32)
                            for name in mixed}
        self.last_scalar_rates = MappingProxyType({})

    def _physics_settings(self):
        return dict(self._settings)

    def _ice_threshold(self, value):
        if self._selection["land_surface_scheme"] == "ruc":
            return None
        return super()._ice_threshold(value)

    def _new_column_state(self, cfg, fnm, fnp, p_top, terrain, xp):
        return _MetalSuiteColumnState(cfg, fnm, fnp, p_top, terrain, xp,
                                     species=self._species, znu=self._znu)

    def _initialize_column_driver(self, state, cfg, **kwargs):
        if self._orographic_statistics:
            kwargs["terrain_drag_static"] = dict(
                self._orographic_statistics, HGT_M=self._terrain_static)
        kwargs["cumulus"] = self._momentum_cumulus
        return _initialize_suite_driver(state, cfg, **kwargs)

    @property
    def selection(self):
        return dict(self._selection)

    @property
    def generation(self):
        return dict(self._generation)

    @property
    def n_soil_levels(self):
        return int(self._identity["n_soil_levels"])

    @property
    def mixed_number_species(self):
        return tuple(self._scalar_out)

    @property
    def last_ruc_census(self):
        value = self._driver.last_ruc_census
        return None if value is None else dict(value)

    def soil_state(self):
        return {name: value.view((value.shape[0], self._ncol)).copy()
                for name in ("tslb", "smois")
                if (value := self._driver.fields.get(name)) is not None}

    def run_phase1(self, **kwargs):
        result = super().run_phase1(**kwargs)
        shape = (self._nz, self._ncol)
        for name in ("du", "dv"):
            self._out[name][...] = self._driver.pbl_raw_rates[name].view(shape)
        if self._momentum_cumulus is not None:
            from woof.metal.bandwidth_family import add_array
            momentum = self._momentum_cumulus.last_momentum
            if momentum is None:
                raise RuntimeError("the selected cumulus returned no momentum rates")
            for name, rate in zip(("du", "dv"), momentum):
                if not add_array(rate.view(shape), self._out[name]):
                    raise NotImplementedError("cumulus momentum has no Metal FP32 layout")
        if self._scalar_out:
            if not result.surface_pbl_ran:
                raise RuntimeError("MYNN number mixing requires PBL on every phase one")
            extra = self._driver.pbl_tendencies.extra_scalars or {}
            if set(extra) != set(self._scalar_out):
                raise RuntimeError("MYNN number-species rate inventory changed")
            for name, target in self._scalar_out.items():
                target[...] = extra[name].view(shape)
            result = MpasColumnSuiteTendencies(
                **{item.name: getattr(result, item.name)
                   for item in dataclasses.fields(result)},
                scalar_rates=MappingProxyType(dict(self._scalar_out)))
        self.last_scalar_rates = getattr(result, "scalar_rates", MappingProxyType({}))
        return result


def run_mpas_column_suite_metal(**kwargs):
    return MetalMpasColumnSuitePhysics(**kwargs)


__all__ = ["MetalMpasColumnSuitePhysics", "run_mpas_column_suite_metal"]


def _seed_ruc_fractional_seaice_optics_metal(driver, cfg, latitude_deg, start_time):
    """Apply the shared column seam's RUC cold-start ice optics."""
    if cfg.sf_surface_physics != 3 or cfg.fractional_seaice != 1:
        return
    from woof.core.landuse import load_landuse_table
    from woof.core.ruc import ruc_ice_category
    from woof.hex.mpasphys.metal_noah_v842 import initialize_landuse
    xp = driver.state._array_module
    fields, params = driver.fields, driver.ruc_params
    table = load_landuse_table(params.dataset_identifier)
    isice = ruc_ice_category(params.vegetation)
    xice = fields['xice'].to_numpy().ravel()
    n = xice.size
    packed = np.zeros((n, 14), np.float32)
    packed[:, 0] = np.asarray(latitude_deg).ravel()
    packed[:, 3] = xice
    packed[:, 13] = params.xice_threshold
    if 'embck' not in fields:
        fields['embck'] = xp.asarray(np.full(fields['xice'].shape, .95, np.float32))
    targets = dict(albbck=4, embck=7, albedo=8, emiss=9)
    for name, column in targets.items():
        packed[:, column] = fields[name].to_numpy().ravel()
    integers = np.zeros((n, 9), np.int32)
    integers[:, 0] = integers[:, 2] = integers[:, 3] = isice
    integers[:, 1] = start_time.timetuple().tm_yday
    integers[:, 4] = xice < np.float32(params.xice_threshold)
    integers[:, 6] = 1
    tables = np.zeros((7, 12, 100), np.float32)
    for season in range(12):
        tables[:, season, :table.lucats] = table.values[min(season, table.luseas-1)].T
    device = xp.asarray(packed)
    status = initialize_landuse(device, xp.asarray(integers), xp.asarray(tables.ravel()))
    if np.any(status.to_numpy()):
        raise ValueError('RUC cold-start sea-ice category is outside the land-use table')
    result = device.to_numpy()
    for name, column in targets.items():
        fields[name][...] = xp.asarray(np.ascontiguousarray(result[:, column].reshape(fields[name].shape)))
