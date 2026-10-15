"""Shared prepared-real physics initialization for GPU forecast runners.

The historical public entry point remains :func:`initialize_hrrr_physics`,
but the actual physics setup consumes the source-neutral Noah surface contract
stored by ``gpuwm-prepared-real-cache-v1``.  GFS and ERA5 cache restores use
that same path instead of re-decoding source-specific soil fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import numpy as np

from woof.config import soil_layer_count
from woof.core.noah import noah_initial_snow_albedo
from woof.ingest.hrrr_surface import surface_fields_to_device
from woof.ingest.vegetation import initial_vegetation_fraction


_CANONICAL_SURFACE_FIELDS = frozenset({
    "TSK", "TSLB", "SMOIS", "SH2O", "TMN", "SEAICE", "XLAND",
    "LANDMASK", "SNOW", "SNOWH",
})

#: The native soil pair a met snapshot carries when nothing has solved
#: the surface yet.  Named here because the prepared-cache writer DROPS
#: it precisely when a solved surface rides along
#: (:func:`woof.ingest.prepared_cache._prepared_met_names`), so these
#: two names are the whole difference between the two roads below.
_NATIVE_SOIL_FIELDS = ("SOILT", "SOILW")


def _validate_prepared_surface(surface, cfg) -> Mapping[str, object]:
    try:
        fields = surface.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics surface must expose a fields mapping") from exc
    if not isinstance(fields, Mapping):
        raise TypeError("prepared physics surface fields must be a mapping")
    if set(fields) != _CANONICAL_SURFACE_FIELDS:
        raise ValueError(
            "prepared physics surface inventory differs from the canonical "
            f"Noah contract: expected {sorted(_CANONICAL_SURFACE_FIELDS)}, "
            f"got {sorted(fields)}")
    horizontal_shape = (int(cfg.ny), int(cfg.nx))
    # The prepared cache stores whatever geometry the run that wrote it used,
    # so the shape gate is the SELECTED scheme's layer count, not a literal.
    # A four-layer cache restored under a nine-layer scheme is refused here
    # by shape rather than broadcast into place downstream.
    soil_shape = (soil_layer_count(cfg), *horizontal_shape)
    host_fields = {}
    for name in sorted(_CANONICAL_SURFACE_FIELDS):
        value = fields[name]
        if hasattr(value, "get"):
            value = value.get()
        array = np.asarray(value)
        host_fields[name] = array
        expected = soil_shape if name in {"TSLB", "SMOIS", "SH2O"} \
            else horizontal_shape
        if array.shape != expected:
            raise ValueError(
                f"prepared surface {name} has shape {array.shape}, "
                f"expected {expected}")
        if (array.dtype.hasobject
                or not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()):
            raise ValueError(
                f"prepared surface {name} must be finite numeric data")
    if not np.array_equal(
            host_fields["XLAND"],
            np.where(host_fields["LANDMASK"] >= 0.5, 1.0, 2.0)):
        raise ValueError(
            "prepared surface XLAND differs from its canonical LANDMASK")
    return fields


def _validate_prepared_near_surface(result, met, cfg) -> Mapping[str, np.ndarray]:
    """Validate exact, physical near-surface arrays before GPU allocation."""

    try:
        met_fields = met.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics meteorology must expose a fields mapping") from exc
    if not isinstance(met_fields, Mapping):
        raise TypeError("prepared physics meteorology fields must be a mapping")
    ny, nx = int(cfg.ny), int(cfg.nx)
    specs = {
        "surface_pressure": (result.surface_pressure, (ny, nx), 1_000.0, 120_000.0),
        "surface_qv": (result.surface_qv, (ny, nx), 0.0, 0.2),
        "T2": (met_fields["T2"], (ny, nx), 100.0, 400.0),
        "SKINTEMP": (met_fields["SKINTEMP"], (ny, nx), 100.0, 400.0),
        "U10": (met_fields["U10"], (ny, nx + 1), -250.0, 250.0),
        "V10": (met_fields["V10"], (ny + 1, nx), -250.0, 250.0),
        "LANDSEA": (met_fields["LANDSEA"], (ny, nx), 0.0, 1.0),
    }
    validated = {}
    for name, (value, expected_shape, lower, upper) in specs.items():
        if hasattr(value, "get"):
            value = value.get()
        array = np.asarray(value)
        if array.shape != expected_shape:
            raise ValueError(
                f"prepared near-surface {name} has shape {array.shape}, "
                f"expected {expected_shape}")
        if (array.dtype.hasobject
                or not np.issubdtype(array.dtype, np.number)
                or not np.isfinite(array).all()):
            raise ValueError(
                f"prepared near-surface {name} must be finite numeric data")
        if array.min() < lower or array.max() > upper:
            # Name the numbers.  This refusal used to say only which field
            # and which range, and a field report ("surface_qv is outside
            # the physical range 0.0..0.2") cost a full re-run of a
            # two-stage preparation to learn that the excursion was two
            # cells at -1.8e-05 rather than, say, a unit error.  The
            # observed extremes and the offending-cell count are already
            # in hand here and decide which of those it is on sight.
            outside = int(np.count_nonzero(
                (array < lower) | (array > upper)))
            raise ValueError(
                f"prepared near-surface {name} is outside the physical "
                f"range {lower}..{upper}: {outside} of {array.size} "
                f"cell(s) are out, observed range "
                f"{float(array.min()):.6g}..{float(array.max()):.6g}")
        validated[name] = array
    if not np.isin(validated["LANDSEA"], (0.0, 1.0)).all():
        raise ValueError("prepared near-surface LANDSEA must be exactly binary")
    return MappingProxyType(validated)


def initialize_prepared_physics(
        result, cfg, met, surface, static, landuse_attrs, grid, valid_time, *,
        center_lat=None, constant_glw_wm2=None, fractional_seaice=True,
        isoilwater=14, p_top=None, column_chunk=None,
        trace_gas_overrides=None, ozone_parent=None, cam_ozone=None,
        simulation_start_time=None):
    """Attach physics to one restored source-neutral prepared initial state.

    ``surface`` is the canonical Noah inventory persisted with direct GFS and
    ERA5 caches.  Every field is validated before device allocation.  The
    actual land-use, diagnostics, physics-driver, and near-surface setup is the
    same implementation used by the native HRRR runner.

    ``valid_time`` dates this domain's analysis and land climatologies.
    ``simulation_start_time`` is the radiation calendar's origin when a
    child starts later: model elapsed seconds are measured from the root
    experiment, so adding them to the child's start would count its delay
    twice. Omission retains the ordinary single-epoch behavior.

    ``constant_glw_wm2`` is the DECLARED constant downward longwave, for a
    suite that runs a land-surface scheme with ``ra_lw_physics = 0``.  The
    caller reads it off the experiment's acknowledgements
    (:func:`woof.runtime.declared_constant_glw`); ``None`` -- the normal
    answer -- lets the attached longwave scheme own the field, and makes
    :func:`~woof.core.physics.initialize_physics` refuse an undeclared
    suite that has no longwave scheme at all.

    Radiation uses the normal selector-driven constructor, including legacy
    RRTMG and its CAM ozone routing. The no-CuPy fused RRTMGP adapter remains a
    specialized construction for the modern RRTMGP Metal path.
    """

    from woof.core.diagnostics import update_diagnostics
    from woof.core.landuse import (initialize_landuse,
                                    usemonalb_landuse_inputs)
    from woof.core.physics import initialize_physics
    from woof.ingest.ruc_mosaic import ruc_mosaic_physics_inputs
    from woof.ingest.lake_physics import lake_physics_inputs
    from woof.static.build import monthly_interp_to_date

    # The public prepared-cache road hands this function the restored
    # source-neutral Noah inventory directly (unlike fresh ingestion, which
    # passes through resolve_prepared_noah_surface first). Adapt its four
    # cached soil layers to the selected RUC geometry before the strict
    # selected-scheme shape gate below.
    surface = _resolve_cached_ruc_surface(surface, cfg)
    fields = _validate_prepared_surface(surface, cfg)
    near_surface_host = _validate_prepared_near_surface(result, met, cfg)
    required_attrs = {"MMINLU", "ISWATER", "ISLAKE", "ISICE"}
    admitted_attrs = (required_attrs | {"ISURBAN"}
                      if getattr(cfg, "sf_surface_mosaic", 0) == 1 else required_attrs)
    if not isinstance(landuse_attrs, Mapping) \
            or not required_attrs <= set(landuse_attrs) <= admitted_attrs:
        raise ValueError(
            "prepared physics land-use identity must contain exactly "
            f"{sorted(required_attrs)}")
    state = result.state
    metal_state = bool(getattr(state, "_metal_setup_state", False))
    metal_direct_radiation = False
    if metal_state:
        from woof.config import radiation_scheme_ids
        from woof.physics_compat import (
            RRTMG_VARIANT_RTE_RRTMGP,
            metal_rrtmg_compatibility_supported,
            rrtmg_variant)

        radiation_pair = radiation_scheme_ids(cfg)
        metal_direct_radiation = (
            radiation_pair in ((4, 0), (4, 4))
            and rrtmg_variant(cfg) == RRTMG_VARIANT_RTE_RRTMGP
            and metal_rrtmg_compatibility_supported(cfg)
            and int(getattr(cfg, "icloud", 1)) == 1)
        # These two checks are specific to the fused no-CuPy RRTMGP
        # constructor below. Other selectors, including legacy RRTMG with
        # its CAM ozone routing, are constructed by make_radiation and are
        # admitted centrally by PhysicsDriver.
        if metal_direct_radiation:
            p_top_value = getattr(state, "p_top", None)
            if state.qv is None:
                raise ValueError(
                    "fused Metal RRTMGP construction requires state.qv")
            if (p_top_value is None or not np.isfinite(p_top_value)
                    or float(p_top_value) <= 0.0):
                raise ValueError(
                    "fused Metal RRTMGP construction requires finite positive p_top")
        xp = state._array_module
    else:
        import cupy as cp
        xp = cp
    update_diagnostics(state, cfg.hypsometric_opt)
    landuse = initialize_landuse(
        static["LU_INDEX"], soil_type=static["SCT_DOM"],
        urban_legend=int(getattr(cfg, "sf_urban_physics", 0)) > 0,
        landmask=static["LANDMASK"], snow=fields["SNOW"],
        xice=fields["SEAICE"], valid_time=valid_time,
        cen_lat=float(
            getattr(grid, "cen_lat", grid.ref_lat)
            if center_lat is None else center_lat),
        mminlu=str(landuse_attrs["MMINLU"]),
        iswater=int(landuse_attrs["ISWATER"]),
        islake=int(landuse_attrs["ISLAKE"]),
        isice=int(landuse_attrs["ISICE"]), fractional_seaice=fractional_seaice,
        isoilwater=int(isoilwater),
        # real.exe's landmask/soil-category reconciliation decides a
        # disagreeing column from its soil temperature, then its SST.
        soil_temperature=fields["TSLB"], sst=fields.get("SST"),
        **usemonalb_landuse_inputs(cfg, static, valid_time))
    vegfra = initial_vegetation_fraction(met, static, valid_time)
    from woof.core.landuse import surface_leaf_area
    lai = surface_leaf_area(cfg, static["LAI12M"], valid_time)
    lat, lon = grid.latlon_mass()
    from woof.core.radiation_composition import make_radiation
    radiation_origin = (valid_time if simulation_start_time is None
                        else simulation_start_time)
    if metal_state and metal_direct_radiation:
        from woof.core.radiation_composition import trace_gas_override_status
        from woof.core.rrtmgp import RRTMGPRadiation, coefficient_gas_names
        from woof.core.trace_gases import trace_gas_subset

        # Match make_radiation's selected-spectrum validation and filtering,
        # while constructing the direct adapter required by the no-CuPy call.
        trace_gas_override_status(cfg, trace_gas_overrides)
        selected_gases = trace_gas_overrides
        if trace_gas_overrides:
            selected_gases = trace_gas_subset(
                trace_gas_overrides, coefficient_gas_names("lw")) or None
        options = {
            "validation_mode": "fused",
            "longwave": radiation_pair[0] == 4,
            "shortwave": radiation_pair[1] == 4,
            "trace_gas_overrides": selected_gases,
        }
        if column_chunk is not None:
            options["column_chunk"] = column_chunk
        radiation = RRTMGPRadiation(
            radiation_origin, np.asarray(lat), np.asarray(lon), **options)
    else:
        radiation = make_radiation(
            cfg, radiation_origin, lat, lon, p_top=p_top,
            column_chunk=column_chunk,
            trace_gas_overrides=trace_gas_overrides,
            ozone_parent=ozone_parent)
    driver = initialize_physics(
        state, cfg, landuse=landuse, radiation=radiation, tsk=fields["TSK"],
        soil_temperature=fields["TSLB"],
        soil_moisture=fields["SMOIS"],
        liquid_moisture=fields["SH2O"],
        ivgtyp=static["LU_INDEX"], isltyp=static["SCT_DOM"],
        vegfra=vegfra, tmn=fields["TMN"], xice=fields["SEAICE"],
        snow=fields["SNOW"], snow_depth=fields["SNOWH"],
        sst=fields.get("SST", fields["TSK"]),
        glw=constant_glw_wm2,
        radiation_start_time=radiation_origin, radiation_latitude=lat,
        radiation_longitude=lon,
        landuse_dataset=str(landuse_attrs["MMINLU"]),
        # The prepared statics carry the sub-grid orographic statistics when
        # the preparation ran with topo_wind / gwd_opt on; read only then.
        terrain_drag_static=static,
        **lake_physics_inputs(cfg, static),
        **ruc_mosaic_physics_inputs(
            cfg, static, landuse_attrs=landuse_attrs, xice=fields["SEAICE"],
            fractional_seaice=fractional_seaice),
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))
    from woof.core.landuse import surface_snow_albedo
    if metal_state:
        # These land-use/vegetation diagnostics are static prepared inputs
        # consumed by Noah and the RRTMGP surface boundary. Upload them once
        # during physics attachment, before the normal first forecast step.
        driver.fields["t2"][...] = xp.asarray(
            near_surface_host["T2"], dtype=np.float32)
        driver.fields["q2"][...] = xp.asarray(
            near_surface_host["surface_qv"], dtype=np.float32)
        driver.fields["psfc"][...] = xp.asarray(
            near_surface_host["surface_pressure"], dtype=np.float32)
        driver.fields["th2"][...] = xp.asarray(
            near_surface_host["T2"]
            * (np.float32(100000.0) / near_surface_host["surface_pressure"])
            ** np.float32(287.0 / 1004.0), dtype=np.float32)
        driver.fields["u10"][...] = xp.asarray(
            0.5 * (near_surface_host["U10"][:, :-1]
                   + near_surface_host["U10"][:, 1:]), dtype=np.float32)
        driver.fields["v10"][...] = xp.asarray(
            0.5 * (near_surface_host["V10"][:-1]
                   + near_surface_host["V10"][1:]), dtype=np.float32)
        # Each land model carries its own subset of these fields.
        if "snoalb" in driver.fields:
            driver.fields["snoalb"][...] = xp.asarray(
                surface_snow_albedo(cfg, static, driver.noah_params),
                dtype=np.float32)
        if "lai" in driver.fields:
            driver.fields["lai"][...] = xp.asarray(lai, dtype=np.float32)
        if "shdmin" in driver.fields:
            driver.fields["shdmin"][...] = xp.asarray(
                100.0 * static["GREENFRAC"].min(axis=0), dtype=np.float32)
        if "shdmax" in driver.fields:
            driver.fields["shdmax"][...] = xp.asarray(
                100.0 * static["GREENFRAC"].max(axis=0), dtype=np.float32)
        return driver
    driver.fields["snoalb"][...] = cp.asarray(
        surface_snow_albedo(cfg, static, driver.noah_params),
        dtype=cp.float32)
    driver.fields["lai"][...] = cp.asarray(lai, dtype=cp.float32)
    driver.fields["shdmin"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].min(axis=0), dtype=cp.float32)
    driver.fields["shdmax"][...] = cp.asarray(
        100.0 * static["GREENFRAC"].max(axis=0), dtype=cp.float32)
    driver.fields["psfc"][...] = cp.asarray(
        near_surface_host["surface_pressure"], dtype=cp.float32)
    near_surface = surface_fields_to_device(
        SimpleNamespace(fields=near_surface_host), cp)
    driver.fields["t2"][...] = near_surface["T2"]
    driver.fields["q2"][...] = cp.asarray(
        near_surface_host["surface_qv"], dtype=cp.float32)
    driver.fields["th2"][...] = (
        driver.fields["t2"]
        * (cp.float32(100000.0) / driver.fields["psfc"])
        ** cp.float32(287.0 / 1004.0))
    driver.fields["u10"][...] = 0.5 * (
        near_surface["U10"][:, :-1] + near_surface["U10"][:, 1:])
    driver.fields["v10"][...] = 0.5 * (
        near_surface["V10"][:-1] + near_surface["V10"][1:])
    if getattr(cfg, "sf_surface_mosaic", 0) == 1:
        from woof.core.noah_mosaic_door import attach_noah_mosaic_to_driver
        attach_noah_mosaic_to_driver(
            driver, cfg, landusef=static.get("LANDUSEF"), processed=False,
            landuse_attrs=landuse_attrs, landmask=static["LANDMASK"],
            fractional_seaice=fractional_seaice)
    return driver


def _resolve_cached_ruc_surface(surface, cfg):
    """Rebuild RUC soil levels from the four-layer surface a cache stores.

    Prepared caches persist the source-neutral Noah surface inventory, whose
    soil arrays are on Noah's four layer midpoints.  A cache restore can then
    select RUC (six or nine levels) even though the cache was prepared before
    that LSM choice.  The fresh-source route reaches the same target through
    ``preprocess_land_surface_soil``; here we apply its shared WRF RUC remap
    to the cached four-layer samples instead of passing them to RUC's
    layer-count allocation unchanged.
    """

    if int(cfg.sf_surface_physics) != 3:
        return surface

    fields = surface.fields
    temperature = np.asarray(fields["TSLB"])
    moisture = np.asarray(fields["SMOIS"])
    target_layers = int(soil_layer_count(cfg))
    expected_shape = (target_layers, int(cfg.ny), int(cfg.nx))
    if temperature.shape == expected_shape and moisture.shape == expected_shape:
        return surface
    if (temperature.ndim != 3 or moisture.ndim != 3
            or temperature.shape[0:1] != (4,)
            or moisture.shape != temperature.shape
            or temperature.shape[1:] != expected_shape[1:]
            or target_layers not in (6, 9)):
        raise ValueError(
            "prepared RUC surface must carry either four cached Noah soil "
            f"layers or {target_layers} RUC soil levels; got temperature "
            f"{temperature.shape} and moisture {moisture.shape}")

    # The canonical cached surface is already resolved at Noah's four layer
    # midpoints.  Feed those samples through the same WRF init_soil_3_real
    # interpolation used by the fresh-source RUC path, with the cached TSK
    # and TMN as the surface and deep anchors.
    from woof.ingest.ruc_soil import remap_soil_to_ruc_levels
    from woof.ingest.soil import NOAH_LAYER_MIDPOINTS_M

    source_depths_cm = np.rint(
        np.asarray(NOAH_LAYER_MIDPOINTS_M, dtype=np.float64) * 100.0
    ).astype(np.int64)
    remapped = remap_soil_to_ruc_levels(
        source_temperature=temperature,
        source_moisture=moisture,
        source_levels_cm=source_depths_cm,
        source_geometry="layers",
        skin_temperature=fields["TSK"],
        deep_temperature=fields["TMN"],
        landmask=fields["LANDMASK"],
        num_soil_layers=target_layers,
        moisture_adjustment=False,
    )
    adapted = dict(fields)
    adapted["TSLB"] = remapped.soil_temperature
    adapted["SMOIS"] = remapped.soil_moisture
    # RUC derives its liquid phase during ruc_cold_start; the canonical
    # prepared inventory nevertheless requires a same-shaped SH2O field.
    adapted["SH2O"] = np.array(remapped.soil_moisture, copy=True)
    return SimpleNamespace(fields=MappingProxyType(adapted))


def resolve_prepared_noah_surface(met, cfg, static, *, surface=None, soil_mesh=None):
    """The canonical Noah surface this state initializes its land model from.

    Two roads arrive here holding the same forecast, and they hold
    DIFFERENT halves of the soil statement.

    The FRESH road has just decoded the source: its met snapshot carries
    the native ``SOILT``/``SOILW`` node column and nothing has solved a
    surface yet, so the surface is derived here
    (``preprocess_land_surface_soil`` -> ``canonical_noah_surface``).

    The RESTORE road holds a prepared cache, and the solved surface is
    exactly what that cache persists -- ``surface/TSLB``, ``SMOIS``,
    ``SH2O`` and the rest of the canonical Noah inventory.  Because the
    surface is stored, the writer deliberately DROPS the native pair from
    the met contract beside it
    (:func:`woof.ingest.prepared_cache._prepared_met_names`: the pair is
    required only when ``surface is None``).  So a restore has the
    answer and not the ingredients.

    Re-deriving on the restore road is therefore not a slow path, it is a
    crash: the soil router, finding no native pair and no mapped or GFS
    markers, falls through to its ERA5-named arm and dies on ``missing
    soil input field(s): ['ST000007', ...]`` -- four frames deep, on a
    cache that was complete.  MEASURED 2026-08-18 on the Linux shakeout's
    prepared-cache restore arm, where the solved soil was sitting in
    ``surface/`` the whole time.

    ``surface`` is therefore consumed when the caller has one and derived
    only when it does not.  A state holding NEITHER is refused here by
    name rather than downstream by KeyError.
    """

    from types import MappingProxyType, SimpleNamespace

    if surface is not None:
        return _resolve_cached_ruc_surface(surface, cfg)

    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from woof.native_wrf_contract import canonical_noah_surface

    try:
        fields = met.fields
    except AttributeError as exc:
        raise TypeError(
            "prepared physics meteorology must expose a fields mapping"
        ) from exc
    absent = [name for name in _NATIVE_SOIL_FIELDS if name not in fields]
    if absent:
        raise ValueError(
            "this prepared state carries no solved Noah surface and no "
            f"native soil to derive one from ({', '.join(absent)} absent "
            "from its met fields), so the land model has no soil state to "
            "start from and the forecast would run on fabricated ground.\n"
            "  A prepared cache written by this release stores the solved "
            "surface (surface/TSLB, SMOIS, SH2O and the rest of the "
            "canonical Noah inventory) and drops the native SOILT/SOILW "
            "pair, so a restore must pass restore_prepared_cache(...)"
            ".surface through to this call rather than re-deriving.\n"
            "  A cache carrying neither is incomplete and must be prepared "
            "again; nothing here can reconstruct soil that was never "
            "written.")
    from woof.core.landuse import (
        ruc_fractional_seaice as _ruc_fractional_seaice)
    soil = preprocess_land_surface_soil(
        fields, sf_surface_physics=int(cfg.sf_surface_physics),
        # real.exe's adjust_for_seaice_pre/post keep the fraction under
        # fractional_seaice = 1 (threshold 0.02) and snap to 0/1 at 0.5
        # otherwise (module_soil_pre.F:216-219, :337-343, :392-393 of the HRRR
        # v4.1.21 fork).
        fractional_seaice=_ruc_fractional_seaice(cfg),
        # The RESOLVED count, the same number :func:`_soil_shape` above
        # allocates from.  Left off, RUC's soil ingest took its own
        # nine-level default, so a six-level config prepared a nine-level
        # surface and met a six-level allocation -- a shape error four
        # frames down instead of a forecast, which is the difference
        # between "six levels is unverified" and "six levels is
        # unreachable on this route".
        num_soil_layers=soil_layer_count(cfg),
        soil_type=static["SCT_DOM"],
        deep_soil_temperature=static["SOILTEMP"], soil_mesh=soil_mesh,
        water_temperature=getattr(met, "water_temperature", None),
        water_temperature_policy=(getattr(met, "water_temperature_receipt", None)
                                  or {}).get("policy"))
    return SimpleNamespace(fields=MappingProxyType(
        canonical_noah_surface(soil)),
        soil_texture_downscale=getattr(soil, "soil_texture_downscale", None),
        soil_temperature_repair=getattr(soil, "soil_temperature_repair", None))


_AUTO_SOIL_MESH = object()


def initialize_hrrr_physics(
        result, cfg, met, static, attrs, grid, valid_time, *,
        constant_glw_wm2=None, surface=None, soil_mesh=_AUTO_SOIL_MESH,
        p_top=None, column_chunk=None, trace_gas_overrides=None, ozone_parent=None, cam_ozone=None):
    """Initialize physics, accepting either host or device ingestion fields.

    ``surface`` is the solved Noah surface when the caller already holds
    one -- what :func:`woof.ingest.prepared_cache.restore_prepared_cache`
    hands back.  Omitted, the surface is derived from the met snapshot's
    native soil, which is what a freshly decoded state carries.  See
    :func:`resolve_prepared_noah_surface` for why the two roads differ.
    Omitting ``soil_mesh`` keeps automatic planning; explicitly passing
    ``None`` preserves the caller's decision to disable soil downscaling.
    """

    if soil_mesh is _AUTO_SOIL_MESH:
        soil_mesh = None
        if surface is None:
            from woof.ingest.hrrr import hrrr_source_grid
            from woof.ingest.soil_downscale import soil_mesh_plan_from_case
            soil_mesh = soil_mesh_plan_from_case(
                None, grid, source_grid=hrrr_source_grid())
    surface = resolve_prepared_noah_surface(
        met, cfg, static, surface=surface, soil_mesh=soil_mesh)
    landuse_attrs = {
        name: attrs[name] for name in ("MMINLU", "ISWATER", "ISLAKE", "ISICE")
    }
    return initialize_prepared_physics(
        result, cfg, met, surface, static, landuse_attrs, grid, valid_time,
        center_lat=attrs["CEN_LAT"], constant_glw_wm2=constant_glw_wm2,
        p_top=p_top, column_chunk=column_chunk, trace_gas_overrides=trace_gas_overrides,
        ozone_parent=ozone_parent,
        **({"cam_ozone": cam_ozone} if cam_ozone is not None else {}))


__all__ = ["initialize_hrrr_physics", "initialize_prepared_physics",
           "resolve_prepared_noah_surface", "initial_vegetation_fraction"]
