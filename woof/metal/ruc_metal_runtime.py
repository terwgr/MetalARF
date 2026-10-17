"""Public Metal RUC driver orchestration over the packed six-stage chain."""

from __future__ import annotations

from math import prod
from weakref import WeakKeyDictionary

import numpy as np

from woof.core import ruc
from woof.core.ruc_runtime import (
    RUC_PROFILE_BINDING,
    RUC_STATE_BINDING,
    sfcdiags_exner_powers,
)
from woof.core.ruc_sfctmp_layout import _SFCTMP_CHECKS

from .array import MetalArray
from .ruc_driver_family import dispatch_full_chain, scatter_committed
from .ruc_metal_workspace import RucMetalWorkspace


_EXTRAS = (
    "psfc", "chs2", "cqs2", "cpm", "qgh", "tsk_save", "tsk_sea",
    "flhc_sea", "flqc_sea", "cpm_sea", "cqs2_sea", "chs2_sea",
    "chs_sea", "qsfc_sea", "qgh_sea", "hfx_sea", "qfx_sea", "lh_sea",
)
_LOCAL_OUTPUTS = ("infiltr", "smelt", "runoff1", "runoff2")
_DIAGNOSTICS = ("t2", "th2", "q2")
_INPUTS = (
    ruc.RUC_DRIVER_COLUMN_STATE + ruc.RUC_DRIVER_COLUMN_FORCING
    + ruc.RUC_DRIVER_ARW_FORCING + _EXTRAS + ruc.RUC_DRIVER_PROFILE_STATE
)
_FINITE_ADMISSIONS = (
    ruc.RUC_DRIVER_PROFILE_STATE + ruc.RUC_DRIVER_COLUMN_STATE
    + ruc.RUC_DRIVER_COLUMN_FORCING + ruc.RUC_DRIVER_ARW_FORCING
)
_OUTPUT_ADMISSIONS = (
    ruc.RUC_DRIVER_COLUMN_STATE + ruc.RUC_DRIVER_PROFILE_STATE
    + _LOCAL_OUTPUTS
)
_WORKSPACES = WeakKeyDictionary()
_TABLE_NAMES = (
    "ifortbl", "z0tbl", "lemitbl", "pctbl", "laitbl", "bb", "drysmc",
    "hc", "maxsmc", "refsmc", "satpsi", "satdk", "wltsmc", "qtz",
    "rstbl", "rgltbl", "tbq", "zshalf", "landusef", "soilctop",
)
# `forms` bits shared by the SFCTMP helpers and the driver prologue
# (ruc_sfctmp_helpers.metalh); CUDA compiles these as module defines.
_FORM_SOILPROP_V461 = 1
_FORM_SNOW_V461 = 2


def _first_level(value, name: str) -> MetalArray:
    if not isinstance(value, MetalArray) or value.ndim < 2:
        raise TypeError(f"RUC Metal atmosphere {name} must be a level-major MetalArray")
    n = prod(value.shape[1:])
    return MetalArray(value.buffer, (n,), value.dtype, byte_offset=value.byte_offset)


def _flat(value) -> MetalArray:
    if not isinstance(value, MetalArray) or not value.flags.c_contiguous:
        raise TypeError("RUC Metal mosaic fractions must be contiguous MetalArrays")
    return MetalArray(value.buffer, (value.size,), value.dtype,
                      byte_offset=value.byte_offset)


def _overlaps(left: MetalArray, right: MetalArray) -> bool:
    return (left.buffer is right.buffer
            and left.byte_offset < right.byte_offset + right.nbytes
            and right.byte_offset < left.byte_offset + left.nbytes)


def _parameter_tables(params, landusef=None, soilctop=None):
    vegetation = params.bundle.vegetation_for(params.dataset_identifier)
    soil = np.asarray([row.values for row in params.bundle.soil.rows], dtype=np.float32)
    tables = {
        "ifortbl": np.asarray([row.ifor for row in vegetation.rows], dtype=np.int32),
        "z0tbl": np.asarray([row.z0 for row in vegetation.rows], dtype=np.float32),
        "lemitbl": np.asarray([row.lemi for row in vegetation.rows], dtype=np.float32),
        "pctbl": np.asarray([row.pc for row in vegetation.rows], dtype=np.float32),
        "laitbl": np.asarray([row.lai for row in vegetation.rows], dtype=np.float32),
        "bb": soil[:, 0], "drysmc": soil[:, 1], "hc": soil[:, 2],
        "maxsmc": soil[:, 3], "refsmc": soil[:, 4], "satpsi": soil[:, 5],
        "satdk": soil[:, 6], "wltsmc": soil[:, 8], "qtz": soil[:, 9],
        "rstbl": np.asarray([row.rs for row in vegetation.rows], dtype=np.float32),
        "rgltbl": np.asarray([row.rgl for row in vegetation.rows], dtype=np.float32),
        "tbq": np.asarray(ruc.ruc_saturation_table(), dtype=np.float32),
        "zshalf": np.asarray(ruc.ruc_zshalf(params.zs), dtype=np.float32),
        # Category-major fractions (nlcat*n, nscat*n); empty when the run
        # reads no mosaic. They change with the run, not the tables, but
        # ride the same per-step tail upload.
        "landusef": (_flat(landusef) if landusef is not None
                     else np.zeros((0,), dtype=np.float32)),
        "soilctop": (_flat(soilctop) if soilctop is not None
                     else np.zeros((0,), dtype=np.float32)),
    }
    if tuple(tables) != _TABLE_NAMES:
        raise RuntimeError("RUC Metal table order differs from the packed ABI")
    return vegetation, tables


def _input_values(fields, atmosphere, precipitation, *, frzfrac_mode: bool):
    values = {
        argument: fields[name]
        for name, argument in RUC_STATE_BINDING.items()
    }
    values.update({
        argument: fields[name]
        for name, argument in RUC_PROFILE_BINDING.items()
    })
    values.update({
        "z3d": _first_level(atmosphere["dz"], "dz"),
        "p8w": _first_level(atmosphere["pressure"], "pressure"),
        "t3d": _first_level(atmosphere["temperature"], "temperature"),
        "qv3d": _first_level(atmosphere["qv"], "qv"),
        "qc3d": _first_level(atmosphere["qc"], "qc"),
        "rho3d": _first_level(atmosphere["rho"], "rho"),
        "tbot": fields["tmn"],
        "rainncv": precipitation.rain_nonconvective,
        "snowncv": precipitation.snow_nonconvective,
        "graupelncv": precipitation.graupel_nonconvective,
        # LSMRUC receives these WRF categorical inputs outside the
        # source-generated float field lists. The packed Metal prologue
        # validates and consumes their dedicated integer rows.
        "ivgtyp": fields["ivgtyp"],
        "isltyp": fields["isltyp"],
    })
    if not frzfrac_mode:
        values["frzfrac"] = fields["sr"]
    for name in _INPUTS:
        if name not in values and name != "frzfrac":
            values[name] = fields[name]
    if frzfrac_mode:
        values.pop("frzfrac", None)
    return values


def _outputs(fields):
    targets = {
        argument: fields[name]
        for name, argument in RUC_STATE_BINDING.items()
    }
    targets.update({
        argument: fields[name]
        for name, argument in RUC_PROFILE_BINDING.items()
    })
    targets.update({name: fields[name] for name in _EXTRAS})
    targets.update({name: fields[name] for name in ("albbck", "chs", "flhc", "flqc")})
    targets.update({name: fields["ruc_" + name] for name in _LOCAL_OUTPUTS})
    targets.update({name: fields[name] for name in _DIAGNOSTICS})
    return targets


def _flag_error(flag_slab, params):
    words = flag_slab.to_numpy()
    driver = words[:20].view(np.uint32)
    sfctmp = words[20:]
    if not (np.any(driver[:36]) or np.any(sfctmp)):
        return driver[36:40].copy()

    for bit, name in enumerate(_FINITE_ADMISSIONS):
        if int(driver[bit // 32]) & (1 << (bit % 32)):
            raise ValueError(f"{name} must be finite")
    for bit in (65, 66):
        if int(driver[bit // 32]) & (1 << (bit % 32)):
            vegetation = params.bundle.vegetation_for(params.dataset_identifier)
            message = (f"RUC ivgtyp is outside 1..{len(vegetation.rows)}"
                       if bit == 65 else
                       f"RUC isltyp is outside 1..{len(params.bundle.soil.rows)}")
            raise ValueError(message)
    for bit in (80, 81, 82, 83, 84, 85):
        if int(driver[bit // 32]) & (1 << (bit % 32)):
            raise ValueError("RUC qsn temperature or lookup index failed admission")
    for index, check in enumerate(_SFCTMP_CHECKS):
        if int(sfctmp[index // 64]) & (1 << (index % 64)):
            if isinstance(check, str):
                raise ValueError(check.replace("1..30", f"1..{len(params.vegetation.rows)}"))
            raise ValueError(f"RUC SFCTMP check {index} failed admission")
    for bit in (1024, 1025):
        if int(driver[bit // 32]) & (1 << (bit % 32)):
            raise ValueError("RUC qsn temperature or lookup index failed admission")
    for index, name in enumerate(_OUTPUT_ADMISSIONS):
        bit = 1056 + index
        if int(driver[bit // 32]) & (1 << (bit % 32)):
            raise ValueError(f"RUC driver produced non-finite {name}")
    raise ValueError("RUC Metal driver admission failed")


def step(fields, atmosphere, *, params, precipitation, dt, itimestep,
         pending_rainbl=None, frzfrac_mode: bool = False,
         lakemodel: int = 0,
         spp_lsm: int = 0, pattern_spp_lsm=None, field_sf=None,
         mosaic_lu=0, mosaic_soil=0, irrigation="wrf_461",
         soilprop="wrf_45", qvg_cold_start="wrf", diagnostic_2m="flux",
         snow="wrf_461"):
    """Run the production RUC chain on Metal and commit only clean outputs.

    The lineage selectors are woof.core.ruc_fused.step's: CUDA compiles
    soilprop/snow into the module, Metal passes them as `forms` bits.
    """
    from woof.core.ruc_mosaic import irrigation_form
    from woof.core.ruc_tier import (ruc_2m_diagnostic_form,
                                     ruc_qvg_cold_start_form, ruc_snow_form,
                                     ruc_soilprop_form)
    from woof.core.ruc_spp import validate_spp_mode
    forms = ((_FORM_SOILPROP_V461 if ruc_soilprop_form(soilprop) == "wrf_461" else 0)
             | (_FORM_SNOW_V461 if ruc_snow_form(snow) == "wrf_461" else 0))
    qvg_air = ruc_qvg_cold_start_form(qvg_cold_start)
    log_profile = ruc_2m_diagnostic_form(diagnostic_2m)
    irrigation_selector = irrigation_form(irrigation)
    if type(itimestep) is not int or itimestep < 1:
        raise ValueError("LSMRUC ktau is one-based and starts at 1")
    if type(frzfrac_mode) is not bool:
        raise TypeError("RUC Metal frzfrac_mode must be bool")
    if type(lakemodel) is not int or lakemodel not in (0, 1):
        raise ValueError("RUC Metal lakemodel must be 0 or 1")
    enabled_spp = validate_spp_mode(spp_lsm)
    levels = np.asarray(params.zs, dtype=np.float32)
    if levels.shape != (9,):
        raise NotImplementedError(
            "the packed Metal RUC source is generated for the nine-level WRF soil grid")
    context = fields["tsk"].buffer.context
    n = int(fields["tsk"].size)
    spp_pattern = spp_field = None
    has_field_sf = False
    if enabled_spp:
        expected_shape = (9, *tuple(fields["tsk"].shape))
        if (not isinstance(pattern_spp_lsm, MetalArray)
                or pattern_spp_lsm.dtype != np.dtype(np.float32)
                or not pattern_spp_lsm.flags.c_contiguous
                or tuple(pattern_spp_lsm.shape) != expected_shape):
            raise ValueError(
                "pattern_spp_lsm must be contiguous float32 Metal storage "
                f"with soil-first shape {expected_shape}")
        if pattern_spp_lsm.buffer.context is not context:
            raise ValueError("RUC Metal SPP pattern belongs to another context")
        pattern_host = pattern_spp_lsm.to_numpy()
        if not bool(np.all(np.isfinite(pattern_host))):
            raise ValueError(
                "pattern_spp_lsm must be finite before soil moisture transport")
        if bool(np.any(pattern_host < np.float32(-1.0))):
            raise ValueError(
                "pattern_spp_lsm below -1 would make hydraulic conductivity negative")
        if field_sf is not None:
            if (not isinstance(field_sf, MetalArray)
                    or field_sf.dtype != np.dtype(np.float32)
                    or not field_sf.flags.c_contiguous
                    or tuple(field_sf.shape) != expected_shape):
                raise ValueError(
                    f"field_sf must be contiguous float32 Metal storage with shape {expected_shape}")
            if field_sf.buffer.context is not context:
                raise ValueError("RUC Metal field_sf belongs to another context")
            if _overlaps(field_sf, pattern_spp_lsm):
                raise ValueError("field_sf must not alias the read-only SPP pattern")
            has_field_sf = True
        spp_pattern = pattern_spp_lsm.view(
            (9 * n,), byte_offset=pattern_spp_lsm.byte_offset)
        if has_field_sf:
            spp_field = field_sf.view((9 * n,), byte_offset=field_sf.byte_offset)
    # The wrf_45 floor reads the fractions whenever the run carries them
    # (WRF v4.5.2 :970 has no mosaic gate); wrf_461 only under mosaic_lu.
    landusef = fields.get("landusef")
    if not (mosaic_lu or (irrigation_selector == 0 and landusef is not None)):
        landusef = None
    soilctop = fields.get("soilctop") if mosaic_soil else None
    if mosaic_lu and landusef is None:
        raise ValueError("RUC mosaic_lu needs LANDUSEF category fractions")
    if mosaic_soil and soilctop is None:
        raise ValueError("RUC mosaic_soil needs SOILCTOP category fractions")
    nlcat = 0 if landusef is None else int(landusef.shape[0])
    nscat = 0 if soilctop is None else int(soilctop.shape[0])
    vegetation, tables = _parameter_tables(params, landusef, soilctop)
    crop, natural = (int(vegetation.scalars[name]) for name in ("CROP", "NATURAL"))
    if nlcat and nlcat < max(crop, natural):
        raise ValueError("RUC landusef omits the table's crop/natural categories; "
                         "LSMRUC irrigation cannot index the source fractions")
    xice_threshold = np.float32(params.xice_threshold)
    table_words = sum(array.size for array in tables.values())
    key = (id(context), n, table_words)
    # Parameter owners live with their forecast, unlike a process-wide
    # context/shape key. Finished ensemble members release their slabs while
    # successive calls on the same domain retain the same workspace.
    owned = _WORKSPACES.setdefault(params, {})
    workspace = owned.get(key)
    if workspace is None or workspace.context is not context:
        workspace = owned[key] = RucMetalWorkspace(
            context, n, table_words=table_words)
    if enabled_spp:
        raw_float = workspace.raw.view(
            (workspace.raw.size,), np.float32,
            byte_offset=workspace.raw.byte_offset)
        if _overlaps(raw_float, spp_pattern) or (
                spp_field is not None and _overlaps(raw_float, spp_field)):
            raise ValueError("RUC Metal SPP resources must not alias the packed workspace")
    offsets = workspace.upload_tables(tables)

    iswater = params.iswater if params.iswater is not None else (
        16 if vegetation.name == "USGS-RUC" else 17)
    isice = params.isice if params.isice is not None else (
        24 if vegetation.name == "USGS-RUC" else 15)
    if type(iswater) is not int or not 1 <= iswater <= len(vegetation.rows):
        raise ValueError(f"RUC iswater {iswater!r} is outside the vegetation table")
    if type(isice) is not int or not 1 <= isice <= len(vegetation.rows):
        raise ValueError(f"RUC isice {isice!r} is outside the vegetation table")

    # WRF's two host-side powf expressions are part of SFCDIAGS_RUCLSM. They
    # run after the soil chain in CUDA; evaluating them here before dispatch
    # leaves the same float32 values in the private workspace and keeps the
    # all-or-nothing public commit.
    psfc = fields["psfc"].to_numpy()
    scale, inverse = sfcdiags_exner_powers(psfc)
    workspace.field_view("scale", (n,))[...] = np.ascontiguousarray(scale, dtype=np.float32).reshape(-1)
    workspace.field_view("inverse", (n,))[...] = np.ascontiguousarray(inverse, dtype=np.float32).reshape(-1)

    values = _input_values(fields, atmosphere, precipitation,
                           frzfrac_mode=frzfrac_mode)
    prologue_parameters = {
        "ktau": itimestep, "dt": np.float32(dt), "iswater": iswater,
        "isice": isice, "nv": len(vegetation.rows),
        "ns": len(params.bundle.soil.rows),
        "icealbedo": np.float32(params.seaice_albedo_default),
        "cn": np.float32(vegetation.scalars["CFACTR_DATA"]),
        "frzfrac_mode": int(frzfrac_mode),
        "nlcat": nlcat, "nscat": nscat,
        "mosaic_lu": int(mosaic_lu), "mosaic_soil": int(mosaic_soil),
        "lakemodel": lakemodel, "qvg_air": int(qvg_air),
        "rdlai2d": int(bool(params.rdlai2d)),
        "xice_threshold": xice_threshold, "forms": forms,
        "table_offset_landusef": offsets["landusef"],
        "table_offset_soilctop": offsets["soilctop"],
        **{f"table_offset_{name}": offsets[name] for name in _TABLE_NAMES[:14] if name != "rstbl"},
    }
    # Prologue's table operands are the 14 vegetation/soil vectors followed
    # by tbq; SFCTMP owns its own two extra vegetation rows and zshalf.
    prologue_parameters["table_offset_ifortbl"] = offsets["ifortbl"]
    prologue_parameters["table_offset_z0tbl"] = offsets["z0tbl"]
    prologue_parameters["table_offset_lemitbl"] = offsets["lemitbl"]
    prologue_parameters["table_offset_pctbl"] = offsets["pctbl"]
    prologue_parameters["table_offset_laitbl"] = offsets["laitbl"]
    prologue_parameters["table_offset_bb"] = offsets["bb"]
    prologue_parameters["table_offset_drysmc"] = offsets["drysmc"]
    prologue_parameters["table_offset_hc"] = offsets["hc"]
    prologue_parameters["table_offset_maxsmc"] = offsets["maxsmc"]
    prologue_parameters["table_offset_refsmc"] = offsets["refsmc"]
    prologue_parameters["table_offset_satpsi"] = offsets["satpsi"]
    prologue_parameters["table_offset_satdk"] = offsets["satdk"]
    prologue_parameters["table_offset_wltsmc"] = offsets["wltsmc"]
    prologue_parameters["table_offset_qtz"] = offsets["qtz"]
    prologue_parameters["table_offset_tbq"] = offsets["tbq"]
    sfctmp_parameters = {
        "table_offset_rstbl": offsets["rstbl"],
        "table_offset_rgltbl": offsets["rgltbl"],
        "table_offset_z0tbl": offsets["z0tbl"],
        "table_offset_lemitbl": offsets["lemitbl"],
        "table_offset_tbq": offsets["tbq"],
        "table_offset_zshalf": offsets["zshalf"],
        "delt": np.float32(dt), "c1sn": np.float32(0.026),
        "c2sn": np.float32(21.0),
        "rsmax": np.float32(vegetation.scalars["RSMAX_DATA"]),
        "isice": isice, "urban": int(vegetation.scalars["URBAN"]),
        "ncategory": len(vegetation.rows), "limit": len(_SFCTMP_CHECKS),
        "forms": forms,
        "spp_lsm": int(enabled_spp),
        "has_field_sf": int(has_field_sf),
    }
    epilogue_parameters = {
        "dt": np.float32(dt), "table_offset_tbq": offsets["tbq"],
        "table_offset_lemitbl": offsets["lemitbl"],
        "table_offset_zshalf": offsets["zshalf"],
        "table_offset_landusef": offsets["landusef"],
        "nlcat": nlcat, "mosaic_lu": int(mosaic_lu),
        "crop": crop, "natural": natural,
        "irrigation": irrigation_selector, "log_profile": int(log_profile),
        "xice_threshold": xice_threshold,
    }
    dispatch_full_chain(
        workspace=workspace, inputs=values,
        prologue_parameters=prologue_parameters,
        sfctmp_parameters=sfctmp_parameters,
        epilogue_parameters=epilogue_parameters,
        pending_rainbl=pending_rainbl,
        pattern_spp_lsm=spp_pattern,
        field_sf=spp_field)
    census = _flag_error(workspace.flag_slab, params)
    scatter_committed(workspace=workspace, outputs=_outputs(fields), admitted=True)
    return dict(zip(("land", "water", "lake", "sea_ice"), map(int, census)))


__all__ = ["step"]
