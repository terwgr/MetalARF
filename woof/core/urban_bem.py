"""``sf_urban_physics = 3``: WRF v4.7.1 BEP+BEM on the GPU.

The multi-layer building effect parameterisation with the building energy
model -- indoor temperature and humidity, air conditioning and its waste
heat, windows, floors, green roofs and rooftop photovoltaics -- is
``phys/module_sf_bep_bem.F`` (``BEP_BEM`` and its own copies of the BEP
routines; WRF keeps two diverging BEP copies and identity means transcribing
the one that runs) and ``phys/module_sf_bem.F``.  Both live, in full, in
``kernels/urban_bep_bem.cu`` and ``kernels/urban_bem.cuh``, generated from the
pinned Fortran by ``tools/transcribe_urban_bem.py``: one thread per urban
column (``FRC_URB2D > 0``, module_sf_bep_bem.F:710), FP32 in Fortran's
evaluation order, no FMA contraction, no flush to zero, glibc 2.43's float
libm words (``glibc_flt32.cuh`` expf/logf/powf, ``glibc_trig_flt32.cuh``
sinf/cosf/asinf/acosf/atanf/tanf, CORE-MATH log10f in ``urban_bem.cuh``).  Every WRF local array lives in a column-private, interleaved
global workspace (``ws[offset * nthreads + tid]``), so the per-thread local
frame stays small.  Every FP32/FP64 operator is emitted as its IEEE
intrinsic (``__fmul_rn`` ...), so no compiler can reassociate or contract
it: NVRTC 13.3 for sm_120 rewrote ``(c*ch)/cm`` as ``c*(ch/cm)`` in
``flux_flat`` until it was, and the RTX 5090 was then up to 4552 ULP off
WRF where the RTX 4090 was bit-identical.

Proof: ``tests/test_urban_bem_wrf471_parity.py`` runs the kernel against WRF's
own ``BEP_BEM`` (``tools/urban_wrf471_oracle/run_bep_bem.F90``, five fixtures,
up to 30 consecutive calls) and asserts max ULP 0 on every output and every
prognostic array; measured on an RTX 4090 (sm_89) and an RTX 5090 (sm_120).  WRF's compiled Fortran is the CPU reference.

Coupling (DESIGN.md 3.4): :func:`after_lsm` zeroes what the land-surface
driver zeroes, runs the column and hands over to the shared BEP couple
(``woof.core.urban_bep_couple``), exactly as ``module_sf_noahdrv.F:1636-1776``
and ``module_sf_noahmpdrv.F:3644-3776`` do.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from woof.core.urban_bem_layout import METADATA

# module_sf_bep_bem.F:22-53, the BEP_BEM module parameters.
NDM = 2
NZ_UM = 18
NG_U = 10
NGR_U = 10
NWR_U = 10
NF_U = 10
NGB_U = 10
NBUI_MAX = 15
NURBMAX = 11
DZ_U = 5.0

#: What check_a_mundo.F:476-485 reads from bep_bem_ndm() .. bep_bem_ngr_u().
DIMENSIONS = MappingProxyType({
    "ndm": NDM, "nz_um": NZ_UM, "ng_u": NG_U, "nwr_u": NWR_U, "nf_u": NF_U,
    "ngb_u": NGB_U, "nbui_max": NBUI_MAX, "ngr_u": NGR_U,
})

#: The option-3 arrays (Registry.EM_COMMON), in infra's row convention:
#: layers 0 = 2-D, a string names an ``urban_map_*`` size or ``ndm``.
STATE_SPEC = MappingProxyType({
    "tsk_rural_bep": ("f4", 0),
    "trb_urb4d": ("f4", "zrd"), "tw1_urb4d": ("f4", "zwd"),
    "tw2_urb4d": ("f4", "zwd"), "tgb_urb4d": ("f4", "gd"),
    "sfw1_urb3d": ("f4", "zd"), "sfw2_urb3d": ("f4", "zd"),
    "sfr_urb3d": ("f4", "zdf"), "sfg_urb3d": ("f4", "ndm"),
    "tlev_urb3d": ("f4", "bd"), "qlev_urb3d": ("f4", "bd"),
    "tw1lev_urb3d": ("f4", "wd"), "tw2lev_urb3d": ("f4", "wd"),
    "tglev_urb3d": ("f4", "gbd"), "tflev_urb3d": ("f4", "fbd"),
    **{name: ("f4", 0) for name in (
        "sf_ac_urb3d", "lf_ac_urb3d", "cm_ac_urb3d", "sfvent_urb3d",
        "lfvent_urb3d", "ep_pv_urb3d", "qgr_urb3d", "tgr_urb3d",
        "draingr_urb3d")},
    "sfwin1_urb3d": ("f4", "wd"), "sfwin2_urb3d": ("f4", "wd"),
    "t_pv_urb3d": ("f4", "zdf"), "trv_urb4d": ("f4", "zgrd"),
    "qr_urb4d": ("f4", "zgrd"), "drain_urb4d": ("f4", "zdf"),
    "sfrv_urb3d": ("f4", "zdf"), "lfrv_urb3d": ("f4", "zdf"),
    "dgr_urb3d": ("f4", "zdf"), "lfr_urb3d": ("f4", "zdf"),
    "dg_urb3d": ("f4", "ndm"), "lfg_urb3d": ("f4", "ndm"),
})

# ---------------------------------------------------------------------------
# the generated layouts
# ---------------------------------------------------------------------------

def _layout(section: str, kind: str) -> Mapping[str, tuple[int, tuple]]:
    return MappingProxyType({
        name: (offset, tuple(shape))
        for name, (k, offset, shape) in METADATA[section].items()
        if k == kind})


#: module_sf_urban arrays BEP_BEM reads, packed in Fortran order.
BEM_TABLE_FLOAT_LAYOUT = _layout("table", "float")
BEM_TABLE_INT_LAYOUT = _layout("table", "int")
#: The ``save``d class arrays BEP_BEM's first call builds (init_para, icBEP).
#: ``clsi[0]`` is the class kernel's error word.
BEM_CLASS_FLOAT_LAYOUT = _layout("class", "float")
BEM_CLASS_INT_LAYOUT = MappingProxyType(
    {"_error": (0, ()), **_layout("class", "int")})
WORKSPACE_FIXED_FLOATS = int(METADATA["fixed"])
WORKSPACE_FLOATS_PER_LEVEL = int(METADATA["per_level"])
COLUMN_ARRAY_ARGUMENTS = tuple(METADATA["array_arguments"])
COLUMN_SCALAR_ARGUMENTS = tuple(METADATA["scalar_arguments"])
#: Error code -> the Fortran check it stands for (file:line, WRF's text).
ERRORS = MappingProxyType({
    number: f"{text} ({where})"
    for where, (number, text) in METADATA["errors"].items()})

#: No FMA contraction (gfortran -O0 contracts nothing) and no flush to zero
#: (gfortran keeps subnormals); the parity test fails with either dropped.
MODULE_OPTIONS = ("-std=c++17", "-fmad=false", "--ftz=false")
MODULE_KEY = "woof.core.urban_bem:urban_bep_bem"
_SOURCES = ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "urban_bem.cuh",
            "urban_bep_bem.cu")

#: Per-category rows are laid out for NURBMAX classes; the hourly profiles
#: are 24 long in every table.
_HOURLY = ("hsequip_tbl", "irho_tbl")


def _length(layout) -> int:
    return max((offset + int(np.prod(shape, dtype=np.int64))
                for offset, shape in layout.values()), default=0)


def pack_bem_tables(tbl: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Pack every module_sf_urban variable BEP_BEM reads (Fortran order).

    ``tbl`` is keyed by the lower-case module_sf_urban names (``capb_tbl``,
    ``street_direction_tbl``, ``icate``, ...).  Per-category arrays may carry
    ICATE (3 or 11) categories; the layout reserves eleven and pads.
    """
    for name in METADATA["table"]:
        if name not in tbl:
            raise KeyError(name)
    icate = int(tbl["icate"])
    if icate not in (3, 11):
        raise ValueError(
            f"icate={icate}: WRF allocates 3 (URBPARM.TBL) or 11 "
            "(URBPARM_LCZ.TBL) urban classes (module_check_a_mundo.F:459-461)")
    packed = []
    for layout, dtype in ((BEM_TABLE_FLOAT_LAYOUT, np.float32),
                          (BEM_TABLE_INT_LAYOUT, np.int32)):
        out = np.zeros(_length(layout), dtype=dtype)
        for name, (offset, shape) in layout.items():
            value = np.asarray(tbl[name], dtype=dtype)
            if shape:
                expected = (shape if name in _HOURLY
                            else shape[:-1] + (icate,))
                if value.shape not in (shape, expected):
                    raise ValueError(
                        f"{name}: expected Fortran shape {expected} or "
                        f"{shape}, got {value.shape}")
                padded = np.zeros(shape, dtype=dtype)
                padded[tuple(slice(0, n) for n in value.shape)] = value
                words = padded.ravel(order="F")
            else:
                if value.shape != ():
                    raise ValueError(f"{name}: expected a scalar, got "
                                     f"{value.shape}")
                words = value.reshape(1)
            out[offset:offset + words.size] = words
        packed.append(out)
    return packed[0], packed[1]


def tables_from_params(params) -> dict[str, object]:
    """The :func:`pack_bem_tables` input from infra's ``UrbanParams``."""
    out: dict[str, object] = {}
    for name in METADATA["table"]:
        if name == "icate":
            out[name] = int(params.icate)
            continue
        value = np.asarray(params.values[name.upper()])
        out[name] = value.item() if value.ndim == 0 else value
    return out


# ---------------------------------------------------------------------------
# the device module
# ---------------------------------------------------------------------------

def module_source() -> str:
    """The exact source string NVRTC compiles."""
    from woof.core.kernels import _preamble

    root = Path(__file__).parent / "kernels"
    return _preamble() + "".join(
        (root / name).read_text(encoding="utf-8") for name in _SOURCES)


@lru_cache(maxsize=None)
def _bem_module():
    """Dedicated loader: this module alone needs both option overrides."""
    import cupy as cp

    from woof.certify.kernel_manifest import record_module

    source = module_source()
    module = cp.RawModule(code=source, options=MODULE_OPTIONS,
                          name_expressions=None)
    module.compile()
    record_module(MODULE_KEY, source=source, options=MODULE_OPTIONS,
                  module=module)
    return module


def launch_bem_view_factors_metal(context, *, z, dxy, street_width,
                                 nz_u: int, nd: int = NDM) -> dict:
    """Build per-column BEM canyon view factors on the resident Metal device.

    ``z`` has shape ``(ncol, 19)`` and ``dxy``/``street_width`` have shape
    ``(ncol, nd)``. The returned factor arrays retain the source direction
    and 18-level capacities, together with an int32 per-direction error
    field. This is the geometry caller for the integrated BEM radiosity
    fragments; the full BEP+BEM column solve remains a separate route.
    """
    from woof.metal.array import MetalArray, MetalArrayModule
    from woof.metal.urban_bep_bem_family import submit_bep_bem_view_factors

    if not 1 <= int(nz_u) <= NZ_UM or int(nd) != NDM:
        raise ValueError("BEM view-factor extent is outside WRF limits")
    arrays = {"z": z, "dxy": dxy, "street_width": street_width}
    if any(not isinstance(value, MetalArray) for value in arrays.values()):
        raise TypeError("BEM view-factor inputs must be resident MetalArrays")
    ncol = int(z.shape[0]) if z.ndim == 2 else 0
    if (z.dtype != np.dtype(np.float32) or tuple(z.shape) != (ncol, 19)
            or tuple(dxy.shape) != (ncol, int(nd))
            or tuple(street_width.shape) != (ncol, int(nd))):
        raise ValueError("BEM view-factor inputs have incompatible shapes")
    if any(value.dtype != np.dtype(np.float32)
           or value.buffer.context is not context
           or not value.flags.c_contiguous for value in arrays.values()):
        raise TypeError("BEM view-factor inputs must be contiguous FP32 on context")
    module = MetalArrayModule(context)
    resources = {
        **arrays,
        "fww": module.zeros((ncol, nd, 18, 18), dtype=np.float32),
        "fwg": module.zeros((ncol, nd, 18), dtype=np.float32),
        "fgw": module.zeros((ncol, nd, 18), dtype=np.float32),
        "fsw": module.zeros((ncol, nd, 18), dtype=np.float32),
        "fws": module.zeros((ncol, nd, 18), dtype=np.float32),
        "fsg": module.zeros((ncol, nd), dtype=np.float32),
        "error": module.zeros((ncol, nd), dtype=np.int32),
    }
    submit_bep_bem_view_factors(context, resources, nz_u=nz_u, nd=nd)
    return resources


class BepBemError(RuntimeError):
    """A WRF fatal (``stop`` / ``wrf_error_fatal``) inside BEP_BEM or BEM."""


_EXTRA_ERRORS = {
    9001: "the column workspace is smaller than WRF's locals need",
    9002: "ICATE is outside the 1..11 class arrays"}


def _message(code: int) -> str:
    return ERRORS.get(code, _EXTRA_ERRORS.get(code, f"error {code}"))


def _refuse(code: int) -> None:
    raise BepBemError(f"WRF BEP_BEM stopped where WRF stops: {_message(code)}")


def _check_errors(err, cols=None) -> None:
    """Immediate read (class tables, tests): name the first failing column."""
    import cupy as cp

    host = cp.asnumpy(err)
    bad = np.flatnonzero(host)
    if not bad.size:
        return
    first = int(bad[0])
    where = ("class initialization" if cols is None
             else f"column {int(cols[first].item())}")
    raise BepBemError(f"WRF BEP_BEM {where}: {_message(int(host[first]))}")


def build_class_tables(tblf, tbli) -> dict:
    """Run BEP_BEM's first-call block (init_para, icBEP) on the device."""
    import cupy as cp

    tblf = cp.asarray(tblf, dtype=cp.float32)
    tbli = cp.asarray(tbli, dtype=cp.int32)
    if tblf.ndim != 1 or tblf.size != _length(BEM_TABLE_FLOAT_LAYOUT):
        raise ValueError("tblf does not match BEM_TABLE_FLOAT_LAYOUT")
    if tbli.ndim != 1 or tbli.size != _length(BEM_TABLE_INT_LAYOUT):
        raise ValueError("tbli does not match BEM_TABLE_INT_LAYOUT")
    cls = {"clsf": cp.zeros(_length(BEM_CLASS_FLOAT_LAYOUT), cp.float32),
           "clsi": cp.zeros(_length(BEM_CLASS_INT_LAYOUT), cp.int32)}
    _bem_module().get_function("bep_bem_class_init")(
        (1,), (1,), (tblf, tbli, cls["clsf"], cls["clsi"]))
    _check_errors(cls["clsi"][:1])
    return cls


def build_class_tables_metal(params, context) -> dict:
    """Run BEP+BEM's class parameter, albedo and profile setup on Metal."""
    from woof.metal.array import MetalArrayModule
    from woof.metal.urban_bep_bem_family import (
        submit_bep_bem_albedo, submit_bep_bem_fragment,
        submit_bep_bem_profile_init)

    nurbm = int(params.icate)
    if not 1 <= nurbm <= NURBMAX:
        raise ValueError(f"BEM ICATE={nurbm} is outside 1..{NURBMAX}")
    module = MetalArrayModule(context)
    table_float, table_int = pack_bem_tables(tables_from_params(params))
    table_float = module.asarray(table_float)
    table_int = module.asarray(table_int)
    cls = {
        "clsf": module.zeros((_length(BEM_CLASS_FLOAT_LAYOUT),),
                             dtype=np.float32),
        "clsi": module.zeros((_length(BEM_CLASS_INT_LAYOUT),),
                             dtype=np.int32),
    }
    submit_bep_bem_fragment(
        context, "urban_bep_bem_class_init_fragment",
        {"table_float": table_float, "table_int": table_int,
         "class_float": cls["clsf"], "class_int": cls["clsi"]},
        {"nurbm": nurbm}, grid_size=nurbm)

    # bb_init_para computes one class window albedo before bb_icbep. Reuse
    # the packed class field as output so the following stages consume one
    # class descriptor, as the CUDA route does.
    albwin = cls["clsf"].view((nurbm,), byte_offset=1052 * 4)
    p_num = module.asarray(np.full((nurbm,), 2, dtype=np.int32))
    q_num = module.asarray(np.full((nurbm,), 4, dtype=np.int32))
    albedo_error = module.zeros((nurbm,), dtype=np.int32)
    submit_bep_bem_albedo(context, p_num, q_num, albwin, albedo_error)

    z_u = module.asarray(
        np.arange(NZ_UM + 1, dtype=np.float32) * np.float32(DZ_U))
    h_b = cls["clsf"].view((nurbm, NZ_UM), byte_offset=242 * 4)
    d_b = cls["clsf"].view((nurbm, NZ_UM), byte_offset=440 * 4)
    profiles = {
        "z_u": z_u,
        "h_b": h_b,
        "d_b": d_b,
        "ss_u": module.zeros((nurbm, NZ_UM + 1), dtype=np.float32),
        "pb_u": module.zeros((nurbm, NZ_UM + 1), dtype=np.float32),
        "nz_u": module.zeros((nurbm,), dtype=np.int32),
        "error": module.zeros((nurbm,), dtype=np.int32),
    }
    submit_bep_bem_profile_init(context, profiles, nz_u=NZ_UM)
    submit_bep_bem_fragment(
        context, "urban_bep_bem_class_profile_pack_fragment",
        {"ss_profile": profiles["ss_u"], "pb_profile": profiles["pb_u"],
         "z_u": z_u, "nz_profile": profiles["nz_u"],
         "profile_error": profiles["error"],
         "class_float": cls["clsf"], "class_int": cls["clsi"]},
        {"nurbm": nurbm}, grid_size=nurbm)
    cls["profiles"] = profiles
    return cls


def launch_bem_class_profiles_metal(params, context) -> dict:
    """Initialize BEM class height profiles in resident Metal buffers.

    This is the `bb_icbep` part of the full class initializer. It consumes
    the pinned parameter tables, dispatches the integrated profile fragment,
    and returns the profile arrays without importing CuPy or reading device
    state back to the host. The remaining class fields and column solve are
    separate stages.
    """
    from woof.metal.array import MetalArrayModule
    from woof.metal.urban_bep_bem_family import submit_bep_bem_profile_init

    nurbm = int(params.icate)
    if not 1 <= nurbm <= NURBMAX:
        raise ValueError(f"BEM ICATE={nurbm} is outside 1..{NURBMAX}")
    tables = tables_from_params(params)
    h_b = np.asarray(tables["height_bin_tbl"], dtype=np.float32)
    d_b = np.asarray(tables["hpercent_bin_tbl"], dtype=np.float32)
    if h_b.ndim != 2 or d_b.shape != h_b.shape or h_b.shape[0] < NZ_UM:
        raise ValueError("BEM height tables do not cover all 18 urban bins")
    h_b = np.ascontiguousarray(h_b[:NZ_UM, :nurbm].T)
    d_b = np.ascontiguousarray(d_b[:NZ_UM, :nurbm].T)
    module = MetalArrayModule(context)
    resources = {
        "z_u": module.asarray(
            np.arange(NZ_UM + 1, dtype=np.float32) * np.float32(DZ_U)),
        "h_b": module.asarray(h_b),
        "d_b": module.asarray(d_b),
        "ss_u": module.zeros((nurbm, NZ_UM + 1), dtype=np.float32),
        "pb_u": module.zeros((nurbm, NZ_UM + 1), dtype=np.float32),
        "nz_u": module.zeros((nurbm,), dtype=np.int32),
        "error": module.zeros((nurbm,), dtype=np.int32),
    }
    submit_bep_bem_profile_init(context, resources, nz_u=NZ_UM)
    return resources


_CLASS_CACHE: dict[tuple, dict] = {}
_METAL_CLASS_CACHE: dict[tuple, dict] = {}


def class_tables(params, *, context=None) -> dict:
    """:func:`build_class_tables` for one ``UrbanParams``, once per device.

    WRF builds these once, on BEP_BEM's first call (``first``,
    module_sf_bep_bem.F:690-705), from the one table every domain reads.
    """
    if context is not None:
        key = (params.sha256, int(params.use_wudapt_lcz), int(params.icate),
               id(context))
        held = _METAL_CLASS_CACHE.get(key)
        if held is None:
            held = build_class_tables_metal(params, context)
            _METAL_CLASS_CACHE[key] = held
        return held

    import cupy as cp

    key = (params.sha256, int(params.use_wudapt_lcz), int(params.icate),
           cp.cuda.Device().id)
    held = _CLASS_CACHE.get(key)
    if held is None:
        held = build_class_tables(*pack_bem_tables(tables_from_params(params)))
        _CLASS_CACHE[key] = held
    return held


def workspace_bytes_per_column(nz: int) -> int:
    """Upper bound of WRF's local arrays for one column (float32 words)."""
    return 4 * (WORKSPACE_FIXED_FLOATS
                + WORKSPACE_FLOATS_PER_LEVEL * (int(nz) + 1))


class ColumnPlan:
    """The urban column list, chunked to a workspace budget, and its buffers.

    FRC_URB2D is fixed for the run once the cold start has set it, so a
    domain builds its plan once (:func:`after_lsm` keeps it on the state)
    and the per-step launch reads nothing back from the device.
    """

    def __init__(self, frc_urb2d, nz: int, workspace_bytes: int) -> None:
        from woof.metal.array import MetalArray, MetalArrayModule

        if isinstance(frc_urb2d, MetalArray):
            if frc_urb2d.dtype != np.dtype(np.float32):
                raise TypeError("frc_urb2d must be FP32")
            columns = np.flatnonzero(
                frc_urb2d.to_numpy().reshape(-1) > np.float32(0)).astype(
                    np.int32, copy=False)
            self.count = int(columns.size)
            self.nz = int(nz)
            self.shape = tuple(frc_urb2d.shape)
            per_column = workspace_bytes_per_column(nz)
            chunk = min(max(self.count, 1), int(workspace_bytes) // per_column)
            if chunk < 1:
                raise ValueError(
                    f"workspace_bytes must be at least {per_column}")
            self.words = per_column // 4
            module = MetalArrayModule(frc_urb2d.buffer.context)
            self.parts = [module.asarray(np.ascontiguousarray(
                columns[start:start + chunk]))
                for start in range(0, self.count, chunk)]
            self.ws = module.empty((self.words * chunk,), dtype=np.float32)
            self.err = module.zeros((chunk,), dtype=np.int32)
            self.metal_context = frc_urb2d.buffer.context
            return

        import cupy as cp

        cols = cp.flatnonzero(frc_urb2d.ravel() > cp.float32(0))
        cols = cols.astype(cp.int32)
        self.count = int(cols.size)
        self.nz = int(nz)
        self.shape = tuple(frc_urb2d.shape)
        per_column = workspace_bytes_per_column(nz)
        chunk = min(max(self.count, 1), int(workspace_bytes) // per_column)
        if chunk < 1:
            raise ValueError(f"workspace_bytes must be at least {per_column}")
        self.words = per_column // 4
        self.parts = [cols[start:start + chunk]
                      for start in range(0, self.count, chunk)]
        self.ws = (cp.empty(self.words * chunk, cp.float32)
                   if self.count else None)
        self.err = cp.zeros(chunk, cp.int32)


def _launch_bep_bem_columns_metal(
        dev: Mapping, cls: Mapping, *, gmt, julday, declin_urb, dt,
        itimestep, num_urban_hi, workspace_bytes, plan=None) -> ColumnPlan:
    """Prepare the BEP record and BEM column geometry/state through bb_param.

    The atmosphere gather is shared with option 2. BEM's selected height
    profile and prognostic arrays are packed into source-local arrays without
    modifying the resident state. The path stops after the per-column
    building-level scan, before the remaining atmospheric/BEM solver chain.
    """
    del gmt, julday, itimestep
    from woof.metal.array import MetalArray, MetalArrayModule
    from woof.metal.urban_bep_family import submit_urban_bep_column_gather
    from woof.metal.urban_bep_bem_family import (
        submit_bem_atmosphere_gather, submit_bem_building_level_scan,
        submit_bem_atmosphere_interp, submit_bem_icbep_xy, submit_bem_param,
        submit_bem_temperature_average, submit_bem_temperature_profile_average,
        submit_bem_shadow_mas, submit_bem_longwave, submit_bem_shortwave,
        submit_bem_upward_rad,
        submit_bem_surface_temp,
        submit_bem_rain_update,
        submit_bem_occupancy_inputs, submit_bem_occupancy,
        submit_bem_indoor_radiation, submit_bem_indoor_fluxes,
        submit_bem_room_balance_inputs, submit_bem_room_control,
        submit_bem_indoor_state_update, submit_bem_consump_total,
        submit_bem_wall_flux, submit_bem_flat_flux,
        submit_bem_flat_flux_pack,
        submit_bem_buildings_tendency,
        submit_bem_urban_meso,
        submit_bem_interp_length,
        submit_bem_final_output, submit_bem_state_scatter,
        submit_bep_bem_fragment)

    fraction = dev.get("frc_urb2d")
    if not isinstance(fraction, MetalArray) or fraction.ndim != 2:
        raise TypeError("Metal BEP_BEM frc_urb2d must be a 2-D MetalArray")
    context = fraction.buffer.context
    if fraction.dtype != np.dtype(np.float32):
        raise TypeError("Metal BEP_BEM frc_urb2d must be FP32")
    ny, nx = map(int, fraction.shape)
    ncol = ny * nx
    nz = int(dev["dz8w"].shape[0])
    if nz < 2:
        raise ValueError("BEP_BEM needs at least two atmospheric mass levels")
    if not 0 < int(num_urban_hi) < NZ_UM:
        raise ValueError(
            f"num_urban_hi={num_urban_hi}: BEP_BEM requires a value below "
            f"nz_um={NZ_UM} (module_sf_bep_bem.F:656)")

    for name in COLUMN_ARRAY_ARGUMENTS:
        array = dev[name]
        dtype = np.dtype(np.int32 if name == "utype_urb2d" else np.float32)
        if (not isinstance(array, MetalArray) or array.dtype != dtype
                or not array.flags.c_contiguous
                or array.buffer.context is not context):
            raise TypeError(
                f"{name}: BEP_BEM requires a same-context contiguous Metal "
                f"{dtype.name} array")
        if tuple(array.shape[-2:]) != (ny, nx):
            raise ValueError(
                f"{name}: horizontal shape {array.shape[-2:]} is not "
                f"{(ny, nx)}")
    for name in ("dz8w", "u_phy", "v_phy", "th_phy", "rho", "p_phy",
                 "qv_phy"):
        if tuple(dev[name].shape) != (nz, ny, nx):
            raise ValueError(f"{name}: expected atmospheric shape {(nz, ny, nx)}")

    class_float, class_int = cls.get("clsf"), cls.get("clsi")
    if (not isinstance(class_float, MetalArray)
            or not isinstance(class_int, MetalArray)
            or class_float.dtype != np.dtype(np.float32)
            or class_int.dtype != np.dtype(np.int32)
            or tuple(class_float.shape) != (_length(BEM_CLASS_FLOAT_LAYOUT),)
            or tuple(class_int.shape) != (_length(BEM_CLASS_INT_LAYOUT),)
            or not class_float.flags.c_contiguous
            or not class_int.flags.c_contiguous
            or class_float.buffer.context is not context
            or class_int.buffer.context is not context):
        raise TypeError("Metal BEP_BEM requires resident typed BEM class tables")
    profiles = cls.get("profiles")
    if not isinstance(profiles, Mapping) or any(
            not isinstance(profiles.get(name), MetalArray)
            or profiles[name].dtype != np.dtype(
                np.int32 if name == "nz_u" else np.float32)
            or profiles[name].buffer.context is not context
            for name in ("ss_u", "pb_u", "nz_u")):
        raise TypeError("Metal BEP_BEM requires initialized BEM class profiles")
    nclass = int(profiles["ss_u"].shape[0])
    if (tuple(profiles["ss_u"].shape) != (nclass, NZ_UM + 1)
            or tuple(profiles["pb_u"].shape) != (nclass, NZ_UM + 1)
            or tuple(profiles["nz_u"].shape) != (nclass,)):
        raise ValueError("BEM class profiles do not match the class table extent")
    if (plan is None or plan.nz != nz or plan.shape != (ny, nx)
            or getattr(plan, "metal_context", None) is not context):
        plan = ColumnPlan(fraction, nz, workspace_bytes)
    if not plan.count:
        return plan

    module = MetalArrayModule(context)
    # This is the same atmosphere -> column record stage used by
    # _launch_bep_columns_metal. It initializes all gathered fields from the
    # real atmospheric arrays and labels active urban columns; it is not a
    # substitute for gathering BEM's persistent state.
    column_x = module.zeros((ncol, 15, nz + 1), dtype=np.float32)
    active_cols = module.zeros((ncol,), dtype=np.int32)
    submit_urban_bep_column_gather(
        context,
        fraction.view((ncol,), byte_offset=fraction.byte_offset),
        dev["dz8w"].view((nz, ncol), byte_offset=dev["dz8w"].byte_offset),
        dev["u_phy"].view((nz, ncol), byte_offset=dev["u_phy"].byte_offset),
        dev["v_phy"].view((nz, ncol), byte_offset=dev["v_phy"].byte_offset),
        dev["th_phy"].view((nz, ncol), byte_offset=dev["th_phy"].byte_offset),
        dev["rho"].view((nz, ncol), byte_offset=dev["rho"].byte_offset),
        dev["p_phy"].view((nz, ncol), byte_offset=dev["p_phy"].byte_offset),
        column_x, active_cols)

    # WRF zeroes hi_urb1d, copies the configured high-rise levels from
    # hi_urb2d, then runs bb_icbephi_xy before constructing nlev/ind_bd.
    # Keep that source order: the profile is per-column and depends on the
    # actual hi_urb2d values, not just the class profile.
    from woof.metal.urban_bep_bem_family import submit_bep_bem_height_init

    hi_source = dev["hi_urb2d"]
    if hi_source.size != int(num_urban_hi) * ncol:
        raise ValueError(
            "hi_urb2d: expected num_urban_hi values per column, "
            f"found {hi_source.size // max(ncol, 1)}")
    hi_domain = hi_source.view(
        (int(num_urban_hi), ncol), byte_offset=hi_source.byte_offset)
    hi_column = module.empty((ncol, NZ_UM), dtype=np.float32)
    height_gather_error = module.zeros((ncol,), dtype=np.int32)
    submit_bep_bem_fragment(
        context, "urban_bep_bem_hi_profile_gather_fragment",
        {"hi_domain": hi_domain,
         "hi_column": hi_column, "error": height_gather_error},
        {"ncol": ncol, "num_urban_hi": int(num_urban_hi)},
        grid_size=ncol)

    def check_stage_error(error, stage):
        codes = error.to_numpy().reshape(-1)
        bad = np.flatnonzero(codes)
        if bad.size:
            col = int(bad[0])
            code = int(codes[col])
            reason = {
                5010: "invalid green-roof flag",
                5011: "more than 15 positive BEM height levels",
                5013: "num_urban_hi is outside the 0..18 range",
                5014: "invalid bb_param extent",
                5015: "UTYPE is outside the configured BEM classes",
                5016: "BEM class geometry/profile is outside WRF limits",
                5531: "UTYPE is outside the configured BEM classes",
                5532: "BEM nzu/nd is outside WRF limits",
                7201: "UTYPE is outside the configured BEM classes",
                7202: "invalid BEM street-direction count",
                7203: "singular BEM ground soil-temperature matrix",
                7301: "UTYPE is outside the configured BEM classes",
                7302: "invalid BEM street-direction count",
                7510: "UTYPE is outside the configured BEM classes",
                1915: "BEM occupancy hour is outside 0..23",
                3811: "UTYPE is outside the BEM class table",
                3812: "invalid indoor radiation geometry or window fraction",
                3813: "indoor radiation surface area is nonpositive",
                3814: "singular indoor shortwave radiosity matrix",
                3815: "singular indoor longwave radiosity matrix",
                3820: "BEM rooftop photovoltaics are not in the ordinary-profile path",
                3821: "UTYPE is outside the BEM class table",
                3822: "invalid indoor flux geometry or window fraction",
                3831: "UTYPE is outside the BEM class table",
                3832: "invalid BEM room thermodynamic state",
                3841: "invalid indoor heat/moisture capacity",
                3842: "nonfinite updated indoor state",
                3850: "BEM material stack exceeds the WRF layer bound",
                3851: "invalid wall material profile or solve state",
                3852: "invalid floor material profile or solve state",
                3853: "invalid ground material profile or solve state",
                3854: "invalid roof material profile or solve state",
                7410: "invalid BEM wind/street geometry for exterior wall flux",
                7420: "invalid BEM ground/roof surface exchange input",
                7430: "invalid level index while packing BEM roof flux",
                7411: "UTYPE is outside the configured BEM classes",
                7601: "invalid canyon geometry in bb_urban_meso",
                7602: "nonpositive atmospheric layer thickness in bb_urban_meso",
                7603: "invalid volume fraction in bb_urban_meso",
                7701: "invalid urban length scale in bb_interp_length",
                7702: "invalid building drag length in bb_interp_length",
                2155: "invalid BEM building level count in consumption sum",
                20: "more than 15 positive BEM building levels",
                12: "BEM high-rise profile requires more than 18 levels",
            }.get(code, f"kernel error {code}")
            raise BepBemError(
                f"Metal BEP_BEM {stage} failed at column {col}: {reason}")

    check_stage_error(height_gather_error, "high-rise profile gather")

    height_profile = {
        "z_u": profiles["z_u"],
        "hi_urb1d": hi_column,
        "hb_u": module.empty((ncol, NZ_UM), dtype=np.float32),
        "ss_u": module.empty((ncol, NZ_UM + 1), dtype=np.float32),
        "pb_u": module.empty((ncol, NZ_UM + 1), dtype=np.float32),
        "nzu_out": module.empty((ncol,), dtype=np.int32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bep_bem_height_init(context, height_profile, dz_u=DZ_U)
    check_stage_error(height_profile["error"], "bb_icbephi_xy height profile")
    ss_urb = height_profile["ss_u"]

    # All state resources use source-flat indexing [packed state, column].
    # These extents are ind_*'s fixed products from module_sf_bep_bem.F:551-647.
    state_extents = {
        "tlev_urb3d": 270, "qlev_urb3d": 270,
        "tw1lev_urb3d": 540, "tw2lev_urb3d": 540,
        "tglev_urb3d": 300, "tflev_urb3d": 5100,
        "sfwin1_urb3d": 540, "sfwin2_urb3d": 540,
        "tw1_urb4d": 5400, "tw2_urb4d": 5400,
        "tgb_urb4d": 20, "trb_urb4d": 360,
        "trv_urb4d": 360, "qr_urb4d": 360,
        "sfw1_urb3d": 540, "sfw2_urb3d": 540,
        "sfg_urb3d": 2, "lfg_urb3d": 2, "dg_urb3d": 2,
        "t_pv_urb3d": 36, "sfr_urb3d": 36, "lfr_urb3d": 36,
        "dgr_urb3d": 36, "sfrv_urb3d": 36, "lfrv_urb3d": 36,
        "drain_urb4d": 36,
    }
    state_resources = {"active_cols": active_cols, "ss_urb": ss_urb}
    for name, extent in state_extents.items():
        array = dev[name]
        if array.size != extent * ncol:
            raise ValueError(
                f"{name}: expected {extent} state values per column, "
                f"found {array.size // max(ncol, 1)}")
        state_resources[name] = array.view(
            (extent, ncol), byte_offset=array.byte_offset)

    packed_shapes = {
        "building_nlev": ((ncol, NBUI_MAX), np.int32),
        "tlev": ((ncol, NBUI_MAX, NZ_UM), np.float32),
        "qlev": ((ncol, NBUI_MAX, NZ_UM), np.float32),
        "tw": ((ncol, NBUI_MAX, NWR_U, NZ_UM, 4), np.float32),
        "tg": ((ncol, NG_U, NDM), np.float32),
        "tr": ((ncol, NWR_U, NZ_UM, NDM), np.float32),
        "trv": ((ncol, NGR_U, NZ_UM, NDM), np.float32),
        "qr": ((ncol, NGR_U, NZ_UM, NDM), np.float32),
        "twlev": ((ncol, NBUI_MAX, NZ_UM, 4), np.float32),
        "tglev": ((ncol, NBUI_MAX, NGB_U, NDM), np.float32),
        "tflev": ((ncol, NBUI_MAX, NF_U, NZ_UM - 1, NDM), np.float32),
        "sfwin": ((ncol, NBUI_MAX, NZ_UM, 4), np.float32),
        "sfw": ((ncol, NBUI_MAX, NZ_UM, 4), np.float32),
        "sfg": ((ncol, NDM), np.float32),
        "lfg": ((ncol, NDM), np.float32),
        "dg": ((ncol, NDM), np.float32),
        "tpv": ((ncol, NZ_UM, NDM), np.float32),
        "sfr": ((ncol, NZ_UM, NDM), np.float32),
        "lfr": ((ncol, NZ_UM, NDM), np.float32),
        "dgr": ((ncol, NZ_UM, NDM), np.float32),
        "sfrv": ((ncol, NZ_UM, NDM), np.float32),
        "lfrv": ((ncol, NZ_UM, NDM), np.float32),
        "drain": ((ncol, NZ_UM, NDM), np.float32),
        "error": ((ncol,), np.int32),
    }
    packed = {
        name: module.empty(shape, dtype=dtype)
        for name, (shape, dtype) in packed_shapes.items()
    }
    # GR_FLAG_U is a scalar class switch, not a per-column state. Read only
    # that four-byte configuration word; all spatial prognostic values stay
    # resident and are consumed by the Metal fragment.
    green_roof = int(cls["clsi"].view(
        (), byte_offset=34 * np.dtype(np.int32).itemsize).to_numpy().item())
    submit_bep_bem_fragment(
        context, "urban_bep_bem_state_gather_fragment",
        {**state_resources, **packed},
        {"ncol": ncol, "green_roof": green_roof}, grid_size=ncol)
    check_stage_error(packed["error"], "prognostic state gather")

    # WRF bb_param consumes the per-column profiles from bb_icbephi_xy and
    # the packed BEM class tables. The outputs are source-local geometry and
    # coefficient arrays; none of the persistent prognostic state is changed.
    param_shapes = {
        "nzurban": ((ncol,), np.int32),
        "ss": ((ncol, NZ_UM + 1), np.float32),
        "pb": ((ncol, NZ_UM + 1), np.float32),
        "ss_urb": ((ncol, NZ_UM + 1), np.float32),
        "pb_urb": ((ncol, NZ_UM + 1), np.float32),
        "csg": ((ncol, NG_U), np.float32),
        "alag": ((ncol, NG_U), np.float32),
        "csgb": ((ncol, NGB_U), np.float32),
        "alagb": ((ncol, NGB_U + 1), np.float32),
        "csf": ((ncol, NF_U), np.float32),
        "alaf": ((ncol, NF_U + 1), np.float32),
        "csr": ((ncol, NWR_U), np.float32),
        "alar": ((ncol, NWR_U + 1), np.float32),
        "csw": ((ncol, NWR_U), np.float32),
        "alaw": ((ncol, NWR_U + 1), np.float32),
        "z0": ((ncol, NDM, NZ_UM), np.float32),
        "strd": ((ncol, NDM), np.float32),
        "drst": ((ncol, NDM), np.float32),
        "ws": ((ncol, NDM), np.float32),
        "bs": ((ncol, NDM), np.float32),
        "ws_urb": ((ncol, NDM), np.float32),
        "bs_urb": ((ncol, NDM), np.float32),
        "dzw": ((ncol, NWR_U), np.float32),
        "dzr": ((ncol, NWR_U), np.float32),
        "dzf": ((ncol, NF_U), np.float32),
        "dzgb": ((ncol, NGB_U), np.float32),
        "error": ((ncol,), np.int32),
    }
    param = {name: module.zeros(shape, dtype=dtype)
             for name, (shape, dtype) in param_shapes.items()}
    param_resources = {
        "active_cols": active_cols,
        "utype": dev["utype_urb2d"].view(
            (ncol,), byte_offset=dev["utype_urb2d"].byte_offset),
        "class_float": class_float,
        "class_int": class_int,
        "ss_class": profiles["ss_u"],
        "pb_class": profiles["pb_u"],
        "nzu_class": profiles["nz_u"],
        "ss_urban": height_profile["ss_u"],
        "pb_urban": height_profile["pb_u"],
        "nzurb_column": height_profile["nzu_out"],
        "lp_urb": dev["lp_urb2d"].view(
            (ncol,), byte_offset=dev["lp_urb2d"].byte_offset),
        "lb_urb": dev["lb_urb2d"].view(
            (ncol,), byte_offset=dev["lb_urb2d"].byte_offset),
        "hgt_urb": dev["hgt_urb2d"].view(
            (ncol,), byte_offset=dev["hgt_urb2d"].byte_offset),
        "frc_urb": fraction.view((ncol,), byte_offset=fraction.byte_offset),
        **param,
    }
    submit_bem_param(context, param_resources, ncol=ncol, nurbm=nclass)
    check_stage_error(param["error"], "bb_param")

    # bb_icbep_xy resets the factor arrays, then evaluates bb_view_factors
    # for id=1..nd_u(utype) using this column's nzurban, strd, and ws.
    # The init_state view-factor cache uses a uniform 18-level profile and
    # therefore cannot stand in for this per-call WRF stage.
    factor = {
        "fww": module.zeros((ncol, NDM, NZ_UM, NZ_UM), dtype=np.float32),
        "fwg": module.zeros((ncol, NDM, NZ_UM), dtype=np.float32),
        "fgw": module.zeros((ncol, NDM, NZ_UM), dtype=np.float32),
        "fsw": module.zeros((ncol, NDM, NZ_UM), dtype=np.float32),
        "fws": module.zeros((ncol, NDM, NZ_UM), dtype=np.float32),
        "fsg": module.zeros((ncol, NDM), dtype=np.float32),
        "error": module.zeros((ncol, NDM), dtype=np.int32),
    }
    factor_resources = {
        "active_cols": active_cols,
        "utype": param_resources["utype"],
        "class_int": class_int,
        "nzurban": param["nzurban"],
        "z_u": profiles["z_u"],
        "strd": param["strd"],
        "ws": param["ws"],
        **factor,
    }
    submit_bem_icbep_xy(
        context, factor_resources, ncol=ncol, nurbm=nclass)
    factor_errors = factor["error"].to_numpy().reshape(ncol, NDM)
    bad_factor = np.argwhere(factor_errors)
    if bad_factor.size:
        col, direction = map(int, bad_factor[0])
        code = int(factor_errors[col, direction])
        reason = {
            6501: "invalid bb_icbep_xy extent",
            6502: "UTYPE is outside the configured BEM classes",
            6503: "nd_u or nzurban is outside WRF limits",
            6504: "street direction/width is nonpositive",
        }.get(code, f"kernel error {code}")
        raise BepBemError(
            "Metal BEP_BEM bb_icbep_xy failed at column "
            f"{col}, direction {direction + 1}: {reason}")

    # The earlier state gather also creates a compact index list, but that
    # pass precedes bb_param. Re-scan bb_param's final ss_urb profile here,
    # matching WRF's post-bb_icbep_xy order at module_sf_bep_bem.F:739-755.
    level_scan = {
        "nlev": module.zeros((ncol, NZ_UM), dtype=np.int32),
        "d_urb": module.zeros((ncol, NZ_UM), dtype=np.float32),
        "nbui": module.zeros((ncol,), dtype=np.int32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    level_scan["building_nlev"] = packed["building_nlev"]
    submit_bem_building_level_scan(
        context,
        {"active_cols": active_cols, "ss_urb": param["ss_urb"],
         **level_scan},
        ncol=ncol)
    check_stage_error(level_scan["error"], "building-level scan")

    # WRF copies the atmospheric column into its local vectors and zeros the
    # initial tendency vectors at module_sf_bep_bem.F:759-778. Reuse the
    # common BEP column record for z/u/v/theta/rho/p/pt0 and gather qv here.
    atmosphere_shapes = {
        "ua1d": ((ncol, nz), np.float32),
        "va1d": ((ncol, nz), np.float32),
        "pt1d": ((ncol, nz), np.float32),
        "da1d": ((ncol, nz), np.float32),
        "pr1d": ((ncol, nz), np.float32),
        "pt01d": ((ncol, nz), np.float32),
        "z1d": ((ncol, nz + 1), np.float32),
        "qv1d": ((ncol, nz), np.float32),
        "a_u1d": ((ncol, nz), np.float32),
        "a_v1d": ((ncol, nz), np.float32),
        "a_t1d": ((ncol, nz), np.float32),
        "a_e1d": ((ncol, nz), np.float32),
        "b_u1d": ((ncol, nz), np.float32),
        "b_v1d": ((ncol, nz), np.float32),
        "b_t1d": ((ncol, nz), np.float32),
        "b_ac1d": ((ncol, nz), np.float32),
        "b_e1d": ((ncol, nz), np.float32),
        "error": ((ncol,), np.int32),
    }
    atmosphere = {
        name: module.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in atmosphere_shapes.items()
    }
    qv_column = dev["qv_phy"].view(
        (nz, ncol), byte_offset=dev["qv_phy"].byte_offset)
    submit_bem_atmosphere_gather(
        context,
        {"active_cols": active_cols, "column_x": column_x,
         "qv": qv_column, **atmosphere},
        ncol=ncol, nz=nz)
    check_stage_error(atmosphere["error"], "atmospheric column gather")

    # bb_bep1d starts with seven ordered vertical-overlap interpolations.
    # Keep its class-specific nzu selection and k-ascending accumulation in
    # the BEM fragment rather than using the BEP ABI's differently packed
    # class record.
    atmosphere_interp = {
        "interp": module.zeros((ncol, 7, NZ_UM), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bem_atmosphere_interp(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "nzu_class": profiles["nz_u"], "z": atmosphere["z1d"],
         "z_u": profiles["z_u"], "ua": atmosphere["ua1d"],
         "va": atmosphere["va1d"], "pt": atmosphere["pt1d"],
         "pt0": atmosphere["pt01d"], "pr": atmosphere["pr1d"],
         "da": atmosphere["da1d"], "qv": atmosphere["qv1d"],
         **atmosphere_interp},
        ncol=ncol, nz=nz, nurbm=nclass)
    check_stage_error(atmosphere_interp["error"], "atmospheric interpolation")

    temperature_average = {
        "tw_av": module.zeros((ncol, 4, NZ_UM), dtype=np.float32),
        "twlev_av": module.zeros((ncol, 4, NZ_UM), dtype=np.float32),
        "sfw_av": module.zeros((ncol, 4, NZ_UM), dtype=np.float32),
        "sfwind_av": module.zeros((ncol, 4, NZ_UM), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bem_temperature_average(
        context,
        {"ss": param["ss"], "pb": param["pb"], "tw": packed["tw"],
         "twlev": packed["twlev"], "sfw": packed["sfw"],
         "sfwin": packed["sfwin"], **temperature_average},
        ncol=ncol)
    check_stage_error(temperature_average["error"], "temperature average")

    temperature_profiles = {
        "tg_av": module.zeros((ncol, NDM), dtype=np.float32),
        "tr_av": module.zeros((ncol, NZ_UM, NDM), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bem_temperature_profile_average(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "class_float": class_float, "tg": packed["tg"],
         "tr": packed["tr"], "trv": packed["trv"],
         **temperature_profiles},
        ncol=ncol)
    check_stage_error(temperature_profiles["error"],
                      "temperature profile aggregation")

    # WRF bb_modif_rad starts with direct-beam shadowing, then performs the
    # longwave and diffuse-shortwave canyon solves per active street direction.
    # Solar inputs are column fields; class coefficients are selected by UTYPE
    # inside each Metal fragment so mixed urban classes stay correctly packed.
    radiation = {
        "rsw": module.zeros((ncol, 4, NZ_UM), dtype=np.float32),
        "rsg": module.zeros((ncol, NDM), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    solar_resources = {
        "active_cols": active_cols,
        "utype": param_resources["utype"],
        "class_int": class_int,
        "nzu_class": profiles["nz_u"],
        "z_u": profiles["z_u"],
        "ws": param["ws"], "drst": param["drst"],
        "ss": param["ss"], "pb": param["pb"],
        "swddir": dev["swddir"].view((ncol,), byte_offset=dev["swddir"].byte_offset),
        "cosz": dev["cosz_urb2d"].view((ncol,), byte_offset=dev["cosz_urb2d"].byte_offset),
        "hour_angle": dev["omg_urb2d"].view((ncol,), byte_offset=dev["omg_urb2d"].byte_offset),
        "latitude": dev["xlat"].view((ncol,), byte_offset=dev["xlat"].byte_offset),
        **radiation,
    }
    submit_bem_shadow_mas(
        context, solar_resources, ncol=ncol, nurbm=nclass,
        declin=float(declin_urb))
    check_stage_error(radiation["error"], "bb_shadow_mas")

    radiation["rlw"] = module.zeros((ncol, 4, NZ_UM), dtype=np.float32)
    radiation["rlg"] = module.zeros((ncol, NDM), dtype=np.float32)
    radiation["error_lw"] = module.zeros((ncol, NDM), dtype=np.int32)
    longwave_resources = {
        "pb": param["pb"], "utype": param_resources["utype"],
        "class_int": class_int, "class_float": class_float,
        "nzu_class": profiles["nz_u"],
        "fwg": factor["fwg"], "fww": factor["fww"],
        "fgw": factor["fgw"], "fsw": factor["fsw"], "fsg": factor["fsg"],
        "tg_av": temperature_profiles["tg_av"],
        "tw": temperature_average["tw_av"],
        "twlev": temperature_average["twlev_av"],
        "rl": dev["glw"].view((ncol,), byte_offset=dev["glw"].byte_offset),
        "rlg": radiation["rlg"], "rlw": radiation["rlw"],
        "error": radiation["error_lw"],
    }
    submit_bem_longwave(
        context, longwave_resources, ncol=ncol, nurbm=nclass,
        sigma=5.67e-8)
    check_stage_error(radiation["error_lw"], "bb_long_rad")

    radiation["error_sw"] = module.zeros((ncol, NDM), dtype=np.int32)
    shortwave_resources = {
        "pb": param["pb"], "utype": param_resources["utype"],
        "class_int": class_int, "class_float": class_float,
        "nzu_class": profiles["nz_u"],
        "fwg": factor["fwg"], "fww": factor["fww"],
        "fgw": factor["fgw"], "fsw": factor["fsw"], "fsg": factor["fsg"],
        "rsw": radiation["rsw"], "rsg": radiation["rsg"],
        "rsdif": dev["swddif"].view((ncol,), byte_offset=dev["swddif"].byte_offset),
        "error": radiation["error_sw"],
    }
    submit_bem_shortwave(
        context, shortwave_resources, ncol=ncol, nurbm=nclass)
    check_stage_error(radiation["error_sw"], "bb_short_rad_dd")

    upward = {
        "rs_abs": module.zeros((ncol,), dtype=np.float32),
        "rl_up": module.zeros((ncol,), dtype=np.float32),
        "emiss": module.zeros((ncol,), dtype=np.float32),
        "grdflx_urb": module.zeros((ncol,), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    upward_resources = {
        "active_cols": active_cols,
        "ws": param["ws"], "bs": param["bs"],
        "pb": param["pb"], "ss": param["ss"],
        "tg_av": temperature_profiles["tg_av"],
        "rlg": radiation["rlg"], "rsg": radiation["rsg"],
        "sfg": packed["sfg"], "lfg": packed["lfg"],
        "tw": temperature_average["tw_av"],
        "rlw": radiation["rlw"], "rsw": radiation["rsw"],
        "sfw": temperature_average["sfw_av"],
        "tr_av": temperature_profiles["tr_av"],
        "twlev": temperature_average["twlev_av"],
        "sfwind": temperature_average["sfwind_av"],
        "tpvlev": packed["tpv"],
        "sfr": packed["sfr"], "sfrv": packed["sfrv"],
        "lfr": packed["lfr"], "lfrv": packed["lfrv"],
        "rld": dev["glw"].view((ncol,), byte_offset=dev["glw"].byte_offset),
        "rs": dev["swdown"].view((ncol,), byte_offset=dev["swdown"].byte_offset),
        "utype": param_resources["utype"], "class_int": class_int,
        "class_float": class_float, "nzu_class": profiles["nz_u"],
        **upward,
    }
    submit_bem_upward_rad(
        context, upward_resources, nitem=ncol, nurbm=nclass,
        sigma=5.67e-8, dz_u=DZ_U)
    check_stage_error(upward["error"], "bb_upward_rad")

    rain_update = {"error": module.zeros((ncol,), dtype=np.int32)}
    submit_bem_rain_update(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "class_int": class_int,
         "rainbl": dev["rainbl"].view((ncol,), byte_offset=dev["rainbl"].byte_offset),
         "lfg": packed["lfg"], "lfr": packed["lfr"],
         "dg": packed["dg"], "dgr": packed["dgr"],
         **rain_update},
        ncol=ncol, nurbm=nclass, dt=float(dt), latent=2.45e6)
    check_stage_error(rain_update["error"], "dg/dgr rain update")

    ground_temperature = {
        "ptg": module.zeros((ncol, NDM), dtype=np.float32),
        "gfg": module.zeros((ncol, NDM), dtype=np.float32),
        "error": module.zeros((ncol, NDM), dtype=np.int32),
    }
    submit_bem_surface_temp(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "class_int": class_int, "class_float": class_float,
         "alag": param["alag"], "csg": param["csg"],
         "interp": atmosphere_interp["interp"],
         "rsg": radiation["rsg"], "rlg": radiation["rlg"],
         "tg": packed["tg"],
         "ptg": ground_temperature["ptg"],
         "sfg": packed["sfg"], "lfg": packed["lfg"],
         "gfg": ground_temperature["gfg"],
         "error": ground_temperature["error"]},
        ncol=ncol, nurbm=nclass, dt=float(dt), sigma=5.67e-8)
    check_stage_error(ground_temperature["error"], "bb_surf_temp")

    # WRF bb_roof_temp_veg follows bb_surf_temp. It advances the 10-layer
    # green-roof heat and moisture columns before bb_flux_flat consumes their
    # surface temperature/fluxes. The class irrigation schedule is selected
    # from the local solar hour angle, matching bb_bep1d's `irho` lookup.
    green_roof_state = {
        "ptrv": module.zeros((ncol, NZ_UM, NDM), dtype=np.float32),
        "hfgroof": module.zeros((ncol, NZ_UM, NDM), dtype=np.float32),
        "gfr": module.zeros((ncol, NZ_UM, NDM), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bep_bem_fragment(
        context, "urban_bep_bem_green_roof_fragment",
        {"active_cols": active_cols,
         "utype": param_resources["utype"],
         "nzurban": param["nzurban"], "class_float": class_float,
         "class_int": class_int,
         "hour_angle": dev["omg_urb2d"].view(
             (ncol,), byte_offset=dev["omg_urb2d"].byte_offset),
         "interp": atmosphere_interp["interp"],
         "alar": param["alar"], "csr": param["csr"],
         "dzr": param["dzr"], "tr": packed["tr"],
         "trv": packed["trv"], "qr": packed["qr"],
         "drain": packed["drain"], "sfrv": packed["sfrv"],
         "lfrv": packed["lfrv"], "gfr": green_roof_state["gfr"],
         "tpv": packed["tpv"],
         "rainbl": dev["rainbl"].view(
             (ncol,), byte_offset=dev["rainbl"].byte_offset),
         "swdown": dev["swdown"].view(
             (ncol,), byte_offset=dev["swdown"].byte_offset),
         "glw": dev["glw"].view(
             (ncol,), byte_offset=dev["glw"].byte_offset),
         **green_roof_state},
        {"ncol": ncol, "dt": float(dt),
         "green_roof": green_roof}, grid_size=ncol)
    check_stage_error(green_roof_state["error"], "bb_roof_temp_veg")

    # bb_flux_flat computes the real column ground/roof sensible and latent
    # terms. Repack its direction-major output before the BEM per-building
    # call, whose persistent state uses level-major, then direction order.
    flat_flux = {
        name: module.zeros((ncol, NDM, NZ_UM), dtype=np.float32)
        for name in ("uhb", "vhb", "sf", "lf", "ehb", "thb", "qhb")
    }
    flat_flux["error"] = module.zeros((ncol, NDM, NZ_UM), dtype=np.int32)
    submit_bem_flat_flux(
        context,
        {"active_cols": active_cols, "interp": atmosphere_interp["interp"],
         "z0": param["z0"], "ss": param["ss"],
         "ptg": ground_temperature["ptg"], "tr": packed["tr"],
         "dg": packed["dg"], "dgr": packed["dgr"], **flat_flux},
        ncol=ncol, g_u=9.81, vk=0.4, p0=100000.0,
        rcp_u=287.0 / 1004.0, cp_u=1004.0, latent=2.45e6,
        dz_u=DZ_U, dgmax=1.0, drmax=1.0)
    check_stage_error(flat_flux["error"], "bb_flux_flat")
    roof_flux = {
        "hsrof": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "lsrof": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bem_flat_flux_pack(
        context,
        {"active_cols": active_cols,
         "building_nlev": level_scan["building_nlev"],
         "sf": flat_flux["sf"], "lf": flat_flux["lf"],
         "sfr": packed["sfr"], "lfr": packed["lfr"],
         "sfg": packed["sfg"], "lfg": packed["lfg"], **roof_flux},
        ncol=ncol)
    check_stage_error(roof_flux["error"], "BEM flat flux repack")

    # bem_fluxeqocc is a pure room-gain calculation. Pack its per-class
    # inputs and evaluate all possible building slots; the indoor solver will
    # consume only the active nlev entries when its complete stage chain is
    # wired. WRF's hour-angle expression reduces to GMT, with one wrap at 24.
    indoor_nitem = ncol * NBUI_MAX
    occupancy_inputs = {
        "hsequip": module.zeros((indoor_nitem, 24), dtype=np.float32),
        "building_width": module.zeros((indoor_nitem,), dtype=np.float32),
        "building_length": module.zeros((indoor_nitem,), dtype=np.float32),
        "floor_fraction": module.zeros((indoor_nitem,), dtype=np.float32),
        "equipment_scale": module.zeros((indoor_nitem,), dtype=np.float32),
        "error": module.zeros((indoor_nitem,), dtype=np.int32),
    }
    submit_bem_occupancy_inputs(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "class_float": class_float, "bs": param["bs"], **occupancy_inputs},
        ncol=ncol, class_float_len=int(class_float.shape[0]))
    check_stage_error(occupancy_inputs["error"], "BEM occupancy input select")
    nhourday = float(gmt)
    if nhourday >= 24.0:
        nhourday -= 24.0
    if nhourday < 0.0:
        nhourday += 24.0
    occupancy = {
        "sensible_gain": module.zeros((indoor_nitem,), dtype=np.float32),
        "latent_gain": module.zeros((indoor_nitem,), dtype=np.float32),
        "error": module.zeros((indoor_nitem,), dtype=np.int32),
    }
    submit_bem_occupancy(
        context,
        {**occupancy_inputs, **occupancy},
        nitem=indoor_nitem, nhourday=nhourday)
    check_stage_error(occupancy["error"], "bem_fluxeqocc")

    # WRF calls bem_int_rsrad/bem_int_rlrad once for every active building
    # level inside bem_bem. The Metal stage gathers the exact packed wall,
    # window, floor/ground/roof temperatures and class coefficients, then
    # writes six radiosities per (column, building slot, level).
    indoor_rad_nitem = ncol * NBUI_MAX * NZ_UM
    indoor_radiation = {
        "rsint": module.zeros((indoor_rad_nitem, 6), dtype=np.float32),
        "rlint": module.zeros((indoor_rad_nitem, 6), dtype=np.float32),
        "error": module.zeros((indoor_rad_nitem,), dtype=np.int32),
    }
    submit_bem_indoor_radiation(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "building_nlev": level_scan["building_nlev"],
         "class_float": class_float, "bs": param["bs"],
         "rsw": radiation["rsw"], "tw": packed["tw"],
         "twlev": packed["twlev"], "tflev": packed["tflev"],
         "tglev": packed["tglev"], "tr": packed["tr"],
         **indoor_radiation},
        ncol=ncol, nitem=indoor_rad_nitem, nurbm=nclass,
        class_float_len=int(class_float.shape[0]), dz_u=DZ_U)
    check_stage_error(indoor_radiation["error"], "bem_int_rsrad/bem_int_rlrad")

    indoor_fluxes = {
        "hrwalout": module.zeros((indoor_rad_nitem, 4), dtype=np.float32),
        "hrwinout": module.zeros((indoor_rad_nitem, 4), dtype=np.float32),
        "hrrof": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "hrwalins": module.zeros((indoor_rad_nitem, 6), dtype=np.float32),
        "hrwinins": module.zeros((indoor_rad_nitem, 4), dtype=np.float32),
        "hswalins": module.zeros((indoor_rad_nitem, 6), dtype=np.float32),
        "hswinins": module.zeros((indoor_rad_nitem, 4), dtype=np.float32),
        "surwal": module.zeros((indoor_rad_nitem, 6), dtype=np.float32),
        "hscond": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "error": module.zeros((indoor_rad_nitem,), dtype=np.int32),
    }
    submit_bem_indoor_fluxes(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "building_nlev": level_scan["building_nlev"],
         "class_float": class_float, "bs": param["bs"],
         "rsw": radiation["rsw"], "rlw": radiation["rlw"],
         "rsint": indoor_radiation["rsint"], "rlint": indoor_radiation["rlint"],
         "tw": packed["tw"], "twlev": packed["twlev"],
         "tflev": packed["tflev"], "tglev": packed["tglev"],
         "tr": packed["tr"], "tlev": packed["tlev"],
         "tr_av": temperature_profiles["tr_av"],
         "swdown": dev["swdown"].view((ncol,), byte_offset=dev["swdown"].byte_offset),
         "glw": dev["glw"].view((ncol,), byte_offset=dev["glw"].byte_offset),
         **indoor_fluxes},
        ncol=ncol, nitem=indoor_rad_nitem, nurbm=nclass,
        class_float_len=int(class_float.shape[0]), dz_u=DZ_U)
    check_stage_error(indoor_fluxes["error"], "bem_radfluxs/bem_hsinsflux/bem_fluxcond")

    # bb_flux_wall supplies the exterior sensible wall/window terms consumed
    # by bem_bem's material matrices. Preserve the returned surface fluxes in
    # the packed state so the final inverse scatter commits the WRF fields.
    wall_flux = {
        name: module.zeros((ncol, NBUI_MAX, NZ_UM, 4), dtype=np.float32)
        for name in ("uva", "vva", "uvb", "vvb", "sfw", "sfwin", "evb")
    }
    wall_flux["error"] = module.zeros(
        (ncol, NBUI_MAX, NZ_UM, 4), dtype=np.int32)
    submit_bem_wall_flux(
        context,
        {"active_cols": active_cols,
         "building_nlev": level_scan["building_nlev"],
         "interp": atmosphere_interp["interp"], "tw": packed["tw"],
         "twlev": packed["twlev"], "pb": param["pb"],
         "bs": param["bs"], "ws": param["ws"], "drst": param["drst"],
         **wall_flux},
        ncol=ncol, p0=100000.0, rcp_u=287.0 / 1004.0,
        cp_u=1004.0, dt=float(dt))
    check_stage_error(wall_flux["error"], "bb_flux_wall")
    packed["sfw"] = wall_flux["sfw"]
    packed["sfwin"] = wall_flux["sfwin"]

    material = {
        "gswal": module.zeros(
            (ncol, NBUI_MAX, NZ_UM, 4), dtype=np.float32),
        "gsrof": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "error": module.zeros((ncol, NBUI_MAX), dtype=np.int32),
    }
    material_resources = {
        "active_cols": active_cols,
        "building_nlev": level_scan["building_nlev"],
        "dzw": param["dzw"], "csw": param["csw"], "alaw": param["alaw"],
        "dzf": param["dzf"], "csf": param["csf"], "alaf": param["alaf"],
        "dzgb": param["dzgb"], "csgb": param["csgb"], "alagb": param["alagb"],
        "dzr": param["dzr"], "csr": param["csr"], "alar": param["alar"],
        "hswalins": indoor_fluxes["hswalins"],
        "hrwalins": indoor_fluxes["hrwalins"],
        "hswalout": wall_flux["sfw"], "hrwalout": indoor_fluxes["hrwalout"],
        "hswinins": indoor_fluxes["hswinins"],
        "hrwinins": indoor_fluxes["hrwinins"],
        "hswinout": wall_flux["sfwin"], "hrwinout": indoor_fluxes["hrwinout"],
        "hsrof": roof_flux["hsrof"], "hrrof": indoor_fluxes["hrrof"],
        "lsrof": roof_flux["lsrof"],
        "tw": packed["tw"], "twlev": packed["twlev"],
        "tflev": packed["tflev"], "tglev": packed["tglev"],
        "tr": packed["tr"], **material,
    }
    material_params = {"ncol": ncol, "dt": float(dt)}
    material_kernel = "urban_bem_material_solve_fragment"
    if green_roof:
        material_kernel = "urban_bem_material_solve_green_roof_fragment"
        material_resources.update({
            "utype": param_resources["utype"],
            "class_float": class_float,
            "hfgroof": green_roof_state["hfgroof"],
        })
        material_params["green_roof"] = green_roof
    submit_bep_bem_fragment(
        context, material_kernel, material_resources, material_params,
        grid_size=ncol * NBUI_MAX)
    check_stage_error(material["error"], "bem_bem material solve")

    room_inputs = {
        name: module.zeros((indoor_rad_nitem,), dtype=np.float32)
        for name in (
            "cpint", "rhoint", "vollev", "tlev", "tout", "latent",
            "humout", "rhoout", "humlev", "beta", "qb", "hsroo",
            "volroo", "hlroo", "shumroo", "cop", "timeon", "timeoff",
            "targtemp", "gaptemp", "targhum", "gaphum")
    }
    room_inputs["swcond"] = module.zeros((indoor_rad_nitem,), dtype=np.int32)
    room_inputs["error"] = module.zeros((indoor_rad_nitem,), dtype=np.int32)
    submit_bem_room_balance_inputs(
        context,
        {"active_cols": active_cols, "utype": param_resources["utype"],
         "building_nlev": level_scan["building_nlev"],
         "class_int": class_int, "class_float": class_float,
         "bs": param["bs"], "tlev_src": packed["tlev"],
         "hum_src": packed["qlev"], "interp": atmosphere_interp["interp"],
         "occupancy_sensible": occupancy["sensible_gain"],
         "occupancy_latent": occupancy["latent_gain"],
         "hscond": indoor_fluxes["hscond"], **room_inputs},
        ncol=ncol, nitem=indoor_rad_nitem, nurbm=nclass,
        class_float_len=int(class_float.shape[0]),
        class_int_len=int(class_int.shape[0]), dz_u=DZ_U)
    check_stage_error(room_inputs["error"], "BEM room-balance input gather")
    room_control = {
        **{name: array for name, array in room_inputs.items()
           if name != "error"},
        "hsvent": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "hlvent": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "hsneed_out": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "hlneed_out": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "hsout": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "hlout": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "consump": module.zeros((indoor_rad_nitem,), dtype=np.float32),
    }
    submit_bem_room_control(
        context, room_control, nitem=indoor_rad_nitem,
        nhourday=nhourday, dt=float(dt))

    # These four per-building profiles are bem_bem's source outputs at
    # module_sf_bep_bem.F:1749-1753. Keep the exact stage results as views:
    # hsroo/hlroo are bem_fluxroo's net room sensible/latent fluxes, and
    # hsvent/hlvent are bem_fluxvent's sensible/latent ventilation fluxes.
    # The later bb_buildings tendency stage consumes all four profiles.
    bem_flux_profiles = {
        "sflev": room_inputs["hsroo"].view(
            (ncol, NBUI_MAX, NZ_UM),
            byte_offset=room_inputs["hsroo"].byte_offset),
        "lflev": room_inputs["hlroo"].view(
            (ncol, NBUI_MAX, NZ_UM),
            byte_offset=room_inputs["hlroo"].byte_offset),
        "sfvlev": room_control["hsvent"].view(
            (ncol, NBUI_MAX, NZ_UM),
            byte_offset=room_control["hsvent"].byte_offset),
        "lfvlev": room_control["hlvent"].view(
            (ncol, NBUI_MAX, NZ_UM),
            byte_offset=room_control["hlvent"].byte_offset),
    }

    staged_indoor_state = {
        "tlev_next": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "humlev_next": module.zeros((indoor_rad_nitem,), dtype=np.float32),
        "error": module.zeros((indoor_rad_nitem,), dtype=np.int32),
    }
    submit_bem_indoor_state_update(
        context,
        {"active_cols": active_cols,
         "building_nlev": level_scan["building_nlev"],
         "tlev": room_inputs["tlev"], "humlev": room_inputs["humlev"],
         "qb": room_inputs["qb"], "vollev": room_inputs["vollev"],
         "rhoint": room_inputs["rhoint"], "latent": room_inputs["latent"],
         "hsroo": room_inputs["hsroo"], "hlroo": room_inputs["hlroo"],
         "hsneed": room_control["hsneed_out"],
         "hlneed": room_control["hlneed_out"],
         **staged_indoor_state},
        ncol=ncol, nitem=indoor_rad_nitem, dt=float(dt))
    check_stage_error(staged_indoor_state["error"], "BEM indoor state update")

    building_items = ncol * NBUI_MAX
    totals = {
        "hsoutbuild": module.zeros((building_items,), dtype=np.float32),
        "consumpbuild": module.zeros((building_items,), dtype=np.float32),
        "error": module.zeros((building_items,), dtype=np.int32),
    }
    submit_bem_consump_total(
        context,
        {"hsout": room_control["hsout"].view((building_items, NZ_UM)),
         "consump": room_control["consump"].view((building_items, NZ_UM)),
         "nlev": level_scan["building_nlev"].view((building_items,)),
         **totals},
        nitem=building_items, nzcanm=NZ_UM)
    check_stage_error(totals["error"], "bem_consump_total")

    # bb_buildings runs after bem_bem has updated roof/material temperatures.
    # Recompute its flat fluxes in that source position; the earlier flat pass
    # supplies the material-solver boundary condition and is intentionally not
    # reused for the post-solve atmospheric tendencies.
    post_solve_flat_flux = {
        name: module.zeros((ncol, NDM, NZ_UM), dtype=np.float32)
        for name in ("uhb", "vhb", "sf", "lf", "ehb", "thb", "qhb")
    }
    post_solve_flat_flux["error"] = module.zeros(
        (ncol, NDM, NZ_UM), dtype=np.int32)
    submit_bem_flat_flux(
        context,
        {"active_cols": active_cols, "interp": atmosphere_interp["interp"],
         "z0": param["z0"], "ss": param["ss"],
         "ptg": ground_temperature["ptg"], "tr": packed["tr"],
         "dg": packed["dg"], "dgr": packed["dgr"], **post_solve_flat_flux},
        ncol=ncol, g_u=9.81, vk=0.4, p0=100000.0,
        rcp_u=287.0 / 1004.0, cp_u=1004.0, latent=2.45e6,
        dz_u=DZ_U, dgmax=1.0, drmax=1.0)
    check_stage_error(post_solve_flat_flux["error"], "bb_buildings flat flux")

    # bem_hsfluxpv runs inside bem_bem before bb_buildings. Keep its panel
    # temperature and energy on device and add its sensible contribution to
    # the post-solve flat tendency consumed by the next Metal stage.
    pv_profiles = {
        "pv_energy": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "pv_hspv": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "sfrpv": module.zeros((ncol, NZ_UM, NDM), dtype=np.float32),
    }
    submit_bep_bem_fragment(
        context, "urban_bep_bem_pv_solve_fragment",
        {"active_cols": active_cols,
         "utype": param_resources["utype"],
         "building_nlev": level_scan["building_nlev"],
         "class_float": class_float,
         "interp": atmosphere_interp["interp"],
         "tr_av": temperature_profiles["tr_av"],
         "tpv": packed["tpv"],
         "swdown": dev["swdown"].view(
             (ncol,), byte_offset=dev["swdown"].byte_offset),
         "glw": dev["glw"].view(
             (ncol,), byte_offset=dev["glw"].byte_offset),
         "flat_thb": post_solve_flat_flux["thb"],
         **pv_profiles},
        {"ncol": ncol, "dt": float(dt), "cp_u": 1004.0},
        grid_size=ncol)

    submit_bep_bem_fragment(
        context, "urban_bep_bem_green_roof_pv_map_fragment",
        {"active_cols": active_cols,
         "utype": param_resources["utype"],
         "building_nlev": level_scan["building_nlev"],
         "class_float": class_float,
         "interp": atmosphere_interp["interp"],
         "pv_hspv": pv_profiles["pv_hspv"],
         "flat_thb": post_solve_flat_flux["thb"],
         "sfrpv": pv_profiles["sfrpv"]},
        {"ncol": ncol, "cp_u": 1004.0}, grid_size=ncol)

    post_solve_roof_flux = {
        "hsrof": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "lsrof": module.zeros((ncol, NBUI_MAX), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bem_flat_flux_pack(
        context,
        {"active_cols": active_cols,
         "building_nlev": level_scan["building_nlev"],
         "sf": post_solve_flat_flux["sf"], "lf": post_solve_flat_flux["lf"],
         "sfr": packed["sfr"], "lfr": packed["lfr"],
         "sfg": packed["sfg"], "lfg": packed["lfg"],
         **post_solve_roof_flux},
        ncol=ncol)
    check_stage_error(post_solve_roof_flux["error"], "bb_buildings roof flux pack")

    # The wall-flux portion of bb_buildings follows bem_bem too, so consume
    # the updated wall/window temperatures from the material solve.
    submit_bem_wall_flux(
        context,
        {"active_cols": active_cols,
         "building_nlev": level_scan["building_nlev"],
         "interp": atmosphere_interp["interp"], "tw": packed["tw"],
         "twlev": packed["twlev"], "pb": param["pb"],
         "bs": param["bs"], "ws": param["ws"], "drst": param["drst"],
         **wall_flux},
        ncol=ncol, p0=100000.0, rcp_u=287.0 / 1004.0,
        cp_u=1004.0, dt=float(dt))
    check_stage_error(wall_flux["error"], "bb_buildings wall flux")
    packed["sfw"] = wall_flux["sfw"]
    packed["sfwin"] = wall_flux["sfwin"]

    # bb_buildings runs bb_flux_flat_roof for each occupied urban level after
    # bem_hsfluxpv. Pack the soil-layer moisture into its row-oriented ABI,
    # call the existing per-level roof-flux fragment, then merge only the
    # active levels (WRF skips levels where ss(iz)==0).
    roof_nitem = ncol * NZ_UM * NDM
    roof_flux_items = {
        name: module.zeros((roof_nitem,), dtype=np.float32)
        for name in ("dz", "z0", "ua", "va", "pt", "pt0", "ptg", "da",
                     "qv", "pr", "rsg", "pv_frac_roof", "uhb", "vhb",
                     "sf", "lf", "ehb", "resg", "rsveg", "f1", "f2",
                     "f3", "f4")
    }
    roof_flux_items["gr_type"] = module.zeros((roof_nitem,), dtype=np.int32)
    roof_flux_items["qr_rows"] = module.zeros(
        (roof_nitem, NGR_U), dtype=np.float32)
    roof_flux_items["error"] = module.zeros((roof_nitem,), dtype=np.int32)
    roof_pack_outputs = {
        name: roof_flux_items[name]
        for name in ("dz", "z0", "ua", "va", "pt", "pt0", "ptg", "da",
                     "qv", "pr", "rsg", "pv_frac_roof", "gr_type",
                     "qr_rows", "sf", "lf")
    }
    submit_bep_bem_fragment(
        context, "urban_bep_bem_green_roof_flux_pack_fragment",
        {"utype": param_resources["utype"], "class_float": class_float,
         "class_int": class_int, "interp": atmosphere_interp["interp"],
         "ptrv": green_roof_state["ptrv"],
         "swdown": dev["swdown"].view(
             (ncol,), byte_offset=dev["swdown"].byte_offset),
         "qr": packed["qr"], "sfrv": packed["sfrv"],
         "lfrv": packed["lfrv"], **roof_pack_outputs},
        {"nitem": roof_nitem, "ncol": ncol, "dz_u": DZ_U},
        grid_size=roof_nitem)
    submit_bep_bem_fragment(
        context, "urban_bep_bem_flat_roof_flux_fragment",
        {"dz": roof_flux_items["dz"], "z0": roof_flux_items["z0"],
         "ua": roof_flux_items["ua"], "va": roof_flux_items["va"],
         "pt": roof_flux_items["pt"], "pt0": roof_flux_items["pt0"],
         "ptg": roof_flux_items["ptg"], "da": roof_flux_items["da"],
         "qv": roof_flux_items["qv"], "pr": roof_flux_items["pr"],
         "rsg": roof_flux_items["rsg"], "qr": roof_flux_items["qr_rows"],
         "pv_frac_roof": roof_flux_items["pv_frac_roof"],
         "gr_type": roof_flux_items["gr_type"],
         "uhb": roof_flux_items["uhb"], "vhb": roof_flux_items["vhb"],
         "sf": roof_flux_items["sf"], "lf": roof_flux_items["lf"],
         "ehb": roof_flux_items["ehb"], "resg": roof_flux_items["resg"],
         "rsveg": roof_flux_items["rsveg"], "f1": roof_flux_items["f1"],
         "f2": roof_flux_items["f2"], "f3": roof_flux_items["f3"],
         "f4": roof_flux_items["f4"], "error": roof_flux_items["error"]},
        {"nitem": roof_nitem, "g_u": 9.81, "vk": 0.4,
         "p0": 100000.0, "rcp_u": 287.0 / 1004.0,
         "cp_u": 1004.0, "latent": 2.45e6}, grid_size=roof_nitem)
    check_stage_error(roof_flux_items["error"], "bb_flux_flat_roof")
    submit_bep_bem_fragment(
        context, "urban_bep_bem_green_roof_flux_merge_fragment",
        {"active_cols": active_cols, "nzurban": param["nzurban"],
         "utype": param_resources["utype"], "class_float": class_float,
         "ss": param["ss"], "sf": roof_flux_items["sf"],
         "lf": roof_flux_items["lf"], "sfrv": packed["sfrv"],
         "lfrv": packed["lfrv"]},
        {"nitem": roof_nitem, "ncol": ncol,
         "green_roof": green_roof}, grid_size=roof_nitem)

    # WRF bb_buildings consumes the four real bem_bem flux profiles above,
    # the wall-flux kernel outputs, and the ordered flat-surface flux results.
    # It emits the arrays consumed by bb_urban_meso; no domain state is written.
    tendency_shapes = {
        name: ((ncol, 4, NZ_UM) if name in (
            "uva", "vva", "uvb", "vvb", "tvb", "tvb_ac", "evb", "qvb")
               else (ncol, NDM, NZ_UM))
        for name in (
            "uva", "vva", "uvb", "vvb", "tvb", "tvb_ac", "evb", "qvb",
            "uhb", "vhb", "thb", "ehb", "qhb_flat")
    }
    building_tendencies = {
        name: module.zeros(shape, dtype=np.float32)
        for name, shape in tendency_shapes.items()
    }
    building_tendencies["error"] = module.zeros(
        (ncol, NDM, NZ_UM), dtype=np.int32)
    submit_bem_buildings_tendency(
        context,
        {"active_cols": active_cols,
         "building_nlev": level_scan["building_nlev"],
         "utype": param_resources["utype"], "d_urb": level_scan["d_urb"],
         "pb": param["pb"], "bs": param["bs"], "class_float": class_float,
         "interp": atmosphere_interp["interp"], **bem_flux_profiles,
         "wall_uva": wall_flux["uva"], "wall_vva": wall_flux["vva"],
         "wall_uvb": wall_flux["uvb"], "wall_vvb": wall_flux["vvb"],
         "wall_sfw": wall_flux["sfw"], "wall_sfwin": wall_flux["sfwin"],
         "wall_evb": wall_flux["evb"],
         "flat_uhb": post_solve_flat_flux["uhb"],
         "flat_vhb": post_solve_flat_flux["vhb"],
         "flat_thb": post_solve_flat_flux["thb"],
         "flat_ehb": post_solve_flat_flux["ehb"],
         "flat_qhb": post_solve_flat_flux["qhb"], **building_tendencies},
        ncol=ncol, nurbm=nclass,
        class_float_len=int(class_float.shape[0]),
        cp_u=1004.0, latent=2.45e6, dz_u=DZ_U)
    check_stage_error(building_tendencies["error"], "bb_buildings tendencies")

    submit_bep_bem_fragment(
        context, "urban_bep_bem_green_roof_tendency_fragment",
        {"active_cols": active_cols,
         "utype": param_resources["utype"], "class_float": class_float,
         "ss": param["ss"], "interp": atmosphere_interp["interp"],
         "sfrv": packed["sfrv"], "lfrv": packed["lfrv"],
         "sfrpv": pv_profiles["sfrpv"],
         "flat_uhb": post_solve_flat_flux["uhb"],
         "flat_vhb": post_solve_flat_flux["vhb"],
         "flat_thb": post_solve_flat_flux["thb"],
         "flat_ehb": post_solve_flat_flux["ehb"],
         "flat_qhb": post_solve_flat_flux["qhb"],
         "roof_uhb": roof_flux_items["uhb"],
         "roof_vhb": roof_flux_items["vhb"],
         "roof_ehb": roof_flux_items["ehb"],
         "uhb": building_tendencies["uhb"],
         "vhb": building_tendencies["vhb"],
         "thb": building_tendencies["thb"],
         "ehb": building_tendencies["ehb"],
         "qhb_flat": building_tendencies["qhb_flat"]},
        {"ncol": ncol, "nitem": roof_nitem,
         "green_roof": green_roof, "cp_u": 1004.0,
         "latent": 2.45e6}, grid_size=roof_nitem)

    # bb_urban_meso consumes the completed per-building tendencies and maps
    # them onto the original atmospheric mass/edge levels.
    meso = {
        "sf": module.zeros((ncol, nz + 1), dtype=np.float32),
        "vl": module.zeros((ncol, nz), dtype=np.float32),
        # bb_bep1d initializes b_q1d to zero and does not change it.
        "b_q": module.zeros((ncol, nz), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    for source, target in (
        ("a_u1d", "a_u"), ("a_v1d", "a_v"), ("a_t1d", "a_t"),
        ("a_e1d", "a_e"), ("b_u1d", "b_u"), ("b_v1d", "b_v"),
        ("b_t1d", "b_t"), ("b_e1d", "b_e"),
        ("b_ac1d", "b_ac")):
        meso[target] = atmosphere[source]
    meso_resources = {
        "active_cols": active_cols, "z": atmosphere["z1d"],
        "dz": dev["dz8w"].view((nz, ncol), byte_offset=dev["dz8w"].byte_offset),
        "z_u": profiles["z_u"], "pb": param["pb"], "ss": param["ss"],
        "bs": param["bs"], "ws": param["ws"],
        "uva": building_tendencies["uva"],
        "vva": building_tendencies["vva"],
        "uvb": building_tendencies["uvb"],
        "vvb": building_tendencies["vvb"],
        "tvb": building_tendencies["tvb"],
        "tvb_ac": building_tendencies["tvb_ac"],
        "evb": building_tendencies["evb"],
        "uhb": building_tendencies["uhb"],
        "vhb": building_tendencies["vhb"],
        "thb": building_tendencies["thb"],
        "ehb": building_tendencies["ehb"],
        "qhb": building_tendencies["qhb_flat"],
        "qvb": building_tendencies["qvb"],
        **meso,
    }
    submit_bem_urban_meso(context, meso_resources, ncol=ncol, nz=nz)
    check_stage_error(meso["error"], "bb_urban_meso")

    # WRF derives the urban and building drag length scales immediately after
    # bb_urban_meso; final PBL aggregation consumes these per-column profiles.
    length_scales = {
        "dlg": module.zeros((ncol, nz), dtype=np.float32),
        "dl_u": module.zeros((ncol, nz), dtype=np.float32),
        "error": module.zeros((ncol,), dtype=np.int32),
    }
    submit_bem_interp_length(
        context,
        {"active_cols": active_cols, "z": atmosphere["z1d"],
         "z_u": profiles["z_u"], "ss": param["ss"],
         "ws": param["ws"], "bs": param["bs"], **length_scales},
        ncol=ncol, nz=nz)
    check_stage_error(length_scales["error"], "bb_interp_length")

    def domain_view(name, shape):
        array = dev[name]
        return array.view(shape, byte_offset=array.byte_offset)

    final_outputs = {
        "active_cols": active_cols,
        "fraction": fraction.view((ncol,), byte_offset=fraction.byte_offset),
        "utype": param_resources["utype"], "class_float": class_float,
        "bs": param["bs"], "ws": param["ws"],
        "building_nlev": level_scan["building_nlev"],
        "d_urb": level_scan["d_urb"],
        "sflev": bem_flux_profiles["sflev"], "lflev": bem_flux_profiles["lflev"],
        "consum": room_control["consump"].view((ncol, NBUI_MAX, NZ_UM)),
        "sfvlev": bem_flux_profiles["sfvlev"], "lfvlev": bem_flux_profiles["lfvlev"],
        "sf": meso["sf"], "vl": meso["vl"],
        "a_u": atmosphere["a_u1d"], "a_v": atmosphere["a_v1d"],
        "a_t": atmosphere["a_t1d"], "a_e": atmosphere["a_e1d"],
        "b_u": atmosphere["b_u1d"], "b_v": atmosphere["b_v1d"],
        "b_t": atmosphere["b_t1d"], "b_e": atmosphere["b_e1d"],
        "b_q": meso["b_q"], "dlg": length_scales["dlg"],
        "dl_u": length_scales["dl_u"],
        "rl_up_in": upward["rl_up"], "rs_abs_in": upward["rs_abs"],
        "emiss_in": upward["emiss"], "grdflx_in": upward["grdflx_urb"],
        "sf_ac": domain_view("sf_ac_urb3d", (ncol,)),
        "lf_ac": domain_view("lf_ac_urb3d", (ncol,)),
        "cm_ac": domain_view("cm_ac_urb3d", (ncol,)),
        "sf_vent": domain_view("sfvent_urb3d", (ncol,)),
        "lf_vent": domain_view("lfvent_urb3d", (ncol,)),
        "ep_pv": domain_view("ep_pv_urb3d", (ncol,)),
        "drain_green": domain_view("draingr_urb3d", (ncol,)),
        "qgr": domain_view("qgr_urb3d", (ncol,)),
        "tgr": domain_view("tgr_urb3d", (ncol,)),
        "rl_up": domain_view("rl_up", (ncol,)),
        "rs_abs": domain_view("rs_abs", (ncol,)),
        "emiss": domain_view("emiss", (ncol,)),
        "grdflx": domain_view("grdflx_urb", (ncol,)),
        "out_sf": domain_view("sf", (nz + 1, ncol)),
        "out_vl": domain_view("vl", (nz, ncol)),
        "out_a_u": domain_view("a_u", (nz, ncol)),
        "out_a_v": domain_view("a_v", (nz, ncol)),
        "out_a_t": domain_view("a_t", (nz, ncol)),
        "out_a_e": domain_view("a_e", (nz, ncol)),
        "out_b_u": domain_view("b_u", (nz, ncol)),
        "out_b_v": domain_view("b_v", (nz, ncol)),
        "out_b_t": domain_view("b_t", (nz, ncol)),
        "out_b_e": domain_view("b_e", (nz, ncol)),
        "out_b_q": domain_view("b_q", (nz, ncol)),
        "out_dlg": domain_view("dlg", (nz, ncol)),
        "out_dl_u": domain_view("dl_u", (nz, ncol)),
    }
    # module_sf_bep_bem.F:1052-1110 final roof diagnostics. Green-roof
    # runoff and vegetation state are reduced from the packed prognostic
    # arrays before the PBL scatter; PV energy comes from bem_hsfluxpv above.
    submit_bep_bem_fragment(
        context, "urban_bep_bem_finalize_fragment",
        {"active_cols": active_cols,
         "fraction": fraction.view((ncol,), byte_offset=fraction.byte_offset),
         "utype": param_resources["utype"], "class_float": class_float,
         "bs": param["bs"], "ws": param["ws"],
         "building_nlev": level_scan["building_nlev"],
         "d_urb": level_scan["d_urb"],
         "trv": packed["trv"], "qr": packed["qr"],
         "drain": packed["drain"],
         "pv_energy": pv_profiles["pv_energy"],
         "ep_pv": domain_view("ep_pv_urb3d", (ncol,)),
         "drain_green": domain_view("draingr_urb3d", (ncol,)),
         "qgr": domain_view("qgr_urb3d", (ncol,)),
         "tgr": domain_view("tgr_urb3d", (ncol,))},
        {"ncol": ncol, "green_roof": green_roof}, grid_size=ncol)

    submit_bem_final_output(
        context, final_outputs, ncol=ncol, nz=nz,
        class_float_len=int(class_float.shape[0]))

    scatter_names = (
        "tlev", "qlev", "tw", "twlev", "tg", "tr", "trv", "qr",
        "tflev", "tglev", "sfwin", "sfw", "sfg", "lfg", "dg", "tpv",
        "sfr", "lfr", "dgr", "sfrv", "lfrv", "drain")
    scatter_resources = {name: packed[name] for name in scatter_names}
    scatter_resources.update({
        "tlev_next": staged_indoor_state["tlev_next"].view(
            (ncol, NBUI_MAX, NZ_UM)),
        "qlev_next": staged_indoor_state["humlev_next"].view(
            (ncol, NBUI_MAX, NZ_UM)),
    })
    scatter_resources.update({
        name: array for name, array in state_resources.items()
        if name != "ss_urb"
    })
    submit_bem_state_scatter(
        context, scatter_resources, ncol=ncol, green_roof=green_roof)
    return plan


def launch_bep_bem_columns(dev: Mapping, cls: Mapping, *, gmt, julday,
                           declin_urb, dt, itimestep, num_urban_hi,
                           workspace_bytes: int = 256 << 20,
                           plan: ColumnPlan | None = None) -> ColumnPlan:
    """BEP_BEM's column loop (module_sf_bep_bem.F:708-1150), in place.

    ``dev`` holds every name in :data:`COLUMN_ARRAY_ARGUMENTS` as C-contiguous
    device arrays in woof's ``(k, ny, nx)`` layout (``(ny, nx)`` for 2-D;
    ``utype_urb2d`` int32, everything else float32).  Atmospheric inputs are
    ``(nz, ny, nx)`` mass levels; the level outputs ``a_u .. vl`` are
    ``(nz, ny, nx)`` and ``sf`` is ``(nz + 1, ny, nx)``.  Columns with
    ``frc_urb2d <= 0`` are never touched.  The urban columns are launched in
    chunks whose workspace fits ``workspace_bytes``; pass back the returned
    :class:`ColumnPlan` to skip rebuilding it.  A WRF ``stop`` inside the
    column is reported through :mod:`woof.core.health_ledger` (at once, or
    at the forecast's drain when one is active) as :class:`BepBemError`.
    """
    from woof.metal.array import MetalArray

    if isinstance(dev.get("frc_urb2d"), MetalArray):
        return _launch_bep_bem_columns_metal(
            dev, cls, gmt=gmt, julday=julday, declin_urb=declin_urb,
            dt=dt, itimestep=itimestep, num_urban_hi=num_urban_hi,
            workspace_bytes=workspace_bytes, plan=plan)

    import cupy as cp

    from woof.core import health_ledger

    nz, ny, nx = dev["dz8w"].shape
    if nz < 2:
        raise ValueError("BEP_BEM needs at least two mass levels: z(kte+1) "
                         "is built from dz8w and urban_meso reads level 2")
    if not 0 < int(num_urban_hi) < NZ_UM:
        raise ValueError(f"num_urban_hi={num_urban_hi}: BEP_BEM stops unless "
                         f"it is below nz_um={NZ_UM} (module_sf_bep_bem.F:656)")
    for name in COLUMN_ARRAY_ARGUMENTS:
        array = dev[name]
        dtype = cp.int32 if name == "utype_urb2d" else cp.float32
        if (not isinstance(array, cp.ndarray) or array.dtype != dtype
                or not array.flags.c_contiguous):
            raise TypeError(f"{name}: requires a C-contiguous device "
                            f"{np.dtype(dtype).name} array")
        if array.shape[-2:] != (ny, nx):
            raise ValueError(f"{name}: horizontal shape {array.shape[-2:]} "
                             f"is not {(ny, nx)}")
    if plan is None or plan.nz != nz or plan.shape != (ny, nx):
        plan = ColumnPlan(dev["frc_urb2d"], nz, workspace_bytes)
    if not plan.count:
        return plan
    scalars = {"gmt": np.float32(gmt), "julday": np.int32(julday),
               "declin_urb": np.float32(declin_urb), "dt": np.float32(dt),
               "itimestep": np.int32(itimestep), "nz": np.int32(nz),
               "ny": np.int32(ny), "nx": np.int32(nx),
               "num_urban_hi": np.int32(num_urban_hi)}
    kernel = _bem_module().get_function("bep_bem_columns")
    for part in plan.parts:
        count = int(part.size)
        err = plan.err[:count]
        args = (part, np.int32(count), cls["clsf"], cls["clsi"])
        args += tuple(dev[name] for name in COLUMN_ARRAY_ARGUMENTS)
        args += tuple(scalars[name] for name in COLUMN_SCALAR_ARGUMENTS)
        args += (plan.ws, np.int32(plan.words), err)
        kernel(((count + 31) // 32,), (32,), args)
        status = err.max().reshape(1)
        code = health_ledger.read_status(status, site="urban_bem",
                                         describe=_refuse)
        if code:
            _check_errors(err, part)
    return plan


# ---------------------------------------------------------------------------
# DESIGN 3.4 entry points (infra's UrbanCoupler calls these by name)
# ---------------------------------------------------------------------------

#: BEP_BEM's atmospheric inputs and the ``_prepare_atmosphere`` key each one
#: is (module_sf_noahdrv.F:1650-1677, module_sf_noahmpdrv.F:3658-3685).
ATMOSPHERE_SOURCES = MappingProxyType({
    "dz8w": "dz", "u_phy": "u", "v_phy": "v", "th_phy": "theta",
    "rho": "rho", "p_phy": "pressure", "qv_phy": "qv",
})
#: Kernel surface output -> its name in ``UrbanState.bep_out``, where the
#: shared couple reads it (``urban_bep_couple.BEP_OUTPUTS``).
SURFACE_OUTPUTS = MappingProxyType({
    "rl_up": "rl_up_urb", "rs_abs": "rs_abs_urb", "emiss": "emiss_urb",
    "grdflx_urb": "grdflx_urb",
})
#: Kernel level output -> its ``UrbanState.pbl_terms`` key.
PBL_OUTPUTS = MappingProxyType({
    "a_u": "a_u_bep", "a_v": "a_v_bep", "a_t": "a_t_bep", "a_e": "a_e_bep",
    "b_u": "b_u_bep", "b_v": "b_v_bep", "b_t": "b_t_bep", "b_e": "b_e_bep",
    "b_q": "b_q_bep", "dlg": "dlg_bep", "dl_u": "dl_u_bep", "sf": "sf_bep",
    "vl": "vl_bep",
})
_FIELD_INPUTS = ("swdown", "glw", "swddir", "swddif", "rainbl")
_SOLAR_INPUTS = MappingProxyType({
    "cosz_urb2d": "coszen", "omg_urb2d": "hrang", "xlat": "xlat",
    "xlong": "xlong"})
#: Kernel arguments that are UrbanState arrays under their own names.
STATE_ARGUMENTS = tuple(
    name for name in COLUMN_ARRAY_ARGUMENTS
    if name not in ATMOSPHERE_SOURCES and name not in SURFACE_OUTPUTS
    and name not in PBL_OUTPUTS and name not in _FIELD_INPUTS
    and name not in _SOLAR_INPUTS)


#: The column workspace budget a forecast domain's plan may hold.  A chunk
#: of urban columns costs about one column's serial time (63 ms on the RTX
#: 4090 at 30 levels) whatever its width up to ~10,000 columns, so the
#: budget sets the cost: measured for 10,000 urban columns, 569 ms per call
#: at 256 MiB (9 chunks) and 79 ms at 2.1 GiB (1 chunk).  The plan holds
#: min(urban columns, budget / per-column) columns' worth
#: (:func:`workspace_bytes_per_column`), so a small city holds only what it
#: uses.
FORECAST_WORKSPACE_BYTES = 1 << 30


class ShortwaveSplitMissing(RuntimeError):
    """Option 3 reached the surface step without SWDDIR/SWDDIF."""


def init_state(state, params, *, tlayer0, tsurface0, tdeep0, smois,
               restart: bool) -> None:
    """Prepare resident BEP+BEM class profiles when the state is on Metal.

    :func:`woof.core.urban_state.urban_var_init_host` transcribes the whole
    option-3 part of ``urban_var_init`` (module_sf_urban.F:2912-2992);
    ``tests/test_urban_bem_wrf471_parity.py::
    test_option3_cold_start_is_urban_var_init`` grades it against the state
    WRF's own ``urban_var_init`` leaves, word for word. The first BEM
    profile stage is a separate typed Metal dispatch; the full class table
    initializer and column solve remain downstream stages.
    """
    del tlayer0, tsurface0, tdeep0, smois, restart
    from woof.metal.array import MetalArray

    fraction = state.fields.get("frc_urb2d")
    if not isinstance(fraction, MetalArray):
        return
    context = fraction.buffer.context
    state.bem_class_tables = class_tables(params, context=context)

    # WRF's bb_view_factors receives STREET_DIRECTION_TBL and
    # STREET_WIDTH_TBL for each urban type. Resolve utype on device so a
    # Metal state never stages that grid field through NumPy during setup.
    from woof.metal.array import MetalArrayModule
    utype = state.fields["utype_urb2d"]
    if (not isinstance(utype, MetalArray) or utype.dtype != np.dtype(np.int32)
            or utype.buffer.context is not context
            or tuple(utype.shape) != tuple(fraction.shape)):
        raise TypeError("BEM Metal init needs same-context int32 utype_urb2d")
    ncol = int(utype.size)
    table = tables_from_params(params)
    nclasses = int(params.icate)
    if not 1 <= nclasses <= NURBMAX:
        raise ValueError("BEM class count is outside the WRF table extent")
    module = MetalArrayModule(context)
    dxy_table = module.asarray(np.ascontiguousarray(
        np.asarray(table["street_direction_tbl"], dtype=np.float32)
        [:NDM, :nclasses]))
    width_table = module.asarray(np.ascontiguousarray(
        np.asarray(table["street_width_tbl"], dtype=np.float32)
        [:NDM, :nclasses]))
    z = module.empty((ncol, NZ_UM + 1), dtype=np.float32)
    dxy = module.empty((ncol, NDM), dtype=np.float32)
    street_width = module.empty((ncol, NDM), dtype=np.float32)
    error = module.zeros((ncol, NDM), dtype=np.int32)
    from woof.metal.generated import kernel_specs
    context.submit_kernel(
        "urban_bem_view_inputs_fragment",
        {"utype": utype.view((ncol,), byte_offset=utype.byte_offset),
         "street_direction_table": dxy_table.view((NDM * nclasses,)),
         "street_width_table": width_table.view((NDM * nclasses,)),
         "z": z.view((z.size,)), "dxy": dxy.view((dxy.size,)),
         "street_width": street_width.view((street_width.size,)),
         "error": error.view((error.size,))},
        {"ncol": ncol, "nclasses": nclasses, "nd": NDM},
        exact_policy=False, grid_size=ncol * NDM,
        threads_per_group=kernel_specs.URBAN_BEM_VIEW_INPUTS_FRAGMENT_THREADS_PER_GROUP,
    )
    state.bem_view_factor_columns = np.arange(ncol, dtype=np.int32)
    state.bem_view_factors = launch_bem_view_factors_metal(
        context, z=module.asarray(z), dxy=module.asarray(dxy),
        street_width=module.asarray(street_width), nz_u=NZ_UM, nd=NDM)


def after_lsm(state, params, *, lsm: int, fields: dict, atmosphere,
              dt: float, itimestep: int, solar, cfg) -> None:
    """BEP_BEM on every urban column, then the shared BEP couple."""
    del cfg
    from woof.core import urban_bep_couple

    for name in ("swddir", "swddif"):
        if name not in fields:
            raise ShortwaveSplitMissing(
                "sf_urban_physics=3 needs the radiation step's direct and "
                f"diffuse surface shortwave ({name.upper()}): BEP_BEM's "
                "shadow_mas and short_rad_dd put the sun on roads, walls and "
                "windows from them, and without them every building would "
                "be heated by zero sun")
    bep_out = _bep_out(state, urban_bep_couple.BEP_OUTPUTS)
    # module_sf_noahdrv.F:1639-1647 / module_sf_noahmpdrv.F:3646-3656:
    # zeroed on every column before every call.
    for name in urban_bep_couple.BEP_OUTPUTS:
        bep_out[name][...] = 0.0
    pbl = state.pbl_terms
    pbl["b_q_bep"][...] = 0.0
    dev = {name: state.fields[name] for name in STATE_ARGUMENTS}
    # These setup products are BEM-specific inputs to the Metal column path;
    # the legacy CUDA argument tuple carries them through its class record.
    dev["_bem_view_factors"] = getattr(state, "bem_view_factors", None)
    for name, source in ATMOSPHERE_SOURCES.items():
        dev[name] = atmosphere[source]
    for name in _FIELD_INPUTS:
        dev[name] = fields[name]
    for name, source in _SOLAR_INPUTS.items():
        dev[name] = getattr(solar, source)
    for name, key in PBL_OUTPUTS.items():
        dev[name] = pbl[key]
    for name, key in SURFACE_OUTPUTS.items():
        dev[name] = bep_out[key]
    from woof.metal.array import MetalArray
    fraction = dev["frc_urb2d"]
    context = fraction.buffer.context if isinstance(fraction, MetalArray) else None
    state.bem_plan = launch_bep_bem_columns(
        dev, class_tables(params, context=context),
        gmt=float(solar.gmt), julday=int(solar.julday),
        declin_urb=float(solar.declin), dt=float(dt),
        itimestep=int(itimestep), num_urban_hi=int(state.num_urban_hi),
        workspace_bytes=FORECAST_WORKSPACE_BYTES,
        plan=getattr(state, "bem_plan", None))
    urban_bep_couple.couple(state, lsm=lsm, fields=fields,
                            atmosphere=atmosphere, dt=dt, bep_out=bep_out)


def after_surface_diagnostics(state, *, lsm: int, fields: dict, atmosphere,
                              cfg) -> None:
    """Options 2 and 3 share the 2 m overrides (the BEP lane's couple)."""
    from woof.core import urban_bep_couple

    urban_bep_couple.after_surface_diagnostics(
        state, lsm=lsm, fields=fields, atmosphere=atmosphere, cfg=cfg)


def _bep_out(state, names) -> dict:
    """The BEP column outputs the couple reads, held on the state.

    Options 2 and 3 share them (``urban_bep_couple.couple`` reads
    ``state.bep_out``); whichever model runs first on a domain allocates.
    """
    from woof.metal.array import MetalArray, MetalArrayModule

    fraction = state.fields["frc_urb2d"]
    if isinstance(fraction, MetalArray):
        xp = MetalArrayModule(fraction.buffer.context)
    else:
        import cupy as cp
        xp = cp

    held = getattr(state, "bep_out", None)
    shape = fraction.shape
    if held is None or any(n not in held or held[n].shape != shape
                           for n in names):
        held = {name: xp.zeros(shape, dtype=np.float32) for name in names}
        state.bep_out = held
    return held
