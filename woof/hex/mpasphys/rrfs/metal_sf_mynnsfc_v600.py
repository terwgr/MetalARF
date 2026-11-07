"""Packed Metal submission for the RRFS MYNN-SFC v600 column scheme."""

from operator import index

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule

from . import sf_mynnsfc_v600 as source
from .sf_mynnsfc_v600 import FIELDS, MPAS_PRESENT, OPTIONS

_TABLE_SIZE = 1001
_INT32_MAX = np.iinfo(np.int32).max
_PROFILE_INPUTS = ("u3d", "v3d", "t3d", "qv3d", "p3d", "dz8w", "th3d", "rho3d")


class MetalMpasSfMynnV600:
    """Bind the fork's surface layer and persistent sea-ice handoff."""

    def restart_state(self):
        return {}

    def restore_restart_state(self, state):
        if state != {}:
            raise ValueError('RRFS MYNN-SFC physical state belongs to native pools')

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('surface_layer', source.SCHEME):
            raise ValueError('RRFS MYNN-SFC requires its surface-layer slot')
        pools = call.pools
        configs, scalar = pools['configs'], pools['scalar_state']
        options = call.options
        return driver(pools['mesh'], pools['diag_physics'], pools['sfc_input'],
                      pools['atm_input'], itimestep=call.itimestep,
                      len_disp=configs['config_len_disp'],
                      xice_threshold=scalar['xice_threshold'],
                      config_do_restart=configs['config_do_restart'],
                      config_do_DAcycling=configs['config_do_DAcycling'],
                      config_frac_seaice=options['config_frac_seaice'],
                      config_lsm_scheme=options['config_lsm_scheme'],
                      config_mynn_sfcflux_land=options['config_mynn_sfcflux_land'],
                      config_mynn_sfcflux_water=options['config_mynn_sfcflux_water'],
                      sea_state=pools['prepared'])


def mynnsfc_tables(context):
    """Build the four default RRFS stability tables once per Metal context."""
    tables = getattr(context, "_rrfs_mynnsfc_v600_tables", None)
    if tables is not None:
        return tables

    tables = MetalArrayModule(context).empty((4, _TABLE_SIZE), np.float32)
    context.submit_kernel(
        "hexrrfs_mynnsfc_tables", {"tables": tables}, {"ntable": _TABLE_SIZE},
        exact_policy=True, grid_size=_TABLE_SIZE)
    # The retained scheme checks the signs at the first nonzero abscissa.
    momentum_stable, momentum_unstable, heat_stable, heat_unstable = (
        tables.to_numpy()[:, 1])
    if not (momentum_stable < 0 and heat_stable < 0
            and momentum_unstable > 0 and heat_unstable > 0):
        raise ValueError("RRFS MYNN-SFC stability tables failed initialization")
    context._rrfs_mynnsfc_v600_tables = tables
    return tables


def run_packed(fields, options, *, vegtype=None, errors=None,
               xice_threshold=0.02, present=MPAS_PRESENT,
               itimestep=None, initflag=None):
    """Update packed fields in place and return the device error words."""
    if (not isinstance(fields, MetalArray) or fields.dtype != np.dtype(np.float32)
            or fields.ndim != 2 or fields.shape[0] != len(FIELDS)
            or not fields.flags.c_contiguous):
        raise TypeError("MYNN-SFC fields must be contiguous float32 (len(FIELDS), ncol)")
    ncol = fields.shape[1]
    if fields.size > _INT32_MAX:
        raise ValueError("MYNN-SFC packed fields exceed the signed column index range")
    context = fields.buffer.context
    arrays = MetalArrayModule(context)

    if vegtype is None:
        vegtype = arrays.ones(ncol, np.int32)
    if errors is None:
        errors = arrays.empty(ncol, np.int32)
    for name, value, shape in (("options", options, (len(OPTIONS),)),
                               ("vegtype", vegtype, (ncol,)),
                               ("errors", errors, (ncol,))):
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.int32)
                or value.shape != shape or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(f"MYNN-SFC {name} must be same-context int32 {shape}")
    if ncol == 0:
        return errors

    mask = index(present)
    if not 0 <= mask < (1 << 64):
        raise ValueError("MYNN-SFC optional-field mask must fit uint64")
    mask_words = np.array([mask & 0xffffffff, mask >> 32], np.uint32).view(np.int32)
    step = -1 if itimestep is None else index(itimestep)
    init = -1 if initflag is None else index(initflag)
    if not (-(1 << 31) <= step <= _INT32_MAX and -(1 << 31) <= init <= _INT32_MAX):
        raise ValueError("MYNN-SFC step and initialization overrides must fit int32")

    context.submit_kernel(
        "hexrrfs_mynnsfc_columns",
        {"fields": fields, "options": options, "vegtype": vegtype,
         "errors": errors, "tables": mynnsfc_tables(context)},
        {"ncol": ncol, "xice_threshold": np.float32(xice_threshold),
         "present_lo": mask_words[0], "present_hi": mask_words[1],
         "step_override": step, "init_override": init},
        exact_policy=True, grid_size=ncol)
    return errors


def driver(mesh, diagnostics, surface, profiles, *, itimestep, len_disp,
           xice_threshold, config_frac_seaice=False, config_do_restart=False,
           config_do_DAcycling=False, config_lsm_scheme="sf_ruc",
           config_mynn_sfcflux_land=0, config_mynn_sfcflux_water=0, spp_pbl=0,
           sea_state=None):
    """Run the native RRFS surface-layer pack, column calls and publication."""
    xland = surface["xland"]
    if not isinstance(xland, MetalArray) or xland.ndim != 1:
        raise TypeError("MYNN-SFC xland must be a resident Metal column plane")
    ncol = xland.size
    context = xland.buffer.context

    def plane(value, name):
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.float32)
                or value.shape != (ncol,) or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise TypeError(f"MYNN-SFC {name} must be same-context float32 ({ncol},)")
        return value

    inputs = {name: diagnostics[name] for name in source._DIAG_INPUT}
    inputs["pblh"] = diagnostics["hpbl"]
    inputs.update((name, surface[key]) for name, key in source._SURFACE.items())
    inputs["psfcpa"] = profiles["psfcpa"]
    for name in _PROFILE_INPUTS:
        value = profiles[name]
        if (not isinstance(value, MetalArray) or value.ndim != 2
                or value.shape[0] < 2 or not value.flags.c_contiguous):
            raise TypeError(f"MYNN-SFC {name} must be contiguous (levels>=2,ncol)")
        inputs[name] = value[0]
        if name in ("u3d", "v3d", "dz8w"):
            inputs[name + "2"] = value[1]
    inputs = {name: plane(value, name) for name, value in inputs.items()}
    outputs = {name: plane(diagnostics[source._DIAG_NAMES.get(name, name)], name)
               for name in source._WRITE}
    handoff = ({name: plane(sea_state[name + "_sea"], name + "_sea")
                for name in source._SEA_HANDOFF}
               if config_frac_seaice and sea_state is not None else {})
    density = plane(mesh["meshDensity"], "meshDensity")
    ice = plane(surface["xice"], "xice")
    if ncol == 0:
        return diagnostics

    packed_words = len(FIELDS) * ncol
    packed_bytes = packed_words * np.dtype(np.float32).itemsize
    if (packed_words > _INT32_MAX or packed_bytes > context.max_buffer_length
            or packed_bytes * (2 if config_frac_seaice else 1)
            > context.recommended_working_set_size):
        raise ValueError("MYNN-SFC packed workspace exceeds the Metal context limits")
    arrays = MetalArrayModule(context)
    packed = arrays.zeros((len(FIELDS), ncol), np.float32)
    sea = arrays.empty(packed.shape, np.float32) if config_frac_seaice else packed
    sst = plane(surface["sst"], "sst") if config_frac_seaice else packed[0]
    context.submit_bookkeeping([(value, packed[FIELDS.index(name)])
                                for name, value in inputs.items()])
    threshold = np.float32(xice_threshold)
    context.submit_kernel(
        "hexrrfs_mynnsfc_from_mpas",
        {"fields": packed, "sea": sea, "density": density, "sst": sst, "ice": ice},
        {"ncol": ncol, "len_disp": np.float32(len_disp),
         "threshold": threshold, "fraction": int(bool(config_frac_seaice))},
        exact_policy=True, grid_size=ncol)

    initflag = int(not (config_do_restart or config_do_DAcycling or itimestep > 1))
    lsm = {"sf_noaa": 1, "sf_noahmp": 2, "sf_ruc": 3}.get(config_lsm_scheme, 4)
    options = arrays.asarray(np.array((
        0, 0, 0, spp_pbl, config_mynn_sfcflux_water, config_mynn_sfcflux_land,
        0, 1, int(config_do_restart), int(config_do_DAcycling), initflag,
        1, lsm, itimestep), dtype=np.int32))
    vegetation = arrays.ones(ncol, np.int32)
    errors = run_packed(packed, options, vegtype=vegetation, xice_threshold=threshold)
    if config_frac_seaice:
        pblh = FIELDS.index("pblh")
        context.submit_bookkeeping([(packed[pblh], sea[pblh])])
        run_packed(sea, options, vegtype=vegetation, errors=errors,
                   xice_threshold=threshold)
        context.submit_kernel(
            "hexrrfs_mynnsfc_to_mpas", {"fields": packed, "sea": sea, "ice": ice},
            {"ncol": ncol, "threshold": threshold}, exact_policy=True, grid_size=ncol)
    context.submit_bookkeeping(
        [(packed[FIELDS.index(name)], value) for name, value in outputs.items()]
        + [(sea[FIELDS.index(name)], value) for name, value in handoff.items()])
    return diagnostics
