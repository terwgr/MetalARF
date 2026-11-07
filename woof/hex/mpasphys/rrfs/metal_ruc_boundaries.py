"""Native RRFS RUC forcing, surface diagnostics and sea-ice boundaries on Metal."""

from operator import index

import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule


_BLEND = ("flhc", "flqc", "cqs", "cpm", "chs", "chs2", "cqs2", "hfx", "lh", "qfx",
          "qgh", "qsfc", "tsk")


def _launch(name, anchor, resources, solve_cells, **parameters):
    if not isinstance(anchor, MetalArray) or anchor.ndim != 1:
        raise TypeError("RRFS RUC requires a one-dimensional Metal surface plane")
    nc = anchor.size
    solve = index(solve_cells)
    if not 0 <= solve <= nc <= np.iinfo(np.int32).max:
        raise ValueError("RRFS RUC solve columns must fit the surface extent")
    if solve:
        anchor.buffer.context.submit_kernel(
            f"hexrrfs_ruc_{name}", resources, {"nc": nc, "solve": solve, **parameters},
            exact_policy=True, grid_size=solve)


def prepare_forcing(inputs, prepared, *, solve_cells, microphysics, convection):
    """Publish source-order precipitation and incident shortwave on solve cells."""
    solve_cells = index(solve_cells)
    if solve_cells == 0:
        return
    gsw = inputs["gsw"]
    resources = {name: inputs[name] for name in ("gsw", "sfc_albedo")}
    for name, active in (("sr", microphysics), ("rainncv", microphysics), ("raincv", convection)):
        if active:
            value = inputs[name]
            if not isinstance(value, MetalArray) or value.shape != gsw.shape:
                raise ValueError(f"RRFS RUC active precipitation field {name} must match the solve extent")
            resources[name] = value
        else:
            resources[name] = gsw.view((1,))
    resources.update((name, prepared[name]) for name in ("sr_p", "rainbl_p", "swdown_p"))
    _launch("forcing", gsw, resources, solve_cells,
            micro_on=int(bool(microphysics)), conv_on=int(bool(convection)))


def prepare_seaice(surface, prepared, *, solve_cells, threshold):
    """Extract ice-only albedo and emissivity for fractional ice columns."""
    resources = {name: surface[name] for name in ("xice", "sfc_albedo", "sfc_emiss")}
    resources.update((name, prepared[name]) for name in ("sfc_albedo_p", "sfc_emiss_p"))
    _launch("seaice_prepare", surface["xice"], resources, solve_cells, xice_threshold=np.float32(threshold))


def broadcast_soil_profile(zs, output, *, solve_cells):
    """Gather the first solve column's soil geometry for every work column."""
    if not isinstance(zs, MetalArray) or zs.ndim != 2 or zs.shape[0] < 1 or zs.size >= 2**31:
        raise ValueError("RRFS RUC soil geometry requires a packed profile with int32 extent")
    _launch("soil_profile", zs[0], {"zs": zs, "output": output}, solve_cells, nsl=zs.shape[0])


def prepare_seaice_background(xice, albbck, output, *, solve_cells, threshold):
    """Apply the native ice background albedo before the column solver."""
    _launch("seaice_background", xice, {"xice": xice, "albbck": albbck, "output": output},
            solve_cells, xice_threshold=np.float32(threshold))


def correct_seaice_skin(xice, tsk, tsk_sea, tsk_ice, *, solve_cells, threshold):
    """Apply the distinct native ice and sea skin-temperature floors."""
    _launch("seaice_skin", xice, dict(xice=xice, tsk=tsk, tsk_sea=tsk_sea, tsk_ice=tsk_ice),
            solve_cells, xice_threshold=np.float32(threshold))


def finish_seaice(prepared, sea, *, solve_cells, threshold, fractional, mynn):
    """Blend prepared ice results with saved ocean fluxes and state."""
    resources = {name: prepared[name] for name in ("xice_p", "sfc_albedo_p", "sfc_emiss_p")}
    resources.update((f"{name}_p", prepared[f"{name}_p"]) for name in _BLEND)
    resources.update((f"{name}_sea", sea[f"{name}_sea"]) for name in _BLEND)
    _launch("seaice_finish", prepared["xice_p"], resources, solve_cells, xice_threshold=np.float32(threshold),
            config_frac_seaice=int(bool(fractional)), mynn=int(bool(mynn)))


def surface_diagnostics(fields, *, solve_cells, cp, rovcp, mynn):
    """Publish two-metre fields and the non-MYNN cqs copy on owned columns.

    The native resident authority contains only solve columns. Any tail in
    these packed work views is outside that authority and remains untouched.
    """
    surface = ("hfx", "qfx", "tsk", "qsfc", "cqs", "cqs2", "chs", "chs2",
               "t2", "th2", "q2", "psfc2d")
    resources = {name: fields[name] for name in surface}
    tsk = resources["tsk"]
    for name in ("t3d", "qv3d", "dz", "rho3d", "p3d"):
        profile = fields[name]
        if (not isinstance(profile, MetalArray) or profile.ndim != 2
                or profile.shape[0] == 0 or profile.shape[1] != tsk.size):
            raise ValueError(f"RRFS RUC diagnostics {name} must contain level-one values for every column")
        resources[name] = profile[0]
    _launch("diagnostics", tsk, resources, solve_cells, cp=np.float32(cp),
            rovcp=np.float32(rovcp), mynn=int(bool(mynn)))


def initialize_columns(fields, tables, *, solve_cells, restart, cycling):
    """Initialize cold-start soil and roughness work views before publication."""
    solve = index(solve_cells)
    if restart or cycling or solve == 0:
        return
    znt, soil = fields["znt"], fields["tslb"]
    if (not isinstance(znt, MetalArray) or znt.ndim != 1
            or not isinstance(soil, MetalArray) or soil.ndim != 2
            or soil.shape[1] != znt.size or soil.shape[0] == 0):
        raise ValueError("RRFS RUC initialization requires packed surface and soil views")
    if soil.size >= 2**31:
        raise ValueError("RRFS RUC initialization soil offsets exceed int32")
    resources = {name: fields[name] for name in ("ivgtyp", "isltyp", "xice", "tslb", "smois",
                                                "sh2o", "smfr3d", "mavail", "znt")}
    error = MetalArrayModule(znt.buffer.context).zeros(1, np.int32)
    resources.update(tables=tables, error=error)
    _launch("initialize", znt, resources, solve, nsl=soil.shape[0])
    if error.to_numpy()[0]:
        raise ValueError("RRFS RUC initialization has an out-of-range soil or vegetation category")
