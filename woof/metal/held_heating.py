"""Metal launcher for the ordinary retained-heating tendency update."""

from __future__ import annotations

from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_held_heating(state):
    """Add one retained ``h_diabatic`` contribution to ``rth_t``.

    The kernel carries the CUDA source's explicit FP32 operation sequence.
    WRF-exact arithmetic remains refused until Metal provides the required
    arithmetic guarantees.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal held_heating supports ordinary non-exact arithmetic only")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal held_heating requires a Metal DomainState")

    nz, ny, nx = state.rth_t.shape
    if any(not 0 < dim <= 0x7FFFFFFF for dim in (nz, ny, nx)):
        raise ValueError("Metal held_heating extents must fit positive int32")
    grid_size = nz * ny * nx
    if grid_size > 0xFFFFFFFF:
        raise ValueError("Metal held_heating grid does not fit the uint grid index")
    if (state.h_diabatic.shape != (nz, ny, nx)
            or state.mub2d.shape != (ny, nx)
            or state.mup.shape != (ny, nx)
            or state.c1h.shape != (nz,)
            or state.c2h.shape != (nz,)
            or state.msft.shape != (ny, nx)):
        raise ValueError("Metal held_heating received incompatible array shapes")

    arrays = {
        "rth": state.rth_t,
        "hd": state.h_diabatic,
        "mub": state.mub2d,
        "mup": state.mup,
        "c1": state.c1h,
        "c2": state.c2h,
        "msft": state.msft,
    }
    output = state.rth_t
    out_lo, out_hi = output.byte_offset, output.byte_offset + output.nbytes
    for name, array in arrays.items():
        if name == "rth":
            continue
        if (array.buffer.handle == output.buffer.handle
                and out_lo < array.byte_offset + array.nbytes
                and array.byte_offset < out_hi):
            raise ValueError(
                f"Metal held_heating does not support rth overlap with {name}")
    parameters = {
        "has_msf": int(bool(state.has_msf)),
        "nz": nz,
        "ny": ny,
        "nx": nx,
    }
    return state._array_module.context.submit_kernel(
        "held_heating", arrays, parameters, exact_policy=False,
        grid_size=grid_size)
