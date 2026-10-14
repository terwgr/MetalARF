"""Metal submission helpers for ordinary and HRRR sixth-order diffusion."""

from __future__ import annotations

import numpy as np

from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_diff6_to_edge(state, f, tend, mut, c1, c2, *, factor, dt, opt,
                         stagger="", phb=None, msfu=None, msfv=None, msft=None,
                         slopeopt=0, thresh=0.10, dx=0.0, dy=0.0, work=None):
    """Run the unchanged diff6 stencil on zero-gradient padded device views.

    Match the CUDA HRRR launcher: three copied halo cells on the low sides,
    four on the high sides, then add only the real-grid tendency. Reuse the
    preflight-priced edge workspace; no host staging or array arithmetic.
    """
    if WRF_EXACT:
        raise NotImplementedError("Metal diff6 does not implement WRF exact-D arithmetic")
    if state is None or not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal edge diff6 requires its owning DomainState")
    if work is None:
        raise ValueError("Metal edge diff6 requires the declared edge workspace")
    from .array import MetalArray
    from woof.core.diff6_edge_workspace import (
        DIFF6_EDGE_PAD, plane_shapes)
    if stagger not in ("", "x", "y", "z"):
        raise ValueError(f"unsupported diff6 stagger {stagger!r}")
    if f.ndim != 3 or tend.shape != f.shape:
        raise ValueError("Metal edge diff6 requires equal 3-D field/tendency shapes")
    context = state._array_module.context
    for array in (f, tend, c1, c2, *work.values()):
        if array is None:
            continue
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float32)
                or array.buffer.context is not context):
            raise TypeError("Metal edge diff6 needs contiguous FP32 arrays on one context")
    nlev, nys, nxs = f.shape
    ny = nys - int(stagger == "y")
    nx = nxs - int(stagger == "x")
    if state.mup.shape != (ny, nx):
        raise ValueError("Metal edge diff6 field does not match the owning mass grid")
    pad = DIFF6_EDGE_PAD
    slope = int(slopeopt) >= 1 and phb is not None and phb.ndim == 3
    if slope and (dx <= 0.0 or dy <= 0.0):
        raise ValueError("diff_6th_slopeopt >= 1 needs positive dx/dy")

    def prefix(array, shape, offset=0):
        count = int(np.prod(shape))
        if offset < 0 or offset + count > array.size:
            raise ValueError("Metal diff6 edge workspace is smaller than the requested view")
        return array.view(shape, byte_offset=array.byte_offset + offset * 4)

    def edge_pad(src, dst):
        if not isinstance(src, MetalArray) or src.dtype != np.dtype(np.float32):
            raise TypeError("Metal diff6 padding requires FP32 device input")
        if src.buffer.context is not context or src.ndim not in (2, 3):
            raise ValueError("Metal diff6 padding input has invalid context or dimensions")
        levels = src.shape[0] if src.ndim == 3 else 1
        rows, cols = src.shape[-2:]
        expected = src.shape[:-2] + (rows + pad, cols + pad)
        if dst.shape != expected:
            raise ValueError("Metal diff6 padded shape does not match its input")
        context.submit_kernel(
            "diff6_edge_pad", {"src": src, "dst": dst},
            {"nlev": levels, "ny": rows, "nx": cols},
            exact_policy=False, grid_size=dst.size)
        return dst

    fp = edge_pad(f, prefix(work["field"], (nlev, nys + pad, nxs + pad)))
    tp = prefix(work["tend"], fp.shape)
    tp.fill(0)
    if mut is None:
        mut = state.scratch((ny, nx), "diff6_mu_t")
        context.submit_kernel(
            "rk_total_mu", {"mub2d": state.mub2d, "mup": state.mup0, "out": mut},
            {"ny": ny, "nx": nx}, exact_policy=False, grid_size=ny * nx)
    padded = []
    offset = 0
    for src, shape in zip((mut, msfu, msfv, msft), plane_shapes(ny, nx)):
        dst = prefix(work["planes"], shape, offset)
        offset += dst.size
        if src is None:
            dst.fill(1)
        else:
            edge_pad(src, dst)
        padded.append(dst)
    phbp = None
    if slope:
        phbp = edge_pad(phb, prefix(work["phb"],
                                   (phb.shape[0], ny + pad, nx + pad)))
    f32 = np.float32
    coef = f32(factor) * f32(0.015625) / (f32(2.0) * f32(dt))
    dzthr_x = f32(thresh) * f32(9.81) * (f32(1.0) / f32(1.0 / dx)) if slope else f32(0)
    dzthr_y = f32(thresh) * f32(9.81) * (f32(1.0) / f32(1.0 / dy)) if slope else f32(0)
    submit_diff6(state, fp, tp, padded[0], c1, c2, coef=coef, mono=opt,
                 stagger=stagger, phb=phbp, msfu=padded[1], msfv=padded[2],
                 msft=padded[3], slopeopt=int(slope),
                 dzthr_x=dzthr_x, dzthr_y=dzthr_y)
    context.submit_kernel(
        "diff6_edge_crop_add", {"padded": tp, "tend": tend},
        {"nlev": nlev, "ny": nys, "nx": nxs},
        exact_policy=False, grid_size=tend.size)


def submit_diff6(state, f, tend, mut, c1, c2, *, coef, mono,
                 stagger="", phb=None, msfu=None, msfv=None, msft=None,
                 slopeopt=0, dzthr_x=0.0, dzthr_y=0.0,
                 boundary_x=False, boundary_y=False):
    """Submit ordinary diff6 and its nonperiodic staggered seam passes.

    The caller owns zeroing ``tend`` and applying WRF's width-three open
    strips after this submission. All Metal work stays ordered on the
    state's serial context queue.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal diff6 does not implement WRF exact-D arithmetic")
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal diff6 requires a Metal DomainState")
    if stagger not in ("", "x", "y", "z"):
        raise ValueError(f"unsupported diff6 stagger {stagger!r}")
    if state is None:
        raise TypeError("Metal diff6 requires its owning DomainState")
    from .array import MetalArray
    arrays = {"f": f, "tend": tend, "c1": c1, "c2": c2}
    if mut is not None:
        arrays["mut"] = mut
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"Metal diff6 {name} must be contiguous FP32")
    context = f.buffer.context
    if any(array.buffer.context is not context for array in arrays.values()):
        raise ValueError("Metal diff6 arrays belong to different contexts")
    nlev, nys, nxs = f.shape
    ny = nys - 1 if stagger == "y" else nys
    nx = nxs - 1 if stagger == "x" else nxs
    if tend.shape != f.shape:
        raise ValueError("Metal diff6 field and tendency shapes differ")
    if c1.shape != (nlev,) or c2.shape != (nlev,):
        raise ValueError("Metal diff6 hybrid coefficients must match nlev")
    if stagger == "x" and nxs != nx + 1:
        raise ValueError("Metal diff6 u field must have nx+1 columns")
    if stagger != "x" and nxs != nx:
        raise ValueError("Metal diff6 field x extent does not match its stagger")
    if stagger == "y" and nys != ny + 1:
        raise ValueError("Metal diff6 v field must have ny+1 rows")
    if stagger != "y" and nys != ny:
        raise ValueError("Metal diff6 field y extent does not match its stagger")
    variant = {"": 0, "x": 1, "y": 2, "z": 0}[stagger]
    if mut is None:
        # Match the production caller's time-t ``mub2d + mup0`` mass without
        # evaluating unsupported MetalArray Python arithmetic on the host.
        mut = state.scratch((ny, nx), "diff6_mu_t")
        state._array_module.context.submit_kernel(
            "rk_total_mu",
            {"mub2d": state.mub2d, "mup": state.mup0, "out": mut},
            {"ny": ny, "nx": nx}, exact_policy=False,
            grid_size=ny * nx)
    if mut.shape != (ny, nx):
        raise ValueError("Metal diff6 total mass must match the mass grid")
    dummy2d = mut
    slope = int(slopeopt) >= 1 and phb is not None and phb.ndim == 3
    if slope and (dzthr_x <= 0.0 or dzthr_y <= 0.0):
        raise ValueError("diff6 terrain taper requires positive thresholds")
    phb = phb if slope else dummy2d
    msfu = state.msfu if msfu is None else msfu
    msfv = state.msfv if msfv is None else msfv
    msft = state.msft if msft is None else msft
    arrays = {"f": f, "tend": tend, "mut": mut, "c1": c1, "c2": c2,
              "phb": phb, "msfu": msfu, "msfv": msfv, "msft": msft}
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous):
            raise TypeError(f"Metal diff6 {name} must be contiguous FP32")
        if array.buffer.context is not context:
            raise ValueError("Metal diff6 arrays belong to different contexts")
    if msfu.shape != (ny, nx + 1) or msfv.shape != (ny + 1, nx):
        raise ValueError("Metal diff6 map factors do not match staggered grid")
    if msft.shape != (ny, nx):
        raise ValueError("Metal diff6 mass map factor does not match grid")
    if slope and (phb.ndim != 3 or phb.shape[-2:] != (ny, nx)
                  or phb.shape[0] < nlev):
        raise ValueError("Metal diff6 terrain geopotential must cover levels/grid")
    common = {"coef": float(np.float32(coef)), "mono": int(mono),
              "slopeopt": int(slope), "dzthr_x": float(np.float32(dzthr_x)),
              "dzthr_y": float(np.float32(dzthr_y)), "nlev": nlev,
              "ny": ny, "nys": nys, "nx": nx, "nxs": nxs,
              "variant": variant, "wstag": int(stagger == "z"),
              "boundary_x": int(bool(boundary_x)),
              "boundary_y": int(bool(boundary_y))}
    context.submit_kernel("diff6", arrays, common, exact_policy=False,
                          grid_size=nlev * nys * nxs)

    # WRF includes u(ide-3)/v(jde-3) in nonperiodic loops. The main kernel
    # skips these cells when enabled; these direct-read passes fill them.
    if stagger == "x" and boundary_x and nx >= 6:
        h0, h1 = ((3, ny - 4) if boundary_y else (0, ny - 1))
        if h1 >= h0:
            seam_params = {k: v for k, v in common.items()
                           if k not in ("nys", "nxs", "variant", "wstag",
                                        "boundary_x", "boundary_y")}
            seam_params.update({"h0": h0, "h1": h1,
                                "boundary_cross": int(bool(boundary_y))})
            context.submit_kernel(
                "diff6_seam_u", arrays, seam_params, exact_policy=False,
                grid_size=nlev * (h1 - h0 + 1))
    elif stagger == "y" and boundary_y and ny >= 6:
        h0, h1 = ((3, nx - 4) if boundary_x else (0, nx - 1))
        if h1 >= h0:
            seam_params = {k: v for k, v in common.items()
                           if k not in ("nys", "nxs", "variant", "wstag",
                                        "boundary_x", "boundary_y")}
            seam_params.update({"h0": h0, "h1": h1,
                                "boundary_cross": int(bool(boundary_x))})
            context.submit_kernel(
                "diff6_seam_v", arrays, seam_params, exact_policy=False,
                grid_size=nlev * (h1 - h0 + 1))
