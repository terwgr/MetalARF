"""Launchers for the flux-form advection kernels (kernels/advection.cu).

5th-order upwind horizontal fluxes on the C-grid, periodic in x/y, zero flux
through the domain top/bottom; the vertical faces take WRF's ``vert_order``
ladder selected by ``vorder`` (3, the ladder every earlier run took, or 5,
WRF's vert_order == 5: 2nd order one face in, flux3 two in, flux5 between).
``dycore`` passes ``cfg.v_sca_adv_order`` for scalars and w and
``cfg.v_mom_adv_order`` for u and v, as WRF's advect_* routines read them.  Each launcher ADDS
the flux divergence into its tendency array.  ``add_advection_tendencies``
assembles the coupled mass fluxes ((c1h*mu+c2h)*u etc.) for the Phase-1
advection-only path, which requires w == 0 (checked): its historical
vertical flux ``rw = -(mu*w)`` was only Omega-correct there, and Task 5
retired it -- the full ``dycore.step`` path advects theta/momentum with
the diagnosed Omega from ``dycore.stage_fluxes`` and the moisture scalars
with the acoustic time-averaged fluxes built on it (WRF sumflux; see
woof.core.moist).

The launchers' ``coord`` argument is anything carrying the vertical-spacing
arrays ``rdnw`` (and ``rdn`` for w): a host ``VerticalCoord`` from
``woof.core.grid`` or a ``DomainState`` (device FP32 copies).
"""

from __future__ import annotations

try:
    import cupy as cp
except ImportError:  # Callers that launch these kernels require CUDA explicitly.
    cp = None
import numpy as np

from woof.config import RunConfig, VERTICAL_ADV_ORDERS
from woof.grid_requirements import FIFTH_ORDER_STENCIL_AXIS
from woof.core.kernels import get_kernel
from woof.core.state import (DTYPE, DomainState, mu_at_u_faces,
                              mu_at_v_faces)
from woof.wrf_exact import ADVECTION_ENABLED as WRF_ADVECTION_EXACT
from woof.wrf_exact import ENABLED as WRF_EXACT

_TPB = 128  # threads per block along i (i fastest)


def vertical_orders(cfg) -> tuple[int, int]:
    """``(v_sca_adv_order, v_mom_adv_order)`` of ``cfg``, 3 when absent.

    Scalars, theta and w take the first (WRF advect_w keys its vertical
    order on ``v_sca_adv_order``); u and v take the second.  A test
    namespace without the fields runs the ladder every earlier run took.
    """
    return (int(getattr(cfg, "v_sca_adv_order", 3) or 3),
            int(getattr(cfg, "v_mom_adv_order", 3) or 3))


def _launch(func: str, q, ru, rv, rw, tend, rdnw, fnm, fnp,
            dx: float, dy: float,
            nz: int, ny: int, nx: int, n_i: int, n_j: int, n_k: int,
            open_x: bool = False, open_y: bool = False,
            msf=None, has_msf=None, msf_shape=None,
            spec: bool = False, vorder: int = 3) -> None:
    """Common launch path: 3D grid over the target points, i fastest.

    ``open_x``/``open_y`` select the kernels' WRF open-lateral-boundary
    path (degraded near-boundary stencils, loop-bound exclusions, non-cb
    open advective terms; see advection.cu header); the defaults are the
    bitwise-unchanged periodic path.  An open axis needs >= 7 cells so the
    WRF degrade bands cannot overlap.

    ``msf`` is the 2-D map factor at the tendency points (Task 3); when
    ``has_msf`` (default: msf given) the kernel weights the horizontal
    divergence by it on BOTH the periodic and the open/specified paths
    (WRF applies mrdx/mrdy unconditionally; the reference case production path is
    specified + Lambert msf).

    ``fnm``/``fnp`` are the stretched-grid half->full interpolation
    weights consumed by the 2nd-order vertical fallback faces and
    advect_w's horizontal advecting velocities (WRF fzm/fzp).  ``spec``
    is WRF's `specified` logical: no non-cb open terms, and the
    upstream-normal-wind substitution in the boundary-adjacent u/v
    fluxes.  ``vorder`` is WRF's vertical advection order, 3 or 5.
    """
    if vorder not in VERTICAL_ADV_ORDERS:
        raise ValueError(
            f"vorder must be one of {VERTICAL_ADV_ORDERS} (WRF vert_order "
            f"ladders the kernels carry), got {vorder!r}")
    if open_x and nx < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError(f"open_x advection needs nx >= 7, got {nx}")
    if open_y and ny < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError(f"open_y advection needs ny >= 7, got {ny}")
    if has_msf is None:
        has_msf = msf is not None
    if msf is None:
        msf = cp.ones(msf_shape, dtype=DTYPE)
    kern = get_kernel("advection", func)
    grid = ((n_i + _TPB - 1) // _TPB, n_j, n_k)
    kern(grid, (_TPB, 1, 1),
         (q, ru, rv, rw, tend, cp.asarray(rdnw, dtype=DTYPE),
          cp.asarray(fnm, dtype=DTYPE), cp.asarray(fnp, dtype=DTYPE), msf,
          DTYPE(1.0 / dx), DTYPE(1.0 / dy),
          np.int32(nz), np.int32(ny), np.int32(nx),
          np.int32(open_x), np.int32(open_y), np.int32(has_msf),
          np.int32(spec), np.int32(vorder)))


def launch_flux_div_scalar(q, ru, rv, rw, tend, coord, dx, dy,
                           open_x=False, open_y=False,
                           msf=None, has_msf=None, spec=False,
                           vorder=3) -> None:
    """Add the flux divergence of scalar ``q`` (mass points) into ``tend``.

    ``msf`` (optional) is the mass-point map factor ``msft (ny, nx)``;
    ``vorder`` is ``cfg.v_sca_adv_order`` (3 or 5)."""
    try:
        from woof.metal.array import MetalArray, MetalArrayModule
    except ImportError:  # The regular CUDA install has no Metal package.
        MetalArray = ()
    if isinstance(q, MetalArray):
        arrays = (ru, rv, rw, tend, coord.rdnw, coord.fnm, coord.fnp)
        if not all(isinstance(array, MetalArray) for array in arrays):
            raise TypeError("Metal scalar advection resources must share the Metal backend")
        if has_msf is None:
            has_msf = msf is not None
        if msf is None:
            msf = MetalArrayModule(q.buffer.context).ones((q.shape[1], q.shape[2]))
        elif not isinstance(msf, MetalArray):
            raise TypeError("Metal scalar advection map factors must be a MetalArray")
        if msf.buffer.context is not q.buffer.context:
            raise ValueError("Metal scalar advection resources use different contexts")
        _require_metal_vorder("flux_div_scalar", vorder)
        nz, ny, nx = q.shape
        if open_x and nx < FIFTH_ORDER_STENCIL_AXIS:
            raise ValueError(f"open_x advection needs nx >= 7, got {nx}")
        if open_y and ny < FIFTH_ORDER_STENCIL_AXIS:
            raise ValueError(f"open_y advection needs ny >= 7, got {ny}")
        parameters = {
            "dx_inv": 1.0 / dx, "dy_inv": 1.0 / dy,
            "nz": nz, "ny": ny, "nx": nx,
            "open_x": int(bool(open_x)), "open_y": int(bool(open_y)),
            "has_msf": int(bool(has_msf)), "spec": int(bool(spec)),
        }
        # Order 3 is the established default kernel and remains available in
        # older bundles.  Only the opt-in order-5 path uses the new ABI.
        kernel_name = "flux_div_scalar"
        if int(vorder) == 5:
            parameters["vorder"] = int(vorder)
            kernel_name = "flux_div_scalar_vorder"
        q.buffer.context.submit_kernel(
            kernel_name,
            {"q": q, "ru": ru, "rv": rv, "rw": rw, "tend": tend,
             "rdnw": coord.rdnw, "fnm": coord.fnm, "fnp": coord.fnp,
             "msf": msf},
            parameters, exact_policy=WRF_ADVECTION_EXACT,
            grid_size=nz * ny * nx)
        return

    nz, ny, nx = q.shape
    _launch("flux_div_scalar", q, ru, rv, rw, tend, coord.rdnw,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx, ny, nz, open_x, open_y,
            msf, has_msf, (ny, nx), spec, vorder)


def launch_flux_div_u(u, ru, rv, rw, tend, coord, dx, dy,
                      open_x=False, open_y=False,
                      msf=None, has_msf=None, spec=False,
                      vorder=3) -> None:
    """Add the flux divergence of ``u`` (u-points) into ``tend``.

    ``msf`` (optional) is the u-point map factor ``msfu (ny, nx+1)``;
    ``vorder`` is ``cfg.v_mom_adv_order`` (3 or 5)."""
    nz, ny, nxp1 = u.shape
    nx = nxp1 - 1
    try:
        from woof.metal.array import MetalArray
    except ImportError:
        MetalArray = ()
    if isinstance(u, MetalArray):
        _launch_metal_advection(
            "flux_div_u", u, ru, rv, rw, tend, coord.rdnw, coord.fnm,
            coord.fnp, dx, dy, nz, ny, nx, nx + 1, open_x, open_y,
            msf, has_msf, (ny, nx + 1), spec, vorder)
        return
    _launch("flux_div_u", u, ru, rv, rw, tend, coord.rdnw,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx + 1, ny, nz, open_x, open_y,
            msf, has_msf, (ny, nx + 1), spec, vorder)


def _require_metal_vorder(func, vorder):
    # Keep this in step with the CUDA launcher's WRF vertical-order ladder.
    if int(vorder) not in VERTICAL_ADV_ORDERS:
        raise NotImplementedError(
            f"Metal {func} supports vertical advection orders "
            f"{VERTICAL_ADV_ORDERS}, got {vorder}")


def _launch_metal_advection(func, q, ru, rv, rw, tend, rdnw, fnm, fnp,
                            dx, dy, nz, ny, nx, n_i, open_x, open_y,
                            msf, has_msf, msf_shape, spec, vorder=3):
    from woof.metal.array import MetalArray, MetalArrayModule
    _require_metal_vorder(func, vorder)
    resources = (ru, rv, rw, tend, rdnw, fnm, fnp)
    if not all(isinstance(array, MetalArray) for array in resources):
        raise TypeError(f"Metal {func} resources must share the Metal backend")
    if has_msf is None:
        has_msf = msf is not None
    if msf is None:
        msf = MetalArrayModule(q.buffer.context).ones(msf_shape)
    elif not isinstance(msf, MetalArray):
        raise TypeError(f"Metal {func} map factors must be a MetalArray")
    if msf.buffer.context is not q.buffer.context:
        raise ValueError(f"Metal {func} resources use different contexts")
    if open_x and nx < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError(f"open_x advection needs nx >= 7, got {nx}")
    if open_y and ny < FIFTH_ORDER_STENCIL_AXIS:
        raise ValueError(f"open_y advection needs ny >= 7, got {ny}")
    parameters = {
        "dx_inv": 1.0 / dx, "dy_inv": 1.0 / dy,
        "nz": nz, "ny": ny, "nx": nx,
        "open_x": int(bool(open_x)), "open_y": int(bool(open_y)),
        "has_msf": int(bool(has_msf)), "spec": int(bool(spec)),
    }
    kernel_name = func
    if int(vorder) == 5:
        parameters["vorder"] = int(vorder)
        kernel_name += "_vorder"
    field = {"flux_div_u": "u", "flux_div_v": "v", "flux_div_w": "w"}[func]
    vertical_spacing = "rdn" if func == "flux_div_w" else "rdnw"
    q.buffer.context.submit_kernel(
        kernel_name, {field: q, "ru": ru, "rv": rv, "rw": rw, "tend": tend,
              vertical_spacing: rdnw, "fnm": fnm, "fnp": fnp, "msf": msf},
        parameters, exact_policy=WRF_ADVECTION_EXACT,
        # V has ny+1 staggered rows; launching the mass-grid count leaves
        # its final row-band/levels unwritten. U and W already have q.size
        # points too, so the actual target extent is the shared launch count.
        grid_size=int(q.size))


def launch_flux_div_v(v, ru, rv, rw, tend, coord, dx, dy,
                      open_x=False, open_y=False,
                      msf=None, has_msf=None, spec=False,
                      vorder=3) -> None:
    """Add the flux divergence of ``v`` (v-points) into ``tend``.

    ``msf`` (optional) is the v-point map factor ``msfv (ny+1, nx)``;
    ``vorder`` is ``cfg.v_mom_adv_order`` (3 or 5)."""
    nz, nyp1, nx = v.shape
    ny = nyp1 - 1
    try:
        from woof.metal.array import MetalArray
    except ImportError:
        MetalArray = ()
    if isinstance(v, MetalArray):
        _launch_metal_advection(
            "flux_div_v", v, ru, rv, rw, tend, coord.rdnw, coord.fnm,
            coord.fnp, dx, dy, nz, ny, nx, nx, open_x, open_y,
            msf, has_msf, (ny + 1, nx), spec, vorder)
        return
    _launch("flux_div_v", v, ru, rv, rw, tend, coord.rdnw,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx, ny + 1, nz, open_x, open_y,
            msf, has_msf, (ny + 1, nx), spec, vorder)


def launch_flux_div_w(w, ru, rv, rw, tend, coord, dx, dy,
                      open_x=False, open_y=False,
                      msf=None, has_msf=None, spec=False,
                      vorder=3) -> None:
    """Add the flux divergence of ``w`` (interior and top w-points)
    into ``tend``.  Uses ``coord.rdn`` (w-level spacing), not ``rdnw``.

    ``msf`` (optional) is the mass-point map factor ``msft (ny, nx)``;
    ``vorder`` is ``cfg.v_sca_adv_order`` (WRF advect_w keys its vertical
    order on the scalar order), 3 or 5."""
    nzp1, ny, nx = w.shape
    nz = nzp1 - 1
    try:
        from woof.metal.array import MetalArray
    except ImportError:
        MetalArray = ()
    if isinstance(w, MetalArray):
        _launch_metal_advection(
            "flux_div_w", w, ru, rv, rw, tend, coord.rdn, coord.fnm,
            coord.fnp, dx, dy, nz, ny, nx, nx, open_x, open_y,
            msf, has_msf, (ny, nx), spec, vorder)
        return
    _launch("flux_div_w", w, ru, rv, rw, tend, coord.rdn,
            coord.fnm, coord.fnp, dx, dy,
            nz, ny, nx, nx, ny, nz + 1, open_x, open_y,
            msf, has_msf, (ny, nx), spec, vorder)


def build_metal_stage_fluxes(state, cfg, *, periodic_faces=False):
    """Build coupled ``ru``, ``rv``, and diagnosed ``ww`` on Metal.

    ``periodic_faces`` is reserved for the legacy advection-only caller,
    whose historical mass-face helpers always used periodic wrapping even
    when the later tendency launch options selected open boundaries.
    Normal RK stage fluxes remain boundary-aware.
    """
    from woof.metal.array import MetalArray
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal stage flux construction needs a Metal DomainState")
    nz, ny, nx = state.p.shape
    mux = state.scratch((ny, nx + 1), "rk_mux")
    muy = state.scratch((ny + 1, nx), "rk_muy")
    ru = state.scratch((nz, ny, nx + 1), "rk_ru")
    rv = state.scratch((nz, ny + 1, nx), "rk_rv")
    ww = state.scratch((nz + 1, ny, nx), "rk_ww")
    arrays = (state.mup, state.mub2d, mux, muy, state.u, state.v,
              state.c1h, state.c2h, state.msfu, state.msfv,
              ru, rv, state.dnw, state.msft, ww)
    if not all(isinstance(value, MetalArray) for value in arrays):
        raise TypeError("Metal stage flux resources cannot mix backends")
    context = state._array_module.context
    forced = bool(getattr(cfg, "specified", False)
                  or getattr(cfg, "nested", False))
    boundary_x = bool(cfg.open_x or forced) and not periodic_faces
    boundary_y = bool(cfg.open_y or forced) and not periodic_faces
    context.submit_kernel(
        "stage_face_masses",
        {"mup": state.mup, "mub2d": state.mub2d,
         "mux": mux, "muy": muy},
        {"wrf_exact": int(WRF_EXACT),
         "boundary_x": int(boundary_x),
         "boundary_y": int(boundary_y),
         "ny": ny, "nx": nx},
        exact_policy=WRF_EXACT, grid_size=ny * nx)
    context.submit_kernel(
        "couple_ru",
        {"u": state.u, "c1h": state.c1h, "c2h": state.c2h,
         "mux": mux, "msfu": state.msfu, "ru": ru},
        {"has_msf": int(state.has_msf), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=WRF_EXACT, grid_size=nz * ny * (nx + 1))
    context.submit_kernel(
        "couple_rv",
        {"v": state.v, "c1h": state.c1h, "c2h": state.c2h,
         "muy": muy, "msfv": state.msfv, "rv": rv},
        {"has_msf": int(state.has_msf),
         "reciprocal": int(WRF_EXACT),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=WRF_EXACT, grid_size=nz * (ny + 1) * nx)
    context.submit_kernel(
        "omega_column_scan",
        {"ru": ru, "rv": rv, "dnw": state.dnw, "c1h": state.c1h,
         "msft": state.msft, "ww": ww},
        {"rdx": float(np.float32(1.0) / np.float32(cfg.dx)),
         "rdy": float(np.float32(1.0) / np.float32(cfg.dy)),
         "nz": nz, "ny": ny, "nx": nx,
         "has_msf": int(state.has_msf)},
        exact_policy=WRF_EXACT, grid_size=ny * nx)
    return ru, rv, ww


def add_advection_tendencies(state: DomainState, cfg: RunConfig) -> None:
    """Accumulate advective tendencies into ``ru_t, rv_t, rw_t, rth_t``.

    Phase-1 advection-only path (``dycore.step(acoustic=False)``): builds
    the coupled mass fluxes ``ru = (c1h*<mu>_x + c2h) * u``,
    ``rv = (c1h*<mu>_y + c2h) * v`` in scratch (mu = mub2d + mu' averaged
    to the staggered faces by the shared ``state`` helpers, periodic), then
    launches the four flux-divergence kernels; theta is advected in coupled
    form (flux divergence of mu*theta with q = total theta).

    The eta-directed vertical flux is zero: the historical placeholder
    ``rw = -(mu*w)`` was dimensionally wrong as Omega = mu*d(eta)/dt except
    in its only valid regime w == 0, so Task 5 retired it -- this path now
    *requires* w == 0 (raises otherwise), and the full ``dycore.step`` path
    advects with the diagnosed Omega from ``dycore.stage_fluxes``.
    """
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # The regular CUDA install has no Metal package.
        MetalArray = ()
    if isinstance(state.w, MetalArray):
        _add_metal_advection_tendencies(state, cfg)
        return

    nz, ny, nx = state.p.shape
    if bool(cp.any(state.w)):
        raise ValueError(
            "add_advection_tendencies (advection-only path) requires "
            "w == 0: its former vertical flux rw = -(mu*w) was only the "
            "eta mass flux Omega for w == 0.  The full dycore.step path "
            "advects with the diagnosed Omega from dycore.stage_fluxes.")
    mu = state.total_mu()                            # (ny, nx) total dry mass
    c1h = state.c1h[:, None, None]
    c2h = state.c2h[:, None, None]

    ru = state.scratch((nz, ny, nx + 1), "adv_ru")
    rv = state.scratch((nz, ny + 1, nx), "adv_rv")
    rw = state.scratch((nz + 1, ny, nx), "adv_rw")

    ru[...] = (c1h * mu_at_u_faces(mu)[None] + c2h) * state.u
    rv[...] = (c1h * mu_at_v_faces(mu)[None] + c2h) * state.v
    if state.has_msf:                                # WRF couple_momentum
        ru /= state.msfu[None]
        rv /= state.msfv[None]
    rw[...] = 0.0                                    # Omega == 0 when w == 0

    vsca, vmom = vertical_orders(cfg)
    launch_flux_div_scalar(state.total_theta(), ru, rv, rw, state.rth_t,
                           state, cfg.dx, cfg.dy,
                           msf=state.msft, has_msf=state.has_msf, vorder=vsca)
    launch_flux_div_u(state.u, ru, rv, rw, state.ru_t, state, cfg.dx, cfg.dy,
                      msf=state.msfu, has_msf=state.has_msf, vorder=vmom)
    launch_flux_div_v(state.v, ru, rv, rw, state.rv_t, state, cfg.dx, cfg.dy,
                      msf=state.msfv, has_msf=state.has_msf, vorder=vmom)
    launch_flux_div_w(state.w, ru, rv, rw, state.rw_t, state, cfg.dx, cfg.dy,
                      msf=state.msft, has_msf=state.has_msf, vorder=vsca)


def _add_metal_advection_tendencies(state: DomainState,
                                    cfg: RunConfig) -> None:
    """Metal realization of the isolated advection-only tendency caller.

    The guard intentionally performs a readback of ``w`` before deciding
    whether this legacy path is valid. That is a synchronization and host
    transfer; it keeps the refusal exact without adding a reduction kernel
    or silently accepting a nonzero vertical wind.
    """
    from woof.metal.array import MetalArray

    if not isinstance(state.w, MetalArray):
        raise TypeError("Metal advection tendencies require Metal state fields")
    if np.any(state.w.to_numpy()):
        raise ValueError(
            "add_advection_tendencies (advection-only path) requires "
            "w == 0: its former vertical flux rw = -(mu*w) was only the "
            "eta mass flux Omega for w == 0.  The full dycore.step path "
            "advects with the diagnosed Omega from dycore.stage_fluxes.")

    nz, ny, nx = state.p.shape
    context = state._array_module.context
    ru, rv, _ww = build_metal_stage_fluxes(
        state, cfg, periodic_faces=True)
    theta = state.scratch((nz, ny, nx), "adv_theta")
    context.submit_kernel(
        "total_theta",
        {"thp": state.thp, "thb": state.thb, "theta": theta},
        {"nz": nz, "ny": ny, "nx": nx,
         "base3d": int(state.thb.ndim == 3)},
        exact_policy=WRF_EXACT, grid_size=nz * ny * nx)
    rw = state.scratch((nz + 1, ny, nx), "adv_rw")
    rw.fill(0.0)

    # The four existing divergence launchers own staggered shapes, map
    # factors, exact-policy selection, and additive tendency semantics.
    launch_flux_div_scalar(theta, ru, rv, rw, state.rth_t, state,
                           cfg.dx, cfg.dy, msf=state.msft,
                           has_msf=state.has_msf)
    launch_flux_div_u(state.u, ru, rv, rw, state.ru_t, state,
                      cfg.dx, cfg.dy, msf=state.msfu,
                      has_msf=state.has_msf)
    launch_flux_div_v(state.v, ru, rv, rw, state.rv_t, state,
                      cfg.dx, cfg.dy, msf=state.msfv,
                      has_msf=state.has_msf)
    launch_flux_div_w(state.w, ru, rv, rw, state.rw_t, state,
                      cfg.dx, cfg.dy, msf=state.msft,
                      has_msf=state.has_msf)


def advect_scalar_rk3_periodic_test(q0, u0: float, dx: float,
                                    t_end: float, cfl: float) -> np.ndarray:
    """1-D uniform-flow RK3 advection driver over the GPU scalar kernel.

    Test harness only: runs ``flux_div_scalar`` with nz = ny = 1 (vertical
    and y fluxes vanish) and constant coupled flux ru = u0, i.e. mu = 1.
    Returns the advected profile after ``t_end`` as a host float32 array.
    """
    q0 = np.asarray(q0)
    nx = q0.size
    q = cp.asarray(q0, dtype=DTYPE).reshape(1, 1, nx)
    ru = cp.full((1, 1, nx + 1), u0, dtype=DTYPE)
    rv = cp.zeros((1, 2, nx), dtype=DTYPE)
    rw = cp.zeros((2, 1, nx), dtype=DTYPE)
    tend = cp.zeros((1, 1, nx), dtype=DTYPE)
    rdnw = np.full(1, -1.0)                          # unused: fz == 0 at nz=1
    kern = get_kernel("advection", "flux_div_scalar")
    grid = ((nx + _TPB - 1) // _TPB, 1, 1)
    args_tail = (cp.asarray(rdnw, dtype=DTYPE),
                 cp.zeros(1, dtype=DTYPE),               # fnm/fnp unused
                 cp.zeros(1, dtype=DTYPE),               # at nz == 1
                 cp.ones((1, nx), dtype=DTYPE),          # msf == 1
                 DTYPE(1.0 / dx),
                 DTYPE(1.0 / dx), np.int32(1), np.int32(1), np.int32(nx),
                 np.int32(0), np.int32(0), np.int32(0),  # periodic path
                 np.int32(0),
                 np.int32(3))                             # vorder: inert at nz=1

    def rhs(qs):
        tend[...] = 0
        kern(grid, (_TPB, 1, 1), (qs, ru, rv, rw, tend) + args_tail)
        return tend

    nsub = max(1, int(np.ceil(t_end / (cfl * dx / abs(u0)))))
    dt = t_end / nsub
    for _ in range(nsub):                            # WRF RK3 (dt/3, dt/2, dt)
        q1 = q + (dt / 3.0) * rhs(q)
        q2 = q + (dt / 2.0) * rhs(q1)
        q = q + dt * rhs(q2)
    return cp.asnumpy(q).ravel()
