"""Metal launcher for the Grell 3D ensemble cumulus (cu_physics=5)."""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.metal.cumulus_launch import submit_column_kernel
from woof.metal.nsas_family import _flat

_FAMILY = "g3_family"
#: Columns per command buffer; one G3 column is a long serial ensemble.
COLUMNS_PER_SUBMIT = 1024


def run_g3(owner, *, atmosphere, fields, state, cfg):
    """Run cu_physics=5 with Metal storage and native column launches."""
    from woof.config import validate_g3_dimensions
    from woof.core.g3 import (CP, ISFC_OUT, LEV_IN, LEV_OUT, SFC_OUT, G3_SLOT_COUNT,
                              XLV, G, R_V, g3_band, spread_band)
    from woof.core.physics import _NativeNtCumulusResult, _model_clock_dt

    dimensions = (cfg.maxiens, cfg.maxens, cfg.maxens2, cfg.maxens3, cfg.ensdim)
    validate_g3_dimensions(*dimensions, cfg.cugd_avedx)
    if tuple(int(d) for d in dimensions) != (1, 3, 3, 16, 144):
        raise NotImplementedError(
            "Metal Grell 3D is built for the default ensemble dimensions")
    module = state._array_module
    context = module.context
    nz, ny, nx = state.p.shape
    ncol, ncell = ny * nx, nz * ny * nx
    volume, surface = (nz, ny, nx), (ny, nx)
    driver = owner._driver

    def lane(name):
        value = None if driver is None else getattr(driver, name, None)
        if value is None:
            value = module.zeros(volume, dtype=np.float32)
        return _flat(value, volume, name, context)

    def mass(name, value):
        return _flat(value, volume, name, context)

    lev_in = module.empty((11 * ncell,), np.float32)
    submit_column_kernel(
        _FAMILY, "g3_pack_levels",
        {"u": mass("u", atmosphere["u"]), "v": mass("v", atmosphere["v"]),
         "t": mass("temperature", atmosphere["temperature"]),
         "q": mass("qv", atmosphere["qv"]),
         "pressure": mass("pressure", atmosphere["pressure"]),
         "exner": mass("exner", atmosphere["exner"]),
         "rho": mass("rho", atmosphere["rho"]),
         "thdyn": lane("gf_rthdynten"), "qvdyn": lane("gf_rqvdynten"),
         "rthlw": lane("rthratenlw"), "rthsw": lane("rthratensw"),
         "thbl": lane("gf_rthblten"), "qvbl": lane("gf_rqvblten"),
         "lev_in": lev_in},
        {"ncell": ncell}, grid_size=ncell)

    def zeroed(name, size):
        array = fields[name]
        array[...] = 0
        return array.view((size,))

    lev = zeroed("g3_lev", len(LEV_OUT) * ncell)
    sfc = zeroed("g3_sfc", len(SFC_OUT) * ncol)
    isfc = zeroed("g3_isfc", len(ISFC_OUT) * ncol)
    gsw = fields.get("gsw")
    gsw = (module.zeros(surface, dtype=np.float32) if gsw is None else gsw)
    sfc_in = module.empty((4 * ncol,), np.float32)
    submit_column_kernel(
        _FAMILY, "g3_pack_surface",
        {"p8w": _flat(atmosphere["p_interface"], (nz + 1, ny, nx), "p8w",
                      context),
         "ht": _flat(state.ht, surface, "ht", context),
         "xland": _flat(fields["xland"], surface, "xland", context),
         "gsw": _flat(gsw, surface, "gsw", context),
         "sfc_in": sfc_in, "sfc_out": sfc},
        {"ncol": ncol, "nz": nz}, grid_size=ncol)

    kpbl = fields["kpbl"] if "kpbl" in fields else module.zeros(
        surface, dtype=np.int32)
    if (not isinstance(kpbl, MetalArray) or kpbl.dtype != np.dtype(np.int32)
            or kpbl.shape != surface or not kpbl.flags.c_contiguous):
        raise TypeError("Grell 3D kpbl must be a contiguous int32 MetalArray")

    ib, ie, jb, je = g3_band(
        nx, ny, not (cfg.open_x or cfg.specified or cfg.nested),
        not (cfg.open_y or cfg.specified or cfg.nested))
    nband = max(0, ie - ib + 1) * max(0, je - jb + 1)
    sib, sie, sjb, sje = spread_band(nx, ny)
    nsp = max(0, sie - sib + 1) * max(0, sje - sjb + 1)
    tile = max(1, min(nband, COLUMNS_PER_SUBMIT))
    spread_tile = max(1, min(nsp, COLUMNS_PER_SUBMIT))
    need = max(G3_SLOT_COUNT * (nz + 1) * tile, 2 * (nz + 1) * spread_tile)
    if owner._ws is None or owner._ws.size < need:
        owner._ws = module.empty((need,), np.float32)
    ws = owner._ws

    dt = float(_model_clock_dt(cfg))
    fqi = int(getattr(state, "qi", None) is not None)
    w_in = _flat(state.w, (nz + 1, ny, nx), "w", context)
    resources = {
        "lev_in": lev_in, "w_in": w_in, "sfc_in": sfc_in,
        "kpbl_in": kpbl.view((ncol,)),
        "apr": fields["g3_apr"].view((8 * ncol,)), "lev_out": lev,
        "sfc_out": sfc, "isfc_out": isfc,
        "xf_ens": fields["g3_xf_ens"].view((int(cfg.ensdim) * ncol,)),
        "pr_ens": fields["g3_pr_ens"].view((int(cfg.ensdim) * ncol,)), "ws": ws}
    for col0 in range(0, nband, tile):
        width = min(tile, nband - col0)
        submit_column_kernel(
            _FAMILY, "g3_column", resources,
            dict(nt=width, col0=col0, nx=nx, ny=ny, nz=nz, ibegc=ib, iendc=ie,
                 jbegc=jb, jendc=je, ishallow=int(cfg.ishallow),
                 avedx=int(cfg.cugd_avedx), ichoice=int(cfg.clos_choice),
                 fqi=fqi, dt=dt, dx=float(cfg.dx), xlv=float(XLV),
                 cp=float(CP), g=float(G), rv=float(R_V)), grid_size=width)

    def lev_copy(name):
        index = LEV_OUT.index(name)
        return module.asarray(lev.view((ncell,), byte_offset=index * ncell * 4),
                              copy=True)

    def sfc_copy(name):
        index = SFC_OUT.index(name)
        return module.asarray(sfc.view((ncol,), byte_offset=index * ncol * 4),
                              copy=True)

    rqccuten = lev_copy("rqccuten_g3")
    rqicuten = lev_copy("rqicuten_g3")
    raincv = sfc_copy("raincv_g3")
    pratec = sfc_copy("pratec_g3")
    rthcuten = module.zeros((ncell,), dtype=np.float32)
    rqvcuten = module.zeros((ncell,), dtype=np.float32)
    spread_resources = {
        "cugd": lev,
        "pi_phy": lev_in.view((ncell,), byte_offset=LEV_IN.index("pi") * ncell * 4),
        "qv": lev_in.view((ncell,), byte_offset=LEV_IN.index("q") * ncell * 4),
        "rthcuten": rthcuten,
        "rqvcuten": rqvcuten, "rqccuten": rqccuten, "rqicuten": rqicuten,
        "raincv": raincv, "pratec": pratec, "ws": ws}
    for col0 in range(0, nsp, spread_tile):
        width = min(spread_tile, nsp - col0)
        submit_column_kernel(
            _FAMILY, "g3_spread", spread_resources,
            dict(nt=width, col0=col0, nx=nx, ny=ny, nz=nz, ibegc=sib,
                 iendc=sie, jbegc=sjb, jendc=sje, avedx=int(cfg.cugd_avedx),
                 fqi=fqi, dt=dt), grid_size=width)

    if cfg.lightning_option == 11:
        cloud_top = state.scratch(surface, "ltng_ktop_deep", dtype=np.int32)
        cloud_top[...] = isfc.view(
            (ncol,), byte_offset=ISFC_OUT.index("ktop_deep") * ncol * 4
        ).view(surface)
    if cfg.cu_diag:
        qi = atmosphere["qi"] if "qi" in atmosphere else module.zeros(
            volume, dtype=np.float32)
        _run_diag(
            fields, module, context, volume,
            cloud=lev.view((ncell,), byte_offset=LEV_OUT.index("gdc") * ncell * 4),
            ice=lev.view((ncell,), byte_offset=LEV_OUT.index("gdc2") * ncell * 4),
            rain=raincv, t=mass("temperature", atmosphere["temperature"]),
            p=mass("pressure", atmosphere["pressure"]),
            qv=mass("qv", atmosphere["qv"]), qc=mass("qc", atmosphere["qc"]),
            qi=mass("qi", qi), fqi=fqi, dt=dt,
            curr=float(state.elapsed_seconds),
            avg=float(cfg.convtrans_avglen_m),
            adaptive=bool(cfg.use_adaptive_time_step))

    def grid(array):
        return array.view(volume)
    return _NativeNtCumulusResult(
        owner=owner, rthcuten=grid(rthcuten), rqvcuten=grid(rqvcuten),
        rqccuten=grid(rqccuten), rqicuten=grid(rqicuten) if fqi else None,
        rainc=raincv.view(surface))


def _advance_clock(clock, dt, avg, curr, adaptive):
    """WRF convtrans_prep's window counter, evaluated as the CUDA kernel does."""
    if not np.isfinite(avg) or np.float32(avg) * np.float32(60.) < np.float32(dt):
        raise ValueError("the averaging period must span one model step: "
                         "MOD(ktau,0) would be undefined")
    words = clock.to_numpy().copy()
    ktau = int(words[0]) + 1
    count = 0 if ktau <= 1 else int(words[1])
    period = np.float32(np.float32(avg) * np.float32(60.))
    stepave = int(np.float32(period / np.float32(dt)))
    if adaptive:
        end = (float(np.float32(curr)) + float(np.float32(dt))
               + float(np.float32(.01))
               >= float(int(float(np.float32(curr)) / float(period) + 1.0))
               * float(period))
        first = float(np.float32(curr)) <= float(period)
    else:
        end = ktau % stepave == 0
        first = ktau <= stepave
    words[:] = (ktau, count + 1, int(first), int(end))
    clock[...] = words


def _run_diag(fields, module, context, volume, *, cloud, ice, rain, t, p, qv,
              qc, qi, fqi, dt, curr, avg, adaptive):
    nz, ny, nx = volume
    clock = fields["g3_diag_clock"]
    _advance_clock(clock, dt, avg, curr, adaptive)
    flat = {name: fields[key].view((fields[key].size,)) for name, key in (
        ("ca", "gd_cloud_a"), ("cb", "qc_cu"), ("ia", "gd_cloud2_a"),
        ("ib", "qi_cu"), ("frac", "gd_cldfr"), ("ra", "raincv_a"),
        ("rb", "raincv_b"))}
    nvol = nz * ny * nx
    submit_column_kernel(
        _FAMILY, "g3_diag",
        {"cloud": cloud, "ice": ice, "rain": rain, "t": t, "p": p, "qv": qv,
         "qc": qc, "qi": qi, **flat, "clock": clock},
        {"fqi": fqi, "nvol": nvol, "ncol": ny * nx}, grid_size=nvol)
    submit_column_kernel(_FAMILY, "g3_diag_finish", {"clock": clock},
                         {"unused": 0}, grid_size=1)
