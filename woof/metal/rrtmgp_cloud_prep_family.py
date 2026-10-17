"""Typed launch boundary for RRTMGP's cloud-fraction preparation."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from ..wrf_exact import ENABLED as WRF_EXACT


def _f32(value, shape, name):
    if (not isinstance(value, MetalArray)
            or value.dtype != np.dtype(np.float32)
            or value.shape != tuple(shape) or not value.flags.c_contiguous):
        raise TypeError(f"{name} must be contiguous float32 with shape {shape}")


def _same_context(named):
    values = tuple(named.values())
    if not values or any(not isinstance(value, MetalArray) for value in values):
        raise TypeError("RRTMGP cloud preparation requires MetalArray values")
    context = values[0].buffer.context
    if any(value.buffer.context is not context for value in values):
        raise ValueError("RRTMGP cloud preparation arrays must share a context")
    return context


def submit_cal_cldfra1(qv, qc, qi, qs, tlay, play, *, f_qi, f_qs):
    """Write WRF's selected Xu-Randall cloud-fraction arm on Metal."""
    arrays = {"qv": qv, "qc": qc, "qi": qi, "qs": qs,
              "tlay": tlay, "play": play}
    shape = qv.shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("RRTMGP cloud-fraction shape must be nonempty (ncol,nlay)")
    for name, value in arrays.items():
        _f32(value, shape, name)
    context = _same_context(arrays)
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal RRTMGP cloud fraction does not implement WRF exact-D math")
    ncell = qv.size
    output = MetalArray(context.buffer(max(ncell * 4, 4)), shape, np.float32)
    context.submit_kernel(
        "rrtmgp_cal_cldfra1",
        {**arrays, "cldfra": output},
        {"ncell": ncell, "ncol": shape[0], "nlay": shape[1],
         "f_qi": int(bool(f_qi)),
         "f_qs": int(bool(f_qs))},
        exact_policy=False, grid_size=ncell, threads_per_group=128)
    return output


def submit_mynn_cloud_supplied(qc, qi, qc_bl, qi_bl, cldfra_bl):
    """Write the two logical MYNN cloud-supplied masks as uint32 words."""
    arrays = {"qc": qc, "qi": qi, "qc_bl": qc_bl, "qi_bl": qi_bl,
              "cldfra_bl": cldfra_bl}
    shape = qc.shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("MYNN radiation clouds need nonempty (ncol,nlay) fields")
    for name, value in arrays.items():
        _f32(value, shape, name)
    context = _same_context(arrays)
    ncell = qc.size
    liquid = MetalArray(context.buffer(max(ncell * 4, 4)), shape, np.uint32)
    ice = MetalArray(context.buffer(max(ncell * 4, 4)), shape, np.uint32)
    context.submit_kernel(
        "rrtmgp_mynn_cloud_supplied",
        {**arrays, "liquid_supplied": liquid, "ice_supplied": ice},
        {"ncell": ncell, "ncol": shape[0], "nlay": shape[1]},
        exact_policy=False, grid_size=ncell,
        threads_per_group=128)
    return liquid, ice


def submit_mynn_cloud_merge(qc, qi, cldfra, *, qc_bl, qi_bl, cldfra_bl,
                            replace_cldfra):
    """Apply WRF's thresholded previous-interval MYNN water merge in place."""
    arrays = {"qc": qc, "qi": qi, "qc_bl": qc_bl, "qi_bl": qi_bl,
              "cldfra_bl": cldfra_bl}
    shape = qc.shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("MYNN radiation clouds need nonempty (ncol,nlay) fields")
    for name, value in arrays.items():
        _f32(value, shape, name)
    has_cldfra = cldfra is not None
    if has_cldfra:
        _f32(cldfra, shape, "cldfra")
        arrays["cldfra"] = cldfra
    else:
        # The shader guards this aliased pointer with has_cldfra. Passing a
        # valid in-place resource avoids inventing a host or device dummy.
        arrays["cldfra"] = qc
    context = _same_context(arrays)
    context.submit_kernel(
        "rrtmgp_mynn_cloud_merge", arrays,
        {"ncell": qc.size, "ncol": shape[0], "nlay": shape[1],
         "has_cldfra": int(has_cldfra),
         "replace_cldfra": int(bool(replace_cldfra))},
        exact_policy=False, grid_size=qc.size, threads_per_group=128)


def submit_mynn_cloud_merge_gsd41(qc, qi, cldfra, *, qc_bl, cldfra_bl, tlay,
                                  first_step):
    """Apply the GSD MYNN v4.1 in-cloud subgrid merge in place."""
    arrays = {"qc": qc, "qi": qi, "qc_bl": qc_bl, "cldfra_bl": cldfra_bl,
              "tlay": tlay}
    shape = qc.shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("MYNN radiation clouds need nonempty (ncol,nlay) fields")
    for name, value in arrays.items():
        _f32(value, shape, name)
    has_cldfra = cldfra is not None
    if has_cldfra:
        _f32(cldfra, shape, "cldfra")
    # Without a fraction the shader's guarded write aliases qc.
    arrays["cldfra"] = cldfra if has_cldfra else qc
    context = _same_context(arrays)
    context.submit_kernel(
        "rrtmgp_mynn_cloud_merge_gsd41", arrays,
        {"ncell": qc.size, "ncol": shape[0], "nlay": shape[1],
         "has_cldfra": int(has_cldfra), "first_step": int(bool(first_step))},
        exact_policy=False, grid_size=qc.size, threads_per_group=128)


_RRTMGP_RETAB = {}


def _radiation_ice_table(context):
    held = _RRTMGP_RETAB.get(id(context))
    if held is not None:
        return held
    from woof.core.rrtmg_legacy_prep import _RETAB
    data = np.ascontiguousarray(_RETAB, dtype=np.float32).reshape(-1)
    if data.size < 76:
        raise ValueError("WRF RRTMG ice-radius table must include indices 0..75")
    held = MetalArray(context.buffer(max(data.nbytes, 4)), data.shape,
                      np.float32)
    held.buffer.write(data.tobytes(), offset=held.byte_offset)
    _RRTMGP_RETAB[id(context)] = held
    return held


def submit_cloudy_background_radii(effc, effi, cldfra, tlay, xland, *,
                                   scheme, supplied_liquid=None,
                                   supplied_ice=None):
    """Apply WRF's cloudy background radii and ice temperature table on Metal."""
    from woof.core.rrtmgp import SCHEME_RADII_COUPLINGS

    if scheme not in SCHEME_RADII_COUPLINGS:
        raise ValueError(
            "cloudy-background radii require one of "
            f"{SCHEME_RADII_COUPLINGS}, got {scheme!r}")
    if not isinstance(effc, MetalArray) or len(effc.shape) != 2:
        raise TypeError("effc must be a rank-two MetalArray")
    shape = effc.shape
    _f32(effc, shape, "effc")
    for name, value in (("effi", effi), ("cldfra", cldfra),
                        ("tlay", tlay)):
        _f32(value, shape, name)
    if (not isinstance(xland, MetalArray)
            or xland.dtype != np.dtype(np.float32)
            or xland.size != shape[0] or not xland.flags.c_contiguous):
        raise TypeError(
            f"xland must be contiguous float32 with {shape[0]} values")
    if xland.shape != (shape[0],):
        xland = xland.view((shape[0],))
    arrays = {"effc": effc, "effi": effi, "cldfra": cldfra,
              "tlay": tlay, "xland": xland}
    context = _same_context(arrays)
    has_liquid = supplied_liquid is not None
    has_ice = supplied_ice is not None
    for name, value, present in (("supplied_liquid", supplied_liquid,
                                  has_liquid),
                                 ("supplied_ice", supplied_ice, has_ice)):
        if present and (not isinstance(value, MetalArray)
                        or value.dtype != np.dtype(np.uint32)
                        or value.shape != shape
                        or value.buffer.context is not context):
            raise TypeError(
                f"{name} must be same-context uint32 logical mask {shape}")
    # No-copy bit views serve as valid, unread resources when MYNN masks are
    # absent. The kernel only indexes them when the matching flag is set.
    liquid = (supplied_liquid if has_liquid else
              effc.view(shape, dtype=np.uint32))
    ice = supplied_ice if has_ice else effi.view(shape, dtype=np.uint32)
    table = _radiation_ice_table(context)
    effc_out = MetalArray(context.buffer(max(effc.nbytes, 4)), shape,
                          np.float32)
    ncell = effc.size
    if scheme == "p3":
        effi_out = effi
    else:
        effi_out = MetalArray(context.buffer(max(effi.nbytes, 4)), shape,
                              np.float32)
    context.submit_kernel(
        "rrtmgp_cloudy_background_radii",
        {"effc": effc, "effi": effi, "cldfra": cldfra, "tlay": tlay,
         "xland": xland, "retab": table,
         "supplied_liquid": liquid, "supplied_ice": ice,
         "effc_out": effc_out, "effi_out": effi_out},
        {"ncell": ncell, "ncol": shape[0], "nlay": shape[1],
         "ntable": table.size,
         "has_supplied_liquid": int(has_liquid),
         "has_supplied_ice": int(has_ice),
         "is_p3": int(scheme == "p3")},
        exact_policy=False, grid_size=ncell, threads_per_group=128)
    return effc_out, effi_out


def submit_surface_emissivity_bands(value, tables, ny, nx, *, validate=True):
    """Broadcast registered emissivity layouts to column-by-band storage."""
    if not isinstance(value, MetalArray):
        raise TypeError("Metal emissivity normalization requires a MetalArray")
    if value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous:
        raise TypeError("surface emissivity must be contiguous float32")
    if validate:
        raise NotImplementedError(
            "Metal surface-emissivity normalization requires caller-side "
            "range validation")
    ny, nx, nband = int(ny), int(nx), int(tables.nband)
    if min(ny, nx, nband) < 1:
        raise ValueError("surface emissivity dimensions must be positive")
    ncol = ny * nx
    layouts = {(): 0, (ny, nx): 1, (nband,): 2,
               (nband, ny, nx): 3}
    layout = layouts.get(value.shape)
    if layout is None:
        raise ValueError(
            "emiss must be scalar, (ny,nx), (nband,), or (nband,ny,nx); "
            f"got {value.shape}")
    output = MetalArray(value.buffer.context.buffer(ncol * nband * 4),
                        (ncol, nband), np.float32)
    value.buffer.context.submit_kernel(
        "rrtmgp_surface_emissivity",
        {"emissivity": value, "emissivity_out": output},
        {"layout": layout, "ny": ny, "nx": nx, "ncol": ncol,
         "nband": nband, "ncell": ncol * nband},
        exact_policy=False, grid_size=ncol * nband, threads_per_group=128)
    return output


def submit_hydrometeor_paths(
        plev, qc, qr=None, qi=None, qs=None, *, microphysics="kessler",
        play=None, tlay=None, nc=None, nr=None, ni=None, ns=None,
        effc=None, effr=None, effi=None, effs=None, cldfra=None,
        snow_treatment="full-snow-mass-into-ice", validate=True):
    """Produce all seven supported microphysics cloud-optics path variants."""
    from woof.core.rrtmgp import SNOW_TREATMENTS

    if snow_treatment not in SNOW_TREATMENTS:
        raise ValueError(
            f"snow_treatment must be one of {SNOW_TREATMENTS}, got "
            f"{snow_treatment!r}")
    scheme = str(microphysics).lower()
    scheme_ids = {"kessler": 0, "wsm6": 1, "thompson": 1, "nssl": 1,
                  "p3": 2, "milbrandt2": 3, "morrison": 4}
    if scheme not in scheme_ids:
        raise ValueError(
            "microphysics must be 'kessler', 'wsm6', 'thompson', 'nssl', "
            "'p3', 'milbrandt2', or 'morrison'")
    if (not isinstance(qc, MetalArray) or qc.ndim != 2
            or qc.dtype != np.dtype(np.float32) or not qc.flags.c_contiguous):
        raise TypeError("qc must be a contiguous rank-two float32 MetalArray")
    ncol, nlay = qc.shape
    if min(ncol, nlay) < 1:
        raise ValueError("hydrometeor paths require nonempty column profiles")
    _f32(plev, (ncol, nlay + 1), "plev")

    def optional(value, name):
        if value is None:
            return qc
        _f32(value, (ncol, nlay), name)
        return value

    if scheme in ("wsm6", "thompson", "nssl"):
        if any(value is None for value in (effc, effi, effs)):
            raise ValueError(f"{scheme} radii require effc, effi, and effs")
    elif scheme == "p3":
        if effc is None or effi is None:
            raise ValueError("p3 radii require effc and effi")
        if effs is not None:
            raise ValueError("p3 supplies no snow effective radius")
    elif scheme == "milbrandt2":
        if any(value is None for value in (nc, ni, ns)):
            raise ValueError("milbrandt2 radii require nc, ni and ns")
        if any(value is not None for value in (effc, effr, effi, effs)):
            raise ValueError("milbrandt2 supplies radii only from nc/ni/ns")
    elif scheme == "morrison":
        if (play is None or tlay is None
                or any(value is None for value in (nc, nr, ni, ns))):
            raise ValueError("Morrison radii require play, tlay, nc, nr, ni, ns")
        effective = (effc, effr, effi, effs)
        if any(value is not None for value in effective) and any(
                value is None for value in effective):
            raise ValueError(
                "Morrison effective radii require effc/effr/effi/effs together")
    if validate:
        raise NotImplementedError(
            "Metal hydrometeor paths require caller-side validation before "
            "device dispatch")

    context = qc.buffer.context
    values = {"plev": plev, "qc": qc,
              "qi": optional(qi, "qi"), "qs": optional(qs, "qs"),
              "play": optional(play, "play"), "tlay": optional(tlay, "tlay"),
              "nc": optional(nc, "nc"), "nr": optional(nr, "nr"),
              "ni": optional(ni, "ni"), "ns": optional(ns, "ns"),
              "effc": optional(effc, "effc"),
              "effr": optional(effr, "effr"),
              "effi": optional(effi, "effi"),
              "effs": optional(effs, "effs"),
              "cldfra": optional(cldfra, "cldfra")}
    if any(value.buffer.context is not context for value in values.values()):
        raise ValueError("hydrometeor path inputs must share one Metal context")

    from woof.core.milbrandt2_constants import CONSTANTS as my2
    outputs = tuple(
        MetalArray(context.buffer(max(qc.nbytes, 4)), qc.shape, np.float32)
        for _ in range(4))
    clwp, ciwp, reliq, dgice = outputs
    use_effective = (scheme == "morrison" and effc is not None)
    constants = {
        "my2_icexc9": float(my2["icexc9"]),
        "my2_icexi9": float(my2["icexi9"]),
        "my2_igs20": float(my2["iGS20"]),
        "my2_idms": float(my2["idms"]),
        "my2_ilam_min": float(np.float32(1.0e-10)),
        "my2_eps_q": float(np.float32(1.0e-14)),
        "my2_eps_n": float(np.float32(1.0e-3)),
        "my2_cloud_radius": float(np.float32(0.664639)),
        "my2_ice_radius": float(np.float32(1.5)),
        "my2_snow_radius": float(np.float32(1.5)),
        "morrison_cloud_lambda_const": float(np.float32(np.pi * 997.0 / 6.0)),
        "morrison_ice_lambda_const": float(np.float32(np.pi * 500.0)),
        "morrison_snow_lambda_const": float(np.float32(np.pi * 100.0)),
    }
    parameters = {
        "ncell": qc.size, "ncol": ncol, "nlay": nlay,
        "scheme": scheme_ids[scheme],
        "snow_discount": int(
            snow_treatment == "wrf-rrtmg-130um-snow-discount"),
        "has_qi": int(qi is not None), "has_qs": int(qs is not None),
        "has_cldfra": int(cldfra is not None),
        "morrison_use_effective": int(use_effective),
        **constants,
    }
    context.submit_kernel(
        "rrtmgp_hydrometeor_paths",
        {**values, "clwp": clwp, "ciwp": ciwp,
         "reliq": reliq, "dgice": dgice},
        parameters, exact_policy=False, grid_size=qc.size,
        threads_per_group=128)
    return outputs


def submit_validate_call(
        *, play, plev, tlay, tlev, tsfc, exner, qv, qc, qr, qi, qs,
        cldfra, emiss, numbers, effective, tables_lw, tables_sw,
        radii_bands, expected_top=None):
    """Run RRTMGP's fused production predicates on device-resident profiles."""
    from .array import MetalArray

    named = {
        "play": play, "plev": plev, "tlay": tlay, "tlev": tlev,
        "tsfc": tsfc, "exner": exner, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "cldfra": cldfra, "emiss": emiss,
        **numbers, **effective,
    }
    if not named or any(not isinstance(value, MetalArray)
                        for value in named.values()):
        raise TypeError("RRTMGP validation inputs must be MetalArray values")
    context = play.buffer.context
    for name, value in named.items():
        if value.buffer.context is not context:
            raise ValueError(
                f"RRTMGP validation resource {name} uses another Metal context")
        expected_dtype = np.dtype(np.float32)
        if (value.dtype != expected_dtype or not value.flags.c_contiguous):
            raise TypeError(
                f"RRTMGP validation resource {name} must be contiguous FP32")

    ncol, nlay = play.shape
    if min(ncol, nlay) < 1:
        raise ValueError("RRTMGP validation requires nonempty profiles")
    dummy = qc

    def optional(values, name):
        return values.get(name, dummy)

    def inward_float32(value, *, lower):
        """Round a FP64 table bound inward to preserve FP32 comparisons."""
        out = np.float32(value)
        if lower and float(out) < float(value):
            out = np.nextafter(out, np.float32(np.inf))
        elif not lower and float(out) > float(value):
            out = np.nextafter(out, np.float32(-np.inf))
        return float(out)

    play_lower = max(float(np.min(tables_lw.press_ref)),
                     float(np.min(tables_sw.press_ref)))
    play_upper = min(float(np.max(tables_lw.press_ref)),
                     float(np.max(tables_sw.press_ref)))
    temp_lower = max(float(np.min(tables_lw.temp_ref)),
                     float(np.min(tables_sw.temp_ref)))
    temp_upper = min(float(np.max(tables_lw.temp_ref)),
                     float(np.max(tables_sw.temp_ref)))
    if radii_bands is None:
        from woof.core.rrtmgp import EFFECTIVE_RADIUS_PLAUSIBLE_UM
        radii_bands = EFFECTIVE_RADIUS_PLAUSIBLE_UM

    flags = MetalArray(context.buffer(4), (1,), np.uint32)
    flags.fill(0)
    resources = {
        "play": play, "plev": plev, "tlay": tlay, "tlev": tlev,
        "tsfc": tsfc, "exner": exner, "qv": qv, "qc": qc, "qr": qr,
        "qi": qi, "qs": qs, "cldfra": cldfra,
        "nc": optional(numbers, "nc"), "nr": optional(numbers, "nr"),
        "ni": optional(numbers, "ni"), "ns": optional(numbers, "ns"),
        "effc": optional(effective, "effc"),
        "effr": optional(effective, "effr"),
        "effi": optional(effective, "effi"),
        "effs": optional(effective, "effs"),
        "emiss": emiss.view((emiss.size,)),
        "flags": flags,
    }
    if expected_top is None:
        has_expected_top = 0
        expected_top_value = 0.0
        top_tolerance = 0.0
    else:
        expected_top = float(expected_top)
        has_expected_top = 1
        expected_top_value = float(np.float32(expected_top))
        top_tolerance = float(np.float32(
            max(1.0e-3, abs(expected_top) * 2.0e-7)))
    parameters = {
        "ncol": ncol, "nlay": nlay, "nemiss": emiss.size, "nflag": 1,
        "morrison": int(bool(numbers)),
        "have_effective": int(bool(effective)),
        "play_lower": inward_float32(play_lower, lower=True),
        "play_upper": inward_float32(play_upper, lower=False),
        "temp_lower": float(np.float32(temp_lower)),
        "temp_upper": float(np.float32(temp_upper)),
        "effc_lower": float(np.float32(radii_bands["effc"][0])),
        "effc_upper": float(np.float32(radii_bands["effc"][1])),
        "effi_lower": float(np.float32(radii_bands["effi"][0])),
        "effi_upper": float(np.float32(radii_bands["effi"][1])),
        "effs_lower": float(np.float32(radii_bands["effs"][0])),
        "effs_upper": float(np.float32(radii_bands["effs"][1])),
        "has_expected_top": has_expected_top,
        "expected_top": expected_top_value,
        "top_tolerance": top_tolerance,
    }
    n = max(play.size, plev.size, emiss.size)
    context.submit_kernel(
        "rrtmgp_validate_call", resources, parameters,
        exact_policy=False, grid_size=n, threads_per_group=128)
    return int(flags.to_numpy()[0])
