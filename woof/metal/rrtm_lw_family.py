"""Typed Metal launch surface for WRF's classic RRTM longwave path.

This family follows the source order in ``woof.core.rrtm_lw``: profile
construction, SETCOEF, the 16 TAUMOL bands, RTRN, and flux assembly.  Each
stage accepts only contiguous FP32 Metal storage so the caller cannot silently
move a column through host memory.
"""

from __future__ import annotations

import numpy as np

from .array import MetalArray

_TABLE_CACHE = {}
_INT32_RESOURCES = {"itr", "ngb0", "jp", "jt", "jt1", "indself",
                    "laytrop", "layswtch", "laylow", "table_index"}
_TAUMOL_BLOB_CACHE = {}


def _validate(arrays, *, context=None):
    if not arrays:
        raise ValueError("classic RRTM LW requires device resources")
    if context is None:
        first = next(iter(arrays.values()))
        if not isinstance(first, MetalArray):
            raise TypeError("classic RRTM LW resources must be MetalArray")
        context = first.buffer.context
    for name, value in arrays.items():
        expected = np.dtype(np.int32) if name in _INT32_RESOURCES else np.dtype(np.float32)
        if (not isinstance(value, MetalArray)
                or value.dtype != expected
                or not value.flags.c_contiguous):
            raise TypeError(f"classic RRTM LW {name} must be contiguous {expected}")
        if value.buffer.context is not context:
            raise ValueError("classic RRTM LW resources must share one context")
    return context


def _shape(value, shape, name):
    if value.shape != tuple(shape):
        raise ValueError(f"classic RRTM LW {name} shape {value.shape} != {tuple(shape)}")


def _upload(context, key, host):
    host = np.ascontiguousarray(host)
    if host.dtype not in (np.dtype(np.float32), np.dtype(np.int32)):
        if host.dtype.kind in "iu":
            host = host.astype(np.int32)
        elif host.dtype.kind == "f":
            host = host.astype(np.float32)
        else:
            raise TypeError(f"classic RRTM LW table {key} has unsupported dtype {host.dtype}")
    full_key = (id(context), key)
    array = _TABLE_CACHE.get(full_key)
    if array is None:
        array = MetalArray(
            context.buffer(max(host.nbytes, 4)), host.shape or (1,), host.dtype)
        array.buffer.write(host.tobytes())
        _TABLE_CACHE[full_key] = array
    return array


def upload_tables(context, tables):
    """Upload classic RRTM's already-combined coefficient tables once/context."""
    result = {"absa": {}, "absb": {}, "selfrefc": {}, "combined": {},
              "statics": {}}
    for family in ("absa", "absb", "selfrefc", "combined", "statics"):
        values = getattr(tables, family)
        for name, value in values.items():
            result[family][name] = _upload(
                context, (id(tables), family, name), value)
    for name in ("tau", "tf", "trans", "corr1", "corr2", "rwgt"):
        result[name] = _upload(
            context, (id(tables), name), getattr(tables, name))
    result["statics"]["TOTPLNK"] = _upload(
        context, (id(tables), "statics", "TOTPLNK_F_ROW_MAJOR"),
        np.asarray(tables.statics["TOTPLNK"]).reshape((181, 16), order="F"))
    from woof.core.rrtm_lw import _o3_climatology
    result["o3wrk"] = _upload(
        context, (id(tables), "o3wrk"), _o3_climatology()[0])
    result["ppwrkh"] = _upload(
        context, (id(tables), "ppwrkh"), _o3_climatology()[1])
    return result


def upload_taumol_blob(context, tables):
    """Pack immutable GASABS coefficient tables into two Metal buffers.

    The first stores contiguous FP32 table values. The second stores
    ``(element_offset, row_count, row_width)`` for each stable table key.
    This keeps the sixteen band kernels within Metal's resource-argument cap.
    """
    key = (id(context), id(tables))
    cached = _TAUMOL_BLOB_CACHE.get(key)
    if cached is not None:
        return cached
    chunks = []
    metadata = []
    table_ids = {}

    def add(name, value):
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 0:
            value = value.reshape((1,))
        value = np.ascontiguousarray(value)
        table_ids[name] = len(metadata)
        metadata.append((sum(chunk.size for chunk in chunks),
                         value.shape[0], int(value.size // value.shape[0])))
        chunks.append(value.reshape(-1))

    absent_band_table = np.zeros((1, 1), dtype=np.float32)
    for band in range(1, 17):
        add(f"absa{band}", tables.absa[band])
        # Some RRTM bands have only a low-pressure coefficient table; band
        # 6 also has no ABSB, and band 10 has no self-reference table. Their
        # kernels select only the present path, so retain a harmless compact
        # placeholder for the unused resource IDs.
        add(f"absb{band}", tables.absb.get(band, absent_band_table))
        add(f"self{band}", tables.selfrefc.get(band, absent_band_table))
    for name in sorted(tables.combined):
        add(f"combined:{name}", tables.combined[name])
    for band in range(1, 17):
        for stream in ("A", "B"):
            name = f"combined:FRACREF{stream}C{band}"
            if name not in table_ids:
                add(name, absent_band_table)
    for name in sorted(tables.statics):
        value = np.asarray(tables.statics[name])
        if value.dtype.kind == "f" and value.size:
            add(f"statics:{name}", value)
    add("corr1", tables.corr1)
    add("corr2", tables.corr2)
    values = np.concatenate(chunks).astype(np.float32, copy=False)
    indices = np.asarray(metadata, dtype=np.int32)
    value_array = _upload(context, (id(tables), "taumol_blob"), values)
    index_array = _upload(context, (id(tables), "taumol_index"), indices)
    cached = (value_array, index_array, table_ids)
    _TAUMOL_BLOB_CACHE[key] = cached
    return cached


def submit_pack_top_down(array):
    """Pack a ``(level,y,x)`` Metal field into top-first ``(column,level)``."""
    if (not isinstance(array, MetalArray) or array.ndim != 3
            or array.dtype != np.dtype(np.float32)
            or not array.flags.c_contiguous):
        raise TypeError("classic RRTM LW packing requires contiguous FP32 rank-three input")
    nlev, ny, nx = array.shape
    if min(nlev, ny, nx) < 1:
        raise ValueError("classic RRTM LW packing dimensions must be positive")
    ncol = ny * nx
    context = array.buffer.context
    output = MetalArray(context.buffer(max(ncol * nlev * 4, 4)),
                        (ncol, nlev), np.float32)
    context.submit_kernel(
        "rrtm_lw_pack_top_down", {"input": array, "output": output},
        {"ncol": ncol, "nlev": nlev, "ny": ny, "nx": nx,
         "n": ncol * nlev}, exact_policy=False,
        grid_size=ncol * nlev, threads_per_group=128)
    return output


def submit_t8w(t, z_interface, fnm, fnp):
    """Build WRF ``phy_prep``'s interface temperatures from top-first columns."""
    arrays = {"t": t, "z_interface": z_interface, "fnm": fnm, "fnp": fnp}
    context = _validate(arrays)
    if t.ndim != 2 or z_interface.shape != (t.shape[0], t.shape[1] + 1):
        raise ValueError("classic RRTM t8w input geometry disagrees")
    ncol, nz = t.shape
    if nz < 3 or fnm.size < nz or fnp.size < nz:
        raise ValueError("classic RRTM t8w needs three levels and full weights")
    output = MetalArray(context.buffer(max(ncol * (nz + 1) * 4, 4)),
                        (ncol, nz + 1), np.float32)
    context.submit_kernel(
        "rrtm_lw_t8w", {**arrays, "tw": output},
        {"ncol": ncol, "nz": nz, "nlev": nz + 1,
         "ncell": ncol * (nz + 1)}, exact_policy=False,
        grid_size=ncol * (nz + 1), threads_per_group=128)
    return output


def submit_mm5atm(*, t, p_pa, dz, qv, qc, qr, qi, qs, qg, cldfra,
                  tw, pw_pa, tsfc, emiss, o3wrk, ppwrkh, pprof, tprof,
                  pavel, tavel, pz, tz, cldfra_out, taucloud, coldry,
                  wkl, wx, tbound, semiss, co2vmr, n2ovmr, ch4vmr,
                  r, g, nlayers):
    """Build the bottom-up MM5ATM profile and Cavallo buffer on Metal."""
    if not isinstance(t, MetalArray) or t.ndim != 2:
        raise TypeError("classic RRTM MM5ATM needs packed layer columns")
    ncol, nz = t.shape
    nlayers = int(nlayers)
    if nlayers < nz or nlayers > 256:
        raise ValueError("classic RRTM MM5ATM buffer geometry is unsupported")
    arrays = {name: value for name, value in locals().copy().items()
              if isinstance(value, MetalArray)}
    context = _validate(arrays)
    for name in ("p_pa", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "cldfra"):
        _shape(arrays[name], (ncol, nz), name)
    for name in ("tw", "pw_pa"):
        _shape(arrays[name], (ncol, nz + 1), name)
    for name in ("tsfc", "emiss"):
        _shape(arrays[name], (ncol,), name)
    for name in ("pavel", "tavel", "coldry", "cldfra_out", "taucloud"):
        _shape(arrays[name], (ncol, nlayers), name)
    for name in ("pz", "tz"):
        _shape(arrays[name], (ncol, nlayers + 1), name)
    _shape(wkl, (ncol, nlayers, 35), "wkl")
    _shape(wx, (ncol, nlayers, 4), "wx")
    _shape(tbound, (ncol,), "tbound")
    _shape(semiss, (ncol, 16), "semiss")
    _shape(o3wrk, (31,), "o3wrk")
    _shape(ppwrkh, (32,), "ppwrkh")
    if pprof.ndim != 1 or tprof.shape != pprof.shape:
        raise ValueError("classic RRTM standard-atmosphere tables disagree")
    context.submit_kernel(
        "rrtm_lw_mm5atm", arrays,
        {"ncol": ncol, "nz": nz, "nlay": nlayers,
         "nprof": pprof.size, "ncell": ncol,
         "co2vmr": float(np.float32(co2vmr)),
         "n2ovmr": float(np.float32(n2ovmr / 0.658090) * np.float32(0.658090)),
         "ch4vmr": float(np.float32(ch4vmr / 1.805423) * np.float32(1.805423)),
         "r": float(np.float32(r)), "g": float(np.float32(g))},
        exact_policy=False, grid_size=ncol, threads_per_group=128)


def submit_setcoef(*, pavel, tavel, coldry, wkl, preflog, tref):
    """Run SETCOEF's pressure/temperature interpolation and gas columns."""
    if not isinstance(pavel, MetalArray) or pavel.ndim != 2:
        raise TypeError("classic RRTM SETCOEF needs packed layer arrays")
    ncol, nlay = pavel.shape
    arrays = {"pavel": pavel, "tavel": tavel, "coldry": coldry,
              "wkl": wkl, "preflog": preflog, "tref": tref}
    context = _validate(arrays)
    for name in ("tavel", "coldry"):
        _shape(arrays[name], (ncol, nlay), name)
    _shape(wkl, (ncol, nlay, 35), "wkl")
    _shape(preflog, (59,), "preflog")
    _shape(tref, (59,), "tref")
    outputs = {}
    for name in ("colh2o", "colco2", "colo3", "coln2o", "colch4",
                 "colo2", "co2mult", "fac00", "fac01", "fac10", "fac11",
                 "forfac", "selffac", "selffrac"):
        outputs[name] = MetalArray(context.buffer(max(ncol * nlay * 4, 4)),
                                   (ncol, nlay), np.float32)
    for name in ("jp", "jt", "jt1", "indself"):
        outputs[name] = MetalArray(context.buffer(max(ncol * nlay * 4, 4)),
                                   (ncol, nlay), np.int32)
    for name in ("laytrop", "layswtch", "laylow"):
        outputs[name] = MetalArray(context.buffer(max(ncol * 4, 4)),
                                   (ncol,), np.int32)
    arrays.update(outputs)
    context.submit_kernel(
        "rrtm_lw_setcoef", arrays,
        {"ncol": ncol, "nlay": nlay, "ncell": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=128)
    return outputs


def submit_taumol_01_02(*, coef, tables):
    """Submit classic GASABS bands 1 and 2 into the 140-point outputs."""
    colh2o = coef["colh2o"]
    ncol, nlay = colh2o.shape
    context = _validate({"colh2o": colh2o})
    host_tables = upload_tables(context, tables)
    arrays = {name: coef[name] for name in (
        "colh2o", "coldry", "fac00", "fac01", "fac10", "fac11",
        "forfac", "selffac", "selffrac", "jp", "jt", "jt1",
        "indself", "laytrop")}
    arrays.update({
        "absa1": _upload(context, (id(tables), "absa", 1), tables.absa[1]),
        "absb1": _upload(context, (id(tables), "absb", 1), tables.absb[1]),
        "self1": _upload(context, (id(tables), "selfrefc", 1), tables.selfrefc[1]),
        "forref1": host_tables["combined"]["FORREFC1"],
        "fraca1": host_tables["combined"]["FRACREFAC1"],
        "fracb1": host_tables["combined"]["FRACREFBC1"],
        "absa2": _upload(context, (id(tables), "absa", 2), tables.absa[2]),
        "absb2": _upload(context, (id(tables), "absb", 2), tables.absb[2]),
        "self2": _upload(context, (id(tables), "selfrefc", 2), tables.selfrefc[2]),
        "forref2": host_tables["combined"]["FORREFC2"],
        "fraca2": host_tables["combined"]["FRACREFAC2"],
        "fracb2": host_tables["combined"]["FRACREFBC2"],
        "refparam2": host_tables["statics"]["TAUGB2__REFPARAM"],
        "corr1": host_tables["corr1"], "corr2": host_tables["corr2"],
    })
    _validate(arrays, context=context)
    for name in arrays:
        if name in coef:
            _shape(arrays[name], (ncol,) if name == "laytrop"
                   else (ncol, nlay), name)
    outputs = {
        "taug": MetalArray(context.buffer(max(ncol * nlay * 140 * 4, 4)),
                           (ncol, nlay, 140), np.float32),
        "pfrac": MetalArray(context.buffer(max(ncol * nlay * 140 * 4, 4)),
                            (ncol, nlay, 140), np.float32),
    }
    outputs["taug"].fill(0.0)
    outputs["pfrac"].fill(0.0)
    arrays.update(outputs)
    context.submit_kernel(
        "rrtm_lw_taumol_01_02", arrays,
        {"ncol": ncol, "nlay": nlay, "ngpt": 140,
         "ncell": ncol * nlay, "nspa1": int(tables.nspa[0]),
         "nspb1": int(tables.nspb[0]), "nspa2": int(tables.nspa[1]),
         "nspb2": int(tables.nspb[1]),
         "nabsa1": int(tables.absa[1].shape[0]),
         "nabsb1": int(tables.absb[1].shape[0]),
         "nself1": int(tables.selfrefc[1].shape[0]),
         "nabsa2": int(tables.absa[2].shape[0]),
         "nabsb2": int(tables.absb[2].shape[0]),
         "nself2": int(tables.selfrefc[2].shape[0]),
         "ncorr": int(tables.corr1.shape[0])}, exact_policy=False,
        grid_size=ncol * nlay, threads_per_group=128)
    return outputs


def submit_taumol_03(*, coef, tables, taug, pfrac):
    """Submit classic RRTM GASABS band 3 to the shared optical-depth arrays."""
    ncol, nlay = coef["colh2o"].shape
    context = _validate({"colh2o": coef["colh2o"], "taug": taug,
                         "pfrac": pfrac})
    table_values, table_index, ids = upload_taumol_blob(context, tables)
    names = ("colh2o", "colco2", "coln2o", "fac00", "fac01", "fac10",
             "fac11", "forfac", "selffac", "selffrac", "jp", "jt",
             "jt1", "indself", "laytrop")
    arrays = {name: coef[name] for name in names}
    arrays.update({"tables": table_values, "table_index": table_index,
                   "taug": taug, "pfrac": pfrac})
    context = _validate(arrays, context=context)
    for name in names:
        _shape(arrays[name], (ncol,) if name == "laytrop" else (ncol, nlay), name)
    _shape(taug, (ncol, nlay, 140), "taug")
    _shape(pfrac, (ncol, nlay, 140), "pfrac")
    def tid(name):
        try:
            return int(ids[name])
        except KeyError as exc:
            raise KeyError(f"RRTM GASABS table is absent: {name}") from exc
    params = {"ncol": ncol, "nlay": nlay, "ncell": ncol * nlay,
              "nspa3": int(tables.nspa[2]), "nspb3": int(tables.nspb[2]),
              "nrowa3": int(tables.absa[3].shape[0]),
              "nrowb3": int(tables.absb[3].shape[0]),
              "id_absa": tid("absa3"), "id_absb": tid("absb3"),
              "id_self3": tid("self3"), "id_forref": tid("combined:FORREFC3"),
              "id_fraca": tid("combined:FRACREFAC3"),
              "id_fracb": tid("combined:FRACREFBC3"),
              "id_absn2oa": tid("combined:ABSN2OAC3"),
              "id_absn2ob": tid("combined:ABSN2OBC3"),
              "id_h2oref": tid("statics:TAUGB3__H2OREF"),
              "id_co2ref": tid("statics:TAUGB3__CO2REF"),
              "id_n2oref": tid("statics:TAUGB3__N2OREF"),
              "id_etaref": tid("statics:TAUGB3__ETAREF")}
    context.submit_kernel("rrtm_lw_taumol_03", arrays, params,
                          exact_policy=False, grid_size=ncol * nlay,
                          threads_per_group=128)


def submit_taumol_06(*, coef, wx, tables, taug, pfrac):
    """Submit classic RRTM's CFC/CO2/H2O band 6."""
    ncol, nlay = coef["colh2o"].shape
    context = _validate({"colh2o": coef["colh2o"], "wx": wx,
                         "taug": taug, "pfrac": pfrac})
    values, table_index, ids = upload_taumol_blob(context, tables)
    names = ("colh2o", "co2mult", "fac00", "fac01", "fac10", "fac11",
             "selffac", "selffrac", "jp", "jt", "jt1", "indself", "laytrop")
    arrays = {name: coef[name] for name in names}
    arrays.update({"wx": wx, "tables": values, "table_index": table_index,
                   "taug": taug, "pfrac": pfrac})
    context = _validate(arrays, context=context)
    for name in names:
        _shape(arrays[name], (ncol,) if name == "laytrop" else (ncol, nlay), name)
    _shape(wx, (ncol, nlay, 4), "wx")
    _shape(taug, (ncol, nlay, 140), "taug")
    _shape(pfrac, (ncol, nlay, 140), "pfrac")
    def tid(name): return int(ids[name])
    params = {"ncol": ncol, "nlay": nlay, "ncell": ncol*nlay,
        "nspa6": int(tables.nspa[5]), "nrowa6": int(tables.absa[6].shape[0]),
        "id_absa6": tid("absa6"), "id_self6": tid("self6"),
        "id_cfc11": tid("combined:CFC11ADJC6"),
        "id_cfc12": tid("combined:CFC12C6"),
        "id_co2": tid("combined:ABSCO2C6"),
        "id_frac": tid("combined:FRACREFAC6")}
    context.submit_kernel("rrtm_lw_taumol_06", arrays, params,
        exact_policy=False, grid_size=ncol*nlay, threads_per_group=128)


def submit_taumol_04_05(*, coef, wx, tables, taug, pfrac):
    """Submit classic RRTM bands 4 and 5 through their shared key-species path."""
    ncol,nlay=coef["colh2o"].shape
    context=_validate({"colh2o":coef["colh2o"],"wx":wx,"taug":taug,"pfrac":pfrac})
    values,index,ids=upload_taumol_blob(context,tables)
    names=("colh2o","colco2","colo3","fac00","fac01","fac10","fac11",
        "selffac","selffrac","jp","jt","jt1","indself","laytrop")
    arrays={name:coef[name] for name in names}
    arrays.update({"wx":wx,"tables":values,"table_index":index,"taug":taug,"pfrac":pfrac})
    context=_validate(arrays,context=context)
    for name in names:_shape(arrays[name],(ncol,) if name=="laytrop" else (ncol,nlay),name)
    _shape(wx,(ncol,nlay,4),"wx")
    for band, offset, ng, lowratio, highratio, nspaix, nspbix, stephi, special in (
            (4,38,14,850.577,35.7416,3,3,6,1),
            (5,52,16,90.4894,0.900502,4,4,5,0)):
        p={"ncol":ncol,"nlay":nlay,"ncell":ncol*nlay,"band":band,
           "offset":offset,"ng":ng,"nspa":int(tables.nspa[nspaix]),
           "nspb":int(tables.nspb[nspbix]),
           "nrowa":int(tables.absa[band].shape[0]),
           "nrowb":int(tables.absb[band].shape[0]),
           "step_low":9,"step_high":stephi,"strrat_low":float(np.float32(lowratio)),
           "strrat_high":float(np.float32(highratio)),"special_high":special,
           "id_absa":int(ids[f"absa{band}"]),"id_absb":int(ids[f"absb{band}"]),
           "id_self":int(ids[f"self{band}"]),
           "id_fraca":int(ids[f"combined:FRACREFAC{band}"]),
           "id_fracb":int(ids[f"combined:FRACREFBC{band}"]),
           "id_ccl4":int(ids.get("combined:CCL4C5",0))}
        context.submit_kernel("rrtm_lw_taumol_04_05",arrays,p,
            exact_policy=False,grid_size=ncol*nlay,threads_per_group=128)


def submit_taumol_07(*,coef,tables,taug,pfrac):
    ncol,nlay=coef["colh2o"].shape
    context=_validate({"colh2o":coef["colh2o"],"taug":taug,"pfrac":pfrac})
    values,index,ids=upload_taumol_blob(context,tables)
    names=("colh2o","colo3","co2mult","fac00","fac01","fac10","fac11",
        "selffac","selffrac","jp","jt","jt1","indself","laytrop")
    arrays={name:coef[name] for name in names}
    arrays.update({"tables":values,"table_index":index,"taug":taug,"pfrac":pfrac})
    context=_validate(arrays,context=context)
    for name in names:_shape(arrays[name],(ncol,) if name=="laytrop" else (ncol,nlay),name)
    params={"ncol":ncol,"nlay":nlay,"ncell":ncol*nlay,
        "nspa":int(tables.nspa[6]),"nspb":int(tables.nspb[6]),
        "nrowa":int(tables.absa[7].shape[0]),"nrowb":int(tables.absb[7].shape[0]),
        "id_absa":int(ids["absa7"]),"id_absb":int(ids["absb7"]),
        "id_self":int(ids["self7"]),"id_co2":int(ids["combined:ABSCO2C7"]),
        "id_fraca":int(ids["combined:FRACREFAC7"]),
        "id_fracb":int(ids["combined:FRACREFBC7"])}
    context.submit_kernel("rrtm_lw_taumol_07",arrays,params,
        exact_policy=False,grid_size=ncol*nlay,threads_per_group=128)


def submit_taumol_08(*,coef,wx,tables,taug,pfrac):
    ncol,nlay=coef["colh2o"].shape
    context=_validate({"colh2o":coef["colh2o"],"wx":wx,"taug":taug,"pfrac":pfrac})
    values,index,ids=upload_taumol_blob(context,tables)
    names=("colh2o","colo3","coln2o","co2mult","fac00","fac01","fac10",
        "fac11","selffac","selffrac","jp","jt","jt1","indself","layswtch")
    arrays={name:coef[name] for name in names}
    arrays.update({"wx":wx,"tables":values,"table_index":index,"taug":taug,"pfrac":pfrac})
    context=_validate(arrays,context=context)
    for name in names:_shape(arrays[name],(ncol,) if name=="layswtch" else (ncol,nlay),name)
    params={"ncol":ncol,"nlay":nlay,"ncell":ncol*nlay,
        "nspa":int(tables.nspa[7]),"nspb":int(tables.nspb[7]),
        "nrowa":int(tables.absa[8].shape[0]),"nrowb":int(tables.absb[8].shape[0]),
        "id_absa":int(ids["absa8"]),"id_absb":int(ids["absb8"]),
        "id_self":int(ids["self8"]),"id_n2oref":int(ids["statics:TAUGB8__N2OREF"]),
        "id_h2oref":int(ids["statics:TAUGB8__H2OREF"]),"id_o3ref":int(ids["statics:TAUGB8__O3REF"]),
        "id_cfc12":int(ids["combined:CFC12C8"]),"id_cfc22":int(ids["combined:CFC22ADJC8"]),
        "id_co2a":int(ids["combined:ABSCO2AC8"]),"id_co2b":int(ids["combined:ABSCO2BC8"]),
        "id_n2oa":int(ids["combined:ABSN2OAC8"]),"id_n2ob":int(ids["combined:ABSN2OBC8"]),
        "id_fraca":int(ids["combined:FRACREFAC8"]),"id_fracb":int(ids["combined:FRACREFBC8"])}
    context.submit_kernel("rrtm_lw_taumol_08",arrays,params,
        exact_policy=False,grid_size=ncol*nlay,threads_per_group=128)


def submit_taumol_10_11(*,coef,tables,taug,pfrac):
    ncol,nlay=coef["colh2o"].shape
    context=_validate({"colh2o":coef["colh2o"],"taug":taug,"pfrac":pfrac})
    values,index,ids=upload_taumol_blob(context,tables)
    names=("colh2o","fac00","fac01","fac10","fac11","selffac","selffrac",
        "jp","jt","jt1","indself","laytrop")
    arrays={name:coef[name] for name in names}
    arrays.update({"tables":values,"table_index":index,"taug":taug,"pfrac":pfrac})
    context=_validate(arrays,context=context)
    for name in names:_shape(arrays[name],(ncol,) if name=="laytrop" else (ncol,nlay),name)
    for band,offset,ng,have_self in ((10,108,6,0),(11,114,8,1)):
        params={"ncol":ncol,"nlay":nlay,"ncell":ncol*nlay,"offset":offset,
          "ng":ng,"nspa":int(tables.nspa[band-1]),"nspb":int(tables.nspb[band-1]),
          "nrowa":int(tables.absa[band].shape[0]),"nrowb":int(tables.absb[band].shape[0]),
          "have_self":have_self,"id_absa":int(ids[f"absa{band}"]),
          "id_absb":int(ids[f"absb{band}"]),"id_self":int(ids[f"self{band}"]),
          "id_fraca":int(ids[f"combined:FRACREFAC{band}"]),
          "id_fracb":int(ids[f"combined:FRACREFBC{band}"])}
        context.submit_kernel("rrtm_lw_taumol_10_11",arrays,params,
          exact_policy=False,grid_size=ncol*nlay,threads_per_group=128)


def submit_taumol_12_16(*,coef,tables,taug,pfrac):
    ncol,nlay=coef["colh2o"].shape
    context=_validate({"colh2o":coef["colh2o"],"taug":taug,"pfrac":pfrac})
    values,index,ids=upload_taumol_blob(context,tables)
    names=("colh2o","colco2","coln2o","colch4","fac00","fac01",
        "fac10","fac11","selffac","selffrac","jp","jt","jt1",
        "indself","laytrop")
    arrays={name:coef[name] for name in names}
    arrays.update({"tables":values,"table_index":index,"taug":taug,"pfrac":pfrac})
    context=_validate(arrays,context=context)
    for name in names:_shape(arrays[name],(ncol,) if name=="laytrop" else (ncol,nlay),name)
    # Bands 12, 13, 15 and 16 use a low-pressure key-species mixture only.
    configs=((12,122,8,1,2,0.009736757,0),
             (13,130,4,1,3,16658.87,0),
             (15,136,2,3,2,0.2883201,0),
             (16,138,2,1,4,830.411,0),
             (14,134,2,2,0,0.0,1))
    for band,offset,ng,gas_a,gas_b,ratio,mode in configs:
        nrowb = (int(tables.absb[band].shape[0]) if band in tables.absb
                 else 1)
        params={"ncol":ncol,"nlay":nlay,"ncell":ncol*nlay,"offset":offset,
          "ng":ng,"nspa":int(tables.nspa[band-1]),"nspb":int(tables.nspb[band-1]),
          "nrowa":int(tables.absa[band].shape[0]),"nrowb":nrowb,
          "gas_a":gas_a,"gas_b":gas_b,"strrat":float(np.float32(ratio)),"mode":mode,
          "have_self":1,"id_absa":int(ids[f"absa{band}"]),"id_absb":int(ids[f"absb{band}"]),
          "id_self":int(ids[f"self{band}"]),"id_fraca":int(ids[f"combined:FRACREFAC{band}"]),
          "id_fracb":int(ids[f"combined:FRACREFBC{band}"])}
        context.submit_kernel("rrtm_lw_taumol_12_16",arrays,params,
          exact_policy=False,grid_size=ncol*nlay,threads_per_group=128)


def submit_taumol_09(*,coef,tables,taug,pfrac):
    ncol,nlay=coef["colh2o"].shape
    context=_validate({"colh2o":coef["colh2o"],"taug":taug,"pfrac":pfrac})
    values,index,ids=upload_taumol_blob(context,tables)
    names=("colh2o","colch4","coln2o","fac00","fac01","fac10","fac11",
        "selffac","selffrac","jp","jt","jt1","indself","laytrop","layswtch","laylow")
    arrays={name:coef[name] for name in names}
    arrays.update({"tables":values,"table_index":index,"taug":taug,"pfrac":pfrac})
    context=_validate(arrays,context=context)
    for name in names:_shape(arrays[name],(ncol,) if name in ("laytrop","layswtch","laylow") else (ncol,nlay),name)
    params={"ncol":ncol,"nlay":nlay,"ncell":ncol*nlay,"nspa":int(tables.nspa[8]),
      "nspb":int(tables.nspb[8]),"nrowa":int(tables.absa[9].shape[0]),
      "nrowb":int(tables.absb[9].shape[0]),"ng":12,
      "id_absa":int(ids["absa9"]),"id_absb":int(ids["absb9"]),
      "id_self":int(ids["self9"]),"id_h2oref":int(ids["statics:TAUGB9__H2OREF"]),
      "id_ch4ref":int(ids["statics:TAUGB9__CH4REF"]),"id_n2oref":int(ids["statics:TAUGB9__N2OREF"]),
      "id_eta":int(ids["statics:TAUGB9__ETAREF"]),"id_absn2o":int(ids["combined:ABSN2OC9"]),
      "id_fraca":int(ids["combined:FRACREFAC9"]),"id_fracb":int(ids["combined:FRACREFBC9"])}
    context.submit_kernel("rrtm_lw_taumol_09",arrays,params,
      exact_policy=False,grid_size=ncol*nlay,threads_per_group=128)


def submit_taumol_all(*,coef,wx,tables):
    """Run the full classic 16-band GASABS column stage on Metal."""
    outputs=submit_taumol_01_02(coef=coef,tables=tables)
    taug,pfrac=outputs["taug"],outputs["pfrac"]
    submit_taumol_03(coef=coef,tables=tables,taug=taug,pfrac=pfrac)
    submit_taumol_04_05(coef=coef,wx=wx,tables=tables,taug=taug,pfrac=pfrac)
    submit_taumol_06(coef=coef,wx=wx,tables=tables,taug=taug,pfrac=pfrac)
    submit_taumol_07(coef=coef,tables=tables,taug=taug,pfrac=pfrac)
    submit_taumol_08(coef=coef,wx=wx,tables=tables,taug=taug,pfrac=pfrac)
    submit_taumol_09(coef=coef,tables=tables,taug=taug,pfrac=pfrac)
    submit_taumol_10_11(coef=coef,tables=tables,taug=taug,pfrac=pfrac)
    submit_taumol_12_16(coef=coef,tables=tables,taug=taug,pfrac=pfrac)
    return outputs


def submit_make_itr(taug, *, bpade=1.0 / 0.278):
    """Build GASABS's rounded 0..5000 Pade table index on Metal."""
    context=_validate({"taug":taug})
    if taug.ndim!=3 or taug.shape[-1]!=140:
        raise ValueError("classic RRTM optical depths must have shape (ncol,nlay,140)")
    npoint=int(np.prod(taug.shape,dtype=np.int64))
    itr=MetalArray(context.buffer(max(npoint*4,4)),taug.shape,np.int32)
    context.submit_kernel("rrtm_lw_make_itr",{"taug":taug,"itr":itr},
        {"npoint":npoint,"bpade":float(np.float32(bpade))},
        exact_policy=False,grid_size=npoint,threads_per_group=128)
    return itr


def submit_rtrn_down(*, pfrac, pz, tz, tavel, cldfrac, taucloud, itr,
                     tbound, semiss, totplnk, delwave, tau_table, tf_table,
                     trans_table, ngb0, bbu, bbutot, atot, abss, down, clear_down,
                     ncol, nlay, bpade):
    """Run RTRN's downward pass, one lane per (column, g-point).

    Per-g-point lanes preserve Fortran's ordered level recurrence and avoid
    large per-thread arrays.  The same way all temporaries remain in device
    storage; the subsequent up-pass and broadband reduction are separate
    ordered submissions.
    """
    ncol, nlay = int(ncol), int(nlay)
    if ncol < 1 or nlay < 1 or nlay > 256:
        raise ValueError("classic RRTM LW Metal supports 1..256 layers")
    ngpt = 140
    arrays = {
        "pfrac": pfrac, "pz": pz, "tz": tz, "tavel": tavel,
        "cldfrac": cldfrac, "taucloud": taucloud, "itr": itr,
        "tbound": tbound, "semiss": semiss, "totplnk": totplnk,
        "delwave": delwave, "tau_table": tau_table, "tf_table": tf_table,
        "trans_table": trans_table, "ngb0": ngb0, "bbu": bbu,
        "bbutot": bbutot, "atot": atot, "abss": abss, "down": down,
        "clear_down": clear_down,
    }
    context = _validate(arrays)
    for name in ("pfrac", "itr"):
        _shape(arrays[name], (ncol, nlay, ngpt), name)
    for name in ("cldfrac", "taucloud", "tavel"):
        _shape(arrays[name], (ncol, nlay), name)
    for name in ("pz", "tz"):
        _shape(arrays[name], (ncol, nlay + 1), name)
    _shape(semiss, (ncol, 16), "semiss")
    _shape(totplnk, (181, 16), "totplnk")
    _shape(delwave, (16,), "delwave")
    _shape(ngb0, (ngpt,), "ngb0")
    for name in ("bbu", "bbutot", "atot", "abss"):
        _shape(arrays[name], (ncol, nlay, ngpt), name)
    for name in ("down", "clear_down"):
        _shape(arrays[name], (ncol, nlay + 1, ngpt), name)
    _shape(tbound, (ncol,), "tbound")
    for name in ("tau_table", "tf_table", "trans_table"):
        _shape(arrays[name], (5001,), name)
    context.submit_kernel(
        "rrtm_lw_rtrn_down", arrays,
        {"ncol": ncol, "nlay": nlay, "ngpt": ngpt,
         "ncell": ncol * ngpt, "bpade": float(np.float32(bpade))}, exact_policy=False,
        grid_size=ncol * ngpt, threads_per_group=128)


def submit_rtrn_up(*, pfrac, tavel, cldfrac, semiss, tbound, totplnk,
                   delwave, ngb0, bbu, bbutot, atot, abss, taucloud,
                   down, clear_down, up,
                   ncol, nlay):
    """Run RTRN's upward pass over the saved downward source terms."""
    ncol, nlay = int(ncol), int(nlay)
    if ncol < 1 or nlay < 1 or nlay > 256:
        raise ValueError("classic RRTM LW Metal supports 1..256 layers")
    arrays = {"pfrac": pfrac, "tavel": tavel, "cldfrac": cldfrac,
              "semiss": semiss, "tbound": tbound, "totplnk": totplnk,
              "delwave": delwave, "ngb0": ngb0, "bbu": bbu,
              "bbutot": bbutot, "atot": atot, "abss": abss,
              "taucloud": taucloud, "down": down,
              "clear_down": clear_down, "up": up}
    context = _validate(arrays)
    ngpt = 140
    for name in ("pfrac", "bbu", "bbutot", "atot", "abss"):
        _shape(arrays[name], (ncol, nlay, ngpt), name)
    _shape(tavel, (ncol, nlay), "tavel")
    _shape(cldfrac, (ncol, nlay), "cldfrac")
    _shape(taucloud, (ncol, nlay), "taucloud")
    _shape(semiss, (ncol, 16), "semiss")
    _shape(tbound, (ncol,), "tbound")
    _shape(totplnk, (181, 16), "totplnk")
    _shape(delwave, (16,), "delwave")
    _shape(ngb0, (ngpt,), "ngb0")
    for name in ("down", "clear_down", "up"):
        _shape(arrays[name], (ncol, nlay + 1, ngpt), name)
    context.submit_kernel(
        "rrtm_lw_rtrn_up", arrays,
        {"ncol": ncol, "nlay": nlay, "ngpt": ngpt,
         "ncell": ncol * ngpt}, exact_policy=False,
        grid_size=ncol * ngpt, threads_per_group=128)


def submit_rtrn_fluxes(*, pz, up, down, exner, rthratenlw, glw, olr,
                       ncol, nmodel, nlay, ny, nx, heatfac, fluxfac):
    """Reduce g-points to fluxes, heating, GLW and model-top OLR."""
    ncol, nmodel, nlay, ny, nx = map(int, (ncol, nmodel, nlay, ny, nx))
    if (ncol != ny * nx or nmodel < 1 or nmodel > nlay
            or up.shape != (ncol, nlay + 1, 140)
            or down.shape != up.shape or pz.shape != (ncol, nlay + 1)
            or exner.shape != (ncol, nmodel)
            or rthratenlw.shape != (nmodel, ny, nx)
            or glw.shape != (ny, nx) or olr.shape != (ny, nx)):
        raise ValueError("classic RRTM LW flux assembly shapes disagree")
    arrays = {"pz": pz, "up": up, "down": down, "exner": exner,
              "rthratenlw": rthratenlw, "glw": glw, "olr": olr}
    context = _validate(arrays)
    context.submit_kernel(
        "rrtm_lw_rtrn_fluxes", arrays,
        {"ncol": ncol, "nmodel": nmodel, "nlay": nlay,
         "ny": ny, "nx": nx, "ncell": ncol * (nlay + 1),
         "heatfac": float(np.float32(heatfac)),
         "fluxfac": float(np.float32(fluxfac))},
        exact_policy=False, grid_size=ncol * (nlay + 1),
        threads_per_group=128)


def submit_rtrn_transfer(*, pfrac, pz, tz, tavel, cldfrac, taucloud, itr,
                         tbound, semiss, totplnk, delwave, tau_table,
                         tf_table, trans_table, ngb0, exner, rthratenlw,
                         glw, olr, ncol, nmodel, nlay, ny, nx, heatfac,
                         bpade, fluxfac):
    """Allocate and submit classic RRTM's RTRN transfer/flux tail."""
    ncol, nlay = int(ncol), int(nlay)
    if pfrac.shape != (ncol, nlay, 140):
        raise ValueError("classic RRTM LW pfrac shape does not match geometry")
    context = _validate({"pfrac": pfrac})
    scratch = {}
    for name, shape in (
            ("bbu", (ncol, nlay, 140)),
            ("bbutot", (ncol, nlay, 140)),
            ("atot", (ncol, nlay, 140)),
            ("abss", (ncol, nlay, 140)),
            ("down", (ncol, nlay + 1, 140)),
            ("clear_down", (ncol, nlay + 1, 140)),
            ("up", (ncol, nlay + 1, 140))):
        count = int(np.prod(shape, dtype=np.int64))
        scratch[name] = MetalArray(
            context.buffer(max(count * np.dtype(np.float32).itemsize, 4)),
            shape, np.float32)
    submit_rtrn_down(
        pfrac=pfrac, pz=pz, tz=tz, tavel=tavel, cldfrac=cldfrac,
        taucloud=taucloud, itr=itr, tbound=tbound, semiss=semiss,
        totplnk=totplnk, delwave=delwave, tau_table=tau_table,
        tf_table=tf_table, trans_table=trans_table, ngb0=ngb0,
        bbu=scratch["bbu"], bbutot=scratch["bbutot"], atot=scratch["atot"],
        abss=scratch["abss"], down=scratch["down"],
        clear_down=scratch["clear_down"], ncol=ncol, nlay=nlay,
        bpade=bpade)
    submit_rtrn_up(
        pfrac=pfrac, tavel=tavel, cldfrac=cldfrac, semiss=semiss,
        tbound=tbound, totplnk=totplnk, delwave=delwave, ngb0=ngb0,
        bbu=scratch["bbu"], bbutot=scratch["bbutot"], atot=scratch["atot"],
        abss=scratch["abss"], taucloud=taucloud, down=scratch["down"],
        clear_down=scratch["clear_down"], up=scratch["up"],
        ncol=ncol, nlay=nlay)
    submit_rtrn_fluxes(
        pz=pz, up=scratch["up"], down=scratch["down"], exner=exner,
        rthratenlw=rthratenlw, glw=glw, olr=olr, ncol=ncol,
        nmodel=nmodel, nlay=nlay, ny=ny, nx=nx, heatfac=heatfac,
        fluxfac=fluxfac)
    return scratch


__all__ = ["submit_rtrn_down", "submit_rtrn_up", "submit_rtrn_fluxes",
           "submit_rtrn_transfer", "submit_pack_top_down", "submit_t8w",
           "submit_mm5atm", "submit_setcoef", "submit_taumol_01_02",
           "submit_taumol_03", "submit_taumol_06", "upload_tables",
           "submit_taumol_04_05", "submit_taumol_07", "submit_taumol_08",
           "submit_taumol_09", "submit_taumol_10_11",
           "submit_taumol_12_16", "submit_taumol_all",
           "submit_make_itr",
           "upload_taumol_blob"]
