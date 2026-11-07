"""Metal caller of the FAST_SBM column unit (woof/core/fast_sbm_cuda.py)."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from woof.core.fast_sbm_contract import (EXTRA_TABLES, FLAG_FIELDS, STATE_FIELDS,
                                         TABLE_ORDER, corrected_flags)
from woof.core.fast_sbm_cuda import DATA
from woof.metal.array import MetalArray

_KERNELS = ("fsbm_grid_fields_f32", "fsbm_prepare_columns_f32", "fsbm_columns_f32")


def table_arrays():
    """The packaged initialiser tables in kernel order, with ikr and the flags."""
    path = DATA / "tables-float32.npz"
    manifest = json.loads((DATA / "packed-manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["outputs"][path.name]:
        raise RuntimeError(f"FAST_SBM initialiser table hash differs: {path}")
    with np.load(path, allow_pickle=False) as data:
        arrays = [np.ascontiguousarray(data[name], dtype=np.float32) for name in TABLE_ORDER]
        for name in EXTRA_TABLES:
            key = "ecoal_own" if name == "ecoalmassm" else name
            kind = (np.int32 if name == "ima" else
                    np.float32 if name in ("dropradii", "rccn") else np.float64)
            arrays.append(np.ascontiguousarray(data[key], dtype=kind))
        flags = np.asarray([corrected_flags()[name] for name in FLAG_FIELDS], dtype=np.int32)
        return arrays, int(data["ikr_spon_break"][0]), flags


def _upload(context, values):
    values = np.ascontiguousarray(values).reshape(-1)
    out = MetalArray(context.buffer(max(values.nbytes, 4)), values.shape, values.dtype)
    out.buffer.write(values.tobytes())
    return out


def device_tables(context):
    """Tables packed by element type, with each table's element offset."""
    cached = context.__dict__.get("_fast_sbm_tables")
    if cached is not None:
        return cached
    arrays, ikr, flags = table_arrays()
    groups = {np.dtype(np.float32): [], np.dtype(np.float64): [], np.dtype(np.int32): []}
    offsets = []
    for a in arrays:
        group = groups[a.dtype]
        offsets.append(sum(g.size for g in group))
        group.append(a.reshape(-1))
    packed = {k: (np.concatenate(v) if v else np.zeros(1, k)) for k, v in groups.items()}
    result = {"tabf": _upload(context, packed[np.dtype(np.float32)]),
              "tabd": _upload(context, packed[np.dtype(np.float64)]),
              "tabi": _upload(context, packed[np.dtype(np.int32)]),
              "tab_offsets": _upload(context, np.asarray(offsets, dtype=np.int32)),
              "flags": _upload(context, flags), "ikr": ikr}
    context.__dict__["_fast_sbm_tables"] = result
    return result


def launch(context, step, dt, dx, dy, dims, fields, scratch):
    """Advance persistent, device-resident fields in the source's packed layout."""
    d = np.asarray(dims, dtype=np.int32)
    if d.shape != (18,):
        raise ValueError("FAST_SBM requires the eighteen source tile bounds")
    dt = np.float32(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("FAST_SBM dt must be finite and positive")
    n = int((d[7] - d[6] + 1) * (d[11] - d[10] + 1) * (d[9] - d[8] + 1))
    nz = int(d[17] - d[16] + 1)
    nc = int((d[13] - d[12] + 1) * (d[15] - d[14] + 1))
    if nz < 2 or nc < 1:
        raise ValueError("FAST_SBM needs at least two levels and a nonempty active tile")
    for name in STATE_FIELDS:
        array = fields[name]
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous or array.buffer.context is not context):
            raise TypeError(f"FAST_SBM {name} must be same-context contiguous FP32 storage")
    tables = device_tables(context)
    bounds = scratch((18,), "fsbm_dims", np.int32)
    bounds.buffer.write(d.tobytes(), offset=bounds.byte_offset)
    resources = {"dims": bounds, **{k: tables[k] for k in
        ("flags", "tab_offsets", "tabf", "tabd", "tabi")}, **fields}
    for name, shape, dtype in (
            ("work", (7 * n + 2 * nc * (nz + 2),), np.float32),
            ("kz", (nc,), np.int32), ("tally", (nc * 17,), np.float64),
            ("status", (nc,), np.int32)):
        slot = {"kz": "cloud_base"}.get(name, name)
        value = scratch(shape, "fsbm_" + slot, dtype)
        value.fill(0)
        resources[name] = value
    params = {"step": int(step), "dt": float(dt), "dx": float(np.float32(dx)),
              "dy": float(np.float32(dy)), "ikr": int(tables["ikr"])}
    halo = ((min(int(d[1]) - 1, int(d[13]) + 1) - max(int(d[12]) - 1, 1) + 1)
            * (min(int(d[3]) - 1, int(d[15]) + 1) - max(int(d[14]) - 1, 1) + 1))
    flat = {name: value.view((value.size,)) for name, value in resources.items()}
    for kernel, count in zip(_KERNELS, (halo, nc, nc)):
        context.submit_kernel(kernel, flat, params, exact_policy=False,
                              grid_size=count, threads_per_group=32)
    errors = resources["status"].to_numpy()
    bad = np.flatnonzero(errors)
    if bad.size:
        raise RuntimeError(f"FAST_SBM stopped at WRF source line {int(errors[bad[0]])} "
                           f"in column {int(bad[0])}; the partial spectral tile cannot be accepted as a forecast step")
    return resources["tally"]


def run(context, step, dt, dx, dy, dims, packed):
    """Advance host-packed fields through the same resident launch interface."""
    fields = {name: _upload(context, packed[name]) for name in STATE_FIELDS}
    slots = {}

    def scratch(shape, name, dtype=np.float32):
        if name not in slots:
            slots[name] = _upload(context, np.zeros(shape, dtype))
        return slots[name]

    tally = launch(context, step, dt, dx, dy, dims, fields, scratch)
    return ({name: fields[name].to_numpy().reshape(packed[name].shape)
             for name in STATE_FIELDS}, tally.to_numpy())


_SURFACE = ("xland", "rainnc", "rainncv", "snownc", "snowncv",
            "graupelnc", "graupelncv")
_NUMBER = {"qnc": "nc", "qnr": "nr", "qni": "ni", "qns": "ns", "qng": "ng", "qna": "nn"}


def _start_bulk_on_host(state, xland):
    """Use upstream's float64 NumPy distribution initializer once at a cold start.

    The density division and target mask retain their upstream FP32 kind.
    Existing spectral cells and all aerosol bins are retained byte for byte.
    This initializer runs on the host; subsequent spectral steps stay on Metal.
    """
    from woof.core.fast_sbm_bulk_start import bins_from_bulk, THRESHOLD
    bins = state.fast_sbm_bins[:99].to_numpy()
    bulk = {n: getattr(state, n).to_numpy() for n in ("qc", "qr", "qi", "qs", "qg")}
    total = sum(bulk.values())
    target = (np.sum(bins, axis=0) == 0) & (total > np.float32(THRESHOLD))
    count = int(np.count_nonzero(target))
    if not count:
        return
    rho = (np.float32(1.) / state.alt.to_numpy()).astype(np.float64)
    selected = {n: np.where(target, a, 0).astype(np.float64) for n, a in bulk.items()}
    planes = bins_from_bulk(np, selected, rho, xland.to_numpy()[None])
    state.fast_sbm_bins[:99][...] = np.where(target[None], planes.astype(np.float32), bins)
    import sys
    print(f"FAST_SBM bulk-to-bin start on host: {count} cells", file=sys.stderr, flush=True)


def apply(state, cfg, dt, *, refl_10cm_due=False):
    """Resident FAST_SBM model adapter, with the upstream host cold initializer."""
    from woof.core.microphysics import MicrophysicsDiagnostics
    from woof.metal.wsm6_family import submit_wsm6_prepare, submit_wsm6_finish
    if getattr(state, "fast_sbm_bins", None) is None:
        raise ValueError("FAST_SBM requires the 142 transported spectral fields; bulk moisture alone does not specify a spectral state")
    if refl_10cm_due:
        raise ValueError("FAST_SBM polar radar diagnostics are not ported")
    dt32 = np.float32(dt)
    if (not np.isfinite(dt32) or dt32 <= 0 or not np.isfinite(cfg.mp_tend_lim)
            or cfg.mp_tend_lim <= 0):
        raise ValueError("FAST_SBM dt and heating limit must be finite and positive")
    driver = getattr(state, "physics", None)
    if driver is None or "xland" not in driver.fields:
        raise ValueError("FAST_SBM requires the initialized surface xland field")
    nz, ny, nx = map(int, state.p.shape)
    width = max(int(cfg.spec_zone), 0) if cfg.specified or cfg.nested else 0
    if nz < 2 or nx <= 2 * width or ny <= 2 * width:
        raise ValueError("FAST_SBM needs at least two levels and a nonempty active tile")
    prior = getattr(state, "fast_sbm_itimestep", None)
    if prior is None and float(getattr(state, "elapsed_seconds", 0.)) == 0.:
        _start_bulk_on_host(state, driver.fields["xland"])
    if prior is None:
        prior = 1 if float(getattr(state, "elapsed_seconds", 0.)) > 0 else 0
    step = int(prior) + 1
    shape = (nz, ny, nx)
    theta = state.scratch(shape, "mp_th")
    rho = state.scratch(shape, "mp_rho")
    pii = state.scratch(shape, "mp_pii")
    dz = state.scratch(shape, "mp_dz8w")
    z = state.scratch((nz + 1, ny, nx), "mp_z8w")
    submit_wsm6_prepare(state, theta=theta, den=rho, pii=pii, dz=dz, z8w=z)
    source = {"th": theta, "rho": rho, "pii": pii, "dz8w": dz, "p": state.p,
              "w": state.w, "u": state.u, "v": state.v,
              "th_old": state.th_old, "qv_old": state.qv_old,
              "xland": driver.fields["xland"]}
    staggered = 1
    mass_winds = getattr(state, "mass_point_winds", None)
    if mass_winds is not None:
        source["u"], source["v"] = mass_winds()
        staggered = 0
    for name in ("qv", "qc", "qr", "qi", "qs", "qg"):
        source[name] = getattr(state, name)
    for name, field in _NUMBER.items():
        source[name] = getattr(state, field)
    for name in _SURFACE[1:]:
        value = state.scratch((ny, nx), "mp_" + name)
        previous = getattr(getattr(driver, "microphysics", None), name, None)
        if name.endswith("ncv") or previous is None:
            value.fill(0)
        elif value is not previous:
            value[...] = previous
        source[name] = value
    packed = {}
    for name in STATE_FIELDS:
        extent = ((143, ny + 2, nz + 1, nx + 2) if name == "chem" else
                  (ny + 2, nx + 2) if name in _SURFACE else
                  (ny + 2, nz + 1, nx + 2))
        value = state.scratch(extent, "fsbm_frame_" + name)
        value.fill(0)
        packed[name] = value
    arrays = {"bins": state.fast_sbm_bins,
              **{"src_" + n: a for n, a in source.items()},
              **{"packed_" + n: a for n, a in packed.items()}}
    context = state.p.buffer.context
    for name, value in arrays.items():
        if (not isinstance(value, MetalArray) or value.dtype != np.dtype(np.float32)
                or value.buffer.context is not context):
            raise TypeError(f"FAST_SBM {name} must be same-context FP32 storage")
    flat = {name: a.view((a.size,)) for name, a in arrays.items()}
    unpack_names = ("bins", "packed_chem") + tuple(prefix + n for n in ("th", "qv", "qc", "qr", "qi", "qs", "qg", "th_old", "qv_old", "qnc", "qnr", "qni", "qns", "qng", "qna") + _SURFACE[1:] for prefix in ("src_", "packed_"))
    params = {"nz": nz, "ny": ny, "nx": nx, "staggered": staggered}
    context.submit_kernel("fsbm_pack_state", flat, params, exact_policy=False,
                          grid_size=nz * ny * nx, threads_per_group=128)
    dims = (1, nx + 1, 1, ny + 1, 1, nz + 1, 0, nx + 1, 0, ny + 1, 1, nz + 1,
            width + 1, nx - width, width + 1, ny - width, 1, nz)
    launch(context, step, dt, cfg.dx, cfg.dy, dims, packed, state.scratch)
    context.submit_kernel("fsbm_unpack_state", {n: flat[n] for n in unpack_names}, params, exact_policy=False,
                          grid_size=nz * ny * nx, threads_per_group=128)
    state.fast_sbm_itimestep = step
    submit_wsm6_finish(state, theta=theta, dt=dt, mp_tend_lim=cfg.mp_tend_lim,
                       no_mp_heating=bool(cfg.no_mp_heating))
    sr = state.scratch((ny, nx), "mp_sr")
    context.submit_kernel("fsbm_frozen_fraction",
        {"total": source["rainncv"].view((ny * nx,)),
         "snow": source["snowncv"].view((ny * nx,)),
         "graupel": source["graupelncv"].view((ny * nx,)), "sr": sr.view((ny * nx,))},
        {"n": ny * nx}, exact_policy=False, grid_size=ny * nx, threads_per_group=128)
    return MicrophysicsDiagnostics(rainnc=source["rainnc"], rainncv=source["rainncv"], sr=sr,
        snownc=source["snownc"], snowncv=source["snowncv"],
        graupelnc=source["graupelnc"], graupelncv=source["graupelncv"])
