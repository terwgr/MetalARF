"""Metal caller of the pinned CCPP TKE-EDMF column (woof/core/satmedmf.py)."""
from __future__ import annotations

import hashlib

import numpy as np

from woof.core.satmedmf import (_INPUTS, WORKSPACE_BUDGET_BYTES, _table_manifest,
                                _validated)
from woof.metal.array import MetalArray


def _upload(context, values):
    values = np.ascontiguousarray(values)
    out = MetalArray(context.buffer(values.nbytes), values.shape, values.dtype)
    out.buffer.write(values.tobytes())
    return out


def table(context):
    """The saturation-pressure table, generated on the device and hash-checked."""
    cached = context.__dict__.get("_satmedmf_table")
    if cached is not None:
        return cached
    manifest = _table_manifest()
    n = int(manifest["elements"])
    out = MetalArray(context.buffer(n * 8), (n,), np.float64)
    context.submit_kernel("satmedmf_fpvs_table", {"table": out}, {"threads": n},
                          exact_policy=False, grid_size=n, threads_per_group=32)
    digest = hashlib.sha256(out.to_numpy().tobytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError(f"SATMEDMF fpvs table hash mismatch: {digest}, "
                         f"expected {manifest['sha256']}")
    context.__dict__["_satmedmf_table"] = out
    return out


def exner(context, p, pi_surface):
    """Layer Exner function and the surface value, in the unit's float64 order."""
    n, k = p.shape
    flat = np.asfortranarray(p).reshape(-1, order="F")
    out = MetalArray(context.buffer(flat.nbytes), flat.shape, np.float64)
    psk = MetalArray(context.buffer(n * 8), (n,), np.float64)
    context.submit_kernel(
        "satmedmf_exner",
        {"p": _upload(context, flat), "pi": _upload(context, pi_surface),
         "exner": out, "psk": psk}, {"nc": n, "km": k},
        exact_policy=False, grid_size=max(n * k, n), threads_per_group=32)
    return out.to_numpy().reshape((n, k), order="F"), psk.to_numpy()


def run_columns(context, inputs, *, dt=60.0):
    """Independent columns over the host float64 inputs the CUDA caller takes."""
    n, k, nt = _validated(inputs, dt)
    tab = table(context)
    shapes = {"du": (n, k), "dv": (n, k), "tdt": (n, k), "rtg": (n, k, nt),
              "hpbl": (n,), "dkt": (n, k), "dku": (n, k), "sflux": (n, 4)}
    out = {key: np.empty(shape, dtype=np.float64) for key, shape in shapes.items()}
    out["kpbl"] = np.empty(n, dtype=np.int32)
    capacity = (220 + 16 * nt) * k + 4096
    chunk = max(1, WORKSPACE_BUDGET_BYTES // (capacity * 8 + ((20 + 2 * nt) * k + 23) * 8))
    dt_bits = _upload(context, np.asarray([dt], dtype=np.float64))
    for first in range(0, n, chunk):
        last = min(n, first + chunk)
        nc = last - first
        packed = np.concatenate([inputs[key][first:last].reshape(nc, -1, order="F")
                                 for key in _INPUTS], axis=1)
        width = (5 + nt) * k + 5
        packed_out = MetalArray(context.buffer(nc * width * 8), (nc * width,), np.float64)
        kpbl = MetalArray(context.buffer(nc * 4), (nc,), np.int32)
        scratch = MetalArray(context.buffer(nc * capacity * 8), (nc * capacity,), np.float64)
        highwater = MetalArray(context.buffer(nc * 4), (nc,), np.int32)
        context.submit_kernel(
            "satmedmf_column",
            {"input": _upload(context, np.ascontiguousarray(packed.T).reshape(-1)),
             "output": packed_out, "kpbl": kpbl, "scratch": scratch,
             "table": tab, "highwater": highwater, "dt_bits": dt_bits},
            {"nc": nc, "km": k, "ntrac": nt},
            exact_policy=False, grid_size=nc, threads_per_group=32)
        if int(highwater.to_numpy().max()) > capacity:
            raise ValueError("SATMEDMF workspace overflow would corrupt adjacent columns")
        result = packed_out.to_numpy().reshape(width, nc).T
        offset = 0
        for key, shape in shapes.items():
            size = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            out[key][first:last] = result[:, offset:offset + size].reshape(
                (nc, *shape[1:]), order="F")
            offset += size
        out["kpbl"][first:last] = kpbl.to_numpy()
    return out
