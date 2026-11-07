"""Typed launchers for the generated TEMPO column on Metal."""

from __future__ import annotations

import numpy as np

from .array import MetalArray, MetalArrayModule

_FLOAT = np.dtype(np.float32)
_DOUBLE = np.dtype(np.float64)
TIERS = (64, 128, 256, 512)
THREADS = 64
SURFACE_NAMES = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                 "graupelncv", "sr")
STATE_FIELDS = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni",
                "nr", "nc", "nwfa", "nifa", "ng", "qb")
UPDATED_FIELDS = ("t", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
                  "nwfa", "nifa", "ng", "qb")

_HELD = {}


def _flat(array, name, floating=_FLOAT):
    if (not isinstance(array, MetalArray) or array.dtype != floating
            or not array.flags.c_contiguous):
        raise TypeError(f"TEMPO {name} requires contiguous {floating} Metal storage")
    return array.view((array.size,), byte_offset=array.byte_offset)


def _words(value):
    bits = int(np.float64(value).view(np.uint64))
    return (int(np.uint32(bits & 0xFFFFFFFF).view(np.int32)),
            int(np.uint32(bits >> 32).view(np.int32)))


def _held(xp, key, make):
    entry = _HELD.get((id(xp.context),) + key)
    if entry is None or entry[0] is not xp.context:
        entry = (xp.context, make())
        _HELD[(id(xp.context),) + key] = entry
    return entry[1]


def _grow(xp, name, words):
    entry = _HELD.get((id(xp.context), name))
    if entry is None or entry[0] is not xp.context or entry[1].size < words:
        entry = (xp.context, xp.zeros((words,), _DOUBLE))
        _HELD[(id(xp.context), name)] = entry
    return entry[1]


def column_bytes(xp, capacity):
    """(work, column state) bytes per column of one level-capacity tier."""
    def measure():
        sizes = xp.zeros((2,), _DOUBLE)
        xp.context.submit_kernel(
            f"tempo_sizes_k{capacity}", {"sizes": _flat(sizes, "sizes", _DOUBLE)},
            {"pad": 0}, exact_policy=False, grid_size=1, threads_per_group=1)
        return tuple(int(v) for v in sizes.to_numpy().view(np.uint64))
    return _held(xp, ("sizes", capacity), measure)


def submit_tempo_column(fields, outputs, *, pii, emission, emission_before, surface,
                        effc, effi, effs, refl, hail, qc_bl, fraction_bl, blob,
                        nz, dt, cfg, first_call, max_work_bytes=1 << 29):
    """Pack, run and unpack TEMPO over every (ny, nx) column.

    ``fields`` maps STATE_FIELDS to float32 Metal arrays holding theta, the
    pressure, ``w`` at the bottom face of each mass level, the layer depth
    and the moments; ``outputs`` maps UPDATED_FIELDS to the arrays written
    back.  ``emission`` (``nwfa2d``) is added before or after the column per
    ``emission_before``.  ``hail``, ``qc_bl`` and ``fraction_bl`` may be None.
    """
    capacity = next((k for k in TIERS if nz <= k), None)
    if capacity is None:
        raise ValueError("generated TEMPO column capacity is 512 levels")
    shape = fields["qv"].shape
    nt = shape[1] * shape[2]
    xp = MetalArrayModule(pii.buffer.context)
    work_bytes, col_bytes = column_bytes(xp, capacity)
    batch = max(1, min(nt, int(max_work_bytes // (work_bytes + col_bytes))))
    cols = _grow(xp, "cols", batch * col_bytes // 8)
    work = _grow(xp, "work", batch * work_bytes // 8)
    dummy = _held(xp, ("dummy",), lambda: xp.zeros((1,), _FLOAT))
    dt_lo, dt_hi = _words(dt)
    idt_lo, idt_hi = _words(1.0 / float(dt))
    emitted = emission is not None
    pack = {"f_" + n: _flat(fields[n], n) for n in STATE_FIELDS}
    pack.update(pii=_flat(pii, "pii"),
                emission=_flat(emission, "emission") if emitted else dummy,
                cols=_flat(cols, "cols", _DOUBLE))
    columns = {"blob": _flat(blob, "blob", _DOUBLE), "cols": pack["cols"],
               "work": _flat(work, "work", _DOUBLE),
               "qc_bl": _flat(qc_bl, "qc_bl") if qc_bl is not None else dummy,
               "fraction_bl": (_flat(fraction_bl, "fraction_bl")
                               if fraction_bl is not None else dummy)}
    unpack = {"o_" + n: _flat(outputs[n], n) for n in UPDATED_FIELDS}
    unpack.update(pii=pack["pii"], emission=pack["emission"], cols=pack["cols"],
                  work=columns["work"])
    unpack.update({n: _flat(surface[n], n) for n in SURFACE_NAMES})
    unpack.update(hail=_flat(hail, "hail") if hail is not None else dummy,
                  effc=_flat(effc, "effc"), effi=_flat(effi, "effi"),
                  effs=_flat(effs, "effs"), refl=_flat(refl, "refl"))
    flag = lambda name: int(bool(getattr(cfg, name)))
    call = None
    for c0 in range(0, nt, batch):
        n = min(batch, nt - c0)
        common = {"nz": int(nz), "nt": int(nt), "c0": int(c0), "n": int(n)}
        submit = lambda name, res, params: xp.context.submit_kernel(
            f"{name}_k{capacity}", res, params, exact_policy=False,
            grid_size=n, threads_per_group=THREADS)
        submit("tempo_pack", pack, {
            "dt_lo": dt_lo, "dt_hi": dt_hi, **common,
            "has_emission": int(emitted and emission_before)})
        submit("tempo_columns", columns, {
            "dt_lo": dt_lo, "dt_hi": dt_hi, "idt_lo": idt_lo, "idt_hi": idt_hi,
            **common,
            "aerosolaware": flag("aerosolaware"), "hailaware": flag("hailaware"),
            "semi_sedi": flag("semi_sedi"),
            "refl10cm_from_melting": flag("refl10cm_from_melting"),
            "cloud_condensation": flag("cloud_condensation"),
            "refl10cm": flag("refl10cm"), "re_cloud": flag("re_cloud"),
            "re_ice": flag("re_ice"), "re_snow": flag("re_snow"),
            "max_hail_diameter": flag("max_hail_diameter"),
            "first_call": int(bool(first_call)),
            "ml_for_bl_nc": int(qc_bl is not None and fraction_bl is not None)})
        call = submit("tempo_unpack", unpack, {
            "dt_lo": dt_lo, "dt_hi": dt_hi, **common,
            "has_emission": int(emitted and not emission_before),
            "has_hail": int(hail is not None), "refl_due": flag("refl10cm")})
    return call
