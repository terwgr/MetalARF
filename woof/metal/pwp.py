"""Metal 3DPWP ocean (woof.ocean.mixed_layer)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .array import MetalArray, MetalArrayModule

_REGISTRY = Path(__file__).with_name("kernel_signatures.json")


@lru_cache(maxsize=1)
def _signatures():
    kernels = [k for k in json.loads(_REGISTRY.read_text())["kernels"]
               if k["source"] == "woof/metal/shaders/pwp_family.metal"]
    return {k["cuda_symbol"]: (k["name"], tuple(r[0] for r in k["resources"]),
                               tuple(p for p in k["parameters"] if p[0] != "levels"),
                               k["threads_per_group"]) for k in kernels}


class Kernels:
    """``woof.ocean.mixed_layer.kernels(levels)`` on Metal storage."""

    def __init__(self, levels):
        self.levels = int(levels)

    def launch(self, cuda_name, count, args):
        name, resources, parameters, tpg = _signatures()[cuda_name]
        arrays = [a for a in args if isinstance(a, MetalArray)]
        scalars = [a for a in args if not isinstance(a, MetalArray)]
        if len(arrays) != len(resources) or len(scalars) != len(parameters):
            raise ValueError(f"3DPWP {cuda_name} takes {len(resources)} arrays and {len(parameters)} scalars")
        context = arrays[0].buffer.context
        if any(a.buffer.context is not context or not a.flags.c_contiguous for a in arrays):
            raise TypeError("3DPWP resources must be contiguous MetalArrays on one context")
        values = {key: (int(v) if kind == "int32" else float(np.float32(v)))
                  for (key, kind), v in zip(parameters, scalars)}
        if count:
            context.submit_kernel(
                name, {k: a.view((a.size,)) for k, a in zip(resources, arrays)},
                {**values, "levels": self.levels}, exact_policy=False,
                grid_size=int(count), threads_per_group=tpg)


def host(value):
    return value.to_numpy() if isinstance(value, MetalArray) else np.asarray(value)


def module_for(value):
    return MetalArrayModule(value.buffer.context)


def initialize(fields, cfg, initial=None, *, start=True):
    """``woof.ocean.mixed_layer.initialize`` with host-side input checks."""
    import warnings
    from woof.ocean.mixed_layer import PROFILE_FIELDS, ocean_land, _launch
    xp = module_for(fields['tsk'])
    initial = initial or {}
    levels = cfg.ocean_levels
    shape = (levels, *fields['tsk'].shape)
    for name in PROFILE_FIELDS:
        value = initial.get(name)
        if value is None:
            fields[name] = xp.zeros(shape, np.float32)
        else:
            a = np.asarray(host(value), dtype=np.float32)
            if a.shape != shape:
                raise ValueError(f'{name.upper()} must have shape {shape}, got {a.shape}')
            fields[name] = xp.asarray(np.ascontiguousarray(a).copy())
    fields['om_ml'] = xp.zeros(shape[1:], np.float32)
    fields['_pwp_base'] = xp.zeros((4*levels, *shape[1:]), np.float32)
    fields['_pwp_tend'] = xp.zeros((4*levels, *shape[1:]), np.float32)
    fields['_pwp_w'] = xp.zeros(shape, np.float32)
    fields['_pwp_p'] = xp.zeros(shape, np.float32)
    fields['_pwp_status'] = xp.zeros((1,), np.int32)
    fields['_pwp_land'] = ocean_land(fields)
    for name in ('_pwp_mx', '_pwp_my'):
        fields[name] = xp.asarray(np.ones(shape[1:], np.float32))
    if not start:
        return
    module = Kernels(levels)
    if cfg.ocean_z:
        from woof.surface_ocean import validate_pwp_profile
        validate_pwp_profile(cfg)
        if initial:
            raise ValueError('ocean_z/ocean_t/ocean_s and explicit ocean_initial both supply profiles')
        for name, values in (('om_depth', cfg.ocean_z), ('om_tmp', cfg.ocean_t), ('om_s', cfg.ocean_s)):
            fields[name][...] = xp.asarray(np.ascontiguousarray(np.broadcast_to(
                np.asarray(values, np.float32)[:, None, None], shape)))
        initial = {name: fields[name] for name in ('om_depth', 'om_tmp', 'om_s')}
    required = ('om_tmp', 'om_s', 'om_depth')
    supplied = [name in initial for name in required]
    if any(supplied) and not all(supplied):
        raise ValueError('3DPWP analysis requires OM_TMP, OM_S and OM_DEPTH together; partial profiles corrupt density and mixing')
    if not any(supplied):
        if cfg.ocean_mld_climatology != 'configured_uniform_mld':
            raise ValueError('WOA18 climatology initialization needs forecast coordinates and UTC start through initialize_ocean; configured_uniform_mld is the explicit spatially uniform alternative')
        if cfg.oml_hml0 <= 0:
            raise ValueError("3DPWP configured_uniform_mld fallback requires positive oml_hml0; no analysis mixed-layer depth was supplied")
        warnings.warn('OCEAN INITIALIZATION: configured_uniform_mld fallback; '
                      f'MLD {cfg.oml_hml0:g} m, 35 PSU, resting currents, configured OML lapse rate, 5 m layers. '
                      'No public ocean analysis was supplied.', RuntimeWarning, stacklevel=3)
        _launch(module, 'pwp_fallback', fields['om_tmp'].size,
                tuple(fields[n] for n in ('om_tmp', 'om_s', 'om_u', 'om_v', 'om_depth', 'tsk')) +
                (cfg.oml_hml0, cfg.oml_gamma, fields['tsk'].size))
    wet = host(fields['_pwp_land']) > 1.5
    z = host(fields['om_depth'])
    with np.errstate(invalid='ignore'):
        bad = (~np.isfinite(z) | (z <= 0))
        bad[1:] |= z[1:] <= z[:-1]
        prof = {name: host(fields[name]) for name in ('om_tmp', 'om_s', 'om_u', 'om_v')}
        for name in prof:
            bad |= ~np.isfinite(prof[name])
        bad |= (prof['om_tmp'] < 260) | (prof['om_tmp'] > 330)
        bad |= (prof['om_s'] < 0) | (prof['om_s'] > 50)
    if bool(np.any(bad & wet[None])):
        raise ValueError('3DPWP analysis has missing, nonfinite or nonphysical ocean columns, or non-increasing positive layer depths; repair the ocean analysis before coupling')
    for name, fallback in (('om_tini', 'om_tmp'), ('om_sini', 'om_s')):
        value = initial.get(name, fields[fallback])
        fields[name][...] = value if isinstance(value, MetalArray) else xp.asarray(
            np.ascontiguousarray(np.asarray(value, np.float32)))
    tini, sini = host(fields['om_tini']), host(fields['om_sini'])
    with np.errstate(invalid='ignore'):
        bad_ref = (~np.isfinite(tini) | ~np.isfinite(sini) | (tini < 260) | (tini > 330) | (sini < 0) | (sini > 50))
    if bool(np.any(bad_ref & wet[None])):
        raise ValueError('3DPWP reference temperature and salinity must be finite physical profiles')
    _launch(module, 'pwp_diagnose', fields['tsk'].size,
            tuple(fields[name] for name in ('om_tmp', 'om_s', 'om_depth', 'om_tini', 'om_sini', 'om_density', 'om_ml', '_pwp_land')) +
            (fields['tsk'].size,))
    tsk = host(fields['tsk'])
    fields['tsk'][...] = xp.asarray(np.where(wet, host(fields['om_tmp'])[0], tsk).astype(np.float32))


def step(fields, *, dt, dx, dy, stbolt=5.67051e-8):
    """The CUDA two-stage predictor/corrector on Metal storage."""
    from woof.ocean.mixed_layer import PROGNOSTIC, _launch
    if dt <= 0 or dx <= 0 or dy <= 0:
        raise ValueError('3DPWP dt, dx and dy must be positive')
    levels, ny, nx = fields['om_tmp'].shape
    n = nx * ny; module = Kernels(levels)
    coupled_ice = int('ice_ocean_coupled' in fields)
    ice_mask = fields['ice_ocean_coupled'] if coupled_ice else fields['_pwp_land']
    base = fields['_pwp_base']
    for j, name in enumerate(PROGNOSTIC):
        base[j * levels:(j + 1) * levels][...] = fields[name]
    common = tuple(fields[name] for name in (*PROGNOSTIC, 'om_depth'))
    for first, stage_dt in ((1, 2 * dt), (0, 0.5 * dt)):
        _launch(module, 'pwp_geometry', n, common +
                tuple(fields[name] for name in ('om_tini', 'om_sini', '_pwp_land', '_pwp_mx', '_pwp_my', '_pwp_w', '_pwp_p')) +
                (1 / dx, 1 / dy, nx, ny))
        _launch(module, 'pwp_transport', n * levels, common +
                tuple(fields[name] for name in ('_pwp_w', '_pwp_p', '_pwp_land', '_pwp_mx', '_pwp_my', '_pwp_tend')) +
                (1 / dx, 1 / dy, nx, ny, first))
        _launch(module, 'pwp_column', n,
                tuple(fields[name] for name in (*PROGNOSTIC, 'om_density', 'om_ml', 'tsk', '_pwp_base', 'om_depth', 'om_tini', 'om_sini',
                                               'hfx', 'lh', 'gsw', 'glw', 'qfx', 'ust', 'uair', 'vair', 'f', '_pwp_land')) +
                (ice_mask,) + tuple(fields[name] for name in ('_pwp_tend',
                                               '_pwp_w', '_pwp_status')) +
                (dt, stage_dt, stbolt, coupled_ice, n))


# Ocean initialization on Metal runs the CUDA loaders' array code on NumPy:
# their kernels are plain binary32 arithmetic, which NumPy evaluates with the
# same IEEE rounding and gradual underflow as the CUDA unit.
_RAD = np.float32(0.01745329252)


def _sinf(x):
    return np.sin(x.astype(np.float64)).astype(np.float32)


def _cosf(x):
    return np.cos(x.astype(np.float64)).astype(np.float32)


def _bin(lat, lon):
    y = np.clip(np.floor(lat + np.float32(90)).astype(np.int64), 0, 179)
    x = lon - np.floor(lon / np.float32(360)) * np.float32(360)
    return y * 360 + np.clip(np.floor(x).astype(np.int64), 0, 359)


def nearest_indices(latitude, longitude, land, source_latitude, source_longitude, valid):
    """``ocean_bin_*`` and ``ocean_nearest``: one-degree bins, exact spherical choice."""
    slat = np.ascontiguousarray(source_latitude, np.float32).ravel()
    slon = np.ascontiguousarray(source_longitude, np.float32).ravel()
    surface = np.where(np.asarray(valid).ravel(), np.float32(300), np.float32('nan'))
    with np.errstate(invalid='ignore'):
        ok = np.isfinite(slat) & np.isfinite(slon) & np.isfinite(surface) & (surface >= 260) & (surface <= 330)
    members = np.flatnonzero(ok)
    bins = _bin(slat[members], slon[members])
    order = np.argsort(bins, kind='stable'); members, bins = members[order], bins[order]
    start = np.searchsorted(bins, np.arange(180 * 360 + 1))
    tlat = np.ascontiguousarray(latitude, np.float32).ravel()
    tlon = np.ascontiguousarray(longitude, np.float32).ravel()
    land = np.asarray(land).ravel()
    index = np.full(tlat.size, -2, np.int32)
    wet = np.flatnonzero(land > 1.5)
    if wet.size == 0:
        return index.reshape(np.shape(latitude))
    tb = _bin(tlat[wet], tlon[wet])
    pole = np.minimum(np.float32(90), np.abs(tlat[wet]) + np.float32(1))
    width = np.minimum(180, np.ceil(np.float32(1) / np.maximum(np.float32(0.00001), _cosf(pole * _RAD))).astype(np.int64) + 1)
    sa, sb = slat * _RAD, slon * _RAD
    for key in np.unique(tb * 1000 + width):
        b, w = divmod(int(key), 1000)
        targets = wet[(tb == b) & (width == w)]
        cy, cx = divmod(b, 360)
        cand = [members[start[y * 360 + (cx + d + 360) % 360]:start[y * 360 + (cx + d + 360) % 360 + 1]]
                for y in range(max(0, cy - 1), min(179, cy + 1) + 1)
                for d in range(-w, min(w, 359 - w) + 1)]
        cand = np.unique(np.concatenate(cand)) if cand else np.zeros(0, np.int64)
        if cand.size == 0:
            index[targets] = -1
            continue
        lat = (tlat[targets] * _RAD)[:, None]; lon = (tlon[targets] * _RAD)[:, None]
        a, bb = sa[cand][None], sb[cand][None]
        dy = _sinf((lat - a) * np.float32(0.5)); dx = _sinf((lon - bb) * np.float32(0.5))
        dist = dy * dy + _cosf(lat) * _cosf(a) * dx * dx
        best = np.argmin(dist, axis=1)  # first minimum: the lowest source index
        score = dist[np.arange(targets.size), best]
        index[targets] = np.where(score > np.float32(0.00007615242), -1, cand[best]).astype(np.int32)
    return index.reshape(np.shape(latitude))


class NumpyKernels:
    """The loaders' profile kernels as NumPy binary32 arithmetic."""

    def __init__(self, levels):
        self.levels = int(levels)

    def get_function(self, name):
        return getattr(self, name)

    def ocean_mld_profiles(self, grid, block, args):
        mld, index, sst, t, s, u, v, z, gamma, n = args
        L, j = self.levels, index.ravel()
        depth = (np.float32(5) * np.arange(1, L + 1, dtype=np.float32))[:, None]
        sst, g = sst.ravel()[None], np.float32(gamma)
        m = np.where(j < 0, np.float32(0), mld.ravel()[np.maximum(j, 0)])[None]
        value = np.fmax(sst - g * np.fmax(depth - m, np.float32(0)), np.fmin(sst, np.float32(271.25)))
        wet = (j >= 0)[None]
        z.reshape(L, -1)[...] = depth
        t.reshape(L, -1)[...] = np.where(wet, value, np.float32(0))
        s.reshape(L, -1)[...] = np.where(wet, np.float32(35), np.float32(0))
        u[...] = 0; v[...] = 0

    def ocean_profiles(self, grid, block, args):
        src, depth, index, out, zout, ns, nt, nk, celsius = args
        L, j = self.levels, index.ravel()
        depth = np.asarray(depth, np.float32).ravel()
        for k in range(L):
            z = np.float32(5) * np.float32(k + 1)
            zout.reshape(L, -1)[k] = z
            hi = 1
            while hi < nk - 1 and depth[hi] < z:
                hi += 1
            lo = hi - 1
            jj = np.maximum(j, 0)
            a, b = src[lo].ravel()[jj], src[hi].ravel()[jj]
            weight = np.fmin(np.fmax((z - depth[lo]) / (depth[hi] - depth[lo]), np.float32(0)), np.float32(1))
            value = a + (b - a) * weight + (np.float32(273.15) if celsius else np.float32(0))
            out.reshape(L, -1)[k] = np.where(j == -2, np.float32(0), np.where(j < 0, np.float32('nan'), value))
