#!/usr/bin/env python3
"""Bounded synchronous Coriolis experiment using the production Metal launcher.

Run from a source checkout after rebuilding woof/libexec/metal. --measure-only
records the broken baseline without accepting it as a passing comparison.
The 900 s experiment isolates horizontal f-plane rotation (w=0, unit maps),
using explicit midpoint integration; it is not a full atmospheric forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from woof.core import constants
from woof.core.dycore import launch_coriolis_curvature
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def fixture():
    nz, ny, nx = 3, 4, 5  # 90 dispatch threads: also exercises the tail guard.
    shapes = dict(ru=(nz, ny, nx + 1), rv=(nz, ny + 1, nx),
                  u=(nz, ny, nx + 1), v=(nz, ny + 1, nx),
                  w=(nz + 1, ny, nx), mut=(ny, nx), msft=(ny, nx),
                  msfu=(ny, nx + 1), msfv=(ny + 1, nx), f=(ny, nx),
                  e=(ny, nx), sina=(ny, nx), cosa=(ny, nx),
                  c1f=(nz + 1,), c2f=(nz + 1,), fnm=(nz,), fnp=(nz,),
                  ru_t=(nz, ny, nx + 1), rv_t=(nz, ny + 1, nx),
                  rw_t=(nz + 1, ny, nx))
    a = {name: np.zeros(shape, np.float32) for name, shape in shapes.items()}
    for name in ('msft', 'msfu', 'msfv', 'cosa', 'c1f'):
        a[name].fill(1)
    a['mut'].fill(100000)
    a['fnm'].fill(.5)
    a['fnp'].fill(.5)
    return a


def reference(a, bounded):
    """Float64 stencil evaluation for w=0/e=0, including hybrid momenta."""
    nz, ny, nx = a['v'].shape[0], a['u'].shape[1], a['v'].shape[2]
    out = {name: a[name].astype(np.float64).copy()
           for name in ('ru_t', 'rv_t', 'rw_t')}
    d = {name: value.astype(np.float64) for name, value in a.items()}
    for k in range(nz):
        u, v, ru, rv = (d[name][k] for name in ('u', 'v', 'ru', 'rv'))
        # Map curvature at mass cells: (u_bar * dm/dy - v_bar * dm/dx).
        vx = ((u[:, :-1] + u[:, 1:]) * .5 * np.diff(d['msfv'], axis=0)
              - (v[:-1] + v[1:]) * .5 * np.diff(d['msfu'], axis=1)) / 3000
        for j in range(ny):
            for i in range(nx + 1):
                if bounded and i in (0, nx):
                    continue
                cols = [(i - 1) % nx, i % nx]
                momentum = rv[j:j + 2, cols].mean()
                out['ru_t'][k, j, i] += momentum * (
                    d['f'][j, cols].mean() + vx[j, cols].mean())
        for j in range(ny + 1):
            if bounded and j in (0, ny):
                continue
            rows = [(j - 1) % ny, j % ny]
            for i in range(nx):
                momentum = ru[rows, i:i + 2].mean()
                out['rv_t'][k, j, i] -= momentum * (
                    d['f'][rows, i].mean() + vx[rows, i].mean())
        if k:
            ruf = .5 * (d['fnm'][k] * (d['ru'][k, :, :-1] + d['ru'][k, :, 1:])
                        + d['fnp'][k] * (d['ru'][k - 1, :, :-1] + d['ru'][k - 1, :, 1:]))
            uf = .5 * (d['fnm'][k] * (d['u'][k, :, :-1] + d['u'][k, :, 1:])
                       + d['fnp'][k] * (d['u'][k - 1, :, :-1] + d['u'][k - 1, :, 1:]))
            rvf = .5 * (d['fnm'][k] * (d['rv'][k, :-1] + d['rv'][k, 1:])
                        + d['fnp'][k] * (d['rv'][k - 1, :-1] + d['rv'][k - 1, 1:]))
            vf = .5 * (d['fnm'][k] * (d['v'][k, :-1] + d['v'][k, 1:])
                       + d['fnp'][k] * (d['v'][k - 1, :-1] + d['v'][k - 1, 1:]))
            out['rw_t'][k] += float(np.float32(constants.RERADIUS)) * (ruf * uf + rvf * vf)
    return out


def dispatch(module, a, bounded=False):
    device = {name: module.asarray(value) for name, value in a.items()}
    args = ('ru', 'rv', 'u', 'v', 'w', 'mut', 'msft', 'msfu', 'msfv',
            'f', 'e', 'c1f', 'c2f', 'fnm', 'fnp')
    launch_coriolis_curvature(
        *(device[name] for name in args), 3000., 3000.,
        *(device[name] for name in ('ru_t', 'rv_t', 'rw_t')),
        sina=device['sina'], cosa=device['cosa'],
        boundary_x=bounded, boundary_y=bounded)
    return {name: device[name].to_numpy() for name in ('ru_t', 'rv_t', 'rw_t')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measure-only', action='store_true')
    args = parser.parse_args()
    original = MetalContext._submit_kernel_batch

    def synchronous(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result is not None:
            result.wait()
        return result

    MetalContext._submit_kernel_batch = synchronous
    module = None
    try:
        module = MetalArrayModule()
        results = []
        for label, bounded, curvature in (
                ('uniform_north', False, False),
                ('hybrid_periodic', False, False),
                ('hybrid_bounded', True, False),
                ('curvature_only', True, True),
                ('southern', False, False)):
            a = fixture()
            a['f'].fill(0 if curvature else np.float32(1.02e-4))
            if label == 'southern':
                a['f'] *= -1
            a['u'].fill(4)
            a['v'].fill(12.8)
            if label.startswith('hybrid') or curvature:
                for name in ('u', 'v'):
                    k, j, i = np.indices(a[name].shape)
                    a[name][:] += .3 * k + .2 * j + .1 * i
                j, i = np.indices(a['mut'].shape)
                a['mut'][:] += 1000 * j + 2000 * i
                a['msfu'][:] = 1 + .01 * np.indices(a['msfu'].shape)[1]
                a['msfv'][:] = 1 + .02 * np.indices(a['msfv'].shape)[0]
                a['c1f'][:] = [1, .8, .6, .4]
                a['c2f'][:] = [0, 10000, 20000, 30000]
                mass = a['mut'].mean()
                # Distinct levels/locations keep every stencil read observable.
                for name, wind, maps in (('ru', 'u', 'msfu'), ('rv', 'v', 'msfv')):
                    k = np.arange(3, dtype=np.float32)[:, None, None]
                    coupling = mass * (1 - .2 * k) + 10000 * k
                    a[name][:] = coupling * a[wind] / a[maps]
            else:
                a['ru'][:] = 100000 * a['u']
                a['rv'][:] = 100000 * a['v']
            for name in ('ru_t', 'rv_t', 'rw_t'):
                a[name].fill(.25)
            got, want = dispatch(module, a, bounded), reference(a, bounded)
            errors = {name: float(np.max(np.abs(got[name] - want[name]))) for name in got}
            if not args.measure_only:
                for name in got:
                    np.testing.assert_allclose(got[name], want[name], rtol=2e-6, atol=5e-5)
                if bounded:
                    np.testing.assert_array_equal(got['ru_t'][:, :, [0, -1]], .25)
                    np.testing.assert_array_equal(got['rv_t'][:, [0, -1], :], .25)
            results.append(dict(case=label, max_abs_error=errors,
                                ru_t_mean=float(got['ru_t'].mean()),
                                rv_sha256=hashlib.sha256(got['rv_t'].tobytes()).hexdigest(),
                                rw_sha256=hashlib.sha256(got['rw_t'].tobytes()).hexdigest()))

        a = fixture()
        f = float(np.float32(1.02e-4))
        a['f'].fill(f)
        def acceleration(u, v):
            a['u'].fill(u); a['v'].fill(v)
            a['ru'][:] = a['u'] * 100000
            a['rv'][:] = a['v'] * 100000
            got = dispatch(module, a)
            return float(got['ru_t'].mean()) / 100000, float(got['rv_t'].mean()) / 100000
        u, v, dt = 0., float(np.float32(12.8)), 10.
        initial_acceleration = acceleration(u, v)[0]
        for _ in range(90):
            du, dv = acceleration(u, v)
            du, dv = acceleration(u + .5 * dt * du, v + .5 * dt * dv)
            u += dt * du; v += dt * dv
        expected = [float(np.float32(12.8)) * np.sin(f * 900),
                    float(np.float32(12.8)) * np.cos(f * 900)]
        if not args.measure_only:
            np.testing.assert_allclose([u, v], expected, rtol=2e-6, atol=2e-6)
        print(json.dumps(dict(
            device=module.context.device_name, synchronous=True,
            mode='measure_only' if args.measure_only else 'verified',
            grid=[3, 4, 5], stencil_cases=results,
            rotation=dict(seconds=900, dt=dt, method='explicit_midpoint',
                          initial_du_dt=initial_acceleration,
                          expected_initial_du_dt=f * float(np.float32(12.8)),
                          final_uv=[u, v], analytical_uv=expected)), indent=2))
    finally:
        if module is not None:
            module.context.close()
        MetalContext._submit_kernel_batch = original

if __name__ == '__main__':
    main()
