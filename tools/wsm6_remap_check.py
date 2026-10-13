#!/usr/bin/env python3
"""CPU checks for the production WSM6 shader and an independent mass integral.

The C++ build strips only Metal address-space declarations and kernel wrappers;
the production numerical expressions are compiled without a second transcription.
This is not CUDA hardware validation. Native checks use the packaged Metal kernel.
"""
from __future__ import annotations

import ctypes
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHADER = ROOT / "woof/metal/shaders/wsm6_family.metal"
VOLUME = ("theta", "qv", "qc", "qi", "qr", "qs", "qg", "den",
          "pressure", "pii", "dz")
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "sr")
RADII = ("effc", "effi", "effs")
INPUTS = VOLUME + SURFACE + RADII
OUTPUTS = VOLUME[:7] + SURFACE + RADII
EPS32 = float(np.finfo(np.float32).eps)


class ShaderCPU:
    """Compile the actual Metal helper bodies with IEEE binary32 C++ arithmetic."""

    def __init__(self, build: Path, shader: Path = SHADER):
        build.mkdir(parents=True, exist_ok=True)
        text = shader.read_text().split("kernel void wsm6_column", 1)[0]
        text = re.sub(r'^#include .*$', '', text, flags=re.M)
        text = text.replace('using namespace metal;',
                            'using std::min; using std::max; using std::sqrt; '
                            'using std::exp; using std::pow; using std::floor;\n'
                            'static inline float pow(float x, int y) '
                            '{ return std::pow(x, float(y)); }')
        text = re.sub(r'\b(device|thread)\s+', '', text)
        text = text.replace('constant int', 'constexpr int')
        wrapper = r'''
extern "C" float check_remap(const float *qa, const float *zi,
    const float *za, const float *dza, int nz, float *qn) {
    return plm_regular_remap(qa, zi, za, dza, nz, qn);
}
extern "C" void check_arrival(const float *q, const float *velocity,
    const float *dz, const float *rho, int nz, float dt, float *zi,
    float *za, float *dza, float *qa) {
    plm_arrival(q, nullptr, velocity, dz, rho, nz, dt, zi, za, dza, qa, nullptr);
}
extern "C" void check_column(float *theta, float *qv, float *qc, float *qi,
    float *qr, float *qs, float *qg, const float *den, const float *p,
    const float *pii, const float *dz, float *rainnc, float *rainncv,
    float *snownc, float *snowncv, float *graupelnc, float *graupelncv,
    float *sr, float *effc, float *effi, float *effs, float dt, int hail,
    int nz, int ncol) {
    for(int col=0; col<ncol; ++col)
        wsm6_column_impl(theta,qv,qc,qi,qr,qs,qg,den,p,pii,dz,rainnc,
            rainncv,snownc,snowncv,graupelnc,graupelncv,sr,effc,effi,effs,
            dt,hail,nz,ncol,col);
}
'''
        unit = '#include <cmath>\n#include <algorithm>\n' + text + wrapper
        compiler = shutil.which('clang++') or shutil.which('c++')
        if not compiler:
            raise RuntimeError('a C++ compiler is required for WSM6 CPU checks')
        flags = ['-O2', '-std=c++17', '-ffp-contract=off', '-fno-fast-math',
                 '-shared', '-fPIC']
        self.settings = {'compiler': subprocess.check_output(
            [compiler, '--version'], text=True).splitlines()[0], 'flags': flags}
        digest = hashlib.sha256((unit + repr(self.settings)).encode()).hexdigest()
        library = build / (digest + '.so')
        if not library.exists():
            source = build / (digest + '.cpp')
            source.write_text(unit)
            subprocess.run([compiler, *flags, str(source), '-o', str(library)],
                           check=True, capture_output=True, timeout=60)
        self.lib = ctypes.CDLL(str(library))
        pointer = ctypes.POINTER(ctypes.c_float)
        self.lib.check_remap.argtypes = [pointer] * 4 + [ctypes.c_int, pointer]
        self.lib.check_remap.restype = ctypes.c_float
        self.lib.check_arrival.argtypes = ([pointer] * 4 +
                                          [ctypes.c_int, ctypes.c_float] + [pointer] * 4)
        self.lib.check_column.argtypes = [pointer] * 21 + [ctypes.c_float] + [ctypes.c_int] * 3

    @staticmethod
    def pointer(array):
        if array.dtype != np.float32 or not array.flags.c_contiguous:
            raise TypeError('CPU shader calls require contiguous binary32 arrays')
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def remap(self, qa, zi, za, dza):
        nz = len(zi) - 1
        qn = np.zeros(nz, np.float32)
        precip = self.lib.check_remap(*(self.pointer(x) for x in (qa, zi, za, dza)),
                                      nz, self.pointer(qn))
        return qn, float(precip)

    def arrival(self, q, velocity, dz, rho, dt):
        nz = len(q)
        arrays = [np.zeros(nz + 1, np.float32) for _ in range(4)]
        self.lib.check_arrival(*(self.pointer(x) for x in (q, velocity, dz, rho)),
                              nz, dt, *(self.pointer(x) for x in arrays))
        return arrays

    def column(self, inputs, dt=30., hail_opt=0):
        arrays = {k: v.copy() for k, v in inputs.items()}
        nz, ny, nx = arrays['theta'].shape
        self.lib.check_column(*(self.pointer(arrays[k]) for k in INPUTS),
                             dt, hail_opt, nz, ny * nx)
        return {k: arrays[k] for k in OUTPUTS}


def integral_reference(qa, zi, za, dza):
    """Double-precision two-point Gauss integration of the limited PLM profile.

    It integrates geometric intersections, including below-ground outflow and
    vacuum above the arrival column, without either production search algorithm.
    """
    qa, zi, za, dza = map(lambda a: np.asarray(a, np.float64), (qa, zi, za, dza))
    nz = len(zi) - 1
    left, right = qa[:-1].copy(), qa[:-1].copy()
    for k in range(1, nz):
        slope_left = (qa[k] - qa[k-1]) / (dza[k] + dza[k-1])
        slope_right = (qa[k+1] - qa[k]) / (dza[k+1] + dza[k])
        if slope_left * slope_right > 0:
            right[k] += .5 * (slope_left + slope_right) * dza[k]
            left[k] = 2 * qa[k] - right[k]
            if min(left[k], right[k]) < 0:
                left[k] = right[k] = qa[k]

    def mass(low, high):
        total = 0.
        for j in range(nz):
            a, b = max(low, za[j]), min(high, za[j+1])
            if b <= a:
                continue
            midpoint, half = (a+b)/2, (b-a)/2
            nodes = midpoint + half * np.array([-1., 1.]) / np.sqrt(3.)
            fractions = (nodes - za[j]) / dza[j]
            values = left[j] * (1-fractions) + right[j] * fractions
            total += half * float(values.sum())
        return total

    return np.asarray([mass(zi[k], zi[k+1])/(zi[k+1]-zi[k]) for k in range(nz)]), mass(-np.inf, 0.)


def rain_reproducer():
    """Neighboring two-layer primitive states plus an empty-upper-rain case."""
    names = ['below', 'equal', 'above', 'upper-rain-zero']
    shape = (2, 1, len(names))
    inputs = {k: np.zeros(shape, np.float32) for k in VOLUME + RADII}
    inputs.update({k: np.zeros((1, len(names)), np.float32) for k in SURFACE})
    inputs['theta'][:] = 300; inputs['pressure'][:] = 100000
    inputs['pii'][:] = 1; inputs['dz'][:] = 150
    inputs['den'][:] = np.float32(100000/(287*300))
    center = np.float32(1e-4)
    inputs['qr'][:, 0, :] = [np.nextafter(center, np.float32(0)), center,
                             np.nextafter(center, np.float32(np.inf)), center]
    inputs['qr'][1, 0, -1] = 0
    return names, inputs


def water_balance(inputs, outputs):
    weights = inputs['den'].astype(float) * inputs['dz'].astype(float)
    mass = lambda fields: sum(fields[k].astype(float) for k in VOLUME[1:7])
    initial = (mass(inputs) * weights).sum(axis=0)
    final = (mass(outputs) * weights).sum(axis=0)
    outflow = outputs['rainnc'].astype(float) - inputs['rainnc'].astype(float)
    return initial, final + outflow - initial


def stress_columns(nz, count=24):
    """Deterministic primitive mixed-phase inputs, including populated top cells."""
    rng = np.random.default_rng(0x57534D36 + nz)
    shape = (nz, 1, count)
    fields = {k: np.zeros(shape, np.float32) for k in VOLUME + RADII}
    fields.update({k: np.zeros((1, count), np.float32) for k in SURFACE})
    fields['theta'][:] = rng.uniform(230, 302, shape)
    fields['pressure'][:] = 100000
    fields['pii'][:] = 1
    fields['dz'][:] = rng.uniform(25, 600, shape)
    t = fields['theta'].astype(float)
    tr = 273.16/t
    xa = -(1846.4-4190)/461.6
    es = 610.78*tr**xa*np.exp((xa+2.5e6/(461.6*273.16))*(1-tr))
    fields['qv'][:] = (287/461.6)*es/(100000-es)*rng.uniform(.2, 1.25, shape)
    fields['den'][:] = 100000/(287*t)
    for key in ('qc', 'qi', 'qr', 'qs', 'qg'):
        fields[key][:] = 10**rng.uniform(-10, np.log10(.003), shape)
    # Empty upper cells distinguish closed-top controls from top-boundary mass.
    for key in ('qc', 'qi', 'qr', 'qs', 'qg'):
        fields[key][-2:, :, ::3] = 0
    return fields


def cpu_stress(shader, count=96, failure_path=None):
    """Check complete arrival arithmetic, geometric integration and total mass."""
    rng = np.random.default_rng(0x504C4D36)
    worst = {'integral_ratio': 0., 'conservation_ratio': 0.}
    total = 0
    for nz in (2, 3, 8, 16, 32, 64):
        for index in range(count):
            q = np.asarray(10**rng.uniform(-12, -2, nz), np.float32)
            rho = np.asarray(rng.uniform(.15, 1.5, nz), np.float32)
            dz = np.asarray(rng.uniform(10, 600, nz), np.float32)
            velocity = np.asarray(rng.uniform(0, 25, nz), np.float32)
            dt = float((.01, 1., 30., 120.)[index % 4])
            if index % 8 < 3:
                # Fall distances neighboring actual regular-cell interfaces.
                dz[:] = 150
                center = np.float32(150/dt)
                velocity[:] = (np.nextafter(center, np.float32(0)), center,
                               np.nextafter(center, np.float32(np.inf)))[index % 8]
            zi, za, dza, qa = shader.arrival(q*rho, velocity, dz, rho, dt)
            if not np.all(dza[:-1] > 0):
                raise AssertionError('arrival interfaces crossed')
            got, rain = shader.remap(qa, zi, za, dza)
            ref, ref_rain = integral_reference(qa, zi, za, dza)
            # Integral and mass budgets scale with column mass rather than
            # individual nearly empty cells. Both are fixed before comparison.
            mass = float(np.dot((q*rho).astype(float), dz.astype(float)))
            budget = 64*EPS32*max(mass, 1e-12)
            integral_error = float(np.sum(np.abs(got-ref)*np.diff(zi.astype(float)))) + abs(rain-ref_rain)
            residual = float(np.dot(got.astype(float), np.diff(zi.astype(float)))) + rain-mass
            worst['integral_ratio'] = max(worst['integral_ratio'], integral_error/budget)
            worst['conservation_ratio'] = max(worst['conservation_ratio'], abs(residual)/budget)
            if integral_error > budget or abs(residual) > budget or (got < 0).any() or rain < 0:
                if failure_path is not None:
                    np.savez(failure_path, q=q, rho=rho, dz=dz, velocity=velocity,
                             dt=np.float32(dt), zi=zi, za=za, dza=dza, qa=qa,
                             got=got, rain=rain, reference=ref, reference_rain=ref_rain)
                raise AssertionError({'nz': nz, 'index': index, 'dt': dt,
                                      'integral_error': integral_error,
                                      'residual': residual, 'budget': budget})
            total += 1
    return {'columns': total, 'seed': '0x504C4D36', **worst}


def check_columns(inputs, outputs, reference=None):
    initial, residual = water_balance(inputs, outputs)
    budget = 32*EPS32*initial + 8*EPS32*.1
    assert np.all(np.abs(residual) <= budget), 'column water balance'
    for key in OUTPUTS:
        assert np.isfinite(outputs[key]).all(), key
    for key in ('qv', 'qc', 'qi', 'qr', 'qs', 'qg'):
        assert (outputs[key] >= 0).all(), key
    differences = {}
    if reference is not None:
        for key in set(OUTPUTS)-set(RADII):
            scale = 300. if key == 'theta' else max(float(np.abs(reference[key]).max()), 1e-4)
            delta = float(np.abs(outputs[key].astype(float)-reference[key]).max())
            assert delta <= 128*EPS32*scale + 1e-8, (key, delta, scale)
            differences[key] = delta
    return {'columns': inputs['theta'].shape[2],
            'max_water_residual': float(np.abs(residual).max()),
            'max_budget_ratio': float((np.abs(residual)/budget).max()),
            'max_native_cpu_difference': differences}


@contextmanager
def coordinated_gpu():
    """Refuse active models/owners and remove only this process's PID lock."""
    lock = Path(os.environ.get('WOOF_METAL_MACHINE_LOCK', '/tmp/woof-gpu.lock'))
    queue = Path(os.environ.get('WOOF_METAL_MACHINE_QUEUE', '/tmp/woof-gpu-queue'))
    processes = subprocess.check_output(['ps', '-axo', 'pid,command'], text=True)
    active = re.compile(r'(?:-m woof(?:\.cli|\.metal\.real_prepared)?|/woof|/metalarf)\s+(?:go|verify|hex|global|sim|--source-cache)\b')
    if active.search(processes) or lock.exists() or (queue.exists() and any(queue.iterdir())):
        raise RuntimeError('GPU occupied; native check was not started')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        if lock.exists() and lock.read_text().strip() == str(os.getpid()):
            lock.unlink()


def native_checks(groups):
    for key in ('WOOF_METAL_RUNTIME_LIB', 'WOOF_METAL_SHADER_LIBRARY',
                'WOOF_METAL_EXACT_SHADER_LIBRARY'):
        if os.environ.get(key):
            raise RuntimeError('unset external bundle override: ' + key)
    os.environ['WOOF_METAL_GPU_SLOTS'] = '1'
    from woof.metal.runtime import MetalContext
    from woof.metal.array import MetalArrayModule
    from woof.metal.wsm6_family import submit_wsm6_column
    from woof.metal.suites import execute
    start = time.monotonic()
    with coordinated_gpu(), MetalContext(batch_kernel_calls=False) as context:
        module = MetalArrayModule(context)
        result = {'device': context.device_name, 'groups': []}
        for label, inputs, hail, reference in groups:
            arrays = {k: module.asarray(v) for k, v in inputs.items()}
            for key in INPUTS:
                assert np.array_equal(inputs[key].view(np.uint32),
                                      arrays[key].to_numpy().view(np.uint32)), key
            submit_wsm6_column(**arrays, dt=30., hail_opt=hail)
            outputs = {k: arrays[k].to_numpy() for k in OUTPUTS}
            result['groups'].append({'label': label, 'hail_opt': hail,
                                    **check_columns(inputs, outputs, reference)})
        fields = {}
        status, detail = execute((6,0,0,0,0,'periodic'), sfclay_option=0,
                                 module=module, steps=2, fields=fields)
        assert status == 'ok', (status, detail)
        assert 250 < fields['theta_min_k'] <= fields['theta_max_k'] < 400
        assert fields['max_wind_m_s'] < 10
        result['driver_smoke'] = {'status': status, 'fields': fields}
    result['elapsed_seconds'] = time.monotonic()-start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', action='store_true', help='request one coordinated synchronous native pass')
    parser.add_argument('--native-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    args.output.mkdir(parents=True, exist_ok=True)
    shader = ShaderCPU(args.output/'cpu-build')
    report = {'source_sha256': hashlib.sha256(SHADER.read_bytes()).hexdigest(),
              'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'compiler': shader.settings,
              'arrival_remap': cpu_stress(shader, failure_path=args.output/'first-remap-failure.npz')}
    groups = []
    for label, inputs in [('reduced', rain_reproducer()[1])] + [(f'nz-{nz}', stress_columns(nz)) for nz in (2,8,64)]:
        for hail in (0,1):
            reference = shader.column(inputs, hail_opt=hail)
            check_columns(inputs, reference)
            groups.append((label, inputs, hail, reference))
    report['full_column_count'] = sum(g[1]['theta'].shape[2] for g in groups)
    # Safe test-only mutation: double retained mass in a copied source file.
    mutation = args.output/'mutated.metal'
    text = SHADER.read_text()
    anchor = 'qn[k] = mass / (zi[k + 1] - zi[k]);'
    assert text.count(anchor) == 1
    mutation.write_text(text.replace(anchor, 'qn[k] = 2.0f * mass / (zi[k + 1] - zi[k]);'))
    mutant = ShaderCPU(args.output/'mutation-build', mutation)
    inputs = rain_reproducer()[1]
    try:
        check_columns(inputs, mutant.column(inputs))
    except AssertionError:
        report['mutation_detected'] = True
    else:
        raise AssertionError('mass-doubling mutation escaped')
    report['native'] = 'not requested'
    if args.native_worker:
        try:
            report['native'] = native_checks(groups)
        except RuntimeError as error:
            report['native'] = {'status': 'not run', 'reason': str(error)}
    elif args.native:
        # Keep a hard parent deadline even if native code stops responding.
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                    '--output', str(args.output), '--native-worker'],
                                   stdout=subprocess.DEVNULL)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            lock = Path(os.environ.get('WOOF_METAL_MACHINE_LOCK', '/tmp/woof-gpu.lock'))
            if lock.exists() and lock.read_text().strip() == str(process.pid):
                lock.unlink()
            report['native'] = {'status': 'failed', 'reason': '60-second native deadline exceeded'}
        else:
            if process.returncode:
                raise RuntimeError('native worker failed; see its traceback')
            report['native'] = json.loads((args.output/'results.json').read_text())['native']
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
