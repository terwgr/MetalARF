"""The fused CUDA kernel bodies, compiled for the CPU, match their numpy specs.

Every ElementwiseKernel in ``woof.globe.spectral.fused`` is documented as
the same arithmetic in the same association as a numpy expression in
``MoistHybridModel``.  A GPU is not available to a CPU test run, so the
kernel C is compiled here with clang (``-ffp-contract=off``, so no fused
multiply-add moves last bits) into a shared library that loops the kernel
body over every output index, and the result is compared BIT-FOR-BIT with
the numpy specification in float64 and float32.  This proves the C text
computes the specification; the device launch (indexing through cupy's
CArray, contraction policy of nvcc) still needs the GPU self-test.

The specifications are imported from ``tools.gpu_selftest_rhs``, which is
the GPU self-test itself, so the functions this file proves the kernel C
bit-identical to are the very ones the device gate compares the launched
kernels against.  A transcription kept on each side could drift apart and
both halves would still pass; one definition cannot.

Skips, naming the reason, when no clang is on this machine.
"""
from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from woof.globe.spectral import fused  # noqa: E402
from tools import gpu_selftest_rhs as rhs_tool  # noqa: E402
from tools.gpu_selftest_rhs import (  # noqa: E402
    momentum_bernoulli_reference,
    scalar_tendency_reference,
    specification_model,
    tracer_flux_reference,
)


def _clang() -> str | None:
    found = shutil.which("clang")
    if found:
        return found
    for candidate in (
        Path(r"C:\Program Files\LLVM\bin\clang.exe"),
        Path("/usr/bin/clang"),
        Path("/usr/local/bin/clang"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


CLANG = _clang()
# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        CLANG is None,
        reason="no clang on this machine: the kernel C cannot be compiled for the CPU"),
]


class _Recorder:
    """Stand-in for cupy that captures ElementwiseKernel arguments."""

    def ElementwiseKernel(self, in_params, out_params, operation, name):
        return {"in": in_params, "out": out_params, "op": operation, "name": name}


def _capture(factory):
    spec = factory(_Recorder())
    fused._CACHE.pop(spec["name"], None)
    return spec


def _params(text):
    out = []
    for item in text.split(","):
        item = item.strip()
        raw = item.startswith("raw ")
        if raw:
            item = item[4:]
        kind, name = item.split()
        out.append((kind, name, raw))
    return out


def _build(tmp_path, spec, ctype, elementwise_inputs):
    """Compile the kernel body into ``run(<inputs...>, <outputs...>, n)``.

    ``raw T`` inputs are passed as flat pointers indexed by the body;
    plain ``T`` inputs are pre-broadcast flat arrays read at ``i``; integer
    params are scalars.
    """
    ins = _params(spec["in"])
    outs = _params(spec["out"])
    args = []
    loads = []
    stores = []
    for kind, name, raw in ins:
        if kind == "T":
            if raw:
                args.append(f"const T* {name}")
            else:
                args.append(f"const T* {name}_arr")
                loads.append(f"const T {name} = {name}_arr[i];")
        elif kind == "int64":
            args.append(f"long long {name}")
        else:
            raise AssertionError(f"unhandled kernel param type {kind}")
    for kind, name, _raw in outs:
        assert kind == "T"
        args.append(f"T* {name}_arr")
        loads.append(f"T {name};")
        stores.append(f"{name}_arr[i] = {name};")
    source = "\n".join([
        f"typedef {ctype} T;",
        "#ifdef _WIN32",
        "#define EXPORT __declspec(dllexport)",
        "#else",
        "#define EXPORT",
        "#endif",
        f"EXPORT void run({', '.join(args)}, long long n) {{",
        "  for (long long i = 0; i < n; ++i) {",
        *("    " + line for line in loads),
        "    {",
        spec["op"],
        "    }",
        *("    " + line for line in stores),
        "  }",
        "}",
    ])
    stem = f"{spec['name']}_{ctype}"
    c_path = tmp_path / f"{stem}.c"
    lib_path = tmp_path / (f"{stem}.dll" if sys.platform == "win32" else f"{stem}.so")
    c_path.write_text(source, encoding="utf-8")
    subprocess.run(
        [CLANG, "-O2", "-ffp-contract=off", "-shared", "-o", str(lib_path), str(c_path)],
        check=True, capture_output=True,
    )
    return ctypes.CDLL(str(lib_path)), ins, outs


def _run(lib, ins, outs, values, dtype, n, out_shape):
    argv = []
    keep = []
    for kind, name, _raw in ins:
        value = values[name]
        if kind == "T":
            array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
            keep.append(array)
            argv.append(array.ctypes.data_as(ctypes.c_void_p))
        else:
            argv.append(ctypes.c_longlong(int(value)))
    results = []
    for _kind, name, _raw in outs:
        array = np.empty(out_shape, dtype=dtype)
        keep.append(array)
        results.append(array)
        argv.append(array.ctypes.data_as(ctypes.c_void_p))
    argv.append(ctypes.c_longlong(int(n)))
    lib.run(*argv)
    return results


def _model(dtype):
    """The GPU self-test's own specification model, at this test's dtype."""
    precision = "float64" if dtype == np.float64 else "float32"
    model = specification_model(7, precision=precision, truncation=5)
    return model.transform, model.vertical, model


@pytest.mark.parametrize("dtype,ctype", [(np.float64, "double"), (np.float32, "float")])
def test_vertical_flux_divergence_kernel_matches_numpy_bitwise(tmp_path, dtype, ctype):
    transform, vertical, model = _model(dtype)
    rng = np.random.default_rng(11)
    nlev = vertical.nlev
    shape = transform.grid.shape
    ps = (1.0e5 * (1.0 + 0.05 * rng.standard_normal(shape))).astype(dtype)
    pressure = vertical.pressure(ps, transform.backend)
    p_half = pressure["p_half"].astype(dtype)
    p_full = pressure["p_full"].astype(dtype)
    # Three stacked tracers: a smooth one, a sign-changing noisy one (both
    # limiter branches and both upstream directions), and a top hat.
    smooth = 300.0 + 0.01 * p_full + rng.standard_normal((nlev, *shape))
    noisy = rng.standard_normal((nlev, *shape))
    hat = np.where((p_full > 3.0e4) & (p_full < 7.0e4), 1.0, 0.0)
    scalar = np.stack([smooth, noisy, hat]).astype(dtype)
    omega = (0.5 * rng.standard_normal((nlev + 1, *shape))).astype(dtype)
    omega[0] = 0.0
    omega[-1] = 0.0
    omega[3, :, :4] = 0.0  # the w >= 0 branch at exact zero

    expected = model._vertical_scalar_flux_divergence(scalar, omega, p_full, p_half)
    assert expected.dtype == dtype

    spec = _capture(fused.vertical_flux_divergence_kernel)
    assert spec["op"] == fused.VERTICAL_FLUX_DIVERGENCE_OPERATION
    lib, ins, outs = _build(tmp_path, spec, ctype, ())
    horiz = shape[0] * shape[1]
    (out,) = _run(
        lib, ins, outs,
        {"s": scalar, "w": omega, "pf": p_full, "ph": p_half, "nlev": nlev, "horiz": horiz},
        dtype, scalar.size, scalar.shape,
    )
    assert np.array_equal(out, expected), (
        f"max |C - numpy| = {np.max(np.abs(out - expected))}"
    )
    # The limiter and both upstream branches were exercised.
    assert np.any(omega[1:nlev] > 0) and np.any(omega[1:nlev] < 0)


@pytest.mark.parametrize("dtype,ctype", [(np.float64, "double"), (np.float32, "float")])
def test_momentum_bernoulli_kernel_matches_numpy_bitwise(tmp_path, dtype, ctype):
    transform, vertical, model = _model(dtype)
    rng = np.random.default_rng(5)
    nlev = vertical.nlev
    shape = transform.grid.shape
    volume = (nlev, *shape)
    fields = {
        "zeta": 1e-5 * rng.standard_normal(volume),
        "cor": np.broadcast_to(np.asarray(model.coriolis, dtype=dtype), volume),
        "u": 20.0 * rng.standard_normal(volume),
        "v": 20.0 * rng.standard_normal(volume),
        "wu": 1e-4 * rng.standard_normal(volume),
        "wv": 1e-4 * rng.standard_normal(volume),
        "tv": 250.0 + 30.0 * rng.standard_normal(volume),
        "pg": np.abs(rng.standard_normal(volume)),
        "ge": np.broadcast_to(1e-6 * rng.standard_normal(shape), volume),
        "gn": np.broadcast_to(1e-6 * rng.standard_normal(shape), volume),
        "phi": 5.0e4 * np.abs(rng.standard_normal(volume)),
        "rgas": np.full(volume, model.gas_constant),
    }
    fields = {k: np.ascontiguousarray(np.asarray(v, dtype=dtype)) for k, v in fields.items()}
    # Numpy specification (MoistHybridModel.rhs, numpy branch), the same
    # function the GPU self-test compares the launched kernel against.
    mu, mv, bern = momentum_bernoulli_reference(
        fields["zeta"], fields["cor"], fields["u"], fields["v"],
        fields["wu"], fields["wv"], fields["tv"], fields["pg"],
        fields["ge"], fields["gn"], fields["phi"], fields["rgas"],
    )

    spec = _capture(fused.momentum_bernoulli_kernel)
    lib, ins, outs = _build(tmp_path, spec, ctype, ())
    c_mu, c_mv, c_bern = _run(lib, ins, outs, fields, dtype, mu.size, volume)
    assert np.array_equal(c_mu, mu.astype(dtype))
    assert np.array_equal(c_mv, mv.astype(dtype))
    assert np.array_equal(c_bern, bern.astype(dtype))


@pytest.mark.parametrize("dtype,ctype", [(np.float64, "double"), (np.float32, "float")])
def test_scalar_tendency_and_tracer_flux_kernels_match_numpy_bitwise(tmp_path, dtype, ctype):
    rng = np.random.default_rng(3)
    volume = (4, 6, 8)
    dp = np.abs(rng.standard_normal(volume)).astype(dtype) + dtype(1.0)
    s = rng.standard_normal(volume).astype(dtype)
    u = rng.standard_normal(volume).astype(dtype)
    v = rng.standard_normal(volume).astype(dtype)
    h = rng.standard_normal(volume).astype(dtype)
    vert = rng.standard_normal(volume).astype(dtype)
    dpt = rng.standard_normal(volume).astype(dtype)

    spec = _capture(fused.tracer_flux_kernel)
    lib, ins, outs = _build(tmp_path, spec, ctype, ())
    fx, fy = _run(lib, ins, outs, {"dp": dp, "s": s, "u": u, "v": v}, dtype, s.size, volume)
    want_fx, want_fy = tracer_flux_reference(dp, s, u, v)
    assert np.array_equal(fx, want_fx)
    assert np.array_equal(fy, want_fy)

    spec = _capture(fused.scalar_tendency_kernel)
    lib, ins, outs = _build(tmp_path, spec, ctype, ())
    (out,) = _run(
        lib, ins, outs, {"h": h, "vert": vert, "s": s, "dpt": dpt, "dp": dp},
        dtype, s.size, volume,
    )
    assert np.array_equal(out, scalar_tendency_reference(h, vert, s, dpt, dp))


@pytest.mark.parametrize("dtype,ctype", [(np.float64, "double"), (np.float32, "float")])
def test_the_gpu_selftest_suites_pass_against_the_kernel_c(tmp_path, dtype, ctype):
    """The device gate's own adversarial data, judged before a card is booked.

    ``tools/gpu_selftest_rhs.py`` chooses the inputs the launched kernels
    will see: zeros, saturated magnitudes, single-point extremes, exact-zero
    interface velocities, both van Leer limiter branches, ``nlev = 2``.  If
    a case disagreed with its reference for a reason that is not the device
    - a broadcast the harness reshapes, a pressure ladder the limiter
    divides by, a dtype the tool casts late - the operator would learn it
    only after reserving a GPU.  Running the same suites through the
    CPU-compiled kernel C settles that here.
    """
    volume_spec = _capture(fused.momentum_bernoulli_kernel)
    momentum_lib, m_ins, m_outs = _build(tmp_path, volume_spec, ctype, ())
    order = ("zeta", "cor", "u", "v", "wu", "wv", "tv", "pg", "ge", "gn", "phi")
    for label, case in rhs_tool.momentum_cases(dtype):
        volume = np.shape(case["u"])
        values = {
            name: np.ascontiguousarray(
                np.broadcast_to(np.asarray(case[name], dtype=dtype), volume)
            )
            for name in order
        }
        values["rgas"] = np.full(volume, 287.0528, dtype=dtype)
        # The extreme-wind case squares 1e30 on purpose; the C overflows to
        # the same infinity and equal_nan compares the pattern.
        with np.errstate(over="ignore"):
            want = momentum_bernoulli_reference(
                *(values[name] for name in order), values["rgas"]
            )
        got = _run(
            momentum_lib, m_ins, m_outs, values, dtype,
            int(np.prod(volume)), volume,
        )
        for field, c_out, spec_out in zip(("mu", "mv", "bern"), got, want):
            assert np.array_equal(c_out, spec_out, equal_nan=True), (
                f"momentum/{label}/{field}: max |C - numpy| = "
                f"{np.nanmax(np.abs(c_out - spec_out))}"
            )

    flux_spec = _capture(fused.vertical_flux_divergence_kernel)
    flux_lib, f_ins, f_outs = _build(tmp_path, flux_spec, ctype, ())
    precision = "float64" if dtype == np.float64 else "float32"
    models: dict[int, object] = {}
    for label, nlev, scalar, omega, p_full, p_half in rhs_tool.vertical_flux_cases(dtype):
        model = models.setdefault(
            nlev, rhs_tool.specification_model(nlev, precision=precision)
        )
        want = model._vertical_scalar_flux_divergence(scalar, omega, p_full, p_half)
        (out,) = _run(
            flux_lib, f_ins, f_outs,
            {
                "s": scalar, "w": omega, "pf": p_full, "ph": p_half,
                "nlev": nlev, "horiz": scalar.shape[-2] * scalar.shape[-1],
            },
            dtype, scalar.size, scalar.shape,
        )
        assert np.array_equal(out, want, equal_nan=True), (
            f"vertical_flux/{label}: max |C - numpy| = "
            f"{np.nanmax(np.abs(out - want))}"
        )


def test_every_fused_kernel_has_a_cpu_harness_case():
    covered = {
        "arwen_rhs_vertical_flux_divergence_vanleer",
        "arwen_rhs_momentum_bernoulli",
        "arwen_rhs_tracer_flux",
        "arwen_rhs_scalar_tendency",
    }
    names = set(re.findall(r'"(arwen_[a-z_]+)"', Path(fused.__file__).read_text(encoding="utf-8")))
    # project_kernel is complex-typed and lives with the transform tests.
    names.discard("arwen_spectral_project")
    assert names == covered, f"kernel names {sorted(names)} vs harness {sorted(covered)}"
