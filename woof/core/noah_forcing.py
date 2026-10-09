"""Noah's forcing prologue and rain epilogue as one launch each.

``PhysicsDriver._run_noah`` fills Noah's lowest-level forcing fields before
the column kernel (SFCPRS from the two lowest interface pressures, SFCTMP,
QV1, DZ8W1, RIB, the snow ratio SR, and the pending RAINBL) and clears the
rain carriers after it.  As CuPy array statements that was nine launches from
Python every default-suite step around one kernel launch.

These kernels make the same writes, in the statements' order, per column:
``sfcprs = 0.5 * (p_interface[0] + p_interface[1])`` rounds the sum and then
the product (``_rn`` intrinsics, so no FMA contraction appears), the copies
are copies, SR is the scheme's value or ``1.0``/``0.0`` from
``temperature <= freezing`` exactly as a boolean array assigned to float32,
and ``rainbl += pending`` is one rounded add.  A caller holding anything
other than C-contiguous float32 CuPy arrays keeps the statements.
"""

from __future__ import annotations

import numpy as np

_SOURCE = r"""
extern "C" __global__
void noah_forcing(const float* p_if0, const float* p_if1,
                  const float* temperature0, const float* qv0,
                  const float* dz0, const float* br, const float* sr_scheme,
                  const int use_scheme_sr, const float freezing,
                  const float* pending,
                  float* sfcprs, float* sfctmp, float* qv1, float* dz8w1,
                  float* rib, float* sr, float* rainbl, const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    sfcprs[c] = __fmul_rn(0.5f, __fadd_rn(p_if0[c], p_if1[c]));
    const float t = temperature0[c];
    sfctmp[c] = t;
    qv1[c] = qv0[c];
    dz8w1[c] = dz0[c];
    rib[c] = br[c];
    sr[c] = use_scheme_sr ? sr_scheme[c] : ((t <= freezing) ? 1.0f : 0.0f);
    rainbl[c] = __fadd_rn(rainbl[c], pending[c]);
}

extern "C" __global__
void noah_rain_clear(float* rainbl, float* pending, const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    rainbl[c] = 0.0f;
    pending[c] = 0.0f;
}
"""

_THREADS = 256
_MODULE = None


def _kernel(name):
    global _MODULE
    if _MODULE is None:
        import cupy as cp
        from woof.certify.kernel_manifest import record_module
        from woof.core.kernels import _compile_observed

        # A forecast translation unit, so the kernel manifest records it
        # (tests/test_kernel_manifest.py SITE_FILES).  CuPy's default
        # options, stated so the record says what was compiled.
        module = cp.RawModule(code=_SOURCE, options=())
        _compile_observed(module, "woof.core.noah_forcing")
        record_module("woof.core.noah_forcing", source=_SOURCE, options=(),
                      module=module)
        _MODULE = module
    return _MODULE.get_function(name)


def device_arrays(shape, *arrays) -> bool:
    """True when arrays can take the fused CuPy or Metal forcing path."""
    # The Metal path is intentionally an all-or-nothing dispatch.  In
    # particular, mixed contexts and mixed host/device inputs keep the
    # caller's original array statements instead of being partially packed
    # into a Metal launch.
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - Metal is optional
        MetalArray = ()
    if MetalArray and any(isinstance(a, MetalArray) for a in arrays):
        if len(shape) != 2:
            return False
        context = next((a.buffer.context for a in arrays
                        if isinstance(a, MetalArray)), None)
        return all(isinstance(a, MetalArray)
                   and a.dtype == np.dtype(np.float32)
                   and a.flags.c_contiguous and a.shape == tuple(shape)
                   and a.buffer.context is context for a in arrays)
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - the driver imports CuPy
        return False
    return all(type(a) is cp.ndarray and a.dtype == np.float32
               and a.flags.c_contiguous and a.shape == shape for a in arrays)


def forcing(fields, *, p_if0, p_if1, temperature0, qv0, dz0, sr_scheme,
            pending, freezing) -> None:
    """The prologue's writes into ``fields`` in one launch.

    ``sr_scheme`` is the scheme's SR array, or ``None`` to take SR from
    ``temperature0 <= freezing``.
    """
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - Metal is optional
        MetalArray = ()
    if MetalArray and any(isinstance(value, MetalArray) for value in
                          (p_if0, p_if1, temperature0, qv0, dz0, pending,
                           fields["br"], fields["sfcprs"], fields["sfctmp"],
                           fields["qv1"], fields["dz8w1"], fields["rib"],
                           fields["sr"], fields["rainbl"])):
        from woof.metal.noah_forcing_family import submit_noah_forcing

        return submit_noah_forcing(
            fields, p_if0=p_if0, p_if1=p_if1, temperature0=temperature0,
            qv0=qv0, dz0=dz0, sr_scheme=sr_scheme, pending=pending,
            freezing=freezing)
    ncol = int(fields["sfcprs"].size)
    use_scheme = sr_scheme is not None
    _kernel("noah_forcing")(
        ((ncol + _THREADS - 1) // _THREADS,), (_THREADS,),
        (p_if0, p_if1, temperature0, qv0, dz0, fields["br"],
         sr_scheme if use_scheme else temperature0, np.int32(use_scheme),
         np.float32(freezing), pending,
         fields["sfcprs"], fields["sfctmp"], fields["qv1"], fields["dz8w1"],
         fields["rib"], fields["sr"], fields["rainbl"], np.int64(ncol)))


def rain_clear(rainbl, pending) -> None:
    """``rainbl[...] = 0`` and ``pending[...] = 0`` in one launch."""
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - Metal is optional
        MetalArray = ()
    if MetalArray and (isinstance(rainbl, MetalArray)
                       or isinstance(pending, MetalArray)):
        from woof.metal.noah_forcing_family import submit_noah_rain_clear

        return submit_noah_rain_clear(rainbl, pending)
    ncol = int(rainbl.size)
    _kernel("noah_rain_clear")(
        ((ncol + _THREADS - 1) // _THREADS,), (_THREADS,),
        (rainbl, pending, np.int64(ncol)))


def rain_add(rainbl, pending) -> None:
    """Accumulate the pending precipitation carrier into ``rainbl``."""
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - Metal is optional
        MetalArray = ()
    if MetalArray and (isinstance(rainbl, MetalArray)
                       or isinstance(pending, MetalArray)):
        from woof.metal.noah_forcing_family import submit_noah_rain_add

        return submit_noah_rain_add(rainbl, pending)
    rainbl += pending


def sr_from_temperature(temperature, sr, *, freezing) -> None:
    """Fill Noah-MP's float snow-ratio field from its lowest temperature level."""
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # pragma: no cover - Metal is optional
        MetalArray = ()
    if isinstance(temperature, MetalArray) or isinstance(sr, MetalArray):
        from woof.metal.noah_forcing_family import submit_noah_sr_from_temperature

        return submit_noah_sr_from_temperature(
            temperature, sr, freezing=freezing)
    sr[...] = temperature[0] <= np.float32(freezing)
