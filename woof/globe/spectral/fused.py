"""CuPy ElementwiseKernel fast paths for the spectral/dynamics hot chains.

Every kernel here is a backend-conditional replacement for a numpy
expression that remains in place at its call site as the specification:
the fused path computes the same arithmetic in the same association (only
compiler contraction of multiply-add may move last bits).  The kernels are
built lazily so this module imports cleanly on machines without CuPy, and
each one is compared against its numpy specification on random and
adversarial inputs by the standalone GPU self-test that ships with the
fusion work (per-op max abs/rel diffs, nonzero exit beyond fp tolerance).

Call-site census that motivated each kernel (T533, one step, measured):
``project`` ran 470 times at 4 backend calls each; the RHS momentum/
bernoulli/scalar-tendency assemblies and the vertical upstream flux chain
were the remaining dynamics-side elementwise launches after the Legendre
m-loops were batched into strided GEMMs.
"""
from __future__ import annotations

from .backend import device_cache_key

#: Compiled kernels, one entry per (name, module, DEVICE).  The device id
#: is in the key because a module-level cache that hands a kernel built on
#: one card to a launch on another is the multi-card defect
#: :func:`~woof.globe.spectral.backend.device_cache_key` documents; gate
#: CARD-1 builds every one of these on two device ids in one process and
#: asserts the entries are distinct.
_CACHE: dict[tuple, object] = {}


def _kernel(xp, name: str, in_params: str, out_params: str, operation: str):
    key = (name, *device_cache_key(xp))
    ker = _CACHE.get(key)
    if ker is None:
        ker = xp.ElementwiseKernel(in_params, out_params, operation, name)
        _CACHE[key] = ker
    return ker


def project_kernel(xp):
    """Fused triangular projection: mask + m=0 realify + fresh output.

    ``code`` is 0 above the triangle (coefficient zeroed), 2 on the m=0
    column inside the triangle (imaginary part dropped), 1 elsewhere
    inside the triangle (copied).  Specification: the numpy branch of
    ``SphericalHarmonicTransform.project``.  Matches it exactly for every
    finite input; a NON-FINITE above-triangle entry becomes an exact zero
    here where the specification's mask multiply would yield NaN.
    """
    return _kernel(
        xp,
        "arwen_spectral_project",
        "T x, int8 code",
        "T y",
        "y = (code == 0) ? T(0) : ((code == 2) ? T(x.real()) : x);",
    )


def momentum_bernoulli_kernel(xp):
    """RHS momentum-forcing pair and Bernoulli function in one launch.

    Specification (numpy branch of ``MoistHybridModel.rhs``)::

        av   = vorticity + coriolis
        pf   = gas_constant * virtual_temperature * pressure_gradient
        mu   = av * v - wu - pf * grad_lnps_east
        mv   = -av * u - wv - pf * grad_lnps_north
        bern = 0.5 * (u**2 + v**2) + geopotential

    ``pg`` is the full-level factor ``grad(ln p_k) / grad(ln ps)`` from
    ``MoistHybridModel._pressure_gradient_factor`` (3-D); ``ge``/``gn`` are
    the 2-D gradient of ``ln ps`` and broadcast over the level axis (DN-3).
    """
    return _kernel(
        xp,
        "arwen_rhs_momentum_bernoulli",
        "T zeta, T cor, T u, T v, T wu, T wv, T tv, T pg, T ge, T gn, "
        "T phi, T rgas",
        "T mu, T mv, T bern",
        """
        const T av = zeta + cor;
        const T pf = rgas * tv * pg;
        mu = av * v - wu - pf * ge;
        mv = -av * u - wv - pf * gn;
        bern = (T)0.5 * (u * u + v * v) + phi;
        """,
    )


def tracer_flux_kernel(xp):
    """Horizontal mass-flux pair ``(dp*s*u, dp*s*v)`` in one launch.

    Specification: the numpy branch of ``MoistHybridModel._scalar_tendency``
    computes ``dp * scalar * u`` and ``dp * scalar * v``; the shared
    ``dp * scalar`` product is associated first in both paths.
    """
    return _kernel(
        xp,
        "arwen_rhs_tracer_flux",
        "T dp, T s, T u, T v",
        "T fx, T fy",
        """
        const T m = dp * s;
        fx = m * u;
        fy = m * v;
        """,
    )


def scalar_tendency_kernel(xp):
    """Flux-form scalar tendency combine ``(-h - vert - s*dp_t) / dp``.

    Specification: the numpy branch of ``MoistHybridModel._scalar_tendency``.
    """
    return _kernel(
        xp,
        "arwen_rhs_scalar_tendency",
        "T h, T vert, T s, T dpt, T dp",
        "T out",
        "out = (-h - vert - s * dpt) / dp;",
    )


def _limited_interface_flux(name: str, j: str) -> str:
    """C for the van Leer limited flux ``{name}`` through interface ``j``.

    Mirrors the numpy specification statement by statement, in the same
    association: upstream layer ``u`` (above when ``w >= 0``), its van
    Leer gradient ``2 * ga * gb / (ga + gb)`` when both one-sided
    gradients share a sign (zero otherwise, and zero for the boundary
    layers), the face value ``s[u] + grad * (p_half[j] - p_full[u])``
    bounded by the two adjacent layer values, times ``w``.  Interfaces
    0 and nlev carry zero flux.  ``base`` is the first index of the
    current tracer block, so no access crosses a tracer boundary.
    """
    return f"""
        T {name} = (T)0;
        {{
            const long long j = {j};
            if (j > 0 && j < nlev) {{
                const T wj = w[j * horiz + col];
                const long long u = (wj >= (T)0) ? (j - 1) : j;
                const T su = s[base + u * horiz + col];
                T grad = (T)0;
                if (u > 0 && u + 1 < nlev) {{
                    const T sa = s[base + (u - 1) * horiz + col];
                    const T sb = s[base + (u + 1) * horiz + col];
                    const T pa = pf[(u - 1) * horiz + col];
                    const T pu = pf[u * horiz + col];
                    const T pb = pf[(u + 1) * horiz + col];
                    const T ga = (su - sa) / (pu - pa);
                    const T gb = (sb - su) / (pb - pu);
                    const T prod = ga * gb;
                    if (prod > (T)0) {{
                        grad = (T)2 * prod / (ga + gb);
                    }}
                }}
                T face = su + grad * (ph[j * horiz + col] - pf[u * horiz + col]);
                const T above = s[base + (j - 1) * horiz + col];
                const T below = s[base + j * horiz + col];
                const T lo = (above < below) ? above : below;
                const T hi = (above > below) ? above : below;
                face = (face > lo) ? face : lo;
                face = (face < hi) ? face : hi;
                {name} = wj * face;
            }}
        }}
    """


VERTICAL_FLUX_DIVERGENCE_OPERATION = (
    """
        const long long ii = (long long)i;
        const long long col = ii % horiz;
        const long long k = (ii / horiz) % nlev;
        const long long base = ii - (k * horiz + col);
    """
    + _limited_interface_flux("up", "k + 1")
    + _limited_interface_flux("dn", "k")
    + """
        out = up - dn;
    """
)


def vertical_flux_divergence_kernel(xp):
    """Van Leer limited vertical flux divergence, one launch, all interfaces.

    Specification: the numpy branch of
    ``MoistHybridModel._vertical_scalar_flux_divergence`` (monotone
    second-order MUSCL flux from the upstream layer; DN-4).  For output
    level ``k`` the result is ``flux[k+1] - flux[k]`` with interface
    fluxes built by ``_limited_interface_flux``.  ``s`` carries any
    leading (tracer) axes ahead of ``(nlev, nlat, nlon)``; ``w`` is the
    shared ``(nlev+1, nlat, nlon)`` interface pressure velocity, ``pf``
    the ``(nlev, nlat, nlon)`` full-level and ``ph`` the ``(nlev+1, nlat,
    nlon)`` half-level pressure; ``horiz = nlat * nlon``.  The level
    guards keep every neighbour access inside the same tracer block.
    """
    return _kernel(
        xp,
        "arwen_rhs_vertical_flux_divergence_vanleer",
        "raw T s, raw T w, raw T pf, raw T ph, int64 nlev, int64 horiz",
        "T out",
        VERTICAL_FLUX_DIVERGENCE_OPERATION,
    )


__all__ = [
    "VERTICAL_FLUX_DIVERGENCE_OPERATION",
    "momentum_bernoulli_kernel",
    "project_kernel",
    "scalar_tendency_kernel",
    "tracer_flux_kernel",
    "vertical_flux_divergence_kernel",
]
