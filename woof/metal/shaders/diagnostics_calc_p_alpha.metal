#include <metal_stdlib>
#include "diagnostics_calc_p_alpha_bindings.metalh"
using namespace metal;
#include "diagnostics_log1pf.metalh"
#include "pressure_power.metalh"

static inline ulong d_idx(uint k, uint j, uint i, uint ny, uint nx) {
    return (ulong(k) * ny + j) * nx + i;
}

// Source-shaped port of woof/core/kernels/diagnostics.cu::calc_p_alpha's
// ordinary (non GPUWM_WRF_EXACT_D_DIAGNOSTICS) branch. The column loop stays
// serial to preserve its per-level operation order. The window decode and
// profile broadcasting match the CUDA caller.
kernel void diagnostics_calc_p_alpha(
    constant DiagnosticsCalcPAlphaResources &r [[buffer(0)]],
    constant DiagnosticsCalcPAlphaParams &q [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (q.hypso != 1 && q.hypso != 2) return;
    if (col >= uint(q.nyw * q.nxw)) return;
    const uint j = uint(q.j0) + col / uint(q.nxw);
    const uint i = uint(q.i0) + col - (col / uint(q.nxw)) * uint(q.nxw);
    const uint ny = uint(q.ny), nx = uint(q.nx), nz = uint(q.nz);
    const ulong st = ulong(ny) * nx;
    const ulong column = ulong(j) * nx + i;
    const ulong kstr = q.base3d ? st : 1ul;
    const ulong coff = q.base3d ? column : 0ul;
    const float mu = r.mub[column] + r.mup[column];

    for (uint k = 0; k < nz; ++k) {
        const ulong h = d_idx(k, j, i, ny, nx);
        const ulong b = ulong(k) * kstr + coff;
        float th = r.thb[b] + r.thp[h];
        if (q.moist) th *= 1.0f + 1.6083624362945557f * r.qv[h];

        const float dphb = (r.phb[ulong(k + 1) * kstr + coff] - r.phb[b])
                         + r.dphbr[b];
        const float dph = dphb
                        + (r.php[d_idx(k + 1, j, i, ny, nx)] - r.php[h]);
        float a, ap;
        if (q.hypso == 2) {
            const float pfu = r.c3f[k + 1] * mu + r.c4f[k + 1] + q.p_top;
            const float dpf = r.dc3f[k] * mu + r.dc4f[k];
            const float phm = r.c3h[k] * mu + r.c4h[k] + q.p_top;
            ap = dph / phm / plm_log1pf(dpf / pfu) - r.alb[b];
            a = ap + r.alb[b];
        } else {
            a = -dph * r.rdnw[k] / (r.c1h[k] * mu + r.c2h[k]);
            ap = a - r.alb[b];
        }
        r.alt[h] = a;
        r.al[h] = ap;
        // A low bias in native pow feeds back into pressure gradients at
        // every RK stage, including small-amplitude gravity-wave states.
        // Keep normalization and storage in FP32 and use the same bounded
        // software-binary64 power as the surface pressure/Exner consumers.
        r.p[h] = 1.0e5f * fp32_pressure_power(287.0f * th,
                                             1.0e5f * a, 1.4f);
    }
}
