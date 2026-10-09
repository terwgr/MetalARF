#include <metal_stdlib>
#include "slow_buoyancy_bindings.metalh"
using namespace metal;

// Keep each operation at FP32 precision and preserve the CUDA rn_* source
// order. This is the ordinary (non-WRF-exact) slow_buoyancy caller.
#pragma clang fp contract(off)

static inline ulong sb_index(uint k, ulong cell, ulong stride) {
    return ulong(k) * stride + cell;
}

static inline float sb_base(device const float *field, uint k, ulong cell,
                            ulong stride, bool base3d) {
    return field[base3d ? sb_index(k, cell, stride) : ulong(k)];
}

static inline float sb_pp(device const float *p, device const float *pb,
                          uint k, ulong cell, ulong stride, bool base3d) {
    return p[sb_index(k, cell, stride)]
         - sb_base(pb, k, cell, stride, base3d);
}

static inline float sb_q_total(
    device const float *qv, device const float *qc, device const float *qr,
    device const float *qi, device const float *qs, device const float *qg,
    device const float *qh, ulong ix, int moist_mode,
    bool has_qi, bool has_qs, bool has_qg, bool has_qh) {
    float q = qv[ix] + qc[ix];
    q = q + qr[ix];
    if (moist_mode >= 2) {
        // Keep all three source rn_add operations in order. A missing mass
        // supplies its scratch-equivalent +0 directly, without allocation.
        q = q + (has_qi ? qi[ix] : 0.0f);
        q = q + (has_qs ? qs[ix] : 0.0f);
        q = q + (has_qg ? qg[ix] : 0.0f);
    }
    if (moist_mode == 3) q = q + (has_qh ? qh[ix] : 0.0f);
    return q;
}

kernel void slow_buoyancy(
    constant SlowBuoyancyResources &r [[buffer(0)]],
    constant SlowBuoyancyParams &q [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint nz = uint(q.nz), ny = uint(q.ny), nx = uint(q.nx);
    const ulong stride = ulong(ny) * ulong(nx);
    const ulong count = ulong(nz) * stride;
    if (ulong(flat) >= count) return;

    // CUDA enumerates k=1..nz while p/q planes are indexed 0..nz-1.
    const uint k = flat / uint(stride) + 1;
    const ulong cell = ulong(flat) - ulong(k - 1) * stride;
    const ulong ix = ulong(k) * stride + cell;
    const ulong im = ix - stride;
    const bool base3d = bool(q.base3d);
    const int moist_mode = q.moist_mode;
    const bool has_qi = bool(q.has_qi), has_qs = bool(q.has_qs);
    const bool has_qg = bool(q.has_qg), has_qh = bool(q.has_qh);

    if (k == nz) {
        float cqw = 0.0f;
        if (moist_mode) {
            const ulong top = ulong(nz - 1) * stride + cell;
            const ulong below = top - stride;
            const float qtop = sb_q_total(r.qv, r.qc, r.qr, r.qi, r.qs,
                r.qg, r.qh, top, moist_mode, has_qi, has_qs, has_qg, has_qh);
            const float qbelow = sb_q_total(r.qv, r.qc, r.qr, r.qi, r.qs,
                r.qg, r.qh, below, moist_mode, has_qi, has_qs, has_qg, has_qh);
            cqw = 0.5f * (qtop + qbelow);
        }
        const float cq1 = 1.0f / (1.0f + cqw);
        const float pp_top = sb_pp(r.p, r.pb, nz - 1, cell, stride, base3d);
        float pressure = ((cq1 * 2.0f) * r.rdnw[nz - 1]) * (0.0f - pp_top);
        float term = pressure - (r.c1f[nz] * r.mup[cell]);
        if (moist_mode) {
            const float loading = cqw * cq1;
            const float base_mass = (r.c1f[nz] * r.mub2d[cell]) + r.c2f[nz];
            term = term - (loading * base_mass);
        }
        float buoy = q.gravity * term;
        if (q.has_msf) buoy = buoy / r.msft[cell];
        r.rw_t[ix] = r.rw_t[ix] + buoy;
        return;
    }

    const float dp = sb_pp(r.p, r.pb, k, cell, stride, base3d)
                   - sb_pp(r.p, r.pb, k - 1, cell, stride, base3d);
    float term;
    if (moist_mode) {
        const float qhere = sb_q_total(r.qv, r.qc, r.qr, r.qi, r.qs,
            r.qg, r.qh, ix, moist_mode, has_qi, has_qs, has_qg, has_qh);
        const float qbelow = sb_q_total(r.qv, r.qc, r.qr, r.qi, r.qs,
            r.qg, r.qh, im, moist_mode, has_qi, has_qs, has_qg, has_qh);
        const float cqw = 0.5f * (qhere + qbelow);
        const float cq1 = 1.0f / (1.0f + cqw);
        term = (cq1 * r.rdn[k]) * dp;
        term = term - (r.c1f[k] * r.mup[cell]);
        const float loading = cqw * cq1;
        const float base_mass = (r.c1f[k] * r.mub2d[cell]) + r.c2f[k];
        term = term - (loading * base_mass);
    } else {
        term = r.rdn[k] * dp;
        term = term - (r.c1f[k] * r.mup[cell]);
    }
    float buoy = q.gravity * term;
    if (q.has_msf) buoy = buoy / r.msft[cell];
    r.rw_t[ix] = r.rw_t[ix] + buoy;
}
