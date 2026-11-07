#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_slow_buoyancy_bindings.metalh"
using namespace metal;

kernel void wrf_exact_slow_buoyancy(constant WrfExactSlowBuoyancyResources &r [[buffer(0)]],
    constant WrfExactSlowBuoyancyParams &q [[buffer(1)]], uint flat [[thread_position_in_grid]]) {
    ulong st = ulong(q.ny) * q.nx;
    if (ulong(flat) >= ulong(q.nz) * st) return;
    uint k = flat / uint(st) + 1;
    ulong c = ulong(flat) - ulong(k - 1) * st, ix = ulong(k) * st + c;
#define A(name) reinterpret_cast<device const Exact32 *>(r.name)
    device Exact32 *out = reinterpret_cast<device Exact32 *>(r.rw_t);
    ulong a = k == uint(q.nz) ? ulong(q.nz - 1) * st + c : ix;
    ulong b = a - st;
    Exact32 cqw = 0.0f;
    if (q.moist_mode) {
        cqw = (cqw + A(qv)[a]) + A(qv)[b];
        cqw = (cqw + A(qc)[a]) + A(qc)[b];
        cqw = (cqw + A(qr)[a]) + A(qr)[b];
        if (q.moist_mode >= 2) {
            cqw = (cqw + (q.has_qi ? A(qi)[a] : Exact32(0.0f))) + (q.has_qi ? A(qi)[b] : Exact32(0.0f));
            cqw = (cqw + (q.has_qs ? A(qs)[a] : Exact32(0.0f))) + (q.has_qs ? A(qs)[b] : Exact32(0.0f));
            cqw = (cqw + (q.has_qg ? A(qg)[a] : Exact32(0.0f))) + (q.has_qg ? A(qg)[b] : Exact32(0.0f));
        }
        if (q.moist_mode == 3)
            cqw = (cqw + (q.has_qh ? A(qh)[a] : Exact32(0.0f))) + (q.has_qh ? A(qh)[b] : Exact32(0.0f));
        cqw = Exact32(0.5f) * cqw;
    }
    Exact32 pp_a = A(p)[a];
    if (!q.diagnostics) pp_a = pp_a - A(pb)[q.base3d ? a : (k == uint(q.nz) ? q.nz - 1 : k)];
    Exact32 term;
    if (k == uint(q.nz)) {
        Exact32 cq1 = Exact32(1.0f) / (Exact32(1.0f) + cqw);
        term = (((cq1 * Exact32(2.0f)) * A(rdnw)[q.nz - 1]) * (Exact32(0.0f) - pp_a)) - (A(c1f)[k] * A(mup)[c]);
        if (q.moist_mode) term = term - ((cqw * cq1) * ((A(c1f)[k] * A(mub2d)[c]) + A(c2f)[k]));
    } else {
        Exact32 pp_b = A(p)[b];
        if (!q.diagnostics) pp_b = pp_b - A(pb)[q.base3d ? b : k - 1];
        Exact32 dp = pp_a - pp_b;
        if (q.moist_mode) {
            Exact32 cq1 = Exact32(1.0f) / (Exact32(1.0f) + cqw);
            term = ((cq1 * A(rdn)[k]) * dp) - (A(c1f)[k] * A(mup)[c]);
            term = term - ((cqw * cq1) * ((A(c1f)[k] * A(mub2d)[c]) + A(c2f)[k]));
        } else term = (A(rdn)[k] * dp) - (A(c1f)[k] * A(mup)[c]);
    }
    Exact32 scale = q.has_msf ? (Exact32(1.0f) / A(msft)[c]) * Exact32(q.gravity) : Exact32(q.gravity);
    out[ix] = out[ix] + scale * term;
#undef A
}
