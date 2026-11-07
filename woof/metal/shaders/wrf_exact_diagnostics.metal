#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_calc_p_alpha_bindings.metalh"
using namespace metal;

// WRF's column operation tree, including the distinct carried-MUTS path.
kernel void wrf_exact_calc_p_alpha(
    constant WrfExactCalcPAlphaResources &r [[buffer(0)]],
    constant WrfExactCalcPAlphaParams &q [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(q.nyw * q.nxw)) return;
    const uint j = uint(q.j0) + col / uint(q.nxw);
    const uint i = uint(q.i0) + col % uint(q.nxw);
    const ulong st = ulong(q.ny) * q.nx, c = ulong(j) * q.nx + i;
    const ulong stride = q.base3d ? st : 1ul, offset = q.base3d ? c : 0ul;
#define A(name) reinterpret_cast<device const Exact32 *>(r.name)
#define O(name) reinterpret_cast<device Exact32 *>(r.name)
    const Exact32 muts = q.carried_muts ? A(muts)[c] : A(mub)[c] + A(mup)[c];
    for (int k = 0; k < q.nz; ++k) {
        const ulong h = ulong(k) * st + c, b = ulong(k) * stride + offset;
        const Exact32 dphp = A(php)[h + st] - A(php)[h];
        Exact32 ap;
        if (q.hypso == 2) {
            const Exact32 pfu = (A(c3f)[k + 1] * muts + A(c4f)[k + 1]) + Exact32(q.p_top);
            const Exact32 pfd = (A(c3f)[k] * muts + A(c4f)[k]) + Exact32(q.p_top);
            const Exact32 phm = (A(c3h)[k] * muts + A(c4h)[k]) + Exact32(q.p_top);
            const Exact32 dph = (dphp + A(phb)[b + stride]) - A(phb)[b];
            ap = (dph / phm) / exact_gfk_log(pfd / pfu) - A(alb)[b];
        } else {
            const Exact32 inverse_mass = Exact32(-1.0f) / (A(c1h)[k] * muts + A(c2h)[k]);
            const Exact32 base_mass = A(alb)[b] * (A(c1h)[k] * A(mup)[c]);
            ap = inverse_mass * (base_mass + A(rdnw)[k] * dphp);
        }
        const Exact32 a = ap + A(alb)[b];
        const Exact32 theta = Exact32(300.0f) + A(thp)[h];
        Exact32 numerator = Exact32(q.rd) * theta;
        if (q.moist) numerator = numerator * (Exact32(1.0f) + Exact32(q.rvovrd) * A(qv)[h]);
        const Exact32 ratio = numerator / (Exact32(q.p0) * a);
        const Exact32 pp = exact_gfk_pow(ratio, Exact32(q.gamma)) * Exact32(q.p0) - A(pb)[b];
        O(alt)[h] = a;
        O(al)[h] = ap;
        O(p_perturbation)[h] = pp;
        O(p)[h] = pp + A(pb)[b];
    }
#undef A
#undef O
}
