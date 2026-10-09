// FP64 norm assembly for the chunked Global semi-Lagrangian Lipschitz gate.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_lipschitz_norms_fragment_bindings.metalh"

using namespace metal;

struct GLJacobi3 {
    ulong d0, d1, d2, o01, o02, o12;
};

static inline ulong glj_abs(ulong x) {
    return soft64_less(x, 0ul) ? soft64_negate_bits(x) : x;
}

static inline void glj_rotate01(thread GLJacobi3 &a) {
    if (a.o01 == 0ul) return;
    const ulong two = 0x4000000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong tau = soft64_div(soft64_sub(a.d1, a.d0), soft64_mul(two, a.o01));
    const ulong root = soft64_sqrt(soft64_add(one, soft64_mul(tau, tau)));
    const ulong den = soft64_add(glj_abs(tau), root);
    const ulong t = soft64_div(soft64_less(tau, 0ul) ? soft64_negate_bits(one) : one, den);
    const ulong c = soft64_div(one, soft64_sqrt(soft64_add(one, soft64_mul(t, t))));
    const ulong s = soft64_mul(t, c);
    const ulong old = a.o01;
    a.d0 = soft64_sub(a.d0, soft64_mul(t, old));
    a.d1 = soft64_add(a.d1, soft64_mul(t, old));
    a.o01 = 0ul;
    const ulong x0 = a.o02, x1 = a.o12;
    a.o02 = soft64_sub(soft64_mul(c, x0), soft64_mul(s, x1));
    a.o12 = soft64_add(soft64_mul(s, x0), soft64_mul(c, x1));
}

static inline void glj_rotate02(thread GLJacobi3 &a) {
    if (a.o02 == 0ul) return;
    const ulong two = 0x4000000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong tau = soft64_div(soft64_sub(a.d2, a.d0), soft64_mul(two, a.o02));
    const ulong root = soft64_sqrt(soft64_add(one, soft64_mul(tau, tau)));
    const ulong den = soft64_add(glj_abs(tau), root);
    const ulong t = soft64_div(soft64_less(tau, 0ul) ? soft64_negate_bits(one) : one, den);
    const ulong c = soft64_div(one, soft64_sqrt(soft64_add(one, soft64_mul(t, t))));
    const ulong s = soft64_mul(t, c);
    const ulong old = a.o02;
    a.d0 = soft64_sub(a.d0, soft64_mul(t, old));
    a.d2 = soft64_add(a.d2, soft64_mul(t, old));
    a.o02 = 0ul;
    const ulong x0 = a.o01, x1 = a.o12;
    a.o01 = soft64_sub(soft64_mul(c, x0), soft64_mul(s, x1));
    a.o12 = soft64_add(soft64_mul(s, x0), soft64_mul(c, x1));
}

static inline void glj_rotate12(thread GLJacobi3 &a) {
    if (a.o12 == 0ul) return;
    const ulong two = 0x4000000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong tau = soft64_div(soft64_sub(a.d2, a.d1), soft64_mul(two, a.o12));
    const ulong root = soft64_sqrt(soft64_add(one, soft64_mul(tau, tau)));
    const ulong den = soft64_add(glj_abs(tau), root);
    const ulong t = soft64_div(soft64_less(tau, 0ul) ? soft64_negate_bits(one) : one, den);
    const ulong c = soft64_div(one, soft64_sqrt(soft64_add(one, soft64_mul(t, t))));
    const ulong s = soft64_mul(t, c);
    const ulong old = a.o12;
    a.d1 = soft64_sub(a.d1, soft64_mul(t, old));
    a.d2 = soft64_add(a.d2, soft64_mul(t, old));
    a.o12 = 0ul;
    const ulong x0 = a.o01, x1 = a.o02;
    a.o01 = soft64_sub(soft64_mul(c, x0), soft64_mul(s, x1));
    a.o02 = soft64_add(soft64_mul(s, x0), soft64_mul(c, x1));
}

static inline ulong glj_top(GLJacobi3 a) {
    for (uint i = 0; i < 8; ++i) {
        glj_rotate01(a); glj_rotate02(a); glj_rotate12(a);
    }
    ulong top = soft64_greater(a.d0, a.d1) ? a.d0 : a.d1;
    top = soft64_greater(top, a.d2) ? top : a.d2;
    return soft64_greater(top, 0ul) ? top : 0ul;
}

kernel void global_semilag_lipschitz_norms_fragment(
    constant GlobalSemilagLipschitzNormsFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagLipschitzNormsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint plane = uint(p.nlat) * uint(p.nlon);
    const uint count = uint(p.chunk_levels) * plane;
    if (tid >= count) return;
    const uint j = (tid % plane) / uint(p.nlon);
    const uint i = tid % uint(p.nlon);
    const ulong one = 0x3ff0000000000000ul;
    const ulong two = 0x4000000000000000ul;
    const ulong lref = r.lref[0];
    const ulong ex0 = soft64_negate_bits(r.sla[i]), ex1 = r.cla[i];
    const ulong ey0 = soft64_negate_bits(soft64_mul(r.sph[j], r.cla[i]));
    const ulong ey1 = soft64_negate_bits(soft64_mul(r.sph[j], r.sla[i]));
    const ulong ey2 = r.cph[j];
    const ulong j00 = soft64_add(soft64_mul(ex0,r.vx_dx[tid]),soft64_mul(ex1,r.vy_dx[tid]));
    const ulong j01 = soft64_add(soft64_mul(ex0,r.vx_dy[tid]),soft64_mul(ex1,r.vy_dy[tid]));
    const ulong j02 = soft64_div(soft64_add(soft64_mul(ex0,r.vx_dk[tid]),soft64_mul(ex1,r.vy_dk[tid])),lref);
    const ulong j10 = soft64_add(soft64_add(soft64_mul(ey0,r.vx_dx[tid]),soft64_mul(ey1,r.vy_dx[tid])),soft64_mul(ey2,r.vz_dx[tid]));
    const ulong j11 = soft64_add(soft64_add(soft64_mul(ey0,r.vx_dy[tid]),soft64_mul(ey1,r.vy_dy[tid])),soft64_mul(ey2,r.vz_dy[tid]));
    const ulong j12 = soft64_div(soft64_add(soft64_add(soft64_mul(ey0,r.vx_dk[tid]),soft64_mul(ey1,r.vy_dk[tid])),soft64_mul(ey2,r.vz_dk[tid])),lref);
    const ulong j20 = soft64_mul(lref,r.rate_dx[tid]);
    const ulong j21 = soft64_mul(lref,r.rate_dy[tid]);
    const ulong j22 = r.rate_dk[tid];
    const ulong frob2 = soft64_add(soft64_add(soft64_add(soft64_mul(j00,j00),soft64_mul(j01,j01)),soft64_mul(j02,j02)),
        soft64_add(soft64_add(soft64_add(soft64_mul(j10,j10),soft64_mul(j11,j11)),soft64_mul(j12,j12)),
        soft64_add(soft64_add(soft64_mul(j20,j20),soft64_mul(j21,j21)),soft64_mul(j22,j22))));
    r.frobenius[tid] = soft64_sqrt(frob2);
    const ulong h00 = soft64_add(soft64_mul(j00,j00),soft64_mul(j10,j10));
    const ulong h11 = soft64_add(soft64_mul(j01,j01),soft64_mul(j11,j11));
    const ulong h01 = soft64_add(soft64_mul(j00,j01),soft64_mul(j10,j11));
    const ulong hhalf = soft64_mul(0x3fe0000000000000ul,soft64_add(h00,h11));
    const ulong diff = soft64_mul(0x3fe0000000000000ul,soft64_sub(h00,h11));
    const ulong hrad = soft64_sqrt(soft64_add(soft64_mul(diff,diff),soft64_mul(h01,h01)));
    const ulong heig = soft64_add(hhalf,hrad);
    r.horizontal[tid] = soft64_sqrt(soft64_greater(heig,0ul) ? heig : 0ul);
    GLJacobi3 spectral = {
        soft64_add(h00,soft64_mul(j20,j20)),
        soft64_add(h11,soft64_mul(j21,j21)),
        soft64_add(soft64_add(soft64_mul(j02,j02),soft64_mul(j12,j12)),soft64_mul(j22,j22)),
        soft64_add(soft64_add(soft64_mul(j00,j01),soft64_mul(j10,j11)),soft64_mul(j20,j21)),
        soft64_add(soft64_add(soft64_mul(j00,j02),soft64_mul(j10,j12)),soft64_mul(j20,j22)),
        soft64_add(soft64_add(soft64_mul(j01,j02),soft64_mul(j11,j12)),soft64_mul(j21,j22))
    };
    r.spectral[tid] = soft64_sqrt(glj_top(spectral));
    const ulong e0 = soft64_mul(j02,lref), e1 = soft64_mul(j12,lref);
    const ulong d0 = r.rate_dx[tid], d1 = r.rate_dy[tid];
    const ulong emag = soft64_sqrt(soft64_add(soft64_mul(e0,e0),soft64_mul(e1,e1)));
    const ulong dmag = soft64_sqrt(soft64_add(soft64_mul(d0,d0),soft64_mul(d1,d1)));
    const ulong gmean = soft64_sqrt(soft64_mul(emag,dmag));
    const ulong se = soft64_greater(emag,0ul) ? soft64_div(gmean,emag) : 0ul;
    const ulong sd = soft64_greater(dmag,0ul) ? soft64_div(gmean,dmag) : 0ul;
    const ulong c02 = soft64_mul(e0,se), c12 = soft64_mul(e1,se);
    const ulong c20 = soft64_mul(d0,sd), c21 = soft64_mul(d1,sd);
    GLJacobi3 balanced = {
        soft64_add(h00,soft64_mul(c20,c20)),
        soft64_add(h11,soft64_mul(c21,c21)),
        soft64_add(soft64_add(soft64_mul(c02,c02),soft64_mul(c12,c12)),soft64_mul(j22,j22)),
        soft64_add(soft64_add(soft64_mul(j00,j01),soft64_mul(j10,j11)),soft64_mul(c20,c21)),
        soft64_add(soft64_add(soft64_mul(j00,c02),soft64_mul(j10,c12)),soft64_mul(c20,j22)),
        soft64_add(soft64_add(soft64_mul(j01,c02),soft64_mul(j11,c12)),soft64_mul(c21,j22))
    };
    r.balanced[tid] = soft64_sqrt(glj_top(balanced));
}
