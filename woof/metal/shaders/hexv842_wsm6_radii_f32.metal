#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "hexv842_wsm6_radii_f32_bindings.metalh"
using namespace ntc;
#pragma clang fp contract(off)

// mp_wsm6_effectrad_run for one solved column of the native wrapper.
kernel void hexv842_wsm6_radii_f32(
    constant Hexv842Wsm6RadiiF32Resources &r [[buffer(0)]],
    constant Hexv842Wsm6RadiiF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ns) || !p.enabled) return;

    int nc = p.nc, nz = p.nz;
    device const F *v = (device const F *)r.v;
    device const F *pi = (device const F *)r.pi;
    device const F *c = (device const F *)r.constants;
    device const F *state = (device const F *)r.state;
    device F *out = (device F *)r.radii;

    // Rows in the native WSM6 state and packed post-column profiles.
    F alpha = state[22 * nc + gid];
    F n0s = state[21 * nc + gid], n0smax = state[20 * nc + gid];
    F pidn0s = state[87 * nc + gid], pidnc = state[88 * nc + gid];
    F obmr = tc_div(F(1.f), F(3.f));
    for (int k = 0; k < nz; ++k) {
        int x = k * nc + int(gid);
        F t = p.wrapper ? tc_mul(v[x], pi[x]) : v[x];
        F rho = v[7 * nz * nc + x];
        F qc = v[2 * nz * nc + x], qi = v[3 * nz * nc + x], qs = v[5 * nz * nc + x];
        F rqc = tc_max(F(1.e-12f), tc_mul(qc, rho));
        F rqi = tc_max(F(1.e-12f), tc_mul(qi, rho));
        F temp = tc_mul(rho, tc_max(qi, c[11]));
        temp = tc_sqrt(tc_sqrt(tc_mul(tc_mul(temp, temp), temp)));
        F ni = tc_min(tc_max(tc_mul(F(5.38e7f), temp), F(1.e3f)), F(1.e6f));
        F rni = tc_max(F(1.e-6f), tc_mul(ni, rho));
        F rqs = tc_max(F(1.e-12f), tc_mul(qs, rho));

        F rc = c[20], ri = c[21], rs = c[22];
        if (rqc > F(1.e-12f)) {
            F lam = tc_pow(tc_div(tc_mul(pidnc, F(3.e8f)), rqc), obmr);
            rc = tc_max(tc_min(tc_mul(F(1.5f), tc_div(F(1.f), lam)), c[23]), F(2.51e-6f));
        }
        if (rqi > F(1.e-12f) && rni > F(1.e-6f)) {
            F dia = tc_mul(F(11.9f), tc_sqrt(tc_div(rqi, ni)));
            ri = tc_max(tc_min(tc_mul(tc_mul(F(.75f), F(.163f)), dia), c[24]), F(10.01e-6f));
        }
        if (rqs > F(1.e-12f)) {
            F sup = tc_sub(c[5], t);
            F fac = tc_max(tc_min(tc_exp(tc_mul(alpha, sup)), tc_div(n0smax, n0s)), F(1.f));
            F lam = tc_sqrt(tc_sqrt(tc_div(tc_mul(pidn0s, fac), rqs)));
            rs = tc_max(tc_min(tc_mul(F(.5f), tc_div(F(1.f), lam)), c[25]), F(25.e-6f));
        }
        out[x] = tc_max(c[20], tc_min(rc, c[23]));
        out[nz * nc + x] = tc_max(c[21], tc_min(ri, c[24]));
        out[2 * nz * nc + x] = tc_max(c[22], tc_min(rs, c[25]));
    }
}
