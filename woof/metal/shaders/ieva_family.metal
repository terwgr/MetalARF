#include <metal_stdlib>
#include "ieva_ww_split_bindings.metalh"
#include "ieva_mut_new_bindings.metalh"
#include "ieva_theta_minus_t0_bindings.metalh"
#include "soft_binary32.metalh"

using namespace metal;

// The CUDA source uses explicit round-to-nearest FP32 operations. Keep each
// Metal expression separate and prevent contraction across the same steps.
#pragma clang fp contract(off)

kernel void ieva_ww_split(constant IevaWwSplitResources &r [[buffer(0)]],
                          constant IevaWwSplitParams &p [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    const ulong st = ulong(p.ny) * ulong(p.nx);
    const ulong count = ulong(p.nz + 1) * st;
    if (ulong(tid) >= count) return;
    const int k = int(ulong(tid) / st);
    const ulong c = ulong(tid) - ulong(k) * st;
    const int j = int(c / ulong(p.nx));
    const int i = int(c - ulong(j) * ulong(p.nx));
    const float w = r.ww[tid];
    if (k == 0 || k == p.nz) {
        r.ww_e[tid] = w;
        r.ww_i[tid] = p.zero;
        return;
    }

    const ulong nxu = ulong(p.nx + 1);
    const ulong ua = (ulong(k) * ulong(p.ny) + ulong(j)) * nxu + ulong(i);
    const ulong ub = (ulong(k - 1) * ulong(p.ny) + ulong(j)) * nxu + ulong(i);
    const float su = ((r.u[ua + 1] + r.u[ua]) + r.u[ub + 1]) + r.u[ub];
    const float cx = (p.quarter * p.rdx) * su;
    const ulong va = (ulong(k) * ulong(p.ny + 1) + ulong(j)) * ulong(p.nx) + ulong(i);
    const ulong vb = (ulong(k - 1) * ulong(p.ny + 1) + ulong(j)) * ulong(p.nx) + ulong(i);
    const float sv = ((r.v[va + ulong(p.nx)] + r.v[va]) +
                      r.v[vb + ulong(p.nx)]) + r.v[vb];
    const float cy = (p.quarter * p.rdy) * sv;
    // __fsqrt_rn: correctly rounded, independent of the math mode.
    float hyp = as_type<float>(soft32_sqrt(as_type<uint>((cx * cx) + (cy * cy))));
    if (p.legacy != 0) {
        // The earlier WW_SPLIT takes the flow on the upwind mass level,
        // including the mass-point map factor, and uses cx+cy.
        const ulong ul = (w < p.zero) ? ua : ub;
        const ulong vl = (w < p.zero) ? va : vb;
        const float msf = p.has_msf != 0 ? r.msft[c] : p.one;
        const float lcx = (msf * p.rdx) *
            (fmax(r.u[ul + 1], p.zero) - fmin(r.u[ul], p.zero));
        const float lcy = (msf * p.rdy) *
            (fmax(r.v[vl + ulong(p.nx)], p.zero) - fmin(r.v[vl], p.zero));
        hyp = lcx + lcy;
    }
    const float cw_max = fmax((p.alpha_max - ((p.dt * p.ceps) * hyp)), p.zero);
    const float mass = (r.c1f[k] * r.mut[c]) + r.c2f[k];
    const float cr = ((w * p.dt) * r.rdnw[k]) / mass;
    if (cw_max > p.zero) {
        const float cw_max2 = cw_max * cw_max;
        const float cw_min = cw_max * p.cmnx_ratio;
        const float cw = fabs(cr);
        float cff;
        if (cw < cw_min) {
            cff = cw_max2;
        } else if (cw < p.cutoff * cw_min) {
            const float d = cw - cw_min;
            cff = cw_max2 + (p.r4cmx * (d * d));
        } else {
            cff = cw_max * cw;
        }
        float wfrac = cw_max2 / cff;
        wfrac = fmax(fmin(wfrac, p.one), p.zero);
        r.ww_e[tid] = w * wfrac;
        r.ww_i[tid] = w * (p.one - wfrac);
    } else {
        r.ww_e[tid] = p.zero;
        r.ww_i[tid] = w;
    }
}

kernel void ieva_mut_new(constant IevaMutNewResources &r [[buffer(0)]],
                         constant IevaMutNewParams &p [[buffer(1)]],
                         uint c [[thread_position_in_grid]]) {
    const ulong st = ulong(p.ny) * ulong(p.nx);
    if (ulong(c) >= st) return;
    const int j = int(c) / p.nx;
    const int i = int(c) - j * p.nx;
    const ulong nxu = ulong(p.nx + 1);
    const ulong fu = ulong(j) * nxu + ulong(i);
    const ulong fv = ulong(j) * ulong(p.nx) + ulong(i);
    const float numer = p.has_msf ? r.msft[c] : p.one;
    float inv_n = p.one;
    float inv_s = p.one;
    if (p.has_msf) {
        inv_n = p.one / r.msfv[fv + ulong(p.nx)];
        inv_s = p.one / r.msfv[fv];
    }
    float dmdt = 0.0f;
    for (int k = 0; k < p.nz; ++k) {
        const ulong uk = ulong(k) * ulong(p.ny) * nxu + fu;
        const ulong vk = ulong(k) * ulong(p.ny + 1) * ulong(p.nx) + fv;
        float ae = ((r.c1h[k] * r.muu[fu + 1]) + r.c2h[k]) * r.u[uk + 1];
        float aw = ((r.c1h[k] * r.muu[fu]) + r.c2h[k]) * r.u[uk];
        if (p.has_msf) {
            ae = ae / r.msfu[fu + 1];
            aw = aw / r.msfu[fu];
        }
        float bn = ((r.c1h[k] * r.muv[fv + ulong(p.nx)]) + r.c2h[k]) *
                    r.v[vk + ulong(p.nx)];
        float bs = ((r.c1h[k] * r.muv[fv]) + r.c2h[k]) * r.v[vk];
        if (p.has_msf) {
            bn = bn * inv_n;
            bs = bs * inv_s;
        }
        const float bracket = (p.rdx * (ae - aw)) + (p.rdy * (bn - bs));
        const float divv = (numer / r.rdnw[k]) * bracket;
        dmdt = dmdt + divv;
    }
    r.mut_new[c] = r.mut_old[c] - (p.dt * dmdt);
}

// WRF t_2 = (thb - t0) + theta' (woof.core.ieva.theta_minus_t0): the base
// subtraction first, so it is exact, then one rounded add.
kernel void ieva_theta_minus_t0(constant IevaThetaMinusT0Resources &r [[buffer(0)]],
                                constant IevaThetaMinusT0Params &p [[buffer(1)]],
                                uint tid [[thread_position_in_grid]]) {
    const ulong plane = ulong(p.ny) * ulong(p.nx);
    if (ulong(tid) >= ulong(p.nz) * plane) return;
    const float base = p.base3d != 0 ? r.thb[tid] : r.thb[ulong(tid) / plane];
    r.out[tid] = (base - p.t0) + r.thp[tid];
}
