#include <metal_stdlib>
#include "ieva_ww_split_bindings.metalh"
#include "ieva_mut_new_bindings.metalh"
#include "ieva_theta_minus_t0_bindings.metalh"
#include "soft_binary32.metalh"

using namespace metal;

// The CUDA source uses explicit round-to-nearest FP32 operations. Keep each
// Metal expression separate and prevent contraction across the same steps.
#pragma clang fp contract(off)

static inline float iadd(float a,float b,int strict) { return strict ? as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))) : a+b; }
static inline float isub(float a,float b,int strict) { return strict ? as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))) : a-b; }
static inline float imul(float a,float b,int strict) { return strict ? as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b))) : a*b; }
static inline float idiv(float a,float b,int strict) { return strict ? as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b))) : a/b; }
static inline bool ilt(float a,float b,int strict) { return strict ? soft32_less(as_type<uint>(a),as_type<uint>(b)) : a<b; }
static inline bool igt(float a,float b,int strict) { return strict ? soft32_greater(as_type<uint>(a),as_type<uint>(b)) : a>b; }
static inline float imin(float a,float b,int strict) { return strict ? (ilt(a,b,1)?a:b) : fmin(a,b); }
static inline float imax(float a,float b,int strict) { return strict ? (igt(a,b,1)?a:b) : fmax(a,b); }
#define IA(a,b) iadd((a),(b),p.strict)
#define IS(a,b) isub((a),(b),p.strict)
#define IM(a,b) imul((a),(b),p.strict)
#define ID(a,b) idiv((a),(b),p.strict)
#define IL(a,b) ilt((a),(b),p.strict)
#define IG(a,b) igt((a),(b),p.strict)
#define IN(a,b) imin((a),(b),p.strict)
#define IX(a,b) imax((a),(b),p.strict)

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
    const float su = IA(IA(IA(r.u[ua + 1], r.u[ua]), r.u[ub + 1]), r.u[ub]);
    const float cx = IM(IM(p.quarter, p.rdx), su);
    const ulong va = (ulong(k) * ulong(p.ny + 1) + ulong(j)) * ulong(p.nx) + ulong(i);
    const ulong vb = (ulong(k - 1) * ulong(p.ny + 1) + ulong(j)) * ulong(p.nx) + ulong(i);
    const float sv = IA(IA(IA(r.v[va + ulong(p.nx)], r.v[va]),
                      r.v[vb + ulong(p.nx)]), r.v[vb]);
    const float cy = IM(IM(p.quarter, p.rdy), sv);
    // __fsqrt_rn: correctly rounded, independent of the math mode.
    float hyp = as_type<float>(soft32_sqrt(as_type<uint>(IA(IM(cx, cx), IM(cy, cy)))));
    if (p.legacy != 0) {
        // The earlier WW_SPLIT takes the flow on the upwind mass level,
        // including the mass-point map factor, and uses cx+cy.
        const ulong ul = IL(w, p.zero) ? ua : ub;
        const ulong vl = IL(w, p.zero) ? va : vb;
        const float msf = p.has_msf != 0 ? r.msft[c] : p.one;
        const float lcx = IM(IM(msf, p.rdx), IS(IX(r.u[ul + 1], p.zero), IN(r.u[ul], p.zero)));
        const float lcy = IM(IM(msf, p.rdy), IS(IX(r.v[vl + ulong(p.nx)], p.zero), IN(r.v[vl], p.zero)));
        hyp = IA(lcx, lcy);
    }
    const float cw_max = IX(IS(p.alpha_max, IM(IM(p.dt, p.ceps), hyp)), p.zero);
    const float mass = IA(IM(r.c1f[k], r.mut[c]), r.c2f[k]);
    const float cr = ID(IM(IM(w, p.dt), r.rdnw[k]), mass);
    if (IG(cw_max, p.zero)) {
        const float cw_max2 = IM(cw_max, cw_max);
        const float cw_min = IM(cw_max, p.cmnx_ratio);
        const float cw = p.strict ? as_type<float>(as_type<uint>(cr) & 0x7fffffffu) : fabs(cr);
        float cff;
        if (IL(cw, cw_min)) {
            cff = cw_max2;
        } else if (IL(cw, IM(p.cutoff, cw_min))) {
            const float d = IS(cw, cw_min);
            cff = IA(cw_max2, IM(p.r4cmx, IM(d, d)));
        } else {
            cff = IM(cw_max, cw);
        }
        float wfrac = ID(cw_max2, cff);
        wfrac = IX(IN(wfrac, p.one), p.zero);
        r.ww_e[tid] = IM(w, wfrac);
        r.ww_i[tid] = IM(w, IS(p.one, wfrac));
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
    if (p.stored_inverse) {
        device const uint *u = reinterpret_cast<device const uint *>(r.u);
        device const uint *v = reinterpret_cast<device const uint *>(r.v);
        device const uint *muu = reinterpret_cast<device const uint *>(r.muu);
        device const uint *muv = reinterpret_cast<device const uint *>(r.muv);
        device const uint *c1 = reinterpret_cast<device const uint *>(r.c1h);
        device const uint *c2 = reinterpret_cast<device const uint *>(r.c2h);
        device const uint *rdnw = reinterpret_cast<device const uint *>(r.rdnw);
        device const uint *old = reinterpret_cast<device const uint *>(r.mut_old);
        device const uint *msft = reinterpret_cast<device const uint *>(r.msft);
        device const uint *msfu = reinterpret_cast<device const uint *>(r.msfu);
        device const uint *inv = reinterpret_cast<device const uint *>(r.msfv);
        uint numer = p.has_msf ? msft[c] : as_type<uint>(p.one);
        uint in = p.has_msf ? inv[fv + ulong(p.nx)] : as_type<uint>(p.one);
        uint is = p.has_msf ? inv[fv] : as_type<uint>(p.one);
        uint dmdt = 0u;
        for (int k = 0; k < p.nz; ++k) {
            ulong uk = ulong(k) * ulong(p.ny) * nxu + fu;
            ulong vk = ulong(k) * ulong(p.ny + 1) * ulong(p.nx) + fv;
            uint ae = soft32_mul(soft32_add(soft32_mul(c1[k], muu[fu+1]), c2[k]), u[uk+1]);
            uint aw = soft32_mul(soft32_add(soft32_mul(c1[k], muu[fu]), c2[k]), u[uk]);
            if (p.has_msf) { ae = soft32_div(ae, msfu[fu+1]); aw = soft32_div(aw, msfu[fu]); }
            uint bn = soft32_mul(soft32_add(soft32_mul(c1[k], muv[fv+ulong(p.nx)]), c2[k]), v[vk+ulong(p.nx)]);
            uint bs = soft32_mul(soft32_add(soft32_mul(c1[k], muv[fv]), c2[k]), v[vk]);
            if (p.has_msf) { bn = soft32_mul(bn, in); bs = soft32_mul(bs, is); }
            uint bracket = soft32_add(soft32_mul(as_type<uint>(p.rdx), soft32_sub(ae, aw)),
                                      soft32_mul(as_type<uint>(p.rdy), soft32_sub(bn, bs)));
            uint divv = soft32_mul(soft32_div(numer, rdnw[k]), bracket);
            dmdt = soft32_add(dmdt, divv);
        }
        reinterpret_cast<device uint *>(r.mut_new)[c] = soft32_sub(old[c], soft32_mul(as_type<uint>(p.dt), dmdt));
        return;
    }
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
    r.out[tid] = IA(IS(base, p.t0), r.thp[tid]);
}
