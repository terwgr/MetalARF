#include <metal_stdlib>
#include "advance_uv_bindings.metalh"

using namespace metal;

static inline float pdmp(device const float *p, device const float *pold,
                         ulong idx, float smdiv) {
    return p[idx] + smdiv * (p[idx] - pold[idx]);
}

static inline float dpn_face(constant AdvanceUvResources &r,
                             constant AdvanceUvParams &q, int kf,
                             ulong cA, ulong cB, ulong st) {
    const int nz = q.nz;
    if (kf == nz) {
        if (!q.top_lid) return 0.0f;
        return 0.5f * (
            q.cf1 * (pdmp(r.p_pp, r.p_pp_old, cA + ulong(nz - 1) * st, q.smdiv)
                   + pdmp(r.p_pp, r.p_pp_old, cB + ulong(nz - 1) * st, q.smdiv))
          + q.cf2 * (pdmp(r.p_pp, r.p_pp_old, cA + ulong(nz - 2) * st, q.smdiv)
                   + pdmp(r.p_pp, r.p_pp_old, cB + ulong(nz - 2) * st, q.smdiv))
          + q.cf3 * (pdmp(r.p_pp, r.p_pp_old, cA + ulong(nz - 3) * st, q.smdiv)
                   + pdmp(r.p_pp, r.p_pp_old, cB + ulong(nz - 3) * st, q.smdiv)));
    }
    if (kf == 0)
        return 0.5f * (q.cf1 * (pdmp(r.p_pp, r.p_pp_old, cA, q.smdiv)
                               + pdmp(r.p_pp, r.p_pp_old, cB, q.smdiv))
                     + q.cf2 * (pdmp(r.p_pp, r.p_pp_old, cA + st, q.smdiv)
                               + pdmp(r.p_pp, r.p_pp_old, cB + st, q.smdiv))
                     + q.cf3 * (pdmp(r.p_pp, r.p_pp_old, cA + 2ul * st, q.smdiv)
                               + pdmp(r.p_pp, r.p_pp_old, cB + 2ul * st, q.smdiv)));
    return 0.5f * (r.fnm[kf] * (pdmp(r.p_pp, r.p_pp_old, cA + ulong(kf) * st, q.smdiv)
                               + pdmp(r.p_pp, r.p_pp_old, cB + ulong(kf) * st, q.smdiv))
                 + r.fnp[kf] * (pdmp(r.p_pp, r.p_pp_old, cA + ulong(kf - 1) * st, q.smdiv)
                               + pdmp(r.p_pp, r.p_pp_old, cB + ulong(kf - 1) * st, q.smdiv)));
}

static inline float pgrad_face(constant AdvanceUvResources &r,
                               constant AdvanceUvParams &q,
                               ulong cA, ulong cB, int k, ulong st,
                               float rd) {
#if GPUWM_WRF_EXACT
    const float muf = 0.5f * (r.mup[cA] + r.mup[cB] + r.mub2d[cA] + r.mub2d[cB]);
#else
    const float muf = 0.5f * ((r.mub2d[cB] + r.mup[cB])
                            + (r.mub2d[cA] + r.mup[cA]));
#endif
    const ulong a = cA + ulong(k) * st;
    const ulong b = cB + ulong(k) * st;
    const float dph = (r.ph_pp[a + st] - r.ph_pp[b + st])
                    + (r.ph_pp[a] - r.ph_pp[b]);
    const float dpe = pdmp(r.p_pp, r.p_pp_old, a, q.smdiv)
                    - pdmp(r.p_pp, r.p_pp_old, b, q.smdiv);
    const float dpb = q.base3d ? r.pb[a] - r.pb[b] : 0.0f;
    float result = 0.5f * rd * (r.c1h[k] * muf + r.c2h[k])
        * (dph + (r.alt[a] + r.alt[b]) * dpe
                + (r.al_pp[a] + r.al_pp[b]) * dpb);
    const float dpn_up = dpn_face(r, q, k + 1, cA, cB, st);
    const float dpn_dn = dpn_face(r, q, k, cA, cB, st);
#if GPUWM_WRF_EXACT
    const ulong ba = q.base3d ? a : ulong(k);
    const ulong bb = q.base3d ? b : ulong(k);
    const ulong bst = q.base3d ? st : 1ul;
    const float half_a = 0.5f * (r.phb[ba] + r.phb[ba + bst]
                               + r.php[a] + r.php[a + st]);
    const float half_b = 0.5f * (r.phb[bb] + r.phb[bb + bst]
                               + r.php[b] + r.php[b + st]);
    const float dphp = half_a - half_b;
    result += rd * dphp * (r.rdnw[k] * (dpn_up - dpn_dn)
                           - 0.5f * ((r.c1h[k] * r.mu_pp[cB])
                                   + (r.c1h[k] * r.mu_pp[cA])));
#else
    float dphp = 0.5f * ((r.php[a] + r.php[a + st])
                       - (r.php[b] + r.php[b + st]));
    if (q.base3d)
        dphp += 0.5f * ((r.phb[a] + r.phb[a + st])
                      - (r.phb[b] + r.phb[b + st]));
    result += rd * dphp * (r.rdnw[k] * (dpn_up - dpn_dn)
                           - 0.5f * r.c1h[k]
                               * (r.mu_pp[cB] + r.mu_pp[cA]));
#endif
    return result;
}

kernel void advance_uv(constant AdvanceUvResources &r [[buffer(0)]],
                       constant AdvanceUvParams &q [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
    const int nyf = q.ny + 1;
    const int nxf = q.nx + 1;
    if (tid >= uint(q.nz * nyf * nxf)) return;
    const int k = int(tid) / (nyf * nxf);
    const int rem = int(tid) - k * nyf * nxf;
    const int j = rem / nxf;
    const int i = rem - j * nxf;
    const ulong st = ulong(q.ny) * ulong(q.nx);

    if (j < q.ny && j >= q.spec_zone && j < q.ny - q.spec_zone
        && i >= q.spec_zone && i <= q.nx - q.spec_zone
        && !(q.radiative_x && (i == 0 || i == q.nx))) {
        const int iA = (i == q.nx ? 0 : i);
        const int iB = (i == 0 ? q.nx - 1 : i - 1);
        const ulong cA = ulong(j) * ulong(q.nx) + ulong(iA);
        const ulong cB = ulong(j) * ulong(q.nx) + ulong(iB);
        const float dpxy = pgrad_face(r, q, cA, cB, k, st, q.rdx);
        const ulong uix = (ulong(k) * ulong(q.ny) + ulong(j))
                        * ulong(nxf) + ulong(i);
#if GPUWM_WRF_EXACT
        r.u_pp[uix] = r.u_pp[uix] + q.dtau * r.ru_t[uix];
        const float cq = q.moist_cq ? r.cqu[uix] : 1.0f;
        r.u_pp[uix] = r.u_pp[uix] - q.dtau * cq * dpxy;
#else
        if (q.moist_cq)
            r.u_pp[uix] += q.dtau * (r.ru_t[uix] - r.cqu[uix] * dpxy);
        else
            r.u_pp[uix] += q.dtau * (r.ru_t[uix] - dpxy);
#endif
    } else if (j < q.ny) {
        const ulong uix = (ulong(k) * ulong(q.ny) + ulong(j))
                        * ulong(nxf) + ulong(i);
        r.u_pp[uix] = r.u_pp[uix] + q.dtau * r.ru_t[uix];
    }

    if (i < q.nx && i >= q.spec_zone && i < q.nx - q.spec_zone
        && j >= q.spec_zone && j <= q.ny - q.spec_zone
        && !(q.radiative_y && (j == 0 || j == q.ny))) {
        const int jA = (j == q.ny ? 0 : j);
        const int jB = (j == 0 ? q.ny - 1 : j - 1);
        const ulong cA = ulong(jA) * ulong(q.nx) + ulong(i);
        const ulong cB = ulong(jB) * ulong(q.nx) + ulong(i);
        const float dpxy = pgrad_face(r, q, cA, cB, k, st, q.rdy);
        const ulong vix = (ulong(k) * ulong(nyf) + ulong(j))
                        * ulong(q.nx) + ulong(i);
#if GPUWM_WRF_EXACT
        r.v_pp[vix] = r.v_pp[vix] + q.dtau * r.rv_t[vix];
        const float cq = q.moist_cq ? r.cqv[vix] : 1.0f;
        r.v_pp[vix] = r.v_pp[vix] - q.dtau * cq * dpxy;
#else
        if (q.moist_cq)
            r.v_pp[vix] += q.dtau * (r.rv_t[vix] - r.cqv[vix] * dpxy);
        else
            r.v_pp[vix] += q.dtau * (r.rv_t[vix] - dpxy);
#endif
    } else if (i < q.nx) {
        const ulong vix = (ulong(k) * ulong(nyf) + ulong(j))
                        * ulong(q.nx) + ulong(i);
        r.v_pp[vix] = r.v_pp[vix] + q.dtau * r.rv_t[vix];
    }
}
