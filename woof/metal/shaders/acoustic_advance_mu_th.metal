#include <metal_stdlib>
#include "advance_mu_th_bindings.metalh"

using namespace metal;

static inline ulong mu_idx3(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

static inline float exact_total_divergence(
    constant AdvanceMuThResources &r, constant AdvanceMuThParams &q,
    int k, int j, int i, ulong cw, ulong ce, ulong cs, ulong cn,
    float msfu_w, float msfu_e, float msfv_s, float msfv_n, float msft) {
    int nxf = q.nx + 1;
    ulong uw = mu_idx3(k, j, i, q.ny, nxf), ue = uw + 1;
    ulong vs = mu_idx3(k, j, i, q.ny + 1, q.nx), vn = vs + ulong(q.nx);
    ulong c = ulong(j) * q.nx + i;
    float muw = 0.5f * (r.mup[c] + r.mup[cw] + r.mub2d[c] + r.mub2d[cw]);
    float mue = 0.5f * (r.mup[ce] + r.mup[c] + r.mub2d[ce] + r.mub2d[c]);
    float mus = 0.5f * (r.mup[c] + r.mup[cs] + r.mub2d[c] + r.mub2d[cs]);
    float mun = 0.5f * (r.mup[cn] + r.mup[c] + r.mub2d[cn] + r.mub2d[c]);
    float vs_full = r.v_pp[vs] + (r.c1h[k] * mus + r.c2h[k]) * r.v[vs] / msfv_s;
    float vn_full = r.v_pp[vn] + (r.c1h[k] * mun + r.c2h[k]) * r.v[vn] / msfv_n;
    float uw_full = r.u_pp[uw] + ((r.c1h[k] * muw + r.c2h[k]) * r.u[uw]) / msfu_w;
    float ue_full = r.u_pp[ue] + ((r.c1h[k] * mue + r.c2h[k]) * r.u[ue]) / msfu_e;
    return msft * msft * (q.rdy * (vn_full - vs_full)
                        + q.rdx * (ue_full - uw_full));
}

kernel void advance_mu_th(constant AdvanceMuThResources &r [[buffer(0)]],
                          constant AdvanceMuThParams &q [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    uint columns = uint(q.ny * q.nx);
    if (tid >= columns) return;
    int c0 = int(tid), j = c0 / q.nx, i = c0 - j * q.nx;
    ulong c = ulong(c0), st = ulong(q.ny) * ulong(q.nx);
    int nxf = q.nx + 1;
    r.mu_pp_old[c] = r.mu_pp[c];
    for (int k = 0; k < q.nz; ++k) {
        ulong h = ulong(k) * st + c;
        r.th_pp_old[h] = r.th_pp[h];
        r.p_pp_old[h] = r.p_pp[h];
    }
    bool frame = i < q.spec_zone || i >= q.nx - q.spec_zone
              || j < q.spec_zone || j >= q.ny - q.spec_zone;
    if (frame) {
#if GPUWM_WRF_EXACT
        if (q.write_mudf) r.mudf[c] = 0.0f;
        return;
#else
        float rmu = r.rmu_t[c];
        if (q.write_mudf) r.mudf[c] = 0.0f;
        {
            // Preserve the CUDA arn_mul/arn_add round points in frame tendencies.
#pragma clang fp contract(off)
            r.mu_pp[c] = r.mu_pp[c] + q.dtau * rmu;
            for (int k = 0; k < q.nz; ++k) {
                ulong h = ulong(k) * st + c;
                r.th_pp[h] = r.th_pp[h] + q.dtau * (r.rth_t[h] + (300.0f * r.c1h[k]) * rmu);
            }
        }
        return;
#endif
    }
    int ip = q.open_x ? min(i + 1, q.nx - 1) : (i + 1 == q.nx ? 0 : i + 1);
    int im = q.open_x ? max(i - 1, 0) : (i == 0 ? q.nx - 1 : i - 1);
    int jp = q.open_y ? min(j + 1, q.ny - 1) : (j + 1 == q.ny ? 0 : j + 1);
    int jm = q.open_y ? max(j - 1, 0) : (j == 0 ? q.ny - 1 : j - 1);
    ulong cw = ulong(j) * q.nx + im, ce = ulong(j) * q.nx + ip;
    ulong cs = ulong(jm) * q.nx + i, cn = ulong(jp) * q.nx + i;
    ulong bstr = q.base3d ? st : 1ul;
    float mc = q.mapped ? r.msft[c] : 1.0f;
    float msfu_w = q.mapped ? r.msfu[ulong(j) * nxf + i] : 1.0f;
    float msfu_e = q.mapped ? r.msfu[ulong(j) * nxf + i + 1] : 1.0f;
    float msfv_s = q.mapped ? r.msfv[ulong(j) * q.nx + i] : 1.0f;
    float msfv_n = q.mapped ? r.msfv[ulong(j + 1) * q.nx + i] : 1.0f;
    float dmdt_pp = 0.0f, dmdt_rf = 0.0f;
    for (int k = 0; k < q.nz; ++k) {
#if GPUWM_WRF_EXACT
        float div = exact_total_divergence(r, q, k, j, i, cw, ce, cs, cn,
                                            msfu_w, msfu_e, msfv_s, msfv_n, mc);
        dmdt_pp += r.dnw[k] * div;
#else
        ulong uw = mu_idx3(k, j, i, q.ny, nxf), ue = uw + 1;
        ulong vs = mu_idx3(k, j, i, q.ny + 1, q.nx), vn = vs + ulong(q.nx);
        float dvp = q.rdx * (r.u_pp[ue] - r.u_pp[uw]) + q.rdy * (r.v_pp[vn] - r.v_pp[vs]);
        float dvr;
        float muw = 0.5f * ((r.mub2d[cw] + r.mup[cw]) + (r.mub2d[c] + r.mup[c]));
        float mue = 0.5f * ((r.mub2d[c] + r.mup[c]) + (r.mub2d[ce] + r.mup[ce]));
        float mus = 0.5f * ((r.mub2d[cs] + r.mup[cs]) + (r.mub2d[c] + r.mup[c]));
        float mun = 0.5f * ((r.mub2d[c] + r.mup[c]) + (r.mub2d[cn] + r.mup[cn]));
        if (q.mapped) {
            dvr = q.rdx * ((r.c1h[k] * mue + r.c2h[k]) * r.u[ue] / msfu_e
                         - (r.c1h[k] * muw + r.c2h[k]) * r.u[uw] / msfu_w)
                 + q.rdy * ((r.c1h[k] * mun + r.c2h[k]) * r.v[vn] / msfv_n
                         - (r.c1h[k] * mus + r.c2h[k]) * r.v[vs] / msfv_s);
        } else {
            // The reference contracts the east/north products, after rounding
            // the west/south products. A subtraction permits either choice.
            float west = (r.c1h[k] * muw + r.c2h[k]) * r.u[uw];
            float south = (r.c1h[k] * mus + r.c2h[k]) * r.v[vs];
            float du = fma(r.c1h[k] * mue + r.c2h[k], r.u[ue], -west);
            float dv = fma(r.c1h[k] * mun + r.c2h[k], r.v[vn], -south);
            dvr = q.rdx * du + q.rdy * dv;
        }
        float map_area = mc * mc;
        dmdt_pp += r.dnw[k] * (map_area * dvp);
        dmdt_rf += r.dnw[k] * (map_area * dvr);
#endif
    }
    float rmu = r.rmu_t[c];
    {
        // Preserve CUDA mass-update round points without changing interior algebra.
#pragma clang fp contract(off)
#if GPUWM_WRF_EXACT
        float mass_tendency = dmdt_pp + rmu;
#else
        float mass_tendency = (dmdt_pp + dmdt_rf) + rmu;
#endif
        if (q.write_mudf) r.mudf[c] = mass_tendency;
        r.mu_pp[c] = r.mu_pp[c] + q.dtau * mass_tendency;
    }
    float wwk = 0.0f;
#if GPUWM_WRF_EXACT
    r.ww_pp[c] = 0.0f - r.ww_ref[c];
    for (int kk = 1; kk < q.nz; ++kk) {
        int k = kk - 1;
        float div = exact_total_divergence(r, q, k, j, i, cw, ce, cs, cn,
                                            msfu_w, msfu_e, msfv_s, msfv_n, mc);
        wwk = wwk - (r.dnw[k] * (r.c1h[k] * dmdt_pp + div + r.c1h[k] * rmu)) / mc;
        ulong hnext = ulong(kk) * st + c;
        r.ww_pp[hnext] = wwk - r.ww_ref[hnext];
    }
#else
    r.ww_pp[c] = 0.0f;
    for (int kk = 1; kk < q.nz; ++kk) {
        int k = kk - 1;
        ulong uw = mu_idx3(k, j, i, q.ny, nxf), ue = uw + 1;
        ulong vs = mu_idx3(k, j, i, q.ny + 1, q.nx), vn = vs + ulong(q.nx);
        float dvp = q.rdx * (r.u_pp[ue] - r.u_pp[uw]) + q.rdy * (r.v_pp[vn] - r.v_pp[vs]);
        if (q.mapped) {
            wwk = wwk - (r.dnw[k] * (r.c1h[k] * (dmdt_pp + rmu) + mc * mc * dvp)) / mc;
        } else {
            // Dividing by a runtime unit map factor would round the product
            // before the subtraction, unlike the flat-grid recurrence.
            float div = r.c1h[k] * (dmdt_pp + rmu) + dvp;
            wwk = fma(-r.dnw[k], div, wwk);
        }
        r.ww_pp[ulong(kk) * st + c] = wwk;
    }
#endif
    r.ww_pp[ulong(q.nz) * st + c] = 0.0f;

    float wdtn_k = 0.0f;
    for (int k2 = 0; k2 < q.nz; ++k2) {
        float wdtn_kp1 = 0.0f;
        if (k2 + 1 < q.nz) {
            ulong hnext = ulong(k2 + 1) * st + c;
            ulong hbelow = ulong(k2) * st + c;
            float t_up = r.thb[ulong(k2 + 1) * bstr + (q.base3d ? c : 0)] + r.thp[hnext];
            float t_dn = r.thb[ulong(k2) * bstr + (q.base3d ? c : 0)] + r.thp[hbelow];
            wdtn_kp1 = r.ww_pp[hnext] * (r.fnm[k2 + 1] * t_up + r.fnp[k2 + 1] * t_dn);
        }
        ulong h = ulong(k2) * st + c;
        float tc = r.thb[ulong(k2) * bstr + (q.base3d ? c : 0)] + r.thp[h];
        float tip = r.thb[ulong(k2) * bstr + (q.base3d ? ce : 0)] + r.thp[mu_idx3(k2,j,ip,q.ny,q.nx)];
        float tim = r.thb[ulong(k2) * bstr + (q.base3d ? cw : 0)] + r.thp[mu_idx3(k2,j,im,q.ny,q.nx)];
        float tjp = r.thb[ulong(k2) * bstr + (q.base3d ? cn : 0)] + r.thp[mu_idx3(k2,jp,i,q.ny,q.nx)];
        float tjm = r.thb[ulong(k2) * bstr + (q.base3d ? cs : 0)] + r.thp[mu_idx3(k2,jm,i,q.ny,q.nx)];
        ulong uw = mu_idx3(k2,j,i,q.ny,nxf), ue = uw + 1;
        ulong vs = mu_idx3(k2,j,i,q.ny+1,q.nx), vn = vs + ulong(q.nx);
        float hx = 0.5f * q.rdx * (r.u_pp[ue] * (tip + tc) - r.u_pp[uw] * (tc + tim));
        float hy = 0.5f * q.rdy * (r.v_pp[vn] * (tjp + tc) - r.v_pp[vs] * (tc + tjm));
#if GPUWM_WRF_EXACT
        r.th_pp[h] = r.th_pp[h] + mc * q.dtau * r.rth_t[h];
        r.th_pp[h] = r.th_pp[h] - q.dtau * mc * (mc * (hx + hy) + r.rdnw[k2] * (wdtn_kp1 - wdtn_k));
#else
        r.th_pp[h] = r.th_pp[h] + q.dtau * mc * (r.rth_t[h] - (mc * (hx + hy) + r.rdnw[k2] * (wdtn_kp1 - wdtn_k)));
#endif
        wdtn_k = wdtn_kp1;
    }
}
