#include <metal_stdlib>
#include "advance_w_phi_bindings.metalh"
#include "upper_wind_limiter_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"

using namespace metal;

constant float WOOF_GRAVITY = 9.81f;
#define WPHI_MAX_LEV 257

// NOAA WRFV3.9 advance_w's saved-wind limiter. One thread owns a
// column; qq's descending recurrence must not be parallelized by level.
kernel void upper_wind_limiter(constant UpperWindLimiterResources &r [[buffer(0)]],
                              constant UpperWindLimiterParams &p [[buffer(1)]],
                              uint col [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (col>=uint(p.ny)*uint(p.nx)) return;
    const int j=int(col/uint(p.nx)), i=int(col%uint(p.nx));
    if (i<p.zone || i>=p.nx-p.zone || j<p.zone || j>=p.ny-p.zone) return;
    float qq=0.0f;
    for (int k=p.nz-1;k>=p.nz-3;--k) {
        const float uk=r.u[(k*p.ny+j)*(p.nx+1)+i];
        const float vk=r.v[(k*(p.ny+1)+j)*p.nx+i];
        qq=max(qq,uk*uk+vk*vk);
    }
    qq=qq/12100.0f;
    if (qq<=1.0f) return;
    const int top=p.nz*p.ny*p.nx+int(col);
    const float htop=(r.php[top]+r.phb[p.base3d ? top : p.nz])/WOOF_GRAVITY;
    const float hbot=htop-p.hdepth;
    for (int k=p.nz-1;k>=1;--k) {
        const int pos=k*p.ny*p.nx+int(col);
        const float hk=(r.php[pos]+r.phb[p.base3d ? pos : k])/WOOF_GRAVITY;
        if (hk>=hbot && qq>1.0f) {
            qq=sqrt((qq-1.0f)*qq);
            float drag=(p.dts/86400.0f)*2.0f;
            drag=drag*qq;
            const float exponent=0.4f*float(k-(p.nz-1));
            const float taper=as_type<float>(soft_gfk_pow_bits(
                as_type<uint>(2.0f),as_type<uint>(exponent)));
            drag=drag*taper;
            const float factor=1.0f-drag;
            const int up=(k*p.ny+j)*(p.nx+1)+i;
            const int vp=(k*(p.ny+1)+j)*p.nx+i;
            r.u[up]=r.u[up]*factor;
            r.v[vp]=r.v[vp]*factor;
        }
    }
}

static inline ulong wphi_idx3(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

static inline float wphi_theta_average(constant AdvanceWPhiResources &r,
                                       constant AdvanceWPhiParams &q,
                                       ulong h, int k, ulong column,
                                       ulong bstr, ulong boff,
                                       float muave, float muts, float mut) {
    float average = 0.5f * ((1.0f + q.epssm) * r.th_pp[h]
                          + (1.0f - q.epssm) * r.th_pp_old[h]);
#if GPUWM_WRF_EXACT
    return average + (r.c1h[k] * muave) * 300.0f
        / ((r.c1h[k] * muts + r.c2h[k]) * (300.0f + r.thp[h]));
#else
    ulong base = ulong(k) * bstr + boff;
    return average / ((r.c1h[k] * muts + r.c2h[k])
                    * (r.thb[base] + r.thp[h]));
#endif
}

static inline void wphi_diagnose_p(constant AdvanceWPhiResources &r,
                                   constant AdvanceWPhiParams &q,
                                   ulong column, ulong st, ulong bstr,
                                   ulong boff, float muts) {
    for (int k = 0; k < q.nz; ++k) {
        ulong h = ulong(k) * st + column;
        float chm = r.c1h[k] * muts + r.c2h[k];
#if GPUWM_WRF_EXACT
        float th_ref = r.thp[h];
        float al = -1.0f / chm
                 * (r.alt[h] * (r.c1h[k] * r.mu_pp[column])
                    + r.rdnw[k] * (r.ph_pp[h + st] - r.ph_pp[h]));
        r.al_pp[h] = al;
        r.p_pp[h] = r.c2a[h]
                  * (r.alt[h] * (r.th_pp[h]
                       - (r.c1h[k] * r.mu_pp[column]) * th_ref)
                       / (chm * (300.0f + th_ref)) - al);
#else
        float th_ref = r.thb[q.base3d ? h : ulong(k)] + r.thp[h];
        float al = -(r.alt[h] * (r.c1h[k] * r.mu_pp[column])
                    + r.rdnw[k] * (r.ph_pp[h + st] - r.ph_pp[h])) / chm;
        r.al_pp[h] = al;
        r.p_pp[h] = r.c2a[h]
                  * (r.alt[h] * (r.th_pp[h]
                       - r.c1h[k] * r.mu_pp[column] * th_ref)
                       / (chm * th_ref) - al);
#endif
    }
}

kernel void advance_w_phi(constant AdvanceWPhiResources &r [[buffer(0)]],
                          constant AdvanceWPhiParams &q [[buffer(1)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.ny * q.nx) || q.nz + 1 > int(WPHI_MAX_LEV)) return;
    int c0 = int(tid), j = c0 / q.nx, i = c0 - j * q.nx;
    if (i < q.spec_zone || i >= q.nx - q.spec_zone
        || j < q.spec_zone || j >= q.ny - q.spec_zone) return;
    ulong column = ulong(c0), st = ulong(q.ny) * ulong(q.nx);
    ulong bstr = q.base3d ? st : 1ul;
    ulong boff = q.base3d ? column : 0ul;
    float mut = r.mub2d[column] + r.mup[column];
    float muts = mut + r.mu_pp[column];
    float muave = 0.5f * ((1.0f + q.epssm) * r.mu_pp[column]
                        + (1.0f - q.epssm) * r.mu_pp_old[column]);
    float msf = q.mapped ? r.msft[column] : 1.0f;
    float msf_inv = 1.0f / msf;

    thread float rhs[WPHI_MAX_LEV];
    rhs[0] = 0.0f;
    for (int k = 1; k <= q.nz; ++k) {
        ulong f = ulong(k) * st + column;
        rhs[k] = q.dtau * (r.rph_t[f]
                           + 0.5f * WOOF_GRAVITY * (1.0f - q.epssm) * r.w_pp[f]);
    }
    float ph_lo = r.phb[boff] + r.php[column];
    float ph_hi = r.phb[bstr + boff] + r.php[st + column];
#if GPUWM_WRF_EXACT
    float wd_lo = 0.5f * (r.ww_pp[st + column] + r.ww_pp[column])
                * r.rdnw[0] * (r.php[st + column] - r.php[column]
                + r.phb[bstr + boff] - r.phb[boff]);
#else
    float wd_lo = 0.5f * (r.ww_pp[st + column] + r.ww_pp[column])
                * r.rdnw[0] * (ph_hi - ph_lo);
#endif
    for (int k = 1; k < q.nz; ++k) {
        ph_lo = ph_hi;
        ph_hi = r.phb[ulong(k + 1) * bstr + boff]
              + r.php[ulong(k + 1) * st + column];
        float wd_hi;
#if GPUWM_WRF_EXACT
        wd_hi = 0.5f * (r.ww_pp[ulong(k + 1) * st + column]
                      + r.ww_pp[ulong(k) * st + column]) * r.rdnw[k]
              * (r.php[ulong(k + 1) * st + column]
                 - r.php[ulong(k) * st + column]
                 + r.phb[ulong(k + 1) * bstr + boff]
                 - r.phb[ulong(k) * bstr + boff]);
#else
        wd_hi = 0.5f * (r.ww_pp[ulong(k + 1) * st + column]
                      + r.ww_pp[ulong(k) * st + column])
              * r.rdnw[k] * (ph_hi - ph_lo);
#endif
        rhs[k] -= q.dtau * (r.fnm[k] * wd_hi + r.fnp[k] * wd_lo);
        wd_lo = wd_hi;
    }
    for (int k = 1; k <= q.nz; ++k) {
        ulong f = ulong(k) * st + column;
        float denom = r.c1f[k] * mut + r.c2f[k];
        rhs[k] = r.ph_pp[f] + msf * rhs[k] / denom;
    }
    if (q.top_lid) rhs[q.nz] = 0.0f;

    int ip = q.boundary_x ? min(i + 1, q.nx - 1) : (i + 1) % q.nx;
    int im = q.boundary_x ? max(i - 1, 0) : (i - 1 + q.nx) % q.nx;
    int jp = q.boundary_y ? min(j + 1, q.ny - 1) : (j + 1) % q.ny;
    int jm = q.boundary_y ? max(j - 1, 0) : (j - 1 + q.ny) % q.ny;
    int nxf = q.nx + 1;
    ulong cip = ulong(j) * q.nx + ip, cim = ulong(j) * q.nx + im;
    ulong cjp = ulong(jp) * q.nx + i, cjm = ulong(jm) * q.nx + i;
    float ue = q.cf1 * r.u_pp[wphi_idx3(0,j,i+1,q.ny,nxf)]
             + q.cf2 * r.u_pp[wphi_idx3(1,j,i+1,q.ny,nxf)]
             + q.cf3 * r.u_pp[wphi_idx3(2,j,i+1,q.ny,nxf)];
    float uw = q.cf1 * r.u_pp[wphi_idx3(0,j,i,q.ny,nxf)]
             + q.cf2 * r.u_pp[wphi_idx3(1,j,i,q.ny,nxf)]
             + q.cf3 * r.u_pp[wphi_idx3(2,j,i,q.ny,nxf)];
    float vn = q.cf1 * r.v_pp[wphi_idx3(0,j+1,i,q.ny+1,q.nx)]
             + q.cf2 * r.v_pp[wphi_idx3(1,j+1,i,q.ny+1,q.nx)]
             + q.cf3 * r.v_pp[wphi_idx3(2,j+1,i,q.ny+1,q.nx)];
    float vs = q.cf1 * r.v_pp[wphi_idx3(0,j,i,q.ny+1,q.nx)]
             + q.cf2 * r.v_pp[wphi_idx3(1,j,i,q.ny+1,q.nx)]
             + q.cf3 * r.v_pp[wphi_idx3(2,j,i,q.ny+1,q.nx)];
    float terrain_v = 0.5f * q.rdy * ((r.ht[cjp] - r.ht[column]) * vn
                                   + (r.ht[column] - r.ht[cjm]) * vs);
    float terrain_u = 0.5f * q.rdx * ((r.ht[cip] - r.ht[column]) * ue
                                   + (r.ht[column] - r.ht[cim]) * uw);
#if GPUWM_WRF_EXACT
    r.w_pp[column] = msf * terrain_v + msf * terrain_u;
#else
    r.w_pp[column] = msf * (terrain_v + terrain_u);
#endif

    float t2_dn = wphi_theta_average(r,q,column,0,column,bstr,boff,muave,muts,mut);
    for (int k = 1; k < q.nz; ++k) {
        ulong h = ulong(k) * st + column, hm = h - st;
        float t2_up = wphi_theta_average(r,q,h,k,column,bstr,boff,muave,muts,mut);
        float dph_up = (1.0f + q.epssm) * (rhs[k+1] - rhs[k])
                     + (1.0f - q.epssm) * (r.ph_pp[ulong(k+1)*st+column]
                                            - r.ph_pp[ulong(k)*st+column]);
        float dph_dn = (1.0f + q.epssm) * (rhs[k] - rhs[k-1])
                     + (1.0f - q.epssm) * (r.ph_pp[ulong(k)*st+column]
                                            - r.ph_pp[ulong(k-1)*st+column]);
        ulong f = ulong(k) * st + column;
        float denom = r.c1h[k] * mut + r.c2h[k];
        float denom_prev = r.c1h[k-1] * mut + r.c2h[k-1];
        float pressure_term = 0.5f * q.dtau * WOOF_GRAVITY * r.rdn[k]
            * (r.c2a[h] * r.rdnw[k] / denom * dph_up
               - r.c2a[hm] * r.rdnw[k-1] / denom_prev * dph_dn);
#if GPUWM_WRF_EXACT
        if (q.moist_cq) {
            r.w_pp[f] = r.w_pp[f] + q.dtau * r.rw_t[f]
                + (msf_inv * r.cqw[f]) * pressure_term
                + q.dtau * WOOF_GRAVITY * msf_inv
                  * (r.rdn[k] * (r.c2a[h] * r.alt[h] * t2_up
                               - r.c2a[hm] * r.alt[hm] * t2_dn)
                     - r.c1f[k] * muave);
        } else {
            r.w_pp[f] = r.w_pp[f] + q.dtau * r.rw_t[f]
                + msf_inv * pressure_term
                + q.dtau * WOOF_GRAVITY * msf_inv
                  * (r.rdn[k] * (r.c2a[h] * r.alt[h] * t2_up
                               - r.c2a[hm] * r.alt[hm] * t2_dn)
                     - r.c1f[k] * muave);
        }
#else
        if (q.moist_cq) {
            r.w_pp[f] += q.dtau * r.rw_t[f]
                + (msf_inv * r.cqw[f]) * pressure_term
                + q.dtau * WOOF_GRAVITY * msf_inv
                  * (r.rdn[k] * (r.c2a[h] * r.alt[h] * t2_up
                               - r.c2a[hm] * r.alt[hm] * t2_dn)
                     - r.c1f[k] * muave);
        } else {
            r.w_pp[f] += q.dtau * r.rw_t[f]
                + msf_inv * pressure_term
                + q.dtau * WOOF_GRAVITY * msf_inv
                  * (r.rdn[k] * (r.c2a[h] * r.alt[h] * t2_up
                               - r.c2a[hm] * r.alt[hm] * t2_dn)
                     - r.c1f[k] * muave);
        }
#endif
        t2_dn = t2_up;
    }
    if (q.top_lid) {
        r.w_pp[ulong(q.nz) * st + column] = 0.0f;
    } else {
        int k = q.nz - 1;
        ulong h = ulong(k) * st + column, f = ulong(q.nz) * st + column;
        float dph_dn = (1.0f + q.epssm) * (rhs[q.nz] - rhs[q.nz - 1])
                     + (1.0f - q.epssm) * (r.ph_pp[f] - r.ph_pp[h]);
        float denom = r.c1h[k] * mut + r.c2h[k];
#if GPUWM_WRF_EXACT
        float top_pressure = (-0.5f * q.dtau * WOOF_GRAVITY / denom)
            * (r.rdnw[k] * r.rdnw[k]) * 2.0f * r.c2a[h] * dph_dn;
#else
        float top_pressure = -q.dtau * WOOF_GRAVITY * r.c2a[h]
            * r.rdnw[k] * r.rdnw[k] / denom * dph_dn;
#endif
        float top_buoyancy = -q.dtau * WOOF_GRAVITY
            * (2.0f * r.rdnw[k] * r.c2a[h] * r.alt[h] * t2_dn
               + r.c1f[q.nz] * muave);
#if GPUWM_WRF_EXACT
        r.w_pp[f] = r.w_pp[f] + q.dtau * r.rw_t[f]
                  + msf_inv * (top_pressure + top_buoyancy);
#else
        r.w_pp[f] += q.dtau * r.rw_t[f]
                   + msf_inv * (top_pressure + top_buoyancy);
#endif
    }

    for (int k = 1; k <= q.nz; ++k) {
        ulong h = ulong(k) * st + column;
        r.w_pp[h] = (r.w_pp[h] - r.a[h] * r.w_pp[h-st]) * r.alpha[h];
    }
    for (int k = q.nz - 1; k >= 1; --k) {
        ulong h = ulong(k) * st + column;
        r.w_pp[h] -= r.gam[h] * r.w_pp[h+st];
    }
    if (q.dampmag > 0.0f) {
        float htop = (r.phb[ulong(q.nz)*bstr+boff] + r.php[ulong(q.nz)*st+column]) / WOOF_GRAVITY;
        float hbot = htop - q.zdamp;
        for (int k = 1; k <= q.nz; ++k) {
            ulong h = ulong(k) * st + column;
            float hk = (r.phb[ulong(k)*bstr+boff] + r.php[h]) / WOOF_GRAVITY;
            if (hk >= hbot) {
                float sn = sin(1.5707963267948966f * (hk - hbot) / q.zdamp);
                float dw = q.dampmag * sn * sn;
                float cfm = r.c1f[k] * mut + r.c2f[k];
                r.w_pp[h] = (r.w_pp[h] - dw * cfm * r.w_ref[h]) / (1.0f + dw);
            }
        }
    }
    for (int k = 1; k <= q.nz; ++k) {
        ulong h = ulong(k) * st + column;
        r.ph_pp[h] = rhs[k] + msf * 0.5f * q.dtau * WOOF_GRAVITY
            * (1.0f + q.epssm) * r.w_pp[h]
            / (r.c1f[k] * muts + r.c2f[k]);
    }
    wphi_diagnose_p(r,q,column,st,bstr,boff,muts);
}
