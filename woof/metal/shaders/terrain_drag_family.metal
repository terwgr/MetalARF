#include <metal_stdlib>
#include "topo_wind_static_bindings.metalh"
#include "terrain_pbl_top_bindings.metalh"
#include "gwdo_column_bindings.metalh"
#include "gwdo_gsl_column_bindings.metalh"

using namespace metal;

// Transcribed from woof/core/kernels/terrain_drag.cu, statement for
// statement, one column per thread.  Metal has no binary64 and no glibc:
// the pinned glibc powf/expf/logf/sinf become precise:: and the
// correctly rounded atan2f becomes precise::atan2, so the port is held to
// the CUDA source's arithmetic, not to its last bit.  The optional
// diagnostic outputs (DTAUX3D, DUSFCG, ...) are not carried.
#pragma clang fp contract(off)

#define TD_KMAX 128
static inline float td_max(float a, float b) { return fmax(a, b); }
static inline float td_min(float a, float b) { return fmin(a, b); }
static inline int td_nint(float x) { return int(round(x)); }
kernel void topo_wind_static(
    constant TopoWindStaticResources &r [[buffer(0)]],
    constant TopoWindStaticParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    device const float *ht = r.ht;
    device const float *var_sso = r.var_sso;
    device const float *var2d = r.var2d;
    device const float *xland = r.xland;
    device float *lap_hgt = r.lap_hgt;
    device float *ctopo = r.ctopo;
    device float *ctopo2 = r.ctopo2;
    const int nx = prm.nx;
    const int ny = prm.ny;
    const int topo_wind = prm.topo_wind;

    int idx = int(tid);
    if (idx >= nx * ny) return;
    int i = idx % nx, j = idx / nx;
    // :1542-1555, the non-periodic arm: neighbours clamp to the domain.
    int im1 = max(i - 1, 0), ip1 = min(i + 1, nx - 1);
    int jm1 = max(j - 1, 0), jp1 = min(j + 1, ny - 1);
    float h = ht[idx];
    // :1558  (ht(ip1)+ht(im1)+ht(jp1)+ht(jm1)-ht*4.)/4.
    float lap = (((ht[j * nx + ip1] + ht[j * nx + im1]) + ht[jp1 * nx + i])
                + ht[jm1 * nx + i]) - h * 4.0f;
    lap = lap / 4.0f;
    lap_hgt[idx] = lap;
    // :1581-1582
    float c = 1.0f, c2 = 1.0f;
    if (topo_wind == 1) {
        // :1587-1592
        if (xland[idx] < 1.5f) c = sqrt(var_sso[idx]);
        if (c <= 2.718f) c = 1.0f;
        else c = precise::log(c);
        // :1594-1608
        if (lap > -10.0f) {
            // ctopo unchanged
        } else if (lap >= -20.0f) {
            float alpha = (lap + 20.0f) / 10.0f;
            c = alpha * c + (1.0f - alpha);
        } else if (lap >= -30.0f) {
            c = (lap + 30.0f) / 10.0f;
            c2 = (lap + 30.0f) / 10.0f;
        } else {
            c = 0.0f;
            c2 = 0.0f;
        }
    } else if (topo_wind == 2) {
        // :1615-1623
        float vfac;
        if (xland[idx] < 1.5f) {
            float x = var2d[idx] * 0.4f / 200.0f + 1.175f;
            vfac = td_min(1.575f, x);
            vfac = vfac * vfac;
        } else {
            vfac = 1.0f;
        }
        c = c * vfac;
    }
    ctopo[idx] = c;
    ctopo2[idx] = c2;
}

kernel void terrain_pbl_top(
    constant TerrainPblTopResources &r [[buffer(0)]],
    constant TerrainPblTopParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    device const float *dz = r.dz;
    device const float *pblh = r.pblh;
    device int *kpbl = r.kpbl;
    const int nz = prm.nz;
    const int ny = prm.ny;
    const int nx = prm.nx;

    ulong col = ulong(tid);
    ulong ncol = (ulong)ny * nx;
    if (col >= ncol) return;
    float depth = 0.0f;
    int top = nz;
    for (int k = 0; k < nz; ++k) {
        float thick = (float)dz[(ulong)k * ncol + col];
        float center = depth + 0.5f * thick;
        depth += thick;
        // Round as the height diagnostic rounds its output.  This keeps
        // its exact first-interior/top-center fallbacks at their level.
        if (k >= 1 && (float)center >= pblh[col]) {
            top = k + 1;
            break;
        }
    }
    kpbl[col] = top;
}

kernel void gwdo_column(
    constant GwdoColumnResources &r [[buffer(0)]],
    constant GwdoColumnParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    device const float *u = r.u;
    device const float *v = r.v;
    device const float *t = r.t;
    device const float *qv = r.qv;
    device const float *p = r.p;
    device const float *p_interface = r.p_interface;
    device const float *pi = r.pi;
    device const float *zi = r.zi;
    device const float *var = r.var;
    device const float *oc1 = r.oc1;
    device const float *oa4 = r.oa4;
    device const float *ol4 = r.ol4;
    device const float *sina = r.sina;
    device const float *cosa = r.cosa;
    device float *du = r.du;
    device float *dv = r.dv;
    const float dxmeter = prm.dxmeter;
    const float deltim = prm.deltim;
    const float g_ = prm.g_;
    const float cp_ = prm.cp_;
    const float rd_ = prm.rd_;
    const float fv_ = prm.fv_;
    const float pi_ = prm.pi_;
    const int nz = prm.nz;
    const int ny = prm.ny;
    const int nx = prm.nx;
    device float *dtaux3d = r.du;
    device float *dtauy3d = r.du;
    device float *dusfcg = r.du;
    device float *dvsfcg = r.du;

    int col = int(tid);
    int st = ny * nx;
    if (col >= st || nz > TD_KMAX) return;
    const float ric = 0.25f, dw2min = 1.0f, rimin = -100.0f, bnv2min = 1.0e-5f;
    const float efmin = 0.0f, efmax = 10.0f, gmax = 1.0f;
    const float veleps = 1.0f, frc = 1.0f, ce = 0.8f, cg = 0.5f;
    const int kpblmin = 2;
    const float frmax = 10.0f, olmin = 1.0e-5f, odmin = 0.1f, odmax = 10.0f;
    const int kte = nz;                    // Fortran kts = 1 .. kte = nz
    const int mdir = 8;
    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    float ca = cosa[col], sa = sina[col];

    // Fortran-indexed accessors (k = 1..nz).
    #define Q(k) ((ulong)((k) - 1) * st + col)
    #define U1(k) (u[Q(k)] * ca - v[Q(k)] * sa)
    #define V1(k) (u[Q(k)] * sa + v[Q(k)] * ca)
    #define VTJ(k) (t[Q(k)] * (1.0f + fv_ * qv[Q(k)]))
    #define VTK(k) (VTJ(k) / pi[Q(k)])
    #define RHO(k) (1.0f / rd_ * p[Q(k)] / VTJ(k))
    #define PRSI(k) (p_interface[(ulong)((k) - 1) * st + col])
    #define PRSL(k) (p[Q(k)])
    #define DEL(k) (PRSI(k) - PRSI((k) + 1))
    #define ZL(k) (0.5f * (zi[Q(k)] + zi[Q((k) + 1)]))

    const float fdir = (float)mdir / (2.0f * pi_);
    float delx = dxmeter, dely = dxmeter;
    float dxy4[4], dxy4p[4];
    dxy4[0] = delx;
    dxy4[1] = dely;
    dxy4[2] = sqrt(precise::pow(delx, 2.0f) + precise::pow(dely, 2.0f));
    dxy4[3] = dxy4[2];
    dxy4p[0] = dxy4[1];
    dxy4p[1] = dxy4[0];
    dxy4p[2] = dxy4[3];
    dxy4p[3] = dxy4[2];
    float cleff = dxmeter;
    float varc = var[col];

    // klowtop: first level whose height above the lowest reaches 2*var.
    float zlowtop = 2.0f * varc;
    int klowtop = 0;
    if (zlowtop > 0.0f) {
        for (int k = 2; k <= kte; ++k) {
            if (ZL(k) - ZL(1) >= zlowtop) { klowtop = k + 1; break; }
        }
    }
    const int kpblmax = kte;
    int kbl = klowtop;
    kbl = max(min(kbl, kpblmax), kpblmin);
    float delks = 1.0f / (PRSI(1) - PRSI(kbl));
    float delks1 = 1.0f / (PRSL(1) - PRSL(kbl));
    float ubar = 0.0f, vbar = 0.0f, rhobar = 0.0f;
    for (int k = 1; k <= kpblmax; ++k) {
        if (k < kbl) {
            float rcsks = DEL(k) * delks;
            float rdelks = DEL(k) * delks;
            ubar = ubar + rcsks * U1(k);
            vbar = vbar + rcsks * V1(k);
            rhobar = rhobar + rdelks * RHO(k);
        }
    }
    // wind direction sector
    float oa, ol, olp, od, dxy, dxyp;
    {
        float oa4c[4], ol4c[4];
        for (int m = 0; m < 4; ++m) {
            oa4c[m] = oa4[(ulong)m * st + col];
            ol4c[m] = ol4[(ulong)m * st + col];
        }
        float wdir = precise::atan2(ubar, vbar) + pi_;
        int idir = td_nint(fdir * wdir) % mdir + 1;
        int nwd = nwdir[idir - 1];
        int m = (nwd - 1) % 4;
        oa = (float)(1 - 2 * ((nwd - 1) / 4)) * oa4c[m];
        ol = ol4c[m];
        float ol4p[4] = {ol4c[1], ol4c[0], ol4c[3], ol4c[2]};
        olp = ol4p[m];
        od = olp / td_max(ol, olmin);
        od = td_min(od, odmax);
        od = td_max(od, odmin);
        dxy = dxy4[m];
        dxyp = dxy4p[m];
    }

    // usqj(k), bnv2(k) for k = 1..kte-1 (bnv2 unclamped in this scheme);
    // bnv2(kte) keeps its initial 0.
    #define TI(k) (2.0f / (t[Q(k)] + t[Q((k) + 1)]))
    #define RDZ(k) (1.0f / (ZL((k) + 1) - ZL(k)))
#define usqj_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        float ti = TI(k); \
        float rdz = RDZ(k); \
        float tem1 = U1(k) - U1(k + 1); \
        float tem2 = V1(k) - V1(k + 1); \
        float dw2 = tem1 * tem1 + tem2 * tem2; \
        float shr2 = td_max(dw2, dw2min) * rdz * rdz; \
        float bvf2 = g_ * (g_ / cp_ + rdz * (VTJ(k + 1) - VTJ(k))) * ti; \
        { td_r = (td_max(bvf2 / shr2, rimin)); break; } \
} while (0); td_r; })
#define bnv2_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        if (k >= kte) { td_r = (0.0f); break; } \
        float rdz = RDZ(k); \
        { td_r = (2.0f * g_ * rdz * (VTK(k + 1) - VTK(k)) \
               / (VTK(k + 1) + VTK(k))); break; } \
} while (0); td_r; })

    float ulow = td_max(sqrt(ubar * ubar + vbar * vbar), 1.0f);
    float rulow = 1.0f / ulow;
#define velco_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        float vc = 0.5f * ((U1(k) + U1(k + 1)) * ubar \
                          + (V1(k) + V1(k + 1)) * vbar); \
        vc = vc * rulow; \
        if (vc < veleps && vc > 0.0f) vc = veleps; \
        { td_r = (vc); break; } \
} while (0); td_r; })

    bool ldrag = velco_at(1) <= 0.0f;
    for (int k = kpblmin; k <= kpblmax; ++k)
        if (k < kbl) ldrag = ldrag || velco_at(k) <= 0.0f;
    float wtkbj = (PRSL(1) - PRSL(2)) * delks1;
    float bnv2_1 = wtkbj * bnv2_at(1);
    for (int k = kpblmin; k <= kpblmax; ++k) {
        if (k < kbl) {
            float rdelks = (PRSL(k) - PRSL(k + 1)) * delks1;
            bnv2_1 = bnv2_1 + bnv2_at(k) * rdelks;
        }
    }
    ldrag = ldrag || bnv2_1 <= 0.0f;
    ldrag = ldrag || ulow == 1.0f;
    ldrag = ldrag || varc <= 0.0f;

    float bnv = 0.0f, fr = 0.0f, xn = 0.0f, yn = 0.0f;
    if (!ldrag) {
        bnv = sqrt(bnv2_1);
        fr = bnv * rulow * varc * od;
        fr = td_min(fr, frmax);
        xn = ubar * rulow;
        yn = vbar * rulow;
    }
    float taub, coefm = 0.0f;
    if (!ldrag) {
        float efact = precise::pow(oa + 2.0f, ce * fr / frc);
        efact = td_min(td_max(efact, efmin), efmax);
        coefm = precise::pow(1.0f + ol, oa + 1.0f);
        float xlinv = coefm / cleff;
        float tem = fr * fr * oc1[col];
        float gfobnv = gmax * tem / ((tem + cg) * bnv);
        taub = xlinv * rhobar * ulow * ulow * ulow * gfobnv * efact;
    } else {
        taub = 0.0f;
        xn = 0.0f;
        yn = 0.0f;
    }

    // Vertical structure of the stress.
    float taup[TD_KMAX + 1];
    for (int k = 1; k <= kte + 1; ++k) taup[k - 1] = 0.0f;
    for (int k = 1; k <= kpblmax; ++k)
        if (k <= kbl) taup[k - 1] = taub;
    bool icrilv = false;
    float brvf = 0.0f;
    for (int k = kpblmin; k <= kte - 1; ++k) {
        int kp1 = k + 1;
        if (k >= kbl) {
            float usqj = usqj_at(k);
            icrilv = icrilv || (usqj < ric) || (velco_at(k) <= 0.0f);
            brvf = td_max(bnv2_at(k), bnv2min);
            brvf = sqrt(brvf);
        }
        if (k >= kbl && !ldrag) {
            if (!icrilv && taup[k - 1] > 0.0f) {
                float velco = velco_at(k);
                float usqj = usqj_at(k);
                float temv = 1.0f / velco;
                float tem1 = coefm / dxy * (RHO(kp1) + RHO(k)) * brvf * velco
                            * 0.5f;
                float hd = sqrt(taup[k - 1] / tem1);
                float fro = brvf * hd * temv;
                float tem2 = sqrt(usqj);
                float tem = 1.0f + tem2 * fro;
                float rim = usqj * (1.0f - fro) / (tem * tem);
                if (rim <= ric) {
                    if ((oa <= 0.0f) || (kp1 >= kpblmin)) {
                        float temc = 2.0f + 1.0f / tem2;
                        hd = velco * (2.0f * sqrt(temc) - temc) / brvf;
                        taup[kp1 - 1] = tem1 * hd * hd;
                    }
                } else {
                    taup[kp1 - 1] = taup[k - 1];
                }
            }
        }
    }
    // Flow blocking, summed into the stress.
    if (!ldrag) {
        int kblk = 0;
        float fbdpe = 0.0f, fbdke = 0.0f, zblk = 0.0f;
        for (int k = kte; k >= kpblmin; --k) {
            if (kblk == 0 && k <= kbl) {
                fbdpe = fbdpe + bnv2_at(k) * (ZL(kbl) - ZL(k)) * DEL(k) / g_
                        / RHO(k);
                fbdke = 0.5f * (precise::pow(U1(k), 2.0f) + precise::pow(V1(k), 2.0f));
                if (fbdpe >= fbdke) {
                    kblk = k;
                    kblk = min(kblk, kbl);
                    zblk = ZL(kblk) - ZL(1);
                }
            }
        }
        if (kblk != 0) {
            float fbdcd = td_max(2.0f - 1.0f / od, 0.0f);
            float taufb = 0.5f * rhobar * coefm / (dxmeter * dxmeter) * fbdcd
                         * dxyp * olp * zblk * (ulow * ulow);
            float tautem = taufb / (float)(kblk - 1);
            taup[0] = taup[0] + taufb;
            for (int k = 2; k <= kblk; ++k) {
                taufb = taufb - tautem;
                taup[k - 1] = taup[k - 1] + taufb;
            }
            // taufb is 0 above kblk: taup there gains + 0.
            for (int k = kblk + 1; k <= kte + 1; ++k)
                taup[k - 1] = taup[k - 1] + 0.0f;
        }
    }
    // -(g) d(tau)/dp and the critical-line limiter.
#define taud_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        { td_r = (1.0f * (taup[k] - taup[k - 1]) * g_ / DEL(k)); break; } \
} while (0); td_r; })
    float dtfac = 1.0f;
    for (int k = 1; k <= kpblmax - 1; ++k) {
        if (k <= kbl) {
            float td = taud_at(k);
            if (td != 0.0f)
                dtfac = td_min(dtfac, fabs(velco_at(k) / (deltim * td)));
        }
    }
    float dusfc = 0.0f, dvsfc = 0.0f;
    for (int k = 1; k <= kte; ++k) {
        float td = taud_at(k) * dtfac;
        float dtaux = td * xn;
        float dtauy = td * yn;
        dusfc = dusfc + dtaux * DEL(k);
        dvsfc = dvsfc + dtauy * DEL(k);
        ulong q = Q(k);
        // module_bl_gwdo.F hands bl_gwdo_run rublten and copies it back;
        // bl_gwdo.F90 rotates the drag to the model grid and adds it.
        du[q] = du[q] + dtaux * ca + dtauy * sa;
        dv[q] = dv[q] - dtaux * sa + dtauy * ca;
        if (false) {
            dtaux3d[q] = dtaux * ca + dtauy * sa;
            dtauy3d[q] = -dtaux * sa + dtauy * ca;
        }
    }
    dusfc = (-1.0f / g_) * dusfc;
    dvsfc = (-1.0f / g_) * dvsfc;
    if (false) {
        dusfcg[col] = dusfc * ca + dvsfc * sa;
        dvsfcg[col] = -dusfc * sa + dvsfc * ca;
    }
    #undef usqj_at
    #undef bnv2_at
    #undef velco_at
    #undef taud_at
    #undef Q
    #undef U1
    #undef V1
    #undef VTJ
    #undef VTK
    #undef RHO
    #undef PRSI
    #undef PRSL
    #undef DEL
    #undef ZL
    #undef TI
    #undef RDZ
}

kernel void gwdo_gsl_column(
    constant GwdoGslColumnResources &r [[buffer(0)]],
    constant GwdoGslColumnParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    device const float *u = r.u;
    device const float *v = r.v;
    device const float *t = r.t;
    device const float *qv = r.qv;
    device const float *p = r.p;
    device const float *p_interface = r.p_interface;
    device const float *pi = r.pi;
    device const float *zi = r.zi;
    device const float *dz = r.dz;
    device const float *var = r.var;
    device const float *oc1 = r.oc1;
    device const float *oa4 = r.oa4;
    device const float *ol4 = r.ol4;
    device const float *varss = r.varss;
    device const float *oc1ss = r.oc1ss;
    device const float *oa4ss = r.oa4ss;
    device const float *ol4ss = r.ol4ss;
    device const float *sina = r.sina;
    device const float *cosa = r.cosa;
    device const float *xland = r.xland;
    device const float *br = r.br;
    device const float *hpbl = r.hpbl;
    device const int *kpbl = r.kpbl;
    device float *du = r.du;
    device float *dv = r.dv;
    const float dxmeter = prm.dxmeter;
    const float deltim = prm.deltim;
    const float g = prm.g;
    const float cp = prm.cp;
    const float rd = prm.rd;
    const float fv = prm.fv;
    const float pi_c = prm.pi_c;
    const int kpblmax = prm.kpblmax;
    const int nz = prm.nz;
    const int ny = prm.ny;
    const int nx = prm.nx;
    device float *dtau = r.du;
    device float *dsfc = r.du;

    int col = int(tid);
    int st = ny * nx;
    if (col >= st || nz > TD_KMAX) return;
    const float dxmin_ss = 1000.0f, dxmax_ss = 12000.0f;
    const float dxmin_ls = 3000.0f, dxmax_ls = 13000.0f;
    const float ric = 0.25f, dw2min = 1.0f, rimin = -100.0f, bnv2min = 1.0e-5f;
    const float efmin = 0.0f, efmax = 10.0f, xl = 4.0e4f, gmax = 1.0f;
    const float veleps = 1.0f, factop = 0.5f, frc = 1.0f, ce = 0.8f, cg = 0.5f;
    const int kpblmin = 2;
    const float varmax_ss = 35.0f, varmax_fd = 160.0f, beta_ss = 0.1f;
    const float beta_fd = 0.2f;
    const float frmax = 10.0f, olmin = 1.0e-5f, odmin = 0.1f, odmax = 10.0f;
    const float clf_coeff = 3.4E+07f, clf_coeff_ss = 0.1f;
    const float a1_coeff = 0.00026615161f, a2_coeff = 0.005363f;
    const float TOFD_coeff = 0.0759f, Hefold_nom = 1500.0f;
    const int kte = nz, kts = 1;
    const int mdir = 8;
    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    const float rcl = 1.0f;
    const float rcs = sqrt(rcl);
    const float cs = 1.0f / sqrt(rcl);
    const float csg = cs * g;
    const int lcap = kte;
    const float fdir = (float)mdir / (2.0f * pi_c);
    float ca = cosa[col], sa = sina[col];

    #define Q(k) ((ulong)((k) - 1) * st + col)
    // gwdo_gsl rotates the winds to zonal/meridional before gwdo2d.
    #define U1(k) (u[Q(k)] * ca - v[Q(k)] * sa)
    #define V1(k) (u[Q(k)] * sa + v[Q(k)] * ca)
    #define VTJ(k) (t[Q(k)] * (1.0f + fv * qv[Q(k)]))
    #define VTK(k) (VTJ(k) / pi[Q(k)])
    #define RO(k) (1.0f / rd * p[Q(k)] / VTJ(k))
    #define PRSI(k) (p_interface[(ulong)((k) - 1) * st + col])
    #define PRSL(k) (p[Q(k)])
    #define DEL(k) (PRSI(k) - PRSI((k) + 1))
    #define ZL(k) (0.5f * (zi[Q(k)] + zi[Q((k) + 1)]))

    // Scale-aware tapers.
    float ls_taper, ss_taper;
    if (dxmeter >= dxmax_ls) {
        ls_taper = 1.0f;
    } else if (dxmeter <= dxmin_ls) {
        ls_taper = 0.0f;
    } else {
        ls_taper = 0.5f * (precise::sin(pi_c * (dxmeter - 0.5f
                                               * (dxmax_ls + dxmin_ls))
                                      / (dxmax_ls - dxmin_ls)) + 1.0f);
    }
    if (dxmeter >= dxmax_ss) {
        ss_taper = 1.0f;
    } else if (dxmeter <= dxmin_ss) {
        ss_taper = 0.0f;
    } else {
        ss_taper = dxmax_ss * (1.0f - dxmin_ss / dxmeter)
                   / (dxmax_ss - dxmin_ss);
    }
    float delx = dxmeter, dely = dxmeter;
    float dxy4[4], dxy4p[4];
    dxy4[0] = delx;
    dxy4[1] = dely;
    dxy4[2] = sqrt(delx * delx + dely * dely);
    dxy4[3] = dxy4[2];
    dxy4p[0] = dxy4[1];
    dxy4p[1] = dxy4[0];
    dxy4p[2] = dxy4[3];
    dxy4p[3] = dxy4[2];

    const float rstoch = 0.0f;
    float varc = var[col], varssc = varss[col];
    float var_stoch = varc + varc * 0.666f * rstoch;
    float varss_stoch = varssc + varssc * 0.666f * rstoch;
    float varmax_ss_stoch = varmax_ss + varmax_ss * 0.5f * rstoch;
    float varmax_fd_stoch = varmax_fd + varmax_fd * 0.5f * rstoch;

    // Reference level: the higher of 2*var and the PBL top.
    float zlowtop = 2.0f * var_stoch;
    int klowtop = 0;
    for (int k = kts + 1; k <= kte; ++k) {
        if (ZL(k) - ZL(1) >= zlowtop) { klowtop = k + 1; break; }
    }
    int kpblc = kpbl[col];
    int kbl = max(kpblc, klowtop);
    kbl = max(min(kbl, kpblmax), kpblmin);
    int komax = klowtop - 1;
    float delks = 1.0f / (PRSI(1) - PRSI(kbl));
    float delks1 = 1.0f / (PRSL(1) - PRSL(kbl));
    float ubar = 0.0f, vbar = 0.0f, roll = 0.0f;
    for (int k = kts; k <= kpblmax; ++k) {
        if (k < kbl) {
            float rcsks = rcs * DEL(k) * delks;
            float rdelks = DEL(k) * delks;
            ubar = ubar + rcsks * U1(k);
            vbar = vbar + rcsks * V1(k);
            roll = roll + rdelks * RO(k);
        }
    }
    float oa, ol, oass, olss, olp, od, dxy, dxyp;
    {
        float oa4c[4], ol4c[4], oa4ssc[4], ol4ssc[4];
        for (int m = 0; m < 4; ++m) {
            oa4c[m] = oa4[(ulong)m * st + col];
            ol4c[m] = ol4[(ulong)m * st + col];
            oa4ssc[m] = oa4ss[(ulong)m * st + col];
            ol4ssc[m] = ol4ss[(ulong)m * st + col];
        }
        float wdir = precise::atan2(ubar, vbar) + pi_c;
        int idir = td_nint(fdir * wdir) % mdir + 1;
        int nwd = nwdir[idir - 1];
        int m = (nwd - 1) % 4;
        float sgn = (float)(1 - 2 * ((nwd - 1) / 4));
        oa = sgn * oa4c[m];
        ol = ol4c[m];
        oass = sgn * oa4ssc[m];
        olss = ol4ssc[m];
        float ol4p[4] = {ol4c[1], ol4c[0], ol4c[3], ol4c[2]};
        olp = ol4p[m];
        od = olp / td_max(ol, olmin);
        od = td_min(od, odmax);
        od = td_max(od, odmin);
        dxy = dxy4[m];
        dxyp = dxy4p[m];
    }

#define bnv2_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        if (k >= kte) { td_r = (0.0f); break; } \
        float rdz = 1.0f / (ZL(k + 1) - ZL(k)); \
        float b = 2.0f * g * rdz * (VTK(k + 1) - VTK(k)) \
                 / (VTK(k + 1) + VTK(k)); \
        { td_r = (td_max(b, bnv2min)); break; } \
} while (0); td_r; })
#define usqj_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        float ti = 2.0f / (t[Q(k)] + t[Q(k + 1)]); \
        float rdz = 1.0f / (ZL(k + 1) - ZL(k)); \
        float tem1 = U1(k) - U1(k + 1); \
        float tem2 = V1(k) - V1(k + 1); \
        float dw2 = rcl * (tem1 * tem1 + tem2 * tem2); \
        float shr2 = td_max(dw2, dw2min) * rdz * rdz; \
        float bvf2 = g * (g / cp + rdz * (VTJ(k + 1) - VTJ(k))) * ti; \
        { td_r = (td_max(bvf2 / shr2, rimin)); break; } \
} while (0); td_r; })

    const bool ls_on = ls_taper > 1.0e-02f;
    const bool ss_on = ss_taper > 1.0e-02f;
    bool ldrag = false;
    float ulow = 0.0f, rulow = 0.0f, xn = 0.0f, yn = 0.0f;
    float taub = 0.0f, coefm = 0.0f, bnv = 0.0f, fr = 0.0f;
#define velco_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        float vc = (0.5f * rcs) * ((U1(k) + U1(k + 1)) * ubar \
                                  + (V1(k) + V1(k + 1)) * vbar); \
        vc = vc * rulow; \
        if (vc < veleps && vc > 0.0f) vc = veleps; \
        { td_r = (vc); break; } \
} while (0); td_r; })
    if (ls_on) {
        ulow = td_max(sqrt(ubar * ubar + vbar * vbar), 1.0f);
        rulow = 1.0f / ulow;
        ldrag = velco_at(1) <= 0.0f;
        for (int k = kpblmin; k <= kpblmax; ++k)
            if (k < kbl) ldrag = ldrag || velco_at(k) <= 0.0f;
        for (int k = kts; k <= kpblmax; ++k)
            if (k < kbl) ldrag = ldrag || bnv2_at(k) < 0.0f;
        float wtkbj = (PRSL(1) - PRSL(2)) * delks1;
        float bnv2_1 = wtkbj * bnv2_at(1);
        for (int k = kpblmin; k <= kpblmax; ++k) {
            if (k < kbl) {
                float rdelks = (PRSL(k) - PRSL(k + 1)) * delks1;
                bnv2_1 = bnv2_1 + bnv2_at(k) * rdelks;
            }
        }
        ldrag = ldrag || bnv2_1 <= 0.0f;
        ldrag = ldrag || ulow == 1.0f;
        ldrag = ldrag || var_stoch <= 0.0f;
        if (!ldrag) {
            bnv = sqrt(bnv2_1);
            fr = bnv * rulow * 2.0f * var_stoch * od;
            fr = td_min(fr, frmax);
            xn = ubar * rulow;
            yn = vbar * rulow;
        }
        if (!ldrag) {
            float efact = precise::pow(oa + 2.0f, ce * fr / frc);
            efact = td_min(td_max(efact, efmin), efmax);
            float cleff = clf_coeff / sqrt(dxmeter);
            coefm = precise::pow(1.0f + ol, oa + 1.0f);
            float xlinv = coefm / cleff;
            float tem = fr * fr * oc1[col];
            float gfobnv = gmax * tem / ((tem + cg) * bnv);
            taub = xlinv * roll * ulow * ulow * ulow * gfobnv * efact;
        } else {
            taub = 0.0f;
            xn = 0.0f;
            yn = 0.0f;
        }
    }

    // Small-scale gravity-wave drag (stable boundary layer).
    float tauwavex0 = 0.0f, tauwavey0 = 0.0f, hpbl2 = 0.0f;
    bool wave = false;
    int kpblw = kpblc;
    float xlandc = xland[col];
    if (ss_on) {
        float hpblc = hpbl[col];
        hpbl2 = hpblc + 10.0f;
        int kpbl2 = kpblc;
        const int kvar = 1;
        // za(k) from the cumulative layer depths, in WRF's order.
        float zq = 0.0f;
        float za_k = 0.0f;
        {
            float zqk = 0.0f;
            int kend = max(kpblc, kts + 1);
            for (int k = kts; k <= kend; ++k) {
                float zqn = dz[Q(k)] + zqk;
                za_k = 0.5f * (zqk + zqn);
                if (k >= kts + 1 && za_k > 300.0f) {
                    kpbl2 = k;
                    if (k == kpblc) hpbl2 = hpblc + 10.0f;
                    else hpbl2 = za_k + 10.0f;
                    break;
                }
                zqk = zqn;
            }
        }
        (void)zq;
        if ((xlandc - 1.5f) <= 0.0f && 2.0f * varss_stoch <= hpblc) {
            float thv1 = t[Q(1)] / pi[Q(1)] * (1.0f + fv * qv[Q(1)]);
            float thvk = t[Q(kpbl2)] / pi[Q(kpbl2)] * (1.0f + fv * qv[Q(kpbl2)]);
            if (br[col] > 0.0f && thvk - thv1 > 0.0f) {
                float cleff_ss = sqrt(dxy * dxy + dxyp * dxyp);
                cleff_ss = clf_coeff_ss * td_max(dxmax_ss, cleff_ss);
                float coefm_ss = precise::pow(1.0f + olss, oass + 1.0f);
                float xlinv = coefm_ss / cleff_ss;
                float govrth = g / (0.5f * (thvk + thv1));
                float xnbv = sqrt(govrth * (thvk - thv1) / hpbl2);
                float var_temp = td_min(varss_stoch, varmax_ss_stoch)
                    + td_max(0.0f, beta_ss * (varss_stoch - varmax_ss_stoch));
                float tv2 = 2.0f * var_temp;
                if (fabs(xnbv / U1(kpbl2)) > xlinv) {
                    tauwavex0 = 0.5f * xnbv * xlinv * (tv2 * tv2) * RO(kvar)
                                * U1(kvar);
                    tauwavex0 = tauwavex0 * ss_taper;
                } else {
                    tauwavex0 = 0.0f;
                }
                if (fabs(xnbv / V1(kpbl2)) > xlinv) {
                    tauwavey0 = 0.5f * xnbv * xlinv * (tv2 * tv2) * RO(kvar)
                                * V1(kvar);
                    tauwavey0 = tauwavey0 * ss_taper;
                } else {
                    tauwavey0 = 0.0f;
                }
                wave = true;
            }
        }
    }
    // Form drag constants.
    float a2 = 0.0f, H_efold = 0.0f;
    bool form = ss_on && (xlandc - 1.5f) <= 0.0f;
    if (form) {
        float var_temp = td_min(varss_stoch, varmax_fd_stoch)
            + td_max(0.0f, beta_fd * (varss_stoch - varmax_fd_stoch));
        float a1 = a1_coeff * (var_temp * var_temp);
        a2 = a1 * a2_coeff;
        H_efold = td_max(2.0f * varss_stoch, hpbl[col]);
        H_efold = td_min(H_efold, Hefold_nom);
    }

    // Large-scale stress profile.
    float taup[TD_KMAX + 1], taufb[TD_KMAX + 1];
    for (int k = 1; k <= kte + 1; ++k) { taup[k - 1] = 0.0f; taufb[k - 1] = 0.0f; }
    if (ls_on) {
        for (int k = kts; k <= kpblmax; ++k)
            if (k <= kbl) taup[k - 1] = taub;
        bool icrilv = false;
        float brvf = 0.0f;
        for (int k = kpblmin; k <= kte - 1; ++k) {
            int kp1 = k + 1;
            if (k >= kbl) {
                icrilv = icrilv || (usqj_at(k) < ric) || (velco_at(k) <= 0.0f);
                brvf = td_max(bnv2_at(k), bnv2min);
                brvf = sqrt(brvf);
            }
            if (k >= kbl && !ldrag) {
                if (!icrilv && taup[k - 1] > 0.0f) {
                    float velco = velco_at(k);
                    float usqj = usqj_at(k);
                    float temv = 1.0f / velco;
                    float tem1 = coefm / dxy * (RO(kp1) + RO(k)) * brvf * velco
                                * 0.5f;
                    float hd = sqrt(taup[k - 1] / tem1);
                    float fro = brvf * hd * temv;
                    float tem2 = sqrt(usqj);
                    float tem = 1.0f + tem2 * fro;
                    float rim = usqj * (1.0f - fro) / (tem * tem);
                    if (rim <= ric) {
                        if ((oa <= 0.0f) || (kp1 >= kpblmin)) {
                            float temc = 2.0f + 1.0f / tem2;
                            hd = velco * (2.0f * sqrt(temc) - temc) / brvf;
                            taup[kp1 - 1] = tem1 * hd * hd;
                        }
                    } else {
                        taup[kp1 - 1] = taup[k - 1];
                    }
                }
            }
        }
        // Flow blocking (kept apart from taup, as WRF keeps it).
        if (!ldrag) {
            int kblk = 0;
            float pe = 0.0f, zblk = 0.0f;
            for (int k = kte; k >= kpblmin; --k) {
                if (kblk == 0 && k <= komax) {
                    pe = pe + bnv2_at(k) * (ZL(komax) - ZL(k)) * DEL(k) / g
                         / RO(k);
                    float ke = 0.5f * (precise::pow(rcs * U1(k), 2.0f)
                                      + precise::pow(rcs * V1(k), 2.0f));
                    if (pe >= ke) {
                        kblk = k;
                        kblk = min(kblk, kbl);
                        zblk = ZL(kblk) - ZL(kts);
                    }
                }
            }
            if (kblk != 0) {
                float cd = td_max(2.0f - 1.0f / od, 0.0f);
                float mx = td_max(dxmax_ls, dxy);
                taufb[0] = 0.5f * roll * coefm / (mx * mx) * cd * dxyp * olp
                           * zblk * (ulow * ulow);
                float tautem = taufb[0] / (float)(kblk - kts);
                for (int k = kts + 1; k <= kblk; ++k)
                    taufb[k - 1] = taufb[k - 2] - tautem;
            }
        }
    }
    // Deceleration terms and the critical-line limiter (ls_on only).
#define taud_ls_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        float td = 1.0f * (taup[k] - taup[k - 1]) * csg / DEL(k); \
        if (k >= lcap) td = td * factop; \
        { td_r = (td); break; } \
} while (0); td_r; })
#define taud_bl_at(K_) ({ int td_k = (K_); float td_r = 0.0f; do { int k = td_k; \
        float td = 1.0f * (taufb[k] - taufb[k - 1]) * csg / DEL(k); \
        if (k >= lcap) td = td * factop; \
        { td_r = (td); break; } \
} while (0); td_r; })
    float dtfac = 1.0f;
    if (ls_on) {
        for (int k = kts; k <= kpblmax - 1; ++k) {
            if (k <= kbl) {
                float s = taud_ls_at(k) + taud_bl_at(k);
                if (s != 0.0f)
                    dtfac = td_min(dtfac, fabs(velco_at(k)
                                                / (deltim * rcs * s)));
            }
        }
    }

    // Per level: the four components in WRF's order of accumulation into
    // dudt (ss, then fd, then ls + bl), rotated and added to the PBL
    // tendencies as gwdo_gsl does.
    float dus[4] = {0.0f, 0.0f, 0.0f, 0.0f}, dvs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float zqk = 0.0f;
    for (int k = kts; k <= kte; ++k) {
        ulong q = Q(k);
        float del = DEL(k);
        float zqn = dz[q] + zqk;
        float za = 0.5f * (zqk + zqn);
        zqk = zqn;
        float dudt = 0.0f, dvdt = 0.0f;
        float ut_ss = 0.0f, vt_ss = 0.0f, ut_fd = 0.0f, vt_fd = 0.0f;
        float ut_ls = 0.0f, vt_ls = 0.0f, ut_bl = 0.0f, vt_bl = 0.0f;
        if (ss_on) {
            if (wave && k <= kpblw) {
                float shape = td_max((1.0f - za / hpbl2), 0.0f);
                ut_ss = -1.0f * tauwavex0 * 2.0f * shape / hpbl2;
                vt_ss = -1.0f * tauwavey0 * 2.0f * shape / hpbl2;
            }
            dudt = dudt + ut_ss;
            dvdt = dvdt + vt_ss;
            dus[2] = dus[2] + ut_ss * del;
            dvs[2] = dvs[2] + vt_ss * del;
            if (form) {
                float u1 = U1(k), v1 = V1(k);
                float wsp = sqrt(u1 * u1 + v1 * v1);
                float ef = precise::exp(-precise::pow(za / H_efold, 1.5f));
                float zp = precise::pow(za, -1.2f);
                ut_fd = -(TOFD_coeff * wsp * u1 * ef * a2 * zp * ss_taper);
                vt_fd = -(TOFD_coeff * wsp * v1 * ef * a2 * zp * ss_taper);
            }
            dudt = dudt + ut_fd;
            dvdt = dvdt + vt_fd;
            dus[3] = dus[3] + ut_fd * del;
            dvs[3] = dvs[3] + vt_fd * del;
        }
        if (ls_on) {
            float tls = taud_ls_at(k) * dtfac * ls_taper;
            float tbl = taud_bl_at(k) * dtfac * ls_taper;
            ut_ls = tls * xn;
            vt_ls = tls * yn;
            ut_bl = tbl * xn;
            vt_bl = tbl * yn;
            dudt = ut_ls + ut_bl + dudt;
            dvdt = vt_ls + vt_bl + dvdt;
            dus[0] = dus[0] + ut_ls * del;
            dvs[0] = dvs[0] + vt_ls * del;
            dus[1] = dus[1] + ut_bl * del;
            dvs[1] = dvs[1] + vt_bl * del;
        }
        du[q] = du[q] + dudt * ca + dvdt * sa;
        dv[q] = dv[q] - dudt * sa + dvdt * ca;
        if (false) {
            ulong n3 = (ulong)nz * st;
            float cx[4] = {ut_ls, ut_bl, ut_ss, ut_fd};
            float cy[4] = {vt_ls, vt_bl, vt_ss, vt_fd};
            for (int m = 0; m < 4; ++m) {
                dtau[(ulong)(2 * m) * n3 + q] = cx[m] * ca + cy[m] * sa;
                dtau[(ulong)(2 * m + 1) * n3 + q] = -cx[m] * sa + cy[m] * ca;
            }
        }
    }
    if (false) {
        for (int m = 0; m < 4; ++m) {
            float us = (-1.0f / g * rcs) * dus[m];
            float vs = (-1.0f / g * rcs) * dvs[m];
            dsfc[(ulong)(2 * m) * st + col] = us * ca + vs * sa;
            dsfc[(ulong)(2 * m + 1) * st + col] = -us * sa + vs * ca;
        }
    }
    #undef bnv2_at
    #undef usqj_at
    #undef velco_at
    #undef taud_ls_at
    #undef taud_bl_at
    #undef Q
    #undef U1
    #undef V1
    #undef VTJ
    #undef VTK
    #undef RO
    #undef PRSI
    #undef PRSL
    #undef DEL
    #undef ZL
}

