#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "milbrandt2_prelim_bindings.metalh"
#include "milbrandt2_geometry_bindings.metalh"
#include "milbrandt2_diagnostics_bindings.metalh"
#include "milbrandt2_cold_bindings.metalh"
#include "milbrandt2_warm_bindings.metalh"
#include "milbrandt2_sediment_64_bindings.metalh"
#include "milbrandt2_sediment_256_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

static inline float my2_polysvp(float t, int wtype) {
    float dt = t - 273.16f;
    if (dt < -80.0f) dt = -80.0f;
    float p;
    if (wtype == 1) {
        p = 6.11147274f + dt * (0.503160820f + dt * (0.188439774e-1f
            + dt * (0.420895665e-3f + dt * (0.615021634e-5f
            + dt * (0.602588177e-7f + dt * (0.385852041e-9f
            + dt * (0.146898966e-11f + 0.252751365e-14f * dt)))))));
    } else {
        p = 6.11239921f + dt * (0.443987641f + dt * (0.142986287e-1f
            + dt * (0.264847430e-3f + dt * (0.302950461e-5f
            + dt * (0.206739458e-7f + dt * (0.640689451e-10f
            + dt * (-0.952447341e-13f + -0.976195544e-15f * dt)))))));
    }
    return p * 100.0f;
}

static inline float my2_qsat(float temp, float pres, int wtype) {
    const float e = my2_polysvp(temp, wtype);
    return 0.622f * e / (pres - e);
}

static inline float my2_N_Cooper(float T) {
    return 5.0f * exp(0.304f * (273.16f - max(233.0f, T)));
}

static inline float my2_Nos_Thompson(float T) {
    return min(2.0e+8f,
               2.0e+6f * exp(-0.12f * min(-0.001f, T - 273.16f)));
}

static inline float my2_NccnFNC(float Win, float Tin, float Pin) {
    const float x = log10(Win * 100.0f);
    const float x2 = x * x, x3 = x2 * x, x4 = x2 * x2;
    const float T = Tin - 273.15f;
    const float T2 = T * T, T3 = T2 * T, T4 = T2 * T2;
    const float p = Pin * 0.01f, p2 = p * p;
    const float a = 0.0f, b = 0.0f;
    const float c = -2.112e-9f * T4 + 3.9836e-8f * T3 + 2.3703e-6f * T2
                    - 1.4542e-4f * T - 0.0698f;
    const float d = -4.210e-8f * T4 + 5.5745e-7f * T3 + 1.8460e-5f * T2
                    + 9.6078e-4f * T + 0.7120f;
    const float e = 1.434e-7f * T4 - 1.6455e-6f * T3 - 4.3334e-5f * T2
                    - 7.6720e-3f * T + 1.0056f;
    const float f = 1.340e-6f * p2 - 3.5114e-3f * p + 1.9453f;
    const float g = 4.226e-3f * x4 - 5.6012e-3f * x3 - 8.7846e-2f * x2
                    + 2.7435e-2f * x + 0.9932f;
    const float h = 5.811e-9f * T4 + 1.5589e-7f * T3 - 3.8623e-5f * T2
                    + 1.4471e-3f * T + 0.1496f;
    const float y = a * x4 + b * x3 + c * x2 + d * x + e + (f * g * h);
    return pow(10.0f, max(0.0f, y)) * 1.0e6f;
}

kernel void milbrandt2_prelim(
    constant Milbrandt2PrelimResources &r [[buffer(0)]],
    constant Milbrandt2PrelimParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.nz * p.ny * p.nx;
    if (tid >= uint(n)) return;
    const int gid = int(tid);
    const int col = gid % (p.ny * p.nx);
    const float ps = r.psfc[col];
    const float sigma = r.p_eos[gid] / ps;
    const float pr = ps * sigma;
    r.pres[gid] = pr;
    float t = r.T[gid];
    const float qsw = my2_qsat(t, pr, 0);
    const float qsi = my2_qsat(t, pr, 1);
    r.QSW[gid] = qsw;
    r.QSI[gid] = qsi;
    const float de = pr / (287.05f * t);
    const float ide = 1.0f / de;
    r.DE[gid] = de;
    r.iDE[gid] = ide;
    float nc = r.NC[gid] * de;
    float nr = r.NR[gid] * de;
    float ny_ = r.NY[gid] * de;
    float nn = r.NN[gid] * de;
    float ng = r.NG[gid] * de;
    float nh = r.NH[gid] * de;
    float q = r.Q[gid];
    float qc = r.QC[gid], qr = r.QR[gid], qi = r.QI[gid];
    float qn = r.QN[gid], qg = r.QG[gid], qh = r.QH[gid];
    const float tmp1 = qsw / max(q, 1.0e-20f);
    const float tmp2 = qsi / max(q, 1.0e-20f);
    float tmp3;
    if (qc > 1.0e-14f && nc < 1.0e-3f) {
        nc = r.ck[50];
    } else if (qc <= 1.0e-14f || (qc < 1.0e-6f && tmp1 < 0.90f)) {
        tmp3 = max(0.0f, qc); q = q + tmp3; t = t - r.ck[4] * tmp3;
        qc = 0.0f; nc = 0.0f;
    }
    if (qr > 1.0e-14f && nr < 1.0e-3f) {
        nr = pow(1.0e+7f * r.ck[54], 3.0f / (4.0f + 0.0f))
             * pow(r.ck[54] * r.ck[59] * de * qr * r.ck[16],
                   (1.0f + 0.0f) / (4.0f + 0.0f));
    } else if (qr <= 1.0e-14f || (qr < 1.0e-6f && tmp1 < 0.90f)) {
        tmp3 = max(0.0f, qr); q = q + tmp3; t = t - r.ck[4] * tmp3;
        qr = 0.0f; nr = 0.0f;
    }
    if (qi > 1.0e-14f && ny_ < 1.0e-3f) {
        ny_ = max(2.0f * 1.0e-3f, my2_N_Cooper(t));
    } else if (qi <= 1.0e-14f || (qi < 1.0e-6f && tmp2 < 0.80f)) {
        tmp3 = max(0.0f, qi); q = q + tmp3; t = t - 0.334e+6f * tmp3;
        qi = 0.0f; ny_ = 0.0f;
    }
    if (qn > 1.0e-14f && nn < 1.0e-3f) {
        const float No_s = my2_Nos_Thompson(t);
        nn = pow(No_s * r.ck[98], r.ck[25] * r.ck[93])
             * pow(r.ck[98] * r.ck[107] * r.ck[26] * de * qn,
                   (1.0f + 0.0f) * r.ck[93]);
    } else if (qn <= 1.0e-14f || (qn < 1.0e-6f && tmp2 < 0.80f)) {
        tmp3 = max(0.0f, qn); q = q + tmp3; t = t - 0.334e+6f * tmp3;
        qn = 0.0f; nn = 0.0f;
    }
    if (qg > 1.0e-14f && ng < 1.0e-3f) {
        ng = pow(4.0e+6f * r.ck[118], 3.0f / (4.0f + 0.0f))
             * pow(r.ck[118] * r.ck[123] * de * qg * r.ck[20],
                   (1.0f + 0.0f) / (4.0f + 0.0f));
    } else if (qg <= 1.0e-14f || (qg < 1.0e-6f && tmp2 < 0.80f)) {
        tmp3 = max(0.0f, qg); q = q + tmp3; t = t - 0.334e+6f * tmp3;
        qg = 0.0f; ng = 0.0f;
    }
    if (qh > 1.0e-14f && nh < 1.0e-3f) {
        nh = pow(1.0e+5f * r.ck[136], 3.0f / (4.0f + 0.0f))
             * pow(r.ck[136] * r.ck[140] * de * qh * r.ck[22],
                   (1.0f + 0.0f) / (4.0f + 0.0f));
    } else if (qh <= 1.0e-14f || (qh < 1.0e-6f && tmp2 < 0.80f)) {
        tmp3 = max(0.0f, qh); q = q + tmp3; t = t - 0.334e+6f * tmp3;
        qh = 0.0f; nh = 0.0f;
    }
    r.T[gid] = t; r.Q[gid] = q;
    r.QC[gid] = qc; r.QR[gid] = qr; r.QI[gid] = qi;
    r.QN[gid] = qn; r.QG[gid] = qg; r.QH[gid] = qh;
    r.NC[gid] = nc; r.NR[gid] = nr; r.NY[gid] = ny_;
    r.NN[gid] = nn; r.NG[gid] = ng; r.NH[gid] = nh;
    r.QC_in[gid] = qc; r.QR_in[gid] = qr;
    r.NC_in[gid] = nc; r.NR_in[gid] = nr;
    r.gamfact[gid] = sqrt(1.225f / de);
}

kernel void milbrandt2_geometry(
    constant Milbrandt2GeometryResources &r [[buffer(0)]],
    constant Milbrandt2GeometryParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.nz * p.ny * p.nx;
    if (tid >= uint(n)) return;
    const int gid = int(tid);
    const int plane = p.ny * p.nx;
    const int k = gid / plane;
    const int col = gid % plane;
    const float iDP = k == 0
        ? 1.0f / (r.psfc[col] - r.pres[gid])
        : 1.0f / (r.pres[gid - plane] - r.pres[gid]);
    const float idz = r.DE[gid] * 9.80616f * iDP;
    r.iDZ[gid] = idz;
    r.DZ[gid] = 1.0f / idz;
}

kernel void milbrandt2_diagnostics(
    constant Milbrandt2DiagnosticsResources &r [[buffer(0)]],
    constant Milbrandt2DiagnosticsParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.nz * p.ny * p.nx;
    if (tid >= uint(n)) return;
    const int gid = int(tid);
    float q = r.Q[gid];
    if (q < 0.0f) q = 0.0f;
    r.Q[gid] = q;
    const float t = r.T[gid], pr = r.pres[gid];
    const float de = pr / (287.05f * t);
    const float tmp9 = de * de;
    const float N_r = r.NR[gid], N_i = r.NY[gid], N_s = r.NN[gid];
    const float N_g = r.NG[gid], N_h = r.NH[gid];
    const float qr = r.QR[gid], qi = r.QI[gid], qn = r.QN[gid];
    const float qg = r.QG[gid], qh = r.QH[gid];
    float tmp1 = 0.0f, tmp2 = 0.0f, tmp3 = 0.0f, tmp4 = 0.0f, tmp5 = 0.0f;
    if (qr > 1.0e-14f && N_r > 1.0e-3f) tmp1 = r.ck[147] * r.ck[149] * tmp9 * qr * qr / N_r;
    if (qi > 1.0e-14f && N_i > 1.0e-3f) tmp2 = r.ck[148] * r.ck[150] * tmp9 * qi * qi / N_i;
    if (qn > 1.0e-14f && N_s > 1.0e-3f) tmp3 = r.ck[148] * r.ck[151] * tmp9 * qn * qn / N_s;
    if (qg > 1.0e-14f && N_g > 1.0e-3f) tmp4 = r.ck[148] * r.ck[152] * tmp9 * qg * qg / N_g;
    if (qh > 1.0e-14f && N_h > 1.0e-3f) tmp5 = r.ck[148] * r.ck[153] * tmp9 * qh * qh / N_h;
    if (t > 273.16f) {
        tmp2 *= 4.464f; tmp3 *= 4.464f; tmp4 *= 4.464f; tmp5 *= 4.464f;
    }
    float zet = tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
    if (zet > 0.0f) zet = 10.0f * log10(zet * 1.0e+18f);
    else zet = -99.0f;
    r.ZET[gid] = max(zet, -99.0f);
    const float ide = (287.05f * t) / pr;
    r.NC[gid] *= ide; r.NR[gid] *= ide; r.NY[gid] *= ide;
    r.NN[gid] *= ide; r.NG[gid] *= ide; r.NH[gid] *= ide;
}

static inline float my2_dim(float x, float y) {
    const float d = x - y;
    return d > 0.0f ? d : 0.0f;
}

static inline float my2_des_OF_Ds(float Ds, float desMax,
                                  float eds_, float fds_) {
    const float v = eds_ * exp(fds_ * log(Ds));
    return min(desMax, v);
}

static inline float my2_Dm_x(float DE, float QX, float iNX,
                             float icmx, float idmx) {
    return exp(idmx * log(DE * QX * iNX * icmx));
}

static inline float my2_iLAMDA_x(float DE, float QX, float iNX,
                                 float icex, float idmx) {
    return exp(idmx * log(DE * QX * iNX * icex));
}

static inline float my2_SxFNC_w(float Win, float Tin, float Pin) {
    const float x = log10(max(Win, 1.0e-20f) * 100.0f);
    const float x2 = x * x, x3 = x2 * x, x4 = x2 * x2;
    const float T = Tin;
    const float T2 = T * T;
    const float pressure = Pin * 0.01f;
    const float p2 = pressure * pressure;
    const float a = 3.80e-5f*T2 + 1.65e-4f*T + 9.88e-2f;
    const float b = -7.38e-5f*T2 - 2.53e-3f*T - 3.23e-1f;
    const float c = 8.39e-5f*T2 + 3.96e-3f*T + 3.50e-1f;
    const float d = -1.88e-6f*T2 - 1.33e-3f*T - 3.73e-2f;
    const float f = -1.9761e-6f*p2 + 4.1473e-3f*pressure - 1.771e0f;
    const float g = 0.1539f*x4 - 0.5575f*x3 + 0.9262f*x2 - 0.3498f*x - 0.1293f;
    const float h = -8.035e-9f*(T2*T2) + 3.162e-7f*(T2*T)
        + 1.029e-5f*T2 - 5.931e-4f*T + 5.62e-2f;
    const float Pcorr = f * g * h;
    float Sw = (a*x3 + b*x2 + c*x + d) + Pcorr;
    Sw = 1.0f + 0.01f * Sw;
    return Win <= 0.0f ? 1.0f : Sw;
}

template <int KMAX>
static inline float my2_sedi_1d(
    device float *QX, device float *NX,
    device const float *DE, device const float *iDE,
    device const float *gamfact, device const float *DZ,
    device const float *iDZ, int st, int nz, int ktop, int cat,
    float epsQ, float epsN, float dmx, float VxMax, float DxMax,
    float dt, float afx, float bfx, float cmx,
    float ckQx1, float ckQx2, float ckQx4) {
    thread float VVQ[KMAX];
    thread float VVN[KMAX];
    const float icmx = 1.0f / cmx;
    float ratio_Vn2Vq = ckQx2 / ckQx1;
    float flux = 0.0f;
    const float iDxMax = 1.0f / DxMax;
    const float idmx = 1.0f / dmx;
    (void)afx; (void)iDE;
    for (int k = 0; k < nz; ++k) { VVQ[k] = 0.0f; VVN[k] = 0.0f; }
    for (int k = 0; k <= ktop; ++k) {
        const ulong o = ulong(k) * ulong(st);
        const float qx = QX[o], nxv = NX[o];
        if (qx > epsQ && nxv > epsN) {
            const float iLAMx = pow((qx * DE[o] / nxv) * ckQx4, idmx);
            const float iLAMxB0 = pow(iLAMx, bfx);
            VVQ[k] = gamfact[o] * iLAMxB0 * ckQx1;
        }
    }
    float vmax = 0.0f;
    for (int k = 0; k < nz; ++k) vmax = max(vmax, VVQ[k]);
    const float Vxmaxx = min(VxMax, vmax);
    float dzMIN = DZ[0];
    for (int k = 1; k < nz; ++k) dzMIN = min(dzMIN, DZ[ulong(k) * ulong(st)]);
    int npassx = int(dt * Vxmaxx / (0.8f * dzMIN) + 0.5f);
    if (npassx < 1) npassx = 1;
    const float dtx = dt / float(npassx);
    for (int nnn = 1; nnn <= npassx; ++nnn) {
        const bool firstPass = nnn == 1;
        for (int k = 0; k <= ktop; ++k) {
            const ulong o = ulong(k) * ulong(st);
            const float qx = QX[o], nxv = NX[o];
            if (qx > epsQ && nxv > epsN) {
                if (firstPass) VVQ[k] = -VVQ[k];
                else {
                    const float iLAMx = pow((qx * DE[o] / nxv) * ckQx4, idmx);
                    const float iLAMxB0 = pow(iLAMx, bfx);
                    VVQ[k] = -(gamfact[o] * iLAMxB0 * ckQx1);
                }
                if (cat == 5) {
                    const float t1 = pow(icmx * qx / nxv, 1.0f / 3.0f);
                    const float t2 = min(50.0f, 0.1f * (1000.0f * t1));
                    ratio_Vn2Vq = ((3.0f+t2)*(2.0f+t2)*(1.0f+t2))
                        / ((3.0f+bfx+t2)*(2.0f+bfx+t2)*(1.0f+bfx+t2));
                }
                VVN[k] = VVQ[k] * ratio_Vn2Vq;
            } else { VVQ[k] = 0.0f; VVN[k] = 0.0f; }
        }
        const ulong o0 = 0ul;
        flux = flux - VVQ[0] * DE[o0] * QX[o0];
        for (int k = 0; k <= ktop; ++k) {
            const ulong o = ulong(k) * ulong(st), o1 = ulong(k+1) * ulong(st);
            QX[o] = QX[o] + dtx * iDE[o]
                * (-DE[o1] * QX[o1] * VVQ[k+1] + DE[o] * QX[o] * VVQ[k])
                * iDZ[o1];
            NX[o] = NX[o] + dtx
                * (-NX[o1] * VVN[k+1] + NX[o] * VVN[k]) * iDZ[o1];
            QX[o] = max(QX[o], 0.0f);
            NX[o] = max(NX[o], 0.0f);
        }
        for (int k = 0; k <= ktop; ++k) {
            const ulong o = ulong(k) * ulong(st);
            if (QX[o] > epsQ && NX[o] < epsN) {
                int kk = k + 1;
                while (kk <= ktop && NX[ulong(kk) * ulong(st)] < epsN) ++kk;
                NX[o] = max(epsN, NX[ulong(kk) * ulong(st)]);
            }
            if (QX[o] > epsQ && NX[o] > epsN) {
                const float Dx = pow(DE[o] * QX[o] / (NX[o] * cmx), idmx);
                if (cat == 1 && Dx > 3.0e-3f) {
                    NX[o] = NX[o] * max(1.0f + 2.0e4f * (Dx-3.0e-3f) * (Dx-3.0e-3f),
                                         pow(Dx * iDxMax, 3.0f));
                } else {
                    NX[o] = NX[o] * pow(max(Dx, DxMax) * iDxMax, dmx);
                }
            }
        }
    }
    return flux / float(npassx);
}

static inline int my2_count_column(device const float *QX, float minQX,
                                   int ktop_sedi, int st) {
    int k = ktop_sedi;
    while (true) {
        --k;
        if (k < 0) return -1;
        if (QX[ulong(k) * ulong(st)] > minQX) return k;
        if (k == 0) return -1;
    }
}

template <int KMAX, typename R, typename P>
static inline void my2_sediment_column(constant R &r, constant P &p, int col) {
    const int st = p.ny * p.nx;
    device const float *DEc = r.DE + col;
    device const float *iDEc = r.iDE + col;
    device const float *DZc = r.DZ + col;
    device const float *iDZc = r.iDZ + col;
    device const float *gfc = r.gamfact + col;
    device float *QRc = r.QR + col, *NRc = r.NR + col;
    device float *QIc = r.QI + col, *NYc = r.NY + col;
    device float *QNc = r.QN + col, *NNc = r.NN + col;
    device float *QGc = r.QG + col, *NGc = r.NG + col;
    device float *QHc = r.QH + col, *NHc = r.NH + col;
    int ktop_sedi = 0;
    float z = 0.0f;
    for (int k = 0; k < p.nz; ++k) {
        z = k == 0 ? DZc[0] : z + DZc[ulong(k) * ulong(st)];
        if (z < 20000.0f) ktop_sedi = k;
    }
    float fluxM_r = 0.0f, fluxM_i = 0.0f, fluxM_s = 0.0f;
    float fluxM_g = 0.0f, fluxM_h = 0.0f;
    int kt = my2_count_column(QRc, 1.0e-8f, ktop_sedi, st);
    if (kt >= 0) fluxM_r = my2_sedi_1d<KMAX>(QRc, NRc, DEc, iDEc, gfc, DZc, iDZc, st, p.nz, kt, 1,
        1.0e-14f, 1.0e-3f, 3.0f, 16.0f, 5.0e-3f, p.dt, 149.1f, 0.5f, r.ck[15], r.ck[70], r.ck[71], r.ck[67]);
    kt = my2_count_column(QIc, 1.0e-10f, ktop_sedi, st);
    if (kt >= 0) fluxM_i = my2_sedi_1d<KMAX>(QIc, NYc, DEc, iDEc, gfc, DZc, iDZc, st, p.nz, kt, 2,
        1.0e-14f, 1.0e-3f, 3.0f, 2.0f, 5.0e-3f, p.dt, 71.34f, 0.6635f, r.ck[17], r.ck[88], r.ck[89], r.ck[90]);
    kt = my2_count_column(QNc, 1.0e-8f, ktop_sedi, st);
    if (kt >= 0) fluxM_s = my2_sedi_1d<KMAX>(QNc, NNc, DEc, iDEc, gfc, DZc, iDZc, st, p.nz, kt, 3,
        1.0e-14f, 1.0e-3f, r.ck[25], 4.0f, 5.0e-3f, p.dt, 11.72f, 0.41f, r.ck[24], r.ck[109], r.ck[110], r.ck[108]);
    kt = my2_count_column(QGc, 1.0e-8f, ktop_sedi, st);
    if (kt >= 0) fluxM_g = my2_sedi_1d<KMAX>(QGc, NGc, DEc, iDEc, gfc, DZc, iDZc, st, p.nz, kt, 4,
        1.0e-14f, 1.0e-3f, 3.0f, 6.0f, 2.0e-3f, p.dt, 19.3f, 0.37f, r.ck[19], r.ck[129], r.ck[130], r.ck[131]);
    kt = my2_count_column(QHc, 1.0e-10f, ktop_sedi, st);
    if (kt >= 0) fluxM_h = my2_sedi_1d<KMAX>(QHc, NHc, DEc, iDEc, gfc, DZc, iDZc, st, p.nz, kt, 5,
        1.0e-14f, 1.0e-3f, 3.0f, 25.0f, 80.0e-3f, p.dt, 206.89f, 0.6384f, r.ck[21], r.ck[144], r.ck[145], r.ck[146]);
    for (int k = p.nz - 1; k >= 0; --k) {
        const ulong o = ulong(k) * ulong(st);
        float qn = QNc[o], nn = NNc[o];
        if (qn > 1.0e-14f && nn > 1.0e-3f) {
            float t1 = 1.0f / nn;
            float iLAMs = max(1.0e-10f, exp((1.0f / r.ck[27]) * log(DEc[o] * qn * t1 * r.ck[108])));
            t1 = min(nn / iLAMs, 1.0e+8f);
            nn = pow(t1, r.ck[25] / (1.0f + r.ck[25]))
                 * pow(r.ck[108] * DEc[o] * qn, 1.0f / (1.0f + r.ck[25]));
            t1 = 1.0f / nn;
            iLAMs = max(1.0e-10f, exp((1.0f / r.ck[27]) * log(DEc[o] * qn * t1 * r.ck[108])));
            const float t2 = 1.0f / iLAMs;
            const float t4 = 0.6f * 500.0f;
            const float t5 = 2.0f * t4;
            float t3 = t2 + t4 * pow(max(0.0f, t5 - t2) / t5, 2.0f);
            t3 = max(t3, 500.0f);
            nn = nn * pow(t3 * iLAMs, r.ck[25]);
            NNc[o] = nn;
        }
    }
    const float idew = r.ck[11];
    float RT_rn1 = fluxM_r * idew, RT_rn2 = 0.0f;
    float RT_fr1 = 0.0f, RT_fr2 = 0.0f;
    float RT_sn1 = fluxM_i * idew, RT_sn2 = fluxM_s * idew;
    float RT_sn3 = fluxM_g * idew, RT_pe1 = fluxM_h * idew, RT_pe2 = 0.0f;
    const ulong obot = ulong(col), otop = ulong(p.nz-1) * ulong(st) + ulong(col);
    const float N_r = NRc[ulong(p.nz-1) * ulong(st)];
    if (QRc[0] > 1.0e-14f && N_r > 1.0e-3f) {
        const float Dm_r = pow(DEc[ulong(p.nz-1)*ulong(st)] * r.ck[16] * QRc[0] / N_r, 1.0f/3.0f);
        if (Dm_r > 200.0e-6f) { RT_rn2 = RT_rn1; RT_rn1 = 0.0f; }
    }
    if (r.T[otop] < 273.16f) { RT_fr1 = RT_rn1; RT_rn1 = 0.0f; RT_fr2 = RT_rn2; RT_rn2 = 0.0f; }
    if (r.T[otop] > (273.16f + 5.0f)) { RT_pe2 = RT_pe1; RT_pe1 = 0.0f; }
    const float ms2mmstp = 1.0e+3f * p.dt;
    const float rncv = (RT_rn1 + RT_rn2 + RT_fr1 + RT_fr2 + RT_sn1 + RT_sn2 + RT_sn3 + RT_pe1 + RT_pe2) * ms2mmstp;
    const float sncv = (RT_sn1 + RT_sn2) * ms2mmstp;
    const float hlcv = (RT_pe1 + RT_pe2) * ms2mmstp;
    const float grcv = RT_sn3 * ms2mmstp;
    r.rainncv[col] = rncv; r.snowncv[col] = sncv;
    r.hailncv[col] = hlcv; r.graupelncv[col] = grcv;
    r.rainnc[col] += rncv; r.snownc[col] += sncv;
    r.hailnc[col] += hlcv; r.graupelnc[col] += grcv;
    r.sr[col] = (sncv + hlcv + grcv) / (rncv + 1.0e-12f);
    (void)obot;
}

template <int KMAX, typename R, typename P>
static inline void my2_sediment_dispatch(constant R &r, constant P &p, uint tid) {
    const int ncol = p.ny * p.nx;
    if (tid >= uint(ncol) || p.nz > KMAX) return;
    my2_sediment_column<KMAX>(r, p, int(tid));
}

kernel void milbrandt2_sediment_64(
    constant Milbrandt2Sediment64Resources &r [[buffer(0)]],
    constant Milbrandt2Sediment64Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    my2_sediment_dispatch<64>(r, p, tid);
}

kernel void milbrandt2_sediment_256(
    constant Milbrandt2Sediment256Resources &r [[buffer(0)]],
    constant Milbrandt2Sediment256Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    my2_sediment_dispatch<256>(r, p, tid);
}

// The source kernel is a per-cell transcription of MY2 Part 2. Integer grid
// setup is replaced by a single flat Metal cell id; the top-level exclusion,
// local statement order, and all source thresholds remain in the body.
#define fminf min
#define fmaxf max
#define powf pow
#define expf exp
#define logf log
#define log10f log10
#define sqrtf sqrt
#define fabsf fabs
#define cbrtf cbrt
static inline float rn_div(float a, float b) { return a / b; }

#define PI2 ck[0]
#define PIov4 ck[1]
#define PIov6 ck[2]
#define CHLS ck[3]
#define LCP ck[4]
#define LFP ck[5]
#define iCHLF ck[6]
#define LSP ck[7]
#define ck5 ck[8]
#define ck6 ck[9]
#define imgo ck[10]
#define idew ck[11]
#define idei ck[12]
#define ideg ck[13]
#define ideh ck[14]
#define cmr ck[15]
#define icmr ck[16]
#define cmi ck[17]
#define icmi ck[18]
#define cmg ck[19]
#define icmg ck[20]
#define cmh ck[21]
#define icmh ck[22]
#define cms_D3 ck[23]
#define cms ck[24]
#define dms ck[25]
#define icms ck[26]
#define idms ck[27]
#define mso ck[28]
#define imso ck[29]
#define eds ck[30]
#define fds ck[31]
#define GS50 ck[32]
#define iMUc ck[33]
#define GC1 ck[34]
#define iGC1 ck[35]
#define GC2 ck[36]
#define GC3 ck[37]
#define GC4 ck[38]
#define GC11 ck[39]
#define GC12 ck[40]
#define GC5 ck[41]
#define iGC5 ck[42]
#define GC6 ck[43]
#define GC7 ck[44]
#define GC8 ck[45]
#define GC13 ck[46]
#define GC14 ck[47]
#define GC15 ck[48]
#define icexc9 ck[49]
#define N_c_SM ck[50]
#define cexr1 ck[51]
#define cexr2 ck[52]
#define GR17 ck[53]
#define GR31 ck[54]
#define iGR31 ck[55]
#define GR32 ck[56]
#define GR33 ck[57]
#define GR34 ck[58]
#define iGR34 ck[59]
#define GR35 ck[60]
#define GR36 ck[61]
#define GR37 ck[62]
#define GR50 ck[63]
#define cexr5 ck[64]
#define cexr6 ck[65]
#define cexr9 ck[66]
#define icexr9 ck[67]
#define cexr3 ck[68]
#define cexr4 ck[69]
#define ckQr1 ck[70]
#define ckQr2 ck[71]
#define ckQr3 ck[72]
#define GI4 ck[73]
#define GI6 ck[74]
#define GI11 ck[75]
#define GI20 ck[76]
#define GI21 ck[77]
#define GI22 ck[78]
#define GI31 ck[79]
#define iGI31 ck[80]
#define GI32 ck[81]
#define GI33 ck[82]
#define GI34 ck[83]
#define GI35 ck[84]
#define GI36 ck[85]
#define GI40 ck[86]
#define icexi9 ck[87]
#define ckQi1 ck[88]
#define ckQi2 ck[89]
#define ckQi4 ck[90]
#define cexs1 ck[91]
#define cexs2 ck[92]
#define icexs2 ck[93]
#define GS09 ck[94]
#define GS11 ck[95]
#define GS12 ck[96]
#define GS13 ck[97]
#define GS31 ck[98]
#define iGS31 ck[99]
#define GS32 ck[100]
#define GS33 ck[101]
#define GS34 ck[102]
#define iGS34 ck[103]
#define GS35 ck[104]
#define GS36 ck[105]
#define GS40 ck[106]
#define iGS40 ck[107]
#define iGS20 ck[108]
#define ckQs1 ck[109]
#define ckQs2 ck[110]
#define GS40_D3 ck[111]
#define iGS20_D3 ck[112]
#define rfact_FvFm ck[113]
#define GG09 ck[114]
#define GG11 ck[115]
#define GG12 ck[116]
#define GG13 ck[117]
#define GG31 ck[118]
#define iGG31 ck[119]
#define GG32 ck[120]
#define GG33 ck[121]
#define GG34 ck[122]
#define iGG34 ck[123]
#define GG35 ck[124]
#define GG36 ck[125]
#define GG40 ck[126]
#define iGG99 ck[127]
#define GG50 ck[128]
#define ckQg1 ck[129]
#define ckQg2 ck[130]
#define ckQg4 ck[131]
#define GH09 ck[132]
#define GH11 ck[133]
#define GH12 ck[134]
#define GH13 ck[135]
#define GH31 ck[136]
#define iGH31 ck[137]
#define GH32 ck[138]
#define GH33 ck[139]
#define iGH34 ck[140]
#define GH40 ck[141]
#define iGH99 ck[142]
#define GH50 ck[143]
#define ckQh1 ck[144]
#define ckQh2 ck[145]
#define ckQh4 ck[146]
#define cxr ck[147]
#define cxi ck[148]
#define Gzr ck[149]
#define Gzi ck[150]
#define Gzs ck[151]
#define Gzg ck[152]
#define Gzh ck[153]

// --------------------------------------------------------------------------
// Scheme parameters (:1026-1176).  Named exactly as the Fortran except MY2_T0
// (the contact-nucleation reference temperature, :1161) which would collide
// with the kernel preamble's base-state T0.
// --------------------------------------------------------------------------
#define MY2_alpha_c   1.0f
#define MY2_alpha_r   0.0f
#define MY2_alpha_i   0.0f
#define MY2_alpha_s   0.0f
#define MY2_alpha_g   0.0f
#define MY2_alpha_h   0.0f
#define MY2_No_s_max  1.0e+8f
#define MY2_lamdas_min 500.0f
#define MY2_No_r_SM   1.0e+7f
#define MY2_No_g_SM   4.0e+6f
#define MY2_No_h_SM   1.0e+5f
#define MY2_afr 149.100f
#define MY2_bfr 0.5000f
#define MY2_afi 71.340f
#define MY2_bfi 0.6635f
#define MY2_afs 11.720f
#define MY2_bfs 0.4100f
#define MY2_afg 19.300f
#define MY2_bfg 0.3700f
#define MY2_afh 206.890f
#define MY2_bfh 0.6384f
#define MY2_epsQ  1.0e-14f
#define MY2_epsN  1.0e-3f
#define MY2_epsQ2 1.0e-6f
#define MY2_iLAMmin1 1.0e-6f
#define MY2_iLAMmin2 1.0e-10f
#define MY2_deg 400.0f
#define MY2_mgo 1.6e-10f
#define MY2_deh 900.0f
#define MY2_dei 500.0f
#define MY2_mio 1.0e-12f
#define MY2_dew 1000.0f
#define MY2_desFix 100.0f
#define MY2_desMax 500.0f
#define MY2_Dso 125.0e-6f
#define MY2_dmr 3.0f
#define MY2_dmi 3.0f
#define MY2_dmg 3.0f
#define MY2_dmh 3.0f
#define MY2_DrMax 5.0e-3f
#define MY2_VrMax 16.0f
#define MY2_epsQr_sedi 1.0e-8f
#define MY2_DiMax 5.0e-3f
#define MY2_ViMax 2.0f
#define MY2_epsQi_sedi 1.0e-10f
#define MY2_DsMax 5.0e-3f
#define MY2_VsMax 4.0f
#define MY2_epsQs_sedi 1.0e-8f
#define MY2_DgMax 2.0e-3f
#define MY2_VgMax 6.0f
#define MY2_epsQg_sedi 1.0e-8f
#define MY2_DhMax 80.0e-3f
#define MY2_VhMax 25.0f
#define MY2_epsQh_sedi 1.0e-10f
#define MY2_CPW 4218.0f
#define MY2_DEo 1.225f
#define MY2_thrd (1.0f / 3.0f)
#define MY2_Ers 1.0f
#define MY2_Eci 1.0f
#define MY2_Eri 1.0f
#define MY2_Erh 1.0f
#define MY2_Avx 0.78f
#define MY2_Bvx 0.30f
#define MY2_Abigg 0.66f
#define MY2_Bbigg 100.0f
#define MY2_fdielec 4.464f
#define MY2_zfact 1.0e+18f
#define MY2_minZET (-99.0f)
#define MY2_Drshed 0.001f
#define MY2_SIGcTHRS 15.0e-6f
#define MY2_KK1 3.03e3f
#define MY2_KK2 2.59e15f
#define MY2_Dhh 82.0e-6f
#define MY2_zMax_sedi 20000.0f
#define MY2_Dr_large 200.0e-6f
#define MY2_Dh_large 1.0e-2f
#define MY2_Dh_min 1.0e-3f
#define MY2_Dr_3cmpThrs 2.5e-3f
#define MY2_Ngh_crit 1.0e+0f
#define MY2_Tc_FZrh (-10.0f)
#define MY2_CNsgThres 1.0f
#define MY2_capFact_i 0.5f
#define MY2_capFact_s 0.5f
#define MY2_Ni_max 1.0e+7f
#define MY2_satw_peak 1.01f
// WRF + kin_1d thermodynamic constants (:1130-1155).  These are the
// scheme's OWN values and deliberately differ from gpuwm's module
// constants (e.g. GRAV=9.80616 here, not the dycore's g).
#define MY2_CPI 0.21153e+4f
#define MY2_TRPL 0.27316e+3f
#define MY2_PI 0.314159265359e+1f
#define MY2_CHLC 0.2501e+7f
#define MY2_CHLF 0.334e+6f
#define MY2_CPD 0.100546e+4f
#define MY2_RGASD 0.28705e+3f
#define MY2_RGASV 0.46151e+3f
#define MY2_EPS1 0.62194800221014f
#define MY2_GRAV 0.980616e+1f
// Contact-nucleation constants (:1160-1165).
#define MY2_LAMa0 6.6e-8f
#define MY2_T0 293.15f
#define MY2_p0 101325.0f
#define MY2_Ra 1.0e-6f
#define MY2_kBoltz 1.381e-23f
#define MY2_KAPa 5.39e5f
// sedi_1D locals (:637-641)
#define MY2_CoMAX 0.8f


kernel void milbrandt2_warm(constant Milbrandt2WarmResources &r [[buffer(0)]],
                            constant Milbrandt2WarmParams &p [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nz * p.ny * p.nx);
    if (tid >= count) return;
    const uint gid = tid;
    const int plane = p.ny * p.nx;
    const int k = int(gid) / plane;
    // Fortran :2848 excludes ktop: k = ktop-kdir, ..., kbot.
    if (k >= p.nz - 1) return;

    const float idt = 1.0f / p.dt;
    float t = r.T[gid], q = r.Q[gid];
    float qc = r.QC[gid], qr = r.QR[gid], qi = r.QI[gid];
    float nc = r.NC[gid], nr = r.NR[gid], ny_ = r.NY[gid];
    float de = r.DE[gid], ide = r.iDE[gid], pr = r.pres[gid];
    const float wz = r.WZ[gid];
    const float qc_in = r.QC_in[gid];
    const float qr_in = r.QR_in[gid];
    float nc_in = r.NC_in[gid], nr_in = r.NR_in[gid];
    device const float *ck = r.ck;

    float RCAUTR = 0.0f, CCACCR = 0.0f, Dc = 0.0f, iLAMc = 0.0f, L = 0.0f;
    float RCACCR = 0.0f, CCSCOC = 0.0f, Dr = 0.0f, iLAMr = 0.0f, TAU = 0.0f;
    float CCAUTR = 0.0f, CRSCOR = 0.0f, SIGc = 0.0f, DrINIT = 0.0f;
    float iLAMc3 = 0.0f, iLAMc6 = 0.0f, iLAMr3 = 0.0f, iLAMr6 = 0.0f;
    float tmp1, tmp2, tmp3, tmp4;
    float iNC, iNR;
    const bool rainPresent = qr_in > MY2_epsQ && nr_in > MY2_epsN;

    if (qc_in > MY2_epsQ && nc_in > MY2_epsN) {
        iNC = 1.0f / nc_in;
        iLAMc = my2_iLAMDA_x(de, qc_in, iNC, icexc9, MY2_thrd);
        iLAMc3 = iLAMc * iLAMc * iLAMc;
        iLAMc6 = iLAMc3 * iLAMc3;
        Dc = iLAMc * pow(GC2 * iGC1, MY2_thrd);
        SIGc = iLAMc * pow(GC3 * iGC1 - (GC2 * iGC1) * (GC2 * iGC1),
                            0.5f * MY2_thrd);
        L = 0.027f * de * qc_in * (6.25e18f * SIGc * SIGc * SIGc * Dc - 0.4f);
        if (SIGc > MY2_SIGcTHRS)
            TAU = 3.7f / (de * qc_in * (0.5e6f * SIGc - 7.5f));
    }

    if (rainPresent) {
        iNR = 1.0f / nr_in;
        Dr = my2_Dm_x(de, qr_in, iNR, icmr, MY2_thrd);
        if (Dr > 3.0e-3f) {
            tmp1 = Dr - 3.0e-3f;
            tmp2 = Dr / MY2_DrMax;
            tmp3 = tmp2 * tmp2 * tmp2;
            nr_in = nr_in * max(1.0f + 2.0e4f * tmp1 * tmp1, tmp3);
            iNR = 1.0f / nr_in;
            Dr = my2_Dm_x(de, qr_in, iNR, icmr, MY2_thrd);
        }
        iLAMr = my2_iLAMDA_x(de, qr_in, iNR, icexr9, MY2_thrd);
        iLAMr3 = iLAMr * iLAMr * iLAMr;
        iLAMr6 = iLAMr3 * iLAMr3;
    }

    if (qc_in > MY2_epsQ && SIGc > MY2_SIGcTHRS) {
        RCAUTR = min(max(L / TAU, 0.0f), qc * idt);
        DrINIT = max(83.0e-6f, 12.6e-4f / (0.5e6f * SIGc - 3.5f));
        const float DrAUT = max(DrINIT, Dr);
        CCAUTR = RCAUTR * de / (cmr * DrAUT * DrAUT * DrAUT);
        CCSCOC = min(MY2_KK2 * nc_in * nc_in * GC3 * iGC1 * iLAMc6,
                     nc_in * idt);
    }

    if ((qr_in > 1.2f * max(L, 0.0f) * ide
         || Dr > max(5.0e-6f, DrINIT)) && rainPresent) {
        if (qc_in > MY2_epsQ && L > 0.0f) {
            if (Dr >= 100.0e-6f) {
                CCACCR = MY2_KK1 * (nc_in * nr_in)
                         * (GC2 * iGC1 * iLAMc3 + GR34 * iGR31 * iLAMr3);
                RCACCR = cmr * ide * MY2_KK1 * (nc_in * nr_in) * iLAMc3
                         * (GC3 * iGC1 * iLAMc3
                            + GC2 * iGC1 * GR34 * iGR31 * iLAMr3);
            } else {
                CCACCR = MY2_KK2 * (nc_in * nr_in)
                         * (GC3 * iGC1 * iLAMc6 + GR37 * iGR31 * iLAMr6);
                tmp1 = cmr * ide;
                tmp2 = MY2_KK2 * (nc_in * nr_in) * iLAMc3;
                RCACCR = tmp1 * tmp2;
                tmp1 = GC4 * iGR31;
                tmp1 = tmp1 * iLAMc6;
                tmp2 = GC2 * iGC1;
                tmp2 = tmp2 * GR37 * iGR31;
                tmp2 = tmp2 * iLAMr6;
                RCACCR = RCACCR * (tmp1 + tmp2);
            }
            CCACCR = min(CCACCR, nc * idt);
            RCACCR = min(RCACCR, qc * idt);
        }
        tmp1 = nr_in * nr_in;
        if (Dr >= 100.0e-6f)
            CRSCOR = MY2_KK1 * tmp1 * GR34 * iGR31 * iLAMr3;
        else
            CRSCOR = MY2_KK2 * tmp1 * GR37 * iGR31 * iLAMr6;
        float Ec = 1.0f;
        if (iLAMr > 300.0e-6f) Ec = 2.0f - exp(2300.0f * (iLAMr - 300.0e-6f));
        CRSCOR = min(Ec * CRSCOR, (0.5f * nr) * idt);
    }

    {
        const float source = qc;
        const float sink = (RCAUTR + RCACCR) * p.dt;
        if (sink > source) {
            const float ratio = source / sink;
            RCAUTR = ratio * RCAUTR;
            RCACCR = ratio * RCACCR;
            CCACCR = ratio * CCACCR;
        }
    }
    qc = max(0.0f, qc + (-RCAUTR - RCACCR) * p.dt);
    qr = max(0.0f, qr + (RCAUTR + RCACCR) * p.dt);
    nc = max(0.0f, nc + (-CCACCR - CCSCOC) * p.dt);
    nr = max(0.0f, nr + (CCAUTR - CRSCOR) * p.dt);

    if (qr > MY2_epsQ && nr > MY2_epsN) {
        iNR = 1.0f / nr;
        Dr = my2_Dm_x(de, qr, iNR, icmr, MY2_thrd);
        if (Dr > 3.0e-3f) {
            tmp1 = Dr - 3.0e-3f; tmp2 = tmp1 * tmp1;
            tmp3 = Dr / MY2_DrMax; tmp4 = tmp3 * tmp3 * tmp3;
            nr = nr * max(1.0f + 2.0e4f * tmp2, tmp4);
        } else if (Dr < MY2_Dhh) {
            qc = qc + qr; nc = nc + nr;
            qr = 0.0f; nr = 0.0f;
        }
    } else { qr = 0.0f; nr = 0.0f; }

    float qsw = my2_qsat(t, pr, 0);
    r.QSW[gid] = qsw;
    float X = q - qsw;
    X = X / (1.0f + ((3.1484e6f - 2370.0f * t) * (3.1484e6f - 2370.0f * t) * qsw)
                    / ((1005.0f * (1.0f + 0.887f * q)) * 461.5f * t * t));
    X = max(X, -qc);
    qc = qc + X; q = q - X; t = t + LCP * X;
    if (X > 0.0f) {
        if (wz > 0.001f) nc = max(nc, my2_NccnFNC(wz, t, pr));
        else nc = max(nc, N_c_SM);
    } else {
        if (qc > MY2_epsQ) nc = max(0.0f, nc + X * nc / max(qc, MY2_epsQ));
        else nc = 0.0f;
    }
    if (qc > MY2_epsQ && nc < MY2_epsN) nc = N_c_SM;

    qsw = my2_qsat(t, pr, 0);
    r.QSW[gid] = qsw;
    if (q < qsw && qr > MY2_epsQ && nr > MY2_epsN) {
        const float Tc = t - MY2_TRPL;
        const float Cdiff = max(1.62e-5f, 2.2157e-5f + 0.0155e-5f * Tc) * 1.0e5f / pr;
        const float MUdyn = max(1.51e-5f, 1.7153e-5f + 0.0050e-5f * Tc);
        const float Ka = max(2.07e-2f, 2.3971e-2f + 0.0078e-2f * Tc);
        const float MUkin = MUdyn * ide;
        const float iMUkin = 1.0f / MUkin;
        const float ScTHRD = pow(MUkin / Cdiff, MY2_thrd);
        X = qsw - q;
        X = X / (1.0f + ((3.1484e6f - 2370.0f * t) * (3.1484e6f - 2370.0f * t) * qsw)
                        / ((1005.0f * (1.0f + 0.887f * q)) * 461.5f * t * t));
        (void)X;
        de = pr / (MY2_RGASD * t);
        ide = 1.0f / de;
        r.DE[gid] = de; r.iDE[gid] = ide;
        const float gam = sqrt(MY2_DEo * ide);
        tmp1 = 1.0f / nr;
        iLAMr = my2_iLAMDA_x(de, qr, tmp1, icexr9, MY2_thrd);
        const float LAMr = 1.0f / iLAMr;
        // WRF :3052-3054 promotes nr, LAMr and iGR31 to binary64, evaluates
        // pow(LAMr, 1 + alpha_r), multiplies left-to-right, then rounds the
        // assignment to REAL(4). alpha_r is fixed at REAL(4) 0.0 (:1026-1078),
        // so the exponent is exactly one and positive finite LAMr**1 is LAMr.
        const ulong no_r64 = soft64_mul(
            soft64_mul(soft64_from_f32_bits(as_type<uint>(nr)),
                       soft64_from_f32_bits(as_type<uint>(LAMr))),
            soft64_from_f32_bits(as_type<uint>(iGR31)));
        const float No_r = as_type<float>(soft64_to_f32_bits(no_r64));
        const float VENTr = MY2_Avx * GR32 * pow(iLAMr, cexr5)
                      + MY2_Bvx * ScTHRD * sqrt(gam * MY2_afr * iMUkin)
                        * GR17 * pow(iLAMr, cexr6);
        const float ABw = MY2_CHLC * MY2_CHLC / (Ka * MY2_RGASV * t * t)
                    + 1.0f / (de * qsw * Cdiff);
        const float ssat = q / qsw - 1.0f;
        const float rate = -(p.dt * (ide * PI2 * ssat * No_r * VENTr / ABw));
        const float QREVP = min(qr, rate);
        tmp1 = qr;
        t = t - LCP * QREVP; q = q + QREVP; qr = qr - QREVP;
        nr = max(0.0f, nr - QREVP * nr / tmp1);
        if (qr < MY2_epsQ || nr < MY2_epsN) {
            q = q + qr; t = t - qr * LCP; qr = 0.0f; nr = 0.0f;
        }
    }

    {
        const float Tc = t - MY2_TRPL;
        if (qc > MY2_epsQ && Tc < -30.0f) {
            const float FRAC = Tc < -35.0f ? 1.0f : 0.0f;
            const float QFZci = FRAC * qc, NFZci = FRAC * nc;
            qc = qc - QFZci; nc = nc - NFZci;
            qi = qi + QFZci; ny_ = ny_ + NFZci; t = t + LFP * QFZci;
            if (qc > MY2_epsQ && nc < MY2_epsN) nc = N_c_SM;
            else if (qc <= MY2_epsQ) { q = q + qc; t = t - LCP * qc; qc = 0.0f; nc = 0.0f; }
            if (qi > MY2_epsQ && ny_ < MY2_epsN) ny_ = max(2.0f * MY2_epsN, my2_N_Cooper(t));
            else if (qi <= MY2_epsQ) { q = q + qi; t = t - LSP * qi; qi = 0.0f; ny_ = 0.0f; }
        }
    }

    r.T[gid] = t; r.Q[gid] = q;
    r.QC[gid] = qc; r.QR[gid] = qr; r.QI[gid] = qi;
    r.NC[gid] = nc; r.NR[gid] = nr; r.NY[gid] = ny_;
    r.NR_in[gid] = nr_in;
    (void)qc_in; (void)nc_in;
}


kernel void milbrandt2_cold(constant Milbrandt2ColdResources &r [[buffer(0)]], constant Milbrandt2ColdParams &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    device float *T = r.T;
    device float *Q = r.Q;
    device float *QC = r.QC;
    device float *QR = r.QR;
    device float *QI = r.QI;
    device float *QN = r.QN;
    device float *QG = r.QG;
    device float *QH = r.QH;
    device float *NC = r.NC;
    device float *NR = r.NR;
    device float *NY = r.NY;
    device float *NN = r.NN;
    device float *NG = r.NG;
    device float *NH = r.NH;
    device const float *WZ = r.WZ;
    device const float *pres = r.pres;
    device const float *DE = r.DE;
    device const float *iDE = r.iDE;
    device const float *gamfact = r.gamfact;
    device const float *QSW = r.QSW;
    device const float *QSI = r.QSI;
    device const float *ck = r.ck;
    const float dt = p.dt;
    const int nz = p.nz, ny = p.ny, nx = p.nx;

    const uint count = uint(p.nz * p.ny * p.nx);
    if (tid >= count) return;
    const uint gid = tid;
    const int plane = p.ny * p.nx;
    const int k = int(gid) / plane;
    // :1592/:1610 -- the active-point and process loops run
    // ``k = ktop-kdir, kbot, -kdir``: the TOP level is excluded.
    if (k >= nz - 1) return;

    float t = T[gid], q = Q[gid];
    float qc = QC[gid], qr = QR[gid], qi = QI[gid];
    float qn = QN[gid], qg = QG[gid], qh = QH[gid];
    float nc = NC[gid], nr = NR[gid], ny_ = NY[gid];
    float nn = NN[gid], ng = NG[gid], nh = NH[gid];
    float qsw = QSW[gid], qsi = QSI[gid];
    float de = DE[gid], ide = iDE[gid], pr = pres[gid];
    float wz = WZ[gid];

    // :1594-1600 -- active grid points
    bool log1 = (qi + qg + qn + qh) < MY2_epsQ;
    bool log2 = (qc + qr) < MY2_epsQ;
    bool log3 = (t > MY2_TRPL) && log1;
    bool log4 = log1 && log2 && (q < qsi);
    if (log3 || log4) return;                            // icephase_ON = .T.

    float Tc = t - MY2_TRPL;                             // :1614
    // :1615-1618 -- WRF prints a warning for |Tc| outside [-120, 50] and does
    // not act on it (the ``stop`` is commented out).  Nothing to reproduce.
    float Cdiff = (2.2157e-5f + 0.0155e-5f * Tc) * 1.0e5f / pr;
    float MUdyn = 1.72e-5f * (393.0f / (t + 120.0f))
                  * pow(rn_div(t, MY2_TRPL), 1.5f);
    float MUkin = MUdyn * ide;
    float iMUkin = 1.0f / MUkin;
    float ScTHRD = pow(MUkin / Cdiff, MY2_thrd);
    float Ka = 2.3971e-2f + 0.0078e-2f * Tc;
    float Kdiff = (9.1018e-11f * t * t + 8.8197e-8f * t - (1.0654e-5f));
    float gam = gamfact[gid];

    // :1629-1634 -- collection efficiencies
    float Eis = min(0.05f * exp(0.1f * Tc), 1.0f);
    float Eig = min(0.01f * exp(0.1f * Tc), 1.0f);
    float Eii = 0.1f * Eis;
    float Ess = Eis, Eih = Eig, Esh = Eig;
    float iEih = 1.0f / Eih;
    float iEsh = 1.0f / Esh;
    (void)Eii;

    float qvs0 = my2_qsat(MY2_TRPL, pr, 0);              // :1641
    float DELqvs = qvs0 - q;                             // :1642

    float tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp8, tmp9, tmp10;
    float iQC = 0.0f, iQR = 0.0f, iQI = 0.0f, iQN = 0.0f, iQG = 0.0f;
    float iQH = 0.0f;
    float iNC = 0.0f, iNR = 0.0f, iNY = 0.0f, iNN = 0.0f, iNG = 0.0f;
    float iNH = 0.0f;

    // Cloud (:1645-1659)
    float Dc, iLAMc, iLAMc2, iLAMc3, iLAMc4, iLAMc5;
    if (qc > MY2_epsQ) {
        iQC = 1.0f / qc;
        iNC = 1.0f / nc;
        Dc = my2_Dm_x(de, qc, iNC, icmr, MY2_thrd);
        iLAMc = my2_iLAMDA_x(de, qc, iNC, icexc9, MY2_thrd);
        iLAMc2 = iLAMc * iLAMc;
        iLAMc3 = iLAMc2 * iLAMc;
        iLAMc4 = iLAMc2 * iLAMc2;
        iLAMc5 = iLAMc3 * iLAMc2;
    } else {
        Dc = 0.0f; iLAMc3 = 0.0f;
        iLAMc = 0.0f; iLAMc4 = 0.0f;
        iLAMc2 = 0.0f; iLAMc5 = 0.0f;
    }

    // Rain (:1662-1680)
    float Dr, iLAMr, iLAMr2, iLAMr3, iLAMr4, iLAMr5, vr0;
    if (qr > MY2_epsQ) {
        iQR = 1.0f / qr;
        iNR = 1.0f / nr;
        Dr = my2_Dm_x(de, qr, iNR, icmr, MY2_thrd);
        iLAMr = max(MY2_iLAMmin1,
                      my2_iLAMDA_x(de, qr, iNR, icexr9, MY2_thrd));
        iLAMr2 = iLAMr * iLAMr;
        iLAMr3 = iLAMr2 * iLAMr;
        iLAMr4 = iLAMr2 * iLAMr2;
        iLAMr5 = iLAMr4 * iLAMr;
        vr0 = (Dr > 40.0e-6f)
              ? gam * ckQr1 * pow(iLAMr, MY2_bfr) : 0.0f;
    } else {
        iLAMr = 0.0f; Dr = 0.0f; vr0 = 0.0f;
        iLAMr2 = 0.0f; iLAMr3 = 0.0f; iLAMr4 = 0.0f; iLAMr5 = 0.0f;
    }

    // Ice (:1683-1700)
    float Di, iLAMi, iLAMi2, iLAMi3, iLAMi4, iLAMi5;
    float iLAMiB0, iLAMiB1, iLAMiB2, vi0;
    if (qi > MY2_epsQ) {
        iQI = 1.0f / qi;
        iNY = 1.0f / ny_;
        iLAMi = max(MY2_iLAMmin2,
                      my2_iLAMDA_x(de, qi, iNY, icexi9, MY2_thrd));
        iLAMi2 = iLAMi * iLAMi;
        iLAMi3 = iLAMi2 * iLAMi;
        iLAMi4 = iLAMi2 * iLAMi2;
        iLAMi5 = iLAMi4 * iLAMi;
        iLAMiB0 = pow(iLAMi, MY2_bfi);
        iLAMiB1 = pow(iLAMi, MY2_bfi + 1.0f);
        iLAMiB2 = pow(iLAMi, MY2_bfi + 2.0f);
        vi0 = gam * ckQi1 * iLAMiB0;
        Di = my2_Dm_x(de, qi, iNY, icmi, MY2_thrd);
    } else {
        iLAMi = 0.0f; vi0 = 0.0f; Di = 0.0f;
        iLAMi2 = 0.0f; iLAMi3 = 0.0f; iLAMi4 = 0.0f; iLAMi5 = 0.0f;
        iLAMiB0 = 0.0f; iLAMiB1 = 0.0f; iLAMiB2 = 0.0f;
    }
    (void)iLAMiB1; (void)iLAMiB2;

    // Snow (:1703-1736)
    float Ds, iLAMs, iLAMs2, iLAMs_D3, iLAMsB0, iLAMsB1, iLAMsB2, vs0;
    float des, No_s = 0.0f, VENTs = 0.0f;
    if (qn > MY2_epsQ) {
        iQN = 1.0f / qn;
        iNN = 1.0f / nn;
        iLAMs = max(MY2_iLAMmin2,
                      my2_iLAMDA_x(de, qn, iNN, iGS20, idms));
        iLAMs_D3 = max(MY2_iLAMmin2,
                         my2_iLAMDA_x(de, qn, iNN, iGS20_D3, MY2_thrd));
        iLAMs2 = iLAMs * iLAMs;
        iLAMsB0 = pow(iLAMs, MY2_bfs);
        iLAMsB1 = pow(iLAMs, MY2_bfs + 1.0f);
        iLAMsB2 = pow(iLAMs, MY2_bfs + 2.0f);
        vs0 = gam * ckQs1 * iLAMsB0;
        Ds = min(MY2_DsMax, my2_Dm_x(de, qn, iNN, icms, idms));
        // snowSpherical = .false. (:1174) -> :1717
        des = my2_des_OF_Ds(Ds, MY2_desMax, eds, fds);
        No_s = nn * iGS31 / iLAMs_D3;                       // :1728
        VENTs = MY2_Avx * GS32 * (iLAMs_D3 * iLAMs_D3)
                + MY2_Bvx * ScTHRD * sqrt(gam * MY2_afs * iMUkin) * GS09
                  * pow(iLAMs_D3, cexs1);                // :1729
    } else {
        iLAMs = 0.0f; vs0 = 0.0f; Ds = 0.0f; iLAMs2 = 0.0f;
        iLAMs_D3 = 0.0f;
        // DIVERGENCE 3: the Fortran :1733 writes iLAMsB1 twice and never
        // zeroes iLAMsB2.  Every iLAMsB2 read is inside the Qs>epsQ arm
        // that also writes it, so the stale value is unreachable; the
        // defined behaviour is implemented.
        iLAMsB0 = 0.0f; iLAMsB1 = 0.0f; iLAMsB2 = 0.0f;
        des = MY2_desFix;                                   // :1734
    }
    float ides = 1.0f / des;                                // :1736
    (void)ides;

    // Graupel (:1740-1755)
    float Dg, iLAMg, iLAMg2, iLAMgB0, iLAMgB1, iLAMgB2, vg0, No_g;
    if (qg > MY2_epsQ) {
        iQG = 1.0f / qg;
        iNG = 1.0f / ng;
        iLAMg = max(MY2_iLAMmin1,
                      my2_iLAMDA_x(de, qg, iNG, iGG99, MY2_thrd));
        iLAMg2 = iLAMg * iLAMg;
        iLAMgB0 = pow(iLAMg, MY2_bfg);
        iLAMgB1 = pow(iLAMg, MY2_bfg + 1.0f);
        iLAMgB2 = pow(iLAMg, MY2_bfg + 2.0f);
        No_g = ng * iGG31 / iLAMg;                          // :1749
        vg0 = gam * ckQg1 * iLAMgB0;
        Dg = my2_Dm_x(de, qg, iNG, icmg, MY2_thrd);
    } else {
        iLAMg = 0.0f; vg0 = 0.0f; Dg = 0.0f; No_g = 0.0f;
        iLAMg2 = 0.0f; iLAMgB0 = 0.0f; iLAMgB1 = 0.0f; iLAMgB2 = 0.0f;
    }

    // Hail (:1758-1772)
    float Dh, iLAMh, iLAMh2, iLAMhB0, iLAMhB1, iLAMhB2, vh0, No_h;
    if (qh > MY2_epsQ) {
        iQH = 1.0f / qh;
        iNH = 1.0f / nh;
        iLAMh = max(MY2_iLAMmin1,
                      my2_iLAMDA_x(de, qh, iNH, iGH99, MY2_thrd));
        iLAMh2 = iLAMh * iLAMh;
        iLAMhB0 = pow(iLAMh, MY2_bfh);
        iLAMhB1 = pow(iLAMh, MY2_bfh + 1.0f);
        iLAMhB2 = pow(iLAMh, MY2_bfh + 2.0f);
        No_h = nh * iGH31 / pow(iLAMh, 1.0f + MY2_alpha_h);   // :1766
        vh0 = gam * ckQh1 * iLAMhB0;
        Dh = my2_Dm_x(de, qh, iNH, icmh, MY2_thrd);
    } else {
        iLAMh = 0.0f; vh0 = 0.0f; Dh = 0.0f; No_h = 0.0f;
        iLAMh2 = 0.0f; iLAMhB0 = 0.0f; iLAMhB1 = 0.0f; iLAMhB2 = 0.0f;
    }
    (void)iQC; (void)iQH; (void)iNH;

    // :1778-1794 -- initialise every source/sink term
    float QNUvi = 0.f, QVDvi = 0.f, QVDvs = 0.f, QVDvg = 0.f, QVDvh = 0.f;
    float QCLcs = 0.f, QCLcg = 0.f, QCLch = 0.f, QFZci = 0.f, QCLri = 0.f;
    float QMLsr = 0.f, QCLrs = 0.f, QCLrg = 0.f, QMLgr = 0.f, QCLrh = 0.f;
    float QMLhr = 0.f, QFZrh = 0.f, QMLir = 0.f, QCLsr = 0.f, QCLsh = 0.f;
    float QCLgr = 0.f, QCNgh = 0.f, QCNis = 0.f, QCLir = 0.f, QCLis = 0.f;
    float QCLih = 0.f, QIMsi = 0.f, QIMgi = 0.f, QCNsg = 0.f, QHwet = 0.f;
    float QCLig = 0.f, QSHhr = 0.f;

    float NCLcs = 0.f, NCLcg = 0.f, NCLch = 0.f, NFZci = 0.f, NMLhr = 0.f;
    float NhCNgh = 0.f, NCLri = 0.f, NCLrs = 0.f, NCLrg = 0.f, NCLrh = 0.f;
    float NMLsr = 0.f, NMLgr = 0.f, NMLir = 0.f, NSHhr = 0.f, NNUvi = 0.f;
    float NVDvi = 0.f, NVDvh = 0.f, NCLir = 0.f, NCLis = 0.f, NCLig = 0.f;
    float NCLih = 0.f, NIMsi = 0.f, NIMgi = 0.f, NiCNis = 0.f;
    float NsCNis = 0.f, NVDvs = 0.f, NCNsg = 0.f, NCLgr = 0.f;
    float NCLsrh = 0.f, NCLss = 0.f, NCLsr = 0.f, NCLsh = 0.f;
    float NCLsrs = 0.f, NCLgrg = 0.f, NgCNgh = 0.f, NVDvg = 0.f;
    float NCLirg = 0.f, NCLsrg = 0.f, NCLgrh = 0.f, NrFZrh = 0.f;
    float NhFZrh = 0.f, NCLirh = 0.f;
    float QCNis1 = 0.f, QCNis2 = 0.f;
    (void)QCNis1; (void)QCNis2;

    float Dirg = 0.f, Dirh = 0.f, Dsrs = 0.f, Dsrg = 0.f, Dsrh = 0.f;
    float Dgrg = 0.f, Dgrh = 0.f;                            // :1794

    float Si = q / qsi;                                      // :1798
    float iABi = 1.0f / (CHLS * CHLS / (Ka * MY2_RGASV * t * t)
                         + 1.0f / (de * qsi * Cdiff));       // :1799
    float VDmax;

    // ---- Collection by SNOW (:1805-1866) ----
    if (qn > MY2_epsQ) {
        if (qc > MY2_epsQ) {
            float Ecs = min(Dc, 30.0e-6f) * 3.333e+4f
                        * sqrt(min(Ds, 1.0e-3f) * 1.0e+3f);
            QCLcs = dt * gam * MY2_afs * cmr * Ecs * PIov4 * ide
                    * (nc * nn) * iGC5 * iGS31
                    * (GC13 * GS13 * iLAMc3 * iLAMsB2
                       + 2.0f * GC14 * GS12 * iLAMc4 * iLAMsB1
                       + GC15 * GS11 * iLAMc5 * iLAMsB0);
            NCLcs = dt * gam * MY2_afs * PIov4 * Ecs * (nc * nn)
                    * iGC5 * iGS31
                    * (GC5 * GS13 * iLAMsB2
                       + 2.0f * GC11 * GS12 * iLAMc * iLAMsB1
                       + GC12 * GS11 * iLAMc2 * iLAMsB0);
            // snowSpherical = .false. -> :1826-1830
            tmp1 = 0.6366f;
            QCLcs = tmp1 * QCLcs;
            NCLcs = tmp1 * NCLcs;
            QCLcs = min(QCLcs, qc);
            NCLcs = min(NCLcs, nc);
        } else {
            QCLcs = 0.0f; NCLcs = 0.0f;
        }
        if (qi > MY2_epsQ) {
            tmp1 = vs0 - vi0;
            tmp3 = sqrt(tmp1 * tmp1 + 0.04f * vs0 * vi0);
            QCLis = dt * cmi * ide * MY2_PI * 6.0f * Eis * (ny_ * nn) * tmp3
                    * iGI31 * iGS31
                    * (0.5f * iLAMs2 * iLAMi3 + 2.0f * iLAMs * iLAMi4
                       + 5.0f * iLAMi5);
            NCLis = dt * PIov4 * Eis * (ny_ * nn) * GI31 * GS31 * tmp3
                    * (GI33 * GS31 * iLAMi2
                       + 2.0f * GI32 * GS32 * iLAMi * iLAMs
                       + GI31 * GS33 * iLAMs2);
            QCLis = min(QCLis, qi);
            NCLis = min(QCLis * (ny_ * iQI), NCLis);
        } else {
            QCLis = 0.0f; NCLis = 0.0f;
        }
        // snow self-collection / aggregation (:1856-1862)
        NCLss = dt * 0.93952f * Ess
                * pow(de * qn, (2.0f + MY2_bfs) * MY2_thrd)
                * pow(nn, (4.0f - MY2_bfs) * MY2_thrd);
        NCLss = min(NCLss, 0.5f * nn);
    } else {
        QCLcs = 0.0f; NCLcs = 0.0f; QCLis = 0.0f; NCLis = 0.0f;
        NCLss = 0.0f;
    }

    // ---- Collection by GRAUPEL (:1869-1928) ----
    float VENTg = 0.0f;
    if (qg > MY2_epsQ) {
        if (qc > MY2_epsQ) {
            float Kstoke = MY2_dew * vg0 * Dc * Dc / (9.0f * MUdyn * Dg);
            Kstoke = max(1.5f, min(10.0f, Kstoke));
            float Ecg = 0.55f * log10(2.51f * Kstoke);
            QCLcg = dt * gam * MY2_afg * cmr * Ecg * PIov4 * ide
                    * (nc * ng) * iGC5 * iGG31
                    * (GC13 * GG13 * iLAMc3 * iLAMgB2
                       + 2.0f * GC14 * GG12 * iLAMc4 * iLAMgB1
                       + GC15 * GG11 * iLAMc5 * iLAMgB0);
            NCLcg = dt * gam * MY2_afg * PIov4 * Ecg * (nc * ng)
                    * iGC5 * iGG31
                    * (GC5 * GG13 * iLAMgB2
                       + 2.0f * GC11 * GG12 * iLAMc * iLAMgB1
                       + GC12 * GG11 * iLAMc2 * iLAMgB0);
            QCLcg = min(QCLcg, qc);
            NCLcg = min(NCLcg, nc);
        } else {
            QCLcg = 0.0f; NCLcg = 0.0f;
        }
        if (qi > MY2_epsQ) {
            tmp1 = vg0 - vi0;
            tmp3 = sqrt(tmp1 * tmp1 + 0.04f * vg0 * vi0);
            QCLig = dt * cmi * ide * MY2_PI * 6.0f * Eig * (ny_ * ng) * tmp3
                    * iGI31 * iGG31
                    * (0.5f * iLAMg2 * iLAMi3 + 2.0f * iLAMg * iLAMi4
                       + 5.0f * iLAMi5);
            NCLig = dt * PIov4 * Eig * (ny_ * ng) * GI31 * GG31 * tmp3
                    * (GI33 * GG31 * iLAMi2
                       + 2.0f * GI32 * GG32 * iLAMi * iLAMg
                       + GI31 * GG33 * iLAMg2);
            QCLig = min(QCLig, qi);
            NCLig = min(QCLig * (ny_ * iQI), NCLig);
        } else {
            QCLig = 0.0f; NCLig = 0.0f;
        }
        VENTg = MY2_Avx * GG32 * iLAMg * iLAMg
                + MY2_Bvx * ScTHRD * sqrt(gam * MY2_afg * iMUkin) * GG09
                  * pow(iLAMg, 2.5f + 0.5f * MY2_bfg + MY2_alpha_g);
        QVDvg = dt * ide * iABi * (PI2 * (Si - 1.0f) * No_g * VENTg);
        VDmax = (q - qsi) / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
        if (Si >= 1.0f) {
            QVDvg = min(max(QVDvg, 0.0f), VDmax);
        } else {
            if (VDmax < 0.0f) QVDvg = max(QVDvg, VDmax);
        }
        NVDvg = 0.0f;                                        // :1923
    } else {
        QCLcg = 0.0f; QCLrg = 0.0f; QCLig = 0.0f;
        NCLcg = 0.0f; NCLrg = 0.0f; NCLig = 0.0f;
    }

    // ---- Collection by HAIL (:1931-2032) ----
    float VENTh = 0.0f;
    if (qh > MY2_epsQ) {
        if (qc > MY2_epsQ) {
            float Ech = exp(-8.68e-7f * pow(Dc, -1.6f) * Dh);
            QCLch = dt * gam * MY2_afh * cmr * Ech * PIov4 * ide
                    * (nc * nh) * iGC5 * iGH31
                    * (GC13 * GH13 * iLAMc3 * iLAMhB2
                       + 2.0f * GC14 * GH12 * iLAMc4 * iLAMhB1
                       + GC15 * GH11 * iLAMc5 * iLAMhB0);
            NCLch = dt * gam * MY2_afh * PIov4 * Ech * (nc * nh)
                    * iGC5 * iGH31
                    * (GC5 * GH13 * iLAMhB2
                       + 2.0f * GC11 * GH12 * iLAMc * iLAMhB1
                       + GC12 * GH11 * iLAMc2 * iLAMhB0);
            QCLch = min(QCLch, qc);
            NCLch = min(NCLch, nc);
        } else {
            QCLch = 0.0f; NCLch = 0.0f;
        }
        if (qr > MY2_epsQ) {
            tmp1 = vh0 - vr0;
            tmp3 = sqrt(tmp1 * tmp1 + 0.04f * vh0 * vr0);
            QCLrh = dt * cmr * MY2_Erh * PIov4 * ide * (nh * nr)
                    * iGR31 * iGH31 * tmp3
                    * (GR36 * GH31 * iLAMr5
                       + 2.0f * GR35 * GH32 * iLAMr4 * iLAMh
                       + GR34 * GH33 * iLAMr3 * iLAMh2);
            NCLrh = dt * PIov4 * MY2_Erh * (nh * nr) * iGR31 * iGH31 * tmp3
                    * (GR33 * GH31 * iLAMr2
                       + 2.0f * GR32 * GH32 * iLAMr * iLAMh
                       + GR31 * GH33 * iLAMh2);
            QCLrh = min(QCLrh, qr);
            NCLrh = min(NCLrh, QCLrh * (nr * iQR));
        } else {
            QCLrh = 0.0f; NCLrh = 0.0f;
        }
        if (qi > MY2_epsQ) {
            tmp1 = vh0 - vi0;
            tmp3 = sqrt(tmp1 * tmp1 + 0.04f * vh0 * vi0);
            QCLih = dt * cmi * ide * MY2_PI * 6.0f * Eih * (ny_ * nh) * tmp3
                    * iGI31 * iGH31
                    * (0.5f * iLAMh2 * iLAMi3 + 2.0f * iLAMh * iLAMi4
                       + 5.0f * iLAMi5);
            NCLih = dt * PIov4 * Eih * (ny_ * nh) * GI31 * GH31 * tmp3
                    * (GI33 * GH31 * iLAMi2
                       + 2.0f * GI32 * GH32 * iLAMi * iLAMh
                       + GI31 * GH33 * iLAMh2);
            QCLih = min(QCLih, qi);
            NCLih = min(QCLih * (ny_ * iQI), NCLih);
        } else {
            QCLih = 0.0f; NCLih = 0.0f;
        }
        if (qn > MY2_epsQ) {
            tmp1 = vh0 - vs0;
            tmp3 = sqrt(tmp1 * tmp1 + 0.04f * vh0 * vs0);
            tmp4 = iLAMs2 * iLAMs2;
            (void)tmp4;
            // snowSpherical = .false. -> the dms=2 form (:1994-1997)
            QCLsh = dt * cms * ide * MY2_PI * 0.25f * Esh * tmp3 * nn * nh
                    * iGS31 * iGH31
                    * (GH33 * GS33 * (iLAMh * iLAMh) * (iLAMs * iLAMs)
                       + 2.0f * GH32 * GS34 * iLAMh
                         * (iLAMs * iLAMs * iLAMs)
                       + GH31 * GS35 * (iLAMs * iLAMs * iLAMs * iLAMs));
            NCLsh = dt * PIov4 * Esh * (nn * nh) * GS31 * GH31 * tmp3
                    * (GS33 * GH31 * iLAMs2
                       + 2.0f * GS32 * GH32 * iLAMs * iLAMh
                       + GS31 * GH33 * iLAMh2);
            QCLsh = min(QCLsh, qn);
            NCLsh = min(min((nn * iQN) * QCLsh, NCLsh), nn);
        } else {
            QCLsh = 0.0f; NCLsh = 0.0f;
        }
        VENTh = MY2_Avx * GH32 * pow(iLAMh, 2.0f + MY2_alpha_h)
                + MY2_Bvx * ScTHRD * sqrt(gam * MY2_afh * iMUkin) * GH09
                  * pow(iLAMh, 2.5f + 0.5f * MY2_bfh + MY2_alpha_h);
        QHwet = max(0.0f,
                      dt * PI2 * (de * MY2_CHLC * Cdiff * DELqvs - Ka * Tc)
                      * No_h * ide / (MY2_CHLF + MY2_CPW * Tc) * VENTh
                      + (QCLih * iEih + QCLsh * iEsh)
                        * (1.0f - MY2_CPI * Tc / (MY2_CHLF + MY2_CPW * Tc)));
        QVDvh = dt * ide * iABi * (PI2 * (Si - 1.0f) * No_h * VENTh);
        VDmax = (q - qsi) / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
        if (Si >= 1.0f) {
            QVDvh = min(max(QVDvh, 0.0f), VDmax);
        } else {
            if (VDmax < 0.0f) QVDvh = max(QVDvh, VDmax);
        }
        NVDvh = 0.0f;                                        // :2027
    } else {
        QCLch = 0.0f; QCLrh = 0.0f; QCLih = 0.0f; QCLsh = 0.0f;
        QHwet = 0.0f;
        NCLch = 0.0f; NCLrh = 0.0f; NCLsh = 0.0f; NCLih = 0.0f;
    }

    if (t > MY2_TRPL) {
        // ------------------------- T > To (:2034-2088) -------------------
        QMLir = qi;                                          // :2041
        qi = 0.0f;
        NMLir = ny_;
        if (qn > MY2_epsQ) {
            QMLsr = dt * (PI2 * ide * iCHLF * No_s * VENTs
                          * (Ka * Tc - MY2_CHLC * Cdiff * DELqvs)
                          + MY2_CPW * iCHLF * Tc * (QCLcs + QCLrs) / dt);
            QMLsr = min(max(QMLsr, 0.0f), qn);
            NMLsr = nn * iQN * QMLsr;
        } else {
            QMLsr = 0.0f; NMLsr = 0.0f;
        }
        if (qg > MY2_epsQ) {
            QMLgr = dt * (PI2 * ide * iCHLF * No_g * VENTg
                          * (Ka * Tc - MY2_CHLC * Cdiff * DELqvs)
                          + MY2_CPW * iCHLF * Tc * (QCLcg + QCLrg) / dt);
            QMLgr = min(max(QMLgr, 0.0f), qg);
            NMLgr = ng * iQG * QMLgr;
        } else {
            QMLgr = 0.0f; NMLgr = 0.0f;
        }
        if (qh > MY2_epsQ && Tc > 5.0f) {
            VENTh = MY2_Avx * GH32 * pow(iLAMh, 2.0f + MY2_alpha_h)
                    + MY2_Bvx * ScTHRD * sqrt(gam * MY2_afh * iMUkin) * GH09
                      * pow(iLAMh, 2.5f + 0.5f * MY2_bfh + MY2_alpha_h);
            QMLhr = dt * (PI2 * ide * iCHLF * No_h * VENTh
                          * (Ka * Tc - MY2_CHLC * Cdiff * DELqvs)
                          + MY2_CPW / MY2_CHLF * Tc * (QCLch + QCLrh) / dt);
            QMLhr = min(max(QMLhr, 0.0f), qh);
            NMLhr = nh * iQH * QMLhr;
            if (QCLrh > 0.0f) NMLhr = NMLhr * 0.1f;
        } else {
            QMLhr = 0.0f; NMLhr = 0.0f;
        }
        // :2079-2088 -- every cold term is reset
        QNUvi = 0.f; QFZci = 0.f; QVDvi = 0.f; QVDvs = 0.f;
        QCLis = 0.f; QCLri = 0.f;
        QCNgh = 0.f; QIMsi = 0.f; QIMgi = 0.f; QCLir = 0.f;
        QCLrs = 0.f; QCLgr = 0.f; QCLrg = 0.f; QCNis = 0.f;
        QCNsg = 0.f; QCLsr = 0.f;
        NNUvi = 0.f; NFZci = 0.f; NCLgr = 0.f; NCLrg = 0.f; NgCNgh = 0.f;
        NCLis = 0.f; NVDvi = 0.f; NVDvs = 0.f; NCLri = 0.f; NCLsr = 0.f;
        NCNsg = 0.f; NhCNgh = 0.f; NiCNis = 0.f; NsCNis = 0.f;
        NIMsi = 0.f; NIMgi = 0.f; NCLir = 0.f; NCLrs = 0.f;
    } else {
        // ------------------------- T < To (:2090-2505) -------------------
        QMLir = 0.f; QMLsr = 0.f; QMLgr = 0.f; QMLhr = 0.f;
        NMLir = 0.f; NMLsr = 0.f; NMLgr = 0.f; NMLhr = 0.f;

        // Probabilistic (Bigg) freezing of rain (:2099-2116)
        if (Tc < MY2_Tc_FZrh && qr > MY2_epsQ) {
            NrFZrh = -dt * MY2_Bbigg * (exp(MY2_Abigg * Tc) - 1.0f)
                     * de * qr * idew;
            float Rz = 1.0f;
            NhFZrh = Rz * NrFZrh;
            QFZrh = NrFZrh * (qr * iNR);
        } else {
            QFZrh = 0.0f; NrFZrh = 0.0f; NhFZrh = 0.0f;
        }

        // Homogeneous freezing of cloud to ice (:2122-2134)
        if (qc > MY2_epsQ) {
            tmp2 = Tc * Tc; tmp3 = tmp2 * Tc; tmp4 = tmp2 * tmp2;
            float JJ = pow(10.0f,
                              max(-20.0f, (-606.3952f - 52.6611f * Tc
                                             - 1.7439f * tmp2
                                             - 0.0265f * tmp3
                                             - 1.536e-4f * tmp4)));
            tmp1 = 1.0e6f * (de * (qc * iNC) * icmr);
            float FRAC = 1.0f - exp(-JJ * PIov6 * tmp1 * dt);
            if (Tc > -30.0f) FRAC = 0.0f;
            if (Tc < -50.0f) FRAC = 1.0f;
            QFZci = FRAC * qc;
            NFZci = FRAC * nc;
        } else {
            QFZci = 0.0f; NFZci = 0.0f;
        }

        // Primary ice nucleation, primIceNucl = 1 (:2136-2193)
        NNUvi = 0.0f; QNUvi = 0.0f;
        {
            float NuDEPSOR = 0.0f, NuCONT = 0.0f;
            float Simax;
            if (qsi > 1.0e-20f) {
                Simax = min(Si, MY2_satw_peak * qsw / qsi);
            } else {
                Simax = 0.0f;
            }
            tmp1 = t - 7.66f;
            float NNUmax = max(0.0f,
                                 rn_div(de, MY2_mio) * (q - qsi)
                                 / (1.0f + ck6 * (qsi / (tmp1 * tmp1))));
            if (Tc < -5.0f && Si > 1.0f) {
                NuDEPSOR = max(0.0f,
                                 1.0e3f * exp(12.96f * (Simax - 1.0f)
                                                 - 0.639f) - ny_);
            }
            if (qc > MY2_epsQ && Tc < -2.0f && wz > 0.001f) {
                float GG = 1.0f * idew
                           / (rn_div(MY2_RGASV * t / (rn_div((qsw * pr), MY2_EPS1)), Cdiff)
                              + MY2_CHLC / Ka / t
                                * (MY2_CHLC / MY2_RGASV / t - 1.0f));
                float Swmax = my2_SxFNC_w(wz, Tc, pr);
                float ssat;
                if (qsw > 1.0e-20f) {
                    ssat = min(q / qsw, Swmax) - 1.0f;
                } else {
                    ssat = 0.0f;
                }
                float Tcc = Tc + GG * ssat * MY2_CHLC / Kdiff;
                float Na = exp(4.11f - 0.262f * Tcc);
                float Kn = MY2_LAMa0 * t * MY2_p0 / (MY2_T0 * pr * MY2_Ra);
                float PSIa = -MY2_kBoltz * Tcc
                             / (6.0f * MY2_PI * MY2_Ra * MUdyn) * (1.0f + Kn);
                float ft = 0.4f * (1.0f + 1.45f * Kn
                                   + 0.4f * Kn * exp(-1.0f / Kn))
                           * (Ka + 2.5f * Kn * MY2_KAPa)
                           / (1.0f + 3.0f * Kn)
                           / (2.0f * Ka + 5.0f * MY2_KAPa * Kn + MY2_KAPa);
                // :2169 -- Dc is RECOMPUTED here without the icmr->thrd
                // helper and overwrites the size-distribution Dc above.
                Dc = pow(de * (qc * iNC) * icmr, MY2_thrd);
                float F1 = PI2 * Dc * Na * nc;
                float F2 = Ka / pr * (Tc - Tcc);
                float NuCONTA = rn_div(-F1 * F2 * MY2_RGASV * t, MY2_CHLC) * ide;
                float NuCONTB = F1 * F2 * ft * ide;
                float NuCONTC = F1 * PSIa;
                NuCONT = max(0.0f, (NuCONTA + NuCONTB + NuCONTC) * dt);
            }
            NNUvi = min(NNUmax, NuDEPSOR + NuCONT);        // icephase_ON
            QNUvi = MY2_mio * ide * NNUvi;
            QNUvi = min(QNUvi, q);
        }

        // ------------------------------ ICE (:2197-2300) -----------------
        if (qi > MY2_epsQ) {
            float No_i = ny_ * iGI31 / iLAMi;                // :2203
            float VENTi = MY2_Avx * GI32 * iLAMi * iLAMi
                          + MY2_Bvx * ScTHRD * sqrt(gam * MY2_afi * iMUkin)
                            * GI6 * pow(iLAMi, 2.5f + 0.5f * MY2_bfi
                                                  + MY2_alpha_i);
            QVDvi = dt * ide * MY2_capFact_i * iABi
                    * (PI2 * (Si - 1.0f) * No_i * VENTi);
            VDmax = (q - qsi)
                    / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
            if (Si >= 1.0f) {
                QVDvi = min(max(QVDvi, 0.0f), VDmax);
            } else {
                if (VDmax < 0.0f) QVDvi = max(QVDvi, VDmax);
            }
            NVDvi = min(0.0f, (ny_ * iQI) * QVDvi);        // :2219

            // Conversion to snow (:2222-2240)
            if (qi + QVDvi > MY2_epsQ && ny_ + NVDvi > MY2_epsN) {
                tmp5 = iLAMi;
                tmp6 = No_i;
                tmp1 = qi + QVDvi;
                tmp2 = ny_ + NVDvi;
                tmp3 = 1.0f / tmp2;
                float iLAMi_h = max(MY2_iLAMmin2,
                                      my2_iLAMDA_x(de, tmp1, tmp3, icexi9,
                                                   MY2_thrd));
                float No_i_h = tmp2 * iGI31 / iLAMi_h;
                tmp4 = exp(-MY2_Dso / iLAMi_h);
                NiCNis = No_i_h * iLAMi_h * tmp4;
                NsCNis = NiCNis;
                QCNis = cmi * No_i_h * tmp4
                        * (MY2_Dso * MY2_Dso * MY2_Dso * iLAMi_h
                           + 3.0f * MY2_Dso * MY2_Dso * iLAMi_h * iLAMi_h
                           + 6.0f * MY2_Dso * iLAMi_h * iLAMi_h * iLAMi_h
                           + 6.0f * iLAMi_h * iLAMi_h * iLAMi_h * iLAMi_h);
                iLAMi = tmp5;
                No_i = tmp6;
                (void)No_i;
            }

            // 3-component freezing, ice + rain (:2249-2282)
            if (qr > MY2_epsQ && qi > MY2_epsQ) {
                tmp1 = vr0 - vi0;
                tmp3 = sqrt(tmp1 * tmp1 + 0.04f * vr0 * vi0);
                QCLir = dt * cmi * MY2_Eri * PIov4 * ide * (nr * ny_)
                        * iGI31 * iGR31 * tmp3
                        * (GI36 * GR31 * iLAMi5
                           + 2.0f * GI35 * GR32 * iLAMi4 * iLAMr
                           + GI34 * GR33 * iLAMi3 * iLAMr2);
                NCLri = dt * PIov4 * MY2_Eri * (nr * ny_) * iGI31 * iGR31
                        * tmp3
                        * (GI33 * GR31 * iLAMi2
                           + 2.0f * GI32 * GR32 * iLAMi * iLAMr
                           + GI31 * GR33 * iLAMr2);
                QCLri = dt * cmr * MY2_Eri * PIov4 * ide * (ny_ * nr)
                        * iGR31 * iGI31 * tmp3
                        * (GR36 * GI31 * iLAMr5
                           + 2.0f * GR35 * GI32 * iLAMr4 * iLAMi
                           + GR34 * GI33 * iLAMr3 * iLAMi2);
                NCLir = min(QCLir * (ny_ * iQI), NCLri);
                QCLri = min(QCLri, qr);
                QCLir = min(QCLir, qi);
                NCLri = min(NCLri, nr);
                NCLir = min(NCLir, ny_);
                tmp1 = max(Di, Dr);
                float dey = (MY2_dei * Di * Di * Di + MY2_dew * Dr * Dr * Dr)
                            / (tmp1 * tmp1 * tmp1);
                if (dey > 0.5f * (MY2_deg + MY2_deh)
                        && Dr > MY2_Dr_3cmpThrs) {
                    Dirg = 0.0f; Dirh = 1.0f;
                } else {
                    Dirg = 1.0f; Dirh = 0.0f;
                }
            } else {
                QCLir = 0.0f; NCLir = 0.0f; QCLri = 0.0f;
                NCLri = 0.0f; Dirh = 0.0f; Dirg = 0.0f;
            }

            // Rime-splintering (:2285-2291)
            float ff = 0.0f;
            if (Tc >= -8.0f && Tc <= -5.0f) ff = 3.5e8f * (Tc + 8.0f)
                                                 * MY2_thrd;
            if (Tc > -5.0f && Tc < -3.0f) ff = 3.5e8f * (-3.0f - Tc) * 0.5f;
            NIMsi = de * ff * QCLcs;
            NIMgi = de * ff * QCLcg;
            QIMsi = MY2_mio * ide * NIMsi;
            QIMgi = MY2_mio * ide * NIMgi;
        } else {
            QVDvi = 0.f; QCNis = 0.f;
            QIMsi = 0.f; QIMgi = 0.f; QCLri = 0.f; QCLir = 0.f;
            NVDvi = 0.f; NCLir = 0.f; NIMsi = 0.f;
            NiCNis = 0.f; NsCNis = 0.f; NIMgi = 0.f; NCLri = 0.f;
        }

        // ----------------------------- SNOW (:2304-2394) -----------------
        if (qn > MY2_epsQ) {
            QVDvs = dt * ide * MY2_capFact_s * iABi
                    * (PI2 * (Si - 1.0f) * No_s * VENTs
                       - CHLS * MY2_CHLF / (Ka * MY2_RGASV * t * t)
                         * QCLcs / dt);
            VDmax = (q - qsi)
                    / (1.0f + ck6 * qsi / ((t - 7.66f) * (t - 7.66f)));
            if (Si >= 1.0f) {
                QVDvs = min(max(QVDvs, 0.0f), VDmax);
            } else {
                if (VDmax < 0.0f) QVDvs = max(QVDvs, VDmax);
            }
            NVDvs = -min(0.0f, (nn * iQN) * QVDvs);

            // Conversion to graupel (:2325-2338)
            if (QCLcs > 0.0f && QCLcs > MY2_CNsgThres * QVDvs) {
                tmp1 = 100.0f;
                QCNsg = min(qn + QCLcs, QCLcs * (tmp1 * QCLcs / qn));
                NCNsg = de * QCNsg / (qn + QCLcs);
            } else {
                QCNsg = 0.0f;
                NCNsg = 0.0f;
            }

            // 3-component freezing, snow + rain (:2341-2387)
            if (qr > MY2_epsQ && qn > MY2_epsQ && Tc < -5.0f) {
                tmp1 = vs0 - vr0;
                tmp2 = sqrt(tmp1 * tmp1 + 0.04f * vs0 * vr0);
                tmp6 = iLAMs2 * iLAMs2 * iLAMs;
                (void)tmp6;
                QCLrs = dt * cmr * MY2_Ers * PIov4 * ide * nn * nr
                        * iGR31 * iGS31 * tmp2
                        * (GR36 * GS31 * iLAMr5
                           + 2.0f * GR35 * GS32 * iLAMr4 * iLAMs
                           + GR34 * GS33 * iLAMr3 * iLAMs2);
                NCLrs = dt * 0.25e0f * MY2_PI * MY2_Ers * (nn * nr)
                        * iGR31 * iGS31 * tmp2
                        * (GR33 * GS31 * iLAMr2
                           + 2.0f * GR32 * GS32 * iLAMr * iLAMs
                           + GR31 * GS33 * iLAMs2);
                // snowSpherical = .false. -> the dms=2 form (:2360-2362)
                QCLsr = dt * cms * ide * MY2_PI * 0.25f * MY2_Ers * tmp2
                        * nn * nr * iGS31 * iGR31
                        * (GR33 * GS33 * (iLAMr * iLAMr) * (iLAMs * iLAMs)
                           + 2.0f * GR32 * GS34 * iLAMr
                             * (iLAMs * iLAMs * iLAMs)
                           + GR31 * GS35 * (iLAMs * iLAMs * iLAMs * iLAMs));
                NCLsr = min(QCLsr * (nn * iQN), NCLrs);
                QCLrs = min(QCLrs, qr);
                QCLsr = min(QCLsr, qn);
                NCLrs = min(NCLrs, nr);
                NCLsr = min(NCLsr, nn);
                Dsrs = 0.0f; Dsrg = 0.0f; Dsrh = 0.0f;
                tmp1 = max(Ds, Dr);
                tmp2 = tmp1 * tmp1 * tmp1;
                float dey = (des * Ds * Ds * Ds + MY2_dew * Dr * Dr * Dr)
                            / tmp2;
                if (dey <= 0.5f * (des + MY2_deg)) Dsrs = 1.0f;
                if (dey > 0.5f * (des + MY2_deg)
                        && dey < 0.5f * (MY2_deg + MY2_deh)) Dsrg = 1.0f;
                if (dey >= 0.5f * (MY2_deg + MY2_deh)) {
                    Dsrh = 1.0f;
                    if (Dr < MY2_Dr_3cmpThrs) {
                        Dsrg = 1.0f; Dsrh = 0.0f;
                    }
                }
            } else {
                QCLrs = 0.f; QCLsr = 0.f; NCLrs = 0.f; NCLsr = 0.f;
            }
        } else {
            QVDvs = 0.f; QCLcs = 0.f; QCNsg = 0.f; QCLsr = 0.f;
            QCLrs = 0.f;
            NVDvs = 0.f; NCLcs = 0.f; NCLsr = 0.f; NCLrs = 0.f;
            NCNsg = 0.f;
        }

        // --------------------------- GRAUPEL (:2398-2482) ----------------
        if (qg > MY2_epsQ && ng > MY2_epsN) {
            if ((QCLcg + QCLrg) > 0.0f) {
                tmp1 = 1.1e4f * de * (qc + qr) + 1.0f;
                tmp1 = max(1.0f, tmp1);
                float D_sll = 2.0f * 0.01f
                              * (exp(min(20.0f, -Tc / tmp1)) - 1.0f);
                D_sll = min(1.0f, max(0.0001f, D_sll));
                tmp1 = iLAMg;
                tmp2 = No_g;
                tmp3 = qg + QCLcg + QCLrg;
                float iLAMg_h = exp(MY2_thrd
                                       * log(de * tmp3
                                                / (ng * 6.0f * cmg)));
                float No_g_h = ng / iLAMg_h;
                tmp4 = exp(-D_sll / iLAMg_h);
                float Ng_tail = No_g_h * iLAMg_h * tmp4;
                if (Ng_tail > MY2_Ngh_crit) {
                    NgCNgh = min(ng, Ng_tail);
                    QCNgh = min(qg,
                                  cmg * No_g_h * tmp4
                                  * (D_sll * D_sll * D_sll * iLAMg_h
                                     + 3.0f * D_sll * D_sll * iLAMg_h
                                       * iLAMg_h
                                     + 6.0f * D_sll * iLAMg_h * iLAMg_h
                                       * iLAMg_h
                                     + 6.0f * iLAMg_h * iLAMg_h * iLAMg_h
                                       * iLAMg_h));
                    float Rz = 1.0f;
                    NhCNgh = Rz * NgCNgh;
                } else {
                    QCNgh = 0.0f; NgCNgh = 0.0f; NhCNgh = 0.0f;
                }
                iLAMg = tmp1;
                No_g = tmp2;
            }
            // 3-component freezing, graupel + rain (:2437-2475)
            if (qr > MY2_epsQ && Tc < -5.0f) {
                tmp1 = vg0 - vr0;
                tmp2 = sqrt(tmp1 * tmp1 + 0.04f * vg0 * vr0);
                tmp8 = iLAMg2 * iLAMg;
                tmp9 = tmp8 * iLAMg;
                tmp10 = tmp9 * iLAMg;
                float Kstoke = MY2_dew * fabs(vg0 - vr0) * Dr * Dr
                               / (9.0f * MUdyn * Dg);
                Kstoke = max(1.5f, min(10.0f, Kstoke));
                float Erg = 0.55f * log10(2.51f * Kstoke);
                QCLrg = dt * cmr * Erg * PIov4 * ide * (ng * nr)
                        * iGR31 * iGG31 * tmp2
                        * (GR36 * GG31 * iLAMr5
                           + 2.0f * GR35 * GG32 * iLAMr4 * iLAMg
                           + GR34 * GG33 * iLAMr3 * iLAMg2);
                NCLrg = dt * PIov4 * Erg * (ng * nr) * iGR31 * iGG31 * tmp2
                        * (GR33 * GG31 * iLAMr2
                           + 2.0f * GR32 * GG32 * iLAMr * iLAMg
                           + GR31 * GG33 * iLAMg2);
                QCLgr = dt * cmg * Erg * PIov4 * ide * (nr * ng)
                        * iGG31 * iGR31 * tmp2
                        * (GG36 * GR31 * tmp10
                           + 2.0f * GG35 * GR32 * tmp9 * iLAMr
                           + GG34 * GR33 * tmp8 * iLAMr2);
                NCLgr = min(NCLrg, QCLgr * (ng * iQG));
                QCLrg = min(QCLrg, qr);
                QCLgr = min(QCLgr, qg);
                NCLrg = min(NCLrg, nr);
                NCLgr = min(NCLgr, ng);
                tmp1 = max(Dg, Dr);
                tmp2 = tmp1 * tmp1 * tmp1;
                float dey = (MY2_deg * Dg * Dg * Dg
                             + MY2_dew * Dr * Dr * Dr) / tmp2;
                if (dey > 0.5f * (MY2_deg + MY2_deh)
                        && Dr > MY2_Dr_3cmpThrs) {
                    Dgrg = 0.0f; Dgrh = 1.0f;
                } else {
                    Dgrg = 1.0f; Dgrh = 0.0f;
                }
            } else {
                QCLgr = 0.f; QCLrg = 0.f; NCLgr = 0.f; NCLrg = 0.f;
            }
        } else {
            QCNgh = 0.f; QCLgr = 0.f; QCLrg = 0.f; NgCNgh = 0.f;
            NhCNgh = 0.f; NCLgr = 0.f; NCLrg = 0.f;
        }

        // ----------------------------- HAIL (:2486-2503) -----------------
        if (qh > MY2_epsQ) {
            if (QHwet < (QCLch + QCLrh + QCLih + QCLsh) && Tc > -40.0f) {
                QCLih = min(QCLih * iEih, qi);
                NCLih = min(NCLih * iEih, ny_);
                QCLsh = min(QCLsh * iEsh, qn);
                NCLsh = min(NCLsh * iEsh, nn);
                tmp3 = QCLrh;
                QCLrh = QHwet - (QCLch + QCLih + QCLsh);
                QSHhr = tmp3 - QCLrh;
                NSHhr = de * QSHhr
                        / (cmr * MY2_Drshed * MY2_Drshed * MY2_Drshed);
            } else {
                NSHhr = 0.0f;
            }
        } else {
            NSHhr = 0.0f;
        }
    }  // end T<To block

    // --- Prevent mass transfer from accretion during melting (:2513-2556) -
    tmp1 = nn + NsCNis - NVDvs - NCNsg - NMLsr - NCLss - NCLsr - NCLsh
           + NCLsrs;
    if (tmp1 < MY2_epsN) {
        QCLcs = 0.0f; NCLcs = 0.0f; QCLrs = 0.0f; NCLrs = 0.0f;
        if (Dsrs == 1.0f) { QCLrs = 0.0f; QCLsr = 0.0f; }
    }
    tmp2 = ng + NCNsg - NCLgr - NVDvg - NMLgr + NCLirg + NCLsrg + NCLgrg
           - NgCNgh;
    if (tmp2 < MY2_epsN) {
        QCLcg = 0.0f; NCLcg = 0.0f; QCLrg = 0.0f; NCLrg = 0.0f;
        if (Dirg == 1.0f) { QCLri = 0.0f; QCLir = 0.0f; }
        if (Dsrg == 1.0f) { QCLrs = 0.0f; QCLsr = 0.0f; }
    }
    tmp3 = nh + NhFZrh + NhCNgh - NMLhr - NVDvh + NCLirh + NCLsrh + NCLgrh;
    if (tmp3 < MY2_epsN) {
        QCLch = 0.0f; NCLch = 0.0f; QCLrh = 0.0f; NCLrh = 0.0f;
        if (Dirh == 1.0f) { QCLri = 0.0f; QCLir = 0.0f; }
        if (Dsrh == 1.0f) { QCLrs = 0.0f; QCLsr = 0.0f; }
    }

    // --- Overdepletion adjustment, two iterations (:2562-2671) -----------
    for (int niter = 0; niter < 2; ++niter) {
        float source, sink, sour, ratio;
        // (1) Vapour
        source = q + my2_dim(-QVDvi, 0.f) + my2_dim(-QVDvs, 0.f)
                 + my2_dim(-QVDvg, 0.f) + my2_dim(-QVDvh, 0.f);
        sink = QNUvi + my2_dim(QVDvi, 0.f) + my2_dim(QVDvs, 0.f);
        sour = max(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QNUvi = ratio * QNUvi; NNUvi = ratio * NNUvi;
            if (QVDvi > 0.f) { QVDvi = ratio * QVDvi; NVDvi = ratio * NVDvi; }
            if (QVDvs > 0.f) { QVDvs = ratio * QVDvs; NVDvs = ratio * NVDvs; }
            QVDvg = ratio * QVDvg; NVDvg = ratio * NVDvg;
            QVDvh = ratio * QVDvh; NVDvh = ratio * NVDvh;
        }
        // (2) Cloud
        source = qc;
        sink = QCLcs + QCLcg + QCLch + QFZci;
        sour = max(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QFZci = ratio * QFZci; NFZci = ratio * NFZci;
            QCLcs = ratio * QCLcs; NCLcs = ratio * NCLcs;
            QCLcg = ratio * QCLcg; NCLcg = ratio * NCLcg;
            QCLch = ratio * QCLch; NCLch = ratio * NCLch;
        }
        // (3) Rain
        source = qr + QMLsr + QMLgr + QMLhr + QMLir;
        sink = QCLri + QCLrs + QCLrg + QCLrh + QFZrh;
        sour = max(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QCLrg = ratio * QCLrg; QCLri = ratio * QCLri;
            NCLri = ratio * NCLri;
            QCLrs = ratio * QCLrs; NCLrs = ratio * NCLrs;
            NCLrg = ratio * NCLrg; QCLrh = ratio * QCLrh;
            NCLrh = ratio * NCLrh;
            QFZrh = ratio * QFZrh; NrFZrh = ratio * NrFZrh;
            NhFZrh = ratio * NhFZrh;
            if (ratio == 0.0f) {
                Dirg = 0.f; Dirh = 0.f; Dgrg = 0.f; Dgrh = 0.f;
                Dsrs = 0.f; Dsrg = 0.f; Dsrh = 0.f;
            }
        }
        // (4) Ice
        source = qi + QNUvi + my2_dim(QVDvi, 0.f) + QFZci;
        sink = QCNis + QCLir + my2_dim(-QVDvi, 0.f) + QCLis + QCLig + QCLih
               + QMLir;
        sour = max(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QMLir = ratio * QMLir; NMLir = ratio * NMLir;
            if (QVDvi < 0.f) { QVDvi = ratio * QVDvi; NVDvi = ratio * NVDvi; }
            QCNis = ratio * QCNis; NiCNis = ratio * NiCNis;
            NsCNis = ratio * NsCNis;
            QCLir = ratio * QCLir; NCLir = ratio * NCLir;
            QCLig = ratio * QCLig;
            QCLis = ratio * QCLis; NCLis = ratio * NCLis;
            QCLih = ratio * QCLih; NCLih = ratio * NCLih;
            if (ratio == 0.0f) { Dirg = 0.f; Dirh = 0.f; }
        }
        // (5) Snow
        source = qn + QCNis + my2_dim(QVDvs, 0.f) + QCLis
                 + Dsrs * (QCLrs + QCLsr) + QCLcs;
        sink = my2_dim(-QVDvs, 0.f) + QCNsg + QMLsr + QCLsr + QCLsh;
        sour = max(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            if (QVDvs <= 0.f) { QVDvs = ratio * QVDvs; NVDvs = ratio * NVDvs; }
            QCNsg = ratio * QCNsg; NCNsg = ratio * NCNsg;
            QMLsr = ratio * QMLsr;
            NMLsr = ratio * NMLsr; QCLsr = ratio * QCLsr;
            NCLsr = ratio * NCLsr;
            QCLsh = ratio * QCLsh; NCLsh = ratio * NCLsh;
            if (ratio == 0.0f) { Dsrs = 0.f; Dsrg = 0.f; Dsrh = 0.f; }
        }
        // (6) Graupel
        source = qg + QCNsg + my2_dim(QVDvg, 0.f) + Dirg * (QCLri + QCLir)
                 + Dgrg * (QCLrg + QCLgr) + QCLcg + Dsrg * (QCLrs + QCLsr)
                 + QCLig;
        sink = my2_dim(-QVDvg, 0.f) + QMLgr + QCNgh + QCLgr;
        sour = max(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QVDvg = ratio * QVDvg; NVDvg = ratio * NVDvg;
            QMLgr = ratio * QMLgr;
            NMLgr = ratio * NMLgr; QCNgh = ratio * QCNgh;
            NgCNgh = ratio * NgCNgh;
            QCLgr = ratio * QCLgr; NCLgr = ratio * NCLgr;
            NhCNgh = ratio * NhCNgh;
            if (ratio == 0.0f) { Dgrg = 0.f; Dgrh = 0.f; }
        }
        // (7) Hail
        source = qh + my2_dim(QVDvh, 0.f) + QCLch + QCLrh
                 + Dirh * (QCLri + QCLir) + QCLih + QCLsh
                 + Dsrh * (QCLrs + QCLsr) + QCNgh + Dgrh * (QCLrg + QCLgr)
                 + QFZrh;
        sink = my2_dim(-QVDvh, 0.f) + QMLhr;
        sour = max(source, 0.0f);
        if (sink > sour) {
            ratio = sour / sink;
            QVDvh = ratio * QVDvh; NVDvh = ratio * NVDvh;
            QMLhr = ratio * QMLhr; NMLhr = ratio * NMLhr;
        }
    }

    // --- N tendencies for 3-comp-freezing destinations (:2675-2698) ------
    NCLirg = 0.f; NCLirh = 0.f; NCLsrs = 0.f; NCLsrg = 0.f;
    NCLsrh = 0.f; NCLgrg = 0.f; NCLgrh = 0.f;
    if (QCLir + QCLri > 0.0f) {
        tmp1 = max(Dr, Di);
        tmp2 = tmp1 * tmp1 * tmp1 * PIov6;
        NCLirg = Dirg * de * (QCLir + QCLri) / (MY2_deg * tmp2);
        NCLirh = Dirh * de * (QCLir + QCLri) / (MY2_deh * tmp2);
    }
    if (QCLsr + QCLrs > 0.0f) {
        tmp1 = max(Dr, Ds);
        tmp2 = tmp1 * tmp1 * tmp1 * PIov6;
        NCLsrs = Dsrs * de * (QCLsr + QCLrs) / (des * tmp2);
        NCLsrg = Dsrg * de * (QCLsr + QCLrs) / (MY2_deg * tmp2);
        NCLsrh = Dsrh * de * (QCLsr + QCLrs) / (MY2_deh * tmp2);
    }
    if (QCLgr + QCLrg > 0.0f) {
        tmp1 = max(Dr, Dg);
        tmp2 = tmp1 * tmp1 * tmp1 * PIov6;
        NCLgrg = Dgrg * de * (QCLgr + QCLrg) / (MY2_deg * tmp2);
        NCLgrh = Dgrh * de * (QCLgr + QCLrg) / (MY2_deh * tmp2);
    }

    // --- Apply all source/sink terms (:2708-2731) ------------------------
    q = q - QNUvi - QVDvi - QVDvs - QVDvg - QVDvh;
    qc = qc - QCLcs - QCLcg - QCLch - QFZci;
    qr = qr - QCLri + QMLsr - QCLrs - QCLrg + QMLgr - QCLrh + QMLhr - QFZrh
         + QMLir;
    qi = qi + QNUvi + QVDvi + QFZci - QCNis - QCLir - QCLis - QCLig
         - QMLir - QCLih + QIMsi + QIMgi;
    qg = qg + QCNsg + QVDvg + QCLcg - QCLgr - QMLgr - QCNgh - QIMgi + QCLig
         + Dirg * (QCLri + QCLir) + Dgrg * (QCLrg + QCLgr)
         + Dsrg * (QCLrs + QCLsr);
    qn = qn + QCNis + QVDvs + QCLcs - QCNsg - QMLsr - QIMsi - QCLsr + QCLis
         - QCLsh + Dsrs * (QCLrs + QCLsr);
    qh = qh + Dirh * (QCLri + QCLir) - QMLhr + QVDvh + QCLch
         + Dsrh * (QCLrs + QCLsr) + QCLih + QCLsh + QFZrh + QCLrh + QCNgh
         + Dgrh * (QCLrg + QCLgr);

    nc = nc - NCLcs - NCLcg - NCLch - NFZci;
    nr = nr - NCLri - NCLrs - NCLrg - NCLrh + NMLsr + NMLgr + NMLhr - NrFZrh
         + NMLir + NSHhr;
    ny_ = ny_ + NNUvi + NVDvi + NFZci - NCLir - NCLis - NCLig - NCLih
          - NMLir + NIMsi + NIMgi - NiCNis;
    nn = nn + NsCNis - NVDvs - NCNsg - NMLsr - NCLss - NCLsr - NCLsh
         + NCLsrs;
    ng = ng + NCNsg - NCLgr - NVDvg - NMLgr + NCLirg + NCLsrg + NCLgrg
         - NgCNgh;
    nh = nh + NhFZrh + NhCNgh - NMLhr - NVDvh + NCLirh + NCLsrh + NCLgrh;

    t = t + LFP * (QCLri + QCLcs + QCLrs + QFZci - QMLsr + QCLcg + QCLrg
                   - QMLir - QMLgr - QMLhr + QCLch + QCLrh + QFZrh)
        + LSP * (QNUvi + QVDvi + QVDvs + QVDvg + QVDvh);

    // --- Ensure consistency between moments (:2733-2816) -----------------
    tmp1 = pr / (MY2_RGASD * t);       // :2734 updated air density
    if (qc > MY2_epsQ && nc < MY2_epsN) {
        nc = N_c_SM;
    } else if (qc <= MY2_epsQ) {
        q = q + qc; t = t - LCP * qc; qc = 0.0f; nc = 0.0f;
    }
    if (qr > MY2_epsQ && nr < MY2_epsN) {
        nr = pow(MY2_No_r_SM * GR31, 3.0f / (4.0f + MY2_alpha_r))
             * pow(GR31 * iGR34 * tmp1 * qr * icmr,
                      (1.0f + MY2_alpha_r) / (4.0f + MY2_alpha_r));
    } else if (qr <= MY2_epsQ) {
        q = q + qr; t = t - LCP * qr; qr = 0.0f; nr = 0.0f;
    }
    if (qi > MY2_epsQ && ny_ < MY2_epsN) {
        ny_ = max(2.0f * MY2_epsN, my2_N_Cooper(t));
    } else if (qi <= MY2_epsQ) {
        q = q + qi; t = t - LSP * qi; qi = 0.0f; ny_ = 0.0f;
    }
    if (qn > MY2_epsQ && nn < MY2_epsN) {
        float Nos = my2_Nos_Thompson(t);
        nn = pow(Nos * GS31, dms * icexs2)
             * pow(GS31 * iGS40 * icms * tmp1 * qn,
                      (1.0f + MY2_alpha_s) * icexs2);
    } else if (qn <= MY2_epsQ) {
        q = q + qn; t = t - LSP * qn; qn = 0.0f; nn = 0.0f;
    }
    if (qg > MY2_epsQ && ng < MY2_epsN) {
        ng = pow(MY2_No_g_SM * GG31, 3.0f / (4.0f + MY2_alpha_g))
             * pow(GG31 * iGG34 * tmp1 * qg * icmg,
                      (1.0f + MY2_alpha_g) / (4.0f + MY2_alpha_g));
    } else if (qg <= MY2_epsQ) {
        q = q + qg; t = t - LSP * qg; qg = 0.0f; ng = 0.0f;
    }
    if (qh > MY2_epsQ && nh < MY2_epsN) {
        nh = pow(MY2_No_h_SM * GH31, 3.0f / (4.0f + MY2_alpha_h))
             * pow(GH31 * iGH34 * tmp1 * qh * icmh,
                      (1.0f + MY2_alpha_h) / (4.0f + MY2_alpha_h));
    } else if (qh <= MY2_epsQ) {
        q = q + qh; t = t - LSP * qh; qh = 0.0f; nh = 0.0f;
    }
    if (qh > MY2_epsQ && nh > MY2_epsN) {
        // :2803-2813 -- transfer small hail to graupel.  NOTE the Fortran
        // uses DE(i,k) (the PART 1 density), not the tmp1 recomputed above.
        tmp1 = 1.0f / nh;
        float Dh_n = my2_Dm_x(de, qh, tmp1, icmh, MY2_thrd);
        if (Dh_n < MY2_Dh_min) {
            qg = qg + qh;
            ng = ng + nh;
            qh = 0.0f;
            nh = 0.0f;
        }
    }
    q = max(q, 0.0f);                        // :2815
    ny_ = min(ny_, MY2_Ni_max);              // :2816
    // DIVERGENCE 1: :2819-2823 prints and STOPs on T<173 or T>323 K.  A
    // device kernel cannot abort the run; gpuwm's step-level health gate
    // owns that decision, so the value is left as computed.

    T[gid] = t;   Q[gid] = q;
    QC[gid] = qc; QR[gid] = qr; QI[gid] = qi;
    QN[gid] = qn; QG[gid] = qg; QH[gid] = qh;
    NC[gid] = nc; NR[gid] = nr; NY[gid] = ny_;
    NN[gid] = nn; NG[gid] = ng; NH[gid] = nh;
}
