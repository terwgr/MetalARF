#include <metal_stdlib>
#include "myjsfc_family_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

#ifndef MYJSFC_KZTM
#define MYJSFC_KZTM 10001
#endif
#define MYJSFC_KZTM2 (MYJSFC_KZTM - 2)

// ---- shared model constants (share/module_model_constants.F) ----
#define MYJ_G        9.81f
#define MYJ_RD       287.0f
#define MYJ_RV       461.6f
#define MYJ_CP       (7.0f * 287.0f / 2.0f)
#define MYJ_XLV      2.5e6f
#define MYJ_P608     (461.6f / 287.0f - 1.0f)
#define MYJ_PQ0      379.90516f
#define MYJ_A2       17.2693882f
#define MYJ_A3       273.16f
#define MYJ_A4       35.86f
#define MYJ_EPSQ2    0.2f
#define MYJ_P1000MB  1.0e5f

// ---- module_sf_myjsfc.F:23-55 ----
#define SFC_ITRMX    5
#define SFC_VKARMAN  0.4f
#define SFC_CAPA     (MYJ_RD / MYJ_CP)
#define SFC_ELOCP    (2.72e6f / MYJ_CP)
#define SFC_RCAP     (1.0f / SFC_CAPA)
#define SFC_GOCP02   (MYJ_G / MYJ_CP * 2.0f)
#define SFC_GOCP10   (MYJ_G / MYJ_CP * 10.0f)
#define SFC_EPSU2    1.0e-6f
#define SFC_EPSUST   1.0e-9f
#define SFC_EPSZT    1.0e-28f
#define SFC_A2S      17.2693882f
#define SFC_A3S      273.16f
#define SFC_A4S      35.86f
#define SFC_SEAFC    0.98f
#define SFC_PQ0SEA   (MYJ_PQ0 * SFC_SEAFC)
#define SFC_EXCML    0.0001f
#define SFC_EXCMS    0.0001f
#define SFC_GLKBR    10.0f
#define SFC_GLKBS    30.0f
#define SFC_QVISC    2.1e-5f
#define SFC_RIC      0.505f
#define SFC_SMALL    0.35f
#define SFC_SQPR     0.84f
#define SFC_SQSC     0.84f
#define SFC_SQVISC   258.2f
#define SFC_TVISC    2.1e-5f
#define SFC_USTC     0.7f
#define SFC_USTR     0.225f
#define SFC_VISC     1.5e-5f
#define SFC_FH       1.01f
#define SFC_WWST     1.2f
#define SFC_ZTFC     1.0f
#define SFC_CZIV     (SFC_SMALL * SFC_GLKBS)
#define SFC_GRRS     (SFC_GLKBR / SFC_GLKBS)
#define SFC_RTVISC   (1.0f / SFC_TVISC)
#define SFC_RVISC    (1.0f / SFC_VISC)
#define SFC_ZQRZT    (SFC_SQSC / SFC_SQPR)
#define SFC_FZQ1     (SFC_RTVISC * SFC_QVISC * SFC_ZQRZT)
#define SFC_FZQ2     (SFC_RTVISC * SFC_QVISC * SFC_ZQRZT)
#define SFC_FZT1     (SFC_RVISC * SFC_TVISC * SFC_SQPR)
#define SFC_FZT2     (SFC_CZIV * SFC_GRRS * SFC_TVISC * SFC_SQPR)
#define SFC_FZU1     (SFC_CZIV * SFC_VISC)
#define SFC_RQVISC   (1.0f / SFC_QVISC)
#define SFC_USTFC    (0.018f / MYJ_G)
#define SFC_WWST2    (SFC_WWST * SFC_WWST)


// CUDA uses explicit round-to-nearest single-precision division at these sites.
// MSL float division is kept as a distinct operation; contraction is disabled.
inline float myjsfc_div_rn(float a, float b) { return a / b; }

inline float myjsfc_table(device const float *psi, float zeta,
                          float ztmin, float dzeta) {
    float rz = (zeta - ztmin) / dzeta;
    int k = int(rz);
    float rdzt = rz - float(k);
    k = min(k, MYJSFC_KZTM2);
    k = max(k, 0);
    return (psi[k + 1] - psi[k]) * rdzt + psi[k];
}

kernel void myjsfc_column(
    constant MyjsfcColumnResources &r [[buffer(0)]],
    constant MyjsfcColumnParams &q [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(q.n)) return;

    const ulong st = ulong(q.n);

    int ntsd = q.itimestep;
    float ust = r.ust[col];
    if (ntsd == 1) ust = 0.1f;            // :186-203 (ARW branch)
    float seamask = r.xland[col] - 1.0f;   // :231
    float psfc = r.psfc[col];
    float thsk = r.tsk[col] / pow(myjsfc_div_rn(psfc,MYJ_P1000MB), SFC_CAPA);  // :227

    // PBLH scan (:263-277): lpbl is the myj (top-down, 1-based) index of
    // the first weak-TKE level above the lowest layer; PBLH=ZHK(LPBL).
    //
    // DELIBERATE DIVERGENCE, gpuwm goes its own way (float32 precision).
    // WRF builds ZHK upward from ZINT(I,KTE+1,J)=HT(I,J) (:162) and then
    // subtracts ZHK(LMH+1) back off (:277), so over terrain it differences
    // two sea-level heights; this kernel accumulates dz directly, which is
    // the same value in float arithmetic and a strictly better-conditioned
    // one in float32.  Same declaration as the PBL kernel's header block;
    // measured by
    // tests/test_myj_port.py::test_the_dropped_terrain_height_cancels_in_float32.
    int lmh = q.nz;
    int lpbl = lmh;
    // ZINT(I,K,J)=ZINT(I,K+1,J)+DZ(I,KFLIP,J) (:177-184) accumulates inside
    // ONE column, so the seed is this thread's own lowest layer: element
    // (0, col) of the (nz, ny, nx) array is r.dz[col], the same index :191
    // reads six lines down.  r.dz[0] was column 0's lowest layer for every
    // thread, wrong by dz[0][0] - dz[0][col] wherever terrain thins it.
    float zcum = r.dz[col];                // interface height above layer 0
    float pblh = 0.0f;
    bool found = false;
    for (int iz = 1; iz < q.nz && !found; ++iz) {
        zcum += r.dz[(ulong)iz * st + col];
        if (2.0f * r.tke[(ulong)iz * st + col] <= MYJ_EPSQ2 * SFC_FH) {
            lpbl = q.nz - iz;
            pblh = zcum;                  // ZHK(LPBL) - ZHK(LMH+1)
            found = true;
        }
    }
    if (!found) { lpbl = 1; pblh = zcum; }
    (void)lpbl;

    // Lowest-layer state (:284-294).
    float plow = r.p1[col];
    float tlow = r.t1[col];
    float thlow = r.th1[col];
    float qv1 = r.qv1[col];
    float qlow = qv1 / (1.0f + qv1);
    float cwmlow = r.qc1[col];
    float thelow = (cwmlow * (-SFC_ELOCP / tlow) + 1.0f) * thlow;
    float ulow = r.u1[col];
    float vlow = r.v1[col];
    float zsl = r.dz[col] * 0.5f;
    float apesfc = pow(myjsfc_div_rn(psfc,MYJ_P1000MB), SFC_CAPA);
    float tz0 = (ntsd == 1) ? r.tsk[col] : r.thz0[col] * apesfc;  // :296-305

    // ---- SFCDIF (:361-1056) ----
    float qs = r.qsfc[col];
    float thz0 = r.thz0[col];
    float qz0 = r.qz0[col];
    float uz0 = r.uz0[col];
    float vz0 = r.vz0[col];
    float z0 = r.znt[col];
    float akhs = r.akhs[col];
    float akms = r.akms[col];
    float z0base = r.z0base[col];
    float mavail = r.mavail[col];

    float rdz = 1.0f / zsl;
    float cxchl = SFC_EXCML * rdz;
    float cxchs = SFC_EXCMS * rdz;
    float btgx = MYJ_G / thlow;
    float elfc = SFC_VKARMAN * btgx;
    float btgh = (pblh > 1000.0f) ? btgx * pblh : btgx * 1000.0f;

    float wstar2 = 0.0f;
    float rlmo = 0.0f;
    float rib = 0.0f, du2 = SFC_EPSU2, dthv = 0.0f;
    float zu = z0, zt = z0;
    float psmz = 0.0f, pshz = 0.0f, ustark = ust * SFC_VKARMAN;

    if (seamask > 0.5f) {
        // ---- sea points (:443-636) ----
        for (int itr = 0; itr < SFC_ITRMX; ++itr) {
            z0 = fmax(SFC_USTFC * ust * ust, 1.59e-5f);
            if (ust < SFC_USTC) {
                if (ust < SFC_USTR) {
                    if (ntsd == 1) {
                        akms = cxchs;
                        akhs = cxchs;
                        qs = qlow;
                    }
                    zu = SFC_FZU1 * sqrt(sqrt(z0 * ust * SFC_RVISC)) / ust;
                    float wght = akms * zu * SFC_RVISC;
                    float rwgh = wght / (wght + 1.0f);
                    uz0 = (ulow * rwgh + uz0) * 0.5f;
                    vz0 = (vlow * rwgh + vz0) * 0.5f;
                    zt = SFC_FZT1 * zu;
                    float zq = SFC_FZQ1 * zt;
                    float wghtt = akhs * zt * SFC_RTVISC;
                    float wghtq = akhs * zq * SFC_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
                if (ust >= SFC_USTR && ust < SFC_USTC) {
                    zu = z0;
                    uz0 = 0.0f;
                    vz0 = 0.0f;
                    zt = SFC_FZT2 * sqrt(sqrt(z0 * ust * SFC_RVISC)) / ust;
                    float zq = SFC_FZQ2 * zt;
                    float wghtt = akhs * zt * SFC_RTVISC;
                    float wghtq = akhs * zq * SFC_RQVISC;
                    if (ntsd > 1) {
                        thz0 = ((wghtt * thlow + thsk) / (wghtt + 1.0f)
                                + thz0) * 0.5f;
                        qz0 = ((wghtq * qlow + qs) / (wghtq + 1.0f)
                               + qz0) * 0.5f;
                    } else {
                        thz0 = (wghtt * thlow + thsk) / (wghtt + 1.0f);
                        qz0 = (wghtq * qlow + qs) / (wghtq + 1.0f);
                    }
                }
            } else {
                zu = z0;
                uz0 = 0.0f;
                vz0 = 0.0f;
                zt = z0;
                thz0 = thsk;
                qz0 = qs;
            }
            float tem = (tlow + tz0) * 0.5f;
            float thm = (thelow + thz0) * 0.5f;
            float a = thm * MYJ_P608;
            float b = (SFC_ELOCP / tem - 1.0f - MYJ_P608) * thm;
            dthv = ((thelow - thz0)
                    * ((qlow + qz0 + cwmlow) * (0.5f * MYJ_P608) + 1.0f)
                    + (qlow - qz0 + cwmlow) * a + cwmlow * b);
            du2 = fmax((ulow - uz0) * (ulow - uz0)
                        + (vlow - vz0) * (vlow - vz0), SFC_EPSU2);
            rib = btgx * dthv * zsl / du2;
            float zslu = zsl + zu;
            float zslt = zsl + zt;
            float rzsu = zslu / zu;
            float rzst = zslt / zt;
            float rlogu = log(rzsu);
            float rlogt = log(rzst);
            rlmo = elfc * akhs * dthv / (ust * ust * ust);
            float zetalu = zslu * rlmo;
            float zetalt = zslt * rlmo;
            float zetau = zu * rlmo;
            float zetat = zt * rlmo;
            zetalu = fmin(fmax(zetalu, q.ztmin1), q.ztmax1);
            zetalt = fmin(fmax(zetalt, q.ztmin1), q.ztmax1);
            zetau = fmin(fmax(zetau, q.ztmin1 / rzsu), q.ztmax1 / rzsu);
            zetat = fmin(fmax(zetat, q.ztmin1 / rzst), q.ztmax1 / rzst);
            psmz = myjsfc_table(r.psim1, zetau, q.ztmin1, q.dzeta1);
            float psmzl = myjsfc_table(r.psim1, zetalu, q.ztmin1, q.dzeta1);
            float simm = psmzl - psmz + rlogu;
            pshz = myjsfc_table(r.psih1, zetat, q.ztmin1, q.dzeta1);
            float pshzl = myjsfc_table(r.psih1, zetalt, q.ztmin1, q.dzeta1);
            float simh = (pshzl - pshz + rlogt) * q.fh01;
            ustark = ust * SFC_VKARMAN;
            akms = fmax(ustark / simm, cxchs);
            akhs = fmax(ustark / simh, cxchs);
            if (dthv <= 0.0f) {
                wstar2 = SFC_WWST2
                    * pow(fabs(btgh * akhs * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmax(sqrt(akms * sqrt(du2 + wstar2)), SFC_EPSUST);
        }
    } else {
        // ---- land points (:644-806) ----
        if (ntsd == 1) qs = qlow;
        zu = z0;
        uz0 = 0.0f;
        vz0 = 0.0f;
        zt = zu * SFC_ZTFC;
        thz0 = thsk;
        qz0 = qs;
        float tem = (tlow + tz0) * 0.5f;
        float thm = (thelow + thz0) * 0.5f;
        float a = thm * MYJ_P608;
        float b = (SFC_ELOCP / tem - 1.0f - MYJ_P608) * thm;
        dthv = ((thelow - thz0)
                * ((qlow + qz0 + cwmlow) * (0.5f * MYJ_P608) + 1.0f)
                + (qlow - qz0 + cwmlow) * a + cwmlow * b);
        du2 = fmax((ulow - uz0) * (ulow - uz0)
                    + (vlow - vz0) * (vlow - vz0), SFC_EPSU2);
        rib = btgx * dthv * zsl / du2;
        float zslu = zsl + zu;
        float rzsu = zslu / zu;
        float rlogu = log(rzsu);
        float zslt = zsl + zu;       // u,v and t are at the same level (:685)
        const float czil = 0.1f;     // :697 (Chen-Zhang block commented out)
        float zilfc = -czil * SFC_VKARMAN * SFC_SQVISC;
        const float czetmax = 10.0f;
        float zzil;
        if (dthv > 0.0f) {
            if (rib < SFC_RIC) {
                zzil = zilfc * (1.0f + (myjsfc_div_rn(rib,SFC_RIC)) * (myjsfc_div_rn(rib,SFC_RIC))
                                * czetmax);
            } else {
                zzil = zilfc * (1.0f + czetmax);
            }
        } else {
            zzil = zilfc;
        }
        for (int itr = 0; itr < SFC_ITRMX; ++itr) {
            zt = fmax(exp(zzil * sqrt(ust * z0base)) * z0base,
                       SFC_EPSZT);                       // :733
            float rzst = zslt / zt;
            float rlogt = log(rzst);
            rlmo = elfc * akhs * dthv / (ust * ust * ust);
            float zetalu = zslu * rlmo;
            float zetalt = zslt * rlmo;
            float zetau = zu * rlmo;
            float zetat = zt * rlmo;
            zetalu = fmin(fmax(zetalu, q.ztmin2), q.ztmax2);
            zetalt = fmin(fmax(zetalt, q.ztmin2), q.ztmax2);
            zetau = fmin(fmax(zetau, q.ztmin2 / rzsu), q.ztmax2 / rzsu);
            zetat = fmin(fmax(zetat, q.ztmin2 / rzst), q.ztmax2 / rzst);
            psmz = myjsfc_table(r.psim2, zetau, q.ztmin2, q.dzeta2);
            float psmzl = myjsfc_table(r.psim2, zetalu, q.ztmin2, q.dzeta2);
            float simm = psmzl - psmz + rlogu;
            pshz = myjsfc_table(r.psih2, zetat, q.ztmin2, q.dzeta2);
            float pshzl = myjsfc_table(r.psih2, zetalt, q.ztmin2, q.dzeta2);
            float simh = (pshzl - pshz + rlogt) * q.fh02;
            ustark = ust * SFC_VKARMAN;
            akms = fmax(ustark / simm, cxchl);
            akhs = fmax(ustark / simh, cxchl);
            if (dthv <= 0.0f) {
                wstar2 = SFC_WWST2
                    * pow(fabs(btgh * akhs * dthv), 2.0f / 3.0f);
            } else {
                wstar2 = 0.0f;
            }
            ust = fmax(sqrt(akms * sqrt(du2 + wstar2)), SFC_EPSUST);
        }
    }
    float ct = 0.0f;                 // :816-825, countergradient commented out

    // ---- 2m / 10m diagnostics (:829-1023) ----
    float umflx = akms * (ulow - uz0);
    float vmflx = akms * (vlow - vz0);
    float hsflx = akhs * (thlow - thz0);
    float hlflx = akhs * (qlow - qz0);
    float zu10 = zu + 10.0f;
    float zt02 = zt + 2.0f;
    float zt10 = zt + 10.0f;
    float rlnu10 = log(zu10 / zu);
    float rlnt02 = log(zt02 / zt);
    float rlnt10 = log(zt10 / zt);
    float ztau10 = zu10 * rlmo;
    float ztat02 = zt02 * rlmo;
    float ztat10 = zt10 * rlmo;
    float akms10, akhs02, akhs10;
    if (seamask > 0.5f) {
        ztau10 = fmin(fmax(ztau10, q.ztmin1), q.ztmax1);
        ztat02 = fmin(fmax(ztat02, q.ztmin1), q.ztmax1);
        ztat10 = fmin(fmax(ztat10, q.ztmin1), q.ztmax1);
        float psm10 = myjsfc_table(r.psim1, ztau10, q.ztmin1, q.dzeta1);
        float simm10 = psm10 - psmz + rlnu10;
        float psh02 = myjsfc_table(r.psih1, ztat02, q.ztmin1, q.dzeta1);
        float simh02 = (psh02 - pshz + rlnt02) * q.fh01;
        float psh10 = myjsfc_table(r.psih1, ztat10, q.ztmin1, q.dzeta1);
        float simh10 = (psh10 - pshz + rlnt10) * q.fh01;
        akms10 = fmax(ustark / simm10, cxchs);
        akhs02 = fmax(ustark / simh02, cxchs);
        akhs10 = fmax(ustark / simh10, cxchs);
    } else {
        ztau10 = fmin(fmax(ztau10, q.ztmin2), q.ztmax2);
        ztat02 = fmin(fmax(ztat02, q.ztmin2), q.ztmax2);
        ztat10 = fmin(fmax(ztat10, q.ztmin2), q.ztmax2);
        float psm10 = myjsfc_table(r.psim2, ztau10, q.ztmin2, q.dzeta2);
        float simm10 = psm10 - psmz + rlnu10;
        float psh02 = myjsfc_table(r.psih2, ztat02, q.ztmin2, q.dzeta2);
        float simh02 = (psh02 - pshz + rlnt02) * q.fh02;
        float psh10 = myjsfc_table(r.psih2, ztat10, q.ztmin2, q.dzeta2);
        float simh10 = (psh10 - pshz + rlnt10) * q.fh02;
        akms10 = fmax(ustark / simm10, cxchl);
        akhs02 = fmax(ustark / simh02, cxchl);
        akhs10 = fmax(ustark / simh10, cxchl);
    }
    float u10 = umflx / akms10 + uz0;
    float v10 = vmflx / akms10 + vz0;
    float th02 = hsflx / akhs02 + thz0;
    if ((thlow > thz0 && (th02 < thz0 || th02 > thlow))
        || (thlow < thz0 && (th02 > thz0 || th02 < thlow))) {
        th02 = thz0 + 2.0f * rdz * (thlow - thz0);
    }
    float th10 = hsflx / akhs10 + thz0;
    if ((thlow > thz0 && (th10 < thz0 || th10 > thlow))
        || (thlow < thz0 && (th10 > thz0 || th10 < thlow))) {
        th10 = thz0 + 10.0f * rdz * (thlow - thz0);
    }
    float q02 = hlflx / akhs02 + qz0;
    float q10 = hlflx / akhs10 + qz0;
    float pshltr = psfc * exp(-0.068283f / tlow);
    float u10e = u10;
    float v10e = v10;
    if (seamask < 0.5f) {
        // Equivalent z0 shelter winds over land (:988-1020).
        float zuuz = fmin(zu * 0.50f, 0.18f);
        float zu_l = fmax(zu * 0.35f, zuuz);
        float zu10_l = zu_l + 10.0f;
        float rzsu_l = zu10_l / zu_l;
        float rlnu10_l = log(rzsu_l);
        float ztau10_l = zu10_l * rlmo;
        ztau10_l = fmin(fmax(ztau10_l, q.ztmin2), q.ztmax2);
        float psm10 = myjsfc_table(r.psim2, ztau10_l, q.ztmin2, q.dzeta2);
        float simm10 = psm10 - psmz + rlnu10_l;
        float ekms10 = fmax(ustark / simm10, cxchl);
        u10e = umflx / ekms10 + uz0;
        v10e = vmflx / ekms10 + vz0;
    }
    // ---- WRF driver arrays (:1029-1053) ----
    float rlow = plow / (MYJ_RD * tlow);
    float chs = akhs;
    float chs2 = akhs02;
    float cqs2 = akhs02;
    float hfx = -rlow * MYJ_CP * hsflx;
    float qfx = -rlow * hlflx * mavail;
    float flx_lh = MYJ_XLV * qfx;
    float flhc = rlow * MYJ_CP * akhs;
    float flqc = rlow * akhs * mavail;
    float qgh = ((1.0f - seamask) * MYJ_PQ0 + seamask * SFC_PQ0SEA) / plow
        * exp(SFC_A2S * (tlow - SFC_A3S) / (tlow - SFC_A4S));
    qgh = qgh / (1.0f - qgh);              // convert to mixing ratio (:1041)
    float cpm = MYJ_CP * (1.0f + 0.8f * qlow);
    if (seamask > 0.5f) {
        float tskl = r.tsk[col];
        qs = SFC_PQ0SEA / psfc
            * exp(SFC_A2S * (tskl - SFC_A3S) / (tskl - SFC_A4S));
        qs = qs / (1.0f - qs);
    }
    // ---- shelter supersaturation removal (module_sf_myjsfc.F:326-349) ----
    float tshltr = th02;      // SFCDIF's TH02 dummy is MYJSFC's TSHLTR
    float qshltr = q02;       // SFCDIF's Q02 dummy is MYJSFC's QSHLTR
    float rapa = apesfc;
    float th02p = tshltr;
    float th10p = th10;
    float th2_grid = tshltr;
    float rapa02 = rapa - SFC_GOCP02 / th02p;
    float rapa10 = rapa - SFC_GOCP10 / th10p;
    float t02p = th02p * rapa02;
    float t10p = th10p * rapa10;
    float t2_grid = th2_grid * apesfc;      // :337
    float p02p = pow(rapa02, SFC_RCAP) * MYJ_P1000MB;
    float p10p = pow(rapa10, SFC_RCAP) * MYJ_P1000MB;
    float qs02 = MYJ_PQ0 / p02p
        * exp(MYJ_A2 * (t02p - MYJ_A3) / (t02p - MYJ_A4));
    float qs10 = MYJ_PQ0 / p10p
        * exp(MYJ_A2 * (t10p - MYJ_A3) / (t10p - MYJ_A4));
    if (qshltr > qs02) qshltr = qs02;
    if (q10 > qs10) q10 = qs10;
    float q02_grid = qshltr / (1.0f - qshltr);   // :349

    // ---- writeback ----
    r.ust[col] = ust;
    r.znt[col] = z0;
    r.thz0[col] = thz0;
    r.qz0[col] = qz0;
    r.uz0[col] = uz0;
    r.vz0[col] = vz0;
    r.qsfc[col] = qs;
    r.akhs[col] = akhs;
    r.akms[col] = akms;
    r.rmol[col] = rlmo;
    r.ct[col] = ct;
    r.pblh[col] = pblh;
    r.rib[col] = rib;
    r.chs[col] = chs;
    r.chs2[col] = chs2;
    r.cqs2[col] = cqs2;
    r.hfx[col] = hfx;
    r.qfx[col] = qfx;
    r.lh[col] = flx_lh;
    r.flhc[col] = flhc;
    r.flqc[col] = flqc;
    r.qgh[col] = qgh;
    r.cpm[col] = cpm;
    r.u10[col] = u10;
    r.v10[col] = v10;
    r.t2[col] = t2_grid;
    r.th2[col] = th2_grid;
    r.tshltr[col] = tshltr;
    r.th10[col] = th10;
    r.q2[col] = q02_grid;
    r.qshltr[col] = qshltr;
    r.q10[col] = q10;
    r.pshltr[col] = pshltr;
    r.u10e[col] = u10e;
    r.v10e[col] = v10e;

}

kernel void myj_driver_wspd(
    constant MyjDriverWspdResources &r [[buffer(0)]],
    constant MyjDriverWspdParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= uint(p.n)) return;
    const float u = r.u[i], v = r.v[i];
    r.wspd[i] = max(sqrt(u * u + v * v), 0.001f);
}
