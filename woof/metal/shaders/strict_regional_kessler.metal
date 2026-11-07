#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#pragma clang fp contract(off)
#define KESS_KMAX 256
#define IDX3(k,j,i) ((ulong(k)*ny+ulong(j))*nx+ulong(i))
static inline F operator/(F a,F b){return f32_word(soft32_div(a.bits,b.bits));}
static inline F operator/(float a,F b){return F(a)/b;}
static inline F operator/(F a,float b){return a/F(b);}
static inline F kf_max(F a,F b){if(soft32_is_nan(a.bits))return b;if(soft32_is_nan(b.bits))return a;if(soft32_equal(a.bits,b.bits))return f32_word(a.bits & b.bits);return a>b?a:b;}
static inline F kf_min(F a,F b){if(soft32_is_nan(a.bits))return b;if(soft32_is_nan(b.bits))return a;if(soft32_equal(a.bits,b.bits))return f32_word(a.bits | b.bits);return a<b?a:b;}
static inline F kf_floor(F a){return F(floor(as_type<float>(a.bits)));}
static inline F kf_exp(F a){return native_tc_math::thompson_aa_expf(a);}
#define SVP1 F(0.6112f)
#define SVP2 F(17.67f)
#define SVP3 F(29.65f)
#define SVPT0 F(273.15f)
#define XLV F(2500000.0f)
#define CP F(1004.5f)
#define EP2 F(0.6217504332755632f)
#define RHOWATER F(1000.0f)
#include "strict_regional_kessler_bindings.metalh"
kernel void strict_regional_kessler(constant StrictRegionalKesslerResources &r [[buffer(0)]],constant StrictRegionalKesslerParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
device F *t=(device F*)r.t;
device F *qv=(device F*)r.qv;
device F *qc=(device F*)r.qc;
device F *qr=(device F*)r.qr;
device const F *rho=(device const F*)r.rho;
device const F *pii=(device const F*)r.pii;
device const F *z=(device const F*)r.z;
device const F *dz8w=(device const F*)r.dz8w;
device F *rainnc=(device F*)r.rainnc;
device F *rainncv=(device F*)r.rainncv;
F dt=F(p.dt);int nz=p.nz,ny=p.ny,nx=p.nx;

    int col = int(gid);
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;

    const F c1k = 0.001f, c2k = 0.001f, c3k = 2.2f, c4k = 0.875f;
    const F max_cr_sedimentation = 0.75f;
    const F f5 = SVP2 * (SVPT0 - SVP3) * XLV / CP;

    F prodk[KESS_KMAX], vt[KESS_KMAX], vtden[KESS_KMAX],
         rhok[KESS_KMAX], rdzk[KESS_KMAX];

    // --- terminal velocity and the stable time-split count
    F crmax = 0.0f;
    F rho0 = rho[IDX3(0, j, i)];
    for (int k = 0; k < nz; ++k) {
        prodk[k] = qr[IDX3(k, j, i)];
        rhok[k] = rho[IDX3(k, j, i)];
        F qrr = kf_max(0.0f, qr[IDX3(k, j, i)] * 0.001f * rhok[k]);
        vtden[k] = ntc::tc_sqrt(rho0 / rhok[k]);
        vt[k] = 36.34f * gfk_pow(qrr, 0.1364f) * vtden[k];
        F rdzw = 1.0f / dz8w[IDX3(k, j, i)];
        crmax = kf_max(vt[k] * dt * rdzw, crmax);
    }
    for (int k = 0; k < nz - 1; ++k)
        rdzk[k] = 1.0f / (z[IDX3(k + 1, j, i)] - z[IDX3(k, j, i)]);
    rdzk[nz - 1] = 1.0f / (z[IDX3(nz - 1, j, i)] - z[IDX3(nz - 2, j, i)]);

    int nfall = max(1, (int)kf_floor(0.5f + __fdiv_rn(crmax, max_cr_sedimentation)
                                   + 0.5f));                  // Fortran NINT
    F dtfall = dt / (F)nfall;
    F time_sediment = dt;

    // --- time-split sedimentation, fallout with flux upstream
    while (nfall > 0) {
        time_sediment -= dtfall;

        F ppt = __fdiv_rn(rhok[0] * prodk[0] * vt[0] * dtfall, RHOWATER);
        rainncv[(size_t)j * nx + i] = ppt * 1000.0f;
        rainnc[(size_t)j * nx + i] += ppt * 1000.0f;          // mm

        // the in-place bottom-up update reads prodk[k+1] before it is
        // overwritten -- identical to the Fortran loop.
        for (int k = 0; k < nz - 1; ++k) {
            F factor = dtfall * rdzk[k] / rhok[k];
            prodk[k] = prodk[k] - factor * (rhok[k] * prodk[k] * vt[k]
                                            - rhok[k + 1] * prodk[k + 1]
                                              * vt[k + 1]);
        }
        prodk[nz - 1] = prodk[nz - 1] - (dtfall * rdzk[nz - 1])
                        * prodk[nz - 1] * vt[nz - 1];

        if (nfall > 1) {          // not the last split step: new vt, nfall
            nfall = nfall - 1;
            crmax = 0.0f;
            for (int k = 0; k < nz; ++k) {
                F qrr = kf_max(0.0f, prodk[k] * 0.001f * rhok[k]);
                vt[k] = 36.34f * gfk_pow(qrr, 0.1364f) * vtden[k];
                F rdzw = 1.0f / dz8w[IDX3(k, j, i)];
                crmax = kf_max(vt[k] * time_sediment * rdzw, crmax);
            }
            int nfall_new = max(1, (int)kf_floor(0.5f + __fdiv_rn(crmax,
                                               max_cr_sedimentation)
                                               + 0.5f));
            if (nfall_new != nfall) {
                nfall = nfall_new;
                dtfall = time_sediment / (F)nfall;
            }
        } else {                  // last split step: prodk is the fallen qr
            nfall = 0;
        }
    }

    // --- production of rain from qc, saturation adjustment, evaporation
    for (int k = 0; k < nz; ++k) {
        size_t idx = IDX3(k, j, i);
        F qrk = qr[idx];                     // PRE-sedimentation qr
        F qck = qc[idx];
        F qvk = qv[idx];
        F tk = t[idx];
        F piik = pii[idx];

        F factorn = 1.0f / (1.0f + c3k * dt
                               * gfk_pow(kf_max(0.0f, qrk), c4k));
        F qrprod = qck * (1.0f - factorn)
                    + factorn * c1k * dt * kf_max(qck - c2k, 0.0f);
        F rcgs = 0.001f * rho[idx];

        qck = kf_max(qck - qrprod, 0.0f);
        // module_mp_kessler.F: qr = (qr + prod - qr).  In float32 that is
        // NOT prod: (qr + prod) rounds before qr is taken back off.  WRF's
        // words are this expression's words, so it is evaluated as written
        // (the WRF column oracle, tests/test_kessler_wrf461_parity.py).
        qrk = (qrk + prodk[k]) - qrk;
        qrk = kf_max(qrk + qrprod, 0.0f);

        F temp = piik * tk;
        F pressure = 1.0e5f * gfk_pow(piik, 1004.0f / 287.0f);
        F gam = 2.5e6f / (1004.0f * piik);
        F es = 1000.0f * SVP1 * kf_exp(SVP2 * (temp - SVPT0)
                                        / (temp - SVP3));
        // WOOF guard, NOT in module_mp_kessler.F: es is capped at 0.99 p,
        // the cap WRF's own Morrison (EVS = min(0.99*pres, ...)) and Ferrier
        // (ESW = MIN(..., 0.99*PP)) apply before the same ep2*es/(p - es).
        // The breakage it prevents (lane/verify-kessler-mp1-refute): with a
        // model top near 1-3 hPa and a 255-275 K stratopause, es >= p, so
        // WRF's qvs goes negative, the adjustment "condenses" ~100 times the
        // vapour present (qv clamped to 0, water created) and heats the top
        // layers without bound: the WRF v4.6.1 column oracle reaches 4,435 K
        // after four 60 s calls from 4 ppmv of vapour.  Where es <= 0.99 p
        // the min returns es itself, so every other word stays WRF's.
        es = kf_min(es, 0.99f * pressure);
        F qvs = EP2 * es / (pressure - es);
        F prod = (qvk - qvs)
                  / (1.0f + pressure / (pressure - es) * qvs * f5
                            / ((temp - SVP3) * (temp - SVP3)));
        F ern = kf_min(kf_min(
            dt * (((1.6f + 124.9f * gfk_pow(rcgs * qrk, 0.2046f))
                   * gfk_pow(rcgs * qrk, 0.525f))
                  / (2.55e8f / (pressure * qvs) + 5.4e5f))
               * (kf_max(qvs - qvk, 0.0f) / (rcgs * qvs)),
            kf_max(-prod - qck, 0.0f)), qrk);

        F product = kf_max(prod, -qck);
        t[idx]  = tk + gam * (product - ern);
        qv[idx] = kf_max(qvk - product + ern, 0.0f);
        qc[idx] = qck + product;
        qr[idx] = qrk - ern;
    }

}
