#include <metal_stdlib>
#include "pressure_power.metalh"
#include "wsm6_column_bindings.metalh"

using namespace metal;

// Keep the CUDA float operation sequence visible. Device parity remains
// unverified until the central runtime build and WRF trajectory gates run.
#pragma clang fp contract(off)

static inline float wsm6_rn_div(float a, float b) { return a / b; }
constant int WSM6_KMAX = 64;

static inline float fmax0(float value) { return max(value, 0.0f); }
static inline float clamp01(float value) { return min(max(value, 0.0f), 1.0f); }

struct Rimed {
    float n0g, deng, avtg, bvtg, lamdagmax;
    float pidn0g, pvtg, pacrg, precg1, precg2;
    float rslopegmax, rslopegbmax, rslopeg2max, rslopeg3max;
};

static inline Rimed rimed_constants(int hail_opt) {
    Rimed r;
    if (hail_opt == 1) {
        r.n0g = 4.0e4f; r.deng = 700.0f; r.avtg = 285.0f;
        r.bvtg = 0.8f; r.lamdagmax = 2.0e4f;
        r.pidn0g = 8.79645943e7f;
        r.pvtg = 8.46323123e2f;
        r.pacrg = 4.19991431e7f;
        r.precg1 = 1.96035382e5f;
        r.precg2 = 2.40250519e6f;
        r.rslopegmax = 5.0e-5f;
        r.rslopegbmax = 3.62389832e-4f;
        r.rslopeg2max = 2.5e-9f;
        r.rslopeg3max = 1.25e-13f;
    } else {
        r.n0g = 4.0e6f; r.deng = 500.0f; r.avtg = 330.0f;
        r.bvtg = 0.8f; r.lamdagmax = 6.0e4f;
        r.pidn0g = 6.28318531e9f;
        r.pvtg = 9.79953090e2f;
        r.pacrg = 4.86305868e9f;
        r.precg1 = 1.96035382e7f;
        r.precg2 = 2.58522814e8f;
        r.rslopegmax = 1.66666667e-5f;
        r.rslopegbmax = 1.50480075e-4f;
        r.rslopeg2max = 2.77777778e-10f;
        r.rslopeg3max = 4.62962963e-15f;
    }
    return r;
}

struct Slope {
    float r, rb, r2, r3, vt;
};

static inline Slope rain_slope(float q, float den,
                                             float denfac) {
    Slope s;
    if (q <= 1.0e-9f) {
        s.r = 1.25e-5f; s.rb = 1.19544062e-4f;
        s.r2 = 1.5625e-10f; s.r3 = 1.953125e-15f;
    } else {
        s.r = 1.0f / sqrt(sqrt(2.51327412e10f / (q * den)));
        s.rb = pow(s.r, 0.8f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = (q > 0.0f) ? 2.50006820e3f * s.rb * denfac : 0.0f;
    return s;
}

static inline Slope snow_slope(float q, float den,
                                             float denfac, float temp,
                                             thread float *n0sfac_out) {
    Slope s;
    float n0sfac = max(min(exp(0.12f * (273.15f - temp)), 5.0e4f),
                         1.0f);
    *n0sfac_out = n0sfac;
    if (q <= 1.0e-9f) {
        s.r = 1.0e-5f; s.rb = 8.91250938e-3f;
        s.r2 = 1.0e-10f; s.r3 = 1.0e-15f;
    } else {
        s.r = 1.0f / sqrt(sqrt(6.28318531e8f * n0sfac / (q * den)));
        s.rb = pow(s.r, 0.41f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = (q > 0.0f) ? 2.00517852e1f * s.rb * denfac : 0.0f;
    return s;
}

static inline Slope graupel_slope(float q, float den,
                                                float denfac,
                                                thread const Rimed &g) {
    Slope s;
    if (q <= 1.0e-9f) {
        s.r = g.rslopegmax; s.rb = g.rslopegbmax;
        s.r2 = g.rslopeg2max; s.r3 = g.rslopeg3max;
    } else {
        s.r = 1.0f / sqrt(sqrt(g.pidn0g / (q * den)));
        s.rb = pow(s.r, g.bvtg); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = (q > 0.0f) ? g.pvtg * s.rb * denfac : 0.0f;
    return s;
}

static inline float diffus(float t, float p) {
    return 8.794e-5f * pow(t, 1.81f) / p;
}
static inline float viscos(float t, float den) {
    return 1.496e-6f * (t * sqrt(t)) / (t + 120.0f) / den;
}
static inline float xka(float t, float den) {
    return 1.414e3f * viscos(t, den) * den;
}
static inline float diffac(float latent, float p, float t,
                                        float den, float qsat) {
    return den * latent * latent / (xka(t, den) * 461.6f * t * t)
         + 1.0f / (qsat * diffus(t, p));
}
static inline float venfac(float p, float t, float den) {
    return pow(viscos(t, den) / diffus(t, p), 0.3333333f)
         / sqrt(viscos(t, den)) * sqrt(sqrt(1.28f / den));
}
static inline float fpvs(float t, bool ice) {
    const float ttp = 273.16f;
    const float xa = -(1846.4f - 4190.0f) / 461.6f;
    const float xb = xa + 2.5e6f / (461.6f * ttp);
    const float xai = -(1846.4f - 2106.0f) / 461.6f;
    const float xbi = xai + 2.85e6f / (461.6f * ttp);
    float tr = ttp / t;
    float aa = (ice && t < ttp) ? xai : xa;
    float bb = (ice && t < ttp) ? xbi : xb;
    return 610.78f * pow(tr, aa) * exp(bb * (1.0f - tr));
}
static inline float qsat(float t, float p, bool ice) {
    float es = min(fpvs(t, ice), 0.99f * p);
    // WRF passes ep2=rd/rv from the driver.  Keep that ratio instead of the
    // commonly rounded 0.622: the rounding is large enough to move a warm
    // saturation-adjustment oracle by O(1e-3 K) in one call.
    return max(1.0e-15f, (287.0f / 461.6f) * es / (p - es));
}

// WRF v4.6.1 nislfv_rain_plm/plm6.  Arrays with nz+1 entries represent
// interfaces.  q/partner enter as mixing ratios and are remapped as density
// concentrations, exactly as the Fortran caller does.
static inline void plm_arrival(
    thread const float *qq, thread const float *qq2, thread const float *ww, thread const float *dz,
    thread const float *den, int nz, float dt, thread float *zi, thread float *za, thread float *dza,
    thread float *qa, thread float *qa2)
{
    float wi[WSM6_KMAX + 1];
    zi[0] = 0.0f;
    for (int k = 0; k < nz; ++k) zi[k + 1] = zi[k] + dz[k];

    // WRF computes a second-order interface interpolation first and then
    // overwrites it with the active third-order interpolation.
    wi[0] = ww[0];
    wi[nz] = ww[nz - 1];
    for (int k = 1; k < nz; ++k)
        wi[k] = (ww[k] * dz[k - 1] + ww[k - 1] * dz[k])
              / (dz[k - 1] + dz[k]);
    wi[0] = ww[0];
    if (nz > 1) wi[1] = 0.5f * (ww[1] + ww[0]);
    for (int k = 2; k < nz - 1; ++k)
        wi[k] = (9.0f / 16.0f) * (ww[k] + ww[k - 1])
              - (1.0f / 16.0f) * (ww[k + 1] + ww[k - 2]);
    if (nz > 1) wi[nz - 1] = 0.5f * (ww[nz - 1] + ww[nz - 2]);
    wi[nz] = ww[nz - 1];

    for (int k = 1; k < nz; ++k)
        if (ww[k] == 0.0f) wi[k] = ww[k - 1];
    for (int k = nz - 1; k >= 0; --k) {
        float decfl = (wi[k + 1] - wi[k]) * dt / dz[k];
        if (decfl > 0.05f) wi[k] = wi[k + 1] - 0.05f * dz[k] / dt;
    }
    for (int k = 0; k <= nz; ++k) za[k] = zi[k] - wi[k] * dt;
    for (int k = 0; k < nz; ++k) dza[k] = za[k + 1] - za[k];
    dza[nz] = zi[nz] - za[nz];
    for (int k = 0; k < nz; ++k) {
        qa[k] = qq[k] * dz[k] / dza[k];
        if (qq2) qa2[k] = qq2[k] * dz[k] / dza[k];
    }
    qa[nz] = 0.0f;
    if (qq2) qa2[nz] = 0.0f;
}

static inline float plm_regular_remap(
    thread const float *qa, thread const float *zi, thread const float *za, thread const float *dza,
    int nz, thread float *qn)
{
    float qmi[WSM6_KMAX + 1], qpi[WSM6_KMAX + 1];
    for (int k = 1; k < nz; ++k) {
        float dip = (qa[k + 1] - qa[k]) / (dza[k + 1] + dza[k]);
        float dim = (qa[k] - qa[k - 1]) / (dza[k - 1] + dza[k]);
        if (dip * dim <= 0.0f) {
            qmi[k] = qa[k]; qpi[k] = qa[k];
        } else {
            qpi[k] = qa[k] + 0.5f * (dip + dim) * dza[k];
            qmi[k] = 2.0f * qa[k] - qpi[k];
            if (qpi[k] < 0.0f || qmi[k] < 0.0f) {
                qpi[k] = qa[k]; qmi[k] = qa[k];
            }
        }
    }
    qpi[0] = qa[0]; qmi[0] = qa[0];
    qpi[nz] = qa[nz]; qmi[nz] = qa[nz];
    for (int k = 0; k < nz; ++k) qn[k] = 0.0f;

    // Integrate the limited profile over each complete regular cell. Arrival
    // cells cover only the falling column; the space above it contains no mass.
    // A persistent source cursor avoids sentinel indices and keeps the overlap
    // walk linear in the number of interfaces.
    int source = 0;
    for (int k = 0; k < nz; ++k) {
        while (source < nz && za[source + 1] <= zi[k]) ++source;
        float mass = 0.0f;
        for (int j = source; j < nz && za[j] < zi[k + 1]; ++j) {
            float low = max(zi[k], za[j]);
            float high = min(zi[k + 1], za[j + 1]);
            if (high > low) {
                float middle = clamp01(0.5f * ((low - za[j]) + (high - za[j]))
                                       / dza[j]);
                float average = qmi[j] * (1.0f - middle) + qpi[j] * middle;
                mass += (high - low) * average;
            }
        }
        qn[k] = mass / (zi[k + 1] - zi[k]);
    }

    // The outflow uses the same linear profile as the retained cell mass,
    // including a partially fallen cell with a nonzero slope.
    float precip = 0.0f;
    for (int j = 0; j < nz && za[j] < 0.0f; ++j) {
        float high = min(0.0f, za[j + 1]);
        float middle = clamp01(0.5f * (high - za[j]) / dza[j]);
        float average = qmi[j] * (1.0f - middle) + qpi[j] * middle;
        precip += (high - za[j]) * average;
    }
    return precip;
}

static inline float sediment_column(thread float *q, thread const float *den,
                                 thread const float *dz, thread const float *temp,
                                 int nz, float dt, int species,
                                 thread const Rimed &g, thread float *partner,
                                 thread float *partner_precip_out) {
    float qq[WSM6_KMAX], qq2[WSM6_KMAX], wd[WSM6_KMAX], ww[WSM6_KMAX];
    float wa[WSM6_KMAX], qn[WSM6_KMAX], qn2[WSM6_KMAX];
    float zi[WSM6_KMAX + 1], za[WSM6_KMAX + 1];
    float dza[WSM6_KMAX + 1], qa[WSM6_KMAX + 1], qa2[WSM6_KMAX + 1];
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        float q1 = fmax0(q[k]);
        float q2 = partner ? fmax0(partner[k]) : 0.0f;
        qq[k] = q1 * den[k]; qq2[k] = q2 * den[k];
        float df = sqrt(1.28f / den[k]);
        if (species == 0) {
            ww[k] = rain_slope(q1, den[k], df).vt;
        } else if (species == 1) {
            float n0s;
            float vs = snow_slope(q1, den[k], df, temp[k], &n0s).vt;
            float vg = graupel_slope(q2, den[k], df, g).vt;
            float total = max(q1 + q2, 1.0e-15f);
            ww[k] = total > 1.0e-15f ? (vs * q1 + vg * q2) / total : 0.0f;
        } else {
            float xniarg = den[k] * max(q1, 1.0e-15f);
            float xni = min(max(5.38e7f * pow(xniarg, 0.75f), 1.0e3f),
                              1.0e6f);
            float diameter = max(min(11.9f * sqrt(den[k] * q1 / xni),
                                         500.0e-6f), 1.0e-25f);
            ww[k] = q1 > 0.0f ? 1.49e4f * pow(diameter, 1.31f) : 0.0f;
        }
        wd[k] = ww[k];
        allold += qq[k] + (partner ? qq2[k] : 0.0f);
    }
    if (partner_precip_out) *partner_precip_out = 0.0f;
    if (allold <= 0.0f) return 0.0f;

    plm_arrival(qq, partner ? qq2 : nullptr, ww, dz, den, nz, dt,
                zi, za, dza, qa, qa2);

    // Rain and the paired snow/graupel categories use WRF iter=1.  Cloud ice
    // uses iter=0, retaining its departure velocity.
    if (species != 2) {
        for (int k = 0; k < nz; ++k) {
            float q1 = qa[k] / den[k];
            float df = sqrt(1.28f / den[k]);
            if (species == 0) {
                wa[k] = rain_slope(q1, den[k], df).vt;
            } else {
                float q2 = qa2[k] / den[k], n0s;
                float vs = snow_slope(q1, den[k], df, temp[k], &n0s).vt;
                float vg = graupel_slope(q2, den[k], df, g).vt;
                float total = max(q1 + q2, 1.0e-15f);
                wa[k] = total > 1.0e-15f
                      ? (vs * q1 + vg * q2) / total : 0.0f;
            }
            ww[k] = 0.5f * (wd[k] + wa[k]);
        }
        plm_arrival(qq, partner ? qq2 : nullptr, ww, dz, den, nz, dt,
                    zi, za, dza, qa, qa2);
    }

    float precip = plm_regular_remap(qa, zi, za, dza, nz, qn);
    float precip2 = partner
                  ? plm_regular_remap(qa2, zi, za, dza, nz, qn2) : 0.0f;
    for (int k = 0; k < nz; ++k) {
        q[k] = fmax0(qn[k] / den[k]);
        if (partner) partner[k] = fmax0(qn2[k] / den[k]);
    }
    if (partner_precip_out) *partner_precip_out = precip2;
    return precip;
}

static inline void wsm6_column_impl(
    device float *theta, device float *qv, device float *qc, device float *qi, device float *qr,
    device float *qs, device float *qg, device const float *den, device const float *p,
    device const float *pii, device const float *dz, device float *rainnc, device float *rainncv,
    device float *snownc, device float *snowncv, device float *graupelnc, device float *graupelncv,
    device float *sr, device float *effc, device float *effi, device float *effs,
    float delt, int hail_opt, int nz, int stride, int col)
{
    const Rimed gc = rimed_constants(hail_opt);
    float temp[WSM6_KMAX], rho[WSM6_KMAX], delz[WSM6_KMAX];
    float qvv[WSM6_KMAX], qcc[WSM6_KMAX], qii[WSM6_KMAX];
    float qrr[WSM6_KMAX], qss[WSM6_KMAX], qgg[WSM6_KMAX];
    float cpm[WSM6_KMAX], xl[WSM6_KMAX];
    float qsw[WSM6_KMAX], qsi[WSM6_KMAX], rhw[WSM6_KMAX], rhi[WSM6_KMAX];

    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        temp[k] = theta[id] * pii[id];
        rho[k] = den[id]; delz[k] = dz[id];
        qvv[k] = qv[id]; qcc[k] = fmax0(qc[id]); qii[k] = fmax0(qi[id]);
        qrr[k] = fmax0(qr[id]); qss[k] = fmax0(qs[id]); qgg[k] = fmax0(qg[id]);
        float vap = max(qvv[k], 1.0e-15f);
        cpm[k] = 1004.5f * (1.0f - vap) + vap * 1846.4f;
        xl[k] = 2.5e6f - 2343.6f * (temp[k] - 273.15f);
    }

    rainncv[col] = 0.0f; snowncv[col] = 0.0f; graupelncv[col] = 0.0f;
    sr[col] = 0.0f;
    int loops = (int)floor(wsm6_rn_div(delt, 120.0f) + 0.5f);
    if (loops < 1) loops = 1;
    float dtcld = delt / (float)loops;

    for (int loop = 0; loop < loops; ++loop) {
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            qsw[k] = qsat(temp[k], p[id], false);
            qsi[k] = qsat(temp[k], p[id], true);
            rhw[k] = max(qvv[k] / qsw[k], 1.0e-15f);
            rhi[k] = max(qvv[k] / qsi[k], 1.0e-15f);
        }

        float pr = sediment_column(qrr, rho, delz, temp, nz, dtcld, 0,
                                    gc, nullptr, nullptr);
        float pg = 0.0f;
        float ps = sediment_column(qss, rho, delz, temp, nz, dtcld, 1,
                                    gc, qgg, &pg);
        // Ice crystals use their own WRF fall-speed law.
        float piice = sediment_column(qii, rho, delz, temp, nz, dtcld, 2,
                                      gc, nullptr, nullptr);

        float total_precip = pr + ps + pg + piice;
        rainncv[col] += total_precip;
        snowncv[col] += ps + piice;
        graupelncv[col] += pg;
        rainnc[col] += total_precip;
        snownc[col] += ps + piice;
        graupelnc[col] += pg;
        if (total_precip > 0.0f)
            sr[col] = (snowncv[col] + graupelncv[col])
                    / (rainncv[col] + 1.0e-12f);

        // Sedimentation-adjacent melting/freezing, then WSM6 process rates.
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            float denfac = sqrt(1.28f / rho[k]);
            float n0sfac;
            Slope rr = rain_slope(qrr[k], rho[k], denfac);
            Slope ss = snow_slope(qss[k], rho[k], denfac, temp[k], &n0sfac);
            Slope gg = graupel_slope(qgg[k], rho[k], denfac, gc);
            float supcol = 273.15f - temp[k];
            float xlf = 2.85e6f - xl[k];

            if (temp[k] > 273.15f) {
                float vf = venfac(p[id], temp[k], rho[k]);
                if (qss[k] > 0.0f) {
                    float coeres = ss.r2 * sqrt(ss.r * ss.rb);
                    float melt = wsm6_rn_div(xka(temp[k], rho[k]), 3.5e5f)
                        * (273.15f - temp[k]) * 1.57079632679f * n0sfac
                        * (5.2e6f * ss.r2 + 1.86818719e7f * vf * coeres)
                        / rho[k];
                    melt = min(max(melt * dtcld, -qss[k]), 0.0f);
                    qss[k] += melt; qrr[k] -= melt;
                    temp[k] += 3.5e5f / cpm[k] * melt;
                }
                if (qgg[k] > 0.0f) {
                    float coeres = gg.r2 * sqrt(gg.r * gg.rb);
                    float melt = wsm6_rn_div(xka(temp[k], rho[k]), 3.5e5f)
                        * (273.15f - temp[k])
                        * (gc.precg1 * gg.r2 + gc.precg2 * vf * coeres)
                        / rho[k];
                    melt = min(max(melt * dtcld, -qgg[k]), 0.0f);
                    qgg[k] += melt; qrr[k] -= melt;
                    temp[k] += 3.5e5f / cpm[k] * melt;
                }
            }

            supcol = 273.15f - temp[k];
            xlf = (supcol < 0.0f) ? 3.5e5f : 2.85e6f - xl[k];
            if (supcol < 0.0f && qii[k] > 0.0f) {
                qcc[k] += qii[k]; temp[k] -= xlf / cpm[k] * qii[k];
                qii[k] = 0.0f;
            }
            if (supcol > 40.0f && qcc[k] > 0.0f) {
                qii[k] += qcc[k]; temp[k] += xlf / cpm[k] * qcc[k];
                qcc[k] = 0.0f;
            }
            if (supcol > 0.0f && qcc[k] > 1.0e-15f) {
                float sc = min(supcol, 50.0f);
                float freeze = min(wsm6_rn_div(wsm6_rn_div(100.0f * (exp(0.66f * sc) - 1.0f)
                    * rho[k], 1000.0f), 3.0e8f) * qcc[k] * qcc[k] * dtcld,
                    qcc[k]);
                qii[k] += freeze; qcc[k] -= freeze;
                temp[k] += xlf / cpm[k] * freeze;
            }
            if (supcol > 0.0f && qrr[k] > 0.0f) {
                float sc = min(supcol, 50.0f);
                float r7 = rr.r3 * rr.r3 * rr.r;
                float freeze = min(20.0f * 9.86960440f * 100.0f * 8.0e6f
                    * 1000.0f / rho[k] * (exp(0.66f * sc) - 1.0f)
                    * r7 * dtcld, qrr[k]);
                qgg[k] += freeze; qrr[k] -= freeze;
                temp[k] += xlf / cpm[k] * freeze;
            }

            // Refresh slopes after phase changes exactly where WRF calls
            // slope_wsm6 for its process block.
            rr = rain_slope(qrr[k], rho[k], denfac);
            ss = snow_slope(qss[k], rho[k], denfac, temp[k], &n0sfac);
            gg = graupel_slope(qgg[k], rho[k], denfac, gc);
            float workw = diffac(xl[k], p[id], temp[k], rho[k], qsw[k]);
            float worki = diffac(2.85e6f, p[id], temp[k], rho[k], qsi[k]);
            float vf = venfac(p[id], temp[k], rho[k]);

            float prevp=0, psdep=0, pgdep=0, praut=0, psaut=0, pgaut=0;
            float pracw=0, praci=0, piacr=0, psaci=0, psacw=0, pracs=0;
            float psacr=0, pgacw=0, paacw=0, pgaci=0, pgacr=0, pgacs=0;
            float pigen=0, pidep=0, pseml=0, pgeml=0, psevp=0, pgevp=0;

            float supsat = max(qvv[k], 1.0e-15f) - qsw[k];
            float satdt = supsat / dtcld;
            if (qcc[k] > 5.02654825e-4f) {
                praut = 6.77389540f * pow(qcc[k], 7.0f/3.0f);
                praut = min(praut, qcc[k]/dtcld);
            }
            if (qrr[k] > 1.0e-9f && qcc[k] > 1.0e-15f)
                pracw = min(2.48133885e10f * rr.r3 * rr.rb * qcc[k]
                              * denfac, qcc[k]/dtcld);
            if (qrr[k] > 0.0f) {
                float coeres = rr.r2 * sqrt(rr.r * rr.rb);
                prevp = (rhw[k]-1.0f) * (3.92070763e7f*rr.r2
                    + 8.25851867e8f*vf*coeres) / workw;
                prevp = prevp < 0.0f
                      ? max(max(prevp, -qrr[k]/dtcld), satdt/2.0f)
                      : min(prevp, satdt/2.0f);
            }

            supcol = 273.15f - temp[k];
            n0sfac = max(min(exp(0.12f*supcol), 5.0e4f), 1.0f);
            supsat = max(qvv[k], 1.0e-15f) - qsi[k];
            satdt = supsat / dtcld;
            int ifsat = 0;
            float xniarg = rho[k]*max(qii[k],1.0e-15f);
            float xni = min(max(5.38e7f*pow(xniarg,0.75f),1.0e3f),1.0e6f);
            float eacrs = exp(-0.07f*supcol);
            float xmi = rho[k]*qii[k]/xni;
            float diameter = min(11.9f*sqrt(max(xmi,0.0f)),500.0e-6f);
            float vt2i = 1.49e4f*pow(max(diameter,0.0f),1.31f);
            float qsum = max(qss[k]+qgg[k],1.0e-15f);
            float vtave = qsum > 1.0e-15f
                        ? (ss.vt*qss[k]+gg.vt*qgg[k])/qsum : 0.0f;

            if (supcol > 0.0f && qii[k] > 1.0e-15f) {
                if (qrr[k] > 1.0e-9f) {
                    float af=2*rr.r3+2*diameter*rr.r2+diameter*diameter*rr.r;
                    praci=3.14159265f*qii[k]*8.0e6f*fabs(rr.vt-vt2i)*af/4;
                    praci*=pow(clamp01(qrr[k]/qii[k]),2); praci=min(praci,qii[k]/dtcld);
                    piacr=9.86960440f*841.9f*8.0e6f*1000.0f*xni*denfac
                         *495.459567f*rr.r3*rr.r3*rr.rb/(24.0f*rho[k]);
                    piacr*=pow(clamp01(qii[k]/qrr[k]),2); piacr=min(piacr,qrr[k]/dtcld);
                }
                if(qss[k]>1.0e-9f){float af=2*ss.r3+2*diameter*ss.r2+diameter*diameter*ss.r;
                    psaci=3.14159265f*qii[k]*eacrs*2.0e6f*n0sfac*fabs(vtave-vt2i)*af/4;
                    psaci=min(psaci,qii[k]/dtcld);}
                if(qgg[k]>1.0e-9f){float egi=exp(-0.07f*supcol);
                    float af=2*gg.r3+2*diameter*gg.r2+diameter*diameter*gg.r;
                    pgaci=3.14159265f*egi*qii[k]*gc.n0g*fabs(vtave-vt2i)*af/4;
                    pgaci=min(pgaci,qii[k]/dtcld);}
            }
            if(qss[k]>1.0e-9f&&qcc[k]>1.0e-15f)
                psacw=min(5.54420857e7f*n0sfac*ss.r3*ss.rb
                    *pow(clamp01(qss[k]/qcc[k]),2)*qcc[k]*denfac,qcc[k]/dtcld);
            if(qgg[k]>1.0e-9f&&qcc[k]>1.0e-15f)
                pgacw=min(gc.pacrg*gg.r3*gg.rb*pow(clamp01(qgg[k]/qcc[k]),2)
                    *qcc[k]*denfac,qcc[k]/dtcld);
            if(qsum>1.0e-15f) paacw=(qss[k]*psacw+qgg[k]*pgacw)/qsum;

            if(qss[k]>1.0e-9f&&qrr[k]>1.0e-9f){
                if(supcol>0){float af=5*ss.r3*ss.r3*rr.r+2*ss.r3*ss.r2*rr.r2
                    +0.5f*ss.r2*ss.r2*rr.r3;
                    pracs=9.86960440f*8.0e6f*2.0e6f*n0sfac*fabs(rr.vt-vtave)
                         *(100.0f/rho[k])*af*pow(clamp01(qrr[k]/qss[k]),2);
                    pracs=min(pracs,qss[k]/dtcld);}
                float af=5*rr.r3*rr.r3*ss.r+2*rr.r3*rr.r2*ss.r2
                    +0.5f*rr.r2*rr.r2*ss.r3;
                psacr=9.86960440f*8.0e6f*2.0e6f*n0sfac*fabs(vtave-rr.vt)
                     *(1000.0f/rho[k])*af*pow(clamp01(qss[k]/qrr[k]),2);
                psacr=min(psacr,qrr[k]/dtcld);
            }
            if(qgg[k]>1.0e-9f&&qrr[k]>1.0e-9f){float af=5*rr.r3*rr.r3*gg.r
                +2*rr.r3*rr.r2*gg.r2+0.5f*rr.r2*rr.r2*gg.r3;
                pgacr=9.86960440f*8.0e6f*gc.n0g*fabs(vtave-rr.vt)
                     *(1000.0f/rho[k])*af*pow(clamp01(qgg[k]/qrr[k]),2);
                pgacr=min(pgacr,qrr[k]/dtcld);}
            pgacs=0.0f;
            if(supcol<=0){
                if(qss[k]>0)pseml=min(max(wsm6_rn_div(4190.0f*supcol*(paacw+psacr), 3.5e5f),
                                               -qss[k]/dtcld),0.0f);
                if(qgg[k]>0)pgeml=min(max(wsm6_rn_div(4190.0f*supcol*(paacw+pgacr), 3.5e5f),
                                               -qgg[k]/dtcld),0.0f);
            }
            if(supcol>0){
                if(qii[k]>0&&!ifsat){pidep=4*diameter*xni*(rhi[k]-1)/worki;
                    float supice=satdt-prevp;
                    pidep=pidep<0?max(max(max(pidep,satdt/2),supice),-qii[k]/dtcld)
                                  :min(min(pidep,satdt/2),supice);
                    if(fabs(prevp+pidep)>=fabs(satdt))ifsat=1;}
                if(qss[k]>0&&!ifsat){float co=ss.r2*sqrt(ss.r*ss.rb);
                    psdep=(rhi[k]-1)*n0sfac*(5.2e6f*ss.r2+1.86818719e7f*vf*co)/worki;
                    float si=satdt-prevp-pidep;
                    psdep=psdep<0?max(max(max(psdep,-qss[k]/dtcld),satdt/2),si)
                                 :min(min(psdep,satdt/2),si);
                    if(fabs(prevp+pidep+psdep)>=fabs(satdt))ifsat=1;}
                if(qgg[k]>0&&!ifsat){float co=gg.r2*sqrt(gg.r*gg.rb);
                    pgdep=(rhi[k]-1)*(gc.precg1*gg.r2+gc.precg2*vf*co)/worki;
                    float si=satdt-prevp-pidep-psdep;
                    pgdep=pgdep<0?max(max(max(pgdep,-qgg[k]/dtcld),satdt/2),si)
                                 :min(min(pgdep,satdt/2),si);
                    if(fabs(prevp+pidep+psdep+pgdep)>=fabs(satdt))ifsat=1;}
                if(supsat>0&&!ifsat){float xni0=1.0e3f*exp(0.1f*supcol);
                    float roqi0=4.92e-11f*pow(xni0,1.33f);
                    float si=satdt-prevp-pidep-psdep-pgdep;
                    pigen=min(min(max(0.0f,(roqi0/rho[k]-fmax0(qii[k]))/dtcld),satdt),si);}
                if(qii[k]>0)psaut=max(0.0f,(8.125e-5f/rho[k]-qii[k])/-dtcld);
                if(qss[k]>0)pgaut=min(max(0.0f,1.0e-3f*exp(-0.09f*supcol)
                                              *(qss[k]-6.0e-4f)),qss[k]/dtcld);
            }
            if(supcol<0){
                if(qss[k]>0&&rhw[k]<1){float co=ss.r2*sqrt(ss.r*ss.rb);
                    psevp=(rhw[k]-1)*n0sfac*(5.2e6f*ss.r2+1.86818719e7f*vf*co)/workw;
                    psevp=min(max(psevp,-qss[k]/dtcld),0.0f);}
                if(qgg[k]>0&&rhw[k]<1){float co=gg.r2*sqrt(gg.r*gg.rb);
                    pgevp=(rhw[k]-1)*(gc.precg1*gg.r2+gc.precg2*vf*co)/workw;
                    pgevp=min(max(pgevp,-qgg[k]/dtcld),0.0f);}
            }

            float d2=(qrr[k]<1e-4f&&qss[k]<1e-4f)?1.0f:0.0f;
            float d3=qrr[k]<1e-4f?1.0f:0.0f;
#define LIMIT(value_, source_, ...) do { float _s=(source_)*dtcld; if(_s>(value_)){float factor=(value_)/_s; __VA_ARGS__;} } while(0)
            if(temp[k]<=273.15f){
                LIMIT(max(1e-15f,qcc[k]),praut+pracw+2*paacw,
                      praut*=factor;pracw*=factor;paacw*=factor;);
                LIMIT(max(1e-15f,qii[k]),psaut-pigen-pidep+praci+psaci+pgaci,
                      psaut*=factor;pigen*=factor;pidep*=factor;praci*=factor;psaci*=factor;pgaci*=factor;);
                LIMIT(max(1e-15f,qrr[k]),-praut-prevp-pracw+piacr+psacr+pgacr,
                      praut*=factor;prevp*=factor;pracw*=factor;piacr*=factor;psacr*=factor;pgacr*=factor;);
                LIMIT(max(1e-15f,qss[k]),-(psdep+psaut-pgaut+paacw+piacr*d3+praci*d3
                      -pracs*(1-d2)+psacr*d2+psaci-pgacs),
                      psdep*=factor;psaut*=factor;pgaut*=factor;paacw*=factor;piacr*=factor;
                      praci*=factor;psaci*=factor;pracs*=factor;psacr*=factor;pgacs*=factor;);
                LIMIT(max(1e-15f,qgg[k]),-(pgdep+pgaut+piacr*(1-d3)+praci*(1-d3)
                      +psacr*(1-d2)+pracs*(1-d2)+pgaci+paacw+pgacr+pgacs),
                      pgdep*=factor;pgaut*=factor;piacr*=factor;praci*=factor;psacr*=factor;
                      pracs*=factor;paacw*=factor;pgaci*=factor;pgacr*=factor;pgacs*=factor;);
                qvv[k]+=-(prevp+psdep+pgdep+pigen+pidep)*dtcld;
                qcc[k]=fmax0(qcc[k]-(praut+pracw+2*paacw)*dtcld);
                qrr[k]=fmax0(qrr[k]+(praut+pracw+prevp-piacr-pgacr-psacr)*dtcld);
                qii[k]=fmax0(qii[k]-(psaut+praci+psaci+pgaci-pigen-pidep)*dtcld);
                qss[k]=fmax0(qss[k]+(psdep+psaut+paacw-pgaut+piacr*d3+praci*d3
                     +psaci-pgacs-pracs*(1-d2)+psacr*d2)*dtcld);
                qgg[k]=fmax0(qgg[k]+(pgdep+pgaut+piacr*(1-d3)+praci*(1-d3)
                     +psacr*(1-d2)+pracs*(1-d2)+pgaci+paacw+pgacr+pgacs)*dtcld);
                float latent=-2.85e6f*(psdep+pgdep+pidep+pigen)-xl[k]*prevp
                    -(2.85e6f-xl[k])*(piacr+2*paacw+pgacr+psacr);
                temp[k]-=latent/cpm[k]*dtcld;
            } else {
                LIMIT(max(1e-15f,qcc[k]),praut+pracw+2*paacw,
                      praut*=factor;pracw*=factor;paacw*=factor;);
                LIMIT(max(1e-15f,qrr[k]),-2*paacw-praut+pseml+pgeml-pracw-prevp,
                      praut*=factor;prevp*=factor;pracw*=factor;paacw*=factor;pseml*=factor;pgeml*=factor;);
                LIMIT(max(1e-9f,qss[k]),pgacs-pseml-psevp,
                      pgacs*=factor;psevp*=factor;pseml*=factor;);
                LIMIT(max(1e-9f,qgg[k]),-(pgacs+pgevp+pgeml),
                      pgacs*=factor;pgevp*=factor;pgeml*=factor;);
                qvv[k]+=-(prevp+psevp+pgevp)*dtcld;
                qcc[k]=fmax0(qcc[k]-(praut+pracw+2*paacw)*dtcld);
                qrr[k]=fmax0(qrr[k]+(praut+pracw+prevp+2*paacw-pseml-pgeml)*dtcld);
                qss[k]=fmax0(qss[k]+(psevp-pgacs+pseml)*dtcld);
                qgg[k]=fmax0(qgg[k]+(pgacs+pgevp+pgeml)*dtcld);
                float latent=-xl[k]*(prevp+psevp+pgevp)-(2.85e6f-xl[k])*(pseml+pgeml);
                temp[k]-=latent/cpm[k]*dtcld;
            }
#undef LIMIT
            float qsat_new=qsat(temp[k],p[id],false);
            float cond=(max(qvv[k],1e-15f)-qsat_new)
                /(1+xl[k]*xl[k]/(461.6f*cpm[k])*qsat_new/(temp[k]*temp[k]));
            float pcond=min(max(cond/dtcld,0.0f),fmax0(qvv[k])/dtcld);
            if(qcc[k]>0&&cond<0)pcond=max(cond,-qcc[k])/dtcld;
            qvv[k]-=pcond*dtcld; qcc[k]=fmax0(qcc[k]+pcond*dtcld);
            temp[k]+=pcond*xl[k]/cpm[k]*dtcld;
            if(qcc[k]<=1e-15f)qcc[k]=0; if(qii[k]<=1e-15f)qii[k]=0;
        }
    }

    for(int k=0;k<nz;++k){
        int id=col+k*stride;
        theta[id]=temp[k]/pii[id]; qv[id]=qvv[k]; qc[id]=qcc[k]; qi[id]=qii[k];
        qr[id]=qrr[k]; qs[id]=qss[k]; qg[id]=qgg[k];
        float rqc=max(1e-12f,qcc[k]*rho[k]);
        float lamc=pow(523.5987756f*3.0e8f/rqc,1.0f/3.0f);
        effc[id]=(rqc>1e-12f)?max(2.51f,min(1.5e6f/lamc,50.0f)):2.49f;
        float itmp=rho[k]*max(qii[k],1e-15f);
        float ni=min(max(5.38e7f*pow(itmp,0.75f),1e3f),1e6f);
        float dia=11.9f*sqrt(max(qii[k]*rho[k]/ni,0.0f));
        effi[id]=(qii[k]*rho[k]>1e-12f)?max(10.01f,min(0.75f*0.163f*dia*1e6f,125.0f)):4.99f;
        float nsfac=max(min(exp(0.12f*(273.15f-temp[k])),5e4f),1.0f);
        float lams=sqrt(sqrt(6.28318531e8f*nsfac/max(qss[k]*rho[k],1e-12f)));
        effs[id]=(qss[k]*rho[k]>1e-12f)?max(25.0f,min(0.5e6f/lams,999.0f)):9.99f;
        effc[id]=max(2.49f,min(effc[id],50.0f));
        effi[id]=max(4.99f,min(effi[id],125.0f));
        effs[id]=max(9.99f,min(effs[id],999.0f));
    }
}


kernel void wsm6_column(
    constant Wsm6ColumnResources &r [[buffer(0)]],
    constant Wsm6ColumnParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny * p.nx);
    if (flat >= ncol || p.nz > WSM6_KMAX) return;
    const int col = int(flat);
    const int stride = p.ny * p.nx;
    wsm6_column_impl(r.theta, r.qv, r.qc, r.qi, r.qr, r.qs, r.qg,
        r.den, r.pressure, r.pii, r.dz, r.rainnc, r.rainncv,
        r.snownc, r.snowncv, r.graupelnc, r.graupelncv, r.sr,
        r.effc, r.effi, r.effs, p.dt, p.hail_opt, p.nz, stride, col);
}

// WRF's moist_physics_prep_em contract: retain pre-MP full theta, form the
// column thermodynamic fields, and derive dz from total geopotential.
struct Wsm6PrepareResources {
    device const float *thb [[id(0)]];
    device const float *thp [[id(1)]];
    device const float *p [[id(2)]];
    device const float *alt [[id(3)]];
    device const float *phb [[id(4)]];
    device const float *php [[id(5)]];
    device float *theta [[id(6)]];
    device float *den [[id(7)]];
    device float *pii [[id(8)]];
    device float *dz [[id(9)]];
    device float *z8w [[id(10)]];
    device float *h_diabatic [[id(11)]];
};
struct Wsm6PrepareParams {
    int nz, ny, nx, nzi, thb_3d, phb_3d;
    float p0, rcp, gravity;
};
static_assert(sizeof(Wsm6PrepareParams) == 36,
              "wsm6_prepare parameter ABI size changed");

kernel void wsm6_prepare(
    constant Wsm6PrepareResources &r [[buffer(0)]],
    constant Wsm6PrepareParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int plane = p.ny * p.nx;
    const int levels = p.nzi;
    if (flat >= uint(levels * plane)) return;
    const int k = int(flat) / plane;
    const int id = int(flat);
    const int thb_id = p.thb_3d ? id : k;
    const int phb_id = p.phb_3d ? id : k;
    const float z = (r.phb[phb_id] + r.php[id]) / p.gravity;
    r.z8w[id] = z;
    if (k < p.nz) {
        const float full_theta = r.thb[thb_id] + r.thp[id];
        r.theta[id] = full_theta;
        r.h_diabatic[id] = full_theta;
        r.den[id] = 1.0f / r.alt[id];
        r.pii[id] = fp32_pressure_power(r.p[id], p.p0, p.rcp);
        const int next = id + plane;
        const int next_phb = p.phb_3d ? next : k + 1;
        const float upper = (r.phb[next_phb] + r.php[next]) / p.gravity;
        r.dz[id] = upper - z;
    }
}

struct Wsm6FinishResources {
    device const float *theta [[id(0)]];
    device float *thp [[id(1)]];
    device float *h_diabatic [[id(2)]];
};
struct Wsm6FinishParams {
    int nz, ny, nx, no_mp_heating;
    float dt, limit;
};
static_assert(sizeof(Wsm6FinishParams) == 24,
              "wsm6_finish parameter ABI size changed");

kernel void wsm6_finish(
    constant Wsm6FinishResources &r [[buffer(0)]],
    constant Wsm6FinishParams &p [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= uint(p.nz * p.ny * p.nx)) return;
    if (p.no_mp_heating != 0) {
        r.h_diabatic[id] = 0.0f;
        return;
    }
    float mpten = r.theta[id] - r.h_diabatic[id];
    if (!isnan(mpten)) {
        mpten = min(p.limit, mpten);
        mpten = max(-p.limit, mpten);
    }
    r.thp[id] += mpten;
    r.h_diabatic[id] = mpten / p.dt;
}
