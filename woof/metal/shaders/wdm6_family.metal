#include <metal_stdlib>
#include "wdm6_column_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Keep explicit CUDA divide sites as separate FP32 operations.
static inline float wdm6_div(float a, float b) { return a / b; }

#define WDM6_KMAX 80

// The limit is the signed counter's first unrepresentable positive value.
// Check the exact FP32 schedule expression before converting it to int.
static inline int wdm6_checked_steps(float value) {
    float count = floor(value);
    if (!isfinite(count) || count < 0.0f || count >= 2147483648.0f) return -1;
    return count < 1.0f ? 1 : (int)count;
}

static inline int wdm6_rain_steps(float den, float dz, float dtcld) {
    // The existing inverse-slope cap bounds future vt, including dry receivers.
    // Its coupled wave speed is <1.25*vt; wave Courant 0.5 gives terminal
    // Courant 0.4. This is the production schedule itself, not a looser estimate.
    float df = sqrt(1.28f / den);
    return wdm6_checked_steps(2.5f * 2.99849272e3f * pow(1.0e-3f, 0.8f)
                             * df / dz * dtcld + 1.0f);
}

static inline float wdm6_max0(float x) { return max(x, 0.0f); }
static inline float wdm6_eff01(float x) {
    // The "reduce collection efficiency" factor min(max(0.,x),1.)**2
    // (B. Wilt, e.g. :1312).
    float c = min(max(x, 0.0f), 1.0f);
    return c * c;
}

// wdm6init :2096-2108 (hail_opt) products; identical to WSM6's because the
// branch sets the same five base constants (n0g/deng/avtg/bvtg/lamdagmax).
struct WdmRimed {
    float n0g, deng, bvtg, pidn0g, pvtg, pacrg, precg1, precg2;
    float rslopegmax, rslopegbmax, rslopeg2max, rslopeg3max;
};

static inline WdmRimed wdm6_rimed_constants(int hail_opt) {
    WdmRimed r;
    if (hail_opt == 1) {
        r.n0g = 4.0e4f; r.deng = 700.0f; r.bvtg = 0.8f;
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
        r.n0g = 4.0e6f; r.deng = 500.0f; r.bvtg = 0.8f;
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

struct WdmSlope {
    float r, rb, r2, r3, vt;
};

// slope_wdm6 rain arm (:2240, :2251-2261, :2284-2288): gamma mu=1 PSD with
// the PROGNOSTIC intercept -- lamdar = ((pidnr*nr)/(q*den))**(1/3),
// pidnr = 4*pi*denr = 1.25663706e4, rslope capped at 1e-3, and the pair of
// fall speeds pvtr (mass, avtr*Gamma(5.8)/24) / pvtrn (number,
// avtr*Gamma(2.8)).
struct WdmRainSlope {
    float r, rb, r2, r3, vt, vtn;
};

struct WdmRainFallout {
    float mass;       // kg m-2 leaving the bottom interface this sub-step
    float number;     // m-2 leaving the bottom interface this sub-step
};

static inline WdmRainFallout wdm6_rain_substep(
    thread float *qr, thread float *nr, thread const float *den, thread const float *dz,
    thread const float *mass_rate, thread const float *number_rate, int nz, float dt)
{
    float incoming_mass = 0.0f, incoming_number = 0.0f;
    for (int k = nz - 1; k >= 0; --k) {
        const float air_mass = den[k] * dz[k];
        const float mass = qr[k] * air_mass;
        const float number = nr[k] * dz[k];
        // The donor bounds its outgoing transfer once. Both adjacent
        // cells use that same amount, including unequal density/depth.
        // This is a conservative flux limit, never a repaired state.
        const float outgoing_mass = min(mass, mass * mass_rate[k] * dt);
        const float outgoing_number = min(number, number * number_rate[k] * dt);
        qr[k] = (mass - outgoing_mass + incoming_mass) / air_mass;
        nr[k] = (number - outgoing_number + incoming_number) / dz[k];
        incoming_mass = outgoing_mass;
        incoming_number = outgoing_number;
    }
    return {incoming_mass, incoming_number};
}

static inline WdmRainSlope wdm6_rain_slope(
    float q, float nr, float den, float denfac) {
    WdmRainSlope s;
    if (q <= 1.0e-9f || nr <= 1.0e-2f) {
        s.r = 2.0e-5f; s.rb = 1.74110113e-4f;
        s.r2 = 4.0e-10f; s.r3 = 8.0e-15f;
    } else {
        float lamdar = exp(log((1.25663706e4f * nr) / (q * den))
                            * 0.33333333f);
        s.r = min(1.0f / lamdar, 1.0e-3f);
        s.rb = pow(s.r, 0.8f); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = 2.99849272e3f * s.rb * denfac;
    s.vtn = 1.41088450e3f * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    if (nr <= 0.0f) s.vtn = 0.0f;
    return s;
}

// slope_wdm6 snow arm (:2241, :2262-2272, :2285/:2289) == slope_snow
// (:2343-2389): fixed n0s with the supercooling intercept factor.
static inline WdmSlope wdm6_snow_slope(
    float q, float den, float denfac, float temp, thread float *n0sfac_out) {
    WdmSlope s;
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
    s.vt = 2.00517852e1f * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// slope_wdm6 graupel arm (:2242, :2273-2283, :2286/:2290) == slope_graup
// (:2391-2435).
static inline WdmSlope wdm6_graupel_slope(
    float q, float den, float denfac, thread const WdmRimed &g) {
    WdmSlope s;
    if (q <= 1.0e-9f) {
        s.r = g.rslopegmax; s.rb = g.rslopegbmax;
        s.r2 = g.rslopeg2max; s.r3 = g.rslopeg3max;
    } else {
        s.r = 1.0f / sqrt(sqrt(g.pidn0g / (q * den)));
        s.rb = pow(s.r, g.bvtg); s.r2 = s.r * s.r; s.r3 = s.r2 * s.r;
    }
    s.vt = g.pvtg * s.rb * denfac;
    if (q <= 0.0f) s.vt = 0.0f;
    return s;
}

// Statement functions (:562-567), identical expressions to WSM6's.
static inline float wdm6_diffus(float t, float p) {
    return 8.794e-5f * pow(t, 1.81f) / p;
}
static inline float wdm6_viscos(float t, float den) {
    return 1.496e-6f * (t * sqrt(t)) / (t + 120.0f) / den;
}
static inline float wdm6_xka(float t, float den) {
    return 1.414e3f * wdm6_viscos(t, den) * den;
}
static inline float wdm6_diffac(float latent, float p, float t,
                                             float den, float qsat) {
    return den * latent * latent / (wdm6_xka(t, den) * 461.6f * t * t)
         + 1.0f / (qsat * wdm6_diffus(t, p));
}
static inline float wdm6_venfac(float p, float t, float den) {
    return pow(wdm6_viscos(t, den) / wdm6_diffus(t, p), 0.3333333f)
         / sqrt(wdm6_viscos(t, den)) * sqrt(sqrt(1.28f / den));
}
// Inline fpvs expansion (:661-695): saturation mixing ratio over water
// (ice=false) / ice (ice=true), with the 0.99p cap, ep2 conversion, and
// the qmin floor applied by the caller exactly as the Fortran does.
static inline float wdm6_fpvs(float t, bool ice) {
    const float ttp = 273.16f;                     // t0c + 0.01
    const float xa = -(1846.4f - 4190.0f) / 461.6f;
    const float xb = xa + 2.5e6f / (461.6f * ttp);
    const float xai = -(1846.4f - 2106.0f) / 461.6f;
    const float xbi = xai + 2.85e6f / (461.6f * ttp);
    float tr = ttp / t;
    float aa = (ice && t < ttp) ? xai : xa;
    float bb = (ice && t < ttp) ? xbi : xb;
    return 610.78f * exp(log(tr) * aa) * exp(bb * (1.0f - tr));
}
static inline float wdm6_qsat(float t, float p, bool ice) {
    float es = min(wdm6_fpvs(t, ice), 0.99f * p);
    float qs = (287.0f / 461.6f) * es / (p - es);
    return max(qs, 1.0e-15f);
}

// nislfv_rain_plm6/plmr arrival-point construction (:2749-2807 pair /
// :2499-2556 single): third-order interface velocity interpolation, the
// raingroup-top and diffusivity CFL limits, arrival coordinates, and the
// deformed density concentrations.  Arrays with nz+1 entries are
// interfaces; qq2 may be null for the single-species (ice) call.
static inline void wdm6_plm_arrival(
    thread const float *qq, thread const float *qq2, thread const float *ww, thread const float *dz,
    int nz, float dt, thread float *zi, thread float *za, thread float *dza,
    thread float *qa, thread float *qa2)
{
    float wi[WDM6_KMAX + 1];
    zi[0] = 0.0f;
    for (int k = 0; k < nz; ++k) zi[k + 1] = zi[k] + dz[k];

    // 2nd-order interface interpolation, then overwritten by the active
    // 3rd-order one (:2760-2774), exactly as the Fortran computes both.
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

    for (int k = 1; k < nz; ++k)                   // :2777-2779
        if (ww[k] == 0.0f) wi[k] = ww[k - 1];
    for (int k = nz - 1; k >= 0; --k) {            // :2782-2788
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

// Monotone PLM interface reconstruction, remap to the regular grid, and
// the rain-out integral (:2843-2939 / :2582-2677).  Returns the fallen
// mass per area (the Fortran's precip).
static inline float wdm6_plm_remap(
    thread const float *qa, thread const float *zi, thread const float *za, thread const float *dza,
    int nz, thread float *qn)
{
    float qmi[WDM6_KMAX + 1], qpi[WDM6_KMAX + 1];
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

    int kb = 0, kt = 0;
    for (int k = 0; k < nz; ++k) {
        kb = (kb > 0) ? kb - 1 : 0;
        kt = (kt > 0) ? kt - 1 : 0;
        if (zi[k] >= za[nz]) break;
        for (int kk = kb; kk < nz; ++kk) {
            if (zi[k] <= za[kk + 1]) { kb = kk; break; }
        }
        for (int kk = kt; kk < nz; ++kk) {
            if (zi[k + 1] <= za[kk]) { kt = kk; break; }
        }
        // DELIBERATE, DOCUMENTED PLM DIVERGENCE from the Fortran.
        // The separate rain-interface correction is documented above.
        // nislfv_rain_plmr:2629 and
        // nislfv_rain_plm6:2891 both do an UNCONDITIONAL `kt = kt - 1`
        // here; this clamps at the first index instead.  The two differ on
        // exactly one input: when find_kt leaves kt at its first index
        // (Fortran kt==1, i.e. zi(k+1) <= za(1)).  There the Fortran's kt
        // falls to 0, which is below kb's floor of 1, so NEITHER the
        // kt==kb nor the kt>kb branch runs and qn(k) keeps its qn=0
        // initialisation -- the layer's mass is dropped on the floor.  The
        // clamp takes the kt==kb piecewise-constant branch and conserves
        // it.  ArWen's standing rule is to implement the DEFINED behaviour
        // where WRF is undefined, and losing mass out of an index guard is
        // not a modelled process, so the clamp is the choice.
        //
        // NOT REACHABLE as far as we can construct, for nz >= 2: the
        // con1 = 0.05 diffusivity limiter above forces dza >= 0.95*dz, so
        // the arrival column cannot compress enough for zi(k+1) to fall to
        // or below za(0), and nz = 1 is below the adapter's
        // VERTICAL_LEVEL_BOUNDS floor of 2 (gpuwm/core/wdm6.py).  It is
        // written down anyway because "unreachable" is an argument, not a
        // measurement, and an oracle campaign that ever does fire it will
        // otherwise be hunting a silent 1-layer mass difference.
        kt = (kt > 0) ? kt - 1 : 0;
        if (kt == kb) {
            float tl = (zi[k] - za[kb]) / dza[kb];
            float th = (zi[k + 1] - za[kb]) / dza[kb];
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qqh = qqd * th * th + qmi[kb] * th;
            float qql = qqd * tl * tl + qmi[kb] * tl;
            qn[k] = (qqh - qql) / (th - tl);
        } else if (kt > kb) {
            float tl = (zi[k] - za[kb]) / dza[kb];
            float qqd = 0.5f * (qpi[kb] - qmi[kb]);
            float qql = qqd * tl * tl + qmi[kb] * tl;
            float zsum = (1.0f - tl) * dza[kb];
            float qsum = (qa[kb] - qql) * dza[kb];
            for (int m = kb + 1; m < kt; ++m) {
                zsum += dza[m]; qsum += qa[m] * dza[m];
            }
            float th = (zi[k + 1] - za[kt]) / dza[kt];
            float dqh = 0.5f * (qpi[kt] - qmi[kt]) * th * th
                      + qmi[kt] * th;
            zsum += th * dza[kt]; qsum += dqh * dza[kt];
            qn[k] = qsum / zsum;
        }
    }

    float precip = 0.0f;
    for (int k = 0; k < nz; ++k) {
        if (za[k] < 0.0f && za[k + 1] < 0.0f) {
            precip += qa[k] * dza[k];
        } else if (za[k] < 0.0f && za[k + 1] >= 0.0f) {
            precip += qa[k] * (0.0f - za[k]);
            break;
        } else {
            break;
        }
    }
    return precip;
}

// Caller block :858-884 + nislfv_rain_plm6 with iter=1 (:2812-2835):
// snow and graupel share the mass-weighted combined fall speed, one
// arrival-velocity iteration averages departure and arrival winds.
static inline void wdm6_sediment_snow_graupel(
    thread float *qs, thread float *qg, thread const float *den, thread const float *dz,
    thread const float *temp, int nz, float dt, thread const WdmRimed &g,
    thread float *precip_s, thread float *precip_g)
{
    float qq[WDM6_KMAX], qq2[WDM6_KMAX], wd[WDM6_KMAX], ww[WDM6_KMAX];
    float qn[WDM6_KMAX], qn2[WDM6_KMAX];
    float zi[WDM6_KMAX + 1], za[WDM6_KMAX + 1];
    float dza[WDM6_KMAX + 1], qa[WDM6_KMAX + 1], qa2[WDM6_KMAX + 1];
    *precip_s = 0.0f; *precip_g = 0.0f;
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        float q1 = qs[k], q2 = qg[k];
        float df = sqrt(1.28f / den[k]);
        float n0s;
        // worka (:858-869): work1(:,2)/(:,3) from slope_wdm6 of the
        // unchanged qs/qg, mass-weighted with the same 1e-15 guard.
        float vs = wdm6_snow_slope(q1, den[k], df, temp[k], &n0s).vt;
        float vg = wdm6_graupel_slope(q2, den[k], df, g).vt;
        float total = max(q1 + q2, 1.0e-15f);
        ww[k] = total > 1.0e-15f ? (vs * q1 + vg * q2) / total : 0.0f;
        wd[k] = ww[k];
        qq[k] = q1 * den[k]; qq2[k] = q2 * den[k];
        allold += qq[k] + qq2[k];
    }
    if (allold <= 0.0f) return;                    // :2740-2746

    wdm6_plm_arrival(qq, qq2, ww, dz, nz, dt, zi, za, dza, qa, qa2);
    // iter = 1 (:2812-2835): one arrival-velocity estimate, mean wind.
    for (int k = 0; k < nz; ++k) {
        float q1 = qa[k] / den[k];
        float q2 = qa2[k] / den[k];
        float df = sqrt(1.28f / den[k]);
        float n0s;
        float vs = wdm6_snow_slope(q1, den[k], df, temp[k], &n0s).vt;
        float vg = wdm6_graupel_slope(q2, den[k], df, g).vt;
        float total = max(q1 + q2, 1.0e-15f);
        float wa = total > 1.0e-15f ? (vs * q1 + vg * q2) / total : 0.0f;
        ww[k] = 0.5f * (wd[k] + wa);
    }
    wdm6_plm_arrival(qq, qq2, ww, dz, nz, dt, zi, za, dza, qa, qa2);

    *precip_s = wdm6_plm_remap(qa, zi, za, dza, nz, qn);
    *precip_g = wdm6_plm_remap(qa2, zi, za, dza, nz, qn2);
    for (int k = 0; k < nz; ++k) {
        qs[k] = max(qn[k] / den[k], 0.0f);       // :873-876
        qg[k] = max(qn2[k] / den[k], 0.0f);
    }
}

// Caller block :955-983 + nislfv_rain_plmr with iter=0 (departure wind,
// no velocity iteration): cloud-ice fallout on its own HDC 5a fall speed.
static inline float wdm6_sediment_ice(
    thread float *qi, thread const float *ww, thread const float *den, thread const float *dz, int nz,
    float dt)
{
    float qq[WDM6_KMAX], qn[WDM6_KMAX];
    float zi[WDM6_KMAX + 1], za[WDM6_KMAX + 1];
    float dza[WDM6_KMAX + 1], qa[WDM6_KMAX + 1];
    float allold = 0.0f;
    for (int k = 0; k < nz; ++k) {
        qq[k] = qi[k] * den[k];
        allold += qq[k];
    }
    if (allold <= 0.0f) return 0.0f;
    wdm6_plm_arrival(qq, nullptr, ww, dz, nz, dt, zi, za, dza, qa,
                     nullptr);
    float precip = wdm6_plm_remap(qa, zi, za, dza, nz, qn);
    for (int k = 0; k < nz; ++k)
        qi[k] = max(qn[k] / den[k], 0.0f);       // :976-980
    return precip;
}

static inline unsigned int wdm6_column_impl(
    device float *theta, device float *qv_g, device float *qc_g, device float *qi_g, device float *qr_g,
    device float *qs_g, device float *qg_g, device float *nn_g, device float *nc_g, device float *nr_g,
    device const float *den_g, device const float *p, device const float *pii,
    device const float *dz_g, device const float *xland,
    device float *rainnc, device float *rainncv, device float *snownc, device float *snowncv,
    device float *graupelnc, device float *graupelncv, device float *sr,
    device float *effc, device float *effi, device float *effs,
    float delt, int hail_opt, int nz, int stride, int col)
{
    if (!isfinite(delt) || delt <= 0.0f) return 1u;
    int loops = wdm6_checked_steps(wdm6_div(delt, 120.0f) + 0.5f);
    if (loops < 0) return 2u;
    float dtcld = delt <= 120.0f ? delt : delt / (float)loops;
    int mstep = 1;
    unsigned int count_flags = 0u;
    const WdmRimed gcn = wdm6_rimed_constants(hail_opt);
    float t[WDM6_KMAX], qv[WDM6_KMAX], qc[WDM6_KMAX], qi[WDM6_KMAX];
    float qr[WDM6_KMAX], qs[WDM6_KMAX], qg[WDM6_KMAX];
    float nn[WDM6_KMAX], nc[WDM6_KMAX], nr[WDM6_KMAX];
    float den[WDM6_KMAX], delz[WDM6_KMAX], cpm[WDM6_KMAX], xl[WDM6_KMAX];
    float qsw[WDM6_KMAX], qsi[WDM6_KMAX], rhw[WDM6_KMAX], rhi[WDM6_KMAX];
    float xni_pre[WDM6_KMAX], work1r[WDM6_KMAX], workn[WDM6_KMAX];
    float rslope3r[WDM6_KMAX];

    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        t[k] = theta[id] * pii[id];
        den[k] = den_g[id]; delz[k] = dz_g[id];
        // This exact schedule depends only on the density, thickness and
        // minor interval, which are constant throughout this scheme call.
        // Check it while loading the column, before any output is written.
        if (!isfinite(den[k]) || den[k] <= 0.0f || !isfinite(delz[k]) || delz[k] <= 0.0f)
            count_flags |= 1u;
        else {
            int steps = wdm6_rain_steps(den[k], delz[k], dtcld);
            if (steps < 0) count_flags |= 2u;
            else if (steps > mstep) mstep = steps;
        }
        // padding for negative values generated by dynamics (:577-588),
        // including the CCN floor/cap of the Oct-2017 KIAPS revision.
        qv[k] = qv_g[id];
        qc[k] = wdm6_max0(qc_g[id]); qi[k] = wdm6_max0(qi_g[id]);
        qr[k] = wdm6_max0(qr_g[id]); qs[k] = wdm6_max0(qs_g[id]);
        qg[k] = wdm6_max0(qg_g[id]);
        nn[k] = min(max(nn_g[id], 1.0e8f), 2.0e10f);
        nc[k] = wdm6_max0(nc_g[id]); nr[k] = wdm6_max0(nr_g[id]);
        // cpm/xl once at scheme entry (:600-605), never refreshed.
        float vap = max(qv[k], 1.0e-15f);
        cpm[k] = 1004.5f * (1.0f - vap) + vap * 1846.4f;
        xl[k] = 2.5e6f - 2343.6f * (t[k] - 273.15f);
    }
    if (count_flags) return count_flags;
    // Land/sea autoconversion threshold (:607-614): xland == 2 is water
    // (module_microphysics_driver.F:2495 "land mask, 1: land, 2: water"),
    // maritime qc0 (xncr0=5e7) against continental qc1 (xncr1=5e8).
    float qcr_col = (xland[col] == 2.0f) ? 8.37758041e-5f : 8.37758041e-4f;

    rainncv[col] = 0.0f; snowncv[col] = 0.0f; graupelncv[col] = 0.0f;
    sr[col] = 0.0f;                                // :625-635

    for (int loop = 0; loop < loops; ++loop) {
        // Saturation state at minor-loop start (:665-695); consumed
        // unrefreshed by the process block exactly as the Fortran's qs/rh
        // arrays are.
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            qsw[k] = wdm6_qsat(t[k], p[id], false);
            qsi[k] = wdm6_qsat(t[k], p[id], true);
            rhw[k] = max(qv[k] / qsw[k], 1.0e-15f);
            rhi[k] = max(qv[k] / qsi[k], 1.0e-15f);
        }
        // Pre-sedimentation ice number column (:771-777): consumed by the
        // ice fall speed (:960) and by nimlt (:1034) AFTER ice has fallen,
        // so it must be the loop-start value, not a live recompute.
        for (int k = 0; k < nz; ++k) {
            float temp = den[k] * max(qi[k], 1.0e-15f);
            temp = sqrt(sqrt(temp * temp * temp));
            xni_pre[k] = min(max(5.38e7f * temp, 1.0e3f), 1.0e6f);
        }

        // ---- rain mass+number fallout: explicit upstream flux with
        // per-column sub-stepping (:790-854) --------------------------------
        for (int k = nz - 1; k >= 0; --k) {
            float df = sqrt(1.28f / den[k]);
            WdmRainSlope rsl = wdm6_rain_slope(qr[k], nr[k], den[k], df);
            work1r[k] = rsl.vt / delz[k];
            workn[k] = rsl.vtn / delz[k];
        }
        float fall_r_sfc = 0.0f;                   // fall(i,kts,1)
        for (int n = 0; n < mstep; ++n) {
            const WdmRainFallout fallout = wdm6_rain_substep(
                qr, nr, den, delz, work1r, workn, nz, dtcld / (float)mstep);
            // Retain the existing precipitation accumulator's units.
            fall_r_sfc += fallout.mass / delz[0] / dtcld;
            // slope_rain refresh for the next sub-step (:840-853).
            for (int k = nz - 1; k >= 0; --k) {
                float df = sqrt(1.28f / den[k]);
                WdmRainSlope rsl = wdm6_rain_slope(qr[k], nr[k], den[k], df);
                work1r[k] = rsl.vt / delz[k];
                workn[k] = rsl.vtn / delz[k];
            }
        }

        // ---- snow + graupel paired semi-Lagrangian fallout (:858-884) -----
        float ps_sfc_mass, pg_sfc_mass;
        wdm6_sediment_snow_graupel(qs, qg, den, delz, t, nz, dtcld, gcn,
                                   &ps_sfc_mass, &pg_sfc_mass);
        float fall2_sfc = ps_sfc_mass / delz[0] / dtcld;   // :881-884
        float fall3_sfc = pg_sfc_mass / delz[0] / dtcld;

        // ---- psmlt/pgmlt with their nr transfers (:885-951) ---------------
        // Consumes the :893 slope_wdm6 refresh; rain rslope3 from that
        // refresh (post-sediment qr/nr, PRE-melt) must survive to pgfrz
        // (:1081/:1089), so it is captured here.
        for (int k = nz - 1; k >= 0; --k) {
            int id = col + k * stride;
            float df = sqrt(1.28f / den[k]);
            WdmRainSlope rr = wdm6_rain_slope(qr[k], nr[k], den[k], df);
            rslope3r[k] = rr.r3;
            float n0sfac;
            WdmSlope ss = wdm6_snow_slope(qs[k], den[k], df, t[k], &n0sfac);
            WdmSlope gg = wdm6_graupel_slope(qg[k], den[k], df, gcn);
            if (t[k] > 273.15f) {
                float xlf = 3.5e5f;
                float work2 = wdm6_venfac(p[id], t[k], den[k]);
                if (qs[k] > 0.0f) {
                    float coeres = ss.r2 * sqrt(ss.r * ss.rb);
                    float psmlt = wdm6_div(wdm6_xka(t[k], den[k]), xlf)
                        * (273.15f - t[k]) * 3.14159265f / 2.0f
                        * n0sfac * (5.2e6f * ss.r2
                                    + 1.86818719e7f * work2 * coeres)
                        / den[k];
                    psmlt = min(max(psmlt * dtcld, -qs[k]), 0.0f);
                    if (qs[k] > 1.0e-9f) {         // nsmlt (:917-920)
                        float sfac = ss.r * 2.0e6f * n0sfac / qs[k];
                        nr[k] = nr[k] - sfac * psmlt;
                    }
                    qs[k] += psmlt; qr[k] -= psmlt;
                    t[k] += xlf / cpm[k] * psmlt;
                }
                if (qg[k] > 0.0f) {
                    // pgmlt reads t AFTER the psmlt update (:932), with
                    // work2 from the block-entry t (:906).
                    float coeres = gg.r2 * sqrt(gg.r * gg.rb);
                    float pgmlt = wdm6_div(wdm6_xka(t[k], den[k]), xlf)
                        * (273.15f - t[k])
                        * (gcn.precg1 * gg.r2
                           + gcn.precg2 * work2 * coeres) / den[k];
                    pgmlt = min(max(pgmlt * dtcld, -qg[k]), 0.0f);
                    if (qg[k] > 1.0e-9f) {         // ngmlt (:940-943)
                        float gfac = gg.r * gcn.n0g / qg[k];
                        nr[k] = nr[k] - gfac * pgmlt;
                    }
                    qg[k] += pgmlt; qr[k] -= pgmlt;
                    t[k] += xlf / cpm[k] * pgmlt;
                }
            }
        }

        // ---- cloud-ice fallout, HDC 5a speed on the PRE-sediment xni
        // (:955-983) --------------------------------------------------------
        {
            float wic[WDM6_KMAX];
            for (int k = nz - 1; k >= 0; --k) {
                if (qi[k] <= 0.0f) {
                    wic[k] = 0.0f;
                } else {
                    float xmi = den[k] * qi[k] / xni_pre[k];
                    float diameter = max(
                        min(11.9f * sqrt(xmi), 500.0e-6f), 1.0e-25f);
                    wic[k] = 1.49e4f * exp(log(diameter) * 1.31f);
                }
            }
            float pi_ice = wdm6_sediment_ice(qi, wic, den, delz, nz, dtcld);
            float fallc_sfc = pi_ice / delz[0] / dtcld;    // :981-983

            // ---- surface precipitation accounting (:988-1017) -------------
            float fallsum = fall_r_sfc + fall2_sfc + fall3_sfc + fallc_sfc;
            float fallsum_qsi = fall2_sfc + fallc_sfc;
            float fallsum_qg = fall3_sfc;
            if (fallsum > 0.0f) {
                float step = wdm6_div(fallsum * delz[0], 1000.0f) * dtcld * 1000.0f;
                rainncv[col] += step;
                rainnc[col] += step;
            }
            if (fallsum_qsi > 0.0f) {
                float step = wdm6_div(fallsum_qsi * delz[0], 1000.0f) * dtcld
                           * 1000.0f;
                snowncv[col] += step;
                snownc[col] += step;
            }
            if (fallsum_qg > 0.0f) {
                float step = wdm6_div(fallsum_qg * delz[0], 1000.0f) * dtcld
                           * 1000.0f;
                graupelncv[col] += step;
                graupelnc[col] += step;
            }
            if (fallsum > 0.0f)                    // :1012-1013
                sr[col] = (snowncv[col] + graupelncv[col])
                        / (rainncv[col] + 1.0e-12f);
        }

        // ---- instantaneous melting/freezing (:1023-1104) -------------------
        for (int k = 0; k < nz; ++k) {
            // rslopec from the loop-start qc/nc (:759-769): nothing has
            // touched qc/nc yet this minor loop, so the live values equal
            // the Fortran's precomputed array as long as they are read
            // BEFORE this iteration's own updates.
            float rslopec3;
            if (qc[k] <= 1.0e-15f || nc[k] <= 1.0e1f) {
                rslopec3 = 8.0e-18f;               // rslopec3max
            } else {
                float lamdac = exp(log((5.23598776e2f * nc[k])
                                         / (qc[k] * den[k]))
                                    * 0.33333333f);
                float rc1 = 1.0f / lamdac;
                rslopec3 = rc1 * rc1 * rc1;
            }
            float supcol = 273.15f - t[k];
            float xlf = 2.85e6f - xl[k];
            if (supcol < 0.0f) xlf = 3.5e5f;
            if (supcol < 0.0f && qi[k] > 0.0f) {   // pimlt/nimlt
                qc[k] += qi[k];
                nc[k] += xni_pre[k];
                t[k] -= xlf / cpm[k] * qi[k];
                qi[k] = 0.0f;
            }
            if (supcol > 40.0f && qc[k] > 0.0f) {  // pihmf/nihmf
                qi[k] += qc[k];
                if (nc[k] > 0.0f) nc[k] = 0.0f;
                t[k] += xlf / cpm[k] * qc[k];
                qc[k] = 0.0f;
            }
            if (supcol > 0.0f && qc[k] > 1.0e-15f) {   // pihtf/nihtf
                float supcolt = min(supcol, 70.0f);
                float pfrzdtc = min(
                    wdm6_div(9.86960440f * 100.0f * (exp(0.66f * supcolt) - 1.0f)
                    * 1000.0f / den[k] * nc[k] * rslopec3 * rslopec3,
                    18.0f) * dtcld, qc[k]);
                if (nc[k] > 1.0e1f) {
                    float nfrzdtc = min(
                        wdm6_div(3.14159265f * 100.0f
                        * (exp(0.66f * supcolt) - 1.0f) * nc[k]
                        * rslopec3, 6.0f) * dtcld, nc[k]);
                    nc[k] -= nfrzdtc;
                }
                qi[k] += pfrzdtc;
                t[k] += xlf / cpm[k] * pfrzdtc;
                qc[k] -= pfrzdtc;
            }
            if (supcol > 0.0f && qr[k] > 0.0f) {   // pgfrz/ngfrz
                float supcolt = min(supcol, 70.0f);
                // rslope3(i,k,1) is the :893 refresh captured above.
                float pfrzdtr = min(
                    140.0f * 9.86960440f * 100.0f * nr[k]
                    * 1000.0f / den[k] * (exp(0.66f * supcolt) - 1.0f)
                    * rslope3r[k] * rslope3r[k] * dtcld, qr[k]);
                if (nr[k] > 1.0e-2f) {
                    float nfrzdtr = min(
                        4.0f * 3.14159265f * 100.0f * nr[k]
                        * (exp(0.66f * supcolt) - 1.0f) * rslope3r[k]
                        * dtcld, nr[k]);
                    nr[k] -= nfrzdtr;
                }
                qg[k] += pfrzdtr;
                t[k] += xlf / cpm[k] * pfrzdtr;
                qr[k] -= pfrzdtr;
            }
            nc[k] = max(nc[k], 0.0f);            // :1099-1104
            nr[k] = max(nr[k], 0.0f);
        }

        // ---- warm + cold process rates, conservation, and update
        // (:1109-1881).  Every Fortran loop in this span is pointwise, so
        // the fusion preserves each cell's exact statement order. ----------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            float denfac = sqrt(1.28f / den[k]);
            WdmRainSlope rr = wdm6_rain_slope(qr[k], nr[k], den[k], denfac);
            float n0sfac;
            WdmSlope ss = wdm6_snow_slope(qs[k], den[k], denfac, t[k],
                                          &n0sfac);
            WdmSlope gg = wdm6_graupel_slope(qg[k], den[k], denfac, gcn);
            // mean-volume diameters (:1125/:1140) and cloud slope refresh.
            float avedia2 = rr.r * 2.88449914f;    // 24**(1/3)
            float rslopec, rslopec2, rslopec3;
            if (qc[k] <= 1.0e-15f || nc[k] <= 1.0e1f) {
                rslopec = 2.0e-6f; rslopec2 = 4.0e-12f; rslopec3 = 8.0e-18f;
            } else {
                float lamdac = exp(log((5.23598776e2f * nc[k])
                                         / (qc[k] * den[k]))
                                    * 0.33333333f);
                rslopec = 1.0f / lamdac;
                rslopec2 = rslopec * rslopec;
                rslopec3 = rslopec2 * rslopec;
            }
            float avedia1 = rslopec;
            float workw = wdm6_diffac(xl[k], p[id], t[k], den[k], qsw[k]);
            float worki = wdm6_diffac(2.85e6f, p[id], t[k], den[k], qsi[k]);
            float vf = wdm6_venfac(p[id], t[k], den[k]);

            float prevp=0, psdep=0, pgdep=0, praut=0, psaut=0, pgaut=0;
            float pracw=0, praci=0, piacr=0, psaci=0, psacw=0, pracs=0;
            float psacr=0, pgacw=0, paacw=0, pgaci=0, pgacr=0, pgacs=0;
            float pigen=0, pidep=0, pseml=0, pgeml=0, psevp=0, pgevp=0;
            float nraut=0, nracw=0, nccol=0, nrcol=0, nsacw=0, ngacw=0;
            float niacr=0, nsacr=0, ngacr=0, naacw=0, nseml=0, ngeml=0;

            // ==== warm rain (:1160-1259) ====
            float supsat = max(qv[k], 1.0e-15f) - qsw[k];
            float satdt = supsat / dtcld;
            float lencon = 2.7e-2f * den[k] * qc[k]
                * (1.0e20f / 16.0f * rslopec2 * rslopec2 - 0.4f);
            float lenconcr = max(1.2f * lencon, 1.0e-9f);
            if (qc[k] > qcr_col && nc[k] > 1.0e1f) {       // praut/nraut
                praut = 4.53466878e3f * pow(qc[k], 7.0f / 3.0f)
                      * pow(nc[k], -1.0f / 3.0f);
                praut = min(praut, qc[k] / dtcld);
                nraut = 3.5e9f * den[k] * praut;
                if (qr[k] > lenconcr)
                    nraut = nr[k] / qr[k] * praut;
                nraut = min(nraut, nc[k] / dtcld);
            }
            if (qr[k] >= lenconcr) {                       // pracw/nracw
                if (avedia2 >= 1.0e-4f) {
                    nracw = min(3.03e3f * nc[k] * nr[k]
                                  * (rslopec3 + 24.0f * rr.r3),
                                  nc[k] / dtcld);
                    pracw = min(3.14159265f / 6.0f
                                  * (1000.0f / den[k]) * 3.03e3f
                                  * nc[k] * nr[k] * rslopec3
                                  * (2.0f * rslopec3 + 24.0f * rr.r3),
                                  qc[k] / dtcld);
                } else {
                    nracw = min(2.59e15f * nc[k] * nr[k]
                                  * (2.0f * rslopec3 * rslopec3
                                     + 5040.0f * rr.r3 * rr.r3),
                                  nc[k] / dtcld);
                    pracw = min(3.14159265f / 6.0f
                                  * (1000.0f / den[k]) * 2.59e15f
                                  * nc[k] * nr[k] * rslopec3
                                  * (6.0f * rslopec3 * rslopec3
                                     + 5040.0f * rr.r3 * rr.r3),
                                  qc[k] / dtcld);
                }
            }
            if (avedia1 >= 1.0e-4f) {                      // nccol
                nccol = 3.03e3f * nc[k] * nc[k] * rslopec3;
            } else {
                nccol = 2.0f * 2.59e15f * nc[k] * nc[k]
                      * rslopec3 * rslopec3;
            }
            if (qr[k] >= lenconcr) {                       // nrcol
                if (avedia2 < 1.0e-4f) {
                    nrcol = 5040.0f * 2.59e15f * nr[k] * nr[k]
                          * rr.r3 * rr.r3;
                } else if (avedia2 < 6.0e-4f) {
                    nrcol = 24.0f * 3.03e3f * nr[k] * nr[k] * rr.r3;
                } else if (avedia2 < 2000.0e-6f) {
                    float coecol = -2.5e3f * (avedia2 - 6.0e-4f);
                    nrcol = 24.0f * exp(coecol) * 3.03e3f
                          * nr[k] * nr[k] * rr.r3;
                } else {
                    nrcol = 0.0f;
                }
            }
            if (qr[k] > 0.0f) {                            // prevp/nrevp
                float coeres = rr.r * sqrt(rr.r * rr.rb);
                prevp = (rhw[k] - 1.0f) * nr[k]
                      * (9.80176908f * rr.r
                         + 2.99269555e2f * vf * coeres) / workw;
                if (prevp < 0.0f) {
                    prevp = max(prevp, -qr[k] / dtcld);
                    prevp = max(prevp, satdt / 2.0f);
                    if (prevp == -qr[k] / dtcld) {         // :1249-1252
                        nn[k] += nr[k];
                        nr[k] = 0.0f;
                    }
                } else {
                    // DELIBERATE, DOCUMENTED DIVERGENCE from :1255 (A144).
                    // This branch caps CONDENSATION at half the water
                    // supersaturation.  WRF also enters it with a rate of
                    // exactly zero -- rain carrying no number (nr = 0) makes
                    // the product -0, and -0 < 0 is false -- and in
                    // subsaturated air min(0, satdt/2) then turns "no drops"
                    // into evaporation of half the saturation deficit,
                    // never bounded by qr (that bound, :1243, sits in the
                    // other branch).  pidep, psdep, pgdep and pigen count
                    // that vapour source in supice (:1527 on) and deposit
                    // against it; the rain limiter (:1670-1680) shrinks
                    // prevp back to the tiny qr but never revisits the
                    // deposition, so vapour goes negative.  WRF's own
                    // Fortran does exactly that on the captured column
                    // (docs/wdm6_oracle_known_deltas.md, section 6).  The
                    // cap keeps its meaning and never changes the sign: a
                    // zero rate stays zero, and the numberless rain joins
                    // the cloud water in the di82 collapse at :1938.
                    prevp = max(min(prevp, satdt / 2.0f), 0.0f);
                }
            }

            // ==== cold rain (:1274-1625) ====
            float supcol = 273.15f - t[k];
            n0sfac = max(min(exp(0.12f * supcol), 5.0e4f), 1.0f);
            supsat = max(qv[k], 1.0e-15f) - qsi[k];
            satdt = supsat / dtcld;
            int ifsat = 0;
            float temp = den[k] * max(qi[k], 1.0e-15f);
            temp = sqrt(sqrt(temp * temp * temp));
            float xni = min(max(5.38e7f * temp, 1.0e3f), 1.0e6f);
            float eacrs = exp(0.07f * (-supcol));
            float xmi = den[k] * qi[k] / xni;
            float diameter = min(11.9f * sqrt(xmi), 500.0e-6f);
            float vt2i = 1.49e4f * pow(diameter, 1.31f);
            float vt2r = 2.99849272e3f * rr.rb * denfac;
            float vt2s = 2.00517852e1f * ss.rb * denfac;
            float vt2g = gcn.pvtg * gg.rb * denfac;
            float qsum = max(qs[k] + qg[k], 1.0e-15f);
            float vt2ave = qsum > 1.0e-15f
                         ? (vt2s * qs[k] + vt2g * qg[k]) / qsum : 0.0f;
            if (supcol > 0.0f && qi[k] > 1.0e-15f) {
                if (qr[k] > 1.0e-9f) {             // praci/piacr
                    float acrfac = 6.0f * rr.r2 + 4.0f * diameter * rr.r
                                 + diameter * diameter;
                    praci = 3.14159265f * qi[k] * nr[k]
                          * abs(vt2r - vt2i) * acrfac / 4.0f;
                    praci *= wdm6_eff01(qr[k] / qi[k]);
                    praci = min(praci, qi[k] / dtcld);
                    piacr = wdm6_div(9.86960440f * 841.9f * nr[k] * 1000.0f * xni
                          * denfac * 3.36666752e3f * rr.r3 * rr.r2
                          * rr.rb, 24.0f) / den[k];
                    piacr *= wdm6_eff01(qi[k] / qr[k]);
                    piacr = min(piacr, qr[k] / dtcld);
                }
                if (nr[k] > 1.0e-2f) {             // niacr (:1329-1335)
                    niacr = 3.14159265f * 841.9f * nr[k] * xni * denfac
                          * 1.78173289e1f * rr.r2 * rr.rb / 4.0f;
                    niacr *= wdm6_eff01(qi[k] / qr[k]);
                    niacr = min(niacr, nr[k] / dtcld);
                }
                if (qs[k] > 1.0e-9f) {             // psaci
                    float acrfac = 2.0f * ss.r3 + 2.0f * diameter * ss.r2
                                 + diameter * diameter * ss.r;
                    psaci = 3.14159265f * qi[k] * eacrs * 2.0e6f * n0sfac
                          * abs(vt2ave - vt2i) * acrfac / 4.0f;
                    psaci = min(psaci, qi[k] / dtcld);
                }
                if (qg[k] > 1.0e-9f) {             // pgaci
                    float egi = exp(0.07f * (-supcol));
                    float acrfac = 2.0f * gg.r3 + 2.0f * diameter * gg.r2
                                 + diameter * diameter * gg.r;
                    pgaci = 3.14159265f * egi * qi[k] * gcn.n0g
                          * abs(vt2ave - vt2i) * acrfac / 4.0f;
                    pgaci = min(pgaci, qi[k] / dtcld);
                }
            }
            if (qs[k] > 1.0e-9f && qc[k] > 1.0e-15f)       // psacw
                psacw = min(5.54420857e7f * n0sfac * ss.r3 * ss.rb
                              * wdm6_eff01(qs[k] / qc[k])
                              * qc[k] * denfac, qc[k] / dtcld);
            if (qs[k] > 1.0e-9f && nc[k] > 1.0e1f)         // nsacw
                nsacw = min(5.54420857e7f * n0sfac * ss.r3 * ss.rb
                              * wdm6_eff01(qs[k] / qc[k])
                              * nc[k] * denfac, nc[k] / dtcld);
            if (qg[k] > 1.0e-9f && qc[k] > 1.0e-15f)       // pgacw
                pgacw = min(gcn.pacrg * gg.r3 * gg.rb * qc[k]
                              * wdm6_eff01(qg[k] / qc[k])
                              * denfac, qc[k] / dtcld);
            if (qg[k] > 1.0e-9f && nc[k] > 1.0e1f)         // ngacw
                ngacw = min(gcn.pacrg * gg.r3 * gg.rb * nc[k]
                              * wdm6_eff01(qg[k] / qc[k])
                              * denfac, nc[k] / dtcld);
            if (qsum > 1.0e-15f) {                         // paacw/naacw
                paacw = (qs[k] * psacw + qg[k] * pgacw) / qsum;
                naacw = (qs[k] * nsacw + qg[k] * ngacw) / qsum;
            }
            if (qs[k] > 1.0e-9f && qr[k] > 1.0e-9f) {
                if (supcol > 0.0f) {               // pracs
                    float acrfac = 5.0f * ss.r3 * ss.r3
                                 + 4.0f * ss.r3 * ss.r2 * rr.r
                                 + 1.5f * ss.r2 * ss.r2 * rr.r2;
                    pracs = 9.86960440f * nr[k] * 2.0e6f * n0sfac
                          * abs(vt2r - vt2ave) * (100.0f / den[k])
                          * acrfac;
                    pracs *= wdm6_eff01(qr[k] / qs[k]);
                    pracs = min(pracs, qs[k] / dtcld);
                }
                // psacr
                float acrfac = 30.0f * rr.r3 * rr.r2 * ss.r
                             + 10.0f * rr.r2 * rr.r2 * ss.r2
                             + 2.0f * rr.r3 * ss.r3;
                psacr = 9.86960440f * nr[k] * 2.0e6f * n0sfac
                      * abs(vt2ave - vt2r) * (1000.0f / den[k])
                      * acrfac;
                psacr *= wdm6_eff01(qs[k] / qr[k]);
                psacr = min(psacr, qr[k] / dtcld);
            }
            if (qs[k] > 1.0e-9f && nr[k] > 1.0e-2f) {      // nsacr
                float acrfac = 1.5f * rr.r2 * ss.r + 1.0f * rr.r * ss.r2
                             + 0.5f * ss.r3;
                nsacr = 3.14159265f * nr[k] * 2.0e6f * n0sfac
                      * abs(vt2ave - vt2r) * acrfac;
                nsacr *= wdm6_eff01(qs[k] / qr[k]);
                nsacr = min(nsacr, nr[k] / dtcld);
            }
            if (qg[k] > 1.0e-9f && qr[k] > 1.0e-9f) {      // pgacr
                float acrfac = 30.0f * rr.r3 * rr.r2 * gg.r
                             + 10.0f * rr.r2 * rr.r2 * gg.r2
                             + 2.0f * rr.r3 * gg.r3;
                pgacr = 9.86960440f * nr[k] * gcn.n0g
                      * abs(vt2ave - vt2r) * (1000.0f / den[k])
                      * acrfac;
                pgacr *= wdm6_eff01(qg[k] / qr[k]);
                pgacr = min(pgacr, qr[k] / dtcld);
            }
            if (qg[k] > 1.0e-9f && nr[k] > 1.0e-2f) {      // ngacr
                float acrfac = 1.5f * rr.r2 * gg.r + 1.0f * rr.r * gg.r2
                             + 0.5f * gg.r3;
                ngacr = 3.14159265f * nr[k] * gcn.n0g
                      * abs(vt2ave - vt2r) * acrfac;
                ngacr *= wdm6_eff01(qg[k] / qr[k]);
                ngacr = min(ngacr, nr[k] / dtcld);
            }
            pgacs = 0.0f;                                  // :1484-1486
            if (supcol <= 0.0f) {                          // pseml/pgeml
                float xlf = 3.5e5f;
                if (qs[k] > 0.0f) {
                    pseml = min(max(wdm6_div(4190.0f * supcol
                                        * (paacw + psacr), xlf),
                                        -qs[k] / dtcld), 0.0f);
                    if (qs[k] > 1.0e-9f) {         // nseml (:1500-1503)
                        float sfac = ss.r * 2.0e6f * n0sfac / qs[k];
                        nseml = -sfac * pseml;
                    }
                }
                if (qg[k] > 0.0f) {
                    pgeml = min(max(wdm6_div(4190.0f * supcol
                                        * (paacw + pgacr), xlf),
                                        -qg[k] / dtcld), 0.0f);
                    if (qg[k] > 1.0e-9f) {         // ngeml (:1515-1518)
                        float gfac = gg.r * gcn.n0g / qg[k];
                        ngeml = -gfac * pgeml;
                    }
                }
            }
            if (supcol > 0.0f) {
                if (qi[k] > 0.0f && ifsat != 1) {          // pidep
                    pidep = 4.0f * diameter * xni * (rhi[k] - 1.0f)
                          / worki;
                    float supice = satdt - prevp;
                    if (pidep < 0.0f) {
                        pidep = max(max(pidep, satdt / 2.0f), supice);
                        pidep = max(pidep, -qi[k] / dtcld);
                    } else {
                        pidep = min(min(pidep, satdt / 2.0f), supice);
                    }
                    if (abs(prevp + pidep) >= abs(satdt)) ifsat = 1;
                }
                if (qs[k] > 0.0f && ifsat != 1) {          // psdep
                    float coeres = ss.r2 * sqrt(ss.r * ss.rb);
                    psdep = (rhi[k] - 1.0f) * n0sfac
                          * (5.2e6f * ss.r2
                             + 1.86818719e7f * vf * coeres) / worki;
                    float supice = satdt - prevp - pidep;
                    if (psdep < 0.0f) {
                        psdep = max(psdep, -qs[k] / dtcld);
                        psdep = max(max(psdep, satdt / 2.0f), supice);
                    } else {
                        psdep = min(min(psdep, satdt / 2.0f), supice);
                    }
                    if (abs(prevp + pidep + psdep) >= abs(satdt))
                        ifsat = 1;
                }
                if (qg[k] > 0.0f && ifsat != 1) {          // pgdep
                    float coeres = gg.r2 * sqrt(gg.r * gg.rb);
                    pgdep = (rhi[k] - 1.0f)
                          * (gcn.precg1 * gg.r2
                             + gcn.precg2 * vf * coeres) / worki;
                    float supice = satdt - prevp - pidep - psdep;
                    if (pgdep < 0.0f) {
                        pgdep = max(pgdep, -qg[k] / dtcld);
                        pgdep = max(max(pgdep, satdt / 2.0f), supice);
                    } else {
                        pgdep = min(min(pgdep, satdt / 2.0f), supice);
                    }
                    if (abs(prevp + pidep + psdep + pgdep)
                        >= abs(satdt)) ifsat = 1;
                }
                if (supsat > 0.0f && ifsat != 1) {         // pigen
                    float supice = satdt - prevp - pidep - psdep - pgdep;
                    float xni0 = 1.0e3f * exp(0.1f * supcol);
                    float roqi0 = 4.92e-11f * pow(xni0, 1.33f);
                    pigen = max(0.0f, (roqi0 / den[k]
                                         - max(qi[k], 0.0f)) / dtcld);
                    pigen = min(min(pigen, satdt), supice);
                }
                if (qi[k] > 0.0f) {                        // psaut
                    float qimax = 8.125e-5f / den[k];
                    psaut = max(0.0f, (qi[k] - qimax) / dtcld);
                }
                if (qs[k] > 0.0f) {                        // pgaut
                    float alpha2 = 1.0e-3f * exp(0.09f * (-supcol));
                    pgaut = min(max(0.0f, alpha2 * (qs[k] - 6.0e-4f)),
                                  qs[k] / dtcld);
                }
            }
            if (supcol < 0.0f) {                           // psevp/pgevp
                if (qs[k] > 0.0f && rhw[k] < 1.0f) {
                    float coeres = ss.r2 * sqrt(ss.r * ss.rb);
                    psevp = (rhw[k] - 1.0f) * n0sfac
                          * (5.2e6f * ss.r2
                             + 1.86818719e7f * vf * coeres) / workw;
                    psevp = min(max(psevp, -qs[k] / dtcld), 0.0f);
                }
                if (qg[k] > 0.0f && rhw[k] < 1.0f) {
                    float coeres = gg.r2 * sqrt(gg.r * gg.rb);
                    pgevp = (rhw[k] - 1.0f)
                          * (gcn.precg1 * gg.r2
                             + gcn.precg2 * vf * coeres) / workw;
                    pgevp = min(max(pgevp, -qg[k] / dtcld), 0.0f);
                }
            }

            // ==== conservation limiting and the coupled update
            // (:1632-1881) ====
            float delta2 = (qr[k] < 1.0e-4f && qs[k] < 1.0e-4f)
                         ? 1.0f : 0.0f;
            float delta3 = (qr[k] < 1.0e-4f) ? 1.0f : 0.0f;
#define WDM6_LIMIT(value_, source_, ...) do { \
    float _src = (source_) * dtcld; \
    if (_src > (value_)) { float factor = (value_) / _src; __VA_ARGS__; } \
} while (0)
            if (t[k] <= 273.15f) {
                WDM6_LIMIT(max(1.0e-15f, qc[k]),
                           praut + pracw + paacw + paacw,
                           praut *= factor; pracw *= factor;
                           paacw *= factor;);
                WDM6_LIMIT(max(1.0e-15f, qi[k]),
                           psaut - pigen - pidep + praci + psaci + pgaci,
                           psaut *= factor; pigen *= factor;
                           pidep *= factor; praci *= factor;
                           psaci *= factor; pgaci *= factor;);
                WDM6_LIMIT(max(1.0e-15f, qr[k]),
                           -praut - prevp - pracw + piacr + psacr + pgacr,
                           praut *= factor; prevp *= factor;
                           pracw *= factor; piacr *= factor;
                           psacr *= factor; pgacr *= factor;);
                WDM6_LIMIT(max(1.0e-15f, qs[k]),
                           -(psdep + psaut - pgaut + paacw
                             + piacr * delta3 + praci * delta3
                             - pracs * (1.0f - delta2) + psacr * delta2
                             + psaci - pgacs),
                           psdep *= factor; psaut *= factor;
                           pgaut *= factor; paacw *= factor;
                           piacr *= factor; praci *= factor;
                           psaci *= factor; pracs *= factor;
                           psacr *= factor; pgacs *= factor;);
                WDM6_LIMIT(max(1.0e-15f, qg[k]),
                           -(pgdep + pgaut
                             + piacr * (1.0f - delta3)
                             + praci * (1.0f - delta3)
                             + psacr * (1.0f - delta2)
                             + pracs * (1.0f - delta2)
                             + pgaci + paacw + pgacr + pgacs),
                           pgdep *= factor; pgaut *= factor;
                           piacr *= factor; praci *= factor;
                           psacr *= factor; pracs *= factor;
                           paacw *= factor; pgaci *= factor;
                           pgacr *= factor; pgacs *= factor;);
                WDM6_LIMIT(max(1.0e1f, nc[k]),
                           nraut + nccol + nracw + naacw + naacw,
                           nraut *= factor; nccol *= factor;
                           nracw *= factor; naacw *= factor;);
                WDM6_LIMIT(max(1.0e-2f, nr[k]),
                           -nraut + nrcol + niacr + nsacr + ngacr,
                           nraut *= factor; nrcol *= factor;
                           niacr *= factor; nsacr *= factor;
                           ngacr *= factor;);
                float work2k = -(prevp + psdep + pgdep + pigen + pidep);
                qv[k] += work2k * dtcld;
                qc[k] = max(qc[k] - (praut + pracw + paacw + paacw)
                              * dtcld, 0.0f);
                qr[k] = max(qr[k] + (praut + pracw + prevp - piacr
                                       - pgacr - psacr) * dtcld, 0.0f);
                qi[k] = max(qi[k] - (psaut + praci + psaci + pgaci
                                       - pigen - pidep) * dtcld, 0.0f);
                qs[k] = max(qs[k] + (psdep + psaut + paacw - pgaut
                                       + piacr * delta3 + praci * delta3
                                       + psaci - pgacs
                                       - pracs * (1.0f - delta2)
                                       + psacr * delta2) * dtcld, 0.0f);
                qg[k] = max(qg[k] + (pgdep + pgaut
                                       + piacr * (1.0f - delta3)
                                       + praci * (1.0f - delta3)
                                       + psacr * (1.0f - delta2)
                                       + pracs * (1.0f - delta2)
                                       + pgaci + paacw + pgacr + pgacs)
                              * dtcld, 0.0f);
                nc[k] = max(nc[k] + (-nraut - nccol - nracw - naacw
                                       - naacw) * dtcld, 0.0f);
                nr[k] = max(nr[k] + (nraut - nrcol - niacr - nsacr
                                       - ngacr) * dtcld, 0.0f);
                float xlf = 2.85e6f - xl[k];
                float xlwork2 = -2.85e6f * (psdep + pgdep + pidep + pigen)
                              - xl[k] * prevp
                              - xlf * (piacr + paacw + paacw + pgacr
                                       + psacr);
                t[k] -= xlwork2 / cpm[k] * dtcld;
            } else {
                WDM6_LIMIT(max(1.0e-15f, qc[k]),
                           praut + pracw + paacw + paacw,
                           praut *= factor; pracw *= factor;
                           paacw *= factor;);
                WDM6_LIMIT(max(1.0e-15f, qr[k]),
                           -paacw - praut + pseml + pgeml - pracw
                           - paacw - prevp,
                           praut *= factor; prevp *= factor;
                           pracw *= factor; paacw *= factor;
                           pseml *= factor; pgeml *= factor;);
                WDM6_LIMIT(max(1.0e-9f, qs[k]),
                           pgacs - pseml - psevp,
                           pgacs *= factor; psevp *= factor;
                           pseml *= factor;);
                WDM6_LIMIT(max(1.0e-9f, qg[k]),
                           -(pgacs + pgevp + pgeml),
                           pgacs *= factor; pgevp *= factor;
                           pgeml *= factor;);
                WDM6_LIMIT(max(1.0e1f, nc[k]),
                           nraut + nccol + nracw + naacw + naacw,
                           nraut *= factor; nccol *= factor;
                           nracw *= factor; naacw *= factor;);
                WDM6_LIMIT(max(1.0e-2f, nr[k]),
                           -nraut + nrcol - nseml - ngeml,
                           nraut *= factor; nrcol *= factor;
                           nseml *= factor; ngeml *= factor;);
                float work2k = -(prevp + psevp + pgevp);
                qv[k] += work2k * dtcld;
                qc[k] = max(qc[k] - (praut + pracw + paacw + paacw)
                              * dtcld, 0.0f);
                qr[k] = max(qr[k] + (praut + pracw + prevp + paacw
                                       + paacw - pseml - pgeml) * dtcld,
                              0.0f);
                qs[k] = max(qs[k] + (psevp - pgacs + pseml) * dtcld,
                              0.0f);
                qg[k] = max(qg[k] + (pgacs + pgevp + pgeml) * dtcld,
                              0.0f);
                nc[k] = max(nc[k] + (-nraut - nccol - nracw - naacw
                                       - naacw) * dtcld, 0.0f);
                nr[k] = max(nr[k] + (nraut - nrcol + nseml + ngeml)
                              * dtcld, 0.0f);
                float xlf = 2.85e6f - xl[k];
                float xlwork2 = -xl[k] * (prevp + psevp + pgevp)
                              - xlf * (pseml + pgeml);
                t[k] -= xlwork2 / cpm[k] * dtcld;
            }
#undef WDM6_LIMIT
        }

        // ---- post-update saturation refresh (:1886-1914): qs1/qs2 and
        // rh1 only; rh2 keeps its loop-start value exactly as in WRF. ------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            qsw[k] = wdm6_qsat(t[k], p[id], false);
            qsi[k] = wdm6_qsat(t[k], p[id], true);
            rhw[k] = max(qv[k] / qsw[k], 1.0e-15f);
        }

        // ---- small-drop collapse, CCN activation, cloud condensation,
        // and the padding clamps (:1916-2032; all pointwise) ---------------
        for (int k = 0; k < nz; ++k) {
            int id = col + k * stride;
            float denfac = sqrt(1.28f / den[k]);
            // Nrevp_s / Prevp_s (:1925-1948): all rain collapses to cloud
            // when the mean-volume raindrop shrinks below di82.
            WdmRainSlope rr = wdm6_rain_slope(qr[k], nr[k], den[k], denfac);
            float avedia2 = rr.r * 2.88449914f;
            if (avedia2 <= 82.0e-6f) {
                nc[k] += nr[k];
                nr[k] = 0.0f;
                qc[k] += qr[k];
                qr[k] = 0.0f;
            }
            // CCN activation (:1951-1969).
            if (rhw[k] > 1.0f) {
                float ncact = max(0.0f,
                    (nn[k] + nc[k])
                    * min(1.0f, pow(wdm6_div(rhw[k], 1.0048f), 0.6f))
                    - nc[k]) / dtcld;
                ncact = min(ncact, max(nn[k], 0.0f) / dtcld);
                float pcact = min(
                    4.0f * 3.14159265f * 1000.0f
                    * (1.5e-6f * 1.5e-6f * 1.5e-6f) * ncact
                    / (3.0f * den[k]),
                    max(qv[k], 0.0f) / dtcld);
                qv[k] = max(qv[k] - pcact * dtcld, 0.0f);
                qc[k] = max(qc[k] + pcact * dtcld, 0.0f);
                nn[k] = max(nn[k] - ncact * dtcld, 0.0f);
                nc[k] = max(nc[k] + ncact * dtcld, 0.0f);
                t[k] += pcact * xl[k] / cpm[k] * dtcld;
            }
            // pcond/ncevp (:1976-1998).
            float qs1 = wdm6_qsat(t[k], p[id], false);
            float work1c = (max(qv[k], 1.0e-15f) - qs1)
                / (1.0f + xl[k] * xl[k] / (461.6f * cpm[k]) * qs1
                   / (t[k] * t[k]));
            float pcond = min(max(work1c / dtcld, 0.0f),
                                max(qv[k], 0.0f) / dtcld);
            if (qc[k] > 0.0f && work1c < 0.0f)
                pcond = max(work1c, -qc[k]) / dtcld;
            if (pcond == -qc[k] / dtcld) {         // ncevp (:1990-1994)
                nn[k] += nc[k];
                nc[k] = 0.0f;
            }
            qv[k] -= pcond * dtcld;
            qc[k] = max(qc[k] + pcond * dtcld, 0.0f);
            t[k] += pcond * xl[k] / cpm[k] * dtcld;

            // padding for small values (:2005-2030), with WDM6's
            // number-adjusting rain/cloud slope clamps.
            if (qc[k] <= 1.0e-15f) qc[k] = 0.0f;
            if (qi[k] <= 1.0e-15f) qi[k] = 0.0f;
            if (qr[k] >= 1.0e-9f && nr[k] >= 1.0e-2f) {
                float lamdr = exp(log((1.25663706e4f * nr[k])
                                        / (den[k] * qr[k]))
                                   * 0.33333333f);
                if (lamdr <= 2.0e3f) {
                    lamdr = 2.0e3f;
                    nr[k] = wdm6_div(den[k] * qr[k] * lamdr * lamdr * lamdr,
                          1.25663706e4f);
                } else if (lamdr >= 5.0e4f) {
                    lamdr = 5.0e4f;
                    nr[k] = wdm6_div(den[k] * qr[k] * lamdr * lamdr * lamdr,
                          1.25663706e4f);
                }
            }
            if (qc[k] >= 1.0e-15f && nc[k] >= 1.0e1f) {
                float lamdc = exp(log((5.23598776e2f * nc[k])
                                        / (den[k] * qc[k]))
                                   * 0.33333333f);
                if (lamdc <= 2.0e4f) {
                    lamdc = 2.0e4f;
                    nc[k] = wdm6_div(den[k] * qc[k] * lamdc * lamdc * lamdc,
                          5.23598776e2f);
                } else if (lamdc >= 5.0e5f) {
                    lamdc = 5.0e5f;
                    nc[k] = wdm6_div(den[k] * qc[k] * lamdc * lamdc * lamdc,
                          5.23598776e2f);
                }
            }
        }
    }

    // ---- writeback + effectRad_wdm6 (:3135-3234) with the wdm6 driver's
    // clamp (:321-325), in gpuwm's radiation-facing micron convention -----
    for (int k = 0; k < nz; ++k) {
        int id = col + k * stride;
        theta[id] = t[k] / pii[id];
        qv_g[id] = qv[k]; qc_g[id] = qc[k]; qi_g[id] = qi[k];
        qr_g[id] = qr[k]; qs_g[id] = qs[k]; qg_g[id] = qg[k];
        nn_g[id] = nn[k]; nc_g[id] = nc[k]; nr_g[id] = nr[k];
        // cloud: gamma PSD with the prognostic nc (:3208-3214);
        // cdm2 = rgmma(5/3) = 0.902619934.
        float rqc = max(1.0e-12f, qc[k] * den[k]);
        float rnc = max(1.0e-6f, nc[k] * den[k]);
        if (rqc > 1.0e-12f && rnc > 1.0e-6f) {
            float lamc = 2.0f * 0.902619934f
                * pow(5.23598776e2f * nc[k] / rqc, 1.0f / 3.0f);
            effc[id] = max(2.51f, min(1.0e6f / lamc, 50.0f));
        } else {
            effc[id] = 2.49f;
        }
        // ice: WSM6's diagnostic number (:3196-3201, :3216-3222).
        float rqi = max(1.0e-12f, qi[k] * den[k]);
        float tempi = den[k] * max(qi[k], 1.0e-15f);
        tempi = sqrt(sqrt(tempi * tempi * tempi));
        float ni = min(max(5.38e7f * tempi, 1.0e3f), 1.0e6f);
        float rni = max(1.0e-6f, ni * den[k]);
        if (rqi > 1.0e-12f && rni > 1.0e-6f) {
            float diai = 11.9f * sqrt(rqi / ni);
            effi[id] = max(10.01f,
                             min(0.75f * 0.163f * diai * 1.0e6f,
                                   125.0f));
        } else {
            effi[id] = 4.99f;
        }
        // snow (:3224-3232).
        float rqs = max(1.0e-12f, qs[k] * den[k]);
        if (rqs > 1.0e-12f) {
            float nsfac = max(min(exp(0.12f * (273.15f - t[k])),
                                      5.0e4f), 1.0f);
            float lamdas = sqrt(sqrt(6.28318531e8f * nsfac / rqs));
            effs[id] = max(25.0f, min(0.5e6f / lamdas, 999.0f));
        } else {
            effs[id] = 9.99f;
        }
        // wdm6 driver clamp (:321-325) == the state background bounds.
        effc[id] = max(2.49f, min(effc[id], 50.0f));
        effi[id] = max(4.99f, min(effi[id], 125.0f));
        effs[id] = max(9.99f, min(effs[id], 999.0f));
    }
    return 0u;
}

kernel void wdm6_column(
    constant Wdm6ColumnResources &r [[buffer(0)]],
    constant Wdm6ColumnParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const int stride = p.ny * p.nx;
    if (col >= uint(stride) || p.nz > WDM6_KMAX) return;
    const unsigned int flags = wdm6_column_impl(
        r.theta, r.qv, r.qc, r.qi, r.qr, r.qs, r.qg, r.nn, r.nc, r.nr,
        r.den, r.p, r.pii, r.dz, r.xland, r.rainnc, r.rainncv, r.snownc,
        r.snowncv, r.graupelnc, r.graupelncv, r.sr, r.effc, r.effi, r.effs,
        p.dt, p.hail_opt, p.nz, stride, int(col));
    if (flags != 0u) {
        device atomic_uint *status = reinterpret_cast<device atomic_uint *>(r.count_status);
        atomic_fetch_or_explicit(status, flags, memory_order_relaxed);
    }
}
