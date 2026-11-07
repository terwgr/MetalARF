// woof/ocean/seaice/thermodynamics.cu, transcribed statement by statement.
// Binary64 runs on Soft8 with glibc exp; binary32 stores round once and
// keep gradual underflow, as the CUDA unit (no FTZ) does.
#include <metal_stdlib>
#include "seaice_initialize_bindings.metalh"
#include "seaice_precipitation_bindings.metalh"
#include "seaice_step_bindings.metalh"
#include "seaice_blend_ocean_bindings.metalh"
#include "seaice_ocean_freeze_bindings.metalh"
#include "seaice_mask_xice_bindings.metalh"
#include "seaice_ocean_owned_bindings.metalh"
#include "seaice_ocean_freeze_pwp_bindings.metalh"
#include "seaice_seed_pwp_profile_bindings.metalh"
#include "thompson_aerosol_soft8.metalh"
using namespace metal;

typedef Soft8 D;
static inline D K(ulong bits) { return D::fromBits(bits); }
static inline D F(float x) { return D(x); }
static inline float R(D x) { return float(x); }
#define RHOI K(0x408ca80000000000ul)
#define RHOS K(0x4074a00000000000ul)
#define LF K(0x41145e1000000000ul)
#define LV K(0x414312d000000000ul)
#define TF K(0x4070f5999999999aul)
#define TM K(0x4071126666666666ul)
#define SIGMA K(0x3e6e714da268c0ccul)
#define ONE K(0x3ff0000000000000ul)
#define ZERO D()

// glibc exp is exp_inline with no tail; NaN is quieted first.
static inline D si_exp(D x) {
    if ((x.bits & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL) return K(x.bits | 0x8000000000000UL);
    return K(sgp_exp_inline(x.bits, 0ul, 0u));
}
static inline D si_qsat(D t, D p, bool ice) {
    const D e = K(0x408319999999999aul) * si_exp((ice ? K(0x403675c28f5c28f6ul) : K(0x4031ab851eb851ecul)) * (t - K(0x4071126666666666ul)) / (t - (ice ? K(0x3fe199999999999aul) : K(0x403da66666666666ul))));
    return K(0x3fe3e76c8b439581ul) * e / (p - e);
}
static inline D si_net(D t, D ta, D qa, D p, D ch, D sw, D lw, D alb, D maxevap,
                       thread D &h, thread D &e) {
    const D rho = p / (K(0x4071f0cccccccccdul) * ta * (ONE + K(0x3fe3851eb851eb85ul) * qa));
    h = rho * K(0x408f640000000000ul) * ch * (t - ta);
    e = fmin(maxevap, rho * ch * (si_qsat(t, p, true) - qa));
    return (ONE - alb) * sw + K(0x3fef5c28f5c28f5cul) * (lw - SIGMA * t * t * t * t) - h - K(0x41459e9200000000ul) * e;
}
static inline float si_add(float acc, D x) { return R(F(acc) + x); }

kernel void seaice_initialize(constant SeaiceInitializeResources &r [[buffer(0)]],
                              constant SeaiceInitializeParams &p [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    r.mask[i] = (F(r.lake[i]) < F(.5f) && (F(r.land[i]) > F(1.5f) || F(r.a[i]) > ZERO)) ? 1.f : 0.f;
    const bool masked = r.mask[i] != 0.f;
    const D frac = masked ? fmin(ONE, fmax(ZERO, F(r.a[i]))) : ZERO;
    if (masked) r.sst[i] = R(fmax(F(r.sst[i]), TF));
    r.v[i] = R(frac * F(r.hi[i])); r.s[i] = R(frac * fmax(ZERO, F(r.snow[i])));
    r.hs[i] = frac > ZERO ? R(F(r.s[i]) / (frac * RHOS)) : 0.f;
    r.skin[i] = R(fmin(TM, F(r.tsk[i])));
    if (frac == ZERO) r.hi[i] = 0.f;
}

kernel void seaice_precipitation(constant SeaicePrecipitationResources &r [[buffer(0)]],
                                 constant SeaicePrecipitationParams &p [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    r.out[i] = aa_fmaxf(0.f, as_type<float>(soft32_add(as_type<uint>(r.rain[i]), as_type<uint>(r.pending[i]))));
    r.snow[i] = as_type<float>(soft32_mul(as_type<uint>(aa_fminf(1.f, aa_fmaxf(0.f, r.ratio[i]))),
                                          as_type<uint>(r.out[i])));
}

kernel void seaice_step(constant SeaiceStepResources &r [[buffer(0)]],
                        constant SeaiceStepParams &p [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n) || F(r.mask[i]) == ZERO) return;
    D a = F(r.ap[i]), v = F(r.vp[i]), s = F(r.sp[i]);
    const D dt = F(p.dt0), qo = F(p.qo0), hnew0 = F(p.hnew0);
    const D oldv = v, olds = s, olda = a, oldenergy = -LF * (RHOI * v + s);
    (void)olds;
    r.coupled[i] = 0.f;
    const D sst = F(r.sst[i]), ta = F(r.ta[i]), qa = F(r.qa[i]), pr = F(r.p[i]);
    const D chp = fmax(ZERO, F(r.ch[i])), sw = F(r.sw[i]), lw = F(r.lw[i]);
    const D water_t = fmax(TF, sst);
    const D rho = pr / (K(0x4071f0cccccccccdul) * ta * (ONE + K(0x3fe3851eb851eb85ul) * qa));
    const D hw = rho * K(0x408f640000000000ul) * chp * (water_t - ta);
    const D ew = rho * chp * (si_qsat(water_t, pr, false) - qa);
    const D qw = K(0x3fee147ae147ae14ul) * sw + K(0x3fef5c28f5c28f5cul) * (lw - SIGMA * water_t * water_t * water_t * water_t) - hw - LV * ew;
    r.ocean_heat[i] = R((ONE - a) * qw - a * qo);
    if (a <= ZERO && sst > TF) return;
    const D snowfall_mass = a * F(r.snowfall[i]); s += snowfall_mass;
    D h = ZERO, e = ZERO, ts = TM, q = ZERO, c = ZERO;
    const D alb = s > K(0x3e45798ee2308c3aul) ? K(0x3fe999999999999aul) : K(0x3fe4cccccccccccdul);
    D atm = ZERO, sublim = ZERO, surf = ZERO;
    if (a > ZERO && v > ZERO) {
        const D hi = v / a, hs = s / (a * RHOS);
        const D conductance = ONE / fmax(hi / K(0x40003d70a3d70a3dul) + hs / K(0x3fd3d70a3d70a3d7ul), K(0x3e45798ee2308c3aul));
        const D maxevap = (s + RHOI * v) / (a * dt);
        D lo = K(0x4062c00000000000ul), up = TM;
        for (int k = 0; k < 36; k++) {
            ts = K(0x3fe0000000000000ul) * (lo + up);
            q = si_net(ts, ta, qa, pr, chp, sw, lw, alb, maxevap, h, e);
            if (q + conductance * (TF - ts) > ZERO) lo = ts; else up = ts;
        }
        ts = K(0x3fe0000000000000ul) * (lo + up);
        q = si_net(ts, ta, qa, pr, chp, sw, lw, alb, maxevap, h, e);
        c = conductance * (TF - ts);
        D surface = fmax(ZERO, q + c) * a * dt;
        sublim = e * a * dt;
        if (sublim > ZERO) {
            sublim = fmin(sublim, s + RHOI * v);
            const D ds = fmin(s, sublim); s -= ds; v -= (sublim - ds) / RHOI;
        } else s -= sublim;
        const D ds = fmin(s, surface / LF); s -= ds; surface -= ds * LF;
        const D dv = fmin(v, surface / (RHOI * LF)); v -= dv; surf += dv;
        const D basal = a * dt * (c - qo) / (RHOI * LF);
        const D basal_excess = fmax(ZERO, -basal - v) * RHOI * LF;
        if (basal >= ZERO) { v += basal; r.growth[i] = si_add(r.growth[i], basal); }
        else { const D dm = fmin(v, -basal); v -= dm; r.melt[i] = si_add(r.melt[i], dm); }
        r.melt[i] = si_add(r.melt[i], surf);
        const D excess = fmax(ZERO, surface - dv * RHOI * LF) + basal_excess;
        r.ocean[i] = si_add(r.ocean[i], excess);
        r.ocean_heat[i] = si_add(r.ocean_heat[i], excess / dt);
        atm = a * dt * (q + LF * e + qo);
        if (excess > ZERO) atm -= excess;
        if (v <= ZERO && s > ZERO) {
            r.ocean[i] = si_add(r.ocean[i], -(s * LF));
            r.ocean_heat[i] = si_add(r.ocean_heat[i], -(s * LF / dt));
            atm += s * LF; s = ZERO;
        }
    }
    const D ow = ONE - olda;
    if (ow > ZERO && sst <= TF) {
        const D freeze = fmax(ZERO, -qw - qo) * ow * dt / (RHOI * LF);
        v += freeze; r.growth[i] = si_add(r.growth[i], freeze);
        r.ocean_heat[i] = si_add(r.ocean_heat[i], freeze * RHOI * LF / dt);
        a = fmin(ONE, a + freeze / hnew0);
        atm -= freeze * RHOI * LF;
    }
    if (olda == ZERO && v > ZERO) ts = TF;
    if (v < oldv && oldv > ZERO) a = olda * sqrt(fmax(ZERO, v / oldv));
    if (v <= ZERO) { v = ZERO; a = ZERO; }
    if (v > ZERO && a <= ZERO) a = fmin(ONE, v / hnew0);
    r.dhi[i] = si_add(r.dhi[i], (a > ZERO ? v / a : ZERO) - (olda > ZERO ? oldv / olda : ZERO));
    r.da[i] = si_add(r.da[i], a - olda);
    const D newenergy = -LF * (RHOI * F(R(v)) + F(R(s)));
    const D expected = -LF * snowfall_mass + atm;
    r.energy[i] = si_add(r.energy[i], expected);
    r.error[i] = si_add(r.error[i], (newenergy - oldenergy) - expected);
    r.ap[i] = R(a); r.vp[i] = R(v); r.sp[i] = R(s); r.tp[i] = R(ts);
    r.hiout[i] = a > ZERO ? R(v / a) : 0.f; r.hsout[i] = a > ZERO ? R(s / (RHOS * a)) : 0.f;
    if (olda > ZERO || a > ZERO) {
        r.coupled[i] = 1.f;
        r.hfx[i] = R(olda * h + (ONE - olda) * hw);
        r.qfx[i] = R(olda * e + (ONE - olda) * ew);
        r.lh[i] = R(LV * F(r.qfx[i]));
        r.qsfc[i] = R(a * si_qsat(ts, pr, true) + (ONE - a) * si_qsat(water_t, pr, false));
        r.tsk[i] = R(a * ts + (ONE - a) * water_t);
        r.albedo[i] = R(a * alb + (ONE - a) * K(0x3faeb851eb851eb8ul)); r.emiss[i] = R(K(0x3fef5c28f5c28f5cul));
        r.snow[i] = a > ZERO ? R(s / a) : 0.f; r.snowh[i] = a > ZERO ? R(s / (RHOS * a)) : 0.f;
        r.znt[i] = R(a * K(0x3f50624dd2f1a9fcul) + (ONE - a) * K(0x3f1a36e2eb1c432dul));
    }
}

kernel void seaice_blend_ocean(constant SeaiceBlendOceanResources &r [[buffer(0)]],
                               constant SeaiceBlendOceanParams &p [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
    if (i < uint(p.n) && F(r.mask[i]) > ZERO && F(r.a[i]) > ZERO) {
        const uint a = as_type<uint>(r.a[i]);
        const uint water = as_type<uint>(aa_fmaxf(271.35f, r.sst[i]));
        r.tsk[i] = as_type<float>(soft32_add(soft32_mul(a, as_type<uint>(r.skin[i])),
                                             soft32_mul(soft32_sub(0x3f800000u, a), water)));
    }
}

kernel void seaice_ocean_freeze(constant SeaiceOceanFreezeResources &r [[buffer(0)]],
                                constant SeaiceOceanFreezeParams &p [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n) || F(r.mask[i]) == ZERO || F(r.tml[i]) >= F(271.35f)) return;
    const D oldv = F(r.v[i]), olda = F(r.a[i]), oldhi = F(r.hi[i]);
    const D heat = (F(271.35f) - F(r.tml[i])) * F(p.water_capacity) * F(r.hml[i]);
    const D vol = heat / (RHOI * LF), nv = oldv + vol;
    const D na = fmin(ONE, olda + vol / F(p.hnew));
    const D s = F(r.s[i]);
    r.v[i] = R(nv); r.a[i] = R(na); r.hi[i] = R(nv / na); r.hs[i] = R(s / (RHOS * na));
    r.snow[i] = R(s / na); r.snowh[i] = r.hs[i];
    r.albedo[i] = R(na * (s > K(0x3e45798ee2308c3aul) ? K(0x3fe999999999999aul) : K(0x3fe4cccccccccccdul)) + (ONE - na) * K(0x3faeb851eb851eb8ul));
    r.znt[i] = R(na * K(0x3f50624dd2f1a9fcul) + (ONE - na) * K(0x3f1a36e2eb1c432dul));
    if (olda == ZERO) r.skin[i] = R(TF);
    r.growth[i] = si_add(r.growth[i], vol); r.energy[i] = si_add(r.energy[i], -heat);
    r.ocean[i] = si_add(r.ocean[i], heat);
    r.error[i] = si_add(r.error[i], -LF * RHOI * (F(r.v[i]) - oldv) + heat);
    r.dhi[i] = si_add(r.dhi[i], F(r.hi[i]) - oldhi); r.da[i] = si_add(r.da[i], F(r.a[i]) - olda);
    r.sst[i] = 271.35f; r.tml[i] = 271.35f;
}

// The Noah sea-ice call sees only the ice columns this model does not own:
// xice where the ocean mask is set reads as zero (cp.where in the driver).
kernel void seaice_mask_xice(constant SeaiceMaskXiceResources &r [[buffer(0)]],
                             constant SeaiceMaskXiceParams &p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    r.out[i] = F(r.mask[i]) > ZERO ? 0.f : r.xice[i];
}

// The ocean takes ice heat only where ice supplied the surface fluxes,
// or the atmosphere classified the masked water column as land.
kernel void seaice_ocean_owned(constant SeaiceOceanOwnedResources &r [[buffer(0)]],
                               constant SeaiceOceanOwnedParams &p [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    r.out[i] = (r.mask[i] > 0.f && (r.coupled[i] > 0.f || r.land[i] <= 1.5f)) ? 1.f : 0.f;
}

kernel void seaice_seed_pwp_profile(constant SeaiceSeedPwpProfileResources &r [[buffer(0)]],
                                    constant SeaiceSeedPwpProfileParams &p [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n) || F(r.mask[i]) == ZERO || F(r.land[i]) <= F(1.5f)) return;
    for (int k = 0; k < p.levels; ++k) {
        uint q = uint(k * p.n) + i;
        r.t[q] = aa_fmaxf(r.t[q], 271.35f);
        r.tini[q] = aa_fmaxf(r.tini[q], 271.35f);
    }
    r.sst[i] = r.t[i];
}

kernel void seaice_ocean_freeze_pwp(constant SeaiceOceanFreezePwpResources &r [[buffer(0)]],
                                    constant SeaiceOceanFreezePwpParams &p [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n) || F(r.mask[i]) == ZERO || F(r.land[i]) <= F(1.5f)) return;
    const D freezing = F(271.35f);
    int m = 0; D depth = ZERO;
    for (int k = 0; k < p.levels; ++k) {
        uint q = uint(k * p.n) + i;
        float top = k ? r.z[q - uint(p.n)] : 0.f;
        if (k && F(top) >= F(r.ml[i])) break;
        depth += F(r.z[q]) - F(top); m = k + 1;
    }
    D h = (F(r.coupled[i]) > ZERO ? F(r.qice[i]) * F(p.dt) : ZERO) + F(r.owed[i]);
    D change = h / (F(p.water_capacity) * depth), heat = ZERO, applied = ZERO;
    for (int k = 0; k < m; ++k) {
        uint q = uint(k * p.n) + i;
        float top = k ? r.z[q - uint(p.n)] : 0.f;
        D dz = F(r.z[q]) - F(top), old = F(r.t[q]), want = old + change;
        if (want < freezing) { heat += (freezing - want) * F(p.water_capacity) * dz; want = freezing; }
        float now = R(want);
        applied += (F(now) - old) * F(p.water_capacity) * dz;
        r.t[q] = now;
    }
    r.owed[i] = R(h + heat - applied);
    if (heat > ZERO) {
        D oldv = F(r.v[i]), olda = F(r.a[i]), oldhi = F(r.hi[i]);
        D vol = heat / (RHOI * LF), nv = oldv + vol;
        D na = fmin(ONE, olda + vol / F(p.hnew)), s = F(r.s[i]);
        r.v[i] = R(nv); r.a[i] = R(na); r.hi[i] = R(nv / na); r.hs[i] = R(s / (RHOS * na));
        r.snow[i] = R(s / na); r.snowh[i] = r.hs[i];
        r.albedo[i] = R(na * (s > K(0x3e45798ee2308c3aul) ? K(0x3fe999999999999aul) : K(0x3fe4cccccccccccdul)) + (ONE - na) * K(0x3faeb851eb851eb8ul));
        r.znt[i] = R(na * K(0x3f50624dd2f1a9fcul) + (ONE - na) * K(0x3f1a36e2eb1c432dul));
        if (olda == ZERO) r.skin[i] = R(TF);
        r.growth[i] = si_add(r.growth[i], vol); r.energy[i] = si_add(r.energy[i], -heat);
        r.ocean[i] = si_add(r.ocean[i], heat);
        r.error[i] = si_add(r.error[i], -LF * RHOI * (F(r.v[i]) - oldv) + heat);
        r.dhi[i] = si_add(r.dhi[i], F(r.hi[i]) - oldhi); r.da[i] = si_add(r.da[i], F(r.a[i]) - olda);
    }
    r.sst[i] = r.t[i];
}
