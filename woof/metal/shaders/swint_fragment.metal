// WRF swint_opt=1: current-time solar cosine, column power-law fit, and
// between-call shortwave evaluation. Every grid-rounded kernel is guarded.
#include <metal_stdlib>
#include "swint_fragment_bindings.metalh"
using namespace metal;

static inline float swint_exponent(float flux, float flux0,
                                    float cosine, float cosine0) {
    const float ratio = cosine / cosine0;
    float value;
    if (ratio < 1.0f) {
        value = precise::log(max(1.0f, flux) / max(1.0f, flux0)) /
                precise::log(min(1.0f - 1.0e-4f, ratio));
    } else if (ratio > 1.0f) {
        value = precise::log(max(1.0f, flux) / max(1.0f, flux0)) /
                precise::log(max(1.0f + 1.0e-4f, ratio));
    } else {
        value = 0.0f;
    }
    return clamp(value, -0.5f, 2.5f);
}

kernel void swint_coszen_loc_fragment(
    constant SwintCoszenLocFragmentResources &r [[buffer(0)]],
    constant SwintCoszenLocFragmentParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    const float obecl = 23.5f * p.degrad;
    const float sinob = precise::sin(obecl);
    const float sxlong = (p.julian >= 80.0f
        ? p.dpd * (p.julian - 80.0f)
        : p.dpd * (p.julian + 285.0f)) * p.degrad;
    const float declin = precise::asin(sinob * precise::sin(sxlong));
    const float da = (6.2831853071795862f * (p.julian - 1.0f)) / 365.0f;
    const float two_da = 2.0f * da;
    float eot = 0.000075f + 0.001868f * precise::cos(da);
    eot = eot - 0.032077f * precise::sin(da);
    eot = eot - 0.014615f * precise::cos(two_da);
    eot = eot - 0.04089f * precise::sin(two_da);
    eot = eot * 229.18f;
    const float xt24 = fmod(p.xtime, 1440.0f) + eot;
    const float local_time = (p.gmt + xt24 / 60.0f) + r.longitude[gid] / 15.0f;
    const float hour_angle = (15.0f * (local_time - 12.0f)) * p.degrad;
    const float latitude = r.latitude[gid] * p.degrad;
    r.coszen_loc[gid] = precise::sin(latitude) * precise::sin(declin) +
        (precise::cos(latitude) * precise::cos(declin)) *
        precise::cos(hour_angle);
}

kernel void swint_update_fragment(
    constant SwintUpdateFragmentResources &r [[buffer(0)]],
    constant SwintUpdateFragmentParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    const float cz = r.coszen[gid];
    const float czl = r.coszen_loc[gid];
    if (cz > 1.0e-4f && czl > 1.0e-4f) {
        float dir0, cz0;
        if (r.bx[gid] <= 0.0f) {
            dir0 = (czl / cz) * r.swddir[gid];
            cz0 = czl;
        } else {
            dir0 = r.swddir_ref[gid];
            cz0 = r.coszen_ref[gid];
        }
        const float b = swint_exponent(r.swddir[gid], dir0, cz, cz0);
        r.bb[gid] = b;
        r.bx[gid] = r.swddir[gid] / precise::pow(cz, b);

        float down0;
        if (r.gx[gid] <= 0.0f) {
            down0 = (czl / cz) * r.swdown[gid];
            cz0 = czl;
        } else {
            down0 = r.swdown_ref[gid];
            cz0 = r.coszen_ref[gid];
        }
        const float g = swint_exponent(r.swdown[gid], down0, cz, cz0);
        r.gg[gid] = g;
        r.gx[gid] = r.swdown[gid] / precise::pow(cz, g);
    } else {
        r.bx[gid] = 0.0f;
        r.bb[gid] = 0.0f;
        r.gx[gid] = 0.0f;
        r.gg[gid] = 0.0f;
    }
    r.coszen_ref[gid] = cz;
    r.swdown_ref[gid] = r.swdown[gid];
    r.swddir_ref[gid] = r.swddir[gid];
}

kernel void swint_interp_fragment(
    constant SwintInterpFragmentResources &r [[buffer(0)]],
    constant SwintInterpFragmentParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.count)) return;
    const float czr = r.coszen_ref[gid];
    const float czl = r.coszen_loc[gid];
    if (czr > 1.0e-4f && czl > 1.0e-4f) {
        const float dir = (r.bb[gid] == -0.5f || r.bb[gid] == 2.5f)
            ? (czl / czr) * r.swddir_ref[gid]
            : r.bx[gid] * precise::pow(czl, r.bb[gid]);
        const float down = (r.gg[gid] == -0.5f || r.gg[gid] == 2.5f)
            ? (czl / czr) * r.swdown_ref[gid]
            : r.gx[gid] * precise::pow(czl, r.gg[gid]);
        r.swddir[gid] = dir;
        r.swdown[gid] = down;
        r.swddif[gid] = down - dir;
        r.swddni[gid] = dir / czl;
        r.gsw[gid] = down * (1.0f - r.albedo[gid]);
    } else {
        r.swddir[gid] = 0.0f;
        r.swdown[gid] = 0.0f;
        r.swddif[gid] = 0.0f;
        r.swddni[gid] = 0.0f;
        r.gsw[gid] = 0.0f;
    }
}
