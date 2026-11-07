// Eclipse obscuration, transcribed statement by statement from the CUDA
// kernel in woof/eclipse_windows.py (WRF v4.6.1 module_ra_eclipse.F).
// Binary64 runs on Soft8; sin, cos, tan and atan are fdlibm's.
#include <metal_stdlib>
#include "eclipse_obscuration_bindings.metalh"
#include "eclipse_scale_bindings.metalh"
#include "eclipse_heating_bindings.metalh"
#include "eclipse_select_bindings.metalh"
#include "soft_fdlibm64_trig.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline Soft8 ec_sqrt(Soft8 x) { return SD8(soft64_sqrt(x.bits)); }

kernel void eclipse_obscuration(constant EclipseObscurationResources &r [[buffer(0)]],
                                constant EclipseObscurationParams &p [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    device const ulong *b = r.besselian;
    const Soft8 one = SD8(0x3FF0000000000000ul), zero = Soft8();
    const Soft8 rad = SD8(0x3F91DF46A2529D39ul);
    const Soft8 eps = SD8(0x3FB4F21A642F5481ul), E = ec_sqrt(one - eps * eps);
    const Soft8 s3600 = SD8(0x40AC200000000000ul);
    const Soft8 t = SD8(b[32]) + SD8(b[31]) / s3600 - SD8(b[2]);
    Soft8 v[6] = {zero, zero, zero, zero, zero, zero};
    for (int k = 0; k < 4; ++k) {
        const Soft8 tk = k == 0 ? one : (k == 1 ? t : (k == 2 ? t * t : t * t * t));
        for (int j = 0; j < 6; ++j) v[j] = v[j] + SD8(b[5 + 4 * j + k]) * tk;
    }
    const Soft8 rd = v[2] * rad, rd1 = fd64_atan(fd64_tan(rd) / E), rd2 = fd64_atan(E * fd64_tan(rd));
    const Soft8 rho1 = ec_sqrt(E * E * fd64_cos(rd) * fd64_cos(rd) + fd64_sin(rd) * fd64_sin(rd));
    const Soft8 rlat = Soft8(r.lat[i]) * rad;
    const Soft8 longitude = -Soft8(r.lon[i])
        - SD8(0x3FF00B3700474D9Cul) * SD8(0x402E000000000000ul) * SD8(b[31]) / s3600;
    const Soft8 H = v[5] * rad - longitude * rad;
    const Soft8 A = ec_sqrt(one - eps * eps * fd64_sin(rlat) * fd64_sin(rlat));
    const Soft8 clat = fd64_cos(rlat) / A, slat = fd64_sin(rlat) * E / A;
    const Soft8 xi = clat * fd64_sin(H);
    const Soft8 eta = slat * fd64_cos(rd1) * E - clat * fd64_sin(rd1) * fd64_cos(H);
    const Soft8 zeta = slat * fd64_sin(rd2) * E + clat * fd64_cos(rd2) * fd64_cos(H);
    const Soft8 dx = xi - v[0], dy = eta - v[1] / rho1;
    const Soft8 delta = ec_sqrt(dx * dx + dy * dy);
    const Soft8 l1 = v[3] - zeta * SD8(b[29]), l2 = v[4] - zeta * SD8(b[30]);
    Soft8 obsc = zero;
    int kind = 0;
    if (zeta > zero && delta <= l1) {
        const Soft8 al2 = SD8(l2.bits & 0x7ffffffffffffffful);
        if (delta <= al2 && l2 < zero) { obsc = one; kind = 2; }
        else { obsc = (l1 - delta) / (l1 + l2); kind = delta <= al2 ? 3 : 1; }
    }
    // fmin(1.0, fmax(0.0, obsc)): a NaN obscuration takes the zero operand.
    const Soft8 lower = obsc > zero ? obsc : zero;
    const Soft8 clipped = lower < one ? lower : one;
    r.obscur[i] = float(clipped);
    r.mask[i] = kind;
}

// EclipseRadiation.apply: transmission = 1 - obscur, products and the
// shortwave change in binary32 with gradual underflow, as CuPy forms them.
static inline uint ec_transmission(device const float *obscur, uint col) {
    return soft32_sub(0x3f800000u, as_type<uint>(obscur[col]));
}

kernel void eclipse_scale(constant EclipseScaleResources &r [[buffer(0)]],
                          constant EclipseScaleParams &p [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    r.dst[i] = as_type<float>(soft32_mul(as_type<uint>(r.src[i]),
                                         ec_transmission(r.obscur, i % uint(p.plane))));
}

kernel void eclipse_heating(constant EclipseHeatingResources &r [[buffer(0)]],
                            constant EclipseHeatingParams &p [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const uint previous = as_type<uint>(r.heating[i]);
    const uint dimmed = soft32_mul(as_type<uint>(r.base[i]),
                                   ec_transmission(r.obscur, i % uint(p.plane)));
    r.heating[i] = as_type<float>(dimmed);
    const uint change = soft32_sub(dimmed, previous);
    const bool changed = !soft32_equal(change, 0u);
    if (changed) r.total[i] = as_type<float>(soft32_add(as_type<uint>(r.total[i]), change));
    r.changed[i] = changed ? 1 : 0;
}

kernel void eclipse_select(constant EclipseSelectResources &r [[buffer(0)]],
                           constant EclipseSelectParams &p [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    if (r.changed[i] != 0) r.dst[i] = r.src[i];
}
