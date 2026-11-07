// WRF v4.6.1 phys/module_sf_oml.F (OML1D, omlinit), transcribed statement by
// statement from woof/core/kernels/oml.cu. The CUDA unit keeps subnormals,
// so every binary32 operation and comparison runs on soft32 bit ops.
#include <metal_stdlib>
#include "ocean_oml_init_bindings.metalh"
#include "ocean_oml_step_bindings.metalh"
#include "ocean_sst_diagnostics_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

static inline uint om_bits(float x) { return as_type<uint>(x); }
static inline uint om_neg(uint a) { return a ^ 0x80000000u; }
// fmaxf: a NaN operand yields the other; +0 wins a signed-zero tie.
static inline uint om_fmax(uint a, uint b) {
    if (soft32_is_nan(a)) return b;
    if (soft32_is_nan(b)) return a;
    if (soft32_equal(a, b)) return a & b;
    return soft32_greater(a, b) ? a : b;
}

kernel void ocean_oml_init(constant OceanOmlInitResources &r [[buffer(0)]],
                           constant OceanOmlInitParams &p [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n) || !p.start) return;
    const uint tsk = om_bits(r.tsk[i]), hml0 = om_bits(p.hml0), zero = 0u;
    r.tml[i] = r.tsk[i]; r.t0ml[i] = r.tsk[i];
    const float cold = as_type<float>(soft32_sub(tsk, 0x40a00000u));
    if (soft32_greater(hml0, zero)) {
        r.hml[i] = p.hml0; r.h0ml[i] = r.hml[i]; r.huml[i] = 0.f; r.hvml[i] = 0.f;
        r.tmoml[i] = cold;
    } else if (soft32_equal(hml0, zero)) {
        r.hml[i] = r.h0ml[i]; r.huml[i] = 0.f; r.hvml[i] = 0.f; r.tmoml[i] = cold;
    } else {
        r.hml[i] = r.h0ml[i];
        const uint tmoml = om_bits(r.tmoml[i]);
        if (soft32_greater(tmoml, 0x43480000u) && soft32_less_equal(tmoml, 0x43490000u))
            r.tmoml[i] = r.tsk[i];
    }
}

kernel void ocean_oml_step(constant OceanOmlStepResources &r [[buffer(0)]],
                           constant OceanOmlStepParams &p [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const bool iced = p.coupled_ice && soft32_greater(om_bits(r.ice_mask[i]), 0u);
    if (!(soft32_greater(om_bits(r.xland[i]), 0x3fc00000u) || iced)) return;
    const uint hlf = 0x3f000000u, one = 0x3f800000u, two = 0x40000000u;
    const uint dt = om_bits(p.dt), gam = om_bits(p.gam), relaxation = om_bits(p.relaxation);
    const uint rhowater = om_bits(p.rhowater), cwater = om_bits(p.cwater);
    const uint stbolt = om_bits(p.stbolt), gravity = om_bits(p.gravity);
    uint tml = om_bits(r.tml[i]), h = om_bits(r.hml[i]);
    const uint t0ml = om_bits(r.t0ml[i]), h0 = om_bits(r.h0ml[i]);
    const uint hu1 = om_bits(r.huml[i]), hv1 = om_bits(r.hvml[i]), tmoml = om_bits(r.tmoml[i]);
    const uint alp = om_fmax(soft32_mul(soft32_sub(tml, 0x43889333u), 0x3727c5acu), 0x358637bdu);
    const uint bv2 = soft32_mul(soft32_mul(alp, gravity), gam);
    const uint thp = soft32_sub(t0ml, soft32_mul(gam, soft32_sub(h, h0)));
    const uint a1 = soft32_sub(soft32_mul(soft32_sub(tml, thp), h),
                               soft32_mul(soft32_mul(soft32_mul(hlf, gam), h), h));
    const uint emiss = om_bits(r.emiss[i]);
    const uint gain = soft32_add(soft32_add(soft32_sub(om_neg(om_bits(r.hfx[i])), om_bits(r.lh[i])),
                                            om_bits(r.gsw[i])),
                                 soft32_mul(om_bits(r.glw[i]), emiss));
    const uint emit = soft32_mul(soft32_mul(soft32_mul(soft32_mul(soft32_mul(stbolt, emiss), tml), tml), tml), tml);
    const uint capacity = soft32_mul(rhowater, cwater);
    uint q = soft32_div(soft32_sub(gain, emit), capacity);
    if (iced) q = soft32_div(om_bits(r.ice_heat[i]), capacity);
    const uint uair = om_bits(r.uair[i]), vair = om_bits(r.vair[i]);
    uint wspd = soft32_sqrt(soft32_add(soft32_mul(uair, uair), soft32_mul(vair, vair)));
    if (soft32_less(wspd, 0x2edbe6ffu)) wspd = 0x2edbe6ffu;
    const uint ust2 = soft32_mul(hlf, om_bits(r.ust[i]));
    const uint inv_rho = soft32_div(one, rhowater);
    const uint taux = soft32_mul(inv_rho, soft32_div(soft32_mul(soft32_mul(ust2, ust2), uair), wspd));
    const uint tauy = soft32_mul(inv_rho, soft32_div(soft32_mul(soft32_mul(ust2, ust2), vair), wspd));
    const uint f = om_bits(r.f[i]);
    const uint hu2 = soft32_add(hu1, soft32_mul(dt, soft32_add(soft32_mul(f, hv1), taux)));
    const uint hv2 = soft32_add(hv1, soft32_mul(dt, soft32_add(soft32_mul(om_neg(f), hu2), tauy)));
    const uint a2 = soft32_add(a1, soft32_mul(q, dt));
    uint huml = hu2, hvml = hv2;
    const uint hold = h;
    const uint b2 = soft32_add(soft32_mul(hu2, hu2), soft32_mul(hv2, hv2));
    const uint hsqrd = soft32_add(
        soft32_div(om_neg(a2), gam),
        soft32_sqrt(soft32_add(soft32_div(soft32_mul(a2, a2), soft32_mul(gam, gam)),
                               soft32_div(soft32_mul(two, b2), bv2))));
    h = soft32_sqrt(om_fmax(hsqrd, 0u));
    if (soft32_less(h, hold)) h = hold;
    if (soft32_greater_equal(tml, tmoml) && !soft32_equal(h, 0u)) {
        tml = om_fmax(soft32_add(soft32_add(soft32_sub(t0ml, soft32_mul(gam, soft32_sub(h, h0))),
                                            soft32_mul(soft32_mul(hlf, gam), h)),
                                 soft32_div(a2, h)),
                      tmoml);
    } else {
        tml = t0ml;
    }
    if (iced && soft32_greater(h, 0u)) {
        const uint deep = om_fmax(0x4387accdu, soft32_sub(t0ml, soft32_mul(gam, soft32_sub(h, h0))));
        tml = soft32_div(soft32_add(soft32_add(soft32_mul(om_bits(r.tml[i]), hold),
                                               soft32_mul(deep, soft32_sub(h, hold))),
                                    soft32_mul(q, dt)),
                         h);
    }
    if (soft32_greater(relaxation, 0u)) {
        tml = soft32_sub(tml, soft32_div(soft32_mul(soft32_sub(tml, t0ml), dt), relaxation));
        h = soft32_sub(h, soft32_div(soft32_mul(soft32_sub(h, h0), dt), relaxation));
        huml = soft32_sub(huml, soft32_div(soft32_mul(huml, dt), relaxation));
        hvml = soft32_sub(hvml, soft32_div(soft32_mul(hvml, dt), relaxation));
    }
    r.tml[i] = as_type<float>(tml); r.hml[i] = as_type<float>(h);
    r.huml[i] = as_type<float>(huml); r.hvml[i] = as_type<float>(hvml);
    r.tsk[i] = as_type<float>(tml);
}

// update_ocean_diagnostics: live SST and its change on open water.
kernel void ocean_sst_diagnostics(constant OceanSstDiagnosticsResources &r [[buffer(0)]],
                                  constant OceanSstDiagnosticsParams &p [[buffer(1)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const bool wet = soft32_greater(om_bits(r.land[i]), 0x3fc00000u)
                  && soft32_less(om_bits(r.lake[i]), 0x3f000000u);
    r.sst[i] = wet ? r.t[i] : 0.f;
    r.change[i] = wet ? as_type<float>(soft32_sub(om_bits(r.t[i]), om_bits(r.t0[i]))) : 0.f;
}
