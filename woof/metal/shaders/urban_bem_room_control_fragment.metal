#include <metal_stdlib>
#include "urban_bem_room_control_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmc_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmc_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmc_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmc_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

// WRF module_sf_bem.F bem_fluxvent/regtemp/reghum/air_cond (1982-2305).
// Source-only stage. Dispatch once per occupied building floor after the BEM
// room heat/moisture balances; the full bem_bem caller must supply qb/hsroo,
// volroo/hlroo and preserve these returned room fluxes in its floor records.
kernel void urban_bem_room_control_fragment(
    constant UrbanBemRoomControlFragmentResources &r [[buffer(0)]],
    constant UrbanBemRoomControlFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    device const float *cpint = r.cpint;
    device const float *rhoint = r.rhoint;
    device const float *vollev = r.vollev;
    device const float *tlev = r.tlev;
    device const float *tout = r.tout;
    device const float *latent = r.latent;
    device const float *humout = r.humout;
    device const float *rhoout = r.rhoout;
    device const float *humlev = r.humlev;
    device const float *beta = r.beta;
    device const float *qb = r.qb;
    device const float *hsroo = r.hsroo;
    device const float *volroo = r.volroo;
    device const float *hlroo = r.hlroo;
    device const float *shumroo = r.shumroo;
    device const float *cop = r.cop;
    device const float *timeon = r.timeon;
    device const float *timeoff = r.timeoff;
    device const float *targtemp = r.targtemp;
    device const float *gaptemp = r.gaptemp;
    device const float *targhum = r.targhum;
    device const float *gaphum = r.gaphum;
    device const int *swcond = r.swcond;
    device float *hsvent = r.hsvent;
    device float *hlvent = r.hlvent;
    device float *hsneed_out = r.hsneed_out;
    device float *hlneed_out = r.hlneed_out;
    device float *hsout = r.hsout;
    device float *hlout = r.hlout;
    device float *consump = r.consump;
    const int nitem = p.nitem;
    const float nhourday = p.nhourday;
    const float dt = p.dt;
    if (item >= uint(nitem)) return;
    hsvent[item] = ubmc_mul(ubmc_mul(ubmc_mul(ubmc_mul(ubmc_sub(1.0f, beta[item]), cpint[item]),
                                     rhoint[item]), ubmc_div(vollev[item], 3600.0f)),
                            ubmc_sub(tout[item], tlev[item]));
    hlvent[item] = ubmc_mul(ubmc_mul(ubmc_mul(ubmc_mul(ubmc_sub(1.0f, beta[item]), latent[item]),
                                     rhoint[item]), ubmc_div(vollev[item], 3600.0f)),
                            ubmc_sub(humout[item], humlev[item]));
    (void)rhoout;

    float hsneed = 0.0f, hlneed = 0.0f;
    if (swcond[item] != 0 && nhourday >= timeon[item] && nhourday <= timeoff[item]) {
        const float templev = ubmc_add(tlev[item], ubmc_mul(ubmc_div(dt, qb[item]), hsroo[item]));
        if (fabs(ubmc_sub(templev, targtemp[item])) > gaptemp[item]) {
            if (templev > ubmc_add(targtemp[item], gaptemp[item])) {
                hsneed = ubmc_sub(hsroo[item], ubmc_mul(ubmc_div(qb[item], dt),
                    ubmc_sub(ubmc_add(targtemp[item], gaptemp[item]), tlev[item])));
                const float alpha = ubmc_div(fabs(ubmc_sub(hsneed, hsroo[item])), qb[item]);
                if (alpha > 0.001f) hsneed = ubmc_add(hsroo[item], ubmc_mul(0.001f, qb[item]));
            } else {
                hsneed = ubmc_sub(hsroo[item], ubmc_mul(ubmc_div(qb[item], dt),
                    ubmc_sub(ubmc_sub(targtemp[item], gaptemp[item]), tlev[item])));
                const float alpha = ubmc_div(fabs(ubmc_sub(hsneed, hsroo[item])), qb[item]);
                if (alpha > 0.001f) hsneed = ubmc_sub(hsroo[item], ubmc_mul(0.001f, qb[item]));
            }
        }

        const float hum = ubmc_add(shumroo[item], ubmc_mul(
            ubmc_div(dt, ubmc_mul(ubmc_mul(latent[item], rhoint[item]), volroo[item])), hlroo[item]));
        if (fabs(ubmc_sub(hum, targhum[item])) > gaphum[item]) {
            const float cap = ubmc_mul(latent[item], ubmc_mul(rhoint[item], volroo[item]));
            if (hum > ubmc_add(targhum[item], gaphum[item])) {
                hlneed = ubmc_sub(hlroo[item], ubmc_mul(ubmc_div(cap, dt),
                    ubmc_sub(ubmc_add(targhum[item], gaphum[item]), shumroo[item])));
                const float beta_h = ubmc_div(fabs(ubmc_sub(hlneed, hlroo[item])), cap);
                if (beta_h > 1.0e-6f) hlneed = ubmc_add(hlroo[item], ubmc_mul(1.0e-6f, cap));
            } else {
                hlneed = ubmc_sub(hlroo[item], ubmc_mul(ubmc_div(cap, dt),
                    ubmc_sub(ubmc_sub(targhum[item], gaphum[item]), shumroo[item])));
                const float beta_h = ubmc_div(fabs(ubmc_sub(hlneed, hlroo[item])), cap);
                if (beta_h > 1.0e-6f) hlneed = ubmc_sub(hlroo[item], ubmc_mul(1.0e-6f, cap));
            }
        }
    }
    hsneed_out[item] = hsneed;
    hlneed_out[item] = hlneed;
    if (hsneed > 0.0f) {
        const float cost = ubmc_mul(ubmc_div(1.0f, cop[item]), ubmc_add(fabs(hsneed), fabs(hlneed)));
        hsout[item] = ubmc_add(cost, hsneed);
        hlout[item] = hlneed;
        consump[item] = cost;
    } else if (hsneed == 0.0f) {
        hsout[item] = 0.0f; hlout[item] = 0.0f; consump[item] = 0.0f;
        hlneed_out[item] = 0.0f;
    } else {
        hsout[item] = 0.0f; hlout[item] = 0.0f;
        consump[item] = ubmc_mul(ubmc_div(1.0f, cop[item]),
                                  ubmc_add(fabs(hsneed), fabs(hlneed)));
        hlneed_out[item] = 0.0f;
    }
}
