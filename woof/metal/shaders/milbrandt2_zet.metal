#include <metal_stdlib>
#include "milbrandt2_zet_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void milbrandt2_zet(
    constant Milbrandt2ZetResources& r [[buffer(0)]],
    constant Milbrandt2ZetParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const float t = r.temperature[idx];
    const float pressure = r.pressure[idx];
    const float de = pressure / (p.rgas_d * t);
    const float tmp9 = de * de;
    float nr = r.nr[idx], ni = r.ni[idx], ns = r.ns[idx];
    float ng = r.ng[idx], nh = r.nh[idx];
    if (!p.concentration_space) {
        nr *= de;
        ni *= de;
        ns *= de;
        ng *= de;
        nh *= de;
    }
    const float qr = r.qr[idx], qi = r.qi[idx], qs = r.qs[idx];
    const float qg = r.qg[idx], qh = r.qh[idx];
    float rain = 0.0f, ice = 0.0f, snow = 0.0f, graupel = 0.0f, hail = 0.0f;
    if (qr > p.eps_q && nr > p.eps_n)
        rain = r.ck[147] * r.ck[149] * tmp9 * qr * qr / nr;
    if (qi > p.eps_q && ni > p.eps_n)
        ice = r.ck[148] * r.ck[150] * tmp9 * qi * qi / ni;
    if (qs > p.eps_q && ns > p.eps_n)
        snow = r.ck[148] * r.ck[151] * tmp9 * qs * qs / ns;
    if (qg > p.eps_q && ng > p.eps_n)
        graupel = r.ck[148] * r.ck[152] * tmp9 * qg * qg / ng;
    if (qh > p.eps_q && nh > p.eps_n)
        hail = r.ck[148] * r.ck[153] * tmp9 * qh * qh / nh;
    if (t > p.trpl) {
        ice *= p.dielectric_factor;
        snow *= p.dielectric_factor;
        graupel *= p.dielectric_factor;
        hail *= p.dielectric_factor;
    }
    const float ze = rain + ice + snow + graupel + hail;
    const float zet = ze > 0.0f
        ? p.dbz_scale * log10(ze * p.z_factor) : p.minimum_zet;
    r.refl[idx] = max(zet, p.minimum_zet);
}
