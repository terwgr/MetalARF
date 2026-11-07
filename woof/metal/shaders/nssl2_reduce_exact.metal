#include <metal_stdlib>
#include "nssl2_reduce_exact_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_word32.metalh"
using namespace metal;

// Strict WRF reducer consumes xfall, not the ordinary integrated exports.
kernel void nssl2_driver_reduce_precipitation_exact(
    constant Nssl2DriverReducePrecipitationExactResources &r [[buffer(0)]],
    constant Nssl2DriverReducePrecipitationExactParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= uint(p.ncol)) return;
    const uint n = uint(p.ncol), zero = 0u, thousand = as_type<uint>(1000.0f);
    const uint xr = as_type<uint>(r.category_export[column]);
    const uint xs = as_type<uint>(r.category_export[2*n+column]);
    const uint xh = as_type<uint>(r.category_export[3*n+column]);
    const uint xhl = as_type<uint>(r.category_export[4*n+column]);
    const uint dtrho = soft32_mul(as_type<uint>(p.dt), as_type<uint>(r.air_density[column]));
    const uint sum = soft32_add(soft32_add(soft32_add(xr,
        soft32_div(soft32_mul(xs, thousand), thousand)),
        soft32_div(soft32_mul(xh, thousand), thousand)),
        soft32_div(soft32_mul(xhl, thousand), thousand));
    const uint rain = soft32_add(zero, soft32_mul(dtrho, sum));
    const uint snow = soft32_add(zero, soft32_div(soft32_mul(soft32_mul(dtrho, xs), thousand), thousand));
    const uint graupel = soft32_add(zero, soft32_div(soft32_mul(soft32_mul(dtrho, xh), thousand), thousand));
    const uint hail = soft32_div(soft32_mul(soft32_mul(dtrho, xhl), thousand), thousand);
    r.rainncv[column] = as_type<float>(rain);
    r.snowncv[column] = as_type<float>(snow);
    r.graupelncv[column] = as_type<float>(graupel);
    r.hailncv[column] = as_type<float>(hail);
    r.rainnc[column] = as_type<float>(soft32_add(as_type<uint>(r.rainnc[column]), rain));
    r.snownc[column] = as_type<float>(soft32_add(as_type<uint>(r.snownc[column]), snow));
    r.hailnc[column] = as_type<float>(soft32_add(as_type<uint>(r.hailnc[column]), hail));
    r.graupelnc[column] = as_type<float>(soft32_add(as_type<uint>(r.graupelnc[column]), graupel));
    r.sr[column] = as_type<float>(soft32_div(soft32_add(soft32_add(snow, hail), graupel),
                                              soft32_add(rain, as_type<uint>(1.e-12f))));
}

// Strict scatter preserves the skipped CCN store and gradual underflow.
kernel void nssl2_driver_scatter_exact(
    constant Nssl2DriverScatterExactResources &r [[buffer(0)]],
    constant Nssl2DriverScatterExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint n = uint(p.n), density = as_type<uint>(r.air_density[idx]);
    r.qv[idx] = r.state[idx];
    r.qc[idx] = r.state[n+idx];
    r.qr[idx] = r.state[2*n+idx];
    r.qi[idx] = r.state[3*n+idx];
    r.qs[idx] = r.state[4*n+idx];
    r.qg[idx] = r.state[5*n+idx];
    r.qh[idx] = r.state[6*n+idx];
    r.qndrop[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[7*n+idx]), density));
    r.qnr[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[8*n+idx]), density));
    r.qni[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[9*n+idx]), density));
    r.qns[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[10*n+idx]), density));
    r.qng[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[11*n+idx]), density));
    r.qnh[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[12*n+idx]), density));
    if (p.predicted_ccn)
        r.qnn[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[13*n+idx]), density));
    r.qvolg[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[14*n+idx]), density));
    r.qvolh[idx] = as_type<float>(soft32_div(as_type<uint>(r.state[15*n+idx]), density));
}

// Hot gather only: first-step and cumulus power branches are refused by the caller.
kernel void nssl2_driver_gather_hot_exact(
    constant Nssl2DriverGatherHotExactResources &r [[buffer(0)]],
    constant Nssl2DriverGatherHotExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint n = uint(p.n), rho = as_type<uint>(r.air_density[idx]);
    r.state[idx] = r.qv[idx];
    r.state[n+idx] = r.qc[idx];
    r.state[2*n+idx] = r.qr[idx];
    r.state[3*n+idx] = r.qi[idx];
    r.state[4*n+idx] = r.qs[idx];
    r.state[5*n+idx] = r.qg[idx];
    r.state[6*n+idx] = r.qh[idx];
    r.state[7*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qndrop[idx]), rho));
    r.state[8*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qnr[idx]), rho));
    r.state[9*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qni[idx]), rho));
    r.state[10*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qns[idx]), rho));
    r.state[11*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qng[idx]), rho));
    r.state[12*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qnh[idx]), rho));
    const uint ccn = p.predicted_ccn ? as_type<uint>(r.qnn[idx]) :
        soft32_sub(as_type<uint>(408163264.0f), as_type<uint>(r.qndrop[idx]));
    r.state[13*n+idx] = as_type<float>(soft32_mul(ccn, rho));
    r.state[14*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qvolg[idx]), rho));
    r.state[15*n+idx] = as_type<float>(soft32_mul(as_type<uint>(r.qvolh[idx]), rho));
}

// Source NUCOND prepass: the table is evaluated on demand and qv is raw.
kernel void nssl2_nucond_supersaturation_exact(
    constant Nssl2NucondSupersaturationExactResources &r [[buffer(0)]],
    constant Nssl2NucondSupersaturationExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint temp = soft32_mul(as_type<uint>(r.full_theta[idx]), as_type<uint>(r.exner[idx]));
    const uint idxword = soft32_add(soft32_div(soft32_sub(temp, as_type<uint>(163.15f)),
                                               as_type<uint>(0.002f)), as_type<uint>(1.5f));
    const int l = clamp(int(as_type<float>(idxword)), 1, 1000001);
    const uint temq = soft32_add(as_type<uint>(163.15f),
        soft32_mul(as_type<uint>(float(l - 1)), as_type<uint>(0.002f)));
    const uint table = soft_gfk_exp(soft32_div(
        soft32_mul(as_type<uint>(17.2693882f), soft32_sub(temq, as_type<uint>(273.15f))),
        soft32_sub(temq, as_type<uint>(35.86f))));
    const uint c1 = soft32_mul(soft32_div(as_type<uint>(380.0f), as_type<uint>(r.pressure_pa[idx])), table);
    uint ss = 0u;
    if ((c1 & 0x80000000u) == 0u && c1 != 0u && c1 <= 0x7f800000u)
        ss = soft32_mul(as_type<uint>(100.0f),
            soft32_sub(soft32_div(as_type<uint>(r.qv[idx]), c1), as_type<uint>(1.0f)));
    r.supersaturation_percent[idx] = as_type<float>(ss);
}

static inline uint nssl2_fmin(uint a, uint b) {
    if (soft32_is_nan(a)) return b;
    if (soft32_is_nan(b)) return a;
    if (((a | b) & 0x7fffffffu) == 0u) return a | b;
    return soft32_less(a, b) ? a : b;
}
static inline uint nssl2_fmax(uint a, uint b) {
    if (soft32_is_nan(a)) return b;
    if (soft32_is_nan(b)) return a;
    if (((a | b) & 0x7fffffffu) == 0u) return a & b;
    return soft32_less(a, b) ? b : a;
}

// Source prepass snapshots temperature and the primary ice target.
kernel void nssl2_prepare_fused_gs_exact(
    constant Nssl2PrepareFusedGsExactResources &r [[buffer(0)]],
    constant Nssl2PrepareFusedGsExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint temperature = soft32_mul(as_type<uint>(r.full_theta[idx]), as_type<uint>(r.exner[idx]));
    r.temperature_k[idx] = as_type<float>(temperature);
    const uint lbits = soft32_add(soft32_div(soft32_sub(temperature, as_type<uint>(163.15f)),
                                             as_type<uint>(0.002f)), as_type<uint>(1.5f));
    const int l = clamp(int(as_type<float>(lbits)), 1, 1000001);
    const uint temq = soft32_add(as_type<uint>(163.15f),
        soft32_mul(as_type<uint>(float(l - 1)), as_type<uint>(0.002f)));
    const uint difference = soft32_sub(temq, as_type<uint>(273.15f));
    const uint tabqvs = soft_gfk_exp(soft32_div(soft32_mul(as_type<uint>(17.2693882f), difference),
                                                soft32_sub(temq, as_type<uint>(35.86f))));
    const uint tabqis = soft_gfk_exp(soft32_div(soft32_mul(as_type<uint>(21.87455f), difference),
                                                soft32_sub(temq, as_type<uint>(7.66f))));
    const uint factor = soft32_div(as_type<uint>(380.0f), as_type<uint>(r.pressure_pa[idx]));
    const uint t8s = soft32_mul(factor, tabqvs), t9s = soft32_mul(factor, tabqis);
    const uint ss = soft32_div(nssl2_fmin(t8s, nssl2_fmax(as_type<uint>(r.state[idx]), 0u)), t9s);
    uint target = 0u;
    if (soft32_less(as_type<uint>(1.0f), ss) && soft32_less_equal(temperature, as_type<uint>(268.15f))) {
        const uint arg = soft32_sub(soft32_mul(as_type<uint>(12.96f),
            soft32_sub(ss, as_type<uint>(1.0f))), as_type<uint>(0.639f));
        target = soft32_mul(soft32_mul(soft32_div(as_type<uint>(r.air_density[idx]),
            as_type<uint>(1.225f)), as_type<uint>(1.0e3f)),
            soft_gfk_exp(nssl2_fmin(as_type<uint>(57.0f), arg)));
    }
    r.target_m3[idx] = as_type<float>(target);
}

// WRF concentration and Registry-number variants differ at the number load.
static inline uint3 nssl2_exact_radii(uint rho, uint qc, uint nc, uint qi, uint ni,
                                       uint qs, uint ns, bool registry) {
    const uint pi6 = soft32_div(as_type<uint>(3.14159265358979323846f), as_type<uint>(6.0f));
    const uint third = soft32_div(as_type<uint>(1.0f), as_type<uint>(3.0f));
    const uint masses[3] = {nssl2_fmax(qc, 0u),
        nssl2_fmax(qi, 0u), nssl2_fmax(qs, 0u)};
    uint numbers[3] = {nssl2_fmax(nc, 0u),
        nssl2_fmax(ni, 0u), nssl2_fmax(ns, 0u)};
    if (registry) for (uint i = 0; i < 3; ++i) numbers[i] = soft32_mul(numbers[i], rho);
    const uint lower[3] = {as_type<uint>(2.51e-6f), as_type<uint>(10.01e-6f), as_type<uint>(25.0e-6f)};
    const uint upper[3] = {as_type<uint>(50.0e-6f), as_type<uint>(125.0e-6f), as_type<uint>(999.0e-6f)};
    const uint density[3] = {as_type<uint>(1000.0f), as_type<uint>(900.0f), as_type<uint>(100.0f)};
    uint output[3];
    for (uint i = 0; i < 3; ++i) {
        output[i] = lower[i];
        if (soft32_less(as_type<uint>(1.0e-13f), masses[i]) &&
            soft32_less(as_type<uint>(1.0e-8f), numbers[i])) {
            uint numer = soft32_mul(soft32_mul(numbers[i], pi6), density[i]);
            uint denom = soft32_mul(masses[i], rho);
            uint factor = as_type<uint>(1.1077321767807007f);
            if (i == 2) {
                numer = soft32_mul(numer, 0x3f6b0d1bu);
                denom = soft32_mul(denom, registry ? 0x4092e831u : 0x4092e832u);
                factor = 0x3f5641b3u;
            }
            const uint lambda = soft_gfk_pow_bits(soft32_div(numer, denom), third);
            const uint raw = soft32_div(soft32_mul(as_type<uint>(0.5f), factor), lambda);
            output[i] = nssl2_fmax(lower[i], nssl2_fmin(raw, upper[i]));
        }
    }
    return uint3(output[0], output[1], output[2]);
}

kernel void nssl2_effective_radius_concentration_exact(
    constant Nssl2EffectiveRadiusConcentrationExactResources &r [[buffer(0)]],
    constant Nssl2EffectiveRadiusConcentrationExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint3 v = nssl2_exact_radii(as_type<uint>(r.air_density[idx]),
        as_type<uint>(r.qc[idx]), as_type<uint>(r.cloud_number[idx]),
        as_type<uint>(r.qi[idx]), as_type<uint>(r.ice_number[idx]),
        as_type<uint>(r.qs[idx]), as_type<uint>(r.snow_number[idx]), false);
    r.re_cloud[idx] = as_type<float>(v.x);
    r.re_ice[idx] = as_type<float>(v.y);
    r.re_snow[idx] = as_type<float>(v.z);
}

kernel void nssl2_effective_radius_registry_exact(
    constant Nssl2EffectiveRadiusRegistryExactResources &r [[buffer(0)]],
    constant Nssl2EffectiveRadiusRegistryExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint3 v = nssl2_exact_radii(as_type<uint>(r.air_density[idx]),
        as_type<uint>(r.qc[idx]), as_type<uint>(r.cloud_number[idx]),
        as_type<uint>(r.qi[idx]), as_type<uint>(r.ice_number[idx]),
        as_type<uint>(r.qs[idx]), as_type<uint>(r.snow_number[idx]), true);
    r.re_cloud[idx] = as_type<float>(v.x);
    r.re_ice[idx] = as_type<float>(v.y);
    r.re_snow[idx] = as_type<float>(v.z);
}

kernel void nssl2_radius_microns_exact(
    constant Nssl2RadiusMicronsExactResources &r [[buffer(0)]],
    constant Nssl2RadiusMicronsExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const uint scale = as_type<uint>(1.0e6f);
    r.cloud[idx] = as_type<float>(soft32_mul(as_type<uint>(r.cloud[idx]), scale));
    r.ice[idx] = as_type<float>(soft32_mul(as_type<uint>(r.ice[idx]), scale));
    r.snow[idx] = as_type<float>(soft32_mul(as_type<uint>(r.snow[idx]), scale));
}

// The radar source mixes REAL and REAL*8; every operation stays in its type.
struct NsslRadar64 { ulong bits; };
static inline NsslRadar64 radar64(F x) { return {soft64_from_f32_bits(x.bits)}; }
static inline NsslRadar64 operator+(NsslRadar64 a, NsslRadar64 b) { return {soft64_add(a.bits,b.bits)}; }
static inline NsslRadar64 operator*(NsslRadar64 a, NsslRadar64 b) { return {soft64_mul(a.bits,b.bits)}; }
static inline NsslRadar64 operator/(NsslRadar64 a, NsslRadar64 b) { return {soft64_div(a.bits,b.bits)}; }
static inline F radar32(NsslRadar64 x) { return f32_word(soft64_to_f32_bits(x.bits)); }
static inline F operator/(F a, F b) { return f32_word(soft32_div(a.bits,b.bits)); }
static inline F radar_max(F a, F b) { return b > a ? b : a; }
static inline F radar_min(F a, F b) { return b < a ? b : a; }
static inline F radar_log10(F x) {
    int hx = as_type<int>(x.bits), k = 0;
    if (hx < 0x00800000) { k = -25; x = x * F(33554432.0f); hx = as_type<int>(x.bits); }
    k += (hx >> 23) - 127;
    const int i = k < 0 ? 1 : 0;
    hx = (hx & 0x007fffff) | ((0x7f - i) << 23);
    const F y(float(k + i));
    const F z = y * f32_word(0x355427dbu) + f32_word(0x3ede5bd9u) * gfk_log(f32_word(as_type<uint>(hx)));
    return z + y * f32_word(0x3e9a2080u);
}

kernel void nssl2_radardd02_exact(
    constant Nssl2Radardd02ExactResources &r [[buffer(0)]],
    constant Nssl2Radardd02ExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const F db(r.air_density[idx]), scale = p.concentration_space ? F(1.0f) : db;
    const F lr(r.qr[idx]), li(r.qi[idx]), ls(r.qs[idx]), lh(r.qg[idx]), lhl(r.qh[idx]);
    const F lnr=F(r.qnr[idx])*scale, lni=F(r.qni[idx])*scale, lns=F(r.qns[idx])*scale;
    const F lnh=F(r.qng[idx])*scale, lnhl=F(r.qnh[idx])*scale;
    const F lvh=F(r.qvolg[idx])*scale, lvhl=F(r.qvolh[idx])*scale;
    const F sixpi = F(6.0f)/(F(3.141592653589793f)*F(1000.0f));
    const NsslRadar64 sixpi2=radar64(sixpi*sixpi), e18=radar64(F(1.e18f));
    F total(0.0f);
    if (lr >= F(1.e-12f) && lnr > F(1.e-3f)) {
        const NsslRadar64 g1=radar64(((F(6.0f)+F(0.0f))*(F(5.0f)+F(0.0f))*(F(4.0f)+F(0.0f)))/
                                      ((F(3.0f)+F(0.0f))*(F(2.0f)+F(0.0f))*(F(1.0f)+F(0.0f))));
        const F v=db*lr;
        const NsslRadar64 zx=(g1*radar64(v*v))/radar64(lnr);
        total=radar32((e18*zx)*sixpi2);
    }
    if (ls >= F(1.e-13f) && lns > F(1.e-7f)) {
        const F ksq(0.189f), qxw(0.0f), snu(-0.8f);
        const F c0=(F(1.e18f)*F(323.3226f))*(F(0.106214f)*F(0.106214f));
        const F numer=(((c0*(ksq*ls+(F(1.0f)-ksq)*qxw))*ls)*(db*db))*f32_word(0x3f63445eu);
        const F denom=((lns*F(840889.0f))*f32_word(0x4092e832u))*gfk_pow(F(1.0f)+snu,F(4.0f)/F(3.0f));
        total=total+numer/denom;
    }
    if (li > F(1.e-13f) && lni > F(1.0f)) {
        const F cinu(0.0f), vr=(db*li)/(F(900.0f)*lni);
        const F term=((((F(0.224f)*F(3.6e18f))*(cinu+F(2.0f)))*lni)*(vr*vr))/(cinu+F(1.0f))*(F(0.9f)*F(0.9f));
        total=total+term;
    }
    if (lh >= F(1.e-12f) && lnh >= F(1.e-8f)) {
        F density=lvh>F(0.0f) ? radar_min(radar_max((db*lh)/lvh,F(100.0f)),F(900.0f)) : F(500.0f);
        F number=lnh;
        if (number > F(0.0f)) {
            F volume=(db*lh)/(density*radar_max(number,F(1.e-3f)));
            if (volume < f32_word(0x2d78b42du) || volume > f32_word(0x368c8d72u)) {
                volume=radar_min(radar_max(volume,f32_word(0x2d78b42du)),f32_word(0x368c8d72u));
                number=(db*lh)/(volume*density);
            }
            const F zero(0.0f);
            const NsslRadar64 g1=radar64((F(6.0f)*F(5.0f)*F(4.0f))/(F(3.0f)*F(2.0f)*F(1.0f)));
            const NsslRadar64 zx=(((g1*radar64(db*db))*radar64(F(0.224f)*lh+F(0.776f)*zero))*radar64(lh))/radar64(number);
            const NsslRadar64 ze=(e18*zx)*sixpi2;
            total=radar32(radar64(total)+ze);
        }
    }
    {
        F density=lvhl>F(0.0f) ? radar_min(radar_max((db*lhl)/lvhl,F(300.0f)),F(900.0f)) : F(900.0f);
        if (lhl >= F(1.e-12f) && lnhl > F(0.0f)) {
            F number=lnhl;
            if (number > F(0.0f)) {
                F volume=(db*lhl)/(density*radar_max(lnhl,F(1.e-9f)));
                if (volume < f32_word(0x2d78b42eu) || volume > f32_word(0x380c8d72u)) {
                    volume=radar_min(radar_max(volume,f32_word(0x2d78b42eu)),f32_word(0x380c8d72u));
                    number=(db*lhl)/(volume*density);
                }
                const F zero(0.0f), alpha(1.0f);
                const NsslRadar64 g1=radar64(((F(6.0f)+alpha)*(F(5.0f)+alpha)*(F(4.0f)+alpha))/
                                             ((F(3.0f)+alpha)*(F(2.0f)+alpha)*(F(1.0f)+alpha)));
                const NsslRadar64 zx=(((g1*radar64(db*db))*radar64(F(0.224f)*lhl+F(0.776f)*zero))*radar64(lhl))/radar64(number);
                const NsslRadar64 ze=(e18*zx)*sixpi2;
                total=radar32(radar64(total)+ze);
            }
        }
    }
    r.refl_10cm[idx] = float(total>F(0.0f) ? radar_max(F(10.0f)*radar_log10(total),F(0.0f)) : F(0.0f));
}

static inline F cloud_min(F a, F b) { return f32_word(nssl2_fmin(a.bits, b.bits)); }
static inline F cloud_max(F a, F b) { return f32_word(nssl2_fmax(a.bits, b.bits)); }

// Serial-column form of the selected strict CUDA cloud sedimentation.
kernel void nssl2_cloud_sediment_exact(
    constant Nssl2CloudSedimentExactResources &r [[buffer(0)]],
    constant Nssl2CloudSedimentExactParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column >= ncol) return;
    F rho[256], cloud[256], number[256], velocity[256], mass_flux[257], number_flux[257];
    const F dt(p.dt), mass_to_diameter=F(6.0f)/(F(3.14159265358979323846f)*F(1000.0f));
    F max_rate(0.0f);
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column;
        rho[k]=F(r.air_density[idx]); cloud[k]=F(r.qc[idx]); number[k]=F(r.qndrop[idx]); velocity[k]=F(0.0f);
        const F positive_cloud=cloud_max(cloud[k],F(0.0f));
        if (positive_cloud > F(1.0e-13f)) {
            const F positive_number=cloud_max(number[k],F(0.0f));
            const F effective_number=positive_number>F(1.0e-8f) ? positive_number :
                cloud_max(F(1.0e-8f),(rho[k]*positive_cloud)/f32_word(0x3078b42bu));
            const F particle_mass=cloud_min(f32_word(0x3078b42bu),
                cloud_max(f32_word(0x2916eacau),(positive_cloud*rho[k])/effective_number));
            const F diameter=gfk_pow(particle_mass*mass_to_diameter,F(1.0f)/F(3.0f));
            const F radius=F(0.5f)*diameter, temperature(r.temperature_k[idx]);
            const F viscosity=(F(1.832e-5f)*(F(416.16f)/(temperature+F(120.0f))))*
                gfk_pow(temperature/F(296.0f),F(1.5f));
            if (viscosity > F(0.0f))
                velocity[k]=cloud_min(F(70.0f),((F(2.0f)*F(9.8f)*F(1000.0f))*(radius*radius))/(F(9.0f)*viscosity));
        }
        max_rate=cloud_max(max_rate,velocity[k]*(F(1.0f)/F(r.dz[idx])));
    }
    if (max_rate==F(0.0f)) { r.cloud_surface_export[column]=0.0f; return; }
    int substeps;
    if (dt*max_rate<F(0.7f)) substeps=1;
    else if (dt>F(20.0f)) substeps=max(2,int(float((dt*max_rate)/F(0.7f)))+1);
    else substeps=1+int(float(dt*max_rate+F(0.301f)));
    const F dt_substep=dt/F(float(substeps)), dt_fraction=dt_substep/dt;
    F surface_mean_flux(0.0f);
    for (int step=0;step<substeps;++step) {
        for (int k=0;k<p.nz;++k) {
            mass_flux[k]=(cloud[k]*velocity[k])*rho[k];
            number_flux[k]=number[k]*velocity[k];
        }
        mass_flux[p.nz]=F(0.0f); number_flux[p.nz]=F(0.0f);
        surface_mean_flux=surface_mean_flux+(cloud[0]*velocity[0])*dt_fraction;
        for (int k=0;k<p.nz;++k) {
            const uint idx=uint(k)*ncol+column;
            const F inverse_dz=F(1.0f)/F(r.dz[idx]);
            cloud[k]=cloud[k]+(dt_substep*(inverse_dz*(F(1.0f)/rho[k])))*(mass_flux[k+1]-mass_flux[k]);
            number[k]=number[k]+(dt_substep*inverse_dz)*(number_flux[k+1]-number_flux[k]);
        }
    }
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column; r.qc[idx]=float(cloud[k]); r.qndrop[idx]=float(number[k]);
    }
    r.cloud_surface_export[column]=float((dt*rho[0])*surface_mean_flux);
}

static inline F nssl2_sediment_z_exact(F q, F number, F rho, F xdn, F g1,
                                        F minimum, F maximum) {
    if (!(q>F(1.0e-12f)) || !(number>F(1.0e-15f))) return F(0.0f);
    F xv=(rho*q)/(xdn*number), adjusted=number;
    if (xv<minimum || xv>maximum) {
        xv=cloud_min(maximum,cloud_max(minimum,xv));
        adjusted=(rho*q)/(xv*xdn);
    }
    const F zx=(((g1*(rho*rho))*q)*q)/adjusted;
    const F scale=F(6.0f)/(F(3.141592653589793f)*F(1000.0f));
    return zx*(scale*scale);
}

static inline F nssl2_sediment_number_exact(F q, F number, F rho, F xdn, F g1,
        F minimum, F maximum, F z_advected, F z_initial, F mass_weighted) {
    F xv=(rho*q)/(xdn*number);
    NsslRadar64 adjusted=radar64(number);
    if (xv<minimum || xv>maximum) {
        xv=cloud_min(maximum,cloud_max(minimum,xv));
        adjusted=radar64((rho*q)/(xv*xdn));
    }
    const NsslRadar64 g=radar64(g1), rho2=radar64(rho*rho), qd=radar64(q);
    NsslRadar64 zx=(((g*rho2)*qd)*qd)/adjusted;
    const F scale=F(6.0f)/(F(3.141592653589793f)*F(1000.0f));
    const NsslRadar64 z=zx*radar64(scale*scale);
    if (soft64_greater(z.bits,radar64(z_advected).bits) &&
        soft64_greater(z.bits,radar64(F(0.0f)).bits) && z_advected>z_initial) {
        zx=radar64(z_advected/(scale*scale));
        const NsslRadar64 nrx=(((g*rho2)*qd)*qd)/zx;
        return cloud_max(cloud_min(radar32(nrx),mass_weighted),number);
    }
    return cloud_max(mass_weighted,number);
}

// Independent serial rain category; the shallow combined parallel path is separate.
kernel void nssl2_rain_sediment_exact(
    constant Nssl2RainSedimentExactResources &r [[buffer(0)]],
    constant Nssl2RainSedimentExactParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column >= ncol) return;
    F rho[256], rain[256], number[256], vm[256], vn[256], vz[256];
    F fm[257], fn[257], fz[257], fmn[257], z0[256], za[256], nm[256];
    const F dt(p.dt), minimum=f32_word(0x2a96eac9u), maximum=f32_word(0x33f2dff3u);
    const F max_speed_volume=maximum/(F(64.0f)/F(6.0f));
    F max_rate(0.0f);
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column;
        rho[k]=F(r.air_density[idx]);rain[k]=F(r.qr[idx]);number[k]=F(r.qnr[idx]);
        vm[k]=F(0.0f);vn[k]=F(0.0f);vz[k]=F(0.0f);
        const F positive=cloud_max(rain[k],F(0.0f));
        if (positive>F(1.0e-12f)) {
            const F local=cloud_max(number[k],F(0.0f));
            F volume=(rho[k]*positive)/(F(1000.0f)*cloud_max(F(1.0e-11f),local));
            if (volume>max_speed_volume) volume=max_speed_volume;
            else if (volume<minimum) volume=minimum;
            const F piinv=F(1.0f)/F(3.141592653589793f);
            const F diameter=gfk_pow(((F(6.0f)*piinv)*volume)/(F(3.0f)*F(2.0f)*F(1.0f)),F(1.0f)/F(3.0f));
            const F density=f32_word(soft32_sqrt((F(1.225f)*cloud_min(F(20.0f),F(1.0f)/rho[k])).bits));
            const F speed=F(1.0f)+F(516.575f)*diameter;
            F a=(density*F(10.0f))*(F(1.0f)-gfk_pow(speed,F(-4.0f)));
            F b=(density*F(10.0f))*(F(1.0f)-gfk_pow(speed,F(-1.0f)));
            F c=(density*F(10.0f))*(F(1.0f)-gfk_pow(speed,F(-7.0f)));
            if (b>a || (a>c && c>F(0.0f))) { a=cloud_max(a,b); c=cloud_max(c,a); }
            vm[k]=cloud_min(F(70.0f),cloud_min(F(150.0f),a));
            vn[k]=cloud_min(F(70.0f),cloud_min(F(150.0f),b));
            vz[k]=cloud_min(F(70.0f),cloud_min(F(150.0f),c));
        }
        const F inverse=F(1.0f)/F(r.dz[idx]);
        max_rate=cloud_max(max_rate,vm[k]*inverse);
        max_rate=cloud_max(max_rate,vn[k]*inverse);
        max_rate=cloud_max(max_rate,vz[k]*inverse);
    }
    if (max_rate==F(0.0f)) { r.rainncv[column]=0.0f; return; }
    int steps;
    if (dt*max_rate<F(0.7f)) steps=1;
    else if (dt>F(20.0f)) steps=max(2,int(float((dt*max_rate)/F(0.7f)))+1);
    else steps=1+int(float(dt*max_rate+F(0.301f)));
    const F dt_sub=dt/F(float(steps)), dt_fraction=dt_sub/dt;
    F surface(0.0f);
    for (int step=0;step<steps;++step) {
        for (int k=0;k<p.nz;++k) {
            z0[k]=nssl2_sediment_z_exact(rain[k],number[k],rho[k],F(1000.0f),F(20.0f),minimum,maximum);
            za[k]=z0[k];nm[k]=number[k];
            fm[k]=(rain[k]*vm[k])*rho[k];fn[k]=number[k]*vn[k];
            fz[k]=z0[k]*vz[k];fmn[k]=number[k]*vm[k];
        }
        fm[p.nz]=F(0.0f);fn[p.nz]=F(0.0f);fz[p.nz]=F(0.0f);fmn[p.nz]=F(0.0f);
        surface=surface+(rain[0]*vm[0])*dt_fraction;
        for (int k=0;k<p.nz;++k) {
            const uint idx=uint(k)*ncol+column;const F inverse=F(1.0f)/F(r.dz[idx]);
            rain[k]=rain[k]+(dt_sub*(inverse*(F(1.0f)/rho[k])))*(fm[k+1]-fm[k]);
            number[k]=number[k]+(dt_sub*inverse)*(fn[k+1]-fn[k]);
            za[k]=za[k]+(dt_sub*inverse)*(fz[k+1]-fz[k]);
            nm[k]=nm[k]+(dt_sub*inverse)*(fmn[k+1]-fmn[k]);
        }
        for (int k=0;k<p.nz;++k) if (za[k]>F(0.0f))
            number[k]=nssl2_sediment_number_exact(rain[k],number[k],rho[k],F(1000.0f),F(20.0f),minimum,maximum,za[k],z0[k],nm[k]);
    }
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column;r.qr[idx]=float(rain[k]);r.qnr[idx]=float(number[k]);
    }
    r.rainncv[column]=float(surface);
    r.rainnc[column]=float(F(r.rainnc[column])+surface);
}

// Independent serial cloud-ice category; shallow production parallel is separate.
kernel void nssl2_ice_sediment_exact(
    constant Nssl2IceSedimentExactResources &r [[buffer(0)]],
    constant Nssl2IceSedimentExactParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column >= ncol) return;
    F rho[256], ice[256], number[256], vm[256], vn[256];
    F fm[257], fn[257], fmn[257], nm[256];
    const F dt(p.dt), minimum(6.88e-13f), maximum(1.0e-8f);
    F max_rate(0.0f);
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column;
        rho[k]=F(r.air_density[idx]);ice[k]=F(r.qi[idx]);number[k]=F(r.qni[idx]);
        vm[k]=F(0.0f);vn[k]=F(0.0f);
        const F positive=cloud_max(ice[k],F(0.0f));
        if (positive>F(1.0e-13f)) {
            F local=cloud_max(number[k],F(0.0f));
            local=cloud_max(local,(rho[k]*positive)/maximum);
            local=cloud_min(local,(rho[k]*positive)/minimum);
            const F particle=cloud_max((rho[k]*positive)/local,minimum);
            const F volume=particle/F(900.0f);
            const F density=f32_word(soft32_sqrt((F(1.225f)*cloud_min(F(20.0f),F(1.0f)/rho[k])).bits));
            const F tmp=(F(47.6273f)*density)/gfk_pow(F(1.0f)/volume,F(0.18333f));
            vn[k]=cloud_min(F(70.0f),tmp*F(0.922766923904419f));
            vm[k]=cloud_min(F(70.0f),tmp*F(1.091937899589539f));
        }
        const F inverse=F(1.0f)/F(r.dz[idx]);
        max_rate=cloud_max(max_rate,vm[k]*inverse);
        max_rate=cloud_max(max_rate,vn[k]*inverse);
    }
    if (max_rate==F(0.0f)) { r.icencv[column]=0.0f; return; }
    int steps;
    if (dt*max_rate<F(0.7f)) steps=1;
    else if (dt>F(20.0f)) steps=max(2,int(float((dt*max_rate)/F(0.7f)))+1);
    else steps=1+int(float(dt*max_rate+F(0.301f)));
    const F dt_sub=dt/F(float(steps)), dt_fraction=dt_sub/dt;
    F surface(0.0f);
    for (int step=0;step<steps;++step) {
        for (int k=0;k<p.nz;++k) {
            fm[k]=(ice[k]*vm[k])*rho[k];fn[k]=number[k]*vn[k];
            fmn[k]=number[k]*vm[k];nm[k]=number[k];
        }
        fm[p.nz]=F(0.0f);fn[p.nz]=F(0.0f);fmn[p.nz]=F(0.0f);
        surface=surface+(ice[0]*vm[0])*dt_fraction;
        for (int k=0;k<p.nz;++k) {
            const uint idx=uint(k)*ncol+column;const F inverse=F(1.0f)/F(r.dz[idx]);
            ice[k]=ice[k]+(dt_sub*(inverse*(F(1.0f)/rho[k])))*(fm[k+1]-fm[k]);
            number[k]=number[k]+(dt_sub*inverse)*(fn[k+1]-fn[k]);
            nm[k]=nm[k]+(dt_sub*inverse)*(fmn[k+1]-fmn[k]);
        }
        for (int k=0;k<p.nz;++k) number[k]=cloud_max(number[k],nm[k]);
    }
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column;r.qi[idx]=float(ice[k]);r.qni[idx]=float(number[k]);
    }
    const F exported=(dt*rho[0])*surface;
    r.icencv[column]=float(exported);r.icenc[column]=float(F(r.icenc[column])+exported);
}

// Independent serial snow category; shallow production parallel is separate.
kernel void nssl2_snow_sediment_exact(
    constant Nssl2SnowSedimentExactResources &r [[buffer(0)]],
    constant Nssl2SnowSedimentExactParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column >= ncol) return;
    F rho[256], snow[256], number[256], vm[256], vn[256], vz[256];
    F fm[257], fn[257], fmn[257], nm[256];
    const F dt(p.dt), minimum=f32_word(0x2616eac9u), maximum=f32_word(0x350c8d72u);
    F max_rate(0.0f);
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column;
        rho[k]=F(r.air_density[idx]);snow[k]=F(r.qs[idx]);number[k]=F(r.qns[idx]);
        vm[k]=F(0.0f);vn[k]=F(0.0f);vz[k]=F(0.0f);
        const F positive=cloud_max(snow[k],F(0.0f));
        if (positive>F(1.0e-13f)) {
            const F local=cloud_max(number[k],F(0.0f));
            F volume=(rho[k]*positive)/(F(100.0f)*cloud_max(F(1.0e-9f),local));
            volume=cloud_min(maximum,cloud_max(minimum,volume));
            const F density=f32_word(soft32_sqrt((F(1.225f)*cloud_min(F(20.0f),F(1.0f)/rho[k])).bits));
            const F size=gfk_pow(volume,F(0.14f));
            vm[k]=cloud_min(F(70.0f),(F(11.9495f)*density)*size);
            vn[k]=cloud_min(F(70.0f),(F(7.02909f)*density)*size);
            vz[k]=cloud_min(F(70.0f),(F(13.3436f)*density)*size);
        }
        const F inverse=F(1.0f)/F(r.dz[idx]);
        max_rate=cloud_max(max_rate,vm[k]*inverse);
        max_rate=cloud_max(max_rate,vn[k]*inverse);
        max_rate=cloud_max(max_rate,vz[k]*inverse);
    }
    if (max_rate==F(0.0f)) { r.snowncv[column]=0.0f; return; }
    int steps;
    if (dt*max_rate<F(0.7f)) steps=1;
    else if (dt>F(20.0f)) steps=max(2,int(float((dt*max_rate)/F(0.7f)))+1);
    else steps=1+int(float(dt*max_rate+F(0.301f)));
    const F dt_sub=dt/F(float(steps)), dt_fraction=dt_sub/dt;
    F surface(0.0f);
    for (int step=0;step<steps;++step) {
        for (int k=0;k<p.nz;++k) {
            fm[k]=(snow[k]*vm[k])*rho[k];fn[k]=number[k]*vn[k];
            fmn[k]=number[k]*vm[k];nm[k]=number[k];
        }
        fm[p.nz]=F(0.0f);fn[p.nz]=F(0.0f);fmn[p.nz]=F(0.0f);
        surface=surface+(snow[0]*vm[0])*dt_fraction;
        for (int k=0;k<p.nz;++k) {
            const uint idx=uint(k)*ncol+column;const F inverse=F(1.0f)/F(r.dz[idx]);
            snow[k]=snow[k]+(dt_sub*(inverse*(F(1.0f)/rho[k])))*(fm[k+1]-fm[k]);
            number[k]=number[k]+(dt_sub*inverse)*(fn[k+1]-fn[k]);
            nm[k]=nm[k]+(dt_sub*inverse)*(fmn[k+1]-fmn[k]);
        }
        for (int k=0;k<p.nz;++k) number[k]=cloud_max(number[k],nm[k]);
    }
    for (int k=0;k<p.nz;++k) {
        const uint idx=uint(k)*ncol+column;r.qs[idx]=float(snow[k]);r.qns[idx]=float(number[k]);
    }
    r.snowncv[column]=float(surface);r.snownc[column]=float(F(r.snownc[column])+surface);
}

// Runtime publishes temperature before sedimentation and after condensation.
kernel void nssl2_temperature_product_exact(
    constant Nssl2TemperatureProductExactResources &r [[buffer(0)]],
    constant Nssl2TemperatureProductExactParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.nz * p.ny * p.nx)) return;
    r.temperature[idx] = as_type<float>(soft32_mul(as_type<uint>(r.theta[idx]),
                                                   as_type<uint>(r.pii[idx])));
}

// Generated from the selected strict NUCOND source by tools/generate_nssl2_nucond_exact.py.
static inline F nucond_sqrt(F x) { return f32_word(soft32_sqrt(x.bits)); }
static inline F nucond_abs(F x) { return f32_word(x.bits & 0x7fffffffu); }
static inline int nucond_lround(F x) {
    const uint magnitude=x.bits & 0x7fffffffu; const int exponent=int(magnitude>>23)-127;
    if (exponent < -1) return 0;
    const uint sig=(magnitude & 0x7fffffu)|0x800000u;
    uint value;
    if (exponent < 0) value=1u;
    else if (exponent >= 23) value=sig << uint(exponent-23);
    else { const uint shift=uint(23-exponent); value=(sig >> shift)+uint((sig & ((1u << shift)-1u)) >= (1u << (shift-1u))); }
    return (x.bits & 0x80000000u) != 0u ? -int(value) : int(value);
}

#define NSSL2_WRF_CWMASN f32_word(0x2916eacau)
#define NSSL2_WRF_CWMASX f32_word(0x3078b42bu)
#define NSSL2_WRF_XVRMN f32_word(0x2a96eac9u)
#define NSSL2_WRF_XVRMX0 f32_word(0x33f2dff3u)
namespace {

constant constexpr float PI = 3.141592653589793f;          // 0x40490FDB
constant constexpr float PIINV = 1.0f / 3.141592653589793f; // 0x3EA2F983
constant constexpr float TFR = 273.15f;
constant constexpr float TFRH = 233.15f;
constant constexpr float NCP = 1004.0f;
constant constexpr float RW = 461.5f;
constant constexpr float RHO00 = 1.225f;
constant constexpr float ADVISC0 = 1.832e-05f;
constant constexpr float ADVISC1 = 1.718e-05f;
constant constexpr float TKA0 = 2.43e-2f;
constant constexpr float CAW = 17.2693882f;
constant constexpr float CBW = 35.86f;
constant constexpr float FQSAT = 0.002f;
constant constexpr int NQSAT = 1000001;
constant constexpr float CXMIN = 1.e-8f;
constant constexpr float AR = 841.99666f;
constant constexpr float BR = 0.8f;
constant constexpr float CWCCN = 5.0e8f;
constant constexpr float QCCN = 408163264.0f;               // 0x4DC2A09E
constant constexpr float CCK = 0.6f;
constant constexpr float MAXLOWTEMPSS = 1.08f;
constant constexpr float MAXSUPERSAT = 1.9f;
constant constexpr float SSMXINIT = 0.4f;
constant constexpr float RESTORECCNFRAC = 1.0f;
constant constexpr float CCNTIMECONST = 3600.0f;
constant constexpr float SUPCB = 0.5f;
constant constexpr float SUPMX = 238.0f;
constant constexpr float QXMIN_C = 1.0e-13f;
constant constexpr float QXMIN_R = 1.0e-12f;
constant constexpr float QXMIN_I = 1.0e-13f;
constant constexpr float QXMIN_S = 1.0e-13f;
constant constexpr float QXMIN_H = 1.0e-12f;
constant constexpr float QXMIN_HL = 1.0e-12f;
constant constexpr float XDN_C = 1000.0f;
constant constexpr float XDN_R = 1000.0f;
constant constexpr float CNOHMN = 1.e-2f;

static inline F ventc() { return f32_word(0x3F649A4E); }
static inline F ventrn() { return f32_word(0x3FE9E6C7); }
static inline F cpi() { return f32_word(0x3A828CC0); }
static inline F cwc1() { return f32_word(0x3AFA543E); }
static inline F c1f3() { return f32_word(0x3EAAAAAB); }
static inline F ccne0() { return f32_word(0x41BFE0CE); }
static inline F cnexp() { return f32_word(0x3EB13B15); }
static inline F cwmasn5() { return f32_word(0x2B136148); }
static inline F cwmas20() { return f32_word(0x2E136148); }
// (4.*pi/3.) and (4.e-6)**3, folded REAL parameters.
static inline F four_pi_third() { return f32_word(0x40860A92); }
static inline F four_micron_cube() { return f32_word(0x249392EE); }
static inline F xvmn_c() { return f32_word(0x241A8A06); }
static inline F xvmx_c() { return f32_word(0x2B7EAC34); }

static inline F nssl2_max(F a, F b) { return (b > a) ? b : a; }
static inline F nssl2_min(F a, F b) { return (b < a) ? b : a; }

static inline F fqsati() { return __fdiv_rn(1.0f, FQSAT); }

// ltemq = (temg-163.15)/fqsat+1.5, clamped (:9927).
static inline int ltemq_of(F temperature)
{
    int l = (int)(__fdiv_rn(temperature - 163.15f, FQSAT) + 1.5f);
    return min(NQSAT, max(1, l));
}

// ltemq1 = (temp1+dtemp-163.15)*fqsati+1.5, clamped (:10772).
static inline int ltemq_fqsati(F temperature)
{
    int l = (int)((temperature - 163.15f) * fqsati() + 1.5f);
    return min(NQSAT, max(1, l));
}

// tabqvs / dtabqvs (module_mp_nssl_2mom.F:1466-1471), evaluated on demand.
static inline F temq_of(int l)
{
    return __fadd_rn(163.15f, __fmul_rn(F((l - 1)), FQSAT));
}

static inline F tabqvs(int l)
{
    const F temq = temq_of(l);
    return gfk_exp(__fdiv_rn(__fmul_rn(CAW, temq - 273.15f), temq - CBW));
}

static inline F dtabqvs(int l)
{
    const F temq = temq_of(l);
    const F t = temq - CBW;
    const F a = __fdiv_rn(__fmul_rn(-CAW, __fadd_rn(-273.15f, temq)), __fmul_rn(t, t));
    const F b = __fdiv_rn(CAW, t);
    return __fmul_rn(__fadd_rn(a, b), tabqvs(l));
}

// QVEXCESS (:6052-6199) for one gathered point.  qcw1 is the caller's
// qx(mgs,lc); only qvex leaves the routine.
static F qvexcess(F qwvp0, F qv0, F qcw1, F pres,
                          F thetap0, F theta0, F pi0, F fcqv1,
                          F felvcp, F ss1, F pk)
{
    const F pqs = __fdiv_rn(380.0f, pres);
    F thetap = thetap0;
    F theta = thetap + theta0;
    F qwvp = qwvp0;
    F qvap = nssl2_max(qwvp0 + qv0, 0.0f);
    F temg = theta * pk;
    F qwv = nssl2_max(qvap, 0.0f);
    F qcw = nssl2_max(qcw1, 0.0f);
    int ltemq = ltemq_of(temg);
    F qvs = pqs * tabqvs(ltemq);
    F qss = (0.01f * ss1 + 1.0f) * qvs;
    for (int itertd = 1; itertd <= 2; ++itertd) {
        F dqcw = 0.0f;
        F dqwv = qwv - qss;
        if (dqwv < 0.0f) {
            if (qcw > -dqwv) {
                dqcw = dqwv;
                dqwv = 0.0f;
            } else {
                dqcw = -qcw;
                dqwv = dqwv + qcw;
            }
            qwvp = qwvp - dqcw;
            qcw = qcw + dqcw;
            thetap = thetap + __fdiv_rn(1.0f, pi0) * (felvcp * dqcw);
        }
        if (dqwv >= 0.0f) {
            const F t = temg - CBW;
            const F dqvcnd = __fdiv_rn(
                dqwv, 1.0f + __fdiv_rn(fcqv1 * qss, t * t));
            dqcw = dqvcnd;
            thetap = thetap + __fdiv_rn(felvcp * dqcw, pi0);
            qwvp = qwvp - dqvcnd;
            qcw = qcw + dqcw;
        }
        theta = thetap + theta0;
        temg = theta * pk;
        qvap = nssl2_max(qwvp + qv0, 0.0f);
        ltemq = ltemq_of(temg);
        qvs = pqs * tabqvs(ltemq);
        qcw = nssl2_max(qcw, 0.0f);
        qwv = nssl2_max(qvap, 0.0f);
        qss = (0.01f * ss1 + 1.0f) * qvs;
    }
    return nssl2_max(qcw - qcw1, 0.0f);
}

// Restoration of evaporated droplets to CCN (:10536-10546, irenuc <= 2,
// lccna = 0, invertccn false).
static inline F restore_ccn(F ccnc, F rho0, F add)
{
    return nssl2_max(ccnc, nssl2_min(QCCN * rho0, ccnc + RESTORECCNFRAC * add));
}

}  // namespace

kernel void nssl2_nucond_stage_exact(
    constant Nssl2NucondStageExactResources &r [[buffer(0)]],
    constant Nssl2NucondStageExactParams &p [[buffer(1)]],
    uint thread_position_in_grid [[thread_position_in_grid]]) {

    const int idx = int(thread_position_in_grid);
    if (idx >= p.n) return;
    const int k0 = idx / p.horizontal_size;
    const int col = idx - k0 * p.horizontal_size;
    const int kgs = k0 + 1;                       // WRF's 1-based level

    const F dn = F(r.air_density[idx]);
    const F pn = F(r.pressure_pa[idx]);
    const F t77 = F(r.exner[idx]);
    const F in_scale = p.concentration_space ? 1.0f : dn;

    // an(...) for this point, NSSL internal units (#/m3 for numbers).
    F an_lt = F(r.full_theta[idx]);
    F an_lv = F(r.qv[idx]);
    F an_lc = F(r.qc[idx]);
    F an_lr = F(r.qr[idx]);
    F an_li = F(r.qi[idx]);
    F an_ls = F(r.qs[idx]);
    F an_lnc = F(r.qndrop[idx]) * in_scale;
    F an_lnr = F(r.qnr[idx]) * in_scale;
    // Ice and snow numbers enter only the tail's sign tests; they are not
    // rescaled, so a Registry-unit caller gets them back untouched.
    F an_lni = F(r.qni[idx]);
    F an_lns = F(r.qns[idx]);
    F an_lccn = F(r.qnn[idx]) * in_scale;

    // ---- gather test (:9922-9947) ----
    const F t0 = an_lt * t77;
    {
        const F pqs1 = __fdiv_rn(380.0f, pn + 0.0f);
        const F temg1 = t0;
        const int ltemq = ltemq_of(temg1);
        const F qvs1 = pqs1 * tabqvs(ltemq);
        const F qss1 = qvs1;
        const bool gathered =
            (temg1 > TFRH || __fdiv_rn(an_lv, qvs1) > MAXLOWTEMPSS)
            && (an_lv > qss1 || an_lc > QXMIN_C || an_lr > QXMIN_R);
        if (gathered) {
            // ---- gather (:9963-10058) ----
            const F qx_lv = an_lv;
            F qc = nssl2_max(an_lc, 0.0f);
            F qr = nssl2_max(an_lr, 0.0f);
            const F theta0 = an_lt;
            F thetap = 0.0f;
            F theta;
            const F qv0 = qx_lv;
            F qwvp = qx_lv - qv0;
            const F pres = pn + 0.0f;
            const F rho0 = dn;
            const F rhoinv = __fdiv_rn(1.0f, rho0);
            const F rhovt = nucond_sqrt(__fdiv_rn(RHO00, rho0));
            const F pi0 = t77 + 0.0f;
            F temg = t0;
            const F pk = t77 + 0.0f;
            const F pqs = __fdiv_rn(380.0f, pres);
            int ltemq = ltemq_of(temg);
            F qvs = pqs * tabqvs(ltemq);
            const F qvap = nssl2_max(qwvp + qv0, 0.0f);
            const F es = 6.1078e2f * tabqvs(ltemq);
            const F qss = qvs;
            F temgx = nssl2_min(temg, 313.15f);
            temgx = nssl2_max(temgx, 233.15f);
            const F felv = 2500837.367f * gfk_pow(
                __fdiv_rn(273.15f, temgx), 0.167f + 3.67e-4f * temgx);
            const F felvcp = felv * cpi();
            const F fcqv1 = (4098.0258f * felv) * cpi();
            const F wvdf = (2.11e-05f * gfk_pow(__fdiv_rn(temg, TFR), 1.94f))
                * __fdiv_rn(101325.0f, 0.0f + pn);
            const F advisc = (ADVISC0 * __fdiv_rn(416.16f, temg + 120.0f))
                * gfk_pow(__fdiv_rn(temg, 296.0f), 1.5f);
            const F tka = __fdiv_rn(TKA0 * advisc, ADVISC1);

            // numbers (:10065-10110); lccn > 1, lccna = lccnuf = 0.
            F cx_lc = nssl2_max(an_lnc, 0.0f);
            const F cwnccn = __fdiv_rn(CWCCN * rho0, RHO00);
            F ccnc = an_lccn;
            F ccna = cwnccn - ccnc;
            F cx_lr = nssl2_max(an_lnr, 0.0f);
            F cn = 0.0f;

            // cnuc (:10113-10129); renucfrac = 0, or 1 with nssl_ccn_on=0.
            const F renucfrac = p.predicted_ccn ? 0.0f : 1.0f;
            F cnuc = nssl2_max(ccnc, cwnccn) * (1.0f - renucfrac)
                + ccnc * renucfrac;
            if (renucfrac >= 0.999f) {
                if (temg < 265.0f) {
                    if (qc > 10.0f * QXMIN_C && F(r.w_interface[idx]) > 2.0f) {
                        cnuc = 0.0f;
                    } else {
                        cnuc = 0.1f * cnuc;
                    }
                }
            }

            // w and ssfilt neighbours (:10378-10397).
            const int kp1 = min(p.nz, kgs + 1);
            const F wvel = 0.5f * (F(r.w_interface[(kp1 - 1) * p.horizontal_size + col])
                                       + F(r.w_interface[idx]));
            const F ssat0 = F(r.initial_supersaturation_percent[idx]);
            const F ssf = F(r.initial_supersaturation_percent[idx]);
            const F ssfkp1 = F(r.initial_supersaturation_percent[(min(p.nz - 1, kgs + 1) - 1) * p.horizontal_size + col]);
            const F ssfkm1 = F(r.initial_supersaturation_percent[(max(1, kgs - 1) - 1) * p.horizontal_size + col]);

            // cloud water (:10407-10436).
            F xmas_lc;
            if (cx_lc > 1.0e6f) {
                xmas_lc = nssl2_min(nssl2_max(__fdiv_rn(qc * rho0, cx_lc),
                                              NSSL2_WRF_CWMASN), NSSL2_WRF_CWMASX);
            } else if (qc > QXMIN_C && cx_lc > CXMIN) {
                xmas_lc = nssl2_min(nssl2_max(__fdiv_rn(qc * rho0, cx_lc),
                                              XDN_C * xvmn_c()), XDN_C * xvmx_c());
                cx_lc = __fdiv_rn(qc * rho0, xmas_lc);
            } else if (qc > QXMIN_C && cx_lc <= CXMIN) {
                cx_lc = nssl2_max(CXMIN, __fdiv_rn(rho0 * qc, NSSL2_WRF_CWMASX));
                xmas_lc = nssl2_min(nssl2_max(__fdiv_rn(qc * rho0, cx_lc),
                                              NSSL2_WRF_CWMASN), NSSL2_WRF_CWMASX);
            } else {
                xmas_lc = NSSL2_WRF_CWMASN;
            }
            F xdia_lc = gfk_pow(xmas_lc * cwc1(), c1f3());

            // rain (:10440-10477); imurain = 1, alpha(lr) = alphar = 0.
            const F alpha_lr = 0.0f;
            F xdia_lr = 1.e-9f;
            if (qr > QXMIN_R) {
                F xv = __fdiv_rn(rho0 * qr, XDN_R * nssl2_max(cx_lr, 1.0e-9f));
                if (xv > NSSL2_WRF_XVRMX0) {
                    xv = NSSL2_WRF_XVRMX0;
                    cx_lr = __fdiv_rn(rho0 * qr, NSSL2_WRF_XVRMX0 * XDN_R);
                } else if (xv < NSSL2_WRF_XVRMN) {
                    xv = NSSL2_WRF_XVRMN;
                    cx_lr = __fdiv_rn(rho0 * qr, NSSL2_WRF_XVRMN * XDN_R);
                }
                xdia_lr = gfk_pow(__fdiv_rn((6.0f * PIINV) * xv,
                                            ((alpha_lr + 3.0f) * (alpha_lr + 2.0f))
                                                * (alpha_lr + 1.0f)),
                                  c1f3());
            }

            // (:10483-10498)
            const F fadvisc = (ADVISC0 * __fdiv_rn(416.16f, temg + 120.0f))
                * gfk_pow(__fdiv_rn(temg, 296.0f), 1.5f);
            const F fakvisc = fadvisc * rhoinv;
            const F fwvdf = (2.11e-05f * gfk_pow(__fdiv_rn(temg, TFR), 1.94f))
                * __fdiv_rn(101325.0f, pres);
            const F fschm = __fdiv_rn(fakvisc, fwvdf);
            const F fvent = gfk_pow(fschm, 1.0f / 3.0f) * gfk_pow(fakvisc, -0.5f);

            // ---- condensation / nucleation (:10507-11637) ----
            F dcloud = 0.0f;
            F dqc = 0.0f;
            bool cycle = (temg <= TFRH && __fdiv_rn(qx_lv, qvs) < MAXLOWTEMPSS);
            if (!cycle) {
                // label selector: 0 = fall to 631, 1 = renucleation block.
                bool renuc_block = false;
                if (ssat0 > 0.0f || ssf > 0.0f) {
                    // 620
                    if (qc > QXMIN_C && cx_lc >= 1.0f) {
                        const F ac1 = __fdiv_rn(felv * felv,
                                                    (tka * RW) * (temg * temg));
                        const F bc = __fdiv_rn(RW * temg, wvdf * es);
                        F d1 = 0.0f, d1r = 0.0f;
                        if (ssf > 0.0f || ssat0 > 0.0f) {
                            if (qc > QXMIN_C) {
                                if (xdia_lc <= 0.0f) {
                                    xmas_lc = NSSL2_WRF_CWMASN;
                                    xdia_lc = gfk_pow(xmas_lc * cwc1(), c1f3());
                                }
                                d1 = __fdiv_rn(1.0f, ac1 + bc) * 4.0f * PI * ventc()
                                    * 0.5f * xdia_lc * cx_lc * rhoinv;
                            } else {
                                d1 = 0.0f;
                            }
                            if (qr > QXMIN_R && cx_lr > 1.e-9f) {
                                const F x = 1.0f + alpha_lr;
                                const F rwvent = 0.78f * x
                                    + 0.308f * ventrn() * fvent * nucond_sqrt(AR * rhovt)
                                        * gfk_pow(xdia_lr, __fdiv_rn(1.0f + BR, 2.0f));
                                d1r = __fdiv_rn(1.0f, ac1 + bc) * 4.0f * PI * rwvent
                                    * 0.5f * xdia_lr * cx_lr * rhoinv;
                            } else {
                                d1r = 0.0f;
                            }
                            const F e1 = __fdiv_rn(felvcp, pi0);
                            const F f1 = pk;
                            F temp1 = temg;
                            const F p380 = __fdiv_rn(380.0f, pres);
                            F ss1 = __fdiv_rn(qx_lv, qvs);
                            F ss2 = ss1;
                            F temp2 = temp1;
                            F qv1 = qx_lv;
                            F qvs1 = qvs;
                            F delta;
                            if (nucond_abs(ss1 - 1.0f) > 1.e-5f) {
                                delta = __fdiv_rn(0.5f * (qv1 - qvs1), d1 * (ss1 - 1.0f));
                            } else {
                                delta = 0.1f * F(p.dt);
                            }
                            const F dtcon1 = nssl2_min(0.2f * delta, 0.05f);
                            int nc = 2 * (int)nucond_lround(__fdiv_rn(F(p.dt) - 4.0f * dtcon1, delta));
                            nc = max(nc, 5);
                            const F dtcon2 = __fdiv_rn(F(p.dt) - 4.0f * dtcon1, F(nc));
                            int nstep = 1;
                            F dt1 = 0.0f;
                            F dqr = 0.0f;
                            while (dt1 < F(p.dt)) {
                                F dtcon = (nstep <= 4) ? dtcon1 : dtcon2;
                                F ss1m;
                                bool exit_loop = false;
                                while (true) {
                                    // 609
                                    const F dqv = -(ss1 - 1.0f) * d1 * dtcon;
                                    const F dqvr = -(ss1 - 1.0f) * d1r * dtcon;
                                    const F dtemp = -0.5f * e1 * f1 * (dqv + dqvr);
                                    const int ltemq1m = ltemq_fqsati(temp1 + dtemp);
                                    const F dqvs = dtemp * p380 * dtabqvs(ltemq1m);
                                    const F qv1m = qv1 + dqv + dqvr;
                                    const F qvs1m = qvs1 + dqvs;
                                    ss1m = __fdiv_rn(qv1m, qvs1m);
                                    // (dqvii + dqvis) == 0 always: no ice here.
                                    if (ss1m < 1.0f) {
                                        dtcon = 0.5f * dtcon;
                                        if (dtcon >= dtcon1) continue;
                                        exit_loop = true;
                                    }
                                    break;
                                }
                                if (exit_loop) break;
                                const F dqv = -(ss1m - 1.0f) * d1 * dtcon;
                                const F dqvr = -(ss1m - 1.0f) * d1r * dtcon;
                                const F dtemp = -e1 * f1 * (dqv + dqvr);
                                const int ltemq1 = ltemq_fqsati(temp1 + dtemp);
                                const F dqvs = dtemp * p380 * dtabqvs(ltemq1);
                                qv1 = qv1 + dqv + dqvr;
                                dqc = dqc - dqv;
                                dqr = dqr - dqvr;
                                qvs1 = qvs1 + dqvs;
                                ss1 = __fdiv_rn(qv1, qvs1);
                                temp1 = temp1 + dtemp;
                                if (temp2 == temp1 || ss2 == ss1 || ss1 == 1.0f
                                        || (nstep > 10 && ss1 < 1.0005f)) {
                                    break;
                                }
                                ss2 = ss1;
                                temp2 = temp1;
                                dt1 = dt1 + dtcon;
                                nstep = nstep + 1;
                            }
                            dcloud = dqc;
                            thetap = thetap + e1 * (dcloud + dqr);
                            qwvp = qwvp - (dcloud + dqr);
                            qc = qc + dcloud;
                            qr = qr + dqr;
                            theta = thetap + theta0;
                            temg = theta * f1;
                            ltemq = ltemq_of(temg);
                            qvs = pqs * tabqvs(ltemq);
                        }
                        // :11023-11033
                        bool go631 = false, go613 = false, at616 = false;
                        if (wvel <= 0.0f) {
                            at616 = true;
                        } else if (cx_lc <= 0.0f) {
                            go613 = true;
                        } else if (kgs > 1 && qc <= QXMIN_C) {
                            go613 = true;
                        } else if (kgs == 1 && wvel > 0.0f) {
                            go613 = true;
                        } else {
                            at616 = true;
                        }
                        if (at616 && !go613) {
                            if (ssf <= SUPCB && wvel > 0.0f) {
                                go631 = true;
                            } else if (kgs > 1 && kgs < p.nz - 1
                                       && (ssfkp1 >= SUPMX || ssf >= SUPMX
                                           || ssfkm1 >= SUPMX)) {
                                go631 = true;
                            } else if (ssf < 1.e-10f || ssf >= SUPMX) {
                                go631 = true;
                            } else {
                                renuc_block = true;
                            }
                        }
                        (void)go631;
                    } else {
                        // qc .le. qxmin(lc) (:10891-11012)
                        if (ssf > 0.0f) {
                            const F ssmx = SSMXINIT;
                            if (ssf > ssmx && ssf < 20.0f && ccnc > 0.05f * cwnccn) {
                                dcloud = qvexcess(qwvp, qv0, qc, pres, thetap, theta0,
                                                  pi0, fcqv1, felvcp, ssmx, pk);
                            } else {
                                dcloud = 0.0f;
                            }
                        } else {
                            dcloud = 0.0f;
                        }
                        thetap = thetap + __fdiv_rn(felvcp * dcloud, pi0);
                        qwvp = qwvp - dcloud;
                        qc = qc + dcloud;
                        theta = thetap + theta0;
                        temg = theta * pk;
                        ltemq = ltemq_of(temg);
                        qvs = pqs * tabqvs(ltemq);
                        cn = 0.0f;
                        const F cnuctmp = cnuc;
                        if (dcloud > QXMIN_C && wvel > 0.0f) {
                            cn = ccne0() * gfk_pow(cnuctmp, __fdiv_rn(2.0f, 2.0f + CCK))
                                * gfk_pow(wvel, cnexp());
                            // iccwflg = 1
                            cn = nssl2_min(__fdiv_rn(CWCCN * rho0, RHO00),
                                           nssl2_max(cn, __fdiv_rn(rho0 * qc,
                                               XDN_C * four_pi_third() * four_micron_cube())));
                        } else {
                            cn = 0.0f;
                            dcloud = 0.0f;
                        }
                        if (cn > 0.0f) {
                            if (cn > ccnc) cn = ccnc;
                            ccnc = nssl2_max(ccnc - cn, 0.0f);
                            ccna = ccna + cn;
                        }
                        if (cn > cx_lc) cx_lc = cn;
                        if (cx_lc > 0.0f && qc <= QXMIN_C) {
                            cx_lc = 0.0f;
                        } else {
                            cx_lc = nssl2_min(cx_lc, __fdiv_rn(rho0 * nssl2_max(qc, 0.0f),
                                                               NSSL2_WRF_CWMASN));
                        }
                    }
                } else if (qc > 0.0f) {
                    // cloud evaporation (:10524-10591)
                    const F tc = temg - CBW;
                    const F r1 = __fdiv_rn(1.0f, 1.0f + __fdiv_rn(
                        CAW * (273.15f - CBW) * qss * felv, NCP * (tc * tc)));
                    const F qevap = nssl2_min(qc, r1 * (qss - qvap));
                    if (qc <= qevap) {
                        qwvp = qwvp + qc;
                        thetap = thetap - __fdiv_rn(felvcp * qc, pi0);
                        qc = 0.0f;
                        ccnc = restore_ccn(ccnc, rho0, cx_lc);
                        cx_lc = 0.0f;
                    } else {
                        const F qctmp = qc;
                        qwvp = qwvp + qevap;
                        qc = qc - qevap;
                        if (qc <= 0.0f) {
                            ccnc = restore_ccn(ccnc, rho0, cx_lc);
                            cx_lc = 0.0f;
                        } else {
                            const F tmp = __fdiv_rn(0.9f * qevap * cx_lc, qctmp);
                            ccnc = restore_ccn(ccnc, rho0, tmp);
                            cx_lc = cx_lc - tmp;
                        }
                        thetap = thetap - __fdiv_rn(felvcp * qevap, pi0);
                    }
                }

                if (renuc_block) {
                    // irenuc = 2 (:11111-11149), then :11508-11513.
                    cn = ccne0() * gfk_pow(cnuc, __fdiv_rn(2.0f, 2.0f + CCK))
                        * gfk_pow(nssl2_max(wvel, 0.0f), cnexp());
                    cn = nssl2_min(cn, ccnc);
                    cn = nssl2_min(cn, __fdiv_rn(0.5f * dqc, NSSL2_WRF_CWMASN));
                    cn = nssl2_min(cn, nssl2_max(cnuc - ccna, 0.0f));
                    cx_lc = cx_lc + cn;
                    ccnc = nssl2_max(ccnc - cn, 0.0f);
                    ccna = ccna + cn;
                    if (cx_lc > 0.0f && qc <= QXMIN_C) cx_lc = 0.0f;
                }

                // 613 / 631: maximum supersaturation (:11523-11577).
                {
                    F ssmx = MAXSUPERSAT;
                    const F qv1 = qv0 + qwvp;
                    const F qvs1 = qvs;
                    if (qv1 > ssmx * qvs1) {
                        ssmx = 100.0f * (ssmx - 1.0f);
                        const F qvex = qvexcess(qwvp, qv0, qc, pres, thetap, theta0,
                                                    pi0, fcqv1, felvcp, ssmx, pk);
                        if (qvex > 0.0f) {
                            thetap = thetap + __fdiv_rn(felvcp * qvex, pi0);
                            qwvp = qwvp - qvex;
                            qc = qc + qvex;
                            cn = nssl2_min(nssl2_max(ccnc, cwnccn),
                                           __fdiv_rn(rho0 * qvex,
                                                     nssl2_max(nssl2_max(xmas_lc, cwmas20()),
                                                               cwmasn5())));
                            ccnc = nssl2_max(ccnc - cn, 0.0f);
                            cx_lc = cx_lc + cn;
                        }
                    }
                }
                // :11587-11600
                if (cx_lc > CXMIN && qc > QXMIN_C) {
                    xmas_lc = __fdiv_rn(rho0 * qc, cx_lc);
                    if (xmas_lc < NSSL2_WRF_CWMASN || xmas_lc > NSSL2_WRF_CWMASX) {
                        xmas_lc = nssl2_min(xmas_lc, NSSL2_WRF_CWMASX);
                        xmas_lc = nssl2_max(xmas_lc, NSSL2_WRF_CWMASN);
                        cx_lc = __fdiv_rn(rho0 * qc, xmas_lc);
                    }
                }
            }

            // ---- scatter (:11649-11695) ----
            an_lt = theta0 + thetap;
            an_lv = qv0 + qwvp;
            an_lc = qc + nssl2_min(an_lc, 0.0f);
            an_lr = qr + nssl2_min(an_lr, 0.0f);
            an_lnc = nssl2_max(cx_lc, 0.0f);
            an_lccn = nssl2_max(ccnc, 0.0f);
            an_lnr = nssl2_max(cx_lr, 0.0f);
        }
    }

    // ---- moment tail (:11732-12206), every point; frac = 1 ----
    const bool zero_lc = an_lnc <= 0.0f;
    const bool zero_lr = an_lnr <= 0.0f;
    const bool zero_li = an_lni <= 0.0f;
    const bool zero_ls = an_lns <= 0.0f;
    if (p.has_frozen) {
        F an_lh = F(r.qg[idx]);
        F an_lhl = F(r.qh[idx]);
        // The frozen arm requires concentration space (launcher-enforced).
        F an_lnh = F(r.qng[idx]);
        F an_lnhl = F(r.qnh[idx]);
        F an_lvh = F(r.qvolg[idx]);
        F an_lvhl = F(r.qvolh[idx]);
        const bool zero_lh = an_lnh <= 0.0f;
        const bool zero_lhl = an_lnhl <= 0.0f;
        // hail (:11799-11892): lhlw = 0, alphahl = 1 > 0.1.
        if (an_lhl < QXMIN_HL || zero_lhl) {
            an_lv = an_lv + an_lhl;
            an_lhl = 0.0f;
            an_lnhl = 0.0f;
            an_lvhl = 0.0f;
        } else {
            F tmp;
            if (an_lvhl > 0.0f) {
                tmp = __fdiv_rn(dn * an_lhl, an_lvhl);
            } else {
                tmp = 900.0f;                     // rho_qhl
                an_lvhl = __fdiv_rn(dn * an_lhl, tmp);
            }
            if (tmp < 500.0f) {                   // xdnmn(lhl)
                tmp = nssl2_max(500.0f, tmp);
                an_lvhl = __fdiv_rn(dn * an_lhl, tmp);
            }
            if (tmp > 900.0f) {                   // xdnmx(lhl)
                tmp = nssl2_min(900.0f, tmp);
                an_lvhl = __fdiv_rn(dn * an_lhl, tmp);
            }
        }
        // graupel (:11935-12033): lhw = 0; alphah = 0 arms the intercept.
        if (an_lh < QXMIN_H || zero_lh) {
            an_lv = an_lv + an_lh;
            an_lh = 0.0f;
            an_lnh = 0.0f;
            an_lvh = 0.0f;
        } else {
            F tmp;
            if (an_lvh > 0.0f) {
                tmp = __fdiv_rn(dn * an_lh, an_lvh);
            } else {
                tmp = 500.0f;                     // rho_qh
                an_lvh = __fdiv_rn(dn * an_lh, tmp);
            }
            if (tmp < 170.0f) {                   // xdnmn(lh)
                tmp = nssl2_max(170.0f, tmp);
                an_lvh = __fdiv_rn(dn * an_lh, tmp);
            }
            if (tmp > 900.0f) {                   // xdnmx(lh)
                tmp = nssl2_min(900.0f, tmp);
                an_lvh = __fdiv_rn(dn * an_lh, tmp);
            }
            if (an_lh > QXMIN_H) {
                F hwdn;
                if (an_lvh > 0.0f) {
                    hwdn = __fdiv_rn(dn * an_lh, an_lvh);
                } else {
                    hwdn = 500.0f;                // xdn0(lh)
                }
                hwdn = nssl2_max(170.0f, hwdn);
                tmp = __fdiv_rn(hwdn * an_lnh, dn * an_lh);
                const F tmpg = an_lnh * gfk_pow(tmp * 3.14159f, 1.0f / 3.0f);
                if (tmpg < CNOHMN) {
                    tmp = gfk_pow(__fdiv_rn(hwdn, dn * an_lh) * 3.14159f, 1.0f / 3.0f);
                    an_lnh = gfk_pow(__fdiv_rn(CNOHMN, tmp), 0.75f);
                }
            }
        }
        r.qg[idx] = float(an_lh);
        r.qh[idx] = float(an_lhl);
        r.qng[idx] = float(an_lnh);
        r.qnh[idx] = float(an_lnhl);
        r.qvolg[idx] = float(an_lvh);
        r.qvolh[idx] = float(an_lvhl);
    }
    // snow (:12036-12092): lvs = lsw = 0, both temperature arms alike.
    if (an_ls < QXMIN_S || zero_ls) {
        an_lv = an_lv + an_ls;
        an_ls = 0.0f;
        an_lns = 0.0f;
    }
    // rain (:12098-12111)
    if (an_lr < QXMIN_R || zero_lr) {
        an_lv = an_lv + an_lr;
        an_lr = 0.0f;
        an_lnr = 0.0f;
    }
    // cloud ice (:12116-12123)
    if (an_li <= QXMIN_I || zero_li) {
        an_lv = an_lv + an_li;
        an_li = 0.0f;
        an_lni = 0.0f;
    }
    // cloud water (:12158-12200)
    if (an_lc <= QXMIN_C || zero_lc) {
        an_lv = an_lv + an_lc;
        an_lc = 0.0f;
        an_lccn = an_lccn + nssl2_max(an_lnc, 0.0f);
        an_lnc = 0.0f;
        an_lccn = nssl2_max(an_lccn, 0.0f);
        const F tmp = an_li + an_ls;
        if (an_lccn > 1.0f && tmp < QXMIN_I) {
            an_lccn = dn * QCCN - nssl2_max(dn * QCCN - an_lccn, 0.0f)
                * gfk_exp(-__fdiv_rn(F(p.dt), CCNTIMECONST));
        }
    }

    const F out_scale = p.concentration_space ? 1.0f : __fdiv_rn(1.0f, dn);
    r.full_theta[idx] = float(an_lt);
    r.qv[idx] = float(an_lv);
    r.qc[idx] = float(an_lc);
    r.qr[idx] = float(an_lr);
    r.qi[idx] = float(an_li);
    r.qs[idx] = float(an_ls);
    r.qndrop[idx] = float(an_lnc * out_scale);
    r.qnr[idx] = float(an_lnr * out_scale);
    r.qni[idx] = float(an_lni);
    r.qns[idx] = float(an_lns);
    r.qnn[idx] = float(an_lccn * out_scale);

}

#undef NSSL2_WRF_CWMASN
#undef NSSL2_WRF_CWMASX
#undef NSSL2_WRF_XVRMN
#undef NSSL2_WRF_XVRMX0
