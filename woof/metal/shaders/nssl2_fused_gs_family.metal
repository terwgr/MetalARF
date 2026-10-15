// WRF v4.6.1 NSSL option-18 fused GS, lowered from the admitted CUDA source.
#include <metal_stdlib>
#include "nssl2_fused_gs_family_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float nssl_add(float a, float b) { return a + b; }
static inline float nssl_div(float a, float b) { return a / b; }
static inline float nssl_mul(float a, float b) { return a * b; }


struct Nssl64 {
    ulong bits;
    Nssl64() : bits(0ul) {}
    Nssl64(float value) : bits(soft64_from_f32_bits(as_type<uint>(value))) {}
    operator float() const { return as_type<float>(soft64_to_f32_bits(bits)); }
    operator int() const { return soft64_to_i32_rz(bits).value; }
};
static inline Nssl64 nssl64_bits(ulong b) { Nssl64 x; x.bits = b; return x; }
static inline Nssl64 operator+(Nssl64 a, Nssl64 b) { Nssl64 x; x.bits=soft64_add(a.bits,b.bits); return x; }
static inline Nssl64 operator+(Nssl64 a, float b) { return a + Nssl64(b); }
static inline Nssl64 operator+(float a, Nssl64 b) { return Nssl64(a) + b; }
static inline Nssl64 operator-(Nssl64 a, Nssl64 b) { Nssl64 x; x.bits=soft64_sub(a.bits,b.bits); return x; }
static inline Nssl64 operator-(Nssl64 a, float b) { return a - Nssl64(b); }
static inline Nssl64 operator-(float a, Nssl64 b) { return Nssl64(a) - b; }
static inline Nssl64 operator-(Nssl64 a) { Nssl64 x; x.bits=soft64_negate_bits(a.bits); return x; }
static inline Nssl64 operator*(Nssl64 a, Nssl64 b) { Nssl64 x; x.bits=soft64_mul(a.bits,b.bits); return x; }
static inline Nssl64 operator*(Nssl64 a, float b) { return a * Nssl64(b); }
static inline Nssl64 operator*(float a, Nssl64 b) { return Nssl64(a) * b; }
static inline Nssl64 operator/(Nssl64 a, Nssl64 b) { Nssl64 x; x.bits=soft64_div(a.bits,b.bits); return x; }
static inline Nssl64 operator/(Nssl64 a, float b) { return a / Nssl64(b); }
static inline Nssl64 operator/(float a, Nssl64 b) { return Nssl64(a) / b; }
static inline thread Nssl64 &operator+=(thread Nssl64 &a, Nssl64 b) { a.bits=soft64_add(a.bits,b.bits); return a; }
static inline thread Nssl64 &operator-=(thread Nssl64 &a, Nssl64 b) { a.bits=soft64_sub(a.bits,b.bits); return a; }
static inline bool operator<(Nssl64 a, Nssl64 b) { return soft64_less(a.bits,b.bits); }
static inline bool operator>(Nssl64 a, Nssl64 b) { return soft64_greater(a.bits,b.bits); }
static inline bool operator<=(Nssl64 a, Nssl64 b) { return soft64_less_equal(a.bits,b.bits); }
static inline bool operator>=(Nssl64 a, Nssl64 b) { return soft64_greater_equal(a.bits,b.bits); }
static inline bool operator<(Nssl64 a, float b) { return a < Nssl64(b); }
static inline bool operator>(Nssl64 a, float b) { return a > Nssl64(b); }
static inline bool operator<=(Nssl64 a, float b) { return a <= Nssl64(b); }
static inline bool operator>=(Nssl64 a, float b) { return a >= Nssl64(b); }
static inline bool operator<(float a, Nssl64 b) { return Nssl64(a) < b; }
static inline bool operator>(float a, Nssl64 b) { return Nssl64(a) > b; }
static inline bool operator<=(float a, Nssl64 b) { return Nssl64(a) <= b; }
static inline bool operator>=(float a, Nssl64 b) { return Nssl64(a) >= b; }
static inline Nssl64 fmax(Nssl64 a, Nssl64 b) { return a < b ? b : a; }
static inline Nssl64 fmin(Nssl64 a, Nssl64 b) { return a > b ? b : a; }
static inline Nssl64 fmax(float a, Nssl64 b) { return fmax(Nssl64(a), b); }
static inline Nssl64 fmax(Nssl64 a, float b) { return fmax(a, Nssl64(b)); }
static inline Nssl64 fmin(float a, Nssl64 b) { return fmin(Nssl64(a), b); }
static inline Nssl64 fmin(Nssl64 a, float b) { return fmin(a, Nssl64(b)); }
static inline Nssl64 log(Nssl64 a) { Nssl64 x; x.bits=soft64_plm_log(a.bits); return x; }
static inline Nssl64 exp(Nssl64 a) { Nssl64 x; x.bits=soft64_plm_exp(a.bits); return x; }
static inline Nssl64 pow(Nssl64 a, Nssl64 b) { Nssl64 x; x.bits=soft64_plm_pow(a.bits,b.bits); return x; }
static inline Nssl64 pow(Nssl64 a, float b) { return pow(a,Nssl64(b)); }
static inline Nssl64 pow(float a, Nssl64 b) { return pow(Nssl64(a),b); }

namespace {

constant constexpr int QV = 0;
constant constexpr int QC = 1;
constant constexpr int QR = 2;
constant constexpr int QI = 3;
constant constexpr int QS = 4;
constant constexpr int QG = 5;
constant constexpr int QH = 6;
constant constexpr int NC = 7;
constant constexpr int NR = 8;
constant constexpr int NI = 9;
constant constexpr int NS = 10;
constant constexpr int NG = 11;
constant constexpr int NH = 12;
constant constexpr int NN = 13;
constant constexpr int VG = 14;
constant constexpr int VH = 15;

constant constexpr float PI = 3.14159265358979323846f;
constant constexpr float CXMIN = 1.0e-8f;
constant constexpr float QC_MIN = 1.0e-13f;
constant constexpr float QR_MIN = 1.0e-12f;
constant constexpr float QI_MIN = 1.0e-13f;
constant constexpr float QS_MIN = 1.0e-13f;
constant constexpr float QD_MIN = 1.0e-12f;

constant constexpr float CLOUD_VMIN =
    0.523599f * (4.0e-6f * 4.0e-6f * 4.0e-6f);
constant constexpr float CLOUD_VMAX =
    0.523599f * (120.0e-6f * 120.0e-6f * 120.0e-6f);
constant constexpr float RAIN_VMIN =
    0.523599f * (80.0e-6f * 80.0e-6f * 80.0e-6f);
constant constexpr float RAIN_VMAX =
    (0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f)) / (64.0f / 6.0f);
constant constexpr float ICE_MMIN = 6.88e-13f;
constant constexpr float ICE_MMAX = 1.0e-8f;
// The generic post-GS two-moment limiter uses setpar-derived ice volume
// bounds (10 um--2 mm diameter), which intentionally differ from the
// narrower initial setvtz crystal-mass bounds above.
constant constexpr float ICE_FINAL_MMIN =
    900.0f * 0.523599f * (10.0e-6f * 10.0e-6f * 10.0e-6f);
constant constexpr float ICE_FINAL_MMAX =
    900.0f * 0.523599f * (2.0e-3f * 2.0e-3f * 2.0e-3f);
constant constexpr float SNOW_VMIN =
    0.523599f * (0.01e-3f * 0.01e-3f * 0.01e-3f);
constant constexpr float SNOW_VMAX =
    0.523599f * (10.0e-3f * 10.0e-3f * 10.0e-3f);
constant constexpr float GRAUPEL_VMIN =
    0.523599f * (0.30e-3f * 0.30e-3f * 0.30e-3f);
constant constexpr float GRAUPEL_INITIAL_VMAX =
    0.523599f * (20.0e-3f * 20.0e-3f * 20.0e-3f);
constant constexpr float GRAUPEL_VMAX =
    (0.523599f * (20.0e-3f * 20.0e-3f * 20.0e-3f)) / (64.0f / 6.0f);
constant constexpr float HAIL_VMIN =
    0.523599f * (0.30e-3f * 0.30e-3f * 0.30e-3f);
constant constexpr float HAIL_INITIAL_VMAX =
    0.523599f * (40.0e-3f * 40.0e-3f * 40.0e-3f);
constant constexpr float HAIL_VMAX =
    (0.523599f * (40.0e-3f * 40.0e-3f * 40.0e-3f)) / (125.0f / 24.0f);

struct State {
    float qv, qc, qr, qi, qs, qg, qh;
    float nc, nr, ni, ns, ng, nh, nn;
    float vg, vh;
    float theta, temperature;
};

// Named WRF rates.  Keeping contributors separate is required: the cloud,
// rain, snow, and hail limiters rescale selected contributors and downstream
// vapor/heat/volume/number sums must then be rebuilt from the scaled values.
struct Rates {
    // Warm-rain mass and number.
    float qrcnw, crcnw, cautn;
    float qracw, cracw;
    float qrcev, crcev;
    float cracr;
    float qwcnr, qwshw, cwshw;

    // Cold-process members are populated by the source-ordered diagnosis
    // sections below this warm foundation.  Their explicit names prevent an
    // aggregate-only implementation from bypassing the WRF resums.
    float qiacw, ciacw;
    float qsacw, csacw;
    float qhacw, chacw, vhacw, qhacwmlr;
    float graupel_cloud_rime_density;
    float qhlacw, chlacw, vhlacw, qhlacwmlr;
    float qwfrz, cwfrz, qwfrzp, cwfrzp, qwfrzc, cwfrzc;
    float qwctfz, cwctfzp, qwctfzc, cwctfzc;
    float qiihr;
    float qiacr, ciacr, qiacrf, ciacrf, qiacrs, ciacrs, viacrf;
    float qrfrz, crfrz, qrfrzf, crfrzf, qrfrzs, crfrzs, vrfrzf;
    float qsacr, csacr;
    float qhacr, chacr, vhacr, qhacrmlr;
    float qhlacr, chlacr, vhlacr, qhlacrmlr;
    float qidpv, cidpv, qisbv, cisbv, qiint, ciint;
    float qsdpv, csdpv, qssbv, cssbv, qscev, cscev;
    float qhdpv, chdpv, qhsbv, chsbv, qhcev, chcev;
    float qhldpv, chldpv, qhlsbv, chlsbv, qhlcev, chlcev;
    float qscni, cscni, qscnvi, cscnvi;
    float qsaci, csaci, qraci, craci, qracif, csacs;
    float qhaci, chaci, qhaci0, chaci0;
    float qhlaci, chlaci, qhlaci0, chlaci0;
    float qhacs, chacs, qhacs0, chacs0;
    float qhlacs, chlacs, qhlacs0, chlacs0;
    float qhcni, chcni, chcnih, vhcni;
    float qhcns, chcns, chcnsh, vhcns;
    float qscnh, cscnh, vscnh;
    float qhlcnh, chlcnh, chlcnhhl, vhlcnh, vhlcnhl;
    float qhcnhl, chcnhl, vhcnhl;
    float qimlr, cimlr, qsmlr, csmlr, csmlrr;
    float qhmlr, chmlr, chmlrr, vhmlr;
    float qhlmlr, chlmlr, chlmlrr, vhlmlr;
    float qsshr, csshr, qhshr, chshr, chshrr;
    float qhlshr, chlshr, chlshrr, qrshr, crshr;
    float vhshdr, vhlshdr, vhsoak, vhlsoak;
    float wetgrowth_g, wetgrowth_h, wetsurface_g, wetsurface_h;
    float qsmul, csmul, qhmul1, chmul1, qhlmul1, chlmul1;
    float qsplinter, csplinter, qsplinter2, csplinter2;
};

struct Aggregates {
    float pqv_i, pqv_d;
    float pqc_i, pqc_d;
    float pqr_i, pqr_d;
    float pqi_i, pqi_d;
    float pqs_i, pqs_d;
    float pqg_i, pqg_d;
    float pqh_i, pqh_d;
    float pnc_i, pnc_d;
    float pnr_i, pnr_d;
    float pni_i, pni_d;
    float pns_i, pns_d;
    float png_i, png_d;
    float pnh_i, pnh_d;
    float pvg_i, pvg_d;
    float pvh_i, pvh_d;
};

struct ParticleProperties {
    float snow_density;
    float graupel_density;
    float hail_density;
};

static inline float latent_vapor(float temperature) {
    const float bounded = fmin(313.15f, fmax(233.15f, temperature));
    return 2500837.367f * pow(
        273.15f / bounded, 0.167f + 3.67e-4f * bounded);
}

static inline float latent_fusion(float temperature) {
    const float bounded = fmin(273.15f, fmax(223.15f, temperature));
    const float celsius = bounded - 273.15f;
    return 333690.6098f + 2030.61425f * celsius
        - 10.46708312f * celsius * celsius;
}

static inline float ice_saturation_mixing_ratio(
    float temperature, float pressure)
{
    int table_index = (int)(nssl_div((temperature - 163.15f), 0.002f) + 1.5f);
    table_index = min(1000001, max(1, table_index));
    // WRF fills TABQVS with separate single-precision multiply and add
    // operations.  NVRTC otherwise contracts this expression to an FMA,
    // moving some table temperatures by one ULP and amplifying the error in
    // the primary-ice target near saturation.
    const float table_temperature = nssl_add(
        163.15f, nssl_mul((float)(table_index - 1), 0.002f));
    return (380.0f / pressure) * exp(
        21.87455f * (table_temperature - 273.15f)
        / (table_temperature - 7.66f));
}

static inline float liquid_saturation_mixing_ratio(
    float temperature, float pressure)
{
    int table_index = (int)(nssl_div((temperature - 163.15f), 0.002f) + 1.5f);
    table_index = min(1000001, max(1, table_index));
    const float table_temperature = nssl_add(
        163.15f, nssl_mul((float)(table_index - 1), 0.002f));
    return (380.0f / pressure) * exp(
        17.2693882f * (table_temperature - 273.15f)
        / (table_temperature - 35.86f));
}

static inline float primary_ice_target(
    float vapor, float temperature, float rho, float pressure)
{
    // WRF 2834-2856, default icenucopt=1.  t7 is diagnosed by the outer
    // scheme from the unchanged column before GS and is not a Registry field.
    const float water_saturation =
        liquid_saturation_mixing_ratio(temperature, pressure);
    const float ice_saturation =
        ice_saturation_mixing_ratio(temperature, pressure);
    const float ice_relative = fmin(water_saturation, fmax(vapor, 0.0f))
        / ice_saturation;
    if (!(ice_relative > 1.0f) || !(temperature <= 268.15f)) return 0.0f;
    return (nssl_div(rho, 1.225f)) * 1.0e3f * exp(fmin(
        57.0f, 12.96f * (ice_relative - 1.0f) - 0.639f));
}

static inline void dense_fall_coefficients(
    float density, thread float* coefficient, thread float* exponent)
{
    const float density_table[9] = {
        50.0f, 150.0f, 250.0f, 350.0f, 450.0f,
        550.0f, 650.0f, 750.0f, 850.0f};
    const float coefficient_table[9] = {
        62.923f, 94.122f, 114.74f, 131.21f, 145.26f,
        157.71f, 168.98f, 179.36f, 189.02f};
    const float exponent_table[9] = {
        0.67819f, 0.63789f, 0.62197f, 0.61240f, 0.60572f,
        0.60066f, 0.59663f, 0.59330f, 0.59048f};
    int table = (int)(nssl_div((density - 50.0f), 100.0f)) + 1;
    table = min(9, max(1, table)) - 1;
    const float fraction = fmax(
        0.0f, 0.01f * (density - density_table[table]));
    *coefficient = coefficient_table[table];
    *exponent = exponent_table[table];
    if (table < 8) {
        *coefficient += fraction
            * (coefficient_table[table + 1] - *coefficient);
        *exponent += fraction
            * (exponent_table[table + 1] - *exponent);
    }
}

// WRF initializes gmoi on a 0.01 grid with its Nssl64-precision Lanczos
// GAMMA_DP routine, then linearly interpolates the table into single-precision
// temporaries.  Dense-particle fall speeds use that interpolated value rather
// than the platform tgammaf result.  The distinction is small (about 5--32
// ppm for the admitted exponents) but is amplified by cloud-number collection.
static inline Nssl64 wrf_gamma_dp(Nssl64 x)
{
    Nssl64 y = x;
    const Nssl64 tmp0 = x + 5.5;
    const Nssl64 tmp = (x + 0.5) * log(tmp0) - tmp0;
    Nssl64 series = nssl64_bits(0x3ff00000000d0ec7ul);
    y += 1.0;
    series += nssl64_bits(0x40530b869f76170aul) / y;
    y += 1.0;
    series += nssl64_bits(0xc055a0572b143656ul) / y;
    y += 1.0;
    series += nssl64_bits(0x4038039bf13b4cc6ul) / y;
    y += 1.0;
    series += nssl64_bits(0xbff3b5348dcdcc8cul) / y;
    y += 1.0;
    series += nssl64_bits(0x3f53cd7319ec12d8ul) / y;
    y += 1.0;
    series += nssl64_bits(0xbed6a1181c0e9888ul) / y;
    return exp(tmp + log(nssl64_bits(0x40040d931ff62706ul) * series / x));
}

static inline float nssl_tgamma(float x) { return (float)wrf_gamma_dp(Nssl64(x)); }

static inline float wrf_gamma_lookup(float argument)
{
    const int index = (int)(100.0 * (Nssl64)argument);
    const float delta = (float)(
        (Nssl64)argument - nssl64_bits(0x3f847ae147ae147bul) * (Nssl64)index);
    const Nssl64 lower = wrf_gamma_dp(nssl64_bits(0x3f847ae147ae147bul) * (Nssl64)index);
    const Nssl64 upper = wrf_gamma_dp(nssl64_bits(0x3f847ae147ae147bul) * (Nssl64)(index + 1));
    return (float)(lower + (upper - lower) * (Nssl64)delta * 100.0);
}

static inline float rain_tail_number_node(int bin) {
    return (float)exp(-0.25 * (Nssl64)bin);
}

static inline float rain_tail_mass_node(int bin) {
    const Nssl64 x = 0.25 * (Nssl64)bin;
    const Nssl64 x2 = x * x;
    return (float)(exp(-x) * (1.0 + x + 0.5 * x2 + x2 * x / 6.0));
}

static inline void rain_tail_fractions(
    float ratio, thread float* number_fraction, thread float* mass_fraction)
{
    ratio = fmin(100.0f, fmax(0.0f, ratio));
    const int bin = min(400, (int)(ratio * 4.0f));
    const int next_bin = min(400, bin + 1);
    const float weight = (ratio - (float)bin * 0.25f) * 4.0f;
    const float number_low = rain_tail_number_node(bin);
    const float mass_low = rain_tail_mass_node(bin);
    *number_fraction = number_low + weight
        * (rain_tail_number_node(next_bin) - number_low);
    *mass_fraction = mass_low + weight
        * (rain_tail_mass_node(next_bin) - mass_low);
}

static inline void bound_liquid(
    float q, float rho, float vmin, float vmax, thread float* number)
{
    if (q <= 0.0f) {
        *number = 0.0f;
    } else if (*number > CXMIN) {
        float volume = rho * q / (1000.0f * *number);
        if (volume < vmin || volume > vmax) {
            volume = fmin(vmax, fmax(vmin, volume));
            *number = rho * q / (1000.0f * volume);
        }
    }
}

static inline void bound_ice(thread State* s, float rho) {
    if (s->qi <= 0.0f) {
        s->ni = 0.0f;
    } else if (s->ni > CXMIN) {
        const float mass = rho * s->qi / s->ni;
        if (mass < ICE_FINAL_MMIN || mass > ICE_FINAL_MMAX) {
            s->ni = rho * s->qi /
                fmin(ICE_FINAL_MMAX, fmax(ICE_FINAL_MMIN, mass));
        }
    }
}

static inline void bound_snow(
    thread State* s, float rho, float density)
{
    if (s->qs <= 0.0f) {
        s->ns = 0.0f;
    } else if (s->ns > CXMIN) {
        float volume = rho * s->qs / (density * s->ns);
        const float maximum = SNOW_VMAX * fmax(
            1.0f, 100.0f / fmin(100.0f, density));
        if (volume < SNOW_VMIN || volume > maximum) {
            volume = fmin(maximum, fmax(SNOW_VMIN, volume));
            s->ns = rho * s->qs / (density * volume);
        }
    }
}

static inline void bound_dense_number(
    float q, thread float* number, float rho, float density,
    float vmin, float vmax)
{
    if (q <= 0.0f) {
        *number = 0.0f;
        return;
    }
    if (*number > CXMIN) {
        float mean_volume = rho * q / (density * *number);
        if (mean_volume < vmin || mean_volume > vmax) {
            mean_volume = fmin(vmax, fmax(vmin, mean_volume));
            *number = rho * q / (density * mean_volume);
        }
    }
}

static inline ParticleProperties normalize_state(
    thread State* s, float rho)
{
    // The admitted option-18 configuration overrides the module's raw
    // rho_qhl=800 declaration with nssl_params(9)=900 at initialization.
    ParticleProperties p{100.0f, 500.0f, 900.0f};

    // WRF 13766-13927 gather/cleanup precedes setvtz.  The exact-zero test is
    // intentionally distinct from the later CXMIN gates: a tiny active
    // rain/snow/graupel/hail mass with a missing number is made available to
    // diagnostic vapor, but that transfer is not part of final qv scatter.
    if (s->qr > QR_MIN) {
        if (s->nr == 0.0f && s->qr < 3.0f * QR_MIN) {
            s->qv += s->qr;
            s->qr = 0.0f;
        } else {
            s->nr = fmax(1.0e-9f, s->nr);
        }
    }
    if (s->qs > QS_MIN) {
        if (s->ns == 0.0f && s->qs < 3.0f * QS_MIN) {
            s->qv += s->qs;
            s->qs = 0.0f;
        } else {
            s->ns = fmax(1.0e-9f, s->ns);
        }
    }
    if (s->qg > QD_MIN) {
        if (s->ng == 0.0f && s->qg < 3.0f * QD_MIN) {
            s->qv += s->qg;
            s->qg = 0.0f;
        } else {
            s->ng = fmax(1.0e-9f, s->ng);
        }
    }
    if (s->qh > QD_MIN) {
        if (s->nh == 0.0f && s->qh < 3.0f * QD_MIN) {
            s->qv += s->qh;
            s->qh = 0.0f;
        } else {
            s->nh = fmax(1.0e-9f, s->nh);
        }
    }

    // Initial setvtz differs from the post-GS moment limiter.  Cloud retains
    // an existing number while clamping only its diagnostic mass/volume.
    if (s->qc > QC_MIN) {
        if (!(s->nc > CXMIN)) {
            s->nc = fmax(
                CXMIN, nssl_div(rho * s->qc, (1000.0f * CLOUD_VMAX)));
        }
        // WRF setvtz (ipconc>=2) clamps the diagnostic cloud mass/volume
        // for an existing number moment but deliberately retains cx itself.
        // diagnose_warm repeats that diagnostic clamp locally; only the
        // post-GS limiter is allowed to rewrite an existing cloud number.
    } else {
        // GS gather clears cloud number whenever qc is not active.
        s->nc = 0.0f;
    }
    if (s->qr > QR_MIN) {
        float volume = rho * s->qr / (1000.0f * fmax(1.0e-11f, s->nr));
        if (volume < RAIN_VMIN || volume > RAIN_VMAX) {
            volume = fmin(RAIN_VMAX, fmax(RAIN_VMIN, volume));
            s->nr = rho * s->qr / (1000.0f * volume);
        }
        // Inactive rain retains the nonnegative number loaded by GS gather.
    }
    if (s->qi > QI_MIN) {
        s->ni = fmax(s->ni, nssl_div(rho * s->qi, ICE_MMAX));
        s->ni = fmin(s->ni, nssl_div(rho * s->qi, ICE_MMIN));
    } else {
        // GS gather clears crystal number whenever qi is not active.
        s->ni = 0.0f;
    }
    if (s->qs > QS_MIN) {
        float volume = rho * s->qs
            / (100.0f * fmax(1.0e-9f, s->ns));
        if (volume < SNOW_VMIN) {
            volume = SNOW_VMIN;
            s->ns = nssl_div(rho * s->qs, (100.0f * volume));
        }
        if (volume > SNOW_VMAX) {
            volume = fmin(SNOW_VMAX, fmax(SNOW_VMIN, volume));
            const float mass = 0.106214f * pow(volume, 2.0f / 3.0f);
            s->ns = rho * s->qs / mass;
            p.snow_density = 0.0346159f * sqrt(s->ns / (s->qs * rho));
        }
    } else {
        // Snow is the setvtz exception: its number is cleared throughout the
        // non-active mixing-ratio branch rather than retained.
        s->ns = 0.0f;
    }
    if (s->qg > QD_MIN) {
        if (s->vg > 0.0f) {
            p.graupel_density = fmin(
                900.0f, fmax(170.0f, rho * s->qg / s->vg));
        }
        s->vg = rho * s->qg / p.graupel_density;
    }
    if (s->qh > QD_MIN) {
        if (s->vh > 0.0f) {
            p.hail_density = fmin(
                900.0f, fmax(500.0f, rho * s->qh / s->vh));
        }
        s->vh = rho * s->qh / p.hail_density;
    }
    if (s->qg > QD_MIN) {
        float mean_volume = rho * s->qg
            / (p.graupel_density * fmax(1.0e-9f, s->ng));
        if (mean_volume < GRAUPEL_VMIN
                || mean_volume > GRAUPEL_INITIAL_VMAX) {
            mean_volume = fmin(
                GRAUPEL_INITIAL_VMAX,
                fmax(GRAUPEL_VMIN, mean_volume));
            s->ng = rho * s->qg / (p.graupel_density * mean_volume);
        }
        // Inactive graupel retains its gathered number moment.
    }
    if (s->qh > QD_MIN) {
        float mean_volume = rho * s->qh
            / (p.hail_density * fmax(1.0e-9f, s->nh));
        if (mean_volume < HAIL_VMIN || mean_volume > HAIL_INITIAL_VMAX) {
            mean_volume = fmin(
                HAIL_INITIAL_VMAX, fmax(HAIL_VMIN, mean_volume));
            s->nh = rho * s->qh / (p.hail_density * mean_volume);
        }
    } else {
        // GS gather clears hail number whenever qh is not active.
        s->nh = 0.0f;
    }
    return p;
}

static inline void final_bounds(
    thread State* s, float rho, thread const ParticleProperties& p)
{
    // WRF can leave a negative FP32 donor-limit cancellation residue in the
    // mass work arrays.  GPUWM's production Registry and downstream physics
    // require exact nonnegativity, matching the final bounds already applied
    // to every number and dense-volume moment below.  Positive mass values
    // are preserved bit-for-bit.
    bound_liquid(s->qc, rho, CLOUD_VMIN, CLOUD_VMAX, &s->nc);
    bound_liquid(s->qr, rho, RAIN_VMIN, RAIN_VMAX, &s->nr);
    bound_ice(s, rho);
    bound_snow(s, rho, p.snow_density);
    bound_dense_number(
        s->qg, &s->ng, rho, p.graupel_density,
        GRAUPEL_VMIN, GRAUPEL_VMAX);
    bound_dense_number(
        s->qh, &s->nh, rho, p.hail_density,
        HAIL_VMIN, HAIL_VMAX);
    s->qv = fmax(s->qv, 0.0f);
    s->qc = fmax(s->qc, 0.0f);
    s->qr = fmax(s->qr, 0.0f);
    s->qi = fmax(s->qi, 0.0f);
    s->qs = fmax(s->qs, 0.0f);
    s->qg = fmax(s->qg, 0.0f);
    s->qh = fmax(s->qh, 0.0f);
    s->nc = fmax(s->nc, 0.0f);
    s->nr = fmax(s->nr, 0.0f);
    s->ni = fmax(s->ni, 0.0f);
    s->ns = fmax(s->ns, 0.0f);
    s->ng = fmax(s->ng, 0.0f);
    s->nh = fmax(s->nh, 0.0f);
    s->nn = fmax(s->nn, 0.0f);
    s->vg = fmax(s->vg, 0.0f);
    s->vh = fmax(s->vh, 0.0f);
}

static inline void diagnose_warm(
    thread const State& s, float rho, float pressure, float dt, thread Rates* r)
{
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    float cloud_number = s.nc;
    float rain_number = s.nr;

    // setvtz cloud and rain reconstruction, option-18 cnu=alphar=0.
    float cloud_volume = CLOUD_VMIN;
    float cloud_diameter = 4.0e-6f;
    if (s.qc > QC_MIN) {
        const float mass_min = 1000.0f * CLOUD_VMIN;
        const float mass_max = 1000.0f * CLOUD_VMAX;
        float mass;
        if (cloud_number > CXMIN) {
            mass = fmin(fmax(s.qc * rho / cloud_number, mass_min), mass_max);
        } else {
            cloud_number = fmax(CXMIN, nssl_div(rho * s.qc, mass_max));
            mass = fmin(fmax(s.qc * rho / cloud_number, mass_min), mass_max);
        }
        cloud_volume = nssl_div(mass, 1000.0f);
        cloud_diameter = pow(mass * (6.0f / (PI * 1000.0f)), 1.0f / 3.0f);
    }

    float rain_volume = RAIN_VMIN;
    float rain_diameter = 80.0e-6f;
    float rain_characteristic = 1.0e-9f;
    if (s.qr > QR_MIN) {
        rain_volume = rho * s.qr /
            (1000.0f * fmax(1.0e-11f, rain_number));
        if (rain_volume < RAIN_VMIN || rain_volume > RAIN_VMAX) {
            rain_volume = fmin(RAIN_VMAX, fmax(RAIN_VMIN, rain_volume));
            rain_number = rho * s.qr / (1000.0f * rain_volume);
        }
        rain_diameter = pow(rain_volume * (6.0f / PI), 1.0f / 3.0f);
        rain_characteristic = pow(nssl_div(rain_volume, PI), 1.0f / 3.0f);
    }

    const Nssl64 rb = (Nssl64)(0.5f * cloud_diameter);
    const float xl2p_prefactor =
        2.7e-2f * 1000.0f * cloud_number * cloud_volume;
    const Nssl64 xl2p_shape =
        (Nssl64)5.0e19f * rb * rb * rb * (Nssl64)cloud_diameter
        - (Nssl64)0.4f;
    const Nssl64 xl2p = fmax(Nssl64(0.0f), (Nssl64)xl2p_prefactor * xl2p_shape);

    if (s.qc > QC_MIN && cloud_number > 1000.0f
            && s.temperature > 237.15f) {
        const float ccmxd = (float)(
            nssl64_bits(0x3fb999999999999aul) * (Nssl64)cloud_number * (Nssl64)dt_inverse);
        const float collision = 2.0f * 9.44e15f
            * (cloud_number * cloud_number)
            * (cloud_volume * cloud_volume);
        r->cautn = (float)fmax(Nssl64(0.0f), (Nssl64)fmin(ccmxd, collision));
        if (rb > nssl64_bits(0x3edf7fcd14c635aful)) {
            const Nssl64 t2s = (Nssl64)3.72f /
                ((Nssl64)1.0e6f * (rb - nssl64_bits(0x3edf75104d551d69ul))
                 * (Nssl64)rho * (Nssl64)s.qc);
            r->qrcnw = (float)fmax(Nssl64(0.0f), xl2p / (t2s * (Nssl64)rho));
            r->crcnw = (float)fmax(
                0.0, fmin(3.5e9 * xl2p / t2s, 0.5f * (Nssl64)r->cautn));
            if ((Nssl64)(s.qr * rho) > nssl64_bits(0x3ff3333333333333ul) * xl2p
                    && rain_number > CXMIN && s.qr > 0.0f) {
                r->crcnw = rain_number / s.qr * r->qrcnw;
            }
            if (r->crcnw < 1.0e-30f) r->qrcnw = 0.0f;
        }
    }

    // Rain collection of cloud water and cloud number.
    if (s.qc > QC_MIN && s.qr > QR_MIN) {
        Nssl64 initiation_radius = nssl64_bits(0x3f057eed45e9185dul);
        if (rb > nssl64_bits(0x3ecd71aae8209043ul)) {
            initiation_radius = fmax(
                nssl64_bits(0x3f057eed45e9185dul),
                nssl64_bits(0x3f44a4d2b2bfdb4dul) / (1.0e6 * (rb - nssl64_bits(0x3ecd5c31593e5fb7ul))));
        }
        const float rain_radius = 0.5f * rain_diameter;
        if ((Nssl64)rain_radius > initiation_radius) {
            if (rain_radius > 50.0e-6f) {
                r->qracw = 5.78e3f * rain_number * cloud_number
                    * (1000.0f * cloud_volume)
                    * (2.0f * cloud_volume + rain_volume) / rho;
                r->cracw = 5.78e3f * rain_number * cloud_number
                    * (cloud_volume + rain_volume);
            } else {
                r->qracw = 9.44e15f * rain_number * s.qc
                    * (6.0f * cloud_volume * cloud_volume
                       + 20.0f * rain_volume * rain_volume);
                r->cracw = 9.44e15f * rain_number * cloud_number
                    * (2.0f * cloud_volume * cloud_volume
                       + 20.0f * rain_volume * rain_volume);
            }
            r->qracw = fmin(
                r->qracw,
                (float)(nssl64_bits(0x3fb999999999999aul) * (Nssl64)s.qc * (Nssl64)dt_inverse));
            if (!(r->qracw > 0.0f)) r->cracw = 0.0f;
        }
    }

    // Rain self-collection/breakup number sink.
    if (s.qr > QR_MIN && rain_diameter - 0.1e-3f <= 1.9e-3f) {
        float efficiency = 1.0f;
        if (rain_diameter >= 6.1e-4f) {
            efficiency = exp(-50.0f * (50.0f * (rain_diameter - 6.0e-4f)));
        }
        if (0.5f * rain_diameter >= 50.0e-6f) {
            r->cracr = (float)((Nssl64)efficiency * (Nssl64)5.78e3f
                * (Nssl64)(rain_number * rain_number)
                * (Nssl64)rain_volume);
        } else {
            const float ratio = (6.0f * 5.0f * 4.0f) /
                (3.0f * 2.0f * 1.0f);
            const float nv = rain_number * rain_volume;
            r->cracr = (float)((Nssl64)efficiency * (Nssl64)9.44e15f
                * (Nssl64)(nv * nv) * (Nssl64)ratio);
        }
    }

    // Rain evaporation. Default rcond=0 makes this a non-positive rate.
    if (s.qr > QR_MIN) {
        int table_index = (int)(nssl_div((s.temperature - 163.15f), 0.002f) + 1.5f);
        table_index = min(1000001, max(1, table_index));
        const float table_temperature = nssl_add(
            163.15f, nssl_mul((float)(table_index - 1), 0.002f));
        const float saturation = (380.0f / pressure) * exp(
            17.2693882f * (table_temperature - 273.15f)
            / (table_temperature - 35.86f));
        const float lv = latent_vapor(s.temperature);
        const float diffusivity = 2.11e-5f
            * pow(nssl_div(s.temperature, 273.15f), 1.94f)
            * (101325.0f / pressure);
        const float viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * pow(nssl_div(s.temperature, 296.0f), 1.5f);
        const float kinematic = viscosity / rho;
        const float conductivity = nssl_div(2.43e-2f * viscosity, 1.718e-5f);
        const float ventilation_factor = pow(
            kinematic / diffusivity, 1.0f / 3.0f) * pow(kinematic, -0.5f);
        const float fall_factor = sqrt(1.225f / fmax(0.05f, rho));
        const float ventilation = 0.78f
            + 0.308f * 1.8273550271987915f * ventilation_factor
            * sqrt(841.99666f * fall_factor)
            * pow(rain_characteristic, 0.9f);
        const float capacitance = 0.5f * rain_characteristic;
        const float resistance = lv * lv /
            (conductivity * 461.5f * s.temperature * s.temperature)
            + 1.0f / (rho * diffusivity * saturation);
        const float growth = (4.0f * PI / rho)
            * (s.qv / saturation - 1.0f) / resistance;
        r->qrcev = fmin(
            growth * rain_number * ventilation * capacitance, 0.0f);
        r->qrcev = fmax(
            r->qrcev,
            -(float)(nssl64_bits(0x3fb999999999999aul) * (Nssl64)s.qr * (Nssl64)dt_inverse));
        if (r->qrcev < 0.0f) {
            r->crcev = (rain_number / s.qr) * r->qrcev;
        }
    }
}

static inline void diagnose_cloud_riming(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float layer_depth,
    float dt,
    thread Rates* r)
{
    // WRF 15493-17238: all frozen collectors diagnose from the same cloud
    // state.  Their independent local caps are followed later by WRF's one
    // shared cloud mass/number donor limiter.
    // The process branches themselves gate on hydrometeor mass, not CXMIN;
    // initial setvtz has already made every active category number nonzero.
    // ICEZVD_GS initializes rimdn(:,:) to the fixed 500 kg m-3 `rimedens`.
    // Graupel/rain collection later (intentionally) reuses the cloud-riming
    // value through WRF's raindn/rimdn typo.
    r->graupel_cloud_rime_density = 500.0f;
    if (!(s.qc > QC_MIN)) return;
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float cloud_mass = fmin(
        1000.0f * CLOUD_VMAX,
        fmax(1000.0f * CLOUD_VMIN, rho * s.qc / s.nc));
    const float cloud_volume = nssl_div(cloud_mass, 1000.0f);
    const float cloud_diameter =
        pow(cloud_volume * (6.0f / PI), 1.0f / 3.0f);
    const float cloud_radius = 0.5f * cloud_diameter;
    const float viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * pow(nssl_div(s.temperature, 296.0f), 1.5f);
    const float cloud_velocity = 2.0f * 9.8f * 1000.0f
        * cloud_radius * cloud_radius / (9.0f * viscosity);
    const float density_factor = sqrt(1.225f / fmax(0.05f, rho));

    // Cloud droplets collected by pristine cloud ice.
    if (s.temperature < 273.15f && s.qi > QI_MIN) {
        const float ice_mass = fmax(rho * s.qi / s.ni, ICE_MMIN);
        const float ice_diameter = 0.1871f * pow(ice_mass, 0.3429f);
        if (cloud_diameter > 15.0e-6f && ice_diameter > 30.0e-6f) {
            const float ice_volume = nssl_div(ice_mass, 900.0f);
            const float ice_velocity = 47.6273f * density_factor
                * pow(ice_volume, 0.18333f) * 1.091937899589539f;
            const float relative_velocity = sqrt(
                (ice_velocity - cloud_velocity)
                    * (ice_velocity - cloud_velocity)
                + 0.04f * ice_velocity * cloud_velocity);
            const float geometry = nssl_tgamma(1.6858f)
                    * ice_diameter * ice_diameter
                + 2.0f * nssl_tgamma(1.3429f) * nssl_tgamma(2.3333333333f)
                    * ice_diameter * cloud_diameter
                + nssl_tgamma(2.6666666667f)
                    * cloud_diameter * cloud_diameter;
            r->qiacw = fmin(
                0.25f * PI * 0.5f * s.ni * s.qc
                    * relative_velocity * geometry,
                0.1f * s.qc * dt_inverse);
            r->ciacw = fmin(
                r->qiacw * rho / cloud_mass,
                0.1f * s.nc * dt_inverse);
        }
    }

    // Cloud droplets collected by snow.
    if (s.qs > QS_MIN) {
        const float snow_volume = fmax(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float collection_count = 0.104f * 5.78e3f
            * s.ns * s.nc * (2.0f * cloud_volume + snow_volume);
        r->qsacw = fmin(
            collection_count * cloud_mass / rho,
            0.1f * s.qc * dt_inverse);
        r->csacw = fmin(
            collection_count, 0.1f * s.nc * dt_inverse);
    }

    const float collection_efficiency = fmin(
        0.9f,
        fmin(
            -0.27544f + cloud_radius
                * (0.26249e6f + cloud_radius
                    * (-1.8896e10f + cloud_radius * 4.4626e14f)),
            1.0f));
    if (!(collection_efficiency > 0.0f) || cloud_diameter < 2.4e-6f) {
        return;
    }

    // Cloud droplets collected by graupel.
    if (s.qg > QD_MIN) {
        const float mean_volume = fmax(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        const float diameter = pow(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic = pow(6.0f, -1.0f / 3.0f) * diameter;
        float fall_a, fall_b;
        dense_fall_coefficients(
            particles.graupel_density, &fall_a, &fall_b);
        const float velocity = density_factor * fall_a * pow(characteristic, fall_b)
            * wrf_gamma_lookup(4.0f + fall_b) / wrf_gamma_lookup(4.0f);
        // WRF's 70 and 150 m s-1 clamps belong only to sedimentation.  GS
        // microphysical collection uses the uncapped vtxbar fall speed.
        const float relative_velocity = fabs(velocity - cloud_velocity);
        // WRF's delbk/delabk tables are initialized through GAMMA_SP and
        // rounded to real.  Preserve those admitted alpha_g=0/alpha_c=2
        // coefficients instead of recomputing them with CUDA tgammaf.
        const float geometry = 0.605706990f * diameter * diameter
            + 1.31048179f * cloud_diameter * diameter
            + 1.50459337f * cloud_diameter * cloud_diameter;
        r->qhacw = fmin(
            0.25f * PI * collection_efficiency * s.ng * s.qc
                * relative_velocity * geometry,
            0.5f * s.qc * dt_inverse);
        r->qhacwmlr = r->qhacw;
        r->chacw = fmin(
            r->qhacw * rho / cloud_mass,
            0.5f * s.nc * dt_inverse);
        float rime_density = 1000.0f;
        if (s.temperature < 273.15f) {
            const float rime_parameter = -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * velocity) / (s.temperature - 273.15f);
            rime_density = fmin(
                900.0f,
                fmax(170.0f, 300.0f * pow(rime_parameter, 0.44f)));
        }
        r->graupel_cloud_rime_density = rime_density;
        r->vhacw = rho * r->qhacw / rime_density;
    }

    // Cloud droplets collected by hail.  WRF caps hail fall speed by dz/dt
    // specifically in this collection path.
    if (s.qh > QD_MIN) {
        const float mean_volume = fmax(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        const float diameter = pow(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic = pow(24.0f, -1.0f / 3.0f) * diameter;
        float fall_a, fall_b;
        dense_fall_coefficients(particles.hail_density, &fall_a, &fall_b);
        float velocity = density_factor * fall_a * pow(characteristic, fall_b)
            * wrf_gamma_lookup(5.0f + fall_b) / wrf_gamma_lookup(5.0f);
        velocity = fmin(layer_depth * dt_inverse, velocity);
        const float relative_velocity = fabs(velocity - cloud_velocity);
        // Static WRF Seifert-table coefficients for admitted alpha_hail=1
        // and alpha_cloud=2.  The mixed term is delabk(lhl,lc,k=1), not the
        // product of two independently recomputed diameter moments.
        const float geometry = 0.721125066f * diameter * diameter
            + 1.65110373f * cloud_diameter * diameter
            + 1.50459337f * cloud_diameter * cloud_diameter;
        r->qhlacw = fmin(
            0.25f * PI * collection_efficiency * s.nh * s.qc
                * relative_velocity * geometry,
            0.5f * s.qc * dt_inverse);
        r->qhlacwmlr = r->qhlacw;
        r->chlacw = fmin(
            r->qhlacw * rho / cloud_mass,
            0.5f * s.nc * dt_inverse);
        float rime_density = 1000.0f;
        if (s.temperature < 273.15f) {
            const float rime_parameter = -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * velocity) / (s.temperature - 273.15f);
            rime_density = fmin(
                900.0f,
                fmax(500.0f, 300.0f * pow(rime_parameter, 0.44f)));
        }
        r->vhlacw = rho * r->qhlacw / rime_density;
    }
}

static inline void diagnose_rain_freezing(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    thread Rates* r)
{
    // WRF 16000-16045, 16742-16927, 17575-17918, 20318-20379.
    // Bigg freezing and rain/ice collision are diagnosed independently, then
    // share one rain heat budget before any number or mass donor limiter.
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float temperature_c = s.temperature - 273.15f;
    float rain_number = s.nr;
    float rain_mean_volume = RAIN_VMIN;
    float rain_mean_diameter = 1.0e-9f;
    float rain_characteristic = 1.0e-9f;
    if (s.qr > QR_MIN) {
        rain_mean_volume = rho * s.qr
            / (1000.0f * fmax(1.0e-11f, rain_number));
        rain_mean_volume = fmin(
            RAIN_VMAX, fmax(RAIN_VMIN, rain_mean_volume));
        rain_number = rho * s.qr / (1000.0f * rain_mean_volume);
        rain_mean_diameter =
            pow(rain_mean_volume * (6.0f / PI), 1.0f / 3.0f);
        rain_characteristic = pow(nssl_div(rain_mean_volume, PI), 1.0f / 3.0f);
    }

    // Default Bigg option 2, including the active <0.30-mm snow split.
    if (s.qr > QR_MIN && temperature_c < -5.0f) {
        const float threshold_volume = exp(16.2f + temperature_c) * 1.0e-6f;
        const float bigg_diameter =
            pow((6.0f / PI) * threshold_volume, 1.0f / 3.0f);
        if (bigg_diameter < 8.0e-3f) {
            float number_fraction, mass_fraction;
            rain_tail_fractions(
                bigg_diameter / rain_characteristic,
                &number_fraction,
                &mass_fraction);
            r->crfrz = number_fraction * rain_number * dt_inverse;
            r->qrfrz = mass_fraction * s.qr * dt_inverse;
            r->crfrzf = r->crfrz;
            r->qrfrzf = r->qrfrz;
            if (r->qrfrz * dt < QD_MIN || r->crfrz * dt < CXMIN) {
                r->crfrz = 0.0f;
                r->qrfrz = 0.0f;
                r->crfrzf = 0.0f;
                r->qrfrzf = 0.0f;
            } else if (bigg_diameter < 0.30e-3f) {
                float dense_number_fraction, dense_mass_fraction;
                rain_tail_fractions(
                    0.30e-3f / rain_characteristic,
                    &dense_number_fraction,
                    &dense_mass_fraction);
                r->crfrzf = dense_number_fraction * rain_number * dt_inverse;
                r->qrfrzf = dense_mass_fraction * s.qr * dt_inverse;
                r->crfrzs = r->crfrz - r->crfrzf;
                r->qrfrzs = r->qrfrz - r->qrfrzf;
            }
            // Preserve the source's normally unreachable inverted correction.
            if (r->qrfrz * dt > s.qr) {
                const float factor = r->qrfrz * dt / s.qr;
                r->qrfrz *= factor;
                r->qrfrzs *= factor;
                r->qrfrzf *= factor;
                r->crfrz *= factor;
                r->crfrzs *= factor;
                r->crfrzf *= factor;
            }
            r->vrfrzf = nssl_div(rho * r->qrfrzf, 900.0f);
        }
    }

    const bool collision_active = s.qr > QR_MIN && s.qi > QI_MIN
        && s.ni > CXMIN;
    float ice_mass = ICE_MMIN;
    float ice_volume = ICE_MMIN / 900.0f;
    float ice_diameter = 1.0e-7f;
    if (collision_active) {
        ice_mass = fmax(rho * s.qi / s.ni, ICE_MMIN);
        ice_volume = nssl_div(ice_mass, 900.0f);
        ice_diameter = 0.1871f * pow(ice_mass, 0.3429f);
    }
    const float density_factor = sqrt(1.225f / fmax(0.05f, rho));
    const float rain_velocity = s.qr > QR_MIN
        ? density_factor * 10.0f * (1.0f - pow(
            1.0f + 516.575f * rain_characteristic, -4.0f))
        : 0.0f;
    const float ice_velocity = collision_active
        ? 47.6273f * density_factor * pow(ice_volume, 0.18333f)
            * 1.091937899589539f
        : 0.0f;

    // Rain collection of cloud-ice mass/number (qraci/craci).
    if (collision_active && ice_diameter >= 10.0e-6f
            && rain_mean_diameter > 100.0e-6f) {
        const float collision_count = 0.1f * 5.78e3f
            * rain_number * s.ni * (2.0f * ice_volume + rain_mean_volume);
        r->qraci = fmin(
            0.1f * s.qi * dt_inverse,
            collision_count * ice_mass / rho);
        r->craci = fmin(
            0.1f * s.ni * dt_inverse, collision_count);
        if (s.temperature > 268.15f) r->qraci = 0.0f;
        r->qracif = r->qraci;
    }

    // Rain frozen by collision with cloud ice (qiacr/ciacr).
    const bool double_moment_collision_gate =
        collision_active && ice_diameter >= 10.0e-6f
        && s.temperature <= 270.15f;
    if (double_moment_collision_gate) {
        const float eligible_ice_number = s.ni
            * exp(-pow(40.0e-6f / ice_diameter, 3.0f));
        float rain_number_fraction, rain_mass_fraction;
        rain_tail_fractions(
            150.0e-6f / rain_characteristic,
            &rain_number_fraction,
            &rain_mass_fraction);
        const float eligible_rain_number = rain_number_fraction * rain_number;
        const float eligible_rain = rain_mass_fraction * s.qr;
        const float relative_velocity = sqrt(
            (rain_velocity - ice_velocity) * (rain_velocity - ice_velocity)
            + 0.04f * rain_velocity * ice_velocity);
        float graupel_diameter = 1.0e-9f;
        if (s.qg > QD_MIN && s.ng > CXMIN) {
            const float graupel_volume = rho * s.qg
                / (particles.graupel_density * s.ng);
            graupel_diameter =
                pow(graupel_volume * (6.0f / PI), 1.0f / 3.0f);
        }
        const float gamma_ice = nssl_tgamma(1.3429f);
        const float one_sixth = 1.0f / 6.0f;
        const float mass_geometry = nssl_tgamma(1.6858f)
                * ice_diameter * ice_diameter
            + 48.0f * gamma_ice * pow(one_sixth, 4.0f / 3.0f)
                * graupel_diameter * ice_diameter
            + 120.0f * pow(one_sixth, 5.0f / 3.0f)
                * rain_mean_diameter * rain_mean_diameter;
        const float number_geometry = nssl_tgamma(1.6858f)
                * ice_diameter * ice_diameter
            + 2.0f * gamma_ice * pow(one_sixth, 1.0f / 3.0f)
                * rain_mean_diameter * ice_diameter
            + 2.0f * pow(one_sixth, 2.0f / 3.0f)
                * rain_mean_diameter * rain_mean_diameter;
        r->qiacr = fmin(
            0.1f * s.qr * dt_inverse,
            0.25f * PI * 0.1f * eligible_ice_number * eligible_rain
                * relative_velocity * mass_geometry);
        r->ciacr = fmin(
            0.1f * rain_number * dt_inverse,
            0.25f * PI * 0.1f * eligible_ice_number * eligible_rain_number
                * relative_velocity * number_geometry);
        r->ciacrf = r->ciacr;
        float dense_fraction = 1.0f;
        if (r->ciacr > QD_MIN) {
            const float frozen_mean_volume =
                rho * r->qiacr / (r->ciacr * 900.0f);
            dense_fraction = 0.5f * (1.0f + tanh(
                0.2e12f * (frozen_mean_volume
                    - 1.15f * GRAUPEL_VMIN)));
            r->qiacrs = (1.0f - dense_fraction) * r->qiacr;
            r->ciacrs = (1.0f - dense_fraction) * r->ciacrf;
        }
        r->qiacrf = dense_fraction * r->qiacr;
        r->ciacrf = dense_fraction * r->ciacrf;
        r->viacrf = nssl_div(rho * r->qiacrf, 900.0f);
    } else {
        // Source-exact dangling-ELSE behavior at WRF 16754-16886: this
        // nominal "single-moment rain" branch pairs with the outer
        // iacr/efficiency/temperature gate.  Therefore it executes for the
        // admitted two-moment configuration whenever that outer gate is
        // false (notably T > 270.15 K), while retaining eri as its multiplier.
        const float efficiency = collision_active
                && ice_diameter >= 10.0e-6f
            ? 0.1f
            : 0.0f;
        const float geometry =
            120.0f * rain_characteristic * rain_characteristic
            + 48.0f * rain_characteristic * ice_diameter
            + 12.0f * ice_diameter * ice_diameter;
        r->qiacr = fmin(
            0.1f * s.qr * dt_inverse,
            (0.25f / 6.0f) * PI * efficiency * s.ni * s.qr
                * fabs(rain_velocity - ice_velocity) * geometry);
        r->qiacrf = r->qiacr;
        r->viacrf = nssl_div(rho * r->qiacrf, 900.0f);
    }

    const float freezing_total = r->qrfrz + r->qiacr + r->qsacr;
    float maximum_freezing = fmax(0.0f, freezing_total);
    if (!(temperature_c < -30.0f)) {
        const float diffusivity = 2.11e-5f
            * pow(nssl_div(s.temperature, 273.15f), 1.94f)
            * (101325.0f / pressure);
        const float viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * pow(nssl_div(s.temperature, 296.0f), 1.5f);
        const float kinematic = viscosity / rho;
        const float ventilation_factor = pow(
            kinematic / diffusivity, 1.0f / 3.0f)
            * pow(kinematic, -0.5f);
        const float rain_ventilation = 0.78f
            + 0.308f * nssl_tgamma(2.9f) * ventilation_factor
                * sqrt(841.99666f * density_factor)
                * pow(rain_characteristic, 0.9f);
        const float bounded_celsius = fmin(
            273.15f, fmax(233.15f, s.temperature)) - 273.15f;
        const float liquid_offset = bounded_celsius - 35.0f;
        const float liquid_heat = 4203.1548f
            + 1.30572e-2f * liquid_offset * liquid_offset
            + 1.60056e-5f * liquid_offset * liquid_offset
                * liquid_offset * liquid_offset;
        const float conductivity = nssl_div(2.43e-2f * viscosity, 1.718e-5f);
        const float wet_growth = (2.0f * PI)
            * (latent_vapor(s.temperature) * diffusivity * rho
                    * (380.0f / pressure - s.qv)
                - conductivity * temperature_c)
            / (rho * (latent_fusion(s.temperature)
                + liquid_heat * temperature_c));
        maximum_freezing = fmax(
            rain_characteristic * rain_ventilation * rain_number
                * wet_growth,
            0.0f);
        maximum_freezing = fmin(freezing_total, maximum_freezing);
        maximum_freezing = fmin(s.qr * dt_inverse, maximum_freezing);
    } else {
        maximum_freezing = s.qr * dt_inverse;
    }

    float factor = 1.0f;
    if (freezing_total > maximum_freezing && freezing_total > QR_MIN) {
        factor = maximum_freezing / freezing_total;
    }
    factor = fmin(1.0f, factor);
    if (s.temperature <= 273.15f && factor < 1.0f) {
        r->qrfrz *= factor;
        r->qrfrzs *= factor;
        r->qrfrzf *= factor;
        r->qiacr *= factor;
        r->qsacr *= factor;
        r->qiacrf *= factor;
        r->qiacrs *= factor;
        r->crfrz *= factor;
        r->crfrzf *= factor;
        r->crfrzs *= factor;
        r->ciacr *= factor;
        r->ciacrf *= factor;
        r->ciacrs *= factor;
        r->vrfrzf *= factor;
        r->viacrf *= factor;
    }
}

static inline void diagnose_cloud_freezing(
    thread const State& s,
    float rho,
    float pressure,
    float dt,
    thread Rates* r)
{
    // WRF 17927-18118, exact default ibfc=1 and icfn=2 paths.
    if (!(s.qc > QC_MIN)) return;
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float cloud_volume = fmin(
        CLOUD_VMAX,
        fmax(
            CLOUD_VMIN,
            rho * s.qc / (1000.0f * fmax(s.nc, 1.0e-20f))));
    const float cloud_mass = 1000.0f * cloud_volume;
    const float cloud_diameter =
        pow(cloud_volume * (6.0f / PI), 1.0f / 3.0f);

    // Homogeneous freezing.  There is no local qcmxd/ccmxd cap for the
    // ipconc>=2 path; the later shared cloud limiters own the competition.
    if (s.temperature < 268.15f && s.nc > CXMIN
            && cloud_diameter > 0.0f) {
        const float threshold_volume =
            exp(16.2f + temperature_c) * 1.0e-6f;
        r->cwfrz = s.nc * exp(-threshold_volume / cloud_volume)
            * dt_inverse;
        r->qwfrz = r->cwfrz * 1000.0f / rho
            * (threshold_volume + cloud_volume);
        // Default crystal routing is entirely to columns/cloud ice.
        r->cwfrzc = r->cwfrz;
        r->qwfrzc = r->qwfrz;
    }

    // Cotton/Meyers contact freezing.  The official gate has no explicit
    // number threshold; a zero number simply makes fn1 and the rate zero.
    if (!(s.temperature < 271.15f)) return;
    const float ccia = exp(4.11f - 0.262f * temperature_c);
    constexpr float AEROSOL_RADIUS = 3.0e-7f;
    constexpr float AEROSOL_CONDUCTIVITY = 5.39e-3f;
    constexpr float BOLTZMANN = 1.3807e-23f;
    const float knudsen = 2.28e-5f * s.temperature
        / (pressure * AEROSOL_RADIUS);
    const float knudsen_adjustment =
        1.257f + 0.4f * exp(-1.1f / knudsen);
    const float viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * pow(nssl_div(s.temperature, 296.0f), 1.5f);
    const float diffusivity = 2.11e-5f
        * pow(nssl_div(s.temperature, 273.15f), 1.94f)
        * (101325.0f / pressure);
    const float conductivity = nssl_div(2.43e-2f * viscosity, 1.718e-5f);
    const float lv = latent_vapor(s.temperature);
    const float ls = lv + latent_fusion(s.temperature);
    const float ice_saturation =
        ice_saturation_mixing_ratio(s.temperature, pressure);
    const float water_saturation =
        liquid_saturation_mixing_ratio(s.temperature, pressure);
    const float fai = ls * ls
        / (conductivity * 461.5f * s.temperature * s.temperature);
    const float fbi = 1.0f / (rho * diffusivity * ice_saturation);
    const float growth_transport = 1.0f / (fai + fbi);
    const float aerosol_diffusivity = BOLTZMANN * s.temperature
        * (1.0f + knudsen_adjustment * knudsen)
        / (6.0f * PI * viscosity * AEROSOL_RADIUS);
    const float fn1 =
        2.0f * PI * cloud_diameter * s.nc * ccia;
    const float fn2 = -growth_transport
        * (s.qv / water_saturation - 1.0f) * lv / pressure;
    const float thermal_factor = 0.4f
        * (1.0f + 1.45f * knudsen
            + 0.4f * knudsen * exp(-1.0f / knudsen))
        * (conductivity + 2.5f * knudsen * AEROSOL_CONDUCTIVITY)
        / ((1.0f + 3.0f * knudsen)
            * (2.0f * conductivity
                + 5.0f * knudsen * AEROSOL_CONDUCTIVITY
                + AEROSOL_CONDUCTIVITY));
    const float brownian = fn1 * aerosol_diffusivity;
    const float thermophoretic = fn1 * fn2 * thermal_factor / rho;
    const float diffusiophoretic = fn1 * fn2 * 1000.0f * s.temperature
        / (lv * rho);
    const float raw_count =
        fmax(brownian + thermophoretic + diffusiophoretic, 0.0f);
    const float cwctfz = fmin(
        raw_count * dt_inverse,
        0.1f * s.nc * dt_inverse);
    r->qwctfz = cloud_mass * cwctfz / rho;
    r->cwctfzc = cwctfz;
    r->qwctfzc = r->qwctfz;
}

static inline void diagnose_frozen_collection(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float layer_depth,
    float dt,
    thread Rates* r)
{
    // WRF 15599-17323 default dry cross-collection.  The ipconc=5
    // snow-rain two-moment branch is intentionally empty, so qsacr remains 0.
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrt(1.225f / fmax(0.05f, rho));

    float rain_volume = RAIN_VMIN;
    float rain_diameter = 1.0e-9f;
    float rain_characteristic = 1.0e-9f;
    float rain_velocity = 0.0f;
    if (s.qr > QR_MIN && s.nr > 0.0f) {
        rain_volume = fmin(
            RAIN_VMAX,
            fmax(RAIN_VMIN, rho * s.qr / (1000.0f * s.nr)));
        rain_diameter = pow(rain_volume * (6.0f / PI), 1.0f / 3.0f);
        rain_characteristic = pow(nssl_div(rain_volume, PI), 1.0f / 3.0f);
        rain_velocity = density_factor * 10.0f * (1.0f - pow(
            1.0f + 516.575f * rain_characteristic, -4.0f));
    }

    float ice_mass = ICE_MMIN;
    float ice_volume = ICE_MMIN / 900.0f;
    float ice_diameter = 1.0e-9f;
    float ice_velocity = 0.0f;
    if (s.qi > QI_MIN && s.ni > 0.0f) {
        ice_mass = fmax(rho * s.qi / s.ni, ICE_MMIN);
        ice_volume = nssl_div(ice_mass, 900.0f);
        ice_diameter = 0.1871f * pow(ice_mass, 0.3429f);
        ice_velocity = 47.6273f * density_factor * pow(ice_volume, 0.18333f)
            * 1.091937899589539f;
    }

    float snow_volume = SNOW_VMIN;
    float snow_diameter = 1.0e-9f;
    float snow_velocity = 0.0f;
    if (s.qs > QS_MIN && s.ns > 0.0f) {
        snow_volume = fmax(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        snow_diameter = pow(snow_volume * (6.0f / PI), 1.0f / 3.0f);
        snow_velocity = 11.9495f * density_factor * pow(snow_volume, 0.14f);
    }

    float graupel_volume = GRAUPEL_VMIN;
    float graupel_diameter = 1.0e-9f;
    float graupel_velocity = 0.0f;
    if (s.qg > QD_MIN && s.ng > 0.0f) {
        graupel_volume = fmax(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        graupel_diameter =
            pow(graupel_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic =
            pow(6.0f, -1.0f / 3.0f) * graupel_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.graupel_density, &coefficient, &exponent);
        graupel_velocity = nssl_div(density_factor * coefficient
            * pow(characteristic, exponent)
            * nssl_tgamma(4.0f + exponent), nssl_tgamma(4.0f));
    }

    float hail_volume = HAIL_VMIN;
    float hail_diameter = 1.0e-9f;
    float hail_velocity = 0.0f;
    if (s.qh > QD_MIN && s.nh > 0.0f) {
        hail_volume = fmax(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        hail_diameter = pow(hail_volume * (6.0f / PI), 1.0f / 3.0f);
        const float characteristic =
            pow(24.0f, -1.0f / 3.0f) * hail_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        hail_velocity = nssl_div(density_factor * coefficient
            * pow(characteristic, exponent)
            * nssl_tgamma(5.0f + exponent), nssl_tgamma(5.0f));
        hail_velocity = fmin(layer_depth * dt_inverse, hail_velocity);
    }

    constexpr float DA0_R = 0.6057068643f;
    constexpr float DA1_R = 6.057068643f;
    constexpr float DA0_I = 0.9060338860f;
    constexpr float DA1_I = 1.527396263f;
    constexpr float DA0_S = 0.6987612361f;
    constexpr float DA1_S = 3.027901273f;
    constexpr float DA0_G = 0.6057068643f;
    constexpr float DA0_H = 0.7211247852f;
    constexpr float DAB0_GI = 0.9816705967f;
    constexpr float DAB1_GI = 1.318283033f;
    constexpr float DAB0_GS = 0.6824619383f;
    constexpr float DAB1_GS = 1.819762383f;
    constexpr float DAB0_GR = 0.6057068643f;
    constexpr float DAB1_GR = 2.422827457f;
    constexpr float DAB0_HI = 1.236827449f;
    constexpr float DAB1_HI = 1.660932543f;
    constexpr float DAB0_HS = 0.8598481618f;
    constexpr float DAB1_HS = 2.292756933f;
    constexpr float DAB0_HR = 0.7631428284f;
    constexpr float DAB1_HR = 3.052571313f;

    if (s.qs > QS_MIN && s.qi > QI_MIN && s.ns > 0.0f && s.ni > 0.0f) {
        const float efficiency = fmin(
            0.1f, 0.1f * exp(0.1f * fmin(temperature_c, 0.0f)));
        if (s.temperature <= 273.15f && efficiency > 0.0f) {
            const float collision = 0.104f * 5.78e3f * s.ns * s.ni
                * (2.0f * ice_volume + snow_volume);
            r->qsaci = fmin(
                0.1f * s.qi * dt_inverse,
                efficiency * collision * ice_mass / rho);
            r->csaci = fmin(
                0.1f * s.ni * dt_inverse,
                efficiency * collision);
        }
    }

    if (s.qg > QD_MIN && s.qi > QI_MIN && s.ng > 0.0f && s.ni > 0.0f) {
        const float efficiency = fmin(
            1.0f,
            fmax(0.0f, 0.1f * exp(0.1f * fmin(temperature_c, 0.0f))));
        const float relative_velocity = sqrt(
            (graupel_velocity - ice_velocity)
                * (graupel_velocity - ice_velocity)
            + 0.04f * graupel_velocity * ice_velocity);
        const float mass_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB1_GI * graupel_diameter * ice_diameter
            + DA1_I * ice_diameter * ice_diameter;
        const float number_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB0_GI * graupel_diameter * ice_diameter
            + DA0_I * ice_diameter * ice_diameter;
        r->qhaci0 = 0.25f * PI * s.ng * s.qi
            * relative_velocity * mass_geometry;
        r->chaci0 = 0.25f * PI * s.ng * s.ni
            * relative_velocity * number_geometry;
        r->qhaci = fmin(
            efficiency * r->qhaci0, 0.1f * s.qi * dt_inverse);
        r->chaci = fmin(
            efficiency * r->chaci0, 0.1f * s.ni * dt_inverse);
    }

    if (s.qg > QD_MIN && s.qs > QS_MIN && s.ng > 0.0f && s.ns > 0.0f
            && s.qc >= QC_MIN) {
        float collision_efficiency = 0.5f;
        if (snow_diameter < 40.0e-6f) {
            collision_efficiency = 0.0f;
        } else if (snow_diameter < 150.0e-6f) {
            collision_efficiency = nssl_div(0.5f
                * (snow_diameter - 40.0e-6f), 110.0e-6f);
        }
        const float conversion_efficiency = fmin(
            0.5f,
            0.1f * exp(0.1f * fmin(temperature_c, 0.0f))
                * fmin(1.0f, nssl_div(fmax(
                    0.0f, particles.graupel_density - 300.0f), 300.0f)));
        if (collision_efficiency > 0.0f && conversion_efficiency > 0.0f) {
            const float relative_velocity = sqrt(
                (graupel_velocity - snow_velocity)
                    * (graupel_velocity - snow_velocity)
                + 0.04f * graupel_velocity * snow_velocity);
            const float mass_geometry = DA0_G * graupel_diameter * graupel_diameter
                + DAB1_GS * graupel_diameter * snow_diameter
                + DA1_S * snow_diameter * snow_diameter;
            const float number_geometry = DA0_G * graupel_diameter * graupel_diameter
                + DAB0_GS * graupel_diameter * snow_diameter
                + DA0_S * snow_diameter * snow_diameter;
            r->qhacs0 = 0.25f * PI * collision_efficiency
                * s.ng * s.qs * relative_velocity * mass_geometry;
            r->chacs0 = 0.25f * PI * collision_efficiency
                * s.ng * s.ns * relative_velocity * number_geometry;
            r->qhacs = fmin(
                conversion_efficiency * r->qhacs0,
                0.1f * s.qs * dt_inverse);
            r->chacs = fmin(
                conversion_efficiency * r->chacs0,
                0.1f * s.ns * dt_inverse);
        }
    }

    if (s.qg > QD_MIN && s.qr > QR_MIN && s.ng > 0.0f && s.nr > 0.0f) {
        const float efficiency = fmin(
            1.0f,
            exp(-40.0e-6f / rain_diameter)
                * exp(-40.0e-6f / graupel_diameter));
        const float relative_velocity = sqrt(
            (graupel_velocity - rain_velocity)
                * (graupel_velocity - rain_velocity)
            + 0.04f * graupel_velocity * rain_velocity);
        const float mass_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB1_GR * graupel_diameter * rain_diameter
            + DA1_R * rain_diameter * rain_diameter;
        const float number_geometry = DA0_G * graupel_diameter * graupel_diameter
            + DAB0_GR * graupel_diameter * rain_diameter
            + DA0_R * rain_diameter * rain_diameter;
        r->qhacr = fmin(
            0.1f * s.qr * dt_inverse,
            0.25f * PI * efficiency * s.ng * s.qr
                * relative_velocity * mass_geometry);
        r->qhacrmlr = r->qhacr;
        r->chacr = fmin(
            0.1f * s.nr * dt_inverse,
            0.25f * PI * efficiency * s.ng * s.nr
                * relative_velocity * number_geometry);
        if (s.temperature > 273.15f) {
            r->qhacr = 0.0f;
            r->chacr = 0.0f;
        } else if (s.temperature == 273.15f) {
            // At exactly freezing WRF leaves collection active but follows
            // the non-cold rain-density branch, whose divisor is liquid water.
            r->vhacr = nssl_div(rho * r->qhacr, 1000.0f);
        } else {
            // WRF 16518-16523 computes a rain-rime density and then
            // overwrites it with clamp(rimdn_g), so the earlier cloud-rime
            // diagnosis (or its 500 default) is the observable divisor.
            r->vhacr = rho * r->qhacr /
                fmin(900.0f, fmax(
                    170.0f, r->graupel_cloud_rime_density));
        }
    }

    if (s.qh > QD_MIN && s.qi > QI_MIN && s.nh > 0.0f && s.ni > 0.0f
            && s.temperature <= 273.15f && s.qc >= QC_MIN) {
        const float relative_velocity = sqrt(
            (hail_velocity - ice_velocity) * (hail_velocity - ice_velocity)
            + 0.04f * hail_velocity * ice_velocity);
        const float mass_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB1_HI * hail_diameter * ice_diameter
            + DA1_I * ice_diameter * ice_diameter;
        const float number_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB0_HI * hail_diameter * ice_diameter
            + DA0_I * ice_diameter * ice_diameter;
        r->qhlaci0 = 0.25f * PI * s.nh * s.qi
            * relative_velocity * mass_geometry;
        r->chlaci0 = 0.25f * PI * s.nh * s.ni
            * relative_velocity * number_geometry;
        r->qhlaci = fmin(
            0.2f * r->qhlaci0, 0.1f * s.qi * dt_inverse);
        r->chlaci = fmin(
            0.2f * r->chlaci0, 0.1f * s.ni * dt_inverse);
    }

    if (s.qh > QD_MIN && s.qs > QS_MIN && s.nh > 0.0f && s.ns > 0.0f) {
        const float efficiency = fmin(
            0.5f, 0.1f * exp(0.1f * fmin(temperature_c, 0.0f)));
        const float relative_velocity = sqrt(
            (hail_velocity - snow_velocity) * (hail_velocity - snow_velocity)
            + 0.04f * hail_velocity * snow_velocity);
        const float mass_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB1_HS * hail_diameter * snow_diameter
            + DA1_S * snow_diameter * snow_diameter;
        const float number_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB0_HS * hail_diameter * snow_diameter
            + DA0_S * snow_diameter * snow_diameter;
        r->qhlacs0 = 0.25f * PI * s.nh * s.qs
            * relative_velocity * mass_geometry;
        r->chlacs0 = 0.25f * PI * s.nh * s.ns
            * relative_velocity * number_geometry;
        r->qhlacs = fmin(
            efficiency * r->qhlacs0, 0.1f * s.qs * dt_inverse);
        r->chlacs = fmin(
            efficiency * r->chlacs0, 0.1f * s.ns * dt_inverse);
    }

    if (s.qh > QD_MIN && s.qr > QR_MIN && s.nh > 0.0f && s.nr > 0.0f) {
        const float relative_velocity = sqrt(
            (hail_velocity - rain_velocity) * (hail_velocity - rain_velocity)
            + 0.04f * hail_velocity * rain_velocity);
        const float mass_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB1_HR * hail_diameter * rain_diameter
            + DA1_R * rain_diameter * rain_diameter;
        const float number_geometry = DA0_H * hail_diameter * hail_diameter
            + DAB0_HR * hail_diameter * rain_diameter
            + DA0_R * rain_diameter * rain_diameter;
        r->qhlacr = fmin(
            0.1f * s.qr * dt_inverse,
            0.25f * PI * s.nh * s.qr
                * relative_velocity * mass_geometry);
        r->qhlacrmlr = r->qhlacr;
        r->chlacr = fmin(
            0.1f * s.nr * dt_inverse,
            0.25f * PI * s.nh * s.nr
                * relative_velocity * number_geometry);
        if (s.temperature > 273.15f) {
            r->qhlacr = 0.0f;
            r->chlacr = 0.0f;
        } else {
            r->vhlacr = nssl_div(rho * r->qhlacr, 900.0f);
        }
    }
}

static inline void diagnose_snow_aggregation(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float dt,
    thread Rates* r)
{
    // WRF 16933-16955.  Snow self-collection removes only number; the shared
    // snow-number limiter later rescales csacs together with its other sinks.
    if (!(s.qs > QS_MIN) || !(s.ns > CXMIN)) return;
    const float temperature_c = s.temperature - 273.15f;
    float efficiency = 0.0f;
    if (temperature_c < 0.0f && temperature_c >= -15.0f) {
        if (temperature_c > -15.0f && temperature_c < -10.0f) {
            efficiency = nssl_div(0.5f * exp(-0.5f)
                * (temperature_c + 15.0f), 5.0f);
        } else if (temperature_c >= -10.0f) {
            efficiency = 0.5f
                * exp(0.05f * fmin(temperature_c, 0.0f));
        }
    }
    if (!(efficiency > 0.0f)) return;

    const float mean_volume = fmax(
        SNOW_VMIN,
        rho * s.qs / (particles.snow_density * s.ns));
    const float swept_volume_cap =
        4.0f * (1.0f / PI) / 3.0f * (0.02f * 0.02f * 0.02f);
    const float collected_volume = fmin(mean_volume, swept_volume_cap);
    const float raw = (float)(
        (Nssl64)0.104f * (Nssl64)5.78e3f * (Nssl64)efficiency
        * (Nssl64)(s.ns * s.ns) * (Nssl64)collected_volume);
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    r->csacs = fmin(
        raw,
        (float)(nssl64_bits(0x3fb999999999999aul) * (Nssl64)s.ns * (Nssl64)dt_inverse));
}

static inline float liquid_heat_capacity(float temperature) {
    if (temperature < 273.15f) {
        const float celsius =
            fmin(273.15f, fmax(233.15f, temperature)) - 273.15f;
        const float offset = celsius - 35.0f;
        return 4203.1548f + 1.30572e-2f * offset * offset
            + 1.60056e-5f * offset * offset * offset * offset;
    }
    const float celsius =
        fmin(308.15f, fmax(273.15f, temperature)) - 273.15f;
    return 4243.1688f + 3.47104e-1f * celsius * celsius;
}

static inline void diagnose_melting(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float pressure,
    float layer_depth,
    float dt,
    thread Rates* r)
{
    // WRF 18534-18904.  Melt rates must exist before frozen-vapor diagnosis:
    // the latter suppresses category sublimation when that category melts.
    // WRF's operative melt/ventilation branches use mass thresholds only;
    // retain active low-positive number moments normalized by setvtz.
    if (!(s.temperature > 273.15f)) return;
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrt(1.225f / fmax(0.05f, rho));

    float snow_diameter = 1.0e-9f;
    float snow_ventilation = 0.0f;
    if (s.qs > QS_MIN) {
        const float snow_volume = fmax(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        snow_diameter = pow(snow_volume * (6.0f / PI), 1.0f / 3.0f);
        const float snow_velocity = 11.9495f * density_factor
            * pow(snow_volume, 0.14f);
        const float dynamic_viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * pow(nssl_div(s.temperature, 296.0f), 1.5f);
        const float diffusivity = 2.11e-5f
            * pow(nssl_div(s.temperature, 273.15f), 1.94f)
            * (101325.0f / pressure);
        const float kinematic = dynamic_viscosity / rho;
        const float ventilation_factor = pow(
            kinematic / diffusivity, 1.0f / 3.0f)
            * pow(kinematic, -0.5f);
        snow_ventilation = 0.65f + 0.44f * ventilation_factor
            * sqrt(snow_velocity * snow_diameter);
    }

    float graupel_volume = GRAUPEL_VMIN;
    float graupel_diameter = 1.0e-9f;
    float graupel_characteristic = 1.0e-9f;
    if (s.qg > QD_MIN) {
        graupel_volume = fmax(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        graupel_diameter =
            pow(graupel_volume * (6.0f / PI), 1.0f / 3.0f);
        graupel_characteristic =
            pow(6.0f, -1.0f / 3.0f) * graupel_diameter;
    }
    float hail_volume = HAIL_VMIN;
    float hail_diameter = 1.0e-9f;
    float hail_characteristic = 1.0e-9f;
    if (s.qh > QD_MIN) {
        hail_volume = fmax(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        hail_diameter = pow(hail_volume * (6.0f / PI), 1.0f / 3.0f);
        hail_characteristic =
            pow(24.0f, -1.0f / 3.0f) * hail_diameter;
    }

    const float dynamic_viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * pow(nssl_div(s.temperature, 296.0f), 1.5f);
    const float kinematic = dynamic_viscosity / rho;
    const float diffusivity = 2.11e-5f
        * pow(nssl_div(s.temperature, 273.15f), 1.94f)
        * (101325.0f / pressure);
    const float ventilation_factor = pow(
        kinematic / diffusivity, 1.0f / 3.0f)
        * pow(kinematic, -0.5f);
    const float conductivity =
        nssl_div(2.43e-2f * dynamic_viscosity, 1.718e-5f);
    const float lv = latent_vapor(s.temperature);
    const float lf = latent_fusion(s.temperature);
    const float cw = liquid_heat_capacity(s.temperature);
    const float fmlt1 = 2.0f * PI
        * (lv * diffusivity * (380.0f / pressure - s.qv)
            - conductivity * temperature_c / rho)
        / lf;
    const float fmlt2 = -cw * temperature_c / lf;

    float graupel_ventilation = 0.0f;
    if (s.qg > QD_MIN) {
        const float drag = fmin(
            1.2f,
            fmax(0.45f, 0.45f + nssl_div(0.55f
                * (800.0f - fmin(
                    800.0f, fmax(170.0f, particles.graupel_density))),
                630.0f)));
        graupel_ventilation = 0.78f * nssl_tgamma(2.0f)
            + 0.308f * nssl_tgamma(2.75f)
                * pow(4.0f * 9.8f / (3.0f * drag), 0.25f)
                * ventilation_factor
                * pow(particles.graupel_density / rho, 0.25f)
                * pow(graupel_characteristic, 0.75f);
    }
    float hail_ventilation = 0.0f;
    if (s.qh > QD_MIN) {
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        // setvtz already caps the corresponding hail fall moments by dz/dt;
        // the analytic ventilation is otherwise the Ferrier alpha=1 branch.
        (void)layer_depth;
        hail_ventilation = 1.56f
            + nssl_tgamma(3.5f + 0.5f * exponent)
                * 0.308f * ventilation_factor
                * pow(
                    hail_characteristic,
                    0.5f + 0.5f * exponent)
                * sqrt(coefficient * density_factor);
    }

    if (s.qs > QS_MIN) {
        const float c1sw = nssl_div(nssl_tgamma(0.5333333333333333f)
            * pow(0.2f, -1.0f / 3.0f), nssl_tgamma(0.2f));
        r->qsmlr = fmin(
            c1sw * fmlt1 * s.ns * snow_ventilation * snow_diameter,
            0.0f);
        r->qsmlr = fmax(r->qsmlr, -0.7f * s.qs * dt_inverse);
        r->csmlr = (s.ns / s.qs) * r->qsmlr;
        // Default gamma-shape conversion from snow donor number to rain.
        r->csmlrr = nssl_div(r->csmlr, 0.30f);
    }
    if (s.qg > QD_MIN) {
        const float raw = fmin(
            fmlt1 * s.ng * graupel_ventilation
                * graupel_characteristic
                + fmlt2 * (r->qhacrmlr + r->qhacwmlr),
            0.0f);
        if (raw < 0.0f && particles.graupel_density < 900.0f) {
            const float available =
                (1.0f - nssl_div(particles.graupel_density, 900.0f))
                * (s.vg + rho * raw / particles.graupel_density)
                * dt_inverse;
            const float refrozen = nssl_div(-rho * raw, 900.0f);
            r->vhsoak = fmin(available, refrozen);
        }
        r->qhmlr = fmax(raw, -0.95f * s.qg * dt_inverse);
        r->chmlr = (s.ng / (s.qg + 1.0e-20f)) * r->qhmlr;
        const float maximum_rain_mass =
            1000.0f * 0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
        const float a = -rho * r->qhmlr / fmin(
            maximum_rain_mass,
            particles.graupel_density * graupel_volume);
        const float three_mm_volume =
            0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f);
        const float b = nssl_div(-rho * r->qhmlr, (1000.0f * three_mm_volume));
        const float interpolation = nssl_div(a
                * (20.0e-3f - graupel_diameter), 12.0e-3f)
            + nssl_div(b * (graupel_diameter - 8.0e-3f), 12.0e-3f);
        r->chmlrr = -fmax(a, fmin(b, interpolation));
        r->vhmlr = r->qhmlr;
    }

    if (s.qh > QD_MIN) {
        const float raw = fmin(
            fmlt1 * s.nh * hail_ventilation * hail_characteristic
                + fmlt2 * (r->qhlacrmlr + r->qhlacwmlr),
            0.0f);
        if (raw < 0.0f && particles.hail_density < 900.0f) {
            const float available =
                (1.0f - nssl_div(particles.hail_density, 900.0f))
                * (s.vh + rho * raw / particles.hail_density)
                * dt_inverse;
            const float refrozen = nssl_div(-rho * raw, 900.0f);
            r->vhlsoak = fmin(available, refrozen);
        }
        r->qhlmlr = fmax(raw, -0.95f * s.qh * dt_inverse);
        r->chlmlr = (s.nh / (s.qh + 1.0e-20f)) * r->qhlmlr;
        const float maximum_rain_mass =
            1000.0f * 0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f);
        const float a = -rho * r->qhlmlr / fmin(
            maximum_rain_mass,
            particles.hail_density * hail_volume);
        const float three_mm_volume =
            0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f);
        const float b = nssl_div(-rho * r->qhlmlr, (1000.0f * three_mm_volume));
        const float interpolation = nssl_div(a
                * (20.0e-3f - hail_diameter), 12.0e-3f)
            + nssl_div(b * (hail_diameter - 8.0e-3f), 12.0e-3f);
        r->chlmlrr = -fmax(a, fmin(b, interpolation));
        r->vhlmlr = r->qhlmlr;
    }
}

static inline void diagnose_wet_growth_shedding(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    thread Rates* r)
{
    // WRF 19436-19769.  qhwet/qhlwet and the shedding decision are formed
    // from the original dry collection rates.  Only afterward can wet growth
    // rewrite the ice/snow collection efficiencies and vapor contributors.
    // As in WRF, active dense categories are mass-gated here; setvtz already
    // guarantees their number moments are positive even below CXMIN.
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrt(1.225f / fmax(0.05f, rho));
    const float dynamic_viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * pow(nssl_div(s.temperature, 296.0f), 1.5f);
    const float kinematic = dynamic_viscosity / rho;
    const float diffusivity = 2.11e-5f
        * pow(nssl_div(s.temperature, 273.15f), 1.94f)
        * (101325.0f / pressure);
    const float ventilation_factor = pow(
        kinematic / diffusivity, 1.0f / 3.0f)
        * pow(kinematic, -0.5f);
    const float conductivity =
        nssl_div(2.43e-2f * dynamic_viscosity, 1.718e-5f);
    const float lv = latent_vapor(s.temperature);
    const float lf = latent_fusion(s.temperature);
    const float cw = liquid_heat_capacity(s.temperature);
    const float bounded_ice_celsius =
        fmin(273.15f, fmax(233.15f, s.temperature)) - 273.15f;
    const float ci =
        (2.118636f + 0.007371f * bounded_ice_celsius) * 1.0e3f;
    const float denominator = lf + cw * temperature_c;
    const float fwet1 = 2.0f * PI
        * (lv * diffusivity * rho * (380.0f / pressure - s.qv)
            - conductivity * temperature_c)
        / (rho * denominator);
    const float fwet2 = 1.0f - ci * temperature_c / denominator;

    float graupel_diameter = 1.0e-9f;
    float graupel_characteristic = 1.0e-9f;
    float graupel_ventilation = 0.0f;
    if (s.qg > QD_MIN) {
        const float volume = fmax(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        graupel_diameter = pow(volume * (6.0f / PI), 1.0f / 3.0f);
        graupel_characteristic =
            pow(6.0f, -1.0f / 3.0f) * graupel_diameter;
        const float drag = fmin(
            1.2f,
            fmax(0.45f, 0.45f + nssl_div(0.55f
                * (800.0f - fmin(
                    800.0f, fmax(170.0f, particles.graupel_density))),
                630.0f)));
        graupel_ventilation = 0.78f * nssl_tgamma(2.0f)
            + 0.308f * nssl_tgamma(2.75f)
                * pow(4.0f * 9.8f / (3.0f * drag), 0.25f)
                * ventilation_factor
                * pow(particles.graupel_density / rho, 0.25f)
                * pow(graupel_characteristic, 0.75f);
    }
    float hail_diameter = 1.0e-9f;
    float hail_characteristic = 1.0e-9f;
    float hail_ventilation = 0.0f;
    if (s.qh > QD_MIN) {
        const float volume = fmax(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        hail_diameter = pow(volume * (6.0f / PI), 1.0f / 3.0f);
        hail_characteristic =
            pow(24.0f, -1.0f / 3.0f) * hail_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        hail_ventilation = 1.56f
            + nssl_tgamma(3.5f + 0.5f * exponent)
                * 0.308f * ventilation_factor
                * pow(
                    hail_characteristic,
                    0.5f + 0.5f * exponent)
                * sqrt(coefficient * density_factor);
    }

    const float graupel_dry =
        r->qhaci + r->qhacs + r->qhacr + r->qhacw;
    const float hail_dry =
        r->qhlaci + r->qhlacs + r->qhlacr + r->qhlacw;
    float graupel_wet = graupel_dry;
    float hail_wet = hail_dry;
    if (s.temperature > 243.15f && s.temperature < 273.15f) {
        graupel_wet = fmax(
            graupel_characteristic * graupel_ventilation * s.ng * fwet1
                + fwet2 * (r->qhaci + r->qhacs),
            0.0f);
        hail_wet = fmax(
            hail_characteristic * hail_ventilation * s.nh * fwet1
                + fwet2 * (r->qhlaci + r->qhlacs),
            0.0f);
    }
    r->qhshr = fmin(0.0f, graupel_wet - graupel_dry);
    r->qhlshr = fmin(0.0f, hail_wet - hail_dry);
    if (s.temperature < 243.15f) {
        r->qhshr = 0.0f;
        r->qhlshr = 0.0f;
    }
    if (s.temperature > 273.15f) {
        r->qsshr = -r->qsacr - r->qsacw;
        r->qhshr = -r->qhacw - r->qhacr;
        r->qhlshr = -r->qhlacw - r->qhlacr;
        r->vhshdr = -r->vhacw - r->vhacr;
        r->vhlshdr = -r->vhlacw - r->vhlacr;
        graupel_wet = 0.0f;
        hail_wet = 0.0f;
    }

    // WRF 19606-19615 final snow decision: shedding and vapor exchange are
    // mutually exclusive.  This occurs after frozen-vapor diagnosis, so the
    // already limited snow deposition/sublimation rates are explicitly
    // cleared whenever warm snow collection sheds to rain.
    if (r->qsshr < 0.0f) {
        r->qsdpv = 0.0f;
        r->qssbv = 0.0f;
    } else {
        r->qsshr = 0.0f;
    }

    r->wetgrowth_g =
        r->qhshr < 0.0f && s.temperature < 273.15f ? 1.0f : 0.0f;
    r->wetgrowth_h =
        r->qhlshr < 0.0f && s.temperature < 273.15f ? 1.0f : 0.0f;
    r->wetsurface_g = (r->wetgrowth_g > 0.0f
        || (r->qhmlr < -QD_MIN && s.temperature > 273.15f)) ? 1.0f : 0.0f;
    r->wetsurface_h = (r->wetgrowth_h > 0.0f
        || (r->qhlmlr < -QD_MIN && s.temperature > 273.15f)) ? 1.0f : 0.0f;

    constexpr float MASS_FACTOR_SHED = 4.5f;
    float graupel_shed_volume =
        0.523599f * (1.0e-3f * 1.0e-3f * 1.0e-3f);
    if (s.qg > QD_MIN) {
        const float weighted = 3.0f * graupel_characteristic;
        if (weighted > 20.0e-3f) {
            graupel_shed_volume =
                0.523599f * (1.5e-3f * 1.5e-3f * 1.5e-3f)
                / MASS_FACTOR_SHED;
        } else if (weighted > 8.0e-3f) {
            graupel_shed_volume =
                0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f)
                / MASS_FACTOR_SHED;
        } else {
            graupel_shed_volume = nssl_div(fmin(
                0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f),
                (6.0f / PI) * particles.graupel_density * 0.001f
                    * weighted * weighted * weighted),
                MASS_FACTOR_SHED);
        }
    }
    float hail_shed_volume =
        0.523599f * (1.0e-3f * 1.0e-3f * 1.0e-3f);
    if (s.qh > QD_MIN) {
        const float weighted = 4.0f * hail_characteristic;
        if (weighted > 20.0e-3f) {
            hail_shed_volume =
                0.523599f * (1.5e-3f * 1.5e-3f * 1.5e-3f)
                / MASS_FACTOR_SHED;
        } else if (weighted > 8.0e-3f) {
            hail_shed_volume =
                0.523599f * (3.0e-3f * 3.0e-3f * 3.0e-3f)
                / MASS_FACTOR_SHED;
        } else {
            hail_shed_volume = nssl_div(fmin(
                0.523599f * (6.0e-3f * 6.0e-3f * 6.0e-3f),
                (6.0f / PI) * particles.hail_density * 0.001f
                    * weighted * weighted * weighted),
                MASS_FACTOR_SHED);
        }
    }
    r->chshrr = rho * r->qhshr / (1000.0f * graupel_shed_volume);
    r->chlshrr = rho * r->qhlshr / (1000.0f * hail_shed_volume);
    r->qrshr = r->qsshr + r->qhshr + r->qhlshr;
    // ipconc=5 has rzxh=1 and rzxhl=0.4375 (alphar=alphah=0,
    // alphahl=1, imurain=1, and no predicted reflectivity moments).
    r->crshr = r->chshrr + nssl_div(r->chlshrr, 0.4375f);

    if (r->wetgrowth_g > 0.0f) {
        r->qhdpv = 0.0f;
        r->chdpv = 0.0f;
        r->qhaci = fmin(r->qhaci0, 0.1f * s.qi * dt_inverse);
        r->chaci = fmin(r->chaci0, 0.1f * s.ni * dt_inverse);
        r->qhacs = fmin(r->qhacs0, 0.1f * s.qs * dt_inverse);
        r->chacs = fmin(r->chacs0, 0.1f * s.ns * dt_inverse);
        r->vhacw = nssl_div(rho * r->qhacw, 900.0f);
        r->vhacr = nssl_div(rho * r->qhacr, 900.0f);
        const float available = particles.graupel_density < 900.0f
            ? (1.0f - nssl_div(particles.graupel_density, 900.0f))
                * s.vg * dt_inverse
            : 0.0f;
        const float refrozen = nssl_div(rho * graupel_wet, 900.0f);
        r->vhsoak = fmin(available, refrozen);
        r->vhshdr = fmin(
            0.0f, refrozen - r->vhacw - r->vhacr);
        r->wetsurface_g = 1.0f;
    }
    if (r->wetgrowth_h > 0.0f) {
        r->qhldpv = 0.0f;
        r->chldpv = 0.0f;
        r->qhlaci = fmin(r->qhlaci0, 0.1f * s.qi * dt_inverse);
        r->chlaci = fmin(r->chlaci0, 0.1f * s.ni * dt_inverse);
        r->qhlacs = fmin(r->qhlacs0, 0.1f * s.qs * dt_inverse);
        r->chlacs = fmin(r->chlacs0, 0.1f * s.ns * dt_inverse);
        r->vhlacw = nssl_div(rho * r->qhlacw, 900.0f);
        r->vhlacr = nssl_div(rho * r->qhlacr, 900.0f);
        const float available = particles.hail_density < 900.0f
            ? (1.0f - nssl_div(particles.hail_density, 900.0f))
                * s.vh * dt_inverse
            : 0.0f;
        const float refrozen = nssl_div(rho * hail_wet, 900.0f);
        r->vhlsoak = fmin(available, refrozen);
        r->vhlshdr = fmin(
            0.0f, refrozen - r->vhlacw - r->vhlacr);
        r->wetsurface_h = 1.0f;
    }
}

static inline void diagnose_crystal_to_snow(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    thread Rates* r)
{
    // WRF 19335-19388, default iscni=4.  This conversion is diagnosed after
    // frozen vapor but before wet-growth can rewrite any collection rates.
    if (!(s.qi > QI_MIN) || !(s.ni > CXMIN) || !(r->qidpv > 0.0f)) {
        return;
    }
    const float ice_mass = fmax(rho * s.qi / s.ni, ICE_MMIN);
    const float ice_diameter = 0.1871f * pow(ice_mass, 0.3429f);
    if (!(ice_diameter >= 100.0e-6f)) return;
    const float fraction = fmin(0.5f, nssl_div(ice_diameter, 200.0e-6f));
    r->qscni = fraction * r->qidpv;
    // WRF's fscni multiplier is one here.  The diameter fraction already
    // entered qscni and must not be applied a second time to its number rate.
    r->cscni = rho * r->qscni / fmax(
        particles.snow_density * SNOW_VMIN, ice_mass);
}

static inline void diagnose_ice_snow_to_graupel(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    thread Rates* r)
{
    // WRF 19774-19844 and 20184-20301.  Donor and receiver number rates are
    // intentionally distinct and remain stale if later cloud/snow limiters
    // rescale the collection rates that caused the conversion.
    const float temperature_c = s.temperature - 273.15f;
    const float density_factor = sqrt(1.225f / fmax(0.05f, rho));
    float cloud_diameter = 1.0e-9f;
    if (s.qc > QC_MIN && s.nc > CXMIN) {
        const float cloud_volume = fmin(
            CLOUD_VMAX,
            fmax(CLOUD_VMIN, rho * s.qc / (1000.0f * s.nc)));
        cloud_diameter =
            pow(cloud_volume * (6.0f / PI), 1.0f / 3.0f);
    }
    if (s.temperature < 273.0f && s.qi > QI_MIN && s.ni > CXMIN
            && r->qiacw - r->qidpv > 0.0f) {
        const float ice_mass = fmax(rho * s.qi / s.ni, ICE_MMIN);
        const float ice_volume = nssl_div(ice_mass, 900.0f);
        const float ice_velocity = 47.6273f * density_factor
            * pow(ice_volume, 0.18333f) * 1.091937899589539f;
        float rime_density = 300.0f * pow(
            -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * ice_velocity) / temperature_c,
            0.44f);
        rime_density = fmin(900.0f, fmax(170.0f, rime_density));
        if (rime_density >= 200.0f) {
            const float converted_density = fmax(
                170.0f, 0.5f * (900.0f + rime_density));
            r->qhcni = r->qiacw - r->qidpv;
            r->chcni = s.ni * r->qhcni / s.qi;
            r->chcnih = fmin(
                r->chcni,
                rho * r->qhcni / (converted_density * GRAUPEL_VMIN));
            r->vhcni = rho * r->qhcni / converted_density;
        }
    }

    if (s.temperature < 273.0f && s.qs > QS_MIN && s.ns > CXMIN
            && r->qsacw > 0.0f && r->qsacw - r->qsdpv > 0.0f) {
        const float snow_volume = fmax(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float snow_velocity = 11.9495f * density_factor
            * pow(snow_volume, 0.14f);
        float rime_density = 300.0f * pow(
            // WRF 20227-20231 intentionally uses the cloud diameter in the
            // snow-to-graupel rime-density diagnosis.
            -(0.5f * 1.0e6f * cloud_diameter)
                * (0.60f * snow_velocity) / temperature_c,
            0.44f);
        // Exact snow-conversion quirk: upper clamp only, no 170 lower clamp.
        rime_density = fmin(900.0f, rime_density);
        if (rime_density >= 200.0f) {
            const float converted_density = fmax(
                170.0f,
                0.5f * (particles.snow_density + rime_density));
            r->qhcns = r->qsacw - r->qsdpv;
            r->chcns = s.ns * r->qhcns / s.qs;
            r->chcnsh = fmin(
                r->chcns,
                rho * r->qhcns / (converted_density * GRAUPEL_VMIN));
            r->vhcns = rho * r->qhcns / converted_density;
        }
    }
}

static inline void diagnose_graupel_to_hail(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    thread Rates* r)
{
    // WRF 19847-20089, default ihlcnh=3.  The wet-growth diameter iteration
    // uses the unscaled Milbrandt-Morrison fall-speed law.  Conversion then
    // takes the upper gamma tails above dg0 from the immutable graupel state.
    if (!(s.qg > QD_MIN) || !(s.ng > CXMIN)) return;

    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float temperature_c = s.temperature - 273.15f;
    const bool riming_candidate =
        (r->qhacw + r->qhacr) * dt > QD_MIN
        && s.qg > 0.1e-3f && s.temperature <= 271.15f;
    float dg0 = 0.1501f;

    if ((riming_candidate && s.temperature > 242.0f)
            || (r->wetgrowth_g > 0.0f && s.qg > 0.1e-3f)) {
        float cloud_efficiency = 0.0f;
        float cloud_diameter = 1.0e-9f;
        float cloud_velocity = 0.0f;
        const float dynamic_viscosity = 1.832e-5f
            * (416.16f / (s.temperature + 120.0f))
            * pow(nssl_div(s.temperature, 296.0f), 1.5f);
        if (s.qc > QC_MIN && s.nc > CXMIN) {
            const float cloud_volume = fmin(
                CLOUD_VMAX,
                fmax(CLOUD_VMIN, rho * s.qc / (1000.0f * s.nc)));
            cloud_diameter = pow(
                cloud_volume * (6.0f / PI), 1.0f / 3.0f);
            const float cloud_radius = 0.5f * cloud_diameter;
            cloud_velocity = 2.0f * 9.8f * 1000.0f
                * cloud_radius * cloud_radius / (9.0f * dynamic_viscosity);
            cloud_efficiency = fmin(
                0.9f,
                fmin(
                    -0.27544f + cloud_radius
                        * (0.26249e6f + cloud_radius
                            * (-1.8896e10f
                                + cloud_radius * 4.4626e14f)),
                    1.0f));
            if (cloud_diameter < 2.4e-6f) cloud_efficiency = 0.0f;
        }

        float rain_efficiency = 0.0f;
        float rain_velocity = 0.0f;
        if (s.qr > QR_MIN && s.nr > CXMIN) {
            const float rain_volume = fmin(
                RAIN_VMAX,
                fmax(RAIN_VMIN, rho * s.qr / (1000.0f * s.nr)));
            const float rain_diameter = pow(
                rain_volume * (6.0f / PI), 1.0f / 3.0f);
            const float rain_characteristic =
                pow(6.0f, -1.0f / 3.0f) * rain_diameter;
            const float graupel_volume = fmax(
                GRAUPEL_VMIN,
                rho * s.qg / (particles.graupel_density * s.ng));
            const float graupel_diameter = pow(
                graupel_volume * (6.0f / PI), 1.0f / 3.0f);
            rain_efficiency = fmin(
                1.0f,
                exp(-40.0e-6f / rain_diameter)
                    * exp(-40.0e-6f / graupel_diameter));
            const float density_factor =
                sqrt(1.225f / fmax(0.05f, rho));
            rain_velocity = density_factor * 10.0f
                * (1.0f - pow(
                    1.0f + 516.575f * rain_characteristic, -4.0f));
        }

        float ice_efficiency = 0.0f;
        float ice_velocity = 0.0f;
        if (s.qi > QI_MIN && s.ni > CXMIN) {
            const float ice_mass = fmax(rho * s.qi / s.ni, ICE_MMIN);
            const float ice_volume = nssl_div(ice_mass, 900.0f);
            ice_efficiency = fmin(
                1.0f,
                fmax(0.0f, 0.1f * exp(0.1f * fmin(temperature_c, 0.0f))));
            ice_velocity = 47.6273f
                * sqrt(1.225f / fmax(0.05f, rho))
                * pow(ice_volume, 0.18333f)
                * nssl_tgamma(2.18333f);
        }

        const float initial_denominator =
            1.1e4f * rho
                * (cloud_efficiency * s.qc + rain_efficiency * s.qr)
            - 1.3e3f * rho * s.qi + 1.0f;
        float dwr = 1.0e30f;
        if (initial_denominator > 1.0e-20f) {
            dwr = 0.01f * (
                exp(fmin(70.0f, -temperature_c / initial_denominator))
                - 1.0f);
        }

        float diameter = dwr;
        if (dwr < 0.2f && dwr > 0.0f
                && rho * (s.qc + s.qr) > 1.0e-4f) {
            const float kinematic_viscosity = dynamic_viscosity / rho;
            const float vapor_diffusivity = 2.11e-5f
                * pow(nssl_div(s.temperature, 273.15f), 1.94f)
                * (101325.0f / pressure);
            const float conductivity =
                nssl_div(2.43e-2f * dynamic_viscosity, 1.718e-5f);
            const float thermal_diffusivity =
                conductivity / (1004.0f * rho);
            const float prandtl =
                kinematic_viscosity / thermal_diffusivity;
            const float density_factor =
                sqrt(1.225f / fmax(0.05f, rho));
            const float sqrt_density_factor = sqrt(density_factor);
            const float heat_ventilation = sqrt_density_factor
                * pow(prandtl, 1.0f / 3.0f)
                * pow(kinematic_viscosity, -0.5f);
            const float h1 = -conductivity * temperature_c
                - latent_vapor(s.temperature) * vapor_diffusivity * rho
                    * (s.qv - 380.0f / pressure);
            const float bounded_ice_celsius =
                fmin(273.15f, fmax(233.15f, s.temperature)) - 273.15f;
            const float ice_heat_capacity =
                (2.118636f + 0.007371f * bounded_ice_celsius) * 1.0e3f;
            const float h2 = ice_efficiency * s.qi * rho
                * ice_heat_capacity * temperature_c;
            const float h3 = fmax(0.0f, cloud_efficiency) * s.qc;
            const float h4 = rain_efficiency * s.qr;
            const float heat_denominator =
                latent_fusion(s.temperature)
                + liquid_heat_capacity(s.temperature) * temperature_c;
            float fall_coefficient;
            float fall_exponent;
            dense_fall_coefficients(
                particles.graupel_density,
                &fall_coefficient,
                &fall_exponent);

            for (int iteration = 0; iteration < 10; ++iteration) {
                diameter = fmax(diameter, 1.0e-4f);
                const float previous = diameter;
                const float graupel_velocity = fall_coefficient
                    * pow(diameter, fall_exponent);
                const float ventilation_argument = heat_ventilation
                    * sqrt_density_factor
                    * sqrt(diameter * graupel_velocity);
                const float heat_factor = ventilation_argument > 1.4f
                    ? 0.78f + 0.308f * ventilation_argument
                    : 1.0f + 0.108f * ventilation_argument
                        * ventilation_argument;
                const float denominator = (
                    fmax(0.001f, graupel_velocity - cloud_velocity) * h3
                    + fmax(0.001f, graupel_velocity - rain_velocity) * h4)
                        * rho * heat_denominator
                    + fmax(0.001f, graupel_velocity - ice_velocity) * h2;
                diameter = 8.0f * heat_factor * h1 / denominator;
                if (fabs(previous - diameter) / previous < 0.05f
                        || (iteration >= 3 && diameter > 0.15f)) {
                    break;
                }
            }
        }
        dg0 = fmin(15.0e-3f, fmax(diameter, 5.0e-3f));
    } else if (s.qg > 0.1e-3f && s.temperature <= 271.15f) {
        dg0 = 15.0e-3f;
    }

    // ihlcnh=3 retains a secondary large-graupel route even when the wet
    // growth iteration itself did not return a usable diameter.
    if (riming_candidate) dg0 = fmin(dg0, 0.1499f);
    const bool wet_growth_test = dg0 > 0.0f && dg0 < 0.15f;
    if (!(wet_growth_test && r->qhacw * dt > QD_MIN
            && s.temperature < 271.15f && s.qg > 0.1e-3f)) {
        return;
    }

    const float mean_volume = fmax(
        GRAUPEL_VMIN,
        rho * s.qg / (particles.graupel_density * s.ng));
    const float mean_diameter = pow(
        mean_volume * (6.0f / PI), 1.0f / 3.0f);
    const float characteristic =
        pow(6.0f, -1.0f / 3.0f) * mean_diameter;
    float number_fraction;
    float mass_fraction;
    rain_tail_fractions(
        fmin(100.0f, dg0 / characteristic),
        &number_fraction,
        &mass_fraction);
    const float converted_mass = s.qg * mass_fraction;
    r->qhlcnh = converted_mass * dt_inverse;
    if (!(converted_mass > 10.0f * QD_MIN)) {
        r->qhlcnh = 0.0f;
        return;
    }
    r->chlcnh = s.ng * number_fraction * dt_inverse;
    r->chlcnhhl = r->chlcnh;
    r->vhlcnh = rho * r->qhlcnh / particles.graupel_density;
    r->vhlcnhl = rho * r->qhlcnh
        / fmax(500.0f, particles.graupel_density);
}

static inline void diagnose_hallett_mossop(
    thread const State& s,
    float rho,
    thread Rates* r)
{
    // WRF 20445-20639, default itype1=0/itype2=2.  H-M reads the exact
    // source-ordered wet-surface flags produced by the shedding decision.
    if (!(s.qc > QC_MIN) || !(s.nc > CXMIN)
            || s.temperature < 265.15f || s.temperature > 271.15f) {
        return;
    }
    const float cloud_volume = fmin(
        CLOUD_VMAX,
        fmax(CLOUD_VMIN, rho * s.qc / (1000.0f * s.nc)));
    const float tail = nssl_div(exp(-7.23e-15f / cloud_volume), 250.0f);
    const float temperature_c = s.temperature - 273.15f;
    const float temperature_factor = fmin(
        1.0f,
        fmax(0.0f,
            -0.11f * temperature_c * temperature_c
                - 1.1f * temperature_c - 1.7f));
    constexpr float SPLINTER_MASS = 6.62e-11f;
    if (s.qg > QD_MIN && !(r->wetsurface_g > 0.0f)) {
        r->chmul1 = temperature_factor * tail * r->chacw;
        r->qhmul1 = SPLINTER_MASS * r->chmul1 / rho;
    }
    if (s.qh > QD_MIN && !(r->wetsurface_h > 0.0f)) {
        r->chlmul1 = temperature_factor * tail * r->chlacw;
        r->qhlmul1 = SPLINTER_MASS * r->chlmul1 / rho;
    }
}

static inline void diagnose_primary_ice(
    thread const State& s,
    float rho,
    float pressure,
    float velocity,
    float layer_depth,
    float target_minus,
    float target_plus,
    float vertical_span,
    float dt,
    thread Rates* r)
{
    // WRF 20708-20823, default icenucopt=1 and column-crystal routing.
    if (!(s.temperature < 268.15f) || !(s.ni < 1.0e6f)
            || !(velocity > 0.0f) || !(layer_depth > 0.0f)
            || !(vertical_span > 0.0f)) {
        return;
    }
    int table_index = (int)(nssl_div((s.temperature - 163.15f), 0.002f) + 1.5f);
    table_index = min(1000001, max(1, table_index));
    const float table_temperature = nssl_add(
        163.15f, nssl_mul((float)(table_index - 1), 0.002f));
    const float ice_saturation = (380.0f / pressure) * exp(
        21.87455f * (table_temperature - 273.15f)
        / (table_temperature - 7.66f));
    if (!(s.qv / ice_saturation > 1.0f)) return;
    const float target_gradient = fmax(target_plus - target_minus, 0.0f);
    if (!(target_gradient > 0.0f)) return;

    const float lv = latent_vapor(s.temperature);
    const float feedback = nssl_div(lv * lv, (1004.0f * 461.5f));
    const float vapor_limit = 0.25f * fmax(
        (s.qv - ice_saturation)
            / (1.0f + feedback * ice_saturation
                / (s.temperature * s.temperature)),
        0.0f);
    constexpr float initial_mass = 6.88e-13f;
    r->qiint = (initial_mass / rho) * velocity * target_gradient
        / (layer_depth * vertical_span);
    r->qiint = fmin(r->qiint, vapor_limit);
    r->ciint = nssl_div(r->qiint * rho, initial_mass);
    r->ciint = fmin(
        r->ciint, fmax(0.0f, 1.0e6f - s.ni) / dt);
    r->qiint = r->ciint * initial_mass / rho;
}

static inline void diagnose_frozen_vapor(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float pressure,
    float exner,
    float dt,
    thread Rates* r)
{
    // WRF 18915-19333.  All four frozen categories share one two-pass
    // saturation-adjustment budget.  The resulting mass rates are scaled
    // together before their number companions are derived.
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    // Default iqis0=2 caps the initial ice table lookup above 273.65 K.
    // The test-adjustment's pass-one trial lookup below intentionally does
    // not use this cap, matching WRF's qisstmp update.
    const float ice_saturation = ice_saturation_mixing_ratio(
        s.temperature <= 273.65f ? s.temperature : 273.15f,
        pressure);
    const float lv = latent_vapor(s.temperature);
    const float lf = latent_fusion(s.temperature);
    const float ls = lv + lf;
    const float diffusivity = 2.11e-5f
        * pow(nssl_div(s.temperature, 273.15f), 1.94f)
        * (101325.0f / pressure);
    const float viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * pow(nssl_div(s.temperature, 296.0f), 1.5f);
    const float kinematic = viscosity / rho;
    const float conductivity = nssl_div(2.43e-2f * viscosity, 1.718e-5f);
    const float schmidt = kinematic / diffusivity;
    const float ventilation_factor = pow(schmidt, 1.0f / 3.0f)
        * pow(kinematic, -0.5f);
    const float resistance = ls * ls
        / (conductivity * 461.5f * s.temperature * s.temperature)
        + 1.0f / (rho * diffusivity * ice_saturation);
    const float vapor_growth = (4.0f * PI / rho)
        * (s.qv / ice_saturation - 1.0f) / resistance;

    float raw_ice = 0.0f;
    if (s.qi > QI_MIN && s.ni > 0.0f) {
        const float mean_mass = fmax(rho * s.qi / s.ni, ICE_MMIN);
        const float diameter = 0.1871f * pow(mean_mass, 0.3429f);
        const float reynolds =
            (1.258e4f * pow(diameter, 2.331f)
             + 5.662e4f * pow(diameter, 2.373f))
            / (0.8241f * pow(diameter, -0.042f) + 1.70f);
        const float argument = pow(schmidt, 1.0f / 3.0f)
            * sqrt(reynolds / kinematic);
        const float ventilation = argument < 1.0f
            ? 1.0f + 0.14f * argument * argument
            : 0.86f + 0.28f * argument;
        const float length = 0.4764f * pow(diameter, 0.958f);
        const float eccentricity = fmin(0.99f, sqrt(fmax(
            0.0f, 1.0f - length * length / (diameter * diameter))));
        const float capacitance = diameter * eccentricity
            / log(fabs((1.0f + eccentricity) / (1.0f - eccentricity)));
        raw_ice = vapor_growth * s.ni * ventilation * capacitance;
    }

    float raw_snow = 0.0f;
    if (s.qs > QS_MIN && s.ns > 0.0f) {
        const float mean_volume = fmax(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float diameter = pow(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float fall_speed = 11.9495f
            * sqrt(1.225f / fmax(0.05f, rho))
            * pow(mean_volume, 0.14f);
        const float ventilation = 0.65f + 0.44f * ventilation_factor
            * sqrt(fall_speed * diameter);
        raw_snow = vapor_growth * s.ns * ventilation * (0.5f * diameter);
    }

    float raw_graupel = 0.0f;
    if (s.qg > QD_MIN && s.ng > 0.0f) {
        const float mean_volume = fmax(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        const float mean_volume_diameter =
            pow(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = pow(6.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        const float drag = fmax(0.45f, fmin(
            1.2f,
            0.45f + nssl_div(0.55f
                * (800.0f - fmax(
                    170.0f, fmin(800.0f, particles.graupel_density))),
                630.0f)));
        const float drag_factor = pow(4.0f * 9.8f / (3.0f * drag), 0.25f);
        const float ventilation = 0.78f
            + 0.308f * 1.6083594560623169f * drag_factor
                * ventilation_factor
                * pow(particles.graupel_density / rho, 0.25f)
                * pow(diameter, 0.75f);
        raw_graupel = vapor_growth * s.ng * ventilation * (0.5f * diameter);
    }

    float raw_hail = 0.0f;
    if (s.qh > QD_MIN && s.nh > 0.0f) {
        const float mean_volume = fmax(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        const float mean_volume_diameter =
            pow(mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = pow(24.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        const float density_table[9] = {
            50.0f, 150.0f, 250.0f, 350.0f, 450.0f,
            550.0f, 650.0f, 750.0f, 850.0f};
        const float fall_a_table[9] = {
            62.923f, 94.122f, 114.74f, 131.21f, 145.26f,
            157.71f, 168.98f, 179.36f, 189.02f};
        const float fall_b_table[9] = {
            0.67819f, 0.63789f, 0.62197f, 0.61240f, 0.60572f,
            0.60066f, 0.59663f, 0.59330f, 0.59048f};
        int table = (int)(nssl_div((particles.hail_density - 50.0f), 100.0f)) + 1;
        table = min(9, max(1, table)) - 1;
        const float fraction = fmax(
            0.0f, 0.01f * (particles.hail_density - density_table[table]));
        float fall_a = fall_a_table[table];
        float fall_b = fall_b_table[table];
        if (table < 8) {
            fall_a += fraction * (fall_a_table[table + 1] - fall_a);
            fall_b += fraction * (fall_b_table[table + 1] - fall_b);
        }
        const float fall_factor = sqrt(1.225f / fmax(0.05f, rho));
        const float ventilation = 1.56f
            + nssl_tgamma(3.5f + 0.5f * fall_b) * 0.308f
                * ventilation_factor
                * pow(diameter, 0.5f + 0.5f * fall_b)
                * sqrt(fall_a * fall_factor);
        raw_hail = vapor_growth * s.nh * ventilation * (0.5f * diameter);
    }

    float maximum_deposition = 0.0f;
    float maximum_sublimation = 0.0f;
    const float total_frozen = s.qi + s.qs + s.qg + s.qh;
    if (total_frozen > QI_MIN) {
        float trial_frozen = total_frozen;
        float trial_vapor = s.qv;
        float trial_theta = s.theta;
        float trial_saturation = ice_saturation;
        float saturation_feedback = 5807.6953f * (nssl_div(ls, 1004.0f));
        float denominator_temperature =
            (s.temperature - 7.66f) * (s.temperature - 7.66f);
        if (s.temperature >= 273.15f) {
            saturation_feedback = 4098.0258f * (nssl_div(lv, 1004.0f));
            denominator_temperature =
                (s.temperature - 35.86f) * (s.temperature - 35.86f);
        }
        #pragma unroll
        for (int iteration = 0; iteration < 2; ++iteration) {
            const float difference = trial_vapor - trial_saturation;
            float adjustment;
            if (difference < 0.0f) {
                adjustment = fmax(difference, -trial_frozen);
            } else {
                adjustment = difference
                    / (1.0f + saturation_feedback * trial_saturation
                        / denominator_temperature);
            }
            trial_vapor -= adjustment;
            trial_frozen += adjustment;
            trial_theta += (ls / (1004.0f * exner)) * adjustment;
            if (iteration == 0) {
                trial_vapor = fmax(trial_vapor, 0.0f);
                trial_frozen = fmax(trial_frozen, 0.0f);
                trial_saturation = ice_saturation_mixing_ratio(
                    trial_theta * exner, pressure);
            }
        }
        trial_frozen = fmax(trial_frozen, 0.0f);
        const float net_adjustment = trial_frozen - total_frozen;
        maximum_deposition = fmax(net_adjustment * dt_inverse, 0.0f);
        maximum_sublimation = fmax(-net_adjustment * dt_inverse, 0.0f);
    }

    r->qidpv = fmax(raw_ice, 0.0f);
    r->qsdpv = fmax(raw_snow, 0.0f);
    r->qhdpv = fmax(raw_graupel, 0.0f);
    r->qhldpv = fmax(raw_hail, 0.0f);
    r->qisbv = fmax(
        fmin(raw_ice, 0.0f),
        fmin(-0.1f * s.qi * dt_inverse, -0.5f * s.qi * dt_inverse));
    r->qssbv = (s.temperature < 273.15f || !(r->qsmlr < 0.0f))
        ? fmax(
            fmin(raw_snow, 0.0f),
            fmin(-0.1f * s.qs * dt_inverse, -0.5f * s.qs * dt_inverse))
        : 0.0f;
    r->qhsbv = (s.temperature < 273.15f || !(r->qhmlr < 0.0f))
        ? fmax(fmin(raw_graupel, 0.0f), -0.1f * s.qg * dt_inverse)
        : 0.0f;
    r->qhlsbv = (s.temperature < 273.15f || !(r->qhlmlr < 0.0f))
        ? fmax(fmin(raw_hail, 0.0f), -0.1f * s.qh * dt_inverse)
        : 0.0f;

    const float deposition_total =
        r->qidpv + r->qsdpv + r->qhdpv + r->qhldpv;
    if (deposition_total > maximum_deposition) {
        const float fraction = maximum_deposition / deposition_total;
        r->qidpv *= fraction;
        r->qsdpv *= fraction;
        r->qhdpv *= fraction;
        r->qhldpv *= fraction;
    }
    const float sublimation_total =
        r->qisbv + r->qssbv + r->qhsbv + r->qhlsbv;
    if (sublimation_total < -maximum_sublimation) {
        const float fraction = -maximum_sublimation / sublimation_total;
        r->qisbv *= fraction;
        r->qssbv *= fraction;
        r->qhsbv *= fraction;
        r->qhlsbv *= fraction;
    }

    r->cidpv = 0.0f;
    r->csdpv = 0.0f;
    r->chdpv = 0.0f;
    r->chldpv = 0.0f;
    r->cisbv = (s.ni / (s.qi + 1.0e-20f)) * r->qisbv;
    r->cssbv = (s.ns / (s.qs + 1.0e-20f)) * r->qssbv;
    r->chsbv = (s.ng / (s.qg + 1.0e-20f)) * r->qhsbv;
    r->chlsbv = (s.nh / (s.qh + 1.0e-20f)) * r->qhlsbv;
}

static inline void diagnose_melting_vapor_exchange(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float pressure,
    float dt,
    thread Rates* r)
{
    // WRF 19214-19275, default evapfac=1/mixedphase=.false.  Once a
    // frozen category is melting, vapor exchange occurs against the liquid
    // coating rather than through that category's ice-saturation tendency.
    // The source leaves all three number companions identically zero.
    if (!(r->qsmlr < 0.0f || r->qhmlr < 0.0f || r->qhlmlr < 0.0f)) {
        return;
    }

    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float dynamic_viscosity = 1.832e-5f
        * (416.16f / (s.temperature + 120.0f))
        * pow(nssl_div(s.temperature, 296.0f), 1.5f);
    const float diffusivity = 2.11e-5f
        * pow(nssl_div(s.temperature, 273.15f), 1.94f)
        * (101325.0f / pressure);
    const float kinematic = dynamic_viscosity / rho;
    const float ventilation_factor = pow(
        kinematic / diffusivity, 1.0f / 3.0f)
        * pow(kinematic, -0.5f);
    const float conductivity =
        nssl_div(2.43e-2f * dynamic_viscosity, 1.718e-5f);
    const float lv = latent_vapor(s.temperature);
    const float liquid_saturation = liquid_saturation_mixing_ratio(
        s.temperature, pressure);
    // qss0 is WRF's 0 C liquid saturation value, not qvs(T).
    const float qss0 = 380.0f / pressure;
    const float fav = lv * lv
        / (conductivity * 461.5f * s.temperature * s.temperature);
    const float fbv = 1.0f / (rho * diffusivity * liquid_saturation);
    const float denominator = qss0 * (fav + fbv);
    const float vapor_difference = s.qv - qss0;
    const float density_factor = sqrt(1.225f / fmax(0.05f, rho));

    if (r->qsmlr < 0.0f && s.qs > QS_MIN && s.ns > 0.0f) {
        const float mean_volume = fmax(
            SNOW_VMIN,
            rho * s.qs / (particles.snow_density * s.ns));
        const float diameter = pow(
            mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float fall_speed = 11.9495f * density_factor
            * pow(mean_volume, 0.14f);
        const float ventilation = 0.65f + 0.44f * ventilation_factor
            * sqrt(fall_speed * diameter);
        const float raw = 4.0f * PI * vapor_difference * s.ns
            * (0.5f * diameter) * ventilation / denominator;
        r->qscev = fmax(
            fmin(0.0f, raw),
            fmin(
                -0.1f * s.qs * dt_inverse,
                -0.5f * s.qs * dt_inverse));
    }

    if (r->qhmlr < 0.0f && s.qg > QD_MIN && s.ng > 0.0f) {
        const float mean_volume = fmax(
            GRAUPEL_VMIN,
            rho * s.qg / (particles.graupel_density * s.ng));
        const float mean_volume_diameter = pow(
            mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = pow(6.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        const float drag = fmax(0.45f, fmin(
            1.2f,
            0.45f + nssl_div(0.55f
                * (800.0f - fmax(
                    170.0f, fmin(800.0f, particles.graupel_density))),
                630.0f)));
        const float ventilation = 0.78f
            + 0.308f * 1.6083594560623169f
                * pow(4.0f * 9.8f / (3.0f * drag), 0.25f)
                * ventilation_factor
                * pow(particles.graupel_density / rho, 0.25f)
                * pow(diameter, 0.75f);
        float raw = 2.0f * PI * vapor_difference * s.ng
            * diameter * ventilation / denominator;
        raw = fmax(raw, -0.1f * s.qg * dt_inverse);
        if (s.temperature > 273.15f) raw = fmin(0.0f, raw);
        r->qhcev = raw;
    }

    if (r->qhlmlr < 0.0f && s.qh > QD_MIN && s.nh > 0.0f) {
        const float mean_volume = fmax(
            HAIL_VMIN,
            rho * s.qh / (particles.hail_density * s.nh));
        const float mean_volume_diameter = pow(
            mean_volume * (6.0f / PI), 1.0f / 3.0f);
        const float diameter = pow(24.0f, -1.0f / 3.0f)
            * mean_volume_diameter;
        float coefficient, exponent;
        dense_fall_coefficients(
            particles.hail_density, &coefficient, &exponent);
        const float ventilation = 1.56f
            + nssl_tgamma(3.5f + 0.5f * exponent)
                * 0.308f * ventilation_factor
                * pow(diameter, 0.5f + 0.5f * exponent)
                * sqrt(coefficient * density_factor);
        float raw = 2.0f * PI * vapor_difference * s.nh
            * diameter * ventilation / denominator;
        raw = fmax(raw, -0.1f * s.qh * dt_inverse);
        if (s.temperature > 273.15f) raw = fmin(0.0f, raw);
        r->qhlcev = raw;
    }
}

static inline void assemble_and_limit(
    thread const State& s,
    thread const ParticleProperties& particles,
    float rho,
    float dt,
    thread Rates* r,
    thread Aggregates* a)
{
    const float dt_inverse = (float)(1.0 / (Nssl64)dt);
    const float cold = s.temperature < 273.15f ? 1.0f : 0.0f;
    const float warm = 1.0f - cold;

    // Number aggregates are formed and limited before mass aggregates.
    // WRF 20908-20970 forms cloud-ice number before cloud-number scaling.
    a->pni_i = r->ciint + cold * (r->cwfrzc + r->cwctfzc)
        + r->chmul1 + r->chlmul1 + r->csplinter + r->csplinter2
        + r->csmul;
    a->pni_d = cold * (-r->cscni - r->cscnvi - r->craci - r->csaci
        - r->chaci - r->chlaci - r->chcni + r->cisbv)
        - warm * r->cimlr;

    // WRF 20979-21078: shared cloud-number donor limiter.
    a->pnc_i = -r->cwshw;
    a->pnc_d = -r->cautn
        + cold * (-r->ciacw - r->cwfrz - r->cwctfzp - r->cwctfzc)
        - r->cracw - r->csacw - r->chacw - r->chlacw;
    if (-a->pnc_d * dt > s.nc) {
        const float fraction = -s.nc / (a->pnc_d * dt);
        a->pnc_d = -s.nc * dt_inverse;
        r->cautn *= fraction;
        r->ciacw *= fraction;
        r->cwfrz *= fraction;
        r->cwfrzp *= fraction;
        r->cwfrzc *= fraction;
        r->cwctfzp *= fraction;
        r->cwctfzc *= fraction;
        r->cracw *= fraction;
        r->csacw *= fraction;
        r->chacw *= fraction;
        r->chlacw *= fraction;
        // Exact source-order quirk: this correction reads the already scaled
        // receiving rates rather than rebuilding pni_i.
        a->pni_i -= (1.0f - fraction) * cold
            * (r->cwfrzc + r->cwctfzc);
    }

    // WRF 21089-21166: shared rain-number donor limiter.
    a->pnr_i = r->crcnw - r->crshr
        + warm * (-r->chmlrr - nssl_div(r->chlmlrr, 0.4375f)
            - r->csmlrr - r->cimlr);
    a->pnr_d = cold * (-r->ciacr - r->crfrz) - r->chacr - r->chlacr
        + r->crcev - r->cracr;
    if (-a->pnr_d * dt > s.nr) {
        const float fraction = -s.nr / (a->pnr_d * dt);
        a->pnr_d = -s.nr * dt_inverse;
        r->ciacr *= fraction;
        r->ciacrf *= fraction;
        r->ciacrs *= fraction;
        r->crfrz *= fraction;
        r->crfrzf *= fraction;
        r->crfrzs *= fraction;
        r->chacr *= fraction;
        r->chlacr *= fraction;
        r->crcev *= fraction;
        r->cracr *= fraction;
    }

    // WRF 21173-21241: limit snow-number donors before crediting the routed
    // frozen-small-rain source (default ifrzs=1).
    a->pns_i = cold * (r->cscni + r->cscnvi) + r->cscnh;
    a->pns_d = -r->chacs - r->chlacs - r->chcns
        + warm * r->csmlr + r->csshr + r->cssbv - r->csacs;
    if (s.ns + dt * (a->pns_i + a->pns_d) < 0.0f) {
        const float fraction = (-s.ns + a->pns_i * dt)
            / (a->pns_d * dt);
        a->pns_d *= fraction;
        r->chacs *= fraction;
        r->chlacs *= fraction;
        r->chcns *= fraction;
        r->csmlr *= fraction;
        r->csshr *= fraction;
        r->cssbv *= fraction;
        r->csacs *= fraction;
    }
    a->pns_i += r->crfrzs + r->ciacrs;

    // WRF 21243-21260 has no shared graupel-number donor limiter.
    a->png_i = r->crfrzf + cold * r->ciacrf + r->chcnsh + r->chcnih
        + r->chcnhl;
    a->png_d = warm * r->chmlr + r->chsbv
        - cold * r->chlcnh - r->cscnh;

    // WRF 21264-21297: hail-number limiter.  png_i remains stale if this
    // scales chcnhl, exactly as in the source.
    a->pnh_i = 0.4375f * r->chlcnhhl;
    a->pnh_d = warm * r->chlmlr + r->chlsbv - r->chcnhl;
    if (s.nh + dt * (a->pnh_i + a->pnh_d) < 0.0f) {
        const float fraction = (-s.nh + a->pnh_i * dt)
            / (a->pnh_d * dt);
        a->pnh_d *= fraction;
        r->chlmlr *= fraction;
        r->chlsbv *= fraction;
        r->chcnhl *= fraction;
    }

    // Initial vapor sum (WRF 21401-21450).  The rain-mass limiter below owns
    // the peculiar conditional patch/re-sum semantics.
    a->pqv_i = -fmin(0.0f, r->qrcev) - fmin(0.0f, r->qhcev)
        - fmin(0.0f, r->qhlcev) - fmin(0.0f, r->qscev)
        - r->qhsbv - r->qhlsbv - r->qssbv - cold * r->qisbv;
    a->pqv_d = -fmax(0.0f, r->qrcev) - fmax(0.0f, r->qhcev)
        - fmax(0.0f, r->qhlcev) - fmax(0.0f, r->qscev)
        + cold * (-r->qiint - r->qhdpv - r->qsdpv - r->qhldpv)
        - cold * r->qidpv;

    // WRF 21454-21500: shared cloud-water mass donor limiter.
    a->pqc_i = r->qwcnr - r->qwshw;
    a->pqc_d = cold * (-r->qiacw - r->qwfrz - r->qwctfz - r->qiihr)
        - r->qracw - r->qsacw - r->qrcnw - r->qhacw - r->qhlacw;
    if (a->pqc_d < 0.0f && -a->pqc_d * dt > s.qc) {
        const float fraction = -fmax(0.0f, s.qc) / (a->pqc_d * dt);
        // Exact WRF quirk: pin the aggregate; do not recompute it from the
        // scaled named rates (qiihr is not scaled).
        a->pqc_d = -s.qc * dt_inverse;
        r->qiacw *= fraction;
        r->qwfrz *= fraction;
        r->qwfrzp *= fraction;
        r->qwfrzc *= fraction;
        r->qwctfz *= fraction;
        r->qwctfzc *= fraction;
        r->qracw *= fraction;
        r->qsacw *= fraction;
        r->qhacw *= fraction;
        r->vhacw *= fraction;
        r->qrcnw *= fraction;
        r->qhlacw *= fraction;
        r->vhlacw *= fraction;
    }

    // Cloud-ice mass is formed after the cloud donor scaling.
    a->pqi_i = cold * (r->qiint + r->qwfrzc + r->qwctfzc)
        + r->qhmul1 + r->qhlmul1
        + r->qsplinter + r->qsplinter2 + r->qsmul
        + cold * (r->qidpv + r->qiacw);
    a->pqi_d = cold * (-r->qscni - r->qscnvi - r->qraci - r->qsaci)
        - r->qhaci - r->qhlaci + cold * r->qisbv
        + warm * r->qimlr - r->qhcni;

    // WRF 21572-21678: shared rain-water donor limiter.  Vapor aggregates
    // are patched first, then conditionally fully rebuilt after scaling.
    a->pqr_i = r->qracw + r->qrcnw + fmax(0.0f, r->qrcev)
        + warm * (-r->qhmlr - r->qsmlr - r->qhlmlr - r->qimlr)
        - r->qrshr;
    a->pqr_d = cold * (-r->qiacr - r->qrfrz) - r->qsacr - r->qhacr
        - r->qhlacr - r->qwcnr + fmin(0.0f, r->qrcev);
    if (a->pqr_d < 0.0f
            && -(a->pqr_d + a->pqr_i) * dt > s.qr) {
        const float fraction =
            (-s.qr + a->pqr_i * dt) / (a->pqr_d * dt);
        const float old_qrcev = r->qrcev;
        a->pqv_i += fmin(0.0f, old_qrcev)
            - fraction * fmin(0.0f, old_qrcev);
        a->pqv_d += fmax(0.0f, old_qrcev)
            - fraction * fmax(0.0f, old_qrcev);
        r->qiacr *= fraction;
        r->qiacrf *= fraction;
        r->qiacrs *= fraction;
        r->viacrf *= fraction;
        r->qrfrz *= fraction;
        r->qrfrzs *= fraction;
        r->qrfrzf *= fraction;
        r->vrfrzf *= fraction;
        r->qsacr *= fraction;
        r->qhacr *= fraction;
        r->vhacr *= fraction;
        r->qhlacr *= fraction;
        r->vhlacr *= fraction;
        r->qrcev *= fraction;
        r->qhcev *= fraction;
        r->qhlcev *= fraction;

        // WRF's post-limit expression moves qsacr inside the cold gate.
        a->pqr_d = cold * (-r->qiacr - r->qrfrz - r->qsacr)
            - r->qhacr - r->qhlacr - r->qwcnr
            + fmin(0.0f, r->qrcev);
        if (r->qrcev != 0.0f) {
            a->pqv_i = -fmin(0.0f, r->qrcev)
                - fmin(0.0f, r->qhcev) - fmin(0.0f, r->qhlcev)
                - fmin(0.0f, r->qscev) - r->qhsbv - r->qhlsbv
                - r->qssbv - cold * r->qisbv;
            a->pqv_d = -fmax(0.0f, r->qrcev)
                - fmax(0.0f, r->qhcev) - fmax(0.0f, r->qhlcev)
                - fmax(0.0f, r->qscev)
                + cold * (-r->qiint - r->qhdpv - r->qsdpv - r->qhldpv)
                - cold * r->qidpv;
        }
    }

    // WRF 21687-21741 snow mass aggregate and donor limiter.  This limiter
    // does not rebuild vapor after scaling qssbv/negative qscev.
    a->pqs_i = cold * (r->qscni + r->qsaci + r->qsdpv + r->qscnvi
        + r->qiacrs + r->qrfrzs + r->qsacr) + fmax(0.0f, r->qscev)
        + r->qsacw + r->qscnh;
    a->pqs_d = -r->qhacs - r->qhlacs - r->qhcns + warm * r->qsmlr
        + r->qsshr + r->qssbv + fmin(0.0f, r->qscev) - r->qsmul;
    if (a->pqs_d < 0.0f
            && s.qs + dt * (a->pqs_i + a->pqs_d) < 0.0f) {
        const float fraction = (-s.qs + a->pqs_i * dt)
            / (a->pqs_d * dt);
        a->pqs_d *= fraction;
        r->qhacs *= fraction;
        r->qhlacs *= fraction;
        r->qhcns *= fraction;
        r->qsmlr *= fraction;
        r->qsshr *= fraction;
        r->qssbv *= fraction;
        r->qsmul *= fraction;
        if (r->qscev < 0.0f) r->qscev *= fraction;
    }

    // WRF 21743-21765 graupel has no shared mass donor limiter.
    a->pqg_i = cold * (r->qrfrzf + r->qiacrf + r->qracif + r->qhdpv)
        + fmax(0.0f, r->qhcev) + r->qhacr + r->qhacw + r->qhacs
        + r->qhaci + r->qhcns + r->qhcni + r->qhcnhl;
    a->pqg_d = r->qhshr + warm * r->qhmlr + r->qhsbv
        + fmin(0.0f, r->qhcev) - r->qhmul1 - r->qhlcnh
        - r->qscnh - r->qsplinter - r->qsplinter2;

    // WRF 21768-21809 hail mass limiter.  Earlier graupel/ice/rain/vapor
    // aggregates intentionally remain stale when this rescales members.
    a->pqh_i = cold * r->qhldpv + fmax(0.0f, r->qhlcev)
        + r->qhlacr + r->qhlacw + r->qhlacs + r->qhlaci + r->qhlcnh;
    a->pqh_d = r->qhlshr + warm * r->qhlmlr + r->qhlsbv
        + fmin(0.0f, r->qhlcev) - r->qhlmul1 - r->qhcnhl;
    if (s.qh + dt * (a->pqh_i + a->pqh_d) < 0.0f) {
        const float fraction = (-s.qh + a->pqh_i * dt)
            / (a->pqh_d * dt);
        a->pqh_d *= fraction;
        r->qhlmlr *= fraction;
        r->qhlsbv *= fraction;
        r->qhcnhl *= fraction;
        r->qhlmul1 *= fraction;
        if (r->qhlcev < 0.0f) r->qhlcev *= fraction;
    }

    // WRF 22493-22671.  Collected ice/snow volume routes are not temperature
    // gated; only dense deposition and frozen-rain routes carry il5.  Liquid
    // coating condensation is credited at liquid-water density.
    a->pvg_i = rho * cold * (
            nssl_div(r->qhdpv, 170.0f) + nssl_div(r->qracif, 900.0f))
        + nssl_div(rho * (r->qhaci + r->qhacs), 170.0f)
        + nssl_div(rho * fmax(0.0f, r->qhcev), 1000.0f)
        + r->vhacw + r->vhacr + r->vhcni + r->vhcns
        + r->viacrf + r->vrfrzf;
    a->pvg_d = rho * (
            warm * r->vhmlr + r->qhsbv + fmin(0.0f, r->qhcev)
            - r->qhmul1) / particles.graupel_density
        - r->vhlcnh + r->vhshdr - r->vhsoak - r->vscnh;
    a->pvh_i = nssl_div(rho * cold * r->qhldpv, 500.0f)
        + nssl_div(rho * (r->qhlaci + r->qhlacs), 500.0f)
        + nssl_div(rho * fmax(0.0f, r->qhlcev), 1000.0f)
        + r->vhlacw + r->vhlacr + r->vhlcnhl;
    a->pvh_d = rho * (
            warm * r->vhlmlr + r->qhlsbv + fmin(0.0f, r->qhlcev)
            - r->qhlmul1) / particles.hail_density
        + r->vhlshdr - r->vhlsoak;
}

static inline void aggregate_once(
    thread State* s, thread const Rates& r, thread const Aggregates& a,
    float rho, float exner, float dt)
{
    s->qv += dt * (a.pqv_i + a.pqv_d);
    s->qc += dt * (a.pqc_i + a.pqc_d);
    s->qr += dt * (a.pqr_i + a.pqr_d);
    s->qi += dt * (a.pqi_i + a.pqi_d);
    s->qs += dt * (a.pqs_i + a.pqs_d);
    s->qg += dt * (a.pqg_i + a.pqg_d);
    s->qh += dt * (a.pqh_i + a.pqh_d);
    s->nc += dt * (a.pnc_i + a.pnc_d);
    s->nr += dt * (a.pnr_i + a.pnr_d);
    s->ni += dt * (a.pni_i + a.pni_d);
    s->ns += dt * (a.pns_i + a.pns_d);
    s->ng += dt * (a.png_i + a.png_d);
    s->nh += dt * (a.pnh_i + a.pnh_d);
    s->vg += dt * (a.pvg_i + a.pvg_d);
    s->vh += dt * (a.pvh_i + a.pvh_d);

    // Default eqtset<=1 latent-heating form.
    const float cold = s->temperature < 273.15f ? 1.0f : 0.0f;
    const float warm = 1.0f - cold;
    const float freezing = warm * (
            r.qhmlr + r.qsmlr + r.qhlmlr)
        + cold * (
            r.qsacw + r.qhacw + r.qhlacw
            + r.qsacr + r.qhacr + r.qhlacr
            + r.qsshr + r.qhshr + r.qhlshr
            + r.qrfrz + r.qiacr
            + r.qwfrz + r.qwctfz + r.qiihr + r.qiacw);
    const float sublimation = cold * (
        r.qsdpv + r.qhdpv + r.qhldpv + r.qidpv + r.qisbv + r.qiint)
        + r.qssbv + r.qhsbv + r.qhlsbv;
    const float liquid_vapor = r.qrcev + r.qhcev + r.qscev + r.qhlcev;
    s->theta += dt / exner * (
        nssl_div(latent_fusion(s->temperature), 1004.0f) * freezing
        + nssl_div((latent_vapor(s->temperature) + latent_fusion(s->temperature)),
            1004.0f) * sublimation
        + nssl_div(latent_vapor(s->temperature), 1004.0f) * liquid_vapor);

    // Dense volume and remaining number routes are populated alongside their
    // named cold rates; no intermediate state is committed here.
    (void)rho;
}

}  // namespace

kernel void nssl2_prepare_fused_gs(
    constant Nssl2PrepareFusedGsResources &r [[buffer(0)]],
    constant Nssl2PrepareFusedGsParams &q [[buffer(1)]],
    uint thread_position_in_grid [[thread_position_in_grid]])
{
    const uint idx = thread_position_in_grid;
    if (idx >= uint(q.n)) return;
    // This separate prepass snapshots the complete immutable pre-GS t0/t7
    // fields.  The fused kernel may then read neighbouring targets without
    // racing a thread that has already advanced the prognostic workspace.
    const float temperature = r.full_theta[idx] * r.exner[idx];
    r.temperature_k[idx] = temperature;
    r.target_m3[idx] = primary_ice_target(
        r.state[QV * q.n + idx],
        temperature,
        r.air_density[idx],
        r.pressure_pa[idx]);

}

kernel void nssl2_fused_gs(
    constant Nssl2FusedGsResources &r [[buffer(0)]],
    constant Nssl2FusedGsParams &q [[buffer(1)]],
    uint thread_position_in_grid [[thread_position_in_grid]])
{
    const uint idx = thread_position_in_grid;
    if (idx >= uint(q.n)) return;

    // GS diagnoses from nonnegative gathered work arrays, but final scatter
    // uses the raw vapor base and restores each raw negative hydromass
    // residual.  Snapshot both views before normalization changes anything.
    const float raw_qv = r.state[QV * q.n + idx];
    const float raw_qc = r.state[QC * q.n + idx];
    const float raw_qr = r.state[QR * q.n + idx];
    const float raw_qi = r.state[QI * q.n + idx];
    const float raw_qs = r.state[QS * q.n + idx];
    const float raw_qg = r.state[QG * q.n + idx];
    const float raw_qh = r.state[QH * q.n + idx];

    State s{};
    s.qv = fmax(raw_qv, 0.0f);
    s.qc = fmax(raw_qc, 0.0f);
    s.qr = fmax(raw_qr, 0.0f);
    s.qi = fmax(raw_qi, 0.0f);
    s.qs = fmax(raw_qs, 0.0f);
    s.qg = fmax(raw_qg, 0.0f);
    s.qh = fmax(raw_qh, 0.0f);
    s.nc = fmax(r.state[NC * q.n + idx], 0.0f);
    s.nr = fmax(r.state[NR * q.n + idx], 0.0f);
    s.ni = fmax(r.state[NI * q.n + idx], 0.0f);
    s.ns = fmax(r.state[NS * q.n + idx], 0.0f);
    s.ng = fmax(r.state[NG * q.n + idx], 0.0f);
    s.nh = fmax(r.state[NH * q.n + idx], 0.0f);
    s.nn = fmax(r.state[NN * q.n + idx], 0.0f);
    s.vg = fmax(r.state[VG * q.n + idx], 0.0f);
    s.vh = fmax(r.state[VH * q.n + idx], 0.0f);
    s.theta = r.full_theta[idx];
    s.temperature = r.temperature_k[idx];

    const float rho = r.air_density[idx];
    const ParticleProperties particles = normalize_state(&s, rho);
    Rates rates{};
    Aggregates aggregates{};
    diagnose_warm(s, rho, r.pressure_pa[idx], q.dt, &rates);
    diagnose_cloud_riming(
        s, particles, rho, r.dz[idx], q.dt, &rates);
    diagnose_rain_freezing(
        s, particles, rho, r.pressure_pa[idx], q.dt, &rates);
    diagnose_cloud_freezing(s, rho, r.pressure_pa[idx], q.dt, &rates);
    diagnose_frozen_collection(
        s, particles, rho, r.dz[idx], q.dt, &rates);
    diagnose_snow_aggregation(s, particles, rho, q.dt, &rates);
    diagnose_melting(
        s, particles, rho, r.pressure_pa[idx], r.dz[idx], q.dt, &rates);

    const int k = idx / q.ncol;
    const int column_index = idx - k * q.ncol;
    // module_mp_nssl_2mom.F:14174-14176
    //   kp1 = Min(q.nz, kgs(mgs)+1)
    //   wvel(mgs) = (0.5)*(w(igs(mgs),jgs,kp1) + w(igs(mgs),jgs,kgs(mgs)))
    // The driver copies the STAGGERED w straight into the GS slab
    // (:2827 wn(ix,1,kz) = w(ix,kz,jy), kz = kts..kte, no de-staggering), so
    // that statement IS the interface-to-mass average and WRF performs it
    // once.  The clamp is therefore on the upper INTERFACE, not on a second
    // mass level: the slab holds only q.nz entries, so the top mass level
    // degenerates to wvel = w(q.nz).  In this kernel's zero-based indexing
    // WRF's kp1-1 is exactly min(k + 1, q.nz - 1).  The explicit RN operations
    // hold WRF's add-then-scale order against FMA contraction.
    const int velocity_kp = min(k + 1, q.nz - 1);
    const float w_center = nssl_mul(0.5f, nssl_add(
        r.vertical_velocity[k * q.ncol + column_index],
        r.vertical_velocity[velocity_kp * q.ncol + column_index]));
    const int target_km = max(k - 1, 0);
    // Exact WRF kgsp=MIN(k+1,q.nz-1) in one-based indexing.
    const int target_kp = min(k + 1, q.nz - 2);
    const float vertical_span = (float)(
        max(target_kp - k, 0) + max(k - target_km, 0));
    diagnose_primary_ice(
        s,
        rho,
        r.pressure_pa[idx],
        w_center,
        r.dz[idx],
        r.primary_ice_target_m3[target_km * q.ncol + column_index],
        r.primary_ice_target_m3[target_kp * q.ncol + column_index],
        vertical_span,
        q.dt,
        &rates);

    diagnose_frozen_vapor(
        s,
        particles,
        rho,
        r.pressure_pa[idx],
        r.exner[idx],
        q.dt,
        &rates);
    diagnose_melting_vapor_exchange(
        s,
        particles,
        rho,
        r.pressure_pa[idx],
        q.dt,
        &rates);
    diagnose_crystal_to_snow(s, particles, rho, &rates);
    diagnose_wet_growth_shedding(
        s, particles, rho, r.pressure_pa[idx], q.dt, &rates);
    diagnose_ice_snow_to_graupel(s, particles, rho, &rates);
    if (q.hail_on) {
        // Every graupel<->hail conversion in WRF lives inside the single
        // IF (lhl .gt. 1) block at module_mp_nssl_2mom.F:19860 with its
        // rates zeroed just above (:19847-19858).  With the hail category
        // absent (nssl_hail_on=0 -> lhab=7, lhl=0 at :1444-1451) the block
        // never runs; skipping the call leaves the zero-initialized Rates
        // members standing, and every other hail-indexed rate is already
        // gated on the (identically zero) hail mass/number moments.
        diagnose_graupel_to_hail(
            s, particles, rho, r.pressure_pa[idx], q.dt, &rates);
    }
    diagnose_hallett_mossop(s, rho, &rates);

    // Source-ordered cold diagnosis is inserted here.  It reads only `s` and
    // the environmental arrays, never an already-updated prognostic.
    assemble_and_limit(
        s, particles, rho, q.dt, &rates, &aggregates);
    // qx(lv) above is the nonnegative diagnostic vapor (and may include a
    // tiny cleaned-up hydromass), whereas WRF advances vapor from raw qv0.
    s.qv = raw_qv;
    aggregate_once(&s, rates, aggregates, rho, r.exner[idx], q.dt);

    // WRF 23187-23224: t0 is diagnosed before warm cloud-ice melting and is
    // not recomputed afterward for option-18 ipconc=5/ibfc=1.
    s.temperature = s.theta * r.exner[idx];
    const float temperature_after_rates = s.temperature;
    if (s.temperature > 273.15f && s.qi > 0.0f) {
        const float melted = s.qi;
        s.qc += melted;
        s.nc += s.ni;
        s.qi = 0.0f;
        s.ni = 0.0f;
        s.theta -= latent_fusion(s.temperature) / (1004.0f * r.exner[idx])
            * melted;
    }

    // WRF 23658-23667 restores the original negative Registry residual to
    // qx before entering the post-GS two-moment size limiter.
    s.qc += fmin(raw_qc, 0.0f);
    s.qr += fmin(raw_qr, 0.0f);
    s.qi += fmin(raw_qi, 0.0f);
    s.qs += fmin(raw_qs, 0.0f);
    s.qg += fmin(raw_qg, 0.0f);
    s.qh += fmin(raw_qh, 0.0f);
    final_bounds(&s, rho, particles);
    r.state[QV * q.n + idx] = s.qv;
    r.state[QC * q.n + idx] = s.qc;
    r.state[QR * q.n + idx] = s.qr;
    r.state[QI * q.n + idx] = s.qi;
    r.state[QS * q.n + idx] = s.qs;
    r.state[QG * q.n + idx] = s.qg;
    r.state[QH * q.n + idx] = s.qh;
    r.state[NC * q.n + idx] = s.nc;
    r.state[NR * q.n + idx] = s.nr;
    r.state[NI * q.n + idx] = s.ni;
    r.state[NS * q.n + idx] = s.ns;
    r.state[NG * q.n + idx] = s.ng;
    r.state[NH * q.n + idx] = s.nh;
    r.state[NN * q.n + idx] = s.nn;
    r.state[VG * q.n + idx] = s.vg;
    r.state[VH * q.n + idx] = s.vh;
    r.full_theta[idx] = s.theta;
    r.temperature_k[idx] = temperature_after_rates;

}
