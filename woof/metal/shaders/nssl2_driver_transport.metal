// WRF v4.6.1 NSSL option-18 state gather, moment initialization, and transport.
#include <metal_stdlib>
#include "nssl2_driver_transport_bindings.metalh"
#include "nssl2_soft64.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float nssl_div_rn(float a, float b) { return a / b; }

kernel void nssl2_driver_gather_initialize(
    constant Nssl2DriverGatherInitializeResources &r [[buffer(0)]],
    constant Nssl2DriverGatherInitializeParams &p [[buffer(1)]],
    uint thread_position_in_grid [[thread_position_in_grid]]) {
    const int idx = thread_position_in_grid;
    if (idx >= p.n) return;

    const float rho = r.air_density[idx];
    const float cxmin = 1.0e-8f;
    const float qxmin_init = 1.0e-8f;
    const float qxmin_cloud = 1.0e-13f;
    const float qxmin_rain = 1.0e-12f;

    float vapor = r.qv[idx];
    float cloud = r.qc[idx];
    float rain = r.qr[idx];
    float ice = r.qi[idx];
    float snow = r.qs[idx];
    float graupel = r.qg[idx];
    float hail = r.qh[idx];

    // Exact driver denscale loop: all number and volume moments enter the
    // internal pipeline in concentration space once, before any processing.
    float cloud_number = r.qndrop[idx] * rho;
    float rain_number = r.qnr[idx] * rho;
    float ice_number = r.qni[idx] * rho;
    float snow_number = r.qns[idx] * rho;
    float graupel_number = r.qng[idx] * rho;
    float hail_number = r.qnh[idx] * rho;
    // module_mp_nssl_2mom.F:2719-2739 is the driver's CCN load.  With
    // predicted CCN (nssl_ccn_on=1 -> Registry package nssl_ccn_opt, so
    // f_qnn and therefore flag_ccn are true at :2464-2466) the prognostic
    // qnn field is loaded at :2727.  With nssl_ccn_on=0 the field does not
    // exist and WRF instead diagnoses the unactivated CCN every step from
    // the constant base concentration at :2734, taking the lccna==0 branch
    // because turn_on_ccna is gated on ccn_on==1 (:1403).  The subtraction
    // happens in per-mass units and only then is the whole slot scaled by
    // density at :2935-2944, so the product is formed after the difference.
    float ccn_number = p.predicted_ccn
        ? r.qnn[idx] * rho
        : (408163264.0f - r.qndrop[idx]) * rho;
    float graupel_volume = r.qvolg[idx] * rho;
    float hail_volume = r.qvolh[idx] * rho;

    if (p.first_step != 0) {
        if (cloud_number <= cxmin && cloud > qxmin_init) {
            const float qccn = 408163264.0f;
            const float cwmas_inverse = 327479132160.0f;
            cloud_number = fmin(qccn, cloud * cwmas_inverse) * rho;
            ccn_number -= cloud_number;
        } else if (cloud <= qxmin_cloud
                   || (cloud_number <= cxmin && cloud <= qxmin_init)) {
            vapor += cloud;
            cloud_number = 0.0f;
            cloud = 0.0f;
        }

        if (ice_number <= cxmin && ice > qxmin_init) {
            const float xims = 4.7123910329460728e-10f;
            ice_number = nssl_div_rn(rho * ice, xims);
        } else if (ice <= qxmin_cloud
                   || (ice_number <= cxmin && ice <= qxmin_init)) {
            vapor += ice;
            ice_number = 0.0f;
            ice = 0.0f;
        }

        if (rain_number <= 0.1f * cxmin && rain > qxmin_init) {
            const float zrfac = 3.9788734806922577e-11f;
            const Nssl2Soft64 lambda_inverse = pow(
                Nssl2Soft64(rho) * Nssl2Soft64(rain) * Nssl2Soft64(zrfac),
                0.25f);
            rain_number = (float)(
                lambda_inverse * Nssl2Soft64(8000000.0f)
                * Nssl2Soft64(20.0f) / Nssl2Soft64(20.0f));
        } else if (rain <= qxmin_rain
                   || (rain_number <= cxmin && rain <= qxmin_init)) {
            vapor += rain;
            rain_number = 0.0f;
            rain = 0.0f;
        }

        if (snow_number <= 0.1f * cxmin && snow > qxmin_init) {
            const float zsfac = 1.0610329281846020e-9f;
            const Nssl2Soft64 lambda_inverse = pow(
                Nssl2Soft64(rho) * Nssl2Soft64(snow) * Nssl2Soft64(zsfac),
                0.25f);
            snow_number = (float)(
                lambda_inverse * Nssl2Soft64(3000000.0f)
                * Nssl2Soft64(6.0000004768371582f) / Nssl2Soft64(20.0f));
        } else if (snow <= qxmin_cloud
                   || (snow_number <= cxmin && snow <= qxmin_init)) {
            vapor += snow;
            snow_number = 0.0f;
            snow = 0.0f;
        }

        if (graupel_number <= 0.1f * cxmin && graupel > qxmin_init) {
            if (graupel_volume <= 0.0f) {
                // Historical WRF quirk: this assignment follows denscale and
                // therefore is intentionally not multiplied by air density.
                graupel_volume = nssl_div_rn(graupel, 700.0f);
            }
            const float zhfac = 2.2736419413860176e-9f;
            const float xgms = 9.8960235561662557e-9f;
            const Nssl2Soft64 lambda_inverse = pow(
                Nssl2Soft64(rho) * Nssl2Soft64(graupel) * Nssl2Soft64(zhfac),
                0.25f);
            const Nssl2Soft64 intercept_number =
                lambda_inverse * Nssl2Soft64(200000.0f);
            const Nssl2Soft64 maximum_number =
                Nssl2Soft64(rho) * Nssl2Soft64(graupel) / Nssl2Soft64(xgms);
            const Nssl2Soft64 diagnosed = fmin(intercept_number, maximum_number);
            if (diagnosed > cxmin) {
                graupel_number = (float)diagnosed;
            } else {
                graupel = 0.0f;
                graupel_number = 0.0f;
                graupel_volume = 0.0f;
            }
        } else if (graupel <= qxmin_rain
                   || (graupel_number <= cxmin
                       && graupel <= qxmin_init)) {
            vapor += graupel;
            graupel = 0.0f;
        }

        if (hail_number <= 0.1f * cxmin && hail > qxmin_init) {
            if (hail_volume <= 0.0f) {
                hail_volume = nssl_div_rn(hail, 900.0f);
            }
            const float zhlfac = 8.8419414012719244e-9f;
            const Nssl2Soft64 lambda_inverse = pow(
                Nssl2Soft64(rho) * Nssl2Soft64(hail) * Nssl2Soft64(zhlfac),
                0.25f);
            hail_number = (float)(
                lambda_inverse * Nssl2Soft64(40000.0f)
                * Nssl2Soft64(8.75f) / Nssl2Soft64(20.0f));
        } else if (hail <= qxmin_rain
                   || (hail_number <= cxmin && hail <= qxmin_init)) {
            vapor += hail;
            hail = 0.0f;
        }
    }

    if (p.cu_used != 0) {
        // calcnfromcuten diagnoses number only. Graupel/hail branches are
        // commented out in WRF 4.6.1, while all four live KF rates are used.
        const float cloud_increment = p.dt * r.qccuten[idx];
        const float ice_increment = p.dt * r.qicuten[idx];
        const float rain_increment = p.dt * r.qrcuten[idx];
        const float snow_increment = p.dt * r.qscuten[idx];
        // WRF 4.6.1 loads qccuten/qicuten into the cloud/ice *mass* slots,
        // but calcnfromcuten gates those branches on the corresponding empty
        // ancuten number slots. Thus both rates are consumed yet diagnose no
        // number increment. Preserve that exact official-source behavior.
        (void)cloud_increment;
        (void)ice_increment;
        if (rain_increment > qxmin_rain) {
            const float zrfac = 3.9788734806922577e-11f;
            const Nssl2Soft64 lambda_inverse = pow(
                Nssl2Soft64(rho) * Nssl2Soft64(rain_increment)
                * Nssl2Soft64(zrfac), 0.25f);
            rain_number += (float)(
                lambda_inverse * Nssl2Soft64(8000000.0f)
                * Nssl2Soft64(20.0f) / Nssl2Soft64(20.0f));
        }
        if (snow_increment > qxmin_cloud) {
            const float zsfac = 1.0610329281846020e-9f;
            const Nssl2Soft64 lambda_inverse = pow(
                Nssl2Soft64(rho) * Nssl2Soft64(snow_increment)
                * Nssl2Soft64(zsfac), 0.25f);
            snow_number += (float)(
                lambda_inverse * Nssl2Soft64(3000000.0f)
                * Nssl2Soft64(6.0000004768371582f) / Nssl2Soft64(20.0f));
        }
    }

    r.state[0 * p.n + idx] = vapor;
    r.state[1 * p.n + idx] = cloud;
    r.state[2 * p.n + idx] = rain;
    r.state[3 * p.n + idx] = ice;
    r.state[4 * p.n + idx] = snow;
    r.state[5 * p.n + idx] = graupel;
    r.state[6 * p.n + idx] = hail;
    r.state[7 * p.n + idx] = cloud_number;
    r.state[8 * p.n + idx] = rain_number;
    r.state[9 * p.n + idx] = ice_number;
    r.state[10 * p.n + idx] = snow_number;
    r.state[11 * p.n + idx] = graupel_number;
    r.state[12 * p.n + idx] = hail_number;
    r.state[13 * p.n + idx] = ccn_number;
    r.state[14 * p.n + idx] = graupel_volume;
    r.state[15 * p.n + idx] = hail_volume;
}

kernel void nssl2_driver_reduce_precipitation(
    constant Nssl2DriverReducePrecipitationResources &r [[buffer(0)]],
    constant Nssl2DriverReducePrecipitationParams &p [[buffer(1)]],
    uint thread_position_in_grid [[thread_position_in_grid]]) {
    const uint column = thread_position_in_grid;
    if (column >= p.ncol) return;
    const float rain = r.category_export[column];
    const float snow = r.category_export[2 * p.ncol + column];
    const float graupel = r.category_export[3 * p.ncol + column];
    const float hail = r.category_export[4 * p.ncol + column];
    const float total = rain + snow + graupel + hail;
    r.rainncv[column] = total;
    r.snowncv[column] = snow;
    r.graupelncv[column] = graupel;
    r.hailncv[column] = hail;
    r.rainnc[column] += total;
    r.snownc[column] += snow;
    r.graupelnc[column] += graupel;
    r.hailnc[column] += hail;
    r.sr[column] = (snow + graupel + hail) / (total + 1.0e-12f);
}

kernel void nssl2_driver_scatter(
    constant Nssl2DriverScatterResources &r [[buffer(0)]],
    constant Nssl2DriverScatterParams &p [[buffer(1)]],
    uint thread_position_in_grid [[thread_position_in_grid]]) {
    const uint idx = thread_position_in_grid;
    if (idx >= p.n) return;
    const uint n = p.n;
    r.qv[idx] = r.state[idx];
    r.qc[idx] = r.state[n + idx];
    r.qr[idx] = r.state[2 * n + idx];
    r.qi[idx] = r.state[3 * n + idx];
    r.qs[idx] = r.state[4 * n + idx];
    r.qg[idx] = r.state[5 * n + idx];
    r.qh[idx] = r.state[6 * n + idx];
    r.qndrop[idx] = r.state[7 * n + idx] / r.air_density[idx];
    r.qnr[idx] = r.state[8 * n + idx] / r.air_density[idx];
    r.qni[idx] = r.state[9 * n + idx] / r.air_density[idx];
    r.qns[idx] = r.state[10 * n + idx] / r.air_density[idx];
    r.qng[idx] = r.state[11 * n + idx] / r.air_density[idx];
    r.qnh[idx] = r.state[12 * n + idx] / r.air_density[idx];
    if (p.predicted_ccn != 0)
        r.qnn[idx] = r.state[13 * n + idx] / r.air_density[idx];
    r.qvolg[idx] = r.state[14 * n + idx] / r.air_density[idx];
    r.qvolh[idx] = r.state[15 * n + idx] / r.air_density[idx];
}
