#include <metal_stdlib>
#include "rrtmgp_cal_cldfra1_bindings.metalh"
#include "rrtmgp_mynn_cloud_supplied_bindings.metalh"
#include "rrtmgp_mynn_cloud_merge_bindings.metalh"
#include "rrtmgp_cloudy_background_radii_bindings.metalh"
#include "rrtmgp_surface_emissivity_bindings.metalh"
#include "rrtmgp_hydrometeor_paths_bindings.metalh"
#include "rrtmgp_validate_call_bindings.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"

using namespace metal;
#pragma clang fp contract(off)

inline float rrtmgp_cloud_exp(float value) {
    return as_type<float>(soft_gfk_exp(as_type<uint>(value)));
}

inline float rrtmgp_cloud_pow(float value, float exponent) {
    return as_type<float>(soft_gfk_pow_bits(as_type<uint>(value),
                                             as_type<uint>(exponent)));
}

kernel void rrtmgp_cal_cldfra1(
    constant RrtmgpCalCldfra1Resources &r [[buffer(0)]],
    constant RrtmgpCalCldfra1Params &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;

    const float qcldmin = 1.0e-12f;
    const float svpt0 = 273.15f;
    const float qv = r.qv[cell];
    const float qc = r.qc[cell];
    const float qi = r.qi[cell];
    const float qs = r.qs[cell];
    const float tlay = r.tlay[cell];
    const float play = r.play[cell];

    const float tc = tlay - svpt0;
    const float esw_exp_arg = (17.2693882f * tc) / (tlay - 35.86f);
    const float esw_base = 1000.0f * 0.61078f;
    const float esw = esw_base * rrtmgp_cloud_exp(esw_exp_arg);
    const float esi_exp_arg = (21.8745584f * tc) / (tlay - 7.66f);
    const float esi_base = 1000.0f * 0.61078f;
    const float esi = esi_base * rrtmgp_cloud_exp(esi_exp_arg);
    const float ep2 = 287.0f / 461.6f;
    const float qvsw = (ep2 * esw) / (play - esw);
    const float qvsi = (ep2 * esi) / (play - esi);

    float qcld;
    float weight;
    if (p.f_qi && p.f_qs) {
        const float qice_snow = qi + qs;
        qcld = qice_snow + qc;
        const float divisor = max(qcld, qcldmin);
        weight = qcld < qcldmin ? 0.0f : qice_snow / divisor;
    } else if (p.f_qi) {
        qcld = qi + qc;
        const float divisor = max(qcld, qcldmin);
        weight = qcld < qcldmin ? 0.0f : qi / divisor;
    } else {
        qcld = qc;
        weight = qcld < qcldmin ? 0.0f
            : (tlay > svpt0 ? 0.0f : 1.0f);
    }

    const float liquid_weight = 1.0f - weight;
    const float qvs_weight = liquid_weight * qvsw + weight * qvsi;
    const float rhum = qv / qvs_weight;
    const float subsat = max(1.0e-10f, qvs_weight - qv);
    const float exponent = rrtmgp_cloud_pow(subsat, 0.49f);
    const float arg = max(-6.9f, (-100.0f * qcld) / exponent);
    const float rhum_floor = max(1.0e-10f, rhum);
    const float fraction = rrtmgp_cloud_pow(rhum_floor, 0.25f)
                           * (1.0f - rrtmgp_cloud_exp(arg));
    const float clipped_fraction = fraction < 0.01f ? 0.0f : fraction;
    r.cldfra[cell] = qcld < qcldmin ? 0.0f
        : (rhum >= 1.0f ? 1.0f : clipped_fraction);
}

kernel void rrtmgp_mynn_cloud_supplied(
    constant RrtmgpMynnCloudSuppliedResources &r [[buffer(0)]],
    constant RrtmgpMynnCloudSuppliedParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;
    const bool cloudy_bl = r.cldfra_bl[cell] > 0.001f;
    r.liquid_supplied[cell] = uint(
        (r.qc[cell] < 1.0e-6f) && cloudy_bl && (r.qc_bl[cell] > 0.0f));
    r.ice_supplied[cell] = uint(
        (r.qi[cell] < 1.0e-8f) && cloudy_bl && (r.qi_bl[cell] > 0.0f));
}

kernel void rrtmgp_mynn_cloud_merge(
    constant RrtmgpMynnCloudMergeResources &r [[buffer(0)]],
    constant RrtmgpMynnCloudMergeParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;
    const bool cloudy_bl = r.cldfra_bl[cell] > 0.001f;
    if (p.has_cldfra && p.replace_cldfra) {
        r.cldfra[cell] = r.cldfra_bl[cell];
    }
    const float qc = r.qc[cell];
    const float qi = r.qi[cell];
    if ((qc < 1.0e-6f) && cloudy_bl) r.qc[cell] = qc + r.qc_bl[cell];
    if ((qi < 1.0e-8f) && cloudy_bl) r.qi[cell] = qi + r.qi_bl[cell];
}

kernel void rrtmgp_cloudy_background_radii(
    constant RrtmgpCloudyBackgroundRadiiResources &r [[buffer(0)]],
    constant RrtmgpCloudyBackgroundRadiiParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;
    const float effc = r.effc[cell];
    const float cloudy = r.cldfra[cell];
    const float land_flag = r.xland[cell / uint(p.nlay)];
    const bool is_cloudy = cloudy > 0.0f;
    bool liquid_background = is_cloudy && (effc <= 2.5f);
    if (p.has_supplied_liquid) {
        liquid_background = liquid_background
            || (is_cloudy && r.supplied_liquid[cell] != 0u);
    }
    const bool over_water = land_flag > 1.5f;
    const bool over_land = land_flag < 1.5f;
    r.effc_out[cell] = liquid_background && over_water ? 10.5f
        : (liquid_background && over_land ? 7.5f : effc);

    if (p.is_p3) return;
    bool redo_ice = is_cloudy && (r.effi[cell] <= 5.0f);
    if (p.has_supplied_ice) {
        redo_ice = redo_ice || (is_cloudy && r.supplied_ice[cell] != 0u);
    }
    const float tlay = r.tlay[cell];
    const float whole = trunc(tlay);
    const int index = clamp(int(trunc(tlay - 179.0f)), 1, 75);
    const float corr = tlay - whole;
    const float table_value = r.retab[index - 1] * (1.0f - corr)
                              + r.retab[index] * corr;
    const float new_ice = max(table_value, 5.0f);
    r.effi_out[cell] = redo_ice ? new_ice : r.effi[cell];
}

kernel void rrtmgp_surface_emissivity(
    constant RrtmgpSurfaceEmissivityResources &r [[buffer(0)]],
    constant RrtmgpSurfaceEmissivityParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.ncell)) return;
    const uint col = index / uint(p.nband);
    const uint band = index - col * uint(p.nband);
    uint source_index = 0u;
    if (p.layout == 1) source_index = col;
    else if (p.layout == 2) source_index = band;
    else if (p.layout == 3) source_index = band * uint(p.ncol) + col;
    r.emissivity_out[index] = r.emissivity[source_index];
}

kernel void rrtmgp_hydrometeor_paths(
    constant RrtmgpHydrometeorPathsResources &r [[buffer(0)]],
    constant RrtmgpHydrometeorPathsParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;
    const int lay = int(cell % uint(p.nlay));
    const float qc = r.qc[cell];
    const float qi = p.has_qi ? r.qi[cell] : 0.0f;
    const float qs = p.has_qs ? r.qs[cell] : 0.0f;
    const uint col = cell / uint(p.nlay);
    const uint pindex = col * uint(p.nlay + 1) + uint(lay);
    const float lower = r.plev[pindex + 1u];
    const float upper = r.plev[pindex];
    const float mass_path = abs(lower - upper) * 101.97162129779283f;
    const float cldfra = p.has_cldfra ? r.cldfra[cell] : 1.0f;
    const float incloud = max(0.01f, cldfra);
    const float clwp_grid = qc * mass_path;
    float clwp = p.has_cldfra ? clwp_grid / incloud : clwp_grid;
    float ciwp_grid = (qi + qs) * mass_path;
    float ciwp = p.has_cldfra ? ciwp_grid / incloud : ciwp_grid;
    float reliq = 10.0f;
    float dgice = 50.0f;
    const float tiny = 1.0e-20f;

    if (p.scheme == 0) {
        // Kessler: cloud-only water path, fixed liquid and ice sizes.
    } else if (p.scheme == 1) {
        const float re_c = r.effc[cell];
        const float re_i = r.effi[cell];
        const float re_s = r.effs[cell];
        float qs_eff = qs;
        float re_s_eff = re_s;
        if (p.snow_discount) {
            const float re_s0 = max(10.0f, re_s);
            const float quotient = 130.0f / re_s0;
            const float factor = re_s0 > 130.0f
                ? min(0.99f, quotient * quotient) : 0.99f;
            re_s_eff = min(re_s0, 130.0f);
            qs_eff = qs * factor;
            const float frozen_path = (qi + qs_eff) * mass_path;
            ciwp = p.has_cldfra ? frozen_path / incloud : frozen_path;
        }
        const float frozen = qi + qs_eff;
        const float numerator = qi * re_i + qs_eff * re_s_eff;
        const float reice = frozen > tiny
            ? numerator / max(frozen, tiny) : 25.0f;
        reliq = qc > tiny ? re_c : 10.0f;
        reliq = clamp(reliq, 2.5f, 21.5f);
        dgice = clamp(2.0f * reice, 10.0f, 180.0f);
    } else if (p.scheme == 2) {
        const float re_c = r.effc[cell];
        const float re_i = r.effi[cell];
        const float re_s = max(10.0f, re_i);
        const float quotient = 130.0f / re_s;
        const float factor = re_s > 130.0f
            ? min(0.99f, quotient * quotient) : 0.99f;
        const float re_s_eff = min(re_s, 130.0f);
        const float qs_eff = qi * factor;
        const float frozen_path = qs_eff * mass_path;
        ciwp = p.has_cldfra ? frozen_path / incloud : frozen_path;
        const float reice = qs_eff > tiny ? re_s_eff : 25.0f;
        reliq = qc > tiny ? re_c : 10.0f;
        reliq = clamp(reliq, 2.5f, 21.5f);
        dgice = clamp(2.0f * reice, 10.0f, 180.0f);
    } else if (p.scheme == 3) {
        const float nc = r.nc[cell];
        const float ni = r.ni[cell];
        const float ns = r.ns[cell];
        const bool active_c = qc > p.my2_eps_q && nc > p.my2_eps_n;
        const bool active_i = qi > p.my2_eps_q && ni > p.my2_eps_n;
        const bool active_s = qs > p.my2_eps_q && ns > p.my2_eps_n;
        const float ratio_c = qc * p.my2_icexc9 / max(nc, tiny);
        const float ratio_i = qi * p.my2_icexi9 / max(ni, tiny);
        const float ratio_s = qs * p.my2_igs20 / max(ns, tiny);
        const float ilam_c = rrtmgp_cloud_pow(max(ratio_c, tiny), 1.0f / 3.0f);
        const float ilam_i = max(p.my2_ilam_min,
            rrtmgp_cloud_pow(max(ratio_i, tiny), 1.0f / 3.0f));
        const float ilam_s = max(p.my2_ilam_min,
            rrtmgp_cloud_pow(max(ratio_s, tiny), p.my2_idms));
        const float re_c = p.my2_cloud_radius * ilam_c * 1.0e6f;
        const float re_i = p.my2_ice_radius * ilam_i * 1.0e6f;
        const float re_s = p.my2_snow_radius * ilam_s * 1.0e6f;
        const float wi = active_i ? ni : 0.0f;
        const float ws = active_s ? ns : 0.0f;
        reliq = active_c ? re_c : 10.0f;
        const float total_n = wi + ws;
        const float numerator = wi * re_i + ws * re_s;
        const float reice = total_n > 0.0f
            ? numerator / max(total_n, tiny) : 25.0f;
        reliq = clamp(reliq, 2.5f, 21.5f);
        dgice = clamp(2.0f * reice, 10.0f, 180.0f);
    } else {
        const float play = r.play[cell];
        const float tlay = r.tlay[cell];
        const float nc = r.nc[cell];
        const float nr = r.nr[cell];
        const float ni = r.ni[cell];
        const float ns = r.ns[cell];
        float re_c;
        float re_i;
        float re_s;
        if (p.morrison_use_effective) {
            re_c = r.effc[cell];
            // Morrison's supplied rain radius is accepted for the upstream
            // interface but intentionally does not enter cloud optics.
            const float unused_reffr = r.effr[cell];
            (void)unused_reffr;
            re_i = r.effi[cell];
            re_s = r.effs[cell];
        } else {
            const float rho_air = play / (287.15f * tlay);
            const float shape = 0.0005714f * (nc / 1.0e6f * rho_air)
                                + 0.2714f;
            const float pgam = clamp(1.0f / (shape * shape) - 1.0f,
                                     2.0f, 10.0f);
            const float gamma_ratio = (pgam + 1.0f) * (pgam + 2.0f)
                                      * (pgam + 3.0f);
            const float cloud_ratio = p.morrison_cloud_lambda_const * nc
                                      * gamma_ratio
                                      / max(qc, tiny);
            const float lam_c = rrtmgp_cloud_pow(
                cloud_ratio, 1.0f / 3.0f);
            re_c = (pgam + 3.0f) / max(2.0f * lam_c, tiny) * 1.0e6f;
            const float ice_ratio = p.morrison_ice_lambda_const * ni
                                    / max(qi, tiny);
            const float snow_ratio = p.morrison_snow_lambda_const * ns
                                     / max(qs, tiny);
            const float lam_i = rrtmgp_cloud_pow(ice_ratio, 1.0f / 3.0f);
            const float lam_s = rrtmgp_cloud_pow(snow_ratio, 1.0f / 3.0f);
            re_i = 1.5e6f / max(lam_i, tiny);
            re_s = 1.5e6f / max(lam_s, tiny);
        }
        const float wc = (qc > 0.0f && nc > 0.0f) ? nc : 0.0f;
        const float wi = (qi > 0.0f && ni > 0.0f) ? ni : 0.0f;
        const float ws = (qs > 0.0f && ns > 0.0f) ? ns : 0.0f;
        reliq = wc > 0.0f ? re_c : 10.0f;
        const float total_n = wi + ws;
        const float numerator = wi * re_i + ws * re_s;
        const float reice = total_n > 0.0f
            ? numerator / max(total_n, tiny) : 25.0f;
        reliq = clamp(reliq, 2.5f, 21.5f);
        dgice = clamp(2.0f * reice, 10.0f, 180.0f);
        (void)nr;
    }

    r.clwp[cell] = clwp;
    r.ciwp[cell] = ciwp;
    r.reliq[cell] = reliq;
    r.dgice[cell] = dgice;
}

inline bool rrtmgp_outside(float value, float lower, float upper) {
    return !isfinite(value) || value < lower || value > upper;
}

inline bool rrtmgp_invalid_nonnegative(float value) {
    return !isfinite(value) || value < 0.0f;
}

kernel void rrtmgp_validate_call(
    constant RrtmgpValidateCallResources &r [[buffer(0)]],
    constant RrtmgpValidateCallParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    const uint ncell = uint(p.ncol * p.nlay);
    const uint nlev_cell = uint(p.ncol * (p.nlay + 1));
    uint found = 0u;
    if (idx < ncell) {
        if (rrtmgp_outside(r.play[idx], p.play_lower, p.play_upper))
            found |= 1u << 0u;
        if (rrtmgp_outside(r.tlay[idx], p.temp_lower, p.temp_upper))
            found |= 1u << 2u;
        if (rrtmgp_invalid_nonnegative(r.qv[idx])) found |= 1u << 5u;
        if (rrtmgp_invalid_nonnegative(r.qc[idx])) found |= 1u << 6u;
        if (rrtmgp_invalid_nonnegative(r.qr[idx])) found |= 1u << 7u;
        if (rrtmgp_invalid_nonnegative(r.qi[idx])) found |= 1u << 8u;
        if (rrtmgp_invalid_nonnegative(r.qs[idx])) found |= 1u << 9u;
        if (rrtmgp_outside(r.cldfra[idx], 0.0f, 1.0f)) found |= 1u << 10u;
        if (p.morrison) {
            if (rrtmgp_invalid_nonnegative(r.nc[idx])) found |= 1u << 11u;
            if (rrtmgp_invalid_nonnegative(r.nr[idx])) found |= 1u << 12u;
            if (rrtmgp_invalid_nonnegative(r.ni[idx])) found |= 1u << 13u;
            if (rrtmgp_invalid_nonnegative(r.ns[idx])) found |= 1u << 14u;
        }
        if (p.have_effective) {
            if (rrtmgp_invalid_nonnegative(r.effc[idx])) found |= 1u << 15u;
            if (rrtmgp_invalid_nonnegative(r.effr[idx])) found |= 1u << 16u;
            if (rrtmgp_invalid_nonnegative(r.effi[idx])) found |= 1u << 17u;
            if (rrtmgp_invalid_nonnegative(r.effs[idx])) found |= 1u << 18u;
            if (r.effc[idx] < p.effc_lower || r.effc[idx] > p.effc_upper)
                found |= 1u << 22u;
            if (r.effi[idx] < p.effi_lower || r.effi[idx] > p.effi_upper)
                found |= 1u << 23u;
            if (r.effs[idx] < p.effs_lower || r.effs[idx] > p.effs_upper)
                found |= 1u << 24u;
        }
        const uint col = idx / uint(p.nlay);
        const uint lay = idx - col * uint(p.nlay);
        const uint pbase = col * uint(p.nlay + 1);
        const float dp = abs(r.plev[pbase + lay + 1u] - r.plev[pbase + lay]);
        if ((dp <= 0.0f) || (r.exner[idx] <= 0.0f)) found |= 1u << 21u;
    }
    if (idx < nlev_cell) {
        const float pressure = r.plev[idx];
        if (!isfinite(pressure) || pressure < 0.0f) found |= 1u << 1u;
        if (rrtmgp_outside(r.tlev[idx], p.temp_lower, p.temp_upper))
            found |= 1u << 3u;
    }
    if (idx < uint(p.ncol)) {
        if (rrtmgp_outside(r.tsfc[idx], p.temp_lower, p.temp_upper))
            found |= 1u << 4u;
        const uint base = idx * uint(p.nlay);
        if (r.play[base] < r.play[base + 1u]) found |= 1u << 20u;
        if (p.has_expected_top) {
            const uint top = idx * uint(p.nlay + 1) + uint(p.nlay);
            if (abs(r.plev[top] - p.expected_top) > p.top_tolerance)
                found |= 1u << 25u;
        }
    }
    if (idx < uint(p.nemiss)
        && rrtmgp_outside(r.emiss[idx], 0.0f, 1.0f)) found |= 1u << 19u;
    if (found != 0u) {
        device atomic_uint *flags = reinterpret_cast<device atomic_uint *>(r.flags);
        atomic_fetch_or_explicit(flags, found, memory_order_relaxed);
    }
}
