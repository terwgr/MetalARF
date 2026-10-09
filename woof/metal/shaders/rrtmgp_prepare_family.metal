#include <metal_stdlib>
#include "rrtmgp_prepare_lw_profile_bindings.metalh"
#include "rrtmgp_prepare_sw_profile_bindings.metalh"
#include "rrtmgp_prepare_solver_temperatures_bindings.metalh"
#include "rrtmgp_prepare_clear_upper_fields_bindings.metalh"
#include "rrtmgp_prepare_gas_vmr_bindings.metalh"
#include "rrtmgp_prepare_pack_columns_bindings.metalh"
#include "rrtmgp_prepare_clamp_top_bindings.metalh"
#include "rrtmgp_prepare_interface_temperatures_bindings.metalh"
#include "soft_libm64.metalh"

using namespace metal;
#pragma clang fp contract(off)

kernel void rrtmgp_prepare_pack_columns(
    constant RrtmgpPreparePackColumnsResources &r [[buffer(0)]],
    constant RrtmgpPreparePackColumnsParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = uint(p.ncol) * uint(p.nlay);
    if (index >= count) return;
    const uint col = index / uint(p.nlay);
    const uint lay = index - col * uint(p.nlay);
    const uint y = col / uint(p.nx), x = col - y * uint(p.nx);
    r.output[index] = r.input[(lay * uint(p.ny) + y) * uint(p.nx) + x];
}

kernel void rrtmgp_prepare_clamp_top(
    constant RrtmgpPrepareClampTopResources &r [[buffer(0)]],
    constant RrtmgpPrepareClampTopParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const uint index = col * uint(p.nlev) + uint(p.nlev - 1);
    r.plev[index] = max(r.plev[index], p.pressure_floor);
}

kernel void rrtmgp_prepare_interface_temperatures(
    constant RrtmgpPrepareInterfaceTemperaturesResources &r [[buffer(0)]],
    constant RrtmgpPrepareInterfaceTemperaturesParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint nlay = uint(p.nlay), stride = nlay + 1u;
    const uint total = uint(p.ncol) * stride;
    if (index >= total || nlay < 2u) return;
    const uint col = index / stride, lev = index - col * stride;
    const uint cell = col * nlay;
    const uint face = col * stride + lev;
    if (lev == 0u) {
        r.tlev[face] = r.tlay[cell]
            + (r.plev[face] - r.play[cell])
            * (r.tlay[cell + 1u] - r.tlay[cell])
            / (r.play[cell + 1u] - r.play[cell]);
    } else if (lev == nlay) {
        r.tlev[face] = r.tlay[cell + nlay - 1u]
            + (r.plev[face] - r.play[cell + nlay - 1u])
            * (r.tlay[cell + nlay - 1u] - r.tlay[cell + nlay - 2u])
            / (r.play[cell + nlay - 1u] - r.play[cell + nlay - 2u]);
    } else {
        const uint lower = cell + lev - 1u, upper = cell + lev;
        const float interface_pressure = r.plev[face];
        const float numerator = r.play[lower] * r.tlay[lower]
            * (interface_pressure - r.play[upper])
            + r.play[upper] * r.tlay[upper]
            * (r.play[lower] - interface_pressure);
        const float denominator = interface_pressure
            * (r.play[lower] - r.play[upper]);
        r.tlev[face] = numerator / denominator;
    }
}

static inline ulong prep_d32(float value) {
    return soft64_from_f32_bits(as_type<uint>(value));
}

static inline ulong prep_interp_climo(float pressure_hpa,
                                      device const float *p,
                                      device const float *t, int n) {
    const ulong x = prep_d32(pressure_hpa);
    if (soft64_less_equal(x, prep_d32(p[0]))) return prep_d32(t[0]);
    if (soft64_greater_equal(x, prep_d32(p[n - 1])))
        return prep_d32(t[n - 1]);
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (soft64_less_equal(prep_d32(p[mid]), x)) lo = mid;
        else hi = mid;
    }
    const ulong x0 = prep_d32(p[lo]);
    const ulong dx = soft64_sub(prep_d32(p[hi]), x0);
    const ulong delta = soft64_sub(prep_d32(t[hi]), prep_d32(t[lo]));
    const ulong slope = soft64_div(delta, dx);
    return soft64_add(prep_d32(t[lo]),
                      soft64_mul(slope, soft64_sub(x, x0)));
}

template <typename R, typename P>
inline ulong prepare_lw_interface_temperature(constant R &r,
                                             constant P &p,
                                             int col, int lev) {
    const int m = p.model_nlay, u = p.upper_nlay;
    if (lev <= m)
        return prep_d32(r.tlev[col * (m + 1) + lev]);

    const int j = lev - m - 1;
    const float top = r.plev[col * (m + 1) + m];
    float pressure = top - 400.0f * float(j + 1);
    if (j == u - 1) pressure = p.pressure_floor;
    const ulong climo_top = prep_interp_climo(
        top * 0.01f, r.climo_pressure, r.climo_temperature, p.nclimo);
    const ulong climo_lev = prep_interp_climo(
        pressure * 0.01f, r.climo_pressure, r.climo_temperature, p.nclimo);
    const ulong shift = soft64_sub(
        prep_d32(r.tlev[col * (m + 1) + m]), climo_top);
    return soft64_add(climo_lev, shift);
}

template <typename R, typename P>
inline void prepare_lw_cell(constant R &r, constant P &p, uint index) {
    const uint extent = uint(p.ncol) * uint(p.total_nlay + 1);
    if (index >= extent) return;
    const int col = int(index) / (p.total_nlay + 1);
    const int lev = int(index) - col * (p.total_nlay + 1);
    const int ncol = p.ncol, m = p.model_nlay, u = p.upper_nlay;
    const int outlev = col * (p.total_nlay + 1) + lev;
    if (lev <= m) {
        r.out_plev[outlev] = r.plev[col * (m + 1) + lev];
        r.out_tlev[outlev] = prep_d32(r.tlev[col * (m + 1) + lev]);
    } else {
        const int j = lev - m - 1;
        const float top = r.plev[col * (m + 1) + m];
        float pressure = top - 400.0f * float(j + 1);
        if (j == u - 1) pressure = p.pressure_floor;
        r.out_plev[outlev] = pressure;
        const float ptop_hpa = top * 0.01f;
        const float plev_hpa = pressure * 0.01f;
        const ulong climo_top = prep_interp_climo(
            ptop_hpa, r.climo_pressure, r.climo_temperature, p.nclimo);
        const ulong climo_lev = prep_interp_climo(
            plev_hpa, r.climo_pressure, r.climo_temperature, p.nclimo);
        const ulong shift = soft64_sub(prep_d32(r.tlev[col * (m + 1) + m]),
                                       climo_top);
        r.out_tlev[outlev] = soft64_add(climo_lev, shift);
    }
    if (lev < p.total_nlay) {
        const int outcell = col * p.total_nlay + lev;
        if (lev < m) {
            r.out_play[outcell] = r.play[col * m + lev];
            r.out_qv[outcell] = r.qv[col * m + lev];
            r.out_tlay[outcell] = prep_d32(r.tlay[col * m + lev]);
        } else {
            const int j = lev - m;
            const float top = r.plev[col * (m + 1) + m];
            const float lower = top - 400.0f * float(j);
            float upper = top - 400.0f * float(j + 1);
            if (j == u - 1) upper = p.pressure_floor;
            r.out_play[outcell] = 0.5f * (lower + upper);
            r.out_qv[outcell] = r.qv[col * m + m - 1];
            // Neighboring interface temperatures are written by other
            // threads in this dispatch.  Derive both locally so this layer
            // never reads an interface before its owner has written it.
            const ulong lower_t = prepare_lw_interface_temperature(
                r, p, col, lev);
            const ulong upper_t = prepare_lw_interface_temperature(
                r, p, col, lev + 1);
            const ulong average = soft64_mul(soft64_add(lower_t, upper_t),
                                            0x3fe0000000000000ul);
            r.out_tlay[outcell] = average;
        }
    }
}

kernel void rrtmgp_prepare_lw_profile(
    constant RrtmgpPrepareLwProfileResources &r [[buffer(0)]],
    constant RrtmgpPrepareLwProfileParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    prepare_lw_cell(r, p, index);
}

template <typename R, typename P>
inline void prepare_sw_cell(constant R &r, constant P &p, uint index) {
    const uint extent = uint(p.ncol) * uint(p.total_nlay + 1);
    if (index >= extent) return;
    const int col = int(index) / (p.total_nlay + 1);
    const int lev = int(index) - col * (p.total_nlay + 1);
    const int m = p.model_nlay;
    if (lev <= m) r.out_plev[index] = r.plev[col * (m + 1) + lev];
    else r.out_plev[index] = p.pressure_floor;
    if (lev <= m) r.out_tlev[index] = r.tlev[col * (m + 1) + lev];
    else r.out_tlev[index] = r.tlev[col * (m + 1) + m];
    if (lev < p.total_nlay) {
        const int outcell = col * p.total_nlay + lev;
        if (lev < m) {
            r.out_play[outcell] = r.play[col * m + lev];
            r.out_qv[outcell] = r.qv[col * m + lev];
            r.out_tlay[outcell] = r.tlay[col * m + lev];
        } else {
            r.out_play[outcell] = 0.5f * r.plev[col * (m + 1) + m];
            r.out_qv[outcell] = r.qv[col * m + m - 1];
            r.out_tlay[outcell] = r.tlev[col * (m + 1) + m];
        }
    }
}

kernel void rrtmgp_prepare_sw_profile(
    constant RrtmgpPrepareSwProfileResources &r [[buffer(0)]],
    constant RrtmgpPrepareSwProfileParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    prepare_sw_cell(r, p, index);
}

kernel void rrtmgp_prepare_solver_temperatures(
    constant RrtmgpPrepareSolverTemperaturesResources &r [[buffer(0)]],
    constant RrtmgpPrepareSolverTemperaturesParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint nlay = uint(p.nlay), total = uint(p.ncol) * (nlay + 1u);
    if (index >= total) return;
    if (index < uint(p.ncol) * nlay)
        r.out_tlay[index] = as_type<float>(
            soft64_to_f32_bits(r.tlay[index]));
    r.out_tlev[index] = as_type<float>(
        soft64_to_f32_bits(r.tlev[index]));
}

kernel void rrtmgp_prepare_clear_upper_fields(
    constant RrtmgpPrepareClearUpperFieldsResources &r [[buffer(0)]],
    constant RrtmgpPrepareClearUpperFieldsParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = uint(p.ncol) * uint(p.nlay);
    if (index >= count) return;
    const int lay = int(index) % p.nlay;
    const bool model = lay < p.model_nlay;
    const int source = int(index / uint(p.nlay)) * p.model_nlay + lay;
    r.out_0[index] = model ? r.in_0[source] : 0.0f;
    r.out_1[index] = model ? r.in_1[source] : 0.0f;
    r.out_2[index] = model ? r.in_2[source] : 0.0f;
    r.out_3[index] = model ? r.in_3[source] : 0.0f;
    r.out_4[index] = model ? r.in_4[source] : 0.0f;
}

kernel void rrtmgp_prepare_gas_vmr(
    constant RrtmgpPrepareGasVmrResources &r [[buffer(0)]],
    constant RrtmgpPrepareGasVmrParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = uint(p.ncells) * uint(p.nslot);
    if (index >= total) return;
    const int cell = int(index) / p.nslot;
    const int slot = int(index) - cell * p.nslot;
    float value = 0.0f;
    if (slot == p.h2o_index) value = r.qv[cell] * p.h2o_scale;
    if (slot == p.o3_index) {
        const ulong pressure = soft64_from_f32_bits(as_type<uint>(r.play[cell]));
        ulong ozone = soft64_from_f32_bits(as_type<uint>(r.ozone_vmr[0]));
        if (soft64_less_equal(pressure, r.ozone_pressure[0])) {
            ozone = soft64_from_f32_bits(as_type<uint>(r.ozone_vmr[0]));
        } else if (soft64_greater_equal(
                       pressure, r.ozone_pressure[p.nozone - 1])) {
            ozone = soft64_from_f32_bits(
                as_type<uint>(r.ozone_vmr[p.nozone - 1]));
        } else {
            int lo = 0, hi = p.nozone - 1;
            while (hi - lo > 1) {
                const int mid = lo + (hi - lo) / 2;
                if (soft64_less_equal(r.ozone_pressure[mid], pressure)) lo = mid;
                else hi = mid;
            }
            const ulong x = soft64_plm_log(pressure);
            const ulong x0 = soft64_plm_log(r.ozone_pressure[lo]);
            const ulong x1 = soft64_plm_log(r.ozone_pressure[hi]);
            const ulong y0 = soft64_from_f32_bits(
                as_type<uint>(r.ozone_vmr[lo]));
            const ulong delta = soft64_sub(
                soft64_from_f32_bits(as_type<uint>(r.ozone_vmr[hi])), y0);
            const ulong slope = soft64_div(delta, soft64_sub(x1, x0));
            ozone = soft64_add(y0,
                               soft64_mul(slope, soft64_sub(x, x0)));
        }
        value = as_type<float>(soft64_to_f32_bits(ozone));
    }
    for (int t = 0; t < p.ntrace; ++t)
        if (r.trace_index[t] == slot) value = r.trace_value[t];
    r.vmr[index] = value;
}
