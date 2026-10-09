// Global operator-energy observer on resident FP64 state.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_insitu_energy_fragment_bindings.metalh"
using namespace metal;

kernel void global_insitu_energy_columns_fragment(
    constant GlobalInsituEnergyColumnsFragmentResources &r [[buffer(0)]],
    constant GlobalInsituEnergyColumnsFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncell = uint(p.nlat) * uint(p.nlon);
    if (col >= ncell) return;
    const uint lat = col / uint(p.nlon);
    const ulong gravity = 0x40239d013a92a305ul; // 9.80665
    const ulong half_bits = 0x3fe0000000000000ul;
    const ulong cp_minus_r = 0x4086680000000000ul; // 717
    const ulong gas_constant = 0x4071f00000000000ul; // 287
    const ulong area = soft64_div(
        soft64_mul(half_bits, r.weights[lat]), soft64_from_i32(p.nlon));
    for (uint k = 0; k < uint(p.nlevel); ++k) {
        const uint i = k * ncell + col;
        const ulong dp_g = soft64_div(r.dp[i], gravity);
        const ulong speed2 = soft64_add(
            soft64_mul(r.u[i], r.u[i]), soft64_mul(r.v[i], r.v[i]));
        const ulong kinetic = soft64_mul(
            soft64_mul(soft64_mul(half_bits, speed2), dp_g), area);
        const ulong temp_mass = soft64_mul(r.temperature[i], dp_g);
        const ulong internal = soft64_mul(
            soft64_mul(cp_minus_r, temp_mass), area);
        const ulong potential = soft64_mul(
            soft64_mul(gas_constant, temp_mass), area);
        const uint row = k * 4u;
        r.columns[(row + 0u) * ncell + col] = kinetic;
        r.columns[(row + 1u) * ncell + col] = internal;
        r.columns[(row + 2u) * ncell + col] = potential;
        r.columns[(row + 3u) * ncell + col] =
            soft64_add(soft64_add(kinetic, internal), potential);
    }
    const uint surface_row = uint(p.nlevel) * 4u;
    const ulong surface = soft64_div(
        soft64_mul(r.surface_geopotential[col], r.ps[col]), gravity);
    r.columns[(surface_row + 0u) * ncell + col] = 0ul;
    r.columns[(surface_row + 1u) * ncell + col] = 0ul;
    r.columns[(surface_row + 2u) * ncell + col] = soft64_mul(surface, area);
    r.columns[(surface_row + 3u) * ncell + col] = soft64_mul(surface, area);
}

kernel void global_insitu_energy_band_columns_fragment(
    constant GlobalInsituEnergyBandColumnsFragmentResources &r [[buffer(0)]],
    constant GlobalInsituEnergyBandColumnsFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncell = uint(p.nlat) * uint(p.nlon);
    if (col >= ncell) return;
    const uint lat = col / uint(p.nlon);
    const ulong gravity = 0x40239d013a92a305ul;
    const ulong half_bits = 0x3fe0000000000000ul;
    const ulong area_n = soft64_div(soft64_mul(half_bits,
        soft64_mul(r.weights[lat], r.north[lat])), soft64_from_i32(p.nlon));
    const ulong area_s = soft64_div(soft64_mul(half_bits,
        soft64_mul(r.weights[lat], r.south[lat])), soft64_from_i32(p.nlon));
    ulong column = 0ul;
    for (uint k = 0; k < uint(p.nlevel); ++k) {
        const uint i = k * ncell + col;
        const ulong speed2 = soft64_add(
            soft64_mul(r.u[i], r.u[i]), soft64_mul(r.v[i], r.v[i]));
        const ulong half_speed2 = soft64_mul(half_bits, speed2);
        column = soft64_add(column, soft64_mul(
            half_speed2, soft64_div(r.dp[i], gravity)));
        if (k == uint(p.level_index)) {
            r.columns[2u * ncell + col] = soft64_mul(half_speed2, area_n);
            r.columns[3u * ncell + col] = soft64_mul(half_speed2, area_s);
        }
    }
    r.columns[col] = soft64_mul(column, area_n);
    r.columns[ncell + col] = soft64_mul(column, area_s);
}

kernel void global_insitu_energy_reduce_fragment(
    constant GlobalInsituEnergyReduceFragmentResources &r [[buffer(0)]],
    constant GlobalInsituEnergyReduceFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.nterm) * uint(p.nchunk);
    if (tid >= total) return;
    const uint term = tid / uint(p.nchunk), chunk = tid % uint(p.nchunk);
    const uint start = chunk * uint(p.chunk_size);
    const uint end = min(uint(p.ncell), start + uint(p.chunk_size));
    ulong value = 0ul;
    for (uint i = start; i < end; ++i)
        value = soft64_add(value, r.columns[term * uint(p.ncell) + i]);
    r.partial[tid] = value;
}

kernel void global_insitu_energy_finalize_fragment(
    constant GlobalInsituEnergyFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalInsituEnergyFinalizeFragmentParams &p [[buffer(1)]],
    uint term [[thread_position_in_grid]]) {
    if (term >= uint(p.nterm)) return;
    ulong value = 0ul;
    for (uint chunk = 0; chunk < uint(p.nchunk); ++chunk)
        value = soft64_add(value, r.partial[term * uint(p.nchunk) + chunk]);
    r.row[uint(p.row_offset) + term] = value;
}

kernel void global_insitu_energy_mask_modes_fragment(
    constant GlobalInsituEnergyMaskModesFragmentResources &r [[buffer(0)]],
    constant GlobalInsituEnergyMaskModesFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlevel) * uint(p.nterm) * uint(p.norder);
    if (tid >= count) return;
    const uint degree = (tid / uint(p.norder)) % uint(p.nterm);
    const ulong mask = r.mask[degree];
    r.filtered_vorticity[2u * tid] = soft64_mul(r.vorticity[2u * tid], mask);
    r.filtered_vorticity[2u * tid + 1u] = soft64_mul(r.vorticity[2u * tid + 1u], mask);
    r.filtered_divergence[2u * tid] = soft64_mul(r.divergence[2u * tid], mask);
    r.filtered_divergence[2u * tid + 1u] = soft64_mul(r.divergence[2u * tid + 1u], mask);
}

kernel void global_insitu_energy_parseval_fragment(
    constant GlobalInsituEnergyParsevalFragmentResources &r [[buffer(0)]],
    constant GlobalInsituEnergyParsevalFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nterm = uint(p.nterm), count = uint(p.nband) + 1u;
    const uint total = 2u * count + 2u * nterm;
    if (tid >= total) return;
    uint field, lo, hi;
    if (tid < 2u * count) {
        field = tid / count;
        lo = 1u;
        hi = nterm - 1u;
    } else {
        const uint offset = tid - 2u * count;
        field = offset / nterm;
        lo = hi = offset % nterm;
    }
    const device ulong *coeff = field == 0u ? r.vorticity : r.divergence;
    ulong result = 0ul;
    for (uint degree = lo; degree <= hi; ++degree) {
        ulong mask = 0x3ff0000000000000ul;
        if (tid < 2u * count && tid % count > 0u)
            mask = r.masks[(tid % count - 1u) * nterm + degree];
        if (mask == 0ul) continue;
        ulong power = 0ul;
        for (uint order = 0; order <= degree; ++order) {
            const uint i = ((uint(p.level_index) * nterm + degree)
                * uint(p.norder)) + order;
            const ulong real = coeff[2u * i], imag = coeff[2u * i + 1u];
            const ulong mag = soft64_add(soft64_mul(real, real), soft64_mul(imag, imag));
            power = soft64_add(power, soft64_mul(mag, r.order_weight[order]));
        }
        result = soft64_add(result, soft64_mul(
            soft64_mul(power, r.factor[degree]), mask));
    }
    r.row[uint(p.row_offset) + tid] = result;
}
