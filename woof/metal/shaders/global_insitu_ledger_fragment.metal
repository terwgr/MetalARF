// Resident Global in-situ ledger: vertical column accounting, then two
// bounded reductions. All real values use the existing soft binary64 ABI.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_insitu_ledger_fragment_bindings.metalh"
using namespace metal;

static inline ulong ledger_max(ulong a, ulong b) { return soft64_greater(a, b) ? a : b; }
static inline ulong ledger_min(ulong a, ulong b) { return soft64_less(a, b) ? a : b; }

kernel void global_insitu_ledger_columns_fragment(
    constant GlobalInsituLedgerColumnsFragmentResources &r [[buffer(0)]],
    constant GlobalInsituLedgerColumnsFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncell = uint(p.nlat) * uint(p.nlon);
    if (col >= ncell) return;
    const uint lat = col / uint(p.nlon);
    const ulong zero = 0ul;
    const ulong half_bits = 0x3fe0000000000000ul;
    const ulong gravity = 0x40239d013a92a305ul; // 9.80665
    const ulong cp = 0x408f600000000000ul;      // 1004
    const ulong lv = 0x414312d000000000ul;      // 2.5e6
    const ulong lf = 0x411462c000000000ul;      // 334000
    const ulong rho = 0x408f400000000000ul;     // 1000
    ulong water_qv = zero, water_qc = zero, water_qr = zero;
    ulong water_qi = zero, water_qs = zero, water_qg = zero;
    ulong dry = zero, kinetic = zero, potential = zero, latent_vapor = zero;
    ulong latent_frozen = zero, column_mass = zero, relative = zero;
    ulong tmin = r.temperature[col], tmax = tmin;
    ulong psmin = r.ps[col], psmax = psmin;
    ulong windmax2 = zero;
    ulong min_qv = r.qv[col], min_qc = r.qc[col], min_qr = r.qr[col];
    ulong min_qi = r.qi[col], min_qs = r.qs[col], min_qg = r.qg[col];
    for (uint k = 0; k < uint(p.nlevel); ++k) {
        const uint i = k * ncell + col;
        const ulong dp_g = soft64_div(r.dp[i], gravity);
        const ulong qv = r.qv[i], qc = r.qc[i], qr = r.qr[i];
        const ulong qi = r.qi[i], qs = r.qs[i], qg = r.qg[i];
        const ulong temp = r.temperature[i], u = r.u[i], v = r.v[i];
        const ulong speed2 = soft64_add(soft64_mul(u, u), soft64_mul(v, v));
        water_qv = soft64_add(water_qv, soft64_mul(qv, dp_g));
        water_qc = soft64_add(water_qc, soft64_mul(qc, dp_g));
        water_qr = soft64_add(water_qr, soft64_mul(qr, dp_g));
        water_qi = soft64_add(water_qi, soft64_mul(qi, dp_g));
        water_qs = soft64_add(water_qs, soft64_mul(qs, dp_g));
        water_qg = soft64_add(water_qg, soft64_mul(qg, dp_g));
        dry = soft64_add(dry, soft64_mul(temp, dp_g));
        kinetic = soft64_add(kinetic, soft64_mul(
            soft64_mul(half_bits, speed2), dp_g));
        potential = soft64_add(potential, soft64_mul(r.geopotential[i], dp_g));
        latent_vapor = soft64_add(latent_vapor, soft64_mul(qv, dp_g));
        const ulong frozen = soft64_add(soft64_add(qi, qs), qg);
        latent_frozen = soft64_add(latent_frozen, soft64_mul(frozen, dp_g));
        column_mass = soft64_add(column_mass, dp_g);
        relative = soft64_add(relative, soft64_mul(u, dp_g));
        tmin = ledger_min(tmin, temp); tmax = ledger_max(tmax, temp);
        windmax2 = ledger_max(windmax2, speed2);
        min_qv = ledger_min(min_qv, qv); min_qc = ledger_min(min_qc, qc);
        min_qr = ledger_min(min_qr, qr); min_qi = ledger_min(min_qi, qi);
        min_qs = ledger_min(min_qs, qs); min_qg = ledger_min(min_qg, qg);
    }
    const ulong surface_water = r.surface_water[col];
    ulong soil = zero;
    for (uint layer = 0; layer < uint(p.nsoil); ++layer) {
        ulong soil_value = ledger_max(r.soil_fraction[layer * ncell + col], zero);
        soil_value = soft64_mul(soil_value, r.soil_thickness[layer]);
        soil_value = soft64_mul(soil_value, rho);
        soil_value = soft64_mul(soil_value, r.land_fraction[col]);
        soil = soft64_add(soil, soil_value);
    }
    const ulong native = soft64_add(
        ledger_max(r.native_canopy[col], zero),
        ledger_max(r.native_snow[col], zero));
    const ulong outflow = r.outflow[col];
    const ulong atmosphere = soft64_add(
        soft64_add(soft64_add(soft64_add(soft64_add(water_qv, water_qc),
            water_qr), water_qi), water_qs), water_qg);
    const ulong held = soft64_add(
        soft64_add(soft64_add(surface_water, soil), native), outflow);
    const ulong surface_potential = soft64_div(
        soft64_mul(r.surface_geopotential[col], r.ps[col]), gravity);
    const ulong dry_energy = soft64_mul(dry, cp);
    const ulong total_dry = soft64_add(soft64_add(dry_energy, kinetic), surface_potential);
    const ulong latent_v = soft64_mul(latent_vapor, lv);
    const ulong latent_f = soft64_mul(latent_frozen, lf);
    const ulong moist = soft64_sub(soft64_add(total_dry, latent_v), latent_f);
    const ulong axial = soft64_add(
        soft64_mul(relative, r.a_coslat[lat]),
        soft64_mul(column_mass, r.planetary[lat]));
    const ulong factor = soft64_div(
        soft64_mul(half_bits, r.weights[lat]), soft64_from_i32(p.nlon));
    device ulong *o = r.columns;
#define PUT(term, value) o[uint(term) * ncell + col] = soft64_mul((value), factor)
    PUT(0, r.ps[col]); PUT(1, water_qv); PUT(2, water_qc); PUT(3, water_qr);
    PUT(4, water_qi); PUT(5, water_qs); PUT(6, water_qg); PUT(7, atmosphere);
    PUT(8, surface_water); PUT(9, soil); PUT(10, native); PUT(11, outflow);
    PUT(12, soft64_add(atmosphere, held)); PUT(13, dry_energy); PUT(14, kinetic);
    PUT(15, potential); PUT(16, surface_potential); PUT(17, total_dry);
    PUT(18, latent_v); PUT(19, latent_f); PUT(20, moist);
    PUT(21, soft64_mul(relative, r.a_coslat[lat])); PUT(22, axial);
#undef PUT
    o[23u * ncell + col] = tmin; o[24u * ncell + col] = tmax;
    o[25u * ncell + col] = psmin; o[26u * ncell + col] = psmax;
    o[27u * ncell + col] = soft64_sqrt(windmax2);
    o[28u * ncell + col] = surface_water;
    o[29u * ncell + col] = min_qv; o[30u * ncell + col] = min_qc;
    o[31u * ncell + col] = min_qr; o[32u * ncell + col] = min_qi;
    o[33u * ncell + col] = min_qs; o[34u * ncell + col] = min_qg;
}

kernel void global_insitu_ledger_reduce_fragment(
    constant GlobalInsituLedgerReduceFragmentResources &r [[buffer(0)]],
    constant GlobalInsituLedgerReduceFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.nterm) * uint(p.nchunk);
    if (tid >= total) return;
    const uint term = tid / uint(p.nchunk), chunk = tid % uint(p.nchunk);
    const uint start = chunk * uint(p.chunk_size);
    const uint end = min(uint(p.ncell), start + uint(p.chunk_size));
    ulong value = r.columns[term * uint(p.ncell) + start];
    for (uint i = start + 1u; i < end; ++i) {
        const ulong candidate = r.columns[term * uint(p.ncell) + i];
        if (term <= 22u) value = soft64_add(value, candidate);
        else if (term == 24u || term == 26u || term == 27u)
            value = ledger_max(value, candidate);
        else value = ledger_min(value, candidate);
    }
    r.partial[tid] = value;
}

kernel void global_insitu_ledger_finalize_fragment(
    constant GlobalInsituLedgerFinalizeFragmentResources &r [[buffer(0)]],
    constant GlobalInsituLedgerFinalizeFragmentParams &p [[buffer(1)]],
    uint term [[thread_position_in_grid]]) {
    if (term >= uint(p.nterm)) return;
    ulong value = r.partial[term * uint(p.nchunk)];
    for (uint i = 1; i < uint(p.nchunk); ++i) {
        const ulong candidate = r.partial[term * uint(p.nchunk) + i];
        if (term <= 22u) value = soft64_add(value, candidate);
        else if (term == 24u || term == 26u || term == 27u)
            value = ledger_max(value, candidate);
        else value = ledger_min(value, candidate);
    }
    r.row[term] = value;
}
