#include <metal_stdlib>
#include "rrtmgp_cloud_optics_bindings.metalh"
#include "rrtmgp_finalize_cloud_lw_bindings.metalh"
#include "rrtmgp_finalize_cloud_sw_bindings.metalh"
#include "rrtmgp_clear_upper_layers_bindings.metalh"
#include "rrtmgp_store_model_flux_pair_bindings.metalh"
#include "rrtmgp_prepare_sw_inputs_bindings.metalh"
#include "rrtmgp_append_profile_fp32_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

inline float interp_table(device const float *table, int size, int band,
                          int nband, float fraction) {
    const ulong lo = ulong(size) * ulong(nband) + ulong(band);
    return table[lo] + fraction * (table[lo + ulong(nband)] - table[lo]);
}

inline float interp_ice(device const float *table, int size, int band,
                        int nband, int nrgh, float fraction) {
    const ulong lo = (ulong(size) * ulong(nband) + ulong(band))
                   * ulong(nrgh) + 1ul;
    const ulong next = (ulong(size + 1) * ulong(nband) + ulong(band))
                     * ulong(nrgh) + 1ul;
    return table[lo] + fraction * (table[next] - table[lo]);
}

kernel void rrtmgp_cloud_optics(
    constant RrtmgpCloudOpticsResources &r [[buffer(0)]],
    constant RrtmgpCloudOpticsParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.ncell * p.nband);
    if (tid >= total) return;
    const int cell = int(tid) / p.nband;
    const int band = int(tid) - cell * p.nband;
    float ltau = 0.0f, lscatter = 0.0f, lgscatter = 0.0f;
    float itau = 0.0f, iscatter = 0.0f, igscatter = 0.0f;
    if (r.clwp[cell] > 0.0f) {
        const float pos = (r.reliq[cell] - p.liq_offset) / p.liq_step;
        const int isize = max(0, min(int(floor(pos)), p.nliq - 2));
        const float fraction = pos - float(isize);
        ltau = r.clwp[cell] * interp_table(r.extliq, isize, band, p.nband, fraction);
        lscatter = ltau * interp_table(r.ssaliq, isize, band, p.nband, fraction);
        lgscatter = lscatter * interp_table(r.asyliq, isize, band, p.nband, fraction);
    }
    if (r.ciwp[cell] > 0.0f) {
        const float pos = (r.dgice[cell] - p.ice_offset) / p.ice_step;
        const int isize = max(0, min(int(floor(pos)), p.nice - 2));
        const float fraction = pos - float(isize);
        itau = r.ciwp[cell] * interp_ice(r.extice, isize, band, p.nband, p.nrgh, fraction);
        iscatter = itau * interp_ice(r.ssaice, isize, band, p.nband, p.nrgh, fraction);
        igscatter = iscatter * interp_ice(r.asyice, isize, band, p.nband, p.nrgh, fraction);
    }
    const float total_tau = ltau + itau;
    const float scatter = lscatter + iscatter;
    r.tau[tid] = total_tau;
    r.ssa[tid] = scatter / fmax(1.1920928955078125e-7f, total_tau);
    r.asym[tid] = (lgscatter + igscatter) / fmax(1.1920928955078125e-7f, scatter);
}

kernel void rrtmgp_finalize_cloud_lw(
    constant RrtmgpFinalizeCloudLwResources &r [[buffer(0)]],
    constant RrtmgpFinalizeCloudLwParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const int gpt = int(tid) % p.ngpt;
    const int cell = int(tid) / p.ngpt;
    const int band = int(r.gpoint_bands[ulong(gpt)]);
    const ulong bi = ulong(cell) * ulong(p.nband) + ulong(band);
    float tc = r.tau_cloud[bi];
    if (p.have_mask != 0) tc = tc * r.cloud_mask[tid];
    const float one_minus_ssa = 1.0f - r.ssa_cloud[bi];
    const float cloud_absorption = tc * one_minus_ssa;
    r.tau[tid] = r.tau_gas[tid] + cloud_absorption;
}

kernel void rrtmgp_finalize_cloud_sw(
    constant RrtmgpFinalizeCloudSwResources &r [[buffer(0)]],
    constant RrtmgpFinalizeCloudSwParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const int gpt = int(tid) % p.ngpt;
    const int cell = int(tid) / p.ngpt;
    const int band = int(r.gpoint_bands[ulong(gpt)]);
    const ulong bi = ulong(cell) * ulong(p.nband) + ulong(band);
    float tc = r.tau_cloud[bi];
    if (p.have_mask != 0) tc = tc * r.cloud_mask[tid];
    const float wc = r.ssa_cloud[bi];
    const float gc = r.g_cloud[bi];
    const float floor = 3.0f * 1.17549435e-38f;
    const float total_tau = r.tau_gas[tid] + tc;
    const float gas_scatter = r.tau_gas[tid] * r.ssa_gas[tid];
    const float cloud_scatter = tc * wc;
    const float scatter = gas_scatter + cloud_scatter;
    const float total_ssa = scatter / fmax(floor, total_tau);
    const float gas_gscatter = gas_scatter * 0.0f;
    const float cloud_gscatter = cloud_scatter * gc;
    const float gscatter = gas_gscatter + cloud_gscatter;
    const float total_g = gscatter / fmax(floor, scatter);
    const float f = total_g * total_g;
    const float wf = total_ssa * f;
    r.tau[tid] = (1.0f - wf) * total_tau;
    r.ssa[tid] = (total_ssa - wf) / fmax(floor, 1.0f - wf);
    r.asym[tid] = (total_g - f) / fmax(floor, 1.0f - f);
}

kernel void rrtmgp_clear_upper_layers(
    constant RrtmgpClearUpperLayersResources &r [[buffer(0)]],
    constant RrtmgpClearUpperLayersParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int total = p.ncol * p.nlay;
    if (tid >= uint(total)) return;
    const int col = int(tid) / p.nlay;
    const int lay = int(tid) - col * p.nlay;
    if (lay < p.model_nlay) {
        const ulong src = ulong(col) * ulong(p.model_nlay) + ulong(lay);
        r.out_0[tid] = r.in_0[src]; r.out_1[tid] = r.in_1[src];
        r.out_2[tid] = r.in_2[src]; r.out_3[tid] = r.in_3[src];
        r.out_4[tid] = r.in_4[src];
    } else {
        r.out_0[tid] = 0.0f; r.out_1[tid] = 0.0f; r.out_2[tid] = 0.0f;
        r.out_3[tid] = 0.0f; r.out_4[tid] = 0.0f;
    }
}

kernel void rrtmgp_store_model_flux_pair(
    constant RrtmgpStoreModelFluxPairResources &r [[buffer(0)]],
    constant RrtmgpStoreModelFluxPairParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int total = p.ncol * p.model_nlev;
    if (tid >= uint(total)) return;
    const int col = int(tid) / p.model_nlev;
    const int lev = int(tid) - col * p.model_nlev;
    if (p.apply_daylight == 0 || r.daylight[col] != 0.0f) {
        const ulong src = ulong(col) * ulong(p.source_nlev) + ulong(lev);
        r.out_up[tid] = r.up[src]; r.out_dn[tid] = r.dn[src];
    } else {
        r.out_up[tid] = 0.0f; r.out_dn[tid] = 0.0f;
    }
}

kernel void rrtmgp_prepare_sw_inputs(
    constant RrtmgpPrepareSwInputsResources &r [[buffer(0)]],
    constant RrtmgpPrepareSwInputsParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint gpt_total = uint(p.ncol * p.ngpt);
    const uint layer_total = uint(p.ncol * p.nlay);
    if (tid < gpt_total) {
        r.albedo_gpt[tid] = r.albedo[tid / uint(p.ngpt)];
        r.inc_gpt[tid] = r.solar[tid % uint(p.ngpt)];
    }
    if (tid < layer_total) r.mu0[tid] = r.mu[tid / uint(p.nlay)];
}

kernel void rrtmgp_append_profile_fp32(
    constant RrtmgpAppendProfileFp32Resources &r [[buffer(0)]],
    constant RrtmgpAppendProfileFp32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlay = p.model_nlay + p.upper_nlay;
    const int total_levels = p.ncol * (nlay + 1);
    if (tid >= uint(total_levels)) return;
    const int col = int(tid) / (nlay + 1);
    const int lev = int(tid) - col * (nlay + 1);
    if (lev <= p.model_nlay) {
        const ulong src = ulong(col) * ulong(p.model_nlay + 1) + ulong(lev);
        r.out_plev[tid] = r.model_plev[src];
        r.out_tlev[tid] = r.model_tlev[src];
    } else {
        const int j = lev - p.model_nlay - 1;
        const ulong src = ulong(col) * ulong(p.sec) + ulong(j) * ulong(p.sel);
        r.out_plev[tid] = r.upper_plev[src];
        r.out_tlev[tid] = r.upper_tlev[ulong(col) * ulong(p.sic) + ulong(j) * ulong(p.sil)];
    }
    if (lev < nlay) {
        const ulong target = ulong(col) * ulong(nlay) + ulong(lev);
        if (lev < p.model_nlay) {
            const ulong src = ulong(col) * ulong(p.model_nlay) + ulong(lev);
            r.out_play[target] = r.model_play[src];
            r.out_tlay[target] = r.model_tlay[src];
            r.out_qv[target] = r.model_qv[src];
        } else {
            const int j = lev - p.model_nlay;
            r.out_play[target] = r.upper_play[ulong(col) * ulong(p.spc) + ulong(j) * ulong(p.spl)];
            r.out_tlay[target] = r.upper_tlay[ulong(col) * ulong(p.stc) + ulong(j) * ulong(p.stl)];
            r.out_qv[target] = r.upper_qv[ulong(col) * ulong(p.sqc) + ulong(j) * ulong(p.sql)];
        }
    }
}

struct RrtmgpMaskWordsToFloatResources {
    device const float *words;
    device float *mask;
};

struct RrtmgpMaskWordsToFloatParams {
    int n;
    int nwords;
};

kernel void rrtmgp_mask_words_to_float(
    constant RrtmgpMaskWordsToFloatResources &r [[buffer(0)]],
    constant RrtmgpMaskWordsToFloatParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n) || (tid >> 2) >= uint(p.nwords)) return;
    const uint packed = as_type<uint>(r.words[tid >> 2]);
    const uint value = (packed >> ((tid & 3u) * 8u)) & 0xffu;
    r.mask[tid] = float(value != 0u);
}

struct RrtmgpExpandBandToGpointResources {
    device const float *values;
    device const float *gpoint_bands;
    device float *output;
};

struct RrtmgpExpandBandToGpointParams {
    int ncol;
    int nband;
    int ngpt;
};

kernel void rrtmgp_expand_band_to_gpoint(
    constant RrtmgpExpandBandToGpointResources &r [[buffer(0)]],
    constant RrtmgpExpandBandToGpointParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint n = uint(p.ncol * p.ngpt);
    if (tid >= n) return;
    const int col = int(tid) / p.ngpt;
    const int gpt = int(tid) - col * p.ngpt;
    const int band = int(r.gpoint_bands[gpt]);
    r.output[tid] = r.values[col * p.nband + band];
}
