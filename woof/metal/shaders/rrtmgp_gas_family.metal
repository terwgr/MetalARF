#include <metal_stdlib>
#include "rrtmgp_interpolation_prepass_bindings.metalh"
#include "rrtmgp_interpolation_inline_reference_bindings.metalh"
#include "rrtmgp_gas_optics_bindings.metalh"
#include "rrtmgp_gas_optics_reference_bindings.metalh"
#include "rrtmgp_planck_sources_bindings.metalh"
#include "rrtmgp_gas_vmr_fill_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Integer metadata tables are transported in float32 Metal resources as raw
// int32 words. This keeps the current typed Metal ABI while preserving every
// index and flag bit. Interpolation metadata uses the same representation.
inline int rrtm_i(float word) { return as_type<int>(word); }
inline float rrtm_meta(int value) { return as_type<float>(uint(value)); }

inline int km_index(int jt, int je, int jp, int g, int neta,
                    int npressk, int ngpt) {
    return (((jt * neta + je) * npressk + jp) * ngpt + g);
}
inline int k2_index(int jt, int je, int k, int neta, int nk) {
    return ((jt * neta + je) * nk + k);
}

template <typename R, typename P>
inline void interpolation_cell(constant R &r, constant P &p, uint cell) {
    if (cell >= uint(p.ncell)) return;
    const float pressure = r.play[cell];
    const float temperature = r.tlay[cell];
    const float dtemp = (r.temp_ref[p.ntemp - 1] - r.temp_ref[0])
                        / float(p.ntemp - 1);
    const float dpress = (log(r.press_ref[p.npres - 1])
                          - log(r.press_ref[0])) / float(p.npres - 1);
    const int iatm = pressure > p.press_ref_trop ? 0 : 1;
    int jt = int((temperature - (r.temp_ref[0] - dtemp)) / dtemp);
    jt = max(1, min(p.ntemp - 1, jt)) - 1;
    const float ftemp = (temperature - r.temp_ref[jt]) / dtemp;
    const float locpress = 1.0f + (log(pressure) - log(r.press_ref[0])) / dpress;
    int jp1 = int(trunc(locpress));
    jp1 = max(1, min(p.npres - 1, jp1));
    r.iatm_out[cell] = rrtm_meta(iatm);
    r.jt_out[cell] = rrtm_meta(jt);
    r.jp_out[cell] = rrtm_meta((jp1 - 1) + iatm);
    r.ftemp_out[cell] = ftemp;
    r.fpress_out[cell] = locpress - float(jp1);
}

template <typename R, typename P>
inline void interpolation_reference_cell(constant R &r, constant P &p,
                                         uint cell) {
    if (cell >= uint(p.ncell)) return;
    const float pressure = r.play[cell];
    const float temperature = r.tlay[cell];
    const float dtemp = (r.temp_ref[p.ntemp - 1] - r.temp_ref[0])
                        / float(p.ntemp - 1);
    const float dpress = (log(r.press_ref[p.npres - 1])
                          - log(r.press_ref[0])) / float(p.npres - 1);
    const int iatm = pressure > p.press_ref_trop ? 0 : 1;
    int jt = int((temperature - (r.temp_ref[0] - dtemp)) / dtemp);
    jt = max(1, min(p.ntemp - 1, jt)) - 1;
    const float ftemp = (temperature - r.temp_ref[jt]) / dtemp;
    const float locpress = 1.0f + (log(pressure) - log(r.press_ref[0])) / dpress;
    int jp1 = int(trunc(locpress));
    jp1 = max(1, min(p.npres - 1, jp1));
    r.iatm_out[cell] = rrtm_meta(iatm);
    r.jt_out[cell] = rrtm_meta(jt);
    r.jp_out[cell] = rrtm_meta((jp1 - 1) + iatm);
    r.ftemp_out[cell] = ftemp;
    r.fpress_out[cell] = locpress - float(jp1);
}

kernel void rrtmgp_interpolation_prepass(
    constant RrtmgpInterpolationPrepassResources &r [[buffer(0)]],
    constant RrtmgpInterpolationPrepassParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    interpolation_cell(r, p, cell);
}

// Kept independent as in CUDA: this is the frozen inline-equation oracle.
kernel void rrtmgp_interpolation_inline_reference(
    constant RrtmgpInterpolationInlineReferenceResources &r [[buffer(0)]],
    constant RrtmgpInterpolationInlineReferenceParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    interpolation_reference_cell(r, p, cell);
}

inline float planck_interp(float t, int band, device const float *table,
                           int nplanck, int nband, float temp_min,
                           float delta) {
    const float val = (t - temp_min) / delta;
    int idx = int(trunc(val));
    idx = max(0, min(nplanck - 2, idx));
    const float frac = val - trunc(val);
    const float lo = table[idx * nband + band];
    return lo + frac * (table[(idx + 1) * nband + band] - lo);
}

template <typename R, typename P>
inline void planck_cell(constant R &r, constant P &p, uint tid) {
    if (p.nlay > 128) return;
    const int total = p.ncol * p.ngpt;
    if (tid >= uint(total)) return;
    const int col = int(tid) / p.ngpt;
    const int gpt = int(tid) - col * p.ngpt;
    const int nlev = p.nlay + 1;
    const int npressk = p.npres + 1;
    const float dplanck = (r.temp_ref[p.ntemp - 1] - r.temp_ref[0])
                          / float(p.nplanck - 1);
    const int sfc_lay = r.play[col * p.nlay]
                        < r.play[col * p.nlay + p.nlay - 1]
        ? p.nlay - 1 : 0;
    float pfrac[128];
    const int band = r.gpoint_bands[gpt];
    for (int lay = 0; lay < p.nlay; ++lay) {
        const int cell = col * p.nlay + lay;
        const float t = r.tlay[cell];
        const int iatm = rrtm_i(r.iatm_meta[cell]);
        const int iflav = rrtm_i(r.gpoint_flavor[iatm * p.ngpt + gpt]);
        const int gas1 = rrtm_i(r.flavor[iflav * 2]);
        const int gas2 = rrtm_i(r.flavor[iflav * 2 + 1]);
        const int jt = rrtm_i(r.jt_meta[cell]);
        const int jp = rrtm_i(r.jp_meta[cell]);
        const float ft = r.ftemp_meta[cell];
        const float fp = r.fpress_meta[cell];
        float pf = 0.0f;
        for (int itemp = 0; itemp < 2; ++itemp) {
            const int jtr = jt + itemp;
            const float ratio = r.vmr_ref[(iatm * (p.ngas + 1) + gas1)
                                           * p.ntemp + jtr]
                / r.vmr_ref[(iatm * (p.ngas + 1) + gas2) * p.ntemp + jtr];
            const float amount1 = gas1 == 0 ? 1.0f
                : r.vmr[cell * (p.ngas + 1) + gas1];
            const float amount2 = gas2 == 0 ? 1.0f
                : r.vmr[cell * (p.ngas + 1) + gas2];
            const float mix = amount1 + ratio * amount2;
            const float eta = mix > 2.0f * 1.17549435e-38f
                ? amount1 / mix : 0.5f;
            const float loce = eta * (p.neta - 1);
            const int je = min(int(loce), p.neta - 2);
            const float fe = loce - trunc(loce);
            const float wt = itemp == 0 ? 1.0f - ft : ft;
            const float w00 = (1.0f - fp) * (1.0f - fe) * wt;
            const float w10 = (1.0f - fp) * fe * wt;
            const float w01 = fp * (1.0f - fe) * wt;
            const float w11 = fp * fe * wt;
            pf += w00 * r.planck_fraction[km_index(
                       jtr, je, jp, gpt, p.neta, npressk, p.ngpt)]
                + w10 * r.planck_fraction[km_index(
                       jtr, je + 1, jp, gpt, p.neta, npressk, p.ngpt)]
                + w01 * r.planck_fraction[km_index(
                       jtr, je, jp + 1, gpt, p.neta, npressk, p.ngpt)]
                + w11 * r.planck_fraction[km_index(
                       jtr, je + 1, jp + 1, gpt, p.neta, npressk, p.ngpt)];
        }
        pfrac[lay] = pf;
        r.lay_source[cell * p.ngpt + gpt] = pf * planck_interp(
            t, band, r.totplnk, p.nplanck, p.nband, r.temp_ref[0], dplanck);
    }
    r.sfc_source[col * p.ngpt + gpt] = pfrac[sfc_lay] * planck_interp(
        r.tsfc[col], band, r.totplnk, p.nplanck, p.nband,
        r.temp_ref[0], dplanck);
    r.lev_source[(col * nlev) * p.ngpt + gpt] = pfrac[0] * planck_interp(
        r.tlev[col * nlev], band, r.totplnk, p.nplanck, p.nband,
        r.temp_ref[0], dplanck);
    for (int lev = 1; lev < p.nlay; ++lev) {
        r.lev_source[(col * nlev + lev) * p.ngpt + gpt] =
            sqrt(pfrac[lev - 1] * pfrac[lev]) * planck_interp(
                r.tlev[col * nlev + lev], band, r.totplnk, p.nplanck,
                p.nband, r.temp_ref[0], dplanck);
    }
    r.lev_source[(col * nlev + p.nlay) * p.ngpt + gpt] = pfrac[p.nlay - 1]
        * planck_interp(r.tlev[col * nlev + p.nlay], band, r.totplnk,
                        p.nplanck, p.nband, r.temp_ref[0], dplanck);
}

kernel void rrtmgp_planck_sources(
    constant RrtmgpPlanckSourcesResources &r [[buffer(0)]],
    constant RrtmgpPlanckSourcesParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    planck_cell(r, p, tid);
}

// Correctly round the source's explicit binary64-to-binary32 store conversion
// from its two raw little-endian words. No Metal double arithmetic is used.
inline uint rrtm_f64_to_f32_bits(uint lo, uint hi) {
    const uint sign = (hi >> 31) << 31;
    const uint exponent = (hi >> 20) & 0x7ffu;
    const ulong fraction = (ulong(hi & 0xfffffu) << 32) | ulong(lo);
    if (exponent == 0x7ffu) {
        if (fraction == 0ul) return sign | 0x7f800000u;
        uint payload = uint(fraction >> 29) & 0x7fffffu;
        payload |= 0x400000u;
        return sign | 0x7f800000u | payload;
    }
    if (exponent == 0u) return sign;
    int unbiased = int(exponent) - 1023;
    ulong significand = (1ul << 52) | fraction;
    if (unbiased > 127) return sign | 0x7f800000u;
    if (unbiased >= -126) {
        ulong rounded = significand >> 29;
        const ulong remainder = significand & ((1ul << 29) - 1ul);
        const ulong halfway = 1ul << 28;
        if (remainder > halfway || (remainder == halfway && (rounded & 1ul)))
            ++rounded;
        if (rounded == (1ul << 24)) {
            rounded >>= 1;
            ++unbiased;
            if (unbiased > 127) return sign | 0x7f800000u;
        }
        return sign | (uint(unbiased + 127) << 23)
             | (uint(rounded) & 0x7fffffu);
    }
    const int shift = -unbiased - 97;
    if (shift >= 64) return sign;
    ulong rounded = significand >> uint(shift);
    const ulong mask = (1ul << uint(shift)) - 1ul;
    const ulong remainder = significand & mask;
    const ulong halfway = 1ul << uint(shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1ul)))
        ++rounded;
    return sign | uint(rounded);
}

kernel void rrtmgp_gas_vmr_fill(
    constant RrtmgpGasVmrFillResources &r [[buffer(0)]],
    constant RrtmgpGasVmrFillParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.ncells) * uint(p.nslot)) return;
    const int cell = int(index) / p.nslot;
    const int slot = int(index) % p.nslot;
    float value = 0.0f;
    if (slot == p.h2o_index) value = r.qv[cell] * p.h2o_scale;
    if (slot == p.o3_index) {
        const uint lo = as_type<uint>(r.o3_words[2 * cell]);
        const uint hi = as_type<uint>(r.o3_words[2 * cell + 1]);
        value = as_type<float>(rrtm_f64_to_f32_bits(lo, hi));
    }
    for (int t = 0; t < p.ntrace; ++t)
        if (rrtm_i(r.trace_index[t]) == slot)
            value = r.trace_value[t];
    r.vmr[index] = value;
}

// Gas-optics work is independent for each (cell,g-point). The CUDA shared
// scaling/flavor staging only removes repeated arithmetic; this single-output
// mapping recomputes the same values and preserves each source operation and
// ascending CSR addition order without warp or dynamic shared memory.
template <typename R, typename P>
inline void gas_optics_one(constant R &r, constant P &p, uint output,
                           bool optimized) {
    const uint count = uint(p.ncol) * uint(p.nlay) * uint(p.ngpt);
    if (output >= count) return;
    const int cell = int(output / uint(p.ngpt));
    const int gpt = int(output - uint(cell) * uint(p.ngpt));
    const int col = cell / p.nlay;
    const int lay = cell - col * p.nlay;
    const int npressk = p.npres + 1;
    const float avogad = 6.02214076e23f;
    const float m_dry = 0.028964f;
    const float m_h2o = 0.018016f;
    const float grav = 9.80665f;
    const float pressure = r.play[cell];
    const float temperature = r.tlay[cell];
    const float h2o_vmr = r.vmr[cell * (p.ngas + 1) + p.idx_h2o];
    const float fact_dry = 1.0f / (1.0f + h2o_vmr);
    const float m_air = optimized
        ? fma(m_h2o, h2o_vmr, m_dry) * fact_dry
        : (m_dry + m_h2o * h2o_vmr) * fact_dry;
    const float dp = fabs(r.plev[col * (p.nlay + 1) + lay + 1]
                          - r.plev[col * (p.nlay + 1) + lay]);
    const float col_dry = dp * avogad * fact_dry
                          / (10000.0f * m_air * grav);
    const int iatm = rrtm_i(r.iatm_meta[cell]);
    const int jt = rrtm_i(r.jt_meta[cell]);
    const int jp = rrtm_i(r.jp_meta[cell]);
    const float ftemp = r.ftemp_meta[cell];
    const float fpress = r.fpress_meta[cell];
    const int nk = iatm == 0 ? p.nk_lower : p.nk_upper;
    device const float *kminor = iatm == 0 ? r.kminor_lower : r.kminor_upper;
    device const float *limits = iatm == 0
        ? r.minor_limits_lower : r.minor_limits_upper;
    device const float *density = iatm == 0
        ? r.density_lower : r.density_upper;
    device const float *complement = iatm == 0
        ? r.complement_lower : r.complement_upper;
    device const float *idx_minor = iatm == 0
        ? r.idx_minor_lower : r.idx_minor_upper;
    device const float *idx_scaling = iatm == 0
        ? r.idx_scaling_lower : r.idx_scaling_upper;
    device const float *starts = iatm == 0
        ? r.kminor_start_lower : r.kminor_start_upper;
    device const float *mstart = iatm == 0
        ? r.minor_gpt_start_lower : r.minor_gpt_start_upper;
    device const float *mlist = iatm == 0
        ? r.minor_gpt_list_lower : r.minor_gpt_list_upper;
    const int nminor = iatm == 0 ? p.nminor_lower : p.nminor_upper;
    const int iflav = rrtm_i(r.gpoint_flavor[iatm * p.ngpt + gpt]);
    int je[2];
    float fm[2][2];
    float cmix[2];
    float tau_abs = 0.0f;
    const int gas1 = rrtm_i(r.flavor[iflav * 2]);
    const int gas2 = rrtm_i(r.flavor[iflav * 2 + 1]);
    for (int itemp = 0; itemp < 2; ++itemp) {
        const int jtr = jt + itemp;
        const float ref1 = r.vmr_ref[(iatm * (p.ngas + 1) + gas1)
                                      * p.ntemp + jtr];
        const float ref2 = r.vmr_ref[(iatm * (p.ngas + 1) + gas2)
                                      * p.ntemp + jtr];
        const float ratio = ref1 / ref2;
        const float col1 = gas1 == 0 ? col_dry
            : r.vmr[cell * (p.ngas + 1) + gas1] * col_dry;
        const float col2 = gas2 == 0 ? col_dry
            : r.vmr[cell * (p.ngas + 1) + gas2] * col_dry;
        cmix[itemp] = optimized ? fma(ratio, col2, col1)
                                : col1 + ratio * col2;
        const float eta = cmix[itemp] > 2.0f * 1.17549435e-38f
            ? col1 / cmix[itemp] : 0.5f;
        const float loceta = eta * (p.neta - 1);
        je[itemp] = min(int(loceta), p.neta - 2);
        const float feta = loceta - trunc(loceta);
        const float wt = itemp == 0 ? 1.0f - ftemp : ftemp;
        fm[0][itemp] = optimized ? (1.0f - feta) * wt
                                 : (1.0f - feta) * wt;
        fm[1][itemp] = feta * wt;
        const float w00 = (1.0f - fpress) * fm[0][itemp];
        const float w10 = (1.0f - fpress) * fm[1][itemp];
        const float w01 = fpress * fm[0][itemp];
        const float w11 = fpress * fm[1][itemp];
        if (optimized) {
            const float major01 = fma(w00,
                r.kmajor[km_index(jtr, je[itemp], jp, gpt, p.neta,
                                 npressk, p.ngpt)],
                w10 * r.kmajor[km_index(jtr, je[itemp] + 1, jp, gpt,
                                        p.neta, npressk, p.ngpt)]);
            const float major2 = fma(w01,
                r.kmajor[km_index(jtr, je[itemp], jp + 1, gpt, p.neta,
                                 npressk, p.ngpt)], major01);
            const float major3 = fma(w11,
                r.kmajor[km_index(jtr, je[itemp] + 1, jp + 1, gpt,
                                 p.neta, npressk, p.ngpt)], major2);
            tau_abs = fma(cmix[itemp], major3, tau_abs);
        } else {
            tau_abs += cmix[itemp] * (
                w00 * r.kmajor[km_index(jtr, je[itemp], jp, gpt, p.neta,
                                        npressk, p.ngpt)]
              + w10 * r.kmajor[km_index(jtr, je[itemp] + 1, jp, gpt,
                                        p.neta, npressk, p.ngpt)]
              + w01 * r.kmajor[km_index(jtr, je[itemp], jp + 1, gpt,
                                        p.neta, npressk, p.ngpt)]
              + w11 * r.kmajor[km_index(jtr, je[itemp] + 1, jp + 1, gpt,
                                        p.neta, npressk, p.ngpt)]);
        }
    }
    const int q_end = rrtm_i(mstart[gpt + 1]);
    for (int qi = rrtm_i(mstart[gpt]); qi < q_end; ++qi) {
        const int m = rrtm_i(mlist[qi]);
        const int gs = rrtm_i(limits[m * p.nbound]);
        const int im = rrtm_i(idx_minor[m]);
        float scaling = im == 0 ? col_dry
            : r.vmr[cell * (p.ngas + 1) + im] * col_dry;
        if (rrtm_i(density[m]) != 0) {
            scaling *= 0.01f * pressure / temperature;
            const int iscale = rrtm_i(idx_scaling[m]);
            if (iscale > 0) {
                const float special = r.vmr[cell * (p.ngas + 1) + iscale]
                                      / (1.0f + h2o_vmr);
                scaling *= rrtm_i(complement[m]) ? 1.0f - special : special;
            }
        }
        const int kk = rrtm_i(starts[m]) + (gpt - gs);
        if (optimized) {
            const float minor01 = fma(fm[0][0],
                kminor[k2_index(jt, je[0], kk, p.neta, nk)],
                fm[1][0] * kminor[k2_index(jt, je[0] + 1, kk, p.neta, nk)]);
            const float minor2 = fma(fm[0][1],
                kminor[k2_index(jt + 1, je[1], kk, p.neta, nk)], minor01);
            const float kval = fma(fm[1][1],
                kminor[k2_index(jt + 1, je[1] + 1, kk, p.neta, nk)], minor2);
            tau_abs = fma(scaling, kval, tau_abs);
        } else {
            const float kval = fm[0][0]
                    * kminor[k2_index(jt, je[0], kk, p.neta, nk)]
                + fm[1][0]
                    * kminor[k2_index(jt, je[0] + 1, kk, p.neta, nk)]
                + fm[0][1]
                    * kminor[k2_index(jt + 1, je[1], kk, p.neta, nk)]
                + fm[1][1]
                    * kminor[k2_index(jt + 1, je[1] + 1, kk, p.neta, nk)];
            tau_abs += scaling * kval;
        }
    }
    if (p.is_sw != 0) {
        device const float *kr = r.rayleigh + iatm * p.ntemp * p.neta * p.ngpt;
        float ray_k;
        if (optimized) {
            const float ray01 = fma(fm[0][0],
                kr[(jt * p.neta + je[0]) * p.ngpt + gpt],
                fm[1][0] * kr[(jt * p.neta + je[0] + 1) * p.ngpt + gpt]);
            const float ray2 = fma(fm[0][1],
                kr[((jt + 1) * p.neta + je[1]) * p.ngpt + gpt], ray01);
            ray_k = fma(fm[1][1],
                kr[((jt + 1) * p.neta + je[1] + 1) * p.ngpt + gpt], ray2);
        } else {
            ray_k = fm[0][0] * kr[(jt * p.neta + je[0]) * p.ngpt + gpt]
                  + fm[1][0] * kr[(jt * p.neta + je[0] + 1) * p.ngpt + gpt]
                  + fm[0][1] * kr[((jt + 1) * p.neta + je[1]) * p.ngpt + gpt]
                  + fm[1][1] * kr[((jt + 1) * p.neta + je[1] + 1) * p.ngpt + gpt];
        }
        const float tau_ray = ray_k * col_dry * (1.0f + h2o_vmr);
        const float total = tau_abs + tau_ray;
        r.tau[output] = total;
        r.ssa[output] = tau_ray / fmax(3.0f * 1.17549435e-38f, total);
    } else {
        r.tau[output] = tau_abs;
    }
    (void)nminor;
}

kernel void rrtmgp_gas_optics(
    constant RrtmgpGasOpticsResources &r [[buffer(0)]],
    constant RrtmgpGasOpticsParams &p [[buffer(1)]],
    uint output [[thread_position_in_grid]]) {
    gas_optics_one(r, p, output, true);
}

kernel void rrtmgp_gas_optics_reference(
    constant RrtmgpGasOpticsReferenceResources &r [[buffer(0)]],
    constant RrtmgpGasOpticsReferenceParams &p [[buffer(1)]],
    uint output [[thread_position_in_grid]]) {
    gas_optics_one(r, p, output, false);
}
