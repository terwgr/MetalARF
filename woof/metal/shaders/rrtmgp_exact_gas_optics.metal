#include <metal_stdlib>
#include "soft_word32.metalh"
#include "rrtmgp_exact_gas_optics_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

inline int rrtm_i(float word) { return as_type<int>(word); }
inline int km_index(int jt, int je, int jp, int g, int neta, int npressk, int ngpt) {
    return (((jt * neta + je) * npressk + jp) * ngpt + g);
}
inline int k2_index(int jt, int je, int k, int neta, int nk) {
    return ((jt * neta + je) * nk + k);
}

kernel void rrtmgp_exact_gas_optics(
    constant RrtmgpExactGasOpticsResources &r [[buffer(0)]],
    constant RrtmgpExactGasOpticsParams &p [[buffer(1)]],
    uint output [[thread_position_in_grid]]) {
    const uint count = uint(p.ncol) * uint(p.nlay) * uint(p.ngpt);
    if (output >= count) return;
    const int cell = int(output / uint(p.ngpt));
    const int gpt = int(output - uint(cell) * uint(p.ngpt));
    const int col = cell / p.nlay;
    const int lay = cell - col * p.nlay;
    const int npressk = p.npres + 1;
    const F avogad = 6.02214076e23f;
    const F m_dry = 0.028964f;
    const F m_h2o = 0.018016f;
    const F grav = 9.80665f;
    const F pressure = r.play[cell];
    const F temperature = r.tlay[cell];
    const F h2o_vmr = r.vmr[cell * (p.ngas + 1) + p.idx_h2o];
    const F fact_dry = aa_div_impl(F(1.0f), F(1.0f) + h2o_vmr);
    const F m_air = (m_dry + m_h2o * h2o_vmr) * fact_dry;
    const F dp = fabs(F(r.plev[col * (p.nlay + 1) + lay])
                          - F(r.plev[col * (p.nlay + 1) + lay + 1]));
    const F col_dry = aa_div_impl(F(10.0f) * dp * avogad * fact_dry,
                                   F(1000.0f) * m_air * 100.0f * grav);
    const F col_h2o = h2o_vmr * col_dry;
    const int iatm = rrtm_i(r.iatm_meta[cell]);
    const int jt = rrtm_i(r.jt_meta[cell]);
    const int jp = rrtm_i(r.jp_meta[cell]);
    const F ftemp = r.ftemp_meta[cell];
    const F fpress = r.fpress_meta[cell];
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
    const int iflav = rrtm_i(r.gpoint_flavor[iatm * p.ngpt + gpt]);
    int je[2];
    F fm[2][2];
    F cmix[2];
    F part[2];
    const int gas1 = rrtm_i(r.flavor[iflav * 2]);
    const int gas2 = rrtm_i(r.flavor[iflav * 2 + 1]);
    for (int itemp = 0; itemp < 2; ++itemp) {
        const int jtr = jt + itemp;
        const F ref1 = r.vmr_ref[(iatm * (p.ngas + 1) + gas1)
                                      * p.ntemp + jtr];
        const F ref2 = r.vmr_ref[(iatm * (p.ngas + 1) + gas2)
                                      * p.ntemp + jtr];
        const F ratio = aa_div_impl(ref1, ref2);
        const F col1 = gas1 == 0 ? col_dry
            : F(r.vmr[cell * (p.ngas + 1) + gas1]) * col_dry;
        const F col2 = gas2 == 0 ? col_dry
            : F(r.vmr[cell * (p.ngas + 1) + gas2]) * col_dry;
        cmix[itemp] = col1 + ratio * col2;
        const F eta = cmix[itemp] > F(2.0f * 1.17549435e-38f)
            ? aa_div_impl(col1, cmix[itemp]) : F(0.5f);
        const F loceta = eta * F(float(p.neta - 1));
        je[itemp] = min(int(loceta), p.neta - 2);
        const F feta = loceta - F(float(je[itemp]));
        const F wt = itemp == 0 ? F(1.0f) - ftemp : ftemp;
        fm[0][itemp] = (F(1.0f) - feta) * wt;
        fm[1][itemp] = feta * wt;
        const F w00 = (F(1.0f) - fpress) * fm[0][itemp];
        const F w10 = (F(1.0f) - fpress) * fm[1][itemp];
        const F w01 = fpress * fm[0][itemp];
        const F w11 = fpress * fm[1][itemp];
        part[itemp] = (
                w00 * r.kmajor[km_index(jtr, je[itemp], jp, gpt, p.neta,
                                        npressk, p.ngpt)]
              + w10 * r.kmajor[km_index(jtr, je[itemp] + 1, jp, gpt,
                                        p.neta, npressk, p.ngpt)]
              + w01 * r.kmajor[km_index(jtr, je[itemp], jp + 1, gpt,
                                        p.neta, npressk, p.ngpt)]
              + w11 * r.kmajor[km_index(jtr, je[itemp] + 1, jp + 1, gpt,
                                        p.neta, npressk, p.ngpt)]);
    }
    F tau_abs = F(0.0f) + (cmix[0] * part[0] + cmix[1] * part[1]);
    const int q_end = rrtm_i(mstart[gpt + 1]);
    for (int qi = rrtm_i(mstart[gpt]); qi < q_end; ++qi) {
        const int m = rrtm_i(mlist[qi]);
        const int gs = rrtm_i(limits[m * 2]);
        const int im = rrtm_i(idx_minor[m]);
        F scaling = im == 0 ? col_dry
            : F(r.vmr[cell * (p.ngas + 1) + im]) * col_dry;
        if (rrtm_i(density[m]) != 0) {
            scaling = scaling * aa_div_impl(F(0.01f) * pressure, temperature);
            const int iscale = rrtm_i(idx_scaling[m]);
            if (iscale > 0) {
                const F vmr_fact = aa_div_impl(F(1.0f), col_dry);
                const F dry_fact = aa_div_impl(F(1.0f), F(1.0f) + col_h2o * vmr_fact);
                const F col_s = F(r.vmr[cell * (p.ngas + 1) + iscale]) * col_dry;
                scaling = rrtm_i(complement[m])
                    ? scaling * (F(1.0f) - col_s * vmr_fact * dry_fact)
                    : scaling * (col_s * vmr_fact * dry_fact);
            }
        }
        const int kk = rrtm_i(starts[m]) + (gpt - gs);
        const F kval = fm[0][0]
                    * kminor[k2_index(jt, je[0], kk, p.neta, nk)]
                + fm[1][0]
                    * kminor[k2_index(jt, je[0] + 1, kk, p.neta, nk)]
                + fm[0][1]
                    * kminor[k2_index(jt + 1, je[1], kk, p.neta, nk)]
                + fm[1][1]
                    * kminor[k2_index(jt + 1, je[1] + 1, kk, p.neta, nk)];
        tau_abs = tau_abs + scaling * kval;
    }
    if (p.is_sw != 0) {
        device const float *kr = r.rayleigh + iatm * p.ntemp * p.neta * p.ngpt;
        F ray_k;
        ray_k = fm[0][0] * kr[(jt * p.neta + je[0]) * p.ngpt + gpt]
                  + fm[1][0] * kr[(jt * p.neta + je[0] + 1) * p.ngpt + gpt]
                  + fm[0][1] * kr[((jt + 1) * p.neta + je[1]) * p.ngpt + gpt]
                  + fm[1][1] * kr[((jt + 1) * p.neta + je[1] + 1) * p.ngpt + gpt];
        const F tau_ray = ray_k * (col_h2o + col_dry);
        const F total = tau_abs + tau_ray;
        r.tau[output] = float(total);
        r.ssa[output] = float(total > F(2.0f * 1.17549435e-38f)
            ? aa_div_impl(tau_ray, total) : F(0.0f));
    } else {
        r.tau[output] = float(tau_abs);
    }
}
