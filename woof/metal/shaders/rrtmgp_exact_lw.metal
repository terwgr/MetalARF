// Binary32 operator order from the retained RRTMGP CUDA longwave solver.
#include <metal_stdlib>
#include "hexv842_mpas_thompson_column_math.metalh"
#include "rrtmgp_exact_lw_noscat_bindings.metalh"
#include "rrtmgp_exact_lw_flux_reduce_bindings.metalh"

using namespace metal;

static inline F lw_div(F a, F b) { return f32_word(soft32_div(a.bits, b.bits)); }
static inline F lw_sqrt(F a) { return f32_word(soft32_sqrt(a.bits)); }

static inline F lw_column_dry(F pa, F pb, F h2o) {
    const F pressure_span = fabs(pa - pb);
    const F reciprocal = lw_div(1.0f, 1.0f + h2o);
    const F air_mass = (F(0.028964f) + F(0.018016f) * h2o) * reciprocal;
    return lw_div(F(10.0f) * pressure_span * 6.02214076e23f * reciprocal,
                  F(1000.0f) * air_mass * 100.0f * 9.80665f);
}

static inline F lw_planck_fraction(
    constant RrtmgpExactLwNoscatResources &r,
    constant RrtmgpExactLwNoscatParams &p, int col, int lay, int gpt) {
    const int cell = col * p.nlay + lay;
    const int gases = p.pk_ngas + 1;
    const int level = col * (p.nlay + 1) + lay;
    const F dry = lw_column_dry(r.pk_plev[level], r.pk_plev[level + 1],
                               r.pk_vmr[cell * gases + p.pk_idx_h2o]);
    const int iatm = as_type<int>(r.pk_iatm[cell]);
    const int flavor = as_type<int>(r.pk_gpoint_flavor[iatm * p.ngpt + gpt]);
    const int gas1 = as_type<int>(r.pk_flavor[flavor * 2]);
    const int gas2 = as_type<int>(r.pk_flavor[flavor * 2 + 1]);
    const F col1 = gas1 == 0 ? dry : F(r.pk_vmr[cell * gases + gas1]) * dry;
    const F col2 = gas2 == 0 ? dry : F(r.pk_vmr[cell * gases + gas2]) * dry;
    const int jt = as_type<int>(r.pk_jt[cell]), jp = as_type<int>(r.pk_jp[cell]);
    const F ft = r.pk_ftemp[cell], fp = r.pk_fpress[cell];
    const int eta_stride = (p.pk_npres + 1) * p.ngpt;
    F part[2];
    for (int it = 0; it < 2; ++it) {
        const int jtr = jt + it;
        const int reference = iatm * gases * p.pk_ntemp + jtr;
        const F ratio = lw_div(r.pk_vmr_ref[reference + gas1 * p.pk_ntemp],
                               r.pk_vmr_ref[reference + gas2 * p.pk_ntemp]);
        const F mix = col1 + ratio * col2;
        const F eta = mix > F(2.0f * 1.17549435e-38f) ? lw_div(col1, mix) : F(0.5f);
        const F loceta = eta * F(float(p.pk_neta - 1));
        const int je = min(int(loceta), p.pk_neta - 2);
        const F feta = loceta - F(float(je));
        const F temp_weight = it == 0 ? F(1.0f) - ft : ft;
        const F f0 = (F(1.0f) - feta) * temp_weight;
        const F f1 = feta * temp_weight;
        const F f11 = (F(1.0f) - fp) * f0;
        const F f21 = (F(1.0f) - fp) * f1;
        const F f12 = fp * f0;
        const F f22 = fp * f1;
        const int index = (jtr * p.pk_neta + je) * eta_stride + jp * p.ngpt + gpt;
        part[it] = f11 * F(r.pk_planck_fraction[index])
                 + f21 * F(r.pk_planck_fraction[index + eta_stride])
                 + f12 * F(r.pk_planck_fraction[index + p.ngpt])
                 + f22 * F(r.pk_planck_fraction[index + eta_stride + p.ngpt]);
    }
    return part[0] + part[1];
}

static inline F lw_planck(F t, int band, device const float *table,
                          int nplanck, int nband, F temp_min, F delta_r) {
    const F value = (t - temp_min) * delta_r;
    const int index = max(0, min(nplanck - 2, int(value)));
    const F fraction = value - F(float(index));
    const F lo = table[index * nband + band];
    return lo + fraction * (F(table[(index + 1) * nband + band]) - lo);
}

kernel void rrtmgp_exact_lw_noscat(
    constant RrtmgpExactLwNoscatResources &r [[buffer(0)]],
    constant RrtmgpExactLwNoscatParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ncol * p.gpt_tile) || p.nlay > 128) return;
    const int col = int(gid) / p.gpt_tile;
    const int gl = int(gid) - col * p.gpt_tile;
    const int gpt = p.gpt0 + gl;
    if (gpt >= p.ngpt) return;
    const int nlev = p.nlay + 1, base = gl * nlev;
    const int top = p.top_at_1 ? 0 : p.nlay;
    const int sfc = p.top_at_1 ? p.nlay : 0;
    const F d = lw_div(1.0f, 0.6096748751f);
    const F pi = 3.14159265358979323846f;
    const F tau_thresh = lw_sqrt(lw_sqrt(0x1p-23f));
    F trans[128], src_up[128];

    const int band = p.pk_on ? as_type<int>(r.pk_gpoint_bands[gpt]) : 0;
    const F t0 = p.pk_on ? F(r.pk_temp_ref[0]) : F(0.0f);
    const F delta_r = p.pk_on
        ? lw_div(1.0f, lw_div(F(r.pk_temp_ref[p.pk_ntemp - 1]) - t0,
                              F(float(p.pk_nplanck - 1)))) : F(1.0f);
    const int first = p.top_at_1 ? 0 : p.nlay - 1;
    F prev = p.pk_on && first > 0 ? lw_planck_fraction(r, p, col, first - 1, gpt) : F(0.0f);
    F cur = p.pk_on ? lw_planck_fraction(r, p, col, first, gpt) : F(0.0f);
    F next = p.pk_on && first + 1 < p.nlay
        ? lw_planck_fraction(r, p, col, first + 1, gpt) : F(0.0f);

    F dn = lw_div(r.incident_flux[col * p.ngpt + gpt], pi);
    r.part_dn[(base + top) * p.ncol + col] = as_type<float>(dn.bits);
    for (int j = 0; j < p.nlay; ++j) {
        const int lay = p.top_at_1 ? j : p.nlay - 1 - j;
        const int cell = col * p.nlay + lay;
        const int cellg = cell * p.ngpt + gpt;
        const int lev = col * nlev + lay;
        F tau = r.tau[cellg];
        if (p.fz_on) {
            const int cloud = cell * p.fz_nband + as_type<int>(r.fz_gpt_bands[gpt]);
            F cloud_tau = r.fz_cld_tau[cloud];
            if (p.fz_have_mask) cloud_tau = cloud_tau * F(r.fz_mask[cellg]);
            tau = tau + cloud_tau * (F(1.0f) - F(r.fz_cld_ssa[cloud]));
        }
        const F path = tau * d;
        const F tr = native_tc_math::thompson_aa_expf(-path);
        trans[lay] = tr;
        const F fact = path > tau_thresh
            ? lw_div(F(1.0f) - tr, path) - tr
            : path * (F(0.5f) + path * (F(-1.0f / 3.0f) + path * F(1.0f / 8.0f)));
        F lays, lo, hi;
        if (p.pk_on) {
            lays = cur * lw_planck(r.pk_tlay[cell], band, r.pk_totplnk,
                                   p.pk_nplanck, p.pk_nband, t0, delta_r);
            const F lo_fraction = lay == 0 ? cur : lw_sqrt(prev * cur);
            const F hi_fraction = lay + 1 == p.nlay ? cur : lw_sqrt(cur * next);
            lo = lo_fraction * lw_planck(r.pk_tlev[lev], band, r.pk_totplnk,
                                         p.pk_nplanck, p.pk_nband, t0, delta_r);
            hi = hi_fraction * lw_planck(r.pk_tlev[lev + 1], band, r.pk_totplnk,
                                         p.pk_nplanck, p.pk_nband, t0, delta_r);
            if (p.top_at_1) {
                prev = cur;
                cur = next;
                next = lay + 2 < p.nlay ? lw_planck_fraction(r, p, col, lay + 2, gpt) : F(0.0f);
            } else {
                next = cur;
                cur = prev;
                prev = lay >= 2 ? lw_planck_fraction(r, p, col, lay - 2, gpt) : F(0.0f);
            }
        } else {
            lays = r.lay_source[cellg];
            lo = r.lev_source[lev * p.ngpt + gpt];
            hi = r.lev_source[(lev + 1) * p.ngpt + gpt];
        }
        const F inc = (F(1.0f) - tr) * hi + F(2.0f) * fact * (lays - hi);
        const F dec = (F(1.0f) - tr) * lo + F(2.0f) * fact * (lays - lo);
        src_up[lay] = p.top_at_1 ? dec : inc;
        dn = tr * dn + (p.top_at_1 ? inc : dec);
        const int dlev = p.top_at_1 ? lay + 1 : lay;
        r.part_dn[(base + dlev) * p.ncol + col] = as_type<float>(dn.bits);
    }
    const int sg = col * p.ngpt + gpt;
    F surface = r.sfc_source[sg];
    if (p.pk_on) {
        const int first_layer = col * p.nlay;
        const int surface_layer = F(r.pk_play[first_layer]) < F(r.pk_play[first_layer + p.nlay - 1])
            ? p.nlay - 1 : 0;
        surface = lw_planck_fraction(r, p, col, surface_layer, gpt)
            * lw_planck(r.pk_tsfc[col], band, r.pk_totplnk, p.pk_nplanck, p.pk_nband, t0, delta_r);
    }
    const F emissivity = r.sfc_emis[sg];
    F up = dn * (F(1.0f) - emissivity) + emissivity * surface;
    r.part_up[(base + sfc) * p.ncol + col] = as_type<float>(up.bits);
    for (int j = 0; j < p.nlay; ++j) {
        const int lay = p.top_at_1 ? p.nlay - 1 - j : j;
        up = trans[lay] * up + src_up[lay];
        const int ulev = p.top_at_1 ? lay : lay + 1;
        r.part_up[(base + ulev) * p.ncol + col] = as_type<float>(up.bits);
    }
}

kernel void rrtmgp_exact_lw_flux_reduce(
    constant RrtmgpExactLwFluxReduceResources &r [[buffer(0)]],
    constant RrtmgpExactLwFluxReduceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ncol * p.nlev)) return;
    const int lev = int(gid) / p.ncol;
    const int col = int(gid) - lev * p.ncol;
    const int output = col * p.nlev + lev;
    F up = p.gpt0 == 0 ? F(0.0f) : F(r.flux_up[output]);
    F down = p.gpt0 == 0 ? F(0.0f) : F(r.flux_dn[output]);
    for (int g = 0; g < p.ntile; ++g) {
        const int input = (g * p.nlev + lev) * p.ncol + col;
        up = up + F(r.part_up[input]);
        down = down + F(r.part_dn[input]);
    }
    if (p.last_tile) {
        const F pi = 3.14159265358979323846f;
        up = pi * up;
        down = pi * down;
    }
    r.flux_up[output] = as_type<float>(up.bits);
    r.flux_dn[output] = as_type<float>(down.bits);
}
