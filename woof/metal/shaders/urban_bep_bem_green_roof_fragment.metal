#include <metal_stdlib>
#include "urban_bep_bem_green_roof_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

constant uint GR_NDM = 2u;
constant uint GR_NZUM = 18u;
constant uint GR_NGR = 10u;
constant uint GR_NWR = 10u;
constant uint GR_INTERP_FIELDS = 7u;
constant uint GR_IRHO = 1263u;
constant uint GR_GREEN_FRACTION = 1206u;
constant uint GR_PV_FRACTION = 1217u;
constant uint GR_TYPE = 35u;

static inline float gr_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float gr_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float gr_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float gr_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float gr_pow(float a, float b) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a), as_type<uint>(b))); }
static inline float gr_log(float a) { return as_type<float>(soft_gfk_log(as_type<uint>(a))); }
static inline float gr_exp(float a) { return as_type<float>(soft_gfk_exp(as_type<uint>(a))); }
static inline float gr_abs(float a) { return as_type<float>(as_type<uint>(a) & 0x7fffffffu); }
static inline float gr_min(float a, float b) { return a < b ? a : b; }
static inline float gr_max(float a, float b) { return a > b ? a : b; }

// WRF module_sf_bep_bem.F bb_roof_temp_veg (2519-2591), including the
// coupled bb_soil_temp_veg and bb_soil_moist solves. One thread owns one
// atmospheric column; its direction, urban-level, and soil-layer loops retain
// the WRF update order so heat and moisture remain coupled in place.
kernel void urban_bep_bem_green_roof_fragment(
    constant UrbanBepBemGreenRoofFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemGreenRoofFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.ncol)) return;
    const uint col = item;
    if (r.active_cols[col] == 0 || p.green_roof != 1) return;
    const int class_id = r.utype[col] - 1;
    if (class_id < 0 || class_id >= 11) return;
    const ulong c = ulong(col), cr = ulong(class_id);
    const float green_fraction = r.class_float[GR_GREEN_FRACTION + cr];
    if (!(green_fraction > 0.0f)) return;
    const int gr_type = r.class_int[GR_TYPE];
    if (gr_type != 1 && gr_type != 2) { r.error[col] = 22; return; }
    const int nzu = r.nzurban[col];
    const float alar_roof = r.alar[c * (GR_NWR + 1u) + 4u];
    const float dzr_roof = r.dzr[c * GR_NWR + 4u];
    const float csr_roof = r.csr[c * GR_NWR + 4u];
    const float pv_fraction = r.class_float[GR_PV_FRACTION + cr];
    const float rain = r.rainbl[col], rs = r.swdown[col], rl = r.glw[col];
    const float hour_angle = r.hour_angle[col];
    float nhourday = gr_add(gr_mul(gr_div(hour_angle, 3.14159265358979323846f), 12.0f), 12.0f);
    int hour = (int(floor(nhourday)) % 24 + 24) % 24;
    float irho_sum = 0.0f;
    for (uint h = 0; h < 24u; ++h)
        irho_sum = gr_add(irho_sum, r.class_float[GR_IRHO + h]);
    const float irho_now = r.class_float[GR_IRHO + uint(hour)];
    const float irrigation = (irho_sum > 0.0f && irho_now != 0.0f)
        ? gr_div(0.0009722f, irho_sum) : 0.0f;

    const float dz[GR_NGR] = {0.1f, 0.003f, 0.06f, 0.003f, 0.05f, 0.04f, 0.02f, 0.0125f, 0.005f, 0.0025f};
    const float qrmax[GR_NGR] = {0.0f, 0.0f, 0.0f, 0.0f, 0.439f, 0.37f, 0.37f, 0.37f, 0.37f, 0.37f};
    const float smax[GR_NGR] = {0.0f, 0.0f, 0.0f, 0.0f, -0.01f, -0.1f, -0.1f, -0.1f, -0.1f, -0.1f};
    const float kmax[GR_NGR] = {0.0f, 0.0f, 0.0f, 0.0f, 3.32e-3f, 2.162e-3f, 2.162e-3f, 2.162e-3f, 2.162e-3f, 2.162e-3f};
    const float b[GR_NGR] = {0.0f, 0.0f, 0.0f, 0.0f, 2.7f, 3.9f, 3.9f, 3.9f, 3.9f, 3.9f};
    const float cd[GR_NGR] = {0.0f, 0.0f, 0.0f, 0.0f, 331500.0f, 1.342e6f, 1.342e6f, 1.342e6f, 1.342e6f, 1.342e6f};
    const float csa[4] = {7.5e4f, 2.1e6f, 4.48e4f, 2.1e6f};
    const float ka[4] = {0.035f, 0.7f, 0.024f, 0.7f};
    const float cw = 4.295e6f, em = gr_type == 1 ? 0.95f : 0.83f;
    const float alb = gr_type == 1 ? 0.3f : 0.154f;
    constexpr float latent = 2.45e6f, rcp_u = 287.0f / 1004.0f;
    constexpr float sigma = 5.67e-8f, ln10 = 2.30258509299404568402f;

    for (int iz = 1; iz < nzu && iz < int(GR_NZUM); ++iz) {
        const ulong z = ulong(iz), level = (c * GR_NZUM + z) * GR_NDM;
        const float pressure = r.interp[c * GR_INTERP_FIELDS * GR_NZUM + 4ul * GR_NZUM + z];
        for (uint d = 0; d < GR_NDM; ++d) {
            const ulong out = level + d;
            float temp[GR_NGR], moisture[GR_NGR], cs[GR_NGR];
            float diffusivity[GR_NGR], suction[GR_NGR], conductivity[GR_NGR], hydraulic_d[GR_NGR];
            for (uint g = 0; g < GR_NGR; ++g) {
                const ulong idx = ((c * GR_NGR + g) * GR_NZUM + z) * GR_NDM + d;
                temp[g] = r.trv[idx];
                moisture[g] = gr_max(r.qr[idx], 1.0e-6f);
                r.qr[idx] = moisture[g];
            }
            for (uint g = 0; g < GR_NGR; ++g) {
                if (g < 4u) {
                    cs[g] = csa[g]; diffusivity[g] = gr_div(ka[g], csa[g]);
                    suction[g] = 0.0f; conductivity[g] = 0.0f; hydraulic_d[g] = 0.0f;
                } else {
                    float qr_m = moisture[g];
                    if (g > 4u)
                        qr_m = gr_div(gr_add(gr_mul(moisture[g], dz[g - 1u]), gr_mul(moisture[g - 1u], dz[g])), gr_add(dz[g], dz[g - 1u]));
                    cs[g] = gr_add(gr_mul(gr_sub(1.0f, qr_m), cd[g]), gr_mul(qr_m, cw));
                    suction[g] = gr_mul(smax[g], gr_pow(gr_div(qrmax[g], qr_m), b[g]));
                    conductivity[g] = gr_mul(kmax[g], gr_pow(gr_div(qr_m, qrmax[g]), gr_add(gr_mul(2.0f, b[g]), 3.0f)));
                    hydraulic_d[g] = gr_div(gr_mul(gr_mul(gr_mul(-b[g], kmax[g]), smax[g]), gr_pow(gr_div(qr_m, qrmax[g]), gr_add(b[g], 3.0f))), qr_m);
                    const float log_s = gr_div(gr_log(gr_abs(suction[g])), ln10);
                    diffusivity[g] = log_s <= 5.1f ? gr_div(gr_mul(gr_exp(-gr_add(log_s, 2.7f)), 418.6f), cs[g]) : gr_div(gr_mul(0.00041f, 418.6f), cs[g]);
                }
            }
            const float roof_temperature = r.tr[(c * GR_NWR + 4u) * GR_NZUM * GR_NDM + z * GR_NDM + d];
            const float hfgroof = gr_div(gr_mul(gr_add(gr_div(alar_roof, csr_roof), diffusivity[0]), gr_sub(temp[0], roof_temperature)), gr_add(dzr_roof, dz[0]));
            r.hfgroof[out] = hfgroof;
            const float tpv = r.tpv[out];
            float rl_eff, rs_eff;
            if (pv_fraction > 0.0f) {
                rl_eff = gr_add(gr_mul(gr_mul(1.0f - pv_fraction, em), rl), gr_mul(gr_mul(gr_mul(em, sigma), gr_pow(tpv, 4.0f)), pv_fraction));
                rs_eff = gr_mul(1.0f - pv_fraction, rs);
            } else { rl_eff = gr_mul(em, rl); rs_eff = rs; }
            const float tsig = temp[GR_NGR - 1u];
            const float surface_sensible = r.sfrv[out];
            const float surface_latent = r.lfrv[out];
            const float alpha = gr_add(gr_sub(gr_add(gr_mul(1.0f - alb, rs_eff), rl_eff), gr_mul(gr_mul(em, sigma), gr_pow(tsig, 4.0f))), gr_add(surface_sensible, surface_latent));
            float cddz[GR_NGR + 1u] = {};
            cddz[0] = gr_div(diffusivity[0], dz[0]);
            for (uint g = 1; g < GR_NGR; ++g) cddz[g] = gr_div(gr_mul(2.0f, diffusivity[g]), gr_add(dz[g], dz[g - 1u]));
            float lower[GR_NGR] = {}, diag[GR_NGR] = {}, upper[GR_NGR] = {}, rhs[GR_NGR];
            diag[0] = 1.0f; rhs[0] = gr_sub(temp[0], gr_div(gr_mul(hfgroof, p.dt), dz[0]));
            for (uint g = 1; g + 1u < GR_NGR; ++g) {
                lower[g] = gr_div(gr_mul(-cddz[g], p.dt), dz[g]);
                diag[g] = gr_add(1.0f, gr_div(gr_mul(p.dt, gr_add(cddz[g], cddz[g + 1u])), dz[g]));
                upper[g] = gr_div(gr_mul(-cddz[g + 1u], p.dt), dz[g]); rhs[g] = temp[g];
            }
            lower[GR_NGR - 1u] = gr_div(gr_mul(-p.dt, cddz[GR_NGR - 1u]), dz[GR_NGR - 1u]);
            diag[GR_NGR - 1u] = gr_add(1.0f, gr_div(gr_mul(p.dt, cddz[GR_NGR - 1u]), dz[GR_NGR - 1u]));
            rhs[GR_NGR - 1u] = gr_add(temp[GR_NGR - 1u], gr_div(gr_div(gr_mul(p.dt, alpha), cs[GR_NGR - 1u]), dz[GR_NGR - 1u]));
            for (int g = int(GR_NGR) - 2; g >= 0; --g) { rhs[g] = gr_sub(rhs[g], gr_div(gr_mul(upper[g], rhs[g + 1]), diag[g + 1])); diag[g] = gr_sub(diag[g], gr_div(gr_mul(upper[g], lower[g + 1]), diag[g + 1])); }
            for (uint g = 1; g < GR_NGR; ++g) rhs[g] = gr_sub(rhs[g], gr_div(gr_mul(lower[g], rhs[g - 1u]), diag[g - 1u]));
            for (uint g = 0; g < GR_NGR; ++g) temp[g] = gr_div(rhs[g], diag[g]);
            r.ptrv[out] = gr_mul(temp[GR_NGR - 1u], gr_pow(gr_div(pressure, 100000.0f), -rcp_u));
            const float rt = gr_sub(gr_add(gr_mul(1.0f - alb, rs_eff), rl_eff), gr_mul(gr_mul(em, sigma), gr_pow(tsig, 4.0f)));
            r.gfr[out] = gr_add(rt, surface_sensible);
            for (uint g = 0; g < GR_NGR; ++g) r.trv[((c * GR_NGR + g) * GR_NZUM + z) * GR_NDM + d] = temp[g];

            float wcddz[GR_NGR + 1u] = {};
            for (uint g = 1; g < GR_NGR; ++g) wcddz[g] = gr_div(gr_mul(2.0f, hydraulic_d[g]), gr_add(dz[g], dz[g - 1u]));
            const float drain = gr_mul(kmax[4], gr_pow(gr_div(moisture[4], qrmax[4]), gr_add(gr_mul(2.0f, b[4]), 3.0f)));
            r.drain[out] = drain;
            const float moisture_alpha = gr_add(gr_add(gr_div(rain, gr_mul(1000.0f, p.dt)), gr_div(gr_div(r.lfrv[out], latent), 1000.0f)), gr_div(irrigation, 1000.0f));
            float mlower[GR_NGR] = {}, mdiag[GR_NGR] = {}, mupper[GR_NGR] = {}, mrhs[GR_NGR];
            for (uint g = 0; g < 4u; ++g) { mdiag[g] = 1.0f; mrhs[g] = moisture[g]; }
            mdiag[4] = gr_add(1.0f, gr_div(gr_mul(p.dt, wcddz[5]), dz[4])); mupper[4] = gr_div(gr_mul(-wcddz[5], p.dt), dz[4]);
            mrhs[4] = gr_add(moisture[4], gr_div(gr_mul(p.dt, gr_sub(conductivity[5], drain)), dz[4]));
            for (uint g = 5u; g + 1u < GR_NGR; ++g) {
                mlower[g] = gr_div(gr_mul(-wcddz[g], p.dt), dz[g]); mdiag[g] = gr_add(1.0f, gr_div(gr_mul(p.dt, gr_add(wcddz[g], wcddz[g + 1u])), dz[g]));
                mupper[g] = gr_div(gr_mul(-wcddz[g + 1u], p.dt), dz[g]); mrhs[g] = gr_add(moisture[g], gr_div(gr_mul(p.dt, gr_sub(conductivity[g + 1u], conductivity[g])), dz[g]));
            }
            mlower[GR_NGR - 1u] = gr_div(gr_mul(-p.dt, wcddz[GR_NGR - 1u]), dz[GR_NGR - 1u]); mdiag[GR_NGR - 1u] = gr_add(1.0f, gr_div(gr_mul(p.dt, wcddz[GR_NGR - 1u]), dz[GR_NGR - 1u]));
            mrhs[GR_NGR - 1u] = gr_sub(gr_add(moisture[GR_NGR - 1u], gr_div(gr_mul(p.dt, moisture_alpha), dz[GR_NGR - 1u])), gr_div(gr_mul(p.dt, conductivity[GR_NGR - 2u]), dz[GR_NGR - 1u]));
            for (int g = int(GR_NGR) - 2; g >= 0; --g) { mrhs[g] = gr_sub(mrhs[g], gr_div(gr_mul(mupper[g], mrhs[g + 1]), mdiag[g + 1])); mdiag[g] = gr_sub(mdiag[g], gr_div(gr_mul(mupper[g], mlower[g + 1]), mdiag[g + 1])); }
            for (uint g = 1; g < GR_NGR; ++g) mrhs[g] = gr_sub(mrhs[g], gr_div(gr_mul(mlower[g], mrhs[g - 1u]), mdiag[g - 1u]));
            for (uint g = 0; g < GR_NGR; ++g) r.qr[((c * GR_NGR + g) * GR_NZUM + z) * GR_NDM + d] = gr_max(gr_min(gr_div(mrhs[g], mdiag[g]), qrmax[g]), 1.0e-6f);
        }
    }
}

// Flatten one (column, level, direction) roof cell into the row-oriented
// input contract used by the existing bb_flux_flat_roof fragment. The merge
// kernel below restores only WRF's ss(iz)>0, iz=2..nzu levels.
kernel void urban_bep_bem_green_roof_flux_pack_fragment(
    constant UrbanBepBemGreenRoofFluxPackFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemGreenRoofFluxPackFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    const uint col = item / (GR_NZUM * GR_NDM);
    const uint rem = item % (GR_NZUM * GR_NDM);
    const uint z = rem / GR_NDM, d = rem % GR_NDM;
    if (col >= uint(p.ncol)) return;
    const ulong profile = ulong(col) * GR_NZUM + z;
    const ulong out = ulong(item);
    const int class_id = r.utype[col] - 1;
    const ulong interp_base = ulong(col) * GR_INTERP_FIELDS * GR_NZUM + ulong(z);
    r.dz[out] = p.dz_u;
    r.z0[out] = 0.01f;
    r.ua[out] = r.interp[interp_base];
    r.va[out] = r.interp[interp_base + GR_NZUM];
    r.pt[out] = r.interp[interp_base + 2ul * GR_NZUM];
    r.pt0[out] = r.interp[interp_base + 3ul * GR_NZUM];
    r.da[out] = r.interp[interp_base + 5ul * GR_NZUM];
    r.qv[out] = r.interp[interp_base + 6ul * GR_NZUM];
    r.pr[out] = r.interp[interp_base + 4ul * GR_NZUM];
    r.ptg[out] = r.ptrv[profile * GR_NDM + d];
    r.rsg[out] = r.swdown[col];
    r.pv_frac_roof[out] = class_id >= 0 && class_id < 11
        ? r.class_float[GR_PV_FRACTION + ulong(class_id)] : 0.0f;
    r.gr_type[out] = r.class_int[GR_TYPE];
    for (uint g = 0; g < GR_NGR; ++g)
        r.qr_rows[out * GR_NGR + g] = r.qr[((ulong(col) * GR_NGR + g) * GR_NZUM + z) * GR_NDM + d];
    // Seed the output profiles so the merge leaves skipped levels untouched.
    r.sf[out] = r.sfrv[profile * GR_NDM + d];
    r.lf[out] = r.lfrv[profile * GR_NDM + d];
}

kernel void urban_bep_bem_green_roof_flux_merge_fragment(
    constant UrbanBepBemGreenRoofFluxMergeFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemGreenRoofFluxMergeFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem) || p.green_roof != 1) return;
    const uint col = item / (GR_NZUM * GR_NDM);
    const uint rem = item % (GR_NZUM * GR_NDM);
    const uint z = rem / GR_NDM, d = rem % GR_NDM;
    if (col >= uint(p.ncol) || r.active_cols[col] == 0) return;
    const int class_id = r.utype[col] - 1;
    if (class_id < 0 || class_id >= 11 ||
        !(r.class_float[GR_GREEN_FRACTION + ulong(class_id)] > 0.0f)) return;
    if (z == 0u || int(z) >= r.nzurban[col]) return;
    if (!(r.ss[ulong(col) * (GR_NZUM + 1u) + ulong(z)] > 0.0f)) return;
    const ulong out = ulong(item), profile = (ulong(col) * GR_NZUM + z) * GR_NDM + d;
    r.sfrv[profile] = r.sf[out];
    r.lfrv[profile] = r.lf[out];
}

// bem_hsfluxpv's scalar sfpv is copied to sfrpv at each building's mapped
// level in WRF order. If slots share a level, WRF's later slot overwrites the
// earlier value; reconcile the aggregate tendency emitted by the existing PV
// solve with that same last-writer mapping.
kernel void urban_bep_bem_green_roof_pv_map_fragment(
    constant UrbanBepBemGreenRoofPvMapFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemGreenRoofPvMapFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.ncol)) return;
    const uint col = item;
    if (r.active_cols[col] == 0) return;
    const int class_id = r.utype[col] - 1;
    if (class_id < 0 || class_id >= 11) return;
    const ulong c = ulong(col), class_row = ulong(class_id);
    const float pv_fraction = r.class_float[GR_PV_FRACTION + class_row];
    const float green_fraction = r.class_float[GR_GREEN_FRACTION + class_row];
    float sum_hspv[GR_NZUM] = {};
    float last_hspv[GR_NZUM] = {};
    bool written[GR_NZUM] = {};
    for (uint building = 0; building < 15u; ++building) {
        const ulong slot = c * 15u + building;
        const int level = r.building_nlev[slot];
        if (level <= 0 || level >= int(GR_NZUM)) continue;
        const float hspv = r.pv_hspv[slot];
        sum_hspv[uint(level)] = gr_add(sum_hspv[uint(level)], hspv);
        last_hspv[uint(level)] = hspv;
        written[uint(level)] = true;
    }
    const ulong interp_base = c * GR_INTERP_FIELDS * GR_NZUM;
    for (uint z = 1; z < GR_NZUM; ++z) {
        if (!written[z]) continue;
        const float da = r.interp[interp_base + 5ul * GR_NZUM + z];
        const float weight = gr_div(gr_mul(gr_mul(1.0f - green_fraction,
                                                   pv_fraction), 1.0f),
                                    gr_mul(da, p.cp_u));
        for (uint d = 0; d < GR_NDM; ++d) {
            const ulong profile = (c * GR_NZUM + z) * GR_NDM + d;
            const ulong flat = (c * GR_NDM + d) * GR_NZUM + z;
            r.sfrpv[profile] = -last_hspv[z];
            r.flat_thb[flat] = gr_add(r.flat_thb[flat],
                gr_mul(gr_sub(last_hspv[z], sum_hspv[z]), weight));
        }
    }
}

// Apply bb_buildings' green-roof flux blend after the independently ported
// flat and roof fluxes have completed. The PV correction already carried by
// flat_thb is retained with the same WRF roof fraction and sfrpv mapping.
kernel void urban_bep_bem_green_roof_tendency_fragment(
    constant UrbanBepBemGreenRoofTendencyFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemGreenRoofTendencyFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    const uint col = item / (GR_NZUM * GR_NDM);
    const uint rem = item % (GR_NZUM * GR_NDM);
    const uint z = rem / GR_NDM, d = rem % GR_NDM;
    if (col >= uint(p.ncol) || r.active_cols[col] == 0 || p.green_roof != 1 || z == 0u) return;
    const int class_id = r.utype[col] - 1;
    if (class_id < 0 || class_id >= 11) return;
    const float green_fraction = r.class_float[GR_GREEN_FRACTION + ulong(class_id)];
    if (!(green_fraction > 0.0f)) return;
    if (!(r.ss[ulong(col) * (GR_NZUM + 1u) + ulong(z)] > 0.0f)) return;
    const ulong c = ulong(col), profile = (c * GR_NZUM + z) * GR_NDM + d;
    const ulong flat = (c * GR_NDM + d) * GR_NZUM + z;
    const ulong interp_base = c * GR_INTERP_FIELDS * GR_NZUM;
    const float da = r.interp[interp_base + 5ul * GR_NZUM + ulong(z)];
    const float pv_fraction = r.class_float[GR_PV_FRACTION + ulong(class_id)];
    const float pv_correction = gr_div(gr_mul(gr_mul(
        gr_sub(1.0f, green_fraction), pv_fraction), -r.sfrpv[profile]), gr_mul(da, p.cp_u));
    const float flat_base = gr_sub(r.flat_thb[flat], pv_correction);
    r.thb[flat] = gr_sub(gr_add(gr_mul(gr_sub(1.0f, green_fraction), flat_base), pv_correction),
                          gr_div(gr_mul(green_fraction, r.sfrv[profile]), gr_mul(da, p.cp_u)));
    r.qhb_flat[flat] = gr_sub(gr_mul(gr_sub(1.0f, green_fraction), r.flat_qhb[flat]),
                               gr_div(gr_mul(green_fraction, r.lfrv[profile]), gr_mul(da, p.latent)));
    const ulong roof = ulong(item);
    r.uhb[flat] = gr_add(gr_mul(gr_sub(1.0f, green_fraction), r.flat_uhb[flat]), gr_mul(green_fraction, r.roof_uhb[roof]));
    r.vhb[flat] = gr_add(gr_mul(gr_sub(1.0f, green_fraction), r.flat_vhb[flat]), gr_mul(green_fraction, r.roof_vhb[roof]));
    r.ehb[flat] = gr_add(gr_mul(gr_sub(1.0f, green_fraction), r.flat_ehb[flat]), gr_mul(green_fraction, r.roof_ehb[roof]));
}
