#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "hexrrfs_ruc_forcing_bindings.metalh"
#include "hexrrfs_ruc_seaice_prepare_bindings.metalh"
#include "hexrrfs_ruc_seaice_finish_bindings.metalh"
#include "hexrrfs_ruc_seaice_skin_bindings.metalh"
#include "hexrrfs_ruc_soil_profile_bindings.metalh"
#include "hexrrfs_ruc_seaice_background_bindings.metalh"
using namespace ntc;
#pragma clang fp contract(off)
inline F get(device const float *p, uint i) { return f32_word(as_type<uint>(p[i])); }
inline void put(device float *p, uint i, F x) { p[i] = as_type<float>(x.bits); }

// The RUC driver takes zs from the first column of the gathered tile.
kernel void hexrrfs_ruc_soil_profile(constant HexrrfsRucSoilProfileResources &r [[buffer(0)]],
                                    constant HexrrfsRucSoilProfileParams &p [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.solve))
        return;
    for (int k = 0; k < p.nsl; ++k) {
        uint row = uint(k) * uint(p.nc);
        r.output[row + i] = r.zs[row];
    }
}

kernel void hexrrfs_ruc_seaice_background(constant HexrrfsRucSeaiceBackgroundResources &r [[buffer(0)]],
                                         constant HexrrfsRucSeaiceBackgroundParams &p [[buffer(1)]],
                                         uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.solve))
        return;
    r.output[i] = get(r.xice, i) < F(p.xice_threshold) ? r.albbck[i] : .65f;
}

// MPAS driver_lsm.F:606-640. Disabled sources can be one-word dummies.
kernel void hexrrfs_ruc_forcing(constant HexrrfsRucForcingResources &r [[buffer(0)]],
                                constant HexrrfsRucForcingParams &p [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.solve))
        return;
    F rain = F(0.f);
    r.sr_p[i] = 0.f;
    if (p.micro_on) {
        r.sr_p[i] = r.sr[i];
        rain = tc_add(rain, get(r.rainncv, i));
    }
    if (p.conv_on)
        rain = tc_add(rain, get(r.raincv, i));
    put(r.rainbl_p, i, rain);
    put(r.swdown_p, i, tc_div(get(r.gsw, i), tc_sub(F(1.f), get(r.sfc_albedo, i))));
}
// MPAS driver_seaice.F:436-442. This mask intentionally excludes xice==1.
kernel void hexrrfs_ruc_seaice_prepare(constant HexrrfsRucSeaicePrepareResources &r [[buffer(0)]],
                                       constant HexrrfsRucSeaicePrepareParams &p [[buffer(1)]],
                                       uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.solve))
        return;
    F x = get(r.xice, i);
    if (x >= F(p.xice_threshold) && x < F(1.f)) {
        put(r.sfc_albedo_p, i,
            tc_div(tc_sub(get(r.sfc_albedo, i), tc_mul(F(.08f), tc_sub(F(1.f), x))), x));
        put(r.sfc_emiss_p, i,
            tc_div(tc_sub(get(r.sfc_emiss, i), tc_mul(F(.98f), tc_sub(F(1.f), x))), x));
    } else {
        r.sfc_albedo_p[i] = r.sfc_albedo[i];
        r.sfc_emiss_p[i] = r.sfc_emiss[i];
    }
}
// MPAS atmphys_lsm_shared.F:22-57. This distinct mask includes xice==1.
kernel void hexrrfs_ruc_seaice_skin(constant HexrrfsRucSeaiceSkinResources &r [[buffer(0)]],
                                    constant HexrrfsRucSeaiceSkinParams &p [[buffer(1)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.solve))
        return;
    F x = get(r.xice, i), sea = get(r.tsk, i), ice = sea;
    if (x >= F(p.xice_threshold) && x <= F(1.f)) {
        sea = tc_max(sea, F(271.4f));
        if (x < F(.2f) && ice < F(253.15f))
            ice = F(253.15f);
        if (x < F(.1f) && ice < F(263.15f))
            ice = F(263.15f);
    }
    put(r.tsk_sea, i, sea);
    put(r.tsk_ice, i, ice);
}
inline F blend(F x, F ice, F sea) { return tc_add(tc_mul(x, ice), tc_mul(tc_sub(F(1.f), x), sea)); }
// MPAS driver_seaice.F:571-601: cqs uses original chs in non-MYNN mode.
kernel void hexrrfs_ruc_seaice_finish(constant HexrrfsRucSeaiceFinishResources &r [[buffer(0)]],
                                      constant HexrrfsRucSeaiceFinishParams &p [[buffer(1)]],
                                      uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.solve) || !p.config_frac_seaice)
        return;
    F x = get(r.xice_p, i);
    if (!(x >= F(p.xice_threshold) && x < F(1.f)))
        return;
    put(r.flhc_p, i, blend(x, get(r.flhc_p, i), get(r.flhc_sea, i)));
    put(r.flqc_p, i, blend(x, get(r.flqc_p, i), get(r.flqc_sea, i)));
    if (p.mynn)
        put(r.cqs_p, i, blend(x, get(r.cqs_p, i), get(r.cqs_sea, i)));
    else
        put(r.cqs_p, i, blend(x, get(r.chs_p, i), get(r.chs_sea, i)));
    put(r.cpm_p, i, blend(x, get(r.cpm_p, i), get(r.cpm_sea, i)));
    put(r.chs_p, i, blend(x, get(r.chs_p, i), get(r.chs_sea, i)));
    put(r.chs2_p, i, blend(x, get(r.chs2_p, i), get(r.chs2_sea, i)));
    put(r.cqs2_p, i, blend(x, get(r.cqs2_p, i), get(r.cqs2_sea, i)));
    put(r.hfx_p, i, blend(x, get(r.hfx_p, i), get(r.hfx_sea, i)));
    put(r.lh_p, i, blend(x, get(r.lh_p, i), get(r.lh_sea, i)));
    put(r.qfx_p, i, blend(x, get(r.qfx_p, i), get(r.qfx_sea, i)));
    put(r.qgh_p, i, blend(x, get(r.qgh_p, i), get(r.qgh_sea, i)));
    put(r.qsfc_p, i, blend(x, get(r.qsfc_p, i), get(r.qsfc_sea, i)));
    put(r.tsk_p, i, blend(x, get(r.tsk_p, i), get(r.tsk_sea, i)));
    put(r.sfc_albedo_p, i, blend(x, get(r.sfc_albedo_p, i), F(.08f)));
    put(r.sfc_emiss_p, i, blend(x, get(r.sfc_emiss_p, i), F(.98f)));
}
