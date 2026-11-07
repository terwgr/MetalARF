#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "hexrrfs_tempo_native_step_diags_bindings.metalh"
#include "hexrrfs_tempo_native_precip_bindings.metalh"
#include "hexrrfs_tempo_native_to_bindings.metalh"
using namespace ntc;
#pragma clang fp contract(off)
inline F get(device const float *p, ulong i) { return f32_word(as_type<uint>(p[i])); }
inline void put(device float *p, ulong i, F a) { p[i] = as_type<float>(a.bits); }
kernel void hexrrfs_tempo_native_step_diags(constant HexrrfsTempoNativeStepDiagsResources &r
                                            [[buffer(0)]],
                                            constant HexrrfsTempoNativeStepDiagsParams &p
                                            [[buffer(1)]],
                                            uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    ulong n = ulong(uint(p.nz)) * uint(p.nc);
    F rain = get(r.out, c), ice = get(r.out, n + c), snow = get(r.out, 2 * n + c),
      graupel = get(r.out, 3 * n + c);
    put(r.s_snowncv, c, tc_add(tc_add(get(r.s_snowncv, c), ice), snow));
    put(r.s_snownc, c, tc_add(tc_add(get(r.s_snownc, c), ice), snow));
    put(r.s_graupelncv, c, tc_add(get(r.s_graupelncv, c), graupel));
    put(r.s_graupelnc, c, tc_add(get(r.s_graupelnc, c), graupel));
    put(r.s_rainncv, c,
        tc_add(tc_add(tc_add(tc_add(get(r.s_rainncv, c), rain), ice), snow), graupel));
    put(r.s_rainnc, c,
        tc_add(tc_add(tc_add(tc_add(get(r.s_rainnc, c), rain), ice), snow), graupel));
    r.s_sr[c] = r.out[5 * n + c];
    put(r.s_frainnc, c, tc_add(get(r.s_frainnc, c), get(r.out, 4 * n + c)));
    r.s_max_hail_diameter_sfc[c] = r.out[6 * n + c];
    r.s_max_hail_diameter_column[c] = r.out[7 * n + c];
    for (int k = 0; k < p.nz; ++k) {
        ulong x = ulong(k) * uint(p.nc) + c;
        r.profile_refl10cm[x] = r.out[8 * n + x];
        r.profile_re_cloud[x] = r.out[9 * n + x];
        r.profile_re_ice[x] = r.out[10 * n + x];
        r.profile_re_snow[x] = r.out[11 * n + x];
    }
}
kernel void hexrrfs_tempo_native_precip(constant HexrrfsTempoNativePrecipResources &r [[buffer(0)]],
                                        constant HexrrfsTempoNativePrecipParams &p [[buffer(1)]],
                                        uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    if (!p.finish) {
        r.s_rainncv[c] = r.s_rainnc[c] = r.s_snowncv[c] = r.s_snownc[c] = r.s_graupelncv[c] =
            r.s_graupelnc[c] = r.s_sr[c] = r.s_frainnc[c] = 0.f;
        r.pr_rainncv[c] = r.pr_snowncv[c] = r.pr_graupelncv[c] = r.pr_sr[c] = 0.f;
        return;
    }
    put(r.pr_max_hail_diameter_sfc_acc, c,
        tc_max(get(r.pr_max_hail_diameter_sfc_acc, c), get(r.s_max_hail_diameter_sfc, c)));
    put(r.pr_max_hail_diameter_column_acc, c,
        tc_max(get(r.pr_max_hail_diameter_column_acc, c), get(r.s_max_hail_diameter_column, c)));
    ulong n = ulong(uint(p.nz)) * uint(p.nc);
    F water = F(0.f);
    for (int k = 0; k < p.nz; ++k) {
        ulong x = ulong(k) * uint(p.nc) + c;
        F qv = get(r.prepared, 4 * n + x),
          rho = tc_div(get(r.prepared, 18 * n + x), tc_add(F(1.f), qv));
        water = tc_add(water, tc_mul(tc_mul(qv, rho), get(r.prepared, 3 * n + x)));
    }
    put(r.pr_precipw, c, water);
    r.pr_rainncv[c] = r.s_rainnc[c];
    put(r.pr_rainnc, c, tc_add(get(r.pr_rainnc, c), get(r.pr_rainncv, c)));
    if (p.accrue && p.bucket > 0.f && get(r.pr_rainnc, c) > F(p.bucket)) {
        r.counter[c] += 1;
        put(r.pr_rainnc, c, tc_sub(get(r.pr_rainnc, c), F(p.bucket)));
    }
    r.pr_snowncv[c] = r.s_snownc[c];
    r.pr_graupelncv[c] = r.s_graupelnc[c];
    put(r.pr_sr, c,
        tc_div(tc_add(get(r.s_snownc, c), get(r.s_graupelnc, c)),
               tc_add(get(r.s_rainnc, c), F(1.e-12f))));
    put(r.pr_snownc, c, tc_add(get(r.pr_snownc, c), get(r.pr_snowncv, c)));
    put(r.pr_graupelnc, c, tc_add(get(r.pr_graupelnc, c), get(r.pr_graupelncv, c)));
    put(r.pr_frainnc, c, tc_add(get(r.pr_frainnc, c), get(r.s_frainnc, c)));
}
kernel void hexrrfs_tempo_native_to(constant HexrrfsTempoNativeToResources &r [[buffer(0)]],
                                    constant HexrrfsTempoNativeToParams &p [[buffer(1)]],
                                    uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    ulong n = ulong(uint(p.nz)) * uint(p.nc);
    F dt = F(p.dt);
    for (int k = 0; k < p.nz; ++k) {
        ulong x = ulong(k) * uint(p.nc) + c;
        if (k < p.top) {
            F oldth = get(r.m_theta_m, x),
              old_dth = tc_div(oldth, tc_add(F(1.f), tc_mul(F(p.rvord), get(r.m_qv, x))));
            r.m_qv[x] = r.prepared[4 * n + x];
            r.m_qc[x] = r.prepared[5 * n + x];
            r.m_qr[x] = r.prepared[6 * n + x];
            F th = tc_mul(get(r.prepared, x),
                          tc_add(F(1.f), tc_mul(F(p.rvord), get(r.prepared, 4 * n + x))));
            put(r.m_theta_m, x, th);
            put(r.d_rt_diabatic_tend, x, tc_div(tc_sub(th, oldth), dt));
            put(r.d_dtheta_dt_mp, x,
                tc_div(
                    tc_sub(tc_div(th, tc_add(F(1.f), tc_mul(F(p.rvord), get(r.m_qv, x)))), old_dth),
                    dt));
            F rtp = tc_sub(tc_mul(get(r.m_rho_zz, x), th), get(r.d_rtheta_base, x));
            put(r.d_rtheta_p, x, rtp);
            F ex = tc_pow(tc_mul(tc_mul(get(r.mesh_zz, x), tc_div(F(p.rd), F(p.p0))),
                                 tc_add(rtp, get(r.d_rtheta_base, x))),
                          F(p.rcv));
            put(r.d_exner, x, ex);
            put(r.d_pressure_p, x,
                tc_mul(tc_mul(get(r.mesh_zz, x), F(p.rd)),
                       tc_add(tc_mul(ex, rtp), tc_mul(tc_sub(ex, get(r.d_exner_base, x)),
                                                      get(r.d_rtheta_base, x)))));
        } else {
            r.m_qc[x] = 0.f;
            r.m_qr[x] = 0.f;
            r.d_rt_diabatic_tend[x] = 0.f;
        }
        r.m_qi[x] = r.prepared[7 * n + x];
        r.m_qs[x] = r.prepared[8 * n + x];
        r.m_qg[x] = r.prepared[9 * n + x];
        r.m_ni[x] = r.prepared[10 * n + x];
        r.m_nr[x] = r.prepared[11 * n + x];
        if (p.aerosol) {
            r.m_nc[x] = r.prepared[12 * n + x];
            r.m_nwfa[x] = r.prepared[13 * n + x];
            r.m_nifa[x] = r.prepared[14 * n + x];
        }
        if (p.hail) {
            r.m_ng[x] = r.prepared[15 * n + x];
            r.m_volg[x] = r.prepared[16 * n + x];
        }
        if (k < p.top) {
            put(r.t_rthmpten, x,
                tc_div(tc_sub(tc_div(get(r.m_theta_m, x),
                                     tc_add(F(1.f), tc_mul(tc_div(F(p.rv), F(p.rd)),
                                                           tc_max(F(0.f), get(r.m_qv, x))))),
                              get(r.saved, x)),
                       dt));
            put(r.t_rqvmpten, x, tc_div(tc_sub(get(r.m_qv, x), get(r.saved, 1 * n + x)), dt));
            put(r.t_rqcmpten, x, tc_div(tc_sub(get(r.m_qc, x), get(r.saved, 2 * n + x)), dt));
            put(r.t_rqrmpten, x, tc_div(tc_sub(get(r.m_qr, x), get(r.saved, 3 * n + x)), dt));
            put(r.t_rqimpten, x, tc_div(tc_sub(get(r.m_qi, x), get(r.saved, 4 * n + x)), dt));
            put(r.t_rqsmpten, x, tc_div(tc_sub(get(r.m_qs, x), get(r.saved, 5 * n + x)), dt));
            put(r.t_rqgmpten, x, tc_div(tc_sub(get(r.m_qg, x), get(r.saved, 6 * n + x)), dt));
        } else {
            r.t_rthmpten[x] = 0.f;
            r.t_rqvmpten[x] = 0.f;
            r.t_rqcmpten[x] = 0.f;
            r.t_rqrmpten[x] = 0.f;
            r.t_rqimpten[x] = 0.f;
            r.t_rqsmpten[x] = 0.f;
            r.t_rqgmpten[x] = 0.f;
        }
        put(r.t_rnimpten, x, tc_div(tc_sub(get(r.m_ni, x), get(r.saved, 7 * n + x)), dt));
        put(r.t_rnrmpten, x, tc_div(tc_sub(get(r.m_nr, x), get(r.saved, 8 * n + x)), dt));
        if (p.aerosol) {
            put(r.t_rncmpten, x, tc_div(tc_sub(get(r.m_nc, x), get(r.saved, 9 * n + x)), dt));
            put(r.t_rnifampten, x, tc_div(tc_sub(get(r.m_nifa, x), get(r.saved, 10 * n + x)), dt));
            put(r.t_rnwfampten, x, tc_div(tc_sub(get(r.m_nwfa, x), get(r.saved, 11 * n + x)), dt));
        }
    }
    if (p.comp_enabled)
        put(r.compmax, c, tc_max(get(r.compmax, c), get(r.comp, c)));
    F dz1 = tc_sub(get(r.mesh_zgrid, uint(p.nc) + c), get(r.mesh_zgrid, c)),
      dz2 = tc_sub(get(r.mesh_zgrid, 2 * uint(p.nc) + c), get(r.mesh_zgrid, uint(p.nc) + c));
    F rho1 = tc_mul(tc_mul(get(r.m_rho_zz, c), get(r.mesh_zz, c)),
                    tc_add(F(1.f), get(r.prepared, 4 * n + c)));
    ulong x = uint(p.nc) + c;
    F rho2 = tc_mul(tc_mul(get(r.m_rho_zz, x), get(r.mesh_zz, x)),
                    tc_add(F(1.f), get(r.prepared, 4 * n + x)));
    F oldp = get(r.d_surface_pressure, c);
    F surface = tc_mul(
        tc_mul(tc_mul(F(0.5f), F(p.grav)), dz1),
        tc_sub(rho1, tc_div(tc_mul(tc_mul(F(0.5f), tc_sub(rho2, rho1)), dz1), tc_add(dz1, dz2))));
    surface = tc_add(tc_add(surface, get(r.d_pressure_p, c)), get(r.d_pressure_base, c));
    put(r.d_surface_pressure, c, surface);
    put(r.d_tend_sfc_pressure, c, tc_div(tc_sub(surface, oldp), dt));
}
