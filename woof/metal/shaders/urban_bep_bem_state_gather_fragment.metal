#include <metal_stdlib>
#include "urban_bep_bem_state_gather_fragment_bindings.metalh"
using namespace metal;

constant uint BEM_NZ_UM = 18u;
constant uint BEM_NBUIMAX = 15u;
constant uint BEM_NWR_U = 10u;
constant uint BEM_NG_U = 10u;
constant uint BEM_NGB_U = 10u;
constant uint BEM_NF_U = 10u;
constant uint BEM_NDM = 2u;

// Exact pre-bb_bep1d prognostic gather from module_sf_bep_bem.F:782-914.
// Inputs are packed as [state-index, domain-column], the flattening of WOOF's
// contiguous [state-index, y, x] buffers. Outputs reverse each source
// Fortran local-array shape after the domain-column axis, so the first
// Fortran index remains the fastest varying packed value. One thread handles
// one domain column. Inactive columns carry a clean error code and no data writes.
kernel void urban_bep_bem_state_gather_fragment(
    constant UrbanBepBemStateGatherFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemStateGatherFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const int *active_cols = r.active_cols;
    device const float *ss_urb = r.ss_urb;
    device const float *tlev_urb3d = r.tlev_urb3d;
    device const float *qlev_urb3d = r.qlev_urb3d;
    device const float *tw1lev_urb3d = r.tw1lev_urb3d;
    device const float *tw2lev_urb3d = r.tw2lev_urb3d;
    device const float *tglev_urb3d = r.tglev_urb3d;
    device const float *tflev_urb3d = r.tflev_urb3d;
    device const float *sfwin1_urb3d = r.sfwin1_urb3d;
    device const float *sfwin2_urb3d = r.sfwin2_urb3d;
    device const float *tw1_urb4d = r.tw1_urb4d;
    device const float *tw2_urb4d = r.tw2_urb4d;
    device const float *tgb_urb4d = r.tgb_urb4d;
    device const float *trb_urb4d = r.trb_urb4d;
    device const float *trv_urb4d = r.trv_urb4d;
    device const float *qr_urb4d = r.qr_urb4d;
    device const float *sfw1_urb3d = r.sfw1_urb3d;
    device const float *sfw2_urb3d = r.sfw2_urb3d;
    device const float *sfg_urb3d = r.sfg_urb3d;
    device const float *lfg_urb3d = r.lfg_urb3d;
    device const float *dg_urb3d = r.dg_urb3d;
    device const float *t_pv_urb3d = r.t_pv_urb3d;
    device const float *sfr_urb3d = r.sfr_urb3d;
    device const float *lfr_urb3d = r.lfr_urb3d;
    device const float *dgr_urb3d = r.dgr_urb3d;
    device const float *sfrv_urb3d = r.sfrv_urb3d;
    device const float *lfrv_urb3d = r.lfrv_urb3d;
    device const float *drain_urb4d = r.drain_urb4d;

    device int *error = r.error;
    device int *building_nlev = r.building_nlev;
    device float *tlev = r.tlev;
    device float *qlev = r.qlev;
    device float *tw = r.tw;
    device float *tg = r.tg;
    device float *tr = r.tr;
    device float *trv = r.trv;
    device float *qr = r.qr;
    device float *twlev = r.twlev;
    device float *tglev = r.tglev;
    device float *tflev = r.tflev;
    device float *sfwin = r.sfwin;
    device float *sfw = r.sfw;
    device float *sfg = r.sfg;
    device float *lfg = r.lfg;
    device float *dg = r.dg;
    device float *tpv = r.tpv;
    device float *sfr = r.sfr;
    device float *lfr = r.lfr;
    device float *dgr = r.dgr;
    device float *sfrv = r.sfrv;
    device float *lfrv = r.lfrv;
    device float *drain = r.drain;

    if (p.ncol <= 0) return;
    const uint ncol = uint(p.ncol);
    if (col >= ncol) return;
    if (active_cols[col] == 0) { error[col] = 0; return; }
    const int green_roof = p.green_roof;
    if (green_roof != 0 && green_roof != 1) { error[col] = 5010; return; }

    // WRF packs each positive ss_urb level into nlev in ascending iz order
    // (module_sf_bep_bem.F:739-748). Refuse overflow rather than truncate a
    // building profile beyond the 15-entry local-array capacity.
    for (uint b = 0; b < BEM_NBUIMAX; ++b)
        building_nlev[ulong(col) * BEM_NBUIMAX + b] = 0;
    uint building = 0u;
    for (uint z = 0; z < BEM_NZ_UM; ++z) {
        if (ss_urb[ulong(col) * (BEM_NZ_UM + 1u) + z] > 0.0f) {
            if (building >= BEM_NBUIMAX) { error[col] = 5011; return; }
            building_nlev[ulong(col) * BEM_NBUIMAX + building] = int(z);
            ++building;
        }
    }
    for (uint b = 0; b < BEM_NBUIMAX; ++b) {
        for (uint z = 0; z < BEM_NZ_UM; ++z) {
            const uint bd = b * BEM_NZ_UM + z;
            const ulong out2 = (ulong(col) * BEM_NBUIMAX + b) * BEM_NZ_UM + z;
            tlev[out2] = tlev_urb3d[ulong(bd) * ncol + col];
            qlev[out2] = qlev_urb3d[ulong(bd) * ncol + col];
        }
    }

    // tw1d(2*id-1/2*id, iz_u, iw, ibui); source ind_zwd packs
    // (ibui, iz_u, iw, id) with id fastest.
    for (uint b = 0; b < BEM_NBUIMAX; ++b) {
        for (uint z = 0; z < BEM_NZ_UM; ++z) {
            for (uint w = 0; w < BEM_NWR_U; ++w) {
                for (uint id = 0; id < BEM_NDM; ++id) {
                    const uint src = (((b * BEM_NZ_UM + z) * BEM_NWR_U + w) * BEM_NDM + id);
                    const ulong out = ((((ulong(col) * BEM_NBUIMAX + b) * BEM_NWR_U + w) * BEM_NZ_UM + z) * 4ul) + 2ul * id;
                    tw[out] = tw1_urb4d[ulong(src) * ncol + col];
                    tw[out + 1ul] = tw2_urb4d[ulong(src) * ncol + col];
                }
            }
        }
    }
    // tg1d(id, ig) = tgb_urb4d(ind_gd(ig,id));
    for (uint g = 0; g < BEM_NG_U; ++g)
        for (uint id = 0; id < BEM_NDM; ++id) {
            const uint src = g * BEM_NDM + id;
            tg[(ulong(col) * BEM_NG_U + g) * BEM_NDM + id] = tgb_urb4d[ulong(src) * ncol + col];
        }
    // tr1d(id,iz,ir), and green-roof trv1d/qr1d use ind_zrd/ind_grd.
    for (uint ir = 0; ir < BEM_NWR_U; ++ir)
        for (uint z = 0; z < BEM_NZ_UM; ++z)
            for (uint id = 0; id < BEM_NDM; ++id) {
                const uint src = (z * BEM_NWR_U + ir) * BEM_NDM + id;
                const ulong out = ((ulong(col) * BEM_NWR_U + ir) * BEM_NZ_UM + z) * BEM_NDM + id;
                tr[out] = trb_urb4d[ulong(src) * ncol + col];
            }
    for (uint ir = 0; ir < BEM_NGB_U; ++ir)
        for (uint z = 0; z < BEM_NZ_UM; ++z)
            for (uint id = 0; id < BEM_NDM; ++id) {
                const uint src = (z * BEM_NGB_U + ir) * BEM_NDM + id;
                const ulong out = ((ulong(col) * BEM_NGB_U + ir) * BEM_NZ_UM + z) * BEM_NDM + id;
                if (green_roof == 1) {
                    trv[out] = trv_urb4d[ulong(src) * ncol + col];
                    qr[out] = qr_urb4d[ulong(src) * ncol + col];
                } else { trv[out] = 0.0f; qr[out] = 0.0f; }
            }

    // twlev1d/sfwin1d use ind_wd(ibui,iz_u,id), with two source values
    // packed in each direction pair.
    for (uint b = 0; b < BEM_NBUIMAX; ++b)
        for (uint z = 0; z < BEM_NZ_UM; ++z)
            for (uint id = 0; id < BEM_NDM; ++id) {
                const uint src = (b * BEM_NZ_UM + z) * BEM_NDM + id;
                const ulong out = ((ulong(col) * BEM_NBUIMAX + b) * BEM_NZ_UM + z) * 4ul + 2ul * id;
                twlev[out] = tw1lev_urb3d[ulong(src) * ncol + col];
                twlev[out + 1ul] = tw2lev_urb3d[ulong(src) * ncol + col];
                sfwin[out] = sfwin1_urb3d[ulong(src) * ncol + col];
                sfwin[out + 1ul] = sfwin2_urb3d[ulong(src) * ncol + col];
            }
    for (uint b = 0; b < BEM_NBUIMAX; ++b)
        for (uint w = 0; w < BEM_NGB_U; ++w)
            for (uint id = 0; id < BEM_NDM; ++id) {
                const uint src = (b * BEM_NGB_U + w) * BEM_NDM + id;
                tglev[((ulong(col) * BEM_NBUIMAX + b) * BEM_NGB_U + w) * BEM_NDM + id] = tglev_urb3d[ulong(src) * ncol + col];
            }
    for (uint b = 0; b < BEM_NBUIMAX; ++b)
        for (uint w = 0; w < BEM_NF_U; ++w)
            for (uint z = 0; z < BEM_NZ_UM - 1u; ++z)
                for (uint id = 0; id < BEM_NDM; ++id) {
                    const uint src = (((b * BEM_NF_U + w) * (BEM_NZ_UM - 1u) + z) * BEM_NDM + id);
                    tflev[(((ulong(col) * BEM_NBUIMAX + b) * BEM_NF_U + w) * (BEM_NZ_UM - 1u) + z) * BEM_NDM + id] = tflev_urb3d[ulong(src) * ncol + col];
                }
    for (uint b = 0; b < BEM_NBUIMAX; ++b)
        for (uint z = 0; z < BEM_NZ_UM; ++z)
            for (uint id = 0; id < BEM_NDM; ++id) {
                const uint src = (b * BEM_NZ_UM + z) * BEM_NDM + id;
                const ulong out = ((ulong(col) * BEM_NBUIMAX + b) * BEM_NZ_UM + z) * 4ul + 2ul * id;
                sfw[out] = sfw1_urb3d[ulong(src) * ncol + col];
                sfw[out + 1ul] = sfw2_urb3d[ulong(src) * ncol + col];
            }

    // Directly indexed two-direction and zdf fields.
    for (uint id = 0; id < BEM_NDM; ++id) {
        const ulong out = ulong(col) * BEM_NDM + id;
        const uint src = id;
        sfg[out] = sfg_urb3d[ulong(src) * ncol + col];
        lfg[out] = lfg_urb3d[ulong(src) * ncol + col];
        dg[out] = dg_urb3d[ulong(src) * ncol + col];
    }
    for (uint z = 0; z < BEM_NZ_UM; ++z)
        for (uint id = 0; id < BEM_NDM; ++id) {
            const uint src = z * BEM_NDM + id;
            const ulong out = (ulong(col) * BEM_NZ_UM + z) * BEM_NDM + id;
            tpv[out] = t_pv_urb3d[ulong(src) * ncol + col];
            sfr[out] = sfr_urb3d[ulong(src) * ncol + col];
            lfr[out] = lfr_urb3d[ulong(src) * ncol + col];
            dgr[out] = dgr_urb3d[ulong(src) * ncol + col];
            if (green_roof == 1) {
                sfrv[out] = sfrv_urb3d[ulong(src) * ncol + col];
                lfrv[out] = lfrv_urb3d[ulong(src) * ncol + col];
                drain[out] = drain_urb4d[ulong(src) * ncol + col];
            } else { sfrv[out] = 0.0f; lfrv[out] = 0.0f; drain[out] = 0.0f; }
        }
    error[col] = 0;
}
