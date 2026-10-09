#include <metal_stdlib>
#include "urban_bep_bem_state_scatter_fragment_bindings.metalh"
using namespace metal;

constant uint BEM_SCATTER_NZ = 18u;
constant uint BEM_SCATTER_B = 15u;
constant uint BEM_SCATTER_W = 10u;
constant uint BEM_SCATTER_NDM = 2u;
constant uint BEM_SCATTER_NF = 10u;
constant uint BEM_SCATTER_NG = 10u;

// Inverse of urban_bep_bem_state_gather_fragment, following the ind_* packing
// in module_sf_bep_bem.F:782-914. One item is one domain column. The driver
// calls this only after all BEM source-order stages have succeeded, so this
// is the single commit point for the resident prognostic arrays.
kernel void urban_bep_bem_state_scatter_fragment(
    constant UrbanBepBemStateScatterFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemStateScatterFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    device const int *active = r.active_cols;
    device const float *tlev = r.tlev, *qlev = r.qlev;
    device const float *tw = r.tw, *twlev = r.twlev, *tg = r.tg;
    device const float *tr = r.tr, *trv = r.trv, *qr = r.qr;
    device const float *tflev = r.tflev, *tglev = r.tglev;
    device const float *sfwin = r.sfwin, *sfw = r.sfw;
    device const float *sfg = r.sfg, *lfg = r.lfg, *dg = r.dg;
    device const float *tpv = r.tpv, *sfr = r.sfr, *lfr = r.lfr;
    device const float *dgr = r.dgr, *sfrv = r.sfrv, *lfrv = r.lfrv;
    device const float *drain = r.drain;
    device const float *tlev_next = r.tlev_next, *qlev_next = r.qlev_next;
    device float *tlev_state = r.tlev_urb3d, *qlev_state = r.qlev_urb3d;
    device float *tw1lev = r.tw1lev_urb3d, *tw2lev = r.tw2lev_urb3d;
    device float *tglev_state = r.tglev_urb3d, *tflev_state = r.tflev_urb3d;
    device float *sfwin1 = r.sfwin1_urb3d, *sfwin2 = r.sfwin2_urb3d;
    device float *tw1 = r.tw1_urb4d, *tw2 = r.tw2_urb4d;
    device float *tgb = r.tgb_urb4d, *trb = r.trb_urb4d;
    device float *trv_state = r.trv_urb4d, *qr_state = r.qr_urb4d;
    device float *sfw1 = r.sfw1_urb3d, *sfw2 = r.sfw2_urb3d;
    device float *sfg_state = r.sfg_urb3d, *lfg_state = r.lfg_urb3d;
    device float *dg_state = r.dg_urb3d, *tpv_state = r.t_pv_urb3d;
    device float *sfr_state = r.sfr_urb3d, *lfr_state = r.lfr_urb3d;
    device float *dgr_state = r.dgr_urb3d, *sfrv_state = r.sfrv_urb3d;
    device float *lfrv_state = r.lfrv_urb3d, *drain_state = r.drain_urb4d;

    if (active[col] == 0) return;
    const ulong c = ulong(col);
    for (uint b = 0; b < BEM_SCATTER_B; ++b) {
        for (uint z = 0; z < BEM_SCATTER_NZ; ++z) {
            const uint bd = b * BEM_SCATTER_NZ + z;
            const ulong out = (ulong(bd) * uint(p.ncol)) + c;
            const ulong air = (c * BEM_SCATTER_B + b) * BEM_SCATTER_NZ + z;
            tlev_state[out] = tlev_next[air];
            qlev_state[out] = qlev_next[air];
            for (uint w = 0; w < BEM_SCATTER_W; ++w) {
                const uint src = ((b * BEM_SCATTER_NZ + z) * BEM_SCATTER_W + w) * BEM_SCATTER_NDM;
                const ulong wall = (((c * BEM_SCATTER_B + b) * BEM_SCATTER_W + w) * BEM_SCATTER_NZ + z) * 4ul;
                for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
                    const ulong state = ulong(src + id) * uint(p.ncol) + c;
                    tw1[state] = tw[wall + 2ul * id];
                    tw2[state] = tw[wall + 2ul * id + 1ul];
                }
            }
            for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
                const uint src = (b * BEM_SCATTER_NZ + z) * BEM_SCATTER_NDM + id;
                const ulong win = ((c * BEM_SCATTER_B + b) * BEM_SCATTER_NZ + z) * 4ul + 2ul * id;
                const ulong state = ulong(src) * uint(p.ncol) + c;
                tw1lev[state] = twlev[win]; tw2lev[state] = twlev[win + 1ul];
                sfwin1[state] = sfwin[win]; sfwin2[state] = sfwin[win + 1ul];
            }
        }
        for (uint g = 0; g < BEM_SCATTER_NG; ++g)
            for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
                const uint src = (b * BEM_SCATTER_NG + g) * BEM_SCATTER_NDM + id;
                const ulong in = ((c * BEM_SCATTER_B + b) * BEM_SCATTER_NG + g) * BEM_SCATTER_NDM + id;
                tglev_state[ulong(src) * uint(p.ncol) + c] = tglev[in];
            }
        for (uint w = 0; w < BEM_SCATTER_NF; ++w)
            for (uint z = 0; z < BEM_SCATTER_NZ - 1u; ++z)
                for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
                    const uint src = (((b * BEM_SCATTER_NF + w) * (BEM_SCATTER_NZ - 1u) + z) * BEM_SCATTER_NDM + id);
                    const ulong in = (((c * BEM_SCATTER_B + b) * BEM_SCATTER_NF + w) * (BEM_SCATTER_NZ - 1u) + z) * BEM_SCATTER_NDM + id;
                    tflev_state[ulong(src) * uint(p.ncol) + c] = tflev[in];
                }
        for (uint z = 0; z < BEM_SCATTER_NZ; ++z)
            for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
                const uint src = (b * BEM_SCATTER_NZ + z) * BEM_SCATTER_NDM + id;
                const ulong in = ((c * BEM_SCATTER_B + b) * BEM_SCATTER_NZ + z) * 4ul + 2ul * id;
                const ulong out = ulong(src) * uint(p.ncol) + c;
                sfw1[out] = sfw[in]; sfw2[out] = sfw[in + 1ul];
            }
    }
    for (uint w = 0; w < BEM_SCATTER_W; ++w)
        for (uint z = 0; z < BEM_SCATTER_NZ; ++z)
            for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
                const uint src = (z * BEM_SCATTER_W + w) * BEM_SCATTER_NDM + id;
                const ulong in = ((c * BEM_SCATTER_W + w) * BEM_SCATTER_NZ + z) * BEM_SCATTER_NDM + id;
                trb[ulong(src) * uint(p.ncol) + c] = tr[in];
                if (p.green_roof == 1) {
                    trv_state[ulong(src) * uint(p.ncol) + c] = trv[in];
                    qr_state[ulong(src) * uint(p.ncol) + c] = qr[in];
                }
            }
    for (uint g = 0; g < BEM_SCATTER_NG; ++g)
        for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
            const ulong in = (c * BEM_SCATTER_NG + g) * BEM_SCATTER_NDM + id;
            const uint src = g * BEM_SCATTER_NDM + id;
            tgb[ulong(src) * uint(p.ncol) + c] = tg[in];
        }
    for (uint z = 0; z < BEM_SCATTER_NZ; ++z)
        for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
            const ulong in = (c * BEM_SCATTER_NZ + z) * BEM_SCATTER_NDM + id;
            const uint src = z * BEM_SCATTER_NDM + id;
            const ulong out = ulong(src) * uint(p.ncol) + c;
            tpv_state[out] = tpv[in]; sfr_state[out] = sfr[in];
            lfr_state[out] = lfr[in]; dgr_state[out] = dgr[in];
            if (p.green_roof == 1) {
                sfrv_state[out] = sfrv[in]; lfrv_state[out] = lfrv[in];
                drain_state[out] = drain[in];
            }
        }
    for (uint id = 0; id < BEM_SCATTER_NDM; ++id) {
        const ulong in = c * BEM_SCATTER_NDM + id;
        sfg_state[ulong(id) * uint(p.ncol) + c] = sfg[in];
        lfg_state[ulong(id) * uint(p.ncol) + c] = lfg[in];
        dg_state[ulong(id) * uint(p.ncol) + c] = dg[in];
    }
}
