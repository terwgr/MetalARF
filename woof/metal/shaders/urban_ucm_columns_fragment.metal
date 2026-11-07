// Standalone SLUCM column entry (CUDA ucm_column_test): n independent urban
// columns from canopy inputs and the grid-state planes, outputs and updated
// state returned in bundles. The canopy is urban_ucm_core.metalh.
#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_urban_trig.metalh"
#include "soft_libm64.metalh"
#include "ucm_columns_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)
#include "urban_ucm_core.metalh"
#include "urban_ucm_state.metalh"

kernel void ucm_columns(
    constant UcmColumnsResources &r [[buffer(0)]],
    constant UcmColumnsParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.n)) return;
    const ulong n = ulong(p.n);
    device const float *in = r.inputs;
    UcmCol c;
    c.utype = r.utype[index];
    c.jmonth = r.jmonth[index];
    c.ta = in[0 * n + index];
    c.qa = in[1 * n + index];
    c.ua = in[2 * n + index];
    c.u1 = in[3 * n + index];
    c.v1 = in[4 * n + index];
    c.ssg = in[5 * n + index];
    c.llg = in[6 * n + index];
    c.rain = in[7 * n + index];
    c.rhoo = in[8 * n + index];
    c.za = in[9 * n + index];
    c.omg = in[10 * n + index];
    c.delt = in[11 * n + index];
    c.znt = in[12 * n + index];
    c.chs = in[13 * n + index];
    c.chs2 = in[14 * n + index];
    device float *state = r.state;
    ucm_load_state(state, index, n, c);
    device const float *tab = r.float_bundle;
    device const float *glob = tab + uint(p.n_utype) * PT_NCOL;
    const int code = ucm_urban(tab, glob, r.isw, c);
    r.codes[index] = code;
    if (code != UCM_OK) return;
    device float *out = r.outputs;
    out[0 * n + index] = c.ts;
    out[1 * n + index] = c.qs;
    out[2 * n + index] = c.sh;
    out[3 * n + index] = c.lh;
    out[4 * n + index] = c.lh_kin;
    out[5 * n + index] = c.sw;
    out[6 * n + index] = c.alb;
    out[7 * n + index] = c.lw;
    out[8 * n + index] = c.g;
    out[9 * n + index] = c.rn;
    out[10 * n + index] = c.psim;
    out[11 * n + index] = c.psih;
    out[12 * n + index] = c.gz1oz0;
    out[13 * n + index] = c.u10;
    out[14 * n + index] = c.v10;
    out[15 * n + index] = c.th2;
    out[16 * n + index] = c.q2;
    out[17 * n + index] = c.ust;
    out[18 * n + index] = c.znt;
    ucm_store_state(state, index, n, c);
}
