#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexrrfs_tempo_columns_bindings.metalh"
#include "hexrrfs_tempo_workspace_sizes_bindings.metalh"

namespace rrfs_tempo {
#define TNZMAX 128
#include "hexrrfs_tempo_core.metalh"
struct TempoState { F f[17][TNZMAX + 3]; };
#undef TNZMAX
}

using rrfs_tempo::TempoCtx;
using rrfs_tempo::TempoParams;
using rrfs_tempo::TempoState;
using rrfs_tempo::TempoWork;
static_assert(sizeof(TempoState) % sizeof(uint) == 0, "TEMPO state alignment");
static_assert(sizeof(TempoWork) % sizeof(ulong) == 0, "TEMPO work alignment");

kernel void hexrrfs_tempo_workspace_sizes(
    constant HexrrfsTempoWorkspaceSizesResources &r [[buffer(0)]],
    constant HexrrfsTempoWorkspaceSizesParams &p [[buffer(1)]],
    uint slot [[thread_position_in_grid]]) {
    if (slot >= uint(p.n)) return;
    r.sizes[0] = sizeof(TempoWork) / sizeof(ulong);
    r.sizes[1] = sizeof(TempoState) / sizeof(uint);
}

kernel void hexrrfs_tempo_columns(
    constant HexrrfsTempoColumnsResources &r [[buffer(0)]],
    constant HexrrfsTempoColumnsParams &p [[buffer(1)]],
    uint slot [[thread_position_in_grid]]) {
    if (slot >= uint(p.batch)) return;
    const uint column = uint(p.offset) + slot;
    const ulong ncol = uint(p.ncol);
    const int nz = p.nz;
    device TempoState &state = reinterpret_cast<device TempoState *>(r.state)[slot];
    device TempoWork &work = reinterpret_cast<device TempoWork *>(r.work)[slot];
    device ulong *work_words = r.work + ulong(slot) * uint(p.work_words);
    for (int i = 0; i < p.work_words; ++i) work_words[i] = 0ul;
    for (int f = 0; f < 17; ++f) {
        for (int k = 1; k <= nz; ++k) {
            const ulong cell = ulong(f * nz + k - 1) * ncol + column;
            state.f[f][k] = f32_word(as_type<uint>(r.fields[cell]));
        }
    }

    const TempoParams tables = {reinterpret_cast<device const Soft8 *>(r.tables)};
    const F dt = f32_word(as_type<uint>(p.dt));
    TempoCtx ctx = {};
    ctx.global_dt = dt;
    ctx.global_inverse_dt = ntc::tc_div(F(1), dt);
    ctx.aerosolaware_flag = p.aerosol;
    ctx.hailaware_flag = p.hail;
    ctx.cloud_condensation_flag = 1;
    ctx.refl10cm_flag = 1;
    ctx.re_cloud_flag = 1;
    ctx.re_ice_flag = 1;
    ctx.re_snow_flag = 1;
    ctx.max_hail_diameter_flag = 1;
    ctx.first_call_main = p.first && column == 0;
    device F (*f)[sizeof(TempoState) / (17 * sizeof(F))] = state.f;
    const int mask = p.presence;
    rrfs_tempo::tempo_main_column(
        tables, ctx, nz, work, f[4], f[5], f[7], f[6], f[8], f[9],
        (mask & 16) ? f[16] : nullptr, f[10], f[11],
        (mask & 1) ? f[12] : nullptr, (mask & 8) ? f[15] : nullptr,
        (mask & 2) ? f[13] : nullptr, (mask & 4) ? f[14] : nullptr,
        f[0], f[1], f[2], f[3]);
    if (p.use_pbl) {
        device F *pbl = reinterpret_cast<device F *>(r.blwork)
            + ulong(slot) * uint(p.bl_words);
        for (int k = 0; k < 2 * nz; ++k) {
            pbl[k] = f32_word(as_type<uint>(r.pbl[ulong(k) * ncol + column]));
        }
        rrfs_tempo::tempo_subgrid_cloud_radii(tables, ctx, nz, work,
            f[5], f[6], f[7], f[8], (mask & 1) ? f[12] : nullptr,
            f[2], pbl, pbl + nz, 1, 0);
    }
    for (int j = 0; j < 17; ++j) {
        for (int k = 1; k <= nz; ++k) {
            const ulong cell = ulong(j * nz + k - 1) * ncol + column;
            r.fields[cell] = as_type<float>(f[j][k].bits);
        }
    }

    F max_hail = work.max_hail_diameter[1];
    for (int k = 2; k <= nz; ++k) {
        const F value = work.max_hail_diameter[k];
        max_hail = max_hail > value ? max_hail : value;
    }
    const F surface[8] = {work.rain_precip, work.ice_liquid_equiv_precip,
        work.snow_liquid_equiv_precip, work.graupel_liquid_equiv_precip,
        work.frz_rain_precip, work.frozen_fraction, work.max_hail_diameter[1], max_hail};
    const device F *profiles[4] = {work.refl10cm, work.re_cloud, work.re_ice, work.re_snow};
    for (int j = 0; j < 15; ++j) {
        for (int k = 0; k < nz; ++k) {
            const ulong cell = ulong(j * nz + k) * ncol + column;
            uint word = 0;
            if (j < 8 && k == 0) word = surface[j].bits;
            if (j >= 8 && j < 12) word = profiles[j - 8][k + 1].bits;
            r.diagnostics[cell] = as_type<float>(word);
        }
    }
}
