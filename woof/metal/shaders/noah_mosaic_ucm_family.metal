// Noah mosaic + SLUCM urban column (noahdrv.F:3733-4057). The mosaic body is
// noah_mosaic_tile.metalh with NOAH_MOSAIC_UCM; the canopy is the shared
// urban_ucm_core.metalh. CUDA passes the UCM grid state as a table of device
// pointers; Metal passes one bundle with the planes in SP_* order (the layered
// planes TRL, TBL, TGL, TGRL and SMR take four planes each) and the per-tile
// planes as ordinary resources.
#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_urban_trig.metalh"
#include "soft_libm64.metalh"
using namespace metal;
#pragma clang fp contract(off)
#include "urban_ucm_core.metalh"

#include "urban_ucm_state.metalh"
#define PLANE(ptrs, i) ((ptrs)[(i)])

static inline float ucm_wind(float u, float v)
{
    float ua = FSQRT(FADD(gfk_pow(u, 2.0f), gfk_pow(v, 2.0f)));
    if (ua < 1.0f) ua = 1.0f;
    return ua;
}
static inline void ucm_flag(device uint* err, int code)
{
    if (code != UCM_OK)
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(err),
                                 1u << (code - 1), memory_order_relaxed);
}
static inline void ucm_floor_exchange(thread float* chs, thread float* chs2, thread float* cqs2, size_t idx)
{
    if (chs[idx] < 1.0e-02f) chs[idx] = 1.0e-02f;
    if (chs2[idx] < 1.0e-02f) chs2[idx] = 1.0e-02f;
    if (cqs2[idx] < 1.0e-02f) cqs2[idx] = 1.0e-02f;
}

#define NOAH_MOSAIC_UCM 1
#include "noah_mosaic_tile.metalh"
