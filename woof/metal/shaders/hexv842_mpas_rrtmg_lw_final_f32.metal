#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define NBNDLW 16
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#include "hexv842_mpas_rrtmg_lw_final_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_final_f32(constant Hexv842MpasRrtmgLwFinalF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwFinalF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol;
 int nl=p.nl;
 float fluxfac=p.fluxfac;
 float heatfac=p.heatfac;
 device const float* pz=r.pz;
 device float* totuflux=r.totuflux;
 device float* totdflux=r.totdflux;
 device float* totuclfl=r.totuclfl;
 device float* totdclfl=r.totdclfl;
 device float* fnet=r.fnet;
 device float* fnetc=r.fnetc;
 device float* htr=r.htr;
 device float* htrc=r.htrc;

    int col = int(gid);
    if (col >= ncol) return;
#define LV(a, l) a[(long)col * (nl + 1) + (l)]
    LV(totuflux, 0) = RLW_MU(LV(totuflux, 0), fluxfac);
    LV(totdflux, 0) = RLW_MU(LV(totdflux, 0), fluxfac);
    LV(fnet, 0) = RLW_SU(LV(totuflux, 0), LV(totdflux, 0));
    LV(totuclfl, 0) = RLW_MU(LV(totuclfl, 0), fluxfac);
    LV(totdclfl, 0) = RLW_MU(LV(totdclfl, 0), fluxfac);
    LV(fnetc, 0) = RLW_SU(LV(totuclfl, 0), LV(totdclfl, 0));
    for (int lev = 1; lev <= nl; ++lev) {
        LV(totuflux, lev) = RLW_MU(LV(totuflux, lev), fluxfac);
        LV(totdflux, lev) = RLW_MU(LV(totdflux, lev), fluxfac);
        LV(fnet, lev) = RLW_SU(LV(totuflux, lev), LV(totdflux, lev));
        LV(totuclfl, lev) = RLW_MU(LV(totuclfl, lev), fluxfac);
        LV(totdclfl, lev) = RLW_MU(LV(totdclfl, lev), fluxfac);
        LV(fnetc, lev) = RLW_SU(LV(totuclfl, lev), LV(totdclfl, lev));
        int l0 = lev - 1;
        LV(htr, l0) = RLW_DV(
            RLW_MU(heatfac, RLW_SU(LV(fnet, l0), LV(fnet, lev))),
            RLW_SU(LV(pz, l0), LV(pz, lev)));
        LV(htrc, l0) = RLW_DV(
            RLW_MU(heatfac, RLW_SU(LV(fnetc, l0), LV(fnetc, lev))),
            RLW_SU(LV(pz, l0), LV(pz, lev)));
    }
    LV(htr, nl) = 0.0f;
    LV(htrc, nl) = 0.0f;
#undef LV

}
