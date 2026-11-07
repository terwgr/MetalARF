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
#include "hexv842_mpas_rrtmg_lw_accum_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_accum_f32(constant Hexv842MpasRrtmgLwAccumF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwAccumF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol;
 int nl=p.nl;
 float wtdiff=p.wtdiff;
 device const float* radld_p=r.radld_p;
 device const float* radclrd_p=r.radclrd_p;
 device const float* radlu_p=r.radlu_p;
 device const float* radclru_p=r.radclru_p;
 device const int* iclddn_p=r.iclddn_p;
 device const float* radlu_sfc=r.radlu_sfc;
 device const float* radclru_sfc=r.radclru_sfc;
 device const int* ngs=r.ngs;
 device const float* delwave=r.delwave;
 device float* totuflux=r.totuflux;
 device float* totdflux=r.totdflux;
 device float* totuclfl=r.totuclfl;
 device float* totdclfl=r.totdclfl;

    long tid = long(gid);
    if (tid >= (long)ncol * (nl + 1)) return;
    int col = (int)(tid / (nl + 1));
    int lev = (int)(tid % (nl + 1));   // 0..nl

    float totu = 0.0f, totd = 0.0f, totuc = 0.0f, totdc = 0.0f;
#define LP(a, lane) a[((long)col * nl + (lev - 1)) * NGPTLW + (lane)]
#define LP0(a, lane) a[((long)col * nl + (lev)) * NGPTLW + (lane)]
    for (int iband = 1; iband <= NBNDLW; ++iband) {
        int g_lo = (iband == 1) ? 0 : ngs[iband - 2];
        int g_hi = ngs[iband - 1];
        float urad = 0.0f, drad = 0.0f, clru = 0.0f, clrd = 0.0f;
        for (int lane = g_lo; lane < g_hi; ++lane) {
            // downward: drad/clrdrad live at levels 0..nl-1
            if (lev <= nl - 1) {
                drad = RLW_AD(drad, LP0(radld_p, lane));
                if (iclddn_p[((long)col * nl + lev) * NGPTLW + lane])
                    clrd = RLW_AD(clrd, LP0(radclrd_p, lane));
                else
                    clrd = drad;
            }
            // upward: urad/clrurad at levels 0 (sfc) and 1..nl
            if (lev == 0) {
                urad = RLW_AD(urad,
                    radlu_sfc[(long)col * NGPTLW + lane]);
                if (iclddn_p[((long)col * nl + 0) * NGPTLW + lane])
                    clru = RLW_AD(clru,
                        radclru_sfc[(long)col * NGPTLW + lane]);
                else
                    clru = urad;
            } else {
                urad = RLW_AD(urad, LP(radlu_p, lane));
                if (iclddn_p[((long)col * nl + 0) * NGPTLW + lane])
                    clru = RLW_AD(clru, LP(radclru_p, lane));
                else
                    clru = urad;
            }
        }
        float dw = delwave[iband - 1];
        totu = RLW_AD(totu, RLW_MU(RLW_MU(urad, wtdiff), dw));
        totd = RLW_AD(totd, RLW_MU(RLW_MU(drad, wtdiff), dw));
        totuc = RLW_AD(totuc, RLW_MU(RLW_MU(clru, wtdiff), dw));
        totdc = RLW_AD(totdc, RLW_MU(RLW_MU(clrd, wtdiff), dw));
    }
    totuflux[(long)col * (nl + 1) + lev] = totu;
    totdflux[(long)col * (nl + 1) + lev] = totd;
    totuclfl[(long)col * (nl + 1) + lev] = totuc;
    totdclfl[(long)col * (nl + 1) + lev] = totdc;
#undef LP
#undef LP0

}
