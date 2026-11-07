#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_exp(x) gfk_exp(x)
#include "hexv842_mpas_rrtmg_lw_cmbgb_f32_bindings.metalh"
constant int mpas_static_ngc[16] = {
10,12,16,14,16,8,12,8,
12,6,8,8,4,2,2,2
};
constant int mpas_static_ngn[140] = {
1,1,2,2,2,2,2,2,
1,1,1,1,1,1,1,1,
1,1,2,2,2,2,1,1,
1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,
1,1,1,3,1,1,1,1,
1,1,1,1,1,1,1,1,
1,1,1,1,2,2,2,2,
2,2,2,2,2,2,1,1,
1,1,1,1,1,1,2,2,
2,2,2,2,2,2,2,2,
1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,
4,4,1,1,2,2,2,2,
3,3,1,1,1,1,2,2,
4,4,3,3,4,6,8,8,
8,8,4,12
};
constant int mpas_static_ngm[256] = {
1,2,3,3,4,4,5,5,
6,6,7,7,8,8,9,10,
1,2,3,4,5,6,7,8,
9,9,10,10,11,11,12,12,
1,2,3,4,5,6,7,8,
9,10,11,12,13,14,15,16,
1,2,3,4,5,6,7,8,
9,10,11,12,13,14,14,14,
1,2,3,4,5,6,7,8,
9,10,11,12,13,14,15,16,
1,1,2,2,3,3,4,4,
5,5,6,6,7,7,8,8,
1,1,2,2,3,4,5,6,
7,8,9,10,11,11,12,12,
1,1,2,2,3,3,4,4,
5,5,6,6,7,7,8,8,
1,2,3,4,5,6,7,8,
9,9,10,10,11,11,12,12,
1,1,2,2,3,3,4,4,
5,5,5,5,6,6,6,6,
1,2,3,3,4,4,5,5,
6,6,7,7,7,8,8,8,
1,2,3,4,5,5,6,6,
7,7,7,7,8,8,8,8,
1,1,1,2,2,2,3,3,
3,3,4,4,4,4,4,4,
1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,
1,1,1,1,1,1,1,1,
2,2,2,2,2,2,2,2,
1,1,1,1,2,2,2,2,
2,2,2,2,2,2,2,2
};
constant float mpas_static_wt[16] = {
0x1.38d6ca0000000p-3f,0x1.3181980000000p-3f,0x1.23034e0000000p-3f,0x1.0db2c80000000p-3f,0x1.e41ff00000000p-4f,0x1.a181720000000p-4f,0x1.5519f40000000p-4f,0x1.00b45e0000000p-4f,
0x1.5c193c0000000p-5f,0x1.2f3c000000000p-8f,0x1.f5be000000000p-9f,0x1.8ca0000000000p-9f,0x1.22fa000000000p-9f,0x1.72ac000000000p-10f,0x1.17720c0000000p-11f,0x1.3a92a40000000p-14f
};
kernel void hexv842_mpas_rrtmg_lw_cmbgb_f32(constant Hexv842MpasRrtmgLwCmbgbF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwCmbgbF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {

 // MPAS: module_ra_rrtmg_lw.F:8331-9982 (all 16 cmbgb routines).
 // Each thread owns one independent point and reduction. The original
 // order within sumk/sumf is preserved; interleaved independent sums
 // need no communication.
 int idx=int(c);int groups=mpas_static_ngc[p.band];
 if(idx>=p.points*groups)return;
 int g=idx/p.points,point=idx%p.points,start=0,offset=0;
 for(int b=0;b<p.band;++b)offset+=mpas_static_ngc[b];
 for(int j=0;j<g;++j)start+=mpas_static_ngn[offset+j];
 float sum=0.0f;
 for(int j=0;j<mpas_static_ngn[offset+g];++j){int old=start+j;
   float term=r.raw[p.first?old+16*point:point+p.points*old];
   if(p.weighted)term=RLW_MU(term,r.rwgt[p.band*16+old]);
   sum=RLW_AD(sum,term);
 }
 r.out[p.first?g+groups*point:point+p.points*g]=sum;
}
