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
#include "hexv842_mpas_rrtmg_lw_rwgt_f32_bindings.metalh"
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
kernel void hexv842_mpas_rrtmg_lw_rwgt_f32(constant Hexv842MpasRrtmgLwRwgtF32Resources &r [[buffer(0)]], constant Hexv842MpasRrtmgLwRwgtF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {

 // MPAS: module_ra_rrtmg_lw.F:8082-8113 (rrtmg_lw_ini).
 if(c)return;
 int total=0;float sum[16];
 for(int band=0;band<16;++band){
   if(mpas_static_ngc[band]<16){
     int old=0;
     for(int g=0;g<mpas_static_ngc[band];++g){float weight=0.0f;
       for(int j=0;j<mpas_static_ngn[total];++j){weight=RLW_AD(weight,mpas_static_wt[old]);++old;}
       sum[g]=weight;++total;
     }
     for(int g=0;g<16;++g)r.rwgt[band*16+g]=RLW_DV(mpas_static_wt[g],sum[mpas_static_ngm[band*16+g]-1]);
   } else {for(int g=0;g<16;++g){r.rwgt[band*16+g]=1.0f;++total;}}
 }
}
