#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
#pragma clang fp contract(off)
#define A(a,b) tc_add((a),(b))
#define S(a,b) tc_sub((a),(b))
#define M(a,b) tc_mul((a),(b))
#define D(a,b) tc_div((a),(b))
#define MAX(a,b) tc_max((a),(b))
#define MIN(a,b) tc_min((a),(b))
#include "hexv842_mpas_kessler_precip_f32_bindings.metalh"
kernel void hexv842_mpas_kessler_precip_f32(constant Hexv842MpasKesslerPrecipF32Resources &r [[buffer(0)]],constant Hexv842MpasKesslerPrecipF32Params &params [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
device F* rho=(device F*)r.rho;
device F* qvp=(device F*)r.qvp;
device F* dz=(device F*)r.dz;
device F* rain_local=(device F*)r.rain_local;
device F* rainnc=(device F*)r.rainnc;
device F* rainncv=(device F*)r.rainncv;
device F* precipw=(device F*)r.precipw;
device int* i_rainnc=(device int*)r.i_rainnc;
F bucket=F(params.bucket);
int accumulate=params.accumulate;
int nz=params.nz;
int nc=params.nc;
int ns=params.ns;
int correct_wb3=params.correct_wb3;

 int c=int(gid);
 if(c>=ns) return;
 precipw[c]=0.0f;
 for(int k=0;k<nz;++k) {
  int p=k*nc+c;
  F rho_a=correct_wb3 ? rho[p] : D(rho[p],A(1.0f,qvp[p]));
  precipw[c]=A(precipw[c],M(M(qvp[p],rho_a),dz[p]));
 }
 rainncv[c]=rain_local[c]; rainnc[c]=A(rainnc[c],rainncv[c]);
 if(accumulate && bucket>0.0f && rainnc[c]>bucket) {
  i_rainnc[c]+=1; rainnc[c]=S(rainnc[c],bucket);
 }

}
