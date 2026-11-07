#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define AD(a,b) mpas_add(a,b)
#define SU(a,b) mpas_sub(a,b)
#define DV(a,b) cu_div(a,b)
#include "hexv842_mpas_noah_to_f32_bindings.metalh"
kernel void hexv842_mpas_noah_to_f32(constant Hexv842MpasNoahToF32Resources &r [[buffer(0)]], constant Hexv842MpasNoahToF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
int n=p.n;
if(c>=uint(n) || r.status[c])return;
const device float* x=r.floats+c*79;
r.skintemp[c]=x[8];
r.hfx[c]=x[9];
r.qfx[c]=x[10];
r.lh[c]=x[11];
r.grdflx[c]=x[12];
r.smstav[c]=x[16];
r.smstot[c]=x[17];
r.sfcrunoff[c]=x[18];
r.udrunoff[c]=x[19];
r.sfc_albedo[c]=x[21];
r.sfc_albbck[c]=x[22];
r.znt[c]=x[23];
r.z0[c]=x[24];
r.sfc_emiss[c]=x[28];
r.snowc[c]=x[30];
r.qsfc[c]=x[31];
r.smois[0*n+c]=x[38];
r.smois[1*n+c]=x[39];
r.smois[2*n+c]=x[40];
r.smois[3*n+c]=x[41];
r.tslb[0*n+c]=x[42];
r.tslb[1*n+c]=x[43];
r.tslb[2*n+c]=x[44];
r.tslb[3*n+c]=x[45];
r.snow[c]=x[46];
r.canwat[c]=x[47];
r.chs[c]=x[48];
r.chs2[c]=x[49];
r.cqs2[c]=x[50];
r.chklowq[c]=x[52];
r.lai[c]=x[53];
r.sh2o[0*n+c]=x[55];
r.sh2o[1*n+c]=x[56];
r.sh2o[2*n+c]=x[57];
r.sh2o[3*n+c]=x[58];
r.snowh[c]=x[59];
r.snoalb[c]=x[60];
r.snotime[c]=x[63];
r.acsnom[c]=x[64];
r.acsnow[c]=x[65];
r.snopcx[c]=x[66];
r.potevp[c]=x[67];
r.smcrel[0*n+c]=x[68];
r.smcrel[1*n+c]=x[69];
r.smcrel[2*n+c]=x[70];
r.smcrel[3*n+c]=x[71];
r.noahres[c]=x[74];
}
