#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define AD(a,b) mpas_add(a,b)
#define SU(a,b) mpas_sub(a,b)
#define DV(a,b) cu_div(a,b)
#include "hexv842_mpas_noah_from_f32_bindings.metalh"
kernel void hexv842_mpas_noah_from_f32(constant Hexv842MpasNoahFromF32Resources &r [[buffer(0)]], constant Hexv842MpasNoahFromF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
int n=p.n;
if(c>=uint(n))return;
device float* x=r.floats+c*79;device int* ix=r.integers+c*7;
float rain=0.0f;if(p.micro)rain=AD(rain,r.rainncv[c]);if(p.convection)rain=AD(rain,r.raincv[c]);
x[0]=r.dz8w[0*n+c];
x[1]=r.dz8w[1*n+c];
x[2]=r.qv[0*n+c];
x[3]=r.qv[1*n+c];
x[4]=r.pres2_hyd[0*n+c];
x[5]=r.pres2_hyd[1*n+c];
x[6]=r.t[0*n+c];
x[7]=r.t[1*n+c];
x[8]=r.skintemp[c];
x[9]=r.hfx[c];
x[10]=r.qfx[c];
x[11]=r.lh[c];
x[12]=r.grdflx[c];
x[13]=r.qgh[c];
x[14]=DV(r.gsw[c],SU(1.0f,r.sfc_albedo[c]));
x[15]=r.glw[c];
x[16]=r.smstav[c];
x[17]=r.smstot[c];
x[18]=r.sfcrunoff[c];
x[19]=r.udrunoff[c];
x[20]=r.vegfra[c];
x[21]=r.sfc_albedo[c];
x[22]=r.sfc_albbck[c];
x[23]=r.znt[c];
x[24]=r.z0[c];
x[25]=r.tmn[c];
x[26]=r.xland[c];
x[27]=r.xice[c];
x[28]=r.sfc_emiss[c];
x[29]=r.sfc_emibck[c];
x[30]=r.snowc[c];
x[31]=r.qsfc[c];
x[32]=rain;
x[33]=p.dt;
x[34]=r.dzs[0*n];
x[35]=r.dzs[1*n];
x[36]=r.dzs[2*n];
x[37]=r.dzs[3*n];
x[38]=r.smois[0*n+c];
x[39]=r.smois[1*n+c];
x[40]=r.smois[2*n+c];
x[41]=r.smois[3*n+c];
x[42]=r.tslb[0*n+c];
x[43]=r.tslb[1*n+c];
x[44]=r.tslb[2*n+c];
x[45]=r.tslb[3*n+c];
x[46]=r.snow[c];
x[47]=r.canwat[c];
x[48]=r.chs[c];
x[49]=r.chs2[c];
x[50]=r.cqs2[c];
x[51]=p.micro?r.sr[c]:0.0f;
x[52]=r.chklowq[c];
x[53]=r.lai[c];
x[54]=0.0f;
x[55]=r.sh2o[0*n+c];
x[56]=r.sh2o[1*n+c];
x[57]=r.sh2o[2*n+c];
x[58]=r.sh2o[3*n+c];
x[59]=r.snowh[c];
x[60]=r.snoalb[c];
x[61]=r.shdmin[c];
x[62]=r.shdmax[c];
x[63]=r.snotime[c];
x[64]=r.acsnom[c];
x[65]=r.acsnow[c];
x[66]=r.snopcx[c];
x[67]=r.potevp[c];
x[68]=r.smcrel[0*n+c];
x[69]=r.smcrel[1*n+c];
x[70]=r.smcrel[2*n+c];
x[71]=r.smcrel[3*n+c];
x[72]=p.xice_threshold;
x[73]=r.br[c];
x[74]=r.noahres[c];
x[75]=0.0f;
x[76]=0.0f;
x[77]=0.0f;
x[78]=0.0f;
ix[0]=r.ivgtyp[c];
ix[1]=r.isltyp[c];
ix[2]=p.isurban;
ix[3]=p.isice;
ix[4]=p.step;
ix[5]=p.usemonalb;
ix[6]=1;
r.locations[2*c]=int(c)+1;r.locations[2*c+1]=1;
}
