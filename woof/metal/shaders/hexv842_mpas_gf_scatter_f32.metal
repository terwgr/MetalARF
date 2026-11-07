#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_gf_scatter_f32_bindings.metalh"
kernel void hexv842_mpas_gf_scatter_f32(constant Hexv842MpasGfScatterF32Resources &r [[buffer(0)]],constant Hexv842MpasGfScatterF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]){
 if(gid>=uint(p.n*p.nz))return; int c=int(gid)%p.n,k=int(gid)/p.n;
 r.rthcuten[gid]=r.levels[(c*6+0)*p.nz+k];
 r.rqvcuten[gid]=r.levels[(c*6+1)*p.nz+k];
 r.rqccuten[gid]=r.levels[(c*6+2)*p.nz+k];
 r.rqicuten[gid]=r.levels[(c*6+3)*p.nz+k];
 r.qc_cu[gid]=r.levels[(c*6+4)*p.nz+k];
 r.qi_cu[gid]=r.levels[(c*6+5)*p.nz+k];
 if(k==0){
  r.cutop[c]=r.surface[c*6+0];
  r.cubot[c]=r.surface[c*6+1];
  r.xmb_total[c]=r.surface[c*6+2];
  r.xmb_shallow[c]=r.surface[c*6+3];
  r.raincv[c]=r.surface[c*6+4];
  r.cuprec[c]=r.surface[c*6+5];
  r.k22_shallow[c]=r.integer[c*3+0];
  r.kbcon_shallow[c]=r.integer[c*3+1];
  r.ktop_shallow[c]=r.integer[c*3+2];
 }
}
