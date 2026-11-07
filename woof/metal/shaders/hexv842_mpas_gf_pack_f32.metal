#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
#include "hexv842_mpas_gf_pack_f32_bindings.metalh"
kernel void hexv842_mpas_gf_pack_f32(constant Hexv842MpasGfPackF32Resources &r [[buffer(0)]], constant Hexv842MpasGfPackF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(p.n*p.nz))return;
 int c=int(gid)%p.n,k=int(gid)/p.n, dst=(c*15)*p.nz+k;
 r.levels[dst+0*p.nz]=r.u[gid];
 r.levels[dst+1*p.nz]=r.v[gid];
 r.levels[dst+2*p.nz]=r.w[gid];
 r.levels[dst+3*p.nz]=r.t[gid];
 r.levels[dst+4*p.nz]=r.q[gid];
 r.levels[dst+5*p.nz]=r.p[gid];
 r.levels[dst+6*p.nz]=r.pi[gid];
 r.levels[dst+7*p.nz]=r.rho[gid];
 r.levels[dst+8*p.nz]=r.dz8w[gid];
 r.levels[dst+9*p.nz]=r.p8w[gid];
 r.levels[dst+10*p.nz]=r.rthdynten[gid];
 r.levels[dst+11*p.nz]=r.rqvdynten[gid];
 r.levels[dst+12*p.nz]=as_type<float>(soft32_add(as_type<uint>(r.rthratenlw[gid]),as_type<uint>(r.rthratensw[gid])));
 r.levels[dst+13*p.nz]=r.rthblten[gid];
 r.levels[dst+14*p.nz]=r.rqvblten[gid];
 if(k==0){
  r.surface[8*c+0]=r.ht[c];
  r.surface[8*c+1]=r.hfx[c];
  r.surface[8*c+2]=r.qfx[c];
  r.surface[8*c+3]=r.xland[c];
  r.surface[8*c+4]=r.gsw[c];
  uint pw=soft_gfk_pow_bits(as_type<uint>(r.meshDensity[c]),0x3e800000u);
  r.surface[8*c+5]=as_type<float>(soft32_div(as_type<uint>(p.len_disp),pw));
  r.surface[8*c+6]=r.areaCell[c];r.surface[8*c+7]=p.dt;
 }
}
