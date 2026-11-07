#include <metal_stdlib>
using namespace metal;
#include "soft_word32.metalh"
#pragma clang fp contract(off)
#define P0 F(100000.0f)
#define RCP F(0.2857142857142857f)
#define G F(9.81f)
#include "strict_regional_kessler_prep_bindings.metalh"
kernel void strict_regional_kessler_prep(constant StrictRegionalKesslerPrepResources &r [[buffer(0)]], constant StrictRegionalKesslerPrepParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
device const F* thb=(device const F*)r.thb;
device const F* thp=(device const F*)r.thp;
device const F* alt=(device const F*)r.alt;
device const F* pressure=(device const F*)r.pressure;
device const F* phb=(device const F*)r.phb;
device const F* php=(device const F*)r.php;
device F* t=(device F*)r.t;
device F* rho=(device F*)r.rho;
device F* pii=(device F*)r.pii;
device F* z=(device F*)r.z;
device F* dz=(device F*)r.dz;
device F* zi=(device F*)r.zi;
device F* saved=(device F*)r.saved;
int nz=p.nz;
int nc=p.nc;
int thb3d=p.thb3d;
int phb3d=p.phb3d;

 int col=int(gid);
 if(col>=nc)return;
 for(int k=0;k<nz;k++){
  int idx=k*nc+col;F theta=__fadd_rn(thb[thb3d?idx:k],thp[idx]);
  t[idx]=theta;saved[idx]=theta;rho[idx]=__fdiv_rn(1.0f,alt[idx]);
  F ratio=__fdiv_rn(pressure[idx],P0);pii[idx]=gfk_pow(ratio,RCP);
 }
 for(int k=0;k<nz+1;k++){
  int idx=k*nc+col;zi[idx]=__fdiv_rn(__fadd_rn(phb[phb3d?idx:k],php[idx]),G);
 }
 for(int k=0;k<nz;k++){
  int idx=k*nc+col;F lo=zi[idx],hi=zi[idx+nc];
  z[idx]=__fmul_rn(0.5f,__fadd_rn(lo,hi));dz[idx]=__fsub_rn(hi,lo);
 }

}
