#include <metal_stdlib>
using namespace metal;
#include "soft_word32.metalh"
#pragma clang fp contract(off)
#define P0 F(100000.0f)
#define RCP F(0.2857142857142857f)
#define G F(9.81f)
struct Resources {
device const F* thb [[id(0)]];
device const F* thp [[id(1)]];
device const F* alt [[id(2)]];
device const F* pressure [[id(3)]];
device const F* phb [[id(4)]];
device const F* php [[id(5)]];
device F* t [[id(6)]];
device F* rho [[id(7)]];
device F* pii [[id(8)]];
device F* dz [[id(9)]];
device F* zi [[id(10)]];
device F* saved [[id(11)]];
};
struct Params {
int nz;
int nc;
int thb3d;
int phb3d;
};
kernel void strict_regional_wsm6_prep(constant Resources &r [[buffer(0)]], constant Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
device const F* thb=r.thb;
device const F* thp=r.thp;
device const F* alt=r.alt;
device const F* pressure=r.pressure;
device const F* phb=r.phb;
device const F* php=r.php;
device F* t=r.t;
device F* rho=r.rho;
device F* pii=r.pii;
device F* dz=r.dz;
device F* zi=r.zi;
device F* saved=r.saved;
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
  dz[idx]=__fsub_rn(hi,lo);
 }

}
