#include <metal_stdlib>
using namespace metal;
#include "soft_word32.metalh"
#pragma clang fp contract(off)
#define P0 F(100000.0f)
#define RCP F(0.2857142857142857f)
#define G F(9.81f)
#include "strict_regional_kessler_finish_bindings.metalh"
kernel void strict_regional_kessler_finish(constant StrictRegionalKesslerFinishResources &r [[buffer(0)]], constant StrictRegionalKesslerFinishParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
device F* thp=(device F*)r.thp;
device const F* t=(device const F*)r.t;
device F* saved=(device F*)r.saved;
int n=p.n;
int no_heating=p.no_heating;
F dt=F(p.dt);
F limit=F(p.limit);

 int i=int(gid);if(i>=n)return;
 if(no_heating){saved[i]=0.0f;return;}
 F v=__fsub_rn(t[i],saved[i]);
 // CuPy minimum/maximum propagate NaNs rather than CUDA fminf/fmaxf.
 if(!isnan(v)){if(v>limit)v=limit;if(v< -limit)v=-limit;}
 thp[i]=__fadd_rn(thp[i],v);saved[i]=__fdiv_rn(v,dt);

}
