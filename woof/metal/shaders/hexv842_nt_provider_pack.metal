#include "hexv842_nt_provider_pack_bindings.metalh"
#include <metal_stdlib>
#include "soft_binary32.metalh"
using namespace metal;

// Inputs are the existing nt_prefix and nt_prefix_math output layouts.
// extra levels are bottom-up u, v, qc, qi. Surface: hfx, qfx, xland.
// All output slabs have an extra row on either side; caller zero-fills them.
kernel void hexv842_nt_provider_pack(constant Hexv842NtProviderPackResources &r [[buffer(0)]], constant Hexv842NtProviderPackParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    device const float *prefix=r.prefix;
    device const float *interfaces=r.interfaces;
    device const float *math=r.math;
    device const float *extra=r.extra;
    device const float *surface=r.surface;
    device uint *env=r.env;
    device float *saved=r.saved;

 if(i>=uint(p.ncol)) return;
 int n=p.ncol,nz=p.nz,s=n*nz,si=n*(nz+1),st=n*(nz+2);
 for(int k=0;k<nz;k++) {
  int a=k*n+i, rev=(nz-1-k)*n+i, o=(k+1)*n+i;
  // pten,pqen,pqsen,puen,pven,pverv,pgeo,paph,pgeoh,pap,ptte,pqte
  env[0*st+o]=as_type<uint>(prefix[3*s+a]);
  env[1*st+o]=as_type<uint>(prefix[6*s+a]);
  env[2*st+o]=as_type<uint>(math[3*n+a]);
  env[3*st+o]=as_type<uint>(extra[rev]);
  env[4*st+o]=as_type<uint>(extra[s+rev]);
  env[5*st+o]=as_type<uint>(prefix[5*s+a]);
  env[6*st+o]=as_type<uint>(prefix[7*s+a]);
  env[7*st+o]=as_type<uint>(interfaces[a]);
  env[8*st+o]=as_type<uint>(interfaces[2*si+a]);
  env[9*st+o]=as_type<uint>(prefix[2*s+a]);
  env[10*st+o]=as_type<uint>(prefix[8*s+a]);
  env[11*st+o]=as_type<uint>(prefix[9*s+a]);
  // Preserve top-down qcf, qif, uf and vf.
  saved[0*s+a]=extra[2*s+rev]; saved[1*s+a]=extra[3*s+rev];
  saved[2*s+a]=extra[rev]; saved[3*s+a]=extra[s+rev];
 }
 int last=(nz+1)*n+i;
 env[7*st+last]=as_type<uint>(interfaces[nz*n+i]);
 env[8*st+last]=as_type<uint>(interfaces[2*si+nz*n+i]);
 env[12*st+i]=as_type<uint>(surface[i]);
 env[12*st+n+i]=as_type<uint>(surface[n+i]);
 float land=as_type<float>(soft32_sub(as_type<uint>(surface[2*n+i]),as_type<uint>(2.0f)));
 env[12*st+2*n+i]=as_type<uint>(int(fabs(land)));
}
