#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexv842_mpas_wind_f32_bindings.metalh"
#include "hexv842_mpas_wind_donor_f32_bindings.metalh"
kernel void hexv842_mpas_wind_f32(constant Hexv842MpasWindF32Resources &r [[buffer(0)]],constant Hexv842MpasWindF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
 if(c>=uint(p.ncells))return;
 bool changed=r.count[c]!=r.sealed_count[c] || as_type<uint>(r.lat[c])!=as_type<uint>(r.sealed_lat[c]) || as_type<uint>(r.lon[c])!=as_type<uint>(r.sealed_lon[c]);
 for(int slot=0;slot<p.maxedges;++slot){int e=int(c)*p.maxedges+slot, k=e*3;changed=changed || r.edges[e]!=r.sealed_edges[e] || as_type<uint>(r.coeff[k])!=as_type<uint>(r.sealed_coeff[k]) || as_type<uint>(r.coeff[k+1])!=as_type<uint>(r.sealed_coeff[k+1]) || as_type<uint>(r.coeff[k+2])!=as_type<uint>(r.sealed_coeff[k+2]);}
 if(changed){atomic_fetch_or_explicit((device atomic_int*)r.invalid,1,memory_order_relaxed);return;}
 for(int k=0;k<p.nlev;++k){int i=k*p.ncells+int(c);r.x[i]=0.f;r.y[i]=0.f;r.z[i]=0.f;}
 int count=r.count[c];if(count<0 || count>p.maxedges){atomic_fetch_or_explicit((device atomic_int*)r.invalid,1,memory_order_relaxed);return;}
 for(int slot=0;slot<count;++slot){int ei=int(c)*p.maxedges+slot,edge=r.edges[ei];if(edge<0 || edge>=p.nedges){atomic_fetch_or_explicit((device atomic_int*)r.invalid,1,memory_order_relaxed);continue;}
  int ci=ei*3;uint cx=as_type<uint>(r.coeff[ci]),cy=as_type<uint>(r.coeff[ci+1]),cz=as_type<uint>(r.coeff[ci+2]);
  if(!soft32_is_finite(cx)||!soft32_is_finite(cy)||!soft32_is_finite(cz))atomic_fetch_or_explicit((device atomic_int*)r.invalid,1,memory_order_relaxed);
  for(int k=0;k<p.nlev;++k){int i=k*p.ncells+int(c);uint value=as_type<uint>(r.velocity[k*p.nedges+edge]);if(!soft32_is_finite(value))atomic_fetch_or_explicit((device atomic_int*)r.invalid,1,memory_order_relaxed);
   r.x[i]=as_type<float>(soft32_add(as_type<uint>(r.x[i]),soft32_mul(cx,value)));
   r.y[i]=as_type<float>(soft32_add(as_type<uint>(r.y[i]),soft32_mul(cy,value)));
   r.z[i]=as_type<float>(soft32_add(as_type<uint>(r.z[i]),soft32_mul(cz,value)));}}
 uint clat=soft_glibc_solar_cosf(as_type<uint>(r.lat[c])),slat=soft_glibc_solar_sinf(as_type<uint>(r.lat[c])),clon=soft_glibc_solar_cosf(as_type<uint>(r.lon[c])),slon=soft_glibc_solar_sinf(as_type<uint>(r.lon[c]));
 if(!soft32_is_finite(clat)||!soft32_is_finite(slat)||!soft32_is_finite(clon)||!soft32_is_finite(slon))atomic_fetch_or_explicit((device atomic_int*)r.invalid,1,memory_order_relaxed);
 for(int k=0;k<p.nlev;++k){int i=k*p.ncells+int(c);uint x=as_type<uint>(r.x[i]),y=as_type<uint>(r.y[i]),z=as_type<uint>(r.z[i]);uint horizontal=soft32_add(soft32_mul(x,clon),soft32_mul(y,slon));
  r.u[i]=as_type<float>(soft32_add(soft32_mul(soft32_sub(0u,x),slon),soft32_mul(y,clon)));
  r.v[i]=as_type<float>(soft32_add(soft32_mul(soft32_sub(0u,horizontal),slat),soft32_mul(z,clat)));}
}
kernel void hexv842_mpas_wind_donor_f32(constant Hexv842MpasWindDonorF32Resources &r [[buffer(0)]],constant Hexv842MpasWindDonorF32Params &p [[buffer(1)]],uint k [[thread_position_in_grid]]) {
 if(k>=uint(p.nlev))return;int row=int(k)*p.ncells;r.u[row+p.garbage]=r.u[row+p.donor];r.v[row+p.garbage]=r.v[row+p.donor];
}
