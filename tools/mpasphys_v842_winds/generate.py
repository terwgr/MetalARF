"""Generate source-ordered native MPAS wind reconstruction kernels."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
name='hexv842_mpas_wind_f32'
shader='''#include <metal_stdlib>
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
'''
(root/f'woof/metal/shaders/{name}.metal').write_text(shader)
resources=[[n,t,a] for n,t,a in [('velocity','float','read'),('edges','int32','read'),('count','int32','read'),('coeff','float','read'),('lat','float','read'),('lon','float','read'),('sealed_edges','int32','read'),('sealed_count','int32','read'),('sealed_coeff','float','read'),('sealed_lat','float','read'),('sealed_lon','float','read'),('x','float','read_write'),('y','float','read_write'),('z','float','read_write'),('u','float','read_write'),('v','float','read_write'),('invalid','int32','read_write')]]
def spec(k,rs,ps):return {'name':k,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/cuda_physics_prep_v841.py','cuda_symbol':'reconstruct_winds_v841_f32','resources':rs,'shapes':{r[0]:[['any_1d']] for r in rs},'parameters':ps,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}
rows=[spec(name,resources,[[n,'int32'] for n in ('nlev','ncells','nedges','maxedges')]),spec('hexv842_mpas_wind_donor_f32',[[n,'float','read_write'] for n in ('u','v')],[[n,'int32'] for n in ('nlev','ncells','garbage','donor')])]
rows[1]['cuda_source']='woof/hex/mpasphys/runtime_v842.py'
rows[1]['cuda_symbol']='_resident_winds'
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels':rows},indent=2)+'\n')
