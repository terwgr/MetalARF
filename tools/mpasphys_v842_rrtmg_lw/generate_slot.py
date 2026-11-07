"""Generate MPAS level-major LW slot packing and publication."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
inputs=('p3d','t3d','dz','pi','qv','qc','qi','qs','cldfra','re_cloud','re_ice','re_snow','p8w','t8w','tsk','emiss','xland','xice','snow','xlat','o3clim')
outputs=('glw','olr','lwcf','lwupt','lwuptc','lwdnt','lwdntc','lwupb','lwupbc','lwdnb','lwdnbc')
def emit(suffix,resources,params,body):
    name='hexv842_mpas_rrtmg_lw_'+suffix+'_f32';typ='Hexv842MpasRrtmgLw'+suffix.title()+'F32'
    code='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
'''+f'#include "{name}_bindings.metalh"\n'
    code+=f'kernel void {name}(constant {typ}Resources &r [[buffer(0)]],constant {typ}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{\n{body}\n}}\n'
    (root/f'woof/metal/shaders/{name}.metal').write_text(code)
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/rrtmg_lw_v842.py','cuda_symbol':{'slotcounts':'mpas_layer_counts','slotpack':'radiation_lw_from_mpas','slotscatter':'radiation_lw_slot','ozstart':'mpas_vinterp_start','ozlevel':'mpas_vinterp_level'}[suffix],'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':[[n,'int32'] for n in params],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
    (root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
emit('slotcounts',[['p8w','float','read'],['counts','int32','read_write']],('nc','nm'),'''if(gid>=uint(p.nc))return;
float top=cu_div(r.p8w[ulong(p.nm)*p.nc+gid],100.0f);
r.counts[gid]=p.nm+max(int(mpas_add(cu_div(top,4.0f),0.5f)),1);''')
resources=[['indices','int32','read']]+[[n,'float','read'] for n in inputs]+[[n,'float','read_write'] for n in ('model','interfaces','surface','ozone')]
# MODEL_FIELDS put the Exner field after cloud radii.
order=('p3d','t3d','dz','qv','qc','qi','qs','cldfra','re_cloud','re_ice','re_snow','pi')
assign='\n'.join(f'r.model[base+{j}*p.nm+k]='+('p.radii?'+f'r.{n}[src]:0.0f' if n.startswith('re_') else f'r.{n}[src]')+';' for j,n in enumerate(order))
surf='\n'.join(f'r.surface[ulong(row)*6+{j}]=r.{n}[col];' for j,n in enumerate(('tsk','emiss','xland','xice','snow','xlat')))
body='''if(ulong(gid)>=ulong(p.ng)*p.nm)return;
uint row=gid/p.nm,k=gid%p.nm,col=uint(r.indices[row]);ulong src=ulong(k)*p.nc+col,base=ulong(row)*12*p.nm;
'''+assign+'''\nulong out=ulong(row)*2*(p.nm+1)+k;
r.interfaces[out]=r.p8w[src];r.interfaces[out+p.nm+1]=r.t8w[src];
if(k==0){
 r.interfaces[ulong(row)*2*(p.nm+1)+p.nm]=r.p8w[ulong(p.nm)*p.nc+col];
 r.interfaces[ulong(row)*2*(p.nm+1)+2*p.nm+1]=r.t8w[ulong(p.nm)*p.nc+col];
'''+surf+'''\nfor(int j=0;j<p.nz;++j)r.ozone[ulong(row)*p.nz+j]=r.o3clim[ulong(j)*p.nc+col];
}'''
emit('slotpack',resources,('nc','ng','nm','nz','radii'),body)
resources=[['indices','int32','read']]+[[n,'float','read'] for n in ('tendency','radii','flux')]+[[n,'float','read_write'] for n in ('tend_out','re_cloud_out','re_ice_out','re_snow_out',*outputs)]
scatter='\n'.join(f'r.{n}[col]=r.flux[ulong(row)*11+{j}];' for j,n in enumerate(outputs))
body='''if(ulong(gid)>=ulong(p.ng)*p.nm)return;
uint row=gid/p.nm,k=gid%p.nm,col=uint(r.indices[row]);ulong dst=ulong(k)*p.nc+col;
r.tend_out[dst]=r.tendency[ulong(row)*p.nm+k];
if(p.write_radii){r.re_cloud_out[dst]=p.radii?r.radii[(ulong(row)*3)*p.nm+k]:0.0f;r.re_ice_out[dst]=p.radii?r.radii[(ulong(row)*3+1)*p.nm+k]:0.0f;r.re_snow_out[dst]=p.radii?r.radii[(ulong(row)*3+2)*p.nm+k]:0.0f;}
if(k==0){\n'''+scatter+'\n}'
emit('slotscatter',resources,('nc','ng','nm','write_radii','radii'),body)
emit('ozstart',[['upper','int32','read'],['start','int32','read_write']],('nc','nz'),'''if(gid>0)return;
int kk=p.nz-1;for(int c=0;c<p.nc;++c)kk=min(kk,r.upper[c]);r.start[0]=kk;''')
emit('ozlevel',[['p3d','float','read'],['pin','float','read'],['ozone','float','read'],['start','int32','read'],['upper','int32','read_write'],['o3vmr','float','read_write']],('nc','nz','k'),'''if(gid>=uint(p.nc))return;
float pressure=cu_div(r.p3d[ulong(p.k)*p.nc+gid],100.0f);int u=r.upper[gid];
for(int kk=r.start[0];kk<p.nz-1;++kk){
 if(soft32_less(as_type<uint>(r.pin[kk]),as_type<uint>(pressure)) &&
    soft32_less_equal(as_type<uint>(pressure),as_type<uint>(r.pin[kk+1]))){u=kk;break;}
}
r.upper[gid]=u;
float value;
if(soft32_less(as_type<uint>(pressure),as_type<uint>(r.pin[0])))value=cu_div(mpas_mul(r.ozone[gid],pressure),r.pin[0]);
else if(soft32_greater(as_type<uint>(pressure),as_type<uint>(r.pin[p.nz-1])))value=r.ozone[ulong(p.nz-1)*p.nc+gid];
else {
 float dpu=mpas_sub(pressure,r.pin[u]),dpl=mpas_sub(r.pin[u+1],pressure);
 value=cu_div(mpas_add(mpas_mul(r.ozone[ulong(u)*p.nc+gid],dpl),mpas_mul(r.ozone[ulong(u+1)*p.nc+gid],dpu)),mpas_add(dpl,dpu));
}
r.o3vmr[ulong(p.k)*p.nc+gid]=value;''')
