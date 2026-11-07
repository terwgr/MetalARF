from pathlib import Path
import json,re,sys
root=Path.cwd();sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
def body(name):
 m=re.search(r'__device__ __noinline__ void '+name+r'\s*\(.*?\)\s*\{',source,re.S);assert m
 j=m.end();d=1
 while d:
  if source[j]=='{':d+=1
  elif source[j]=='}':d-=1
  j+=1
 return source[m.end():j-1]
def clean(s):
 s=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',s)
 s=re.sub(r'\w+\.rebase\(\{kts\}\);\n','',s)
 s=re.sub(r'MynnArray<float, 1> \w+ = mynn_array<float, 1>\(arena, .*?;\n','',s)
 return s.replace('mynn_ipow<2>','mynn_square')
full=body('f_mym_length');sw=full.index('switch (bl_mynn_mixlength)');start=full.index('case 1: {',sw)+len('case 1: {');end=full.index('break; }',start)
length=clean(full[:sw]+full[start:end]);boulac=clean(body('f_boulac_length'))
for a,b in {
 '((qtke(iz)) < (zup))':'soft32_less(as_type<uint>(qtke(iz)),as_type<uint>(zup))',
 '((qtke(iz)) >= (zup_inf))':'soft32_greater_equal(as_type<uint>(qtke(iz)),as_type<uint>(zup_inf))',
 '((qtke(iz)) < (zdo))':'soft32_less(as_type<uint>(qtke(iz)),as_type<uint>(zdo))',
 '((qtke(iz)) >= (zdo_sup))':'soft32_greater_equal(as_type<uint>(qtke(iz)),as_type<uint>(zdo_sup))',
 '((bbb) != (0.f))':'!soft32_equal(as_type<uint>(bbb),0u)',
 '((theta(izz)) != (theta(iz)))':'!soft32_equal(as_type<uint>(theta(izz)),as_type<uint>(theta(iz)))',
}.items():
 assert a in boulac,a;boulac=boulac.replace(a,b)
# BouLac's theta refers to the staggered thetaw array; its outputs are elblmin/avg.
boulac=re.sub(r'\btheta\(', 'thetaw(',boulac).replace('lb1(', 'elblmin(').replace('lb2(', 'elblavg(')
call='f_boulac_length(kts, kte, zw, dz, qtke, thetaw, elblmin, elblavg);';assert length.count(call)==1
length=length.replace(call,'{\n'+boulac+'\n}')
for a,b in {
 '((zwk) <= (mpas_add(zi2, h1)))':'soft32_less_equal(as_type<uint>(zwk),as_type<uint>(mpas_add(zi2,h1)))',
 '((dtv(k)) > (0.0f))':'soft32_greater(as_type<uint>(dtv(k)),0u)',
 '((rmo) > (0.0f))':'soft32_greater(as_type<uint>(rmo),0u)',
}.items():
 assert a in length,a;length=length.replace(a,b)
assert 'Mynn' not in length and 'f_boulac' not in length
name='hexv842_mpas_bl_mynn_length1_f32';typ='Hexv842MpasBlMynnLength1F32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_length2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_length2_f32_bindings.metalh',name+'_bindings.metalh').replace('#include "hexv842_mpas_bl_mynn_length2_validate_f32_bindings.metalh"','')
fields=('dz','u1','v1','qke','dtv','theta','edmf_w1','edmf_a1','qkw','el')
scratch=('qtke','thetaw','elblmin','elblavg','dlu','dld')
macros=''.join(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]\n' for n in fields)
macros+=''.join(f'#define {n}(i) r.scratch[(ulong(col)*{len(scratch)}+{j})*nl+((i)-1)]\n' for j,n in enumerate(scratch))
macros+='#define zw(i) r.zw[ulong(col)*(nl+1)+((i)-1)]\n'
undef=''.join(f'#undef {n}\n' for n in fields+scratch+('zw',))
text=header+f'\nkernel void {name}(constant {typ}Resources &r [[buffer(0)]],constant {typ}Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{\nif(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float grav=cfg.gravity,karman=cfg.karman,gtr=mynn_div(grav,300.f);float zi=r.zi[col],rmo=r.rmo[col],fltv=r.fltv[col],psig_bl=r.psig_bl[col];constexpr float qmin=0.f,zmax=1.f,onethird=1.f/3.f;\n'+macros+length+'\n'+undef+'}\n'
resources=[[n,'float','read'] for n in fields[:8]+('zw','zi','rmo','fltv','psig_bl')]+[[n,'float','read_write'] for n in ('qkw','el','scratch')]
params=[['nc','int32'],['nl','int32'],['gravity','float32'],['karman','float32']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_length','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
vname='hexv842_mpas_bl_mynn_length1_validate_f32';vtyp='Hexv842MpasBlMynnLength1ValidateF32'
text=text.replace('#include "'+name+'_bindings.metalh"', '#include "'+name+'_bindings.metalh"\n#include "'+vname+'_bindings.metalh"')
text+=f"""\nkernel void {vname}(constant {vtyp}Resources &r [[buffer(0)]],constant {vtyp}Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{
if(col>=uint(cfg.nc))return;int nl=cfg.nl;bool bad=nl<2,stop=false;float zi=r.zi[col];
uint zbits=as_type<uint>(zi),rmobits=as_type<uint>(r.rmo[col]),fbits=as_type<uint>(r.fltv[col]),pbits=as_type<uint>(r.psig_bl[col]);
bad|=(zbits&0x7fffffffu)>=0x7f800000u || soft32_less(zbits,0u) || (rmobits&0x7fffffffu)>=0x7f800000u || (fbits&0x7fffffffu)>=0x7f800000u || (pbits&0x7fffffffu)>=0x7f800000u;
float zi2=mynn_max(zi,300.f),h1=mynn_min(mynn_max(mynn_mul(.3f,zi2),300.f),600.f),threshold=mynn_add(zi2,h1);
for(int k=0;k<=nl;k++){{uint z=as_type<uint>(r.zw[ulong(col)*(nl+1)+k]);bad|=(z&0x7fffffffu)>=0x7f800000u;if(k>=1&&soft32_greater(z,as_type<uint>(threshold)))stop=true;
if(k<nl){{ulong j=ulong(col)*nl+k;uint d=as_type<uint>(r.dz[j]);bad|=d<0x00800000u || d>=0x7f800000u;
"""+''.join(f'bad|=(as_type<uint>(r.{n}[j])&0x7fffffffu)>=0x7f800000u;\n' for n in fields[1:8])+"""}}
r.invalid[col]=int(bad||!stop);
}
"""
vr=[[n,'float','read'] for n in fields[:8]+('zw','zi','rmo','fltv','psig_bl')]+[['invalid','int32','read_write']]
spec['kernels'].append({'name':vname,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_mym_length','resources':vr,'shapes':{x[0]:[['any_1d']] for x in vr},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']})
start=text.index('\nkernel void '+name+'(')
end=text.index('\nkernel void '+vname+'(',start)
text=(text[:start]+text[end:]).replace('#include "'+name+'_bindings.metalh"\n','')
kernel=spec['kernels'][1];kernel['module']=vname;kernel['source']=f'woof/metal/shaders/{vname}.metal'
(out/(vname+'.metal')).write_text(text)
(out/(vname+'.json')).write_text(json.dumps({'kernels':[kernel]},indent=2)+'\n')
print('generated',name)
