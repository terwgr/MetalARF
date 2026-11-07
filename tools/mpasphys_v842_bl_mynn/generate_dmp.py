"""Generate the native MYNN eight-plume mass-flux column (no chemistry)."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source

def extract(name,ret='void'):
    m=re.search(r'__device__ __noinline__ '+ret+' '+name+r'\s*\(.*?\)\s*\{',source,re.S);assert m
    j=m.end();d=1
    while d:
        if source[j]=='{':d+=1
        elif source[j]=='}':d-=1
        j+=1
    return m.group(),source[m.end():j-1]
def remove_block(s,needle):
    i=s.index(needle);o=s.index('{',i);j=o+1;d=1
    while d:
        if s[j]=='{':d+=1
        elif s[j]=='}':d-=1
        j+=1
    return s[:i]+s[j:]

sig,raw=extract('f_dmp_mf');profiles=re.findall(r'MynnArray<float, 1> (\w+)',sig)
# The native caller allocates the mass-flux carriers and zw with an extra interface.
interfaces={'zw'}|{n for n in profiles if n.startswith('s_aw')}
s=raw
while 'if (mix_chem)' in s:s=remove_block(s,'if (mix_chem)')
while 'if (((debug_mf) == (1)))' in s:s=remove_block(s,'if (((debug_mf) == (1)))')
s=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',s);s=re.sub(r'\w+\.rebase\(\{.*?\}\);\n','',s)
# Exclude the unexecuted chemistry storage; remaining scratch is native.
scratch1=re.findall(r'MynnArray<float, 1> (\w+) = mynn_array',s);scratch1.remove('chemn')
scratch2=re.findall(r'MynnArray<float, 2> (\w+) = mynn_array',s);scratch2.remove('edmf_chem')
scratchi=re.findall(r'MynnArray<int, 2> (\w+) = mynn_array',s)
s=re.sub(r'MynnArray<[^\n]+ mynn_array<[^\n]+\n','',s)
# Fortran array section copies; every 1D slice here spans 1..kte.
s=re.sub(r'for \(int mynn_element=0; mynn_element < \((\w+)\.slice\(MynnRange\{kts, kte\}\)\)\.size\(\); \+\+mynn_element\) \{ \1\.slice\(MynnRange\{kts, kte\}\)\.linear\(mynn_element\) = (\w+)\.slice\(MynnRange\{kts, kte\}\)\.linear\(mynn_element\); \}',r'for(int mynn_element=1;mynn_element<=nl;++mynn_element) { \1(mynn_element)=\2(mynn_element); }',s)
# Explicit first-index-fastest linear mapping of local plume arrays.
for name in scratch2:
    nlev='nl' if name in ('ent','entf') else '(nl+1)'
    s=s.replace(f'({name}).size()',f'(8*{nlev})')
    s=re.sub(r'\b'+name+r'\.linear\(mynn_element\)',f'{name}((mynn_element%{nlev})+1,(mynn_element/{nlev})+1)',s)
for name in profiles+scratch1:
    nlev='(nl+1)' if name in interfaces or name in ('s_aw2','envi_a','envi_w') else 'nl'
    s=s.replace(f'({name}).size()',nlev).replace(f'{name}.linear(mynn_element)',f'{name}(mynn_element+1)')
s=s.replace('mynn_ipow<2>','mynn_square')
s=s.replace('f_qsat_blend(tk(k), t0c, tice, p(k), {})','f_qsat_blend(tk(k), t0c, tice, p(k), 0)')
s=s.replace('f_condensation_edmf(qtn, thln, pk, zw(((k) + (1))), thvn, qcn);','f_condensation_edmf(qtn,thln,pk,zw(((k)+(1))),thvn,qcn,p1000mb,rcp,xlvcp,t0c,rvovrd);')
s=s.replace('maxqc = mynn_maxval(edmf_qc.slice(MynnRange{1, ktop}));','maxqc=edmf_qc(1); for(int l=2;l<=ktop;l++) maxqc=mynn_max(maxqc,edmf_qc(l));')
assert 'Mynn' not in s and '.linear(' not in s and '.size(' not in s and 'chem1(' not in s
# Convert each source floating comparison, retaining integer control comparisons.
float_names=set(re.findall(r'\bfloat\s+(\w+)',s))|set(profiles)|set(scratch1)|set(scratch2)|set(re.findall(r'\bfloat\s*&?\s*(\w+)',sig))
float_names|={'t0c','tice','p1000mb','rcp','xlvcp','rvovrd','r_d','r_v','ep_2','ep_3','cice','cliq','cpv','xls','xlv','tv0','karman','gravity'}
int_names=set(re.findall(r'\b(?:int|bool)\s+(\w+)',s))|{'kts','kte','kpbl','spp_pbl','ii','nchem'}
def is_float(expr):
    ids=set(re.findall(r'\b[a-zA-Z_]\w*\b',expr))
    return bool(ids&float_names) or bool(re.search(r'\bmpas_|\bmynn_(?:min|max|sqrt|pow|exp|tanh|atan|abs)|\d\.\d|\d(?:e|E)[+-]?\d',expr))
def match_close(text,i):
    depth=1;j=i+1
    while depth and j<len(text):
        if text[j]=='(':depth+=1
        elif text[j]==')':depth-=1
        j+=1
    return j-1 if not depth else -1
for _ in range(8):
    changes=[]
    for i,ch in enumerate(s):
        if ch!='(' or i+1>=len(s) or s[i+1]!='(':continue
        mid=match_close(s,i+1)
        if mid<0:continue
        m=re.match(r'\s*(<=|>=|==|!=|<|>)\s*\(',s[mid+1:])
        if not m:continue
        rhsopen=mid+1+m.end()-1;rhsend=match_close(s,rhsopen)
        if rhsend<0 or rhsend+1>=len(s) or s[rhsend+1]!=')':continue
        lhs=s[i+2:mid];rhs=s[rhsopen+1:rhsend];op=m.group(1)
        if not(is_float(lhs) or is_float(rhs)):continue
        fn={'<':'less','>':'greater','<=':'less_equal','>=':'greater_equal','==':'equal','!=':'equal'}[op]
        expr=f'soft32_{fn}(as_type<uint>(float({lhs})),as_type<uint>(float({rhs})))'
        if op=='!=':expr='(!'+expr+')'
        changes.append((i,rhsend+2,expr))
    # innermost nonoverlapping replacements
    chosen=[]
    for change in sorted(changes,key=lambda x:(x[1]-x[0],x[0])):
        if not any(change[0]<b and a<change[1] for a,b,_ in chosen):chosen.append(change)
    if not chosen:break
    for a,b,v in sorted(chosen,reverse=True):s=s[:a]+v+s[b:]
print('profiles',len(profiles),'interfaces',len(interfaces),'scratch1',len(scratch1),'scratch2',len(scratch2),'int',len(scratchi))
name='hexv842_mpas_bl_mynn_dmp_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_edmf_condensation_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_edmf_condensation_f32_bindings.metalh',name+'_bindings.metalh')
for suffix in ('pack0','pack1','pack2','scatter'):
    header+=f'#include "hexv842_mpas_bl_mynn_dmp_{suffix}_f32_bindings.metalh"\n'
header+='''\nstatic inline int mynn_min(int a,int b){return a<b?a:b;}\n'''
_,esat=extract('f_esat_blend','float');esat=esat.replace('MynnFrame mynn_frame(arena);\n','').replace('((t) >= (t0c))','soft32_greater_equal(as_type<uint>(t),as_type<uint>(t0c))').replace('((t) <= (tice))','soft32_less_equal(as_type<uint>(t),as_type<uint>(tice))')
header+='static inline float f_esat_blend(float t,float t0c,float tice){\n'+esat+'\n}\n'
_,cond=extract('f_condensation_edmf');cond=cond.replace('MynnFrame mynn_frame(arena);\n','').replace('f_qsat_blend(t, t0c, tice, p, {})','f_qsat_blend(t,t0c,tice,p,0)').replace('((mynn_abs(mpas_sub(qc, qcold))) < (diff))','soft32_less(as_type<uint>(mynn_abs(mpas_sub(qc,qcold))),as_type<uint>(diff))').replace('((zagl) < (100.f))','soft32_less(as_type<uint>(zagl),as_type<uint>(100.f))')
header+='static inline void f_condensation_edmf(float qt,float thl,float p,float zagl,thread float &thv,thread float &qc,float p1000mb,float rcp,float xlvcp,float t0c,float rvovrd){constexpr float tice=240.f,onethird=1.f/3.f;constexpr bool env_subs=false;\n'+cond+'\n}\n'
lanes=len(scratch1)+8*len(scratch2)
macros=''.join(f'#define {n}(i) r.packed[(ulong({j})*cfg.nc+col)*(nl+1)+((i)-1)]\n' for j,n in enumerate(profiles))
macros+=''.join(f'#define {n}(i) r.scratch[(ulong(col)*{lanes}+{j})*(nl+1)+((i)-1)]\n' for j,n in enumerate(scratch1))
macros+=''.join(f'#define {n}(i,j) r.scratch[(ulong(col)*{lanes}+{len(scratch1)+8*k}+((j)-1))*(nl+1)+((i)-1)]\n' for k,n in enumerate(scratch2))
macros+='#define enti(i,j) r.intscratch[(ulong(col)*8+((j)-1))*(nl+1)+((i)-1)]\n'
undef=''.join(f'#undef {n}\n' for n in profiles+scratch1+scratch2+scratchi)
columns=('ust','flt','fltv','flq','flqv','pblh','dx','landsea','ts','psig_shcu')
constants=('dt','p1000mb','rcp','xlvcp','t0c','rvovrd','r_v','ep_2','ep_3','cice','cliq','cpv','xls','xlv','cp','p608','gravity','ep1')
flags=('momentum_opt','tke_opt','scalar_opt')
text=header+f'\nkernel void {name}(constant Hexv842MpasBlMynnDmpF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDmpF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{\nif(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl,ii=int(col)+1,spp_pbl=0,kpbl=r.kpbl[col];constexpr float tice=240.f,onethird=1.f/3.f;constexpr bool env_subs=false;\n'+''.join('float '+n+'=r.'+n+'[col];\n' for n in columns)+''.join('float '+n+'=cfg.'+n+';\n' for n in constants)+''.join('bool '+n+'=cfg.'+n+'!=0;\n' for n in flags)+'float grav=gravity,gtr=mynn_div(grav,300.f),tv0=mynn_mul(ep1,300.f),maxwidth=r.maxwidth[col],maxmf=r.maxmf[col],ztop=r.ztop[col];int ktop=r.ktop[col];\n'+macros+s+'\nr.maxwidth[col]=maxwidth;r.maxmf[col]=maxmf;r.ztop[col]=ztop;r.ktop[col]=ktop;\n'+undef+'}\n'
resources=[['packed','float','read_write'],['scratch','float','read_write'],['intscratch','int32','read_write']]+[[n,'float','read'] for n in columns]+[['kpbl','int32','read']]+[[n,'float','read_write'] for n in ('maxwidth','maxmf','ztop')]+[['ktop','int32','read_write']]
params=[['nc','int32'],['nl','int32']]+[[n,'float32'] for n in constants]+[[n,'int32'] for n in flags]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_dmp_mf','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
outputs=('vt','vq','edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','s_aw','s_awthl','s_awqt','s_awqv','s_awqc','s_awu','s_awv','s_awqke','s_awqnc','s_awqni','s_awqnwfa','s_awqnifa','s_awqnbca','sub_thl','sub_sqv','sub_u','sub_v','det_thl','det_sqv','det_sqc','det_u','det_v','qc_bl1d','cldfra_bl1d')
for suffix,fields,scatter in (('pack0',profiles[:25],False),('pack1',profiles[25:50],False),('pack2',profiles[50:],False),('scatter',outputs,True)):
    kernel=f'hexv842_mpas_bl_mynn_dmp_{suffix}_f32';typename='Hexv842MpasBlMynnDmp'+suffix.title()+'F32'
    rs=[['packed','float','read' if scatter else 'read_write']]+[[n,'float','read_write' if scatter else 'read'] for n in fields];p=[['nc','int32'],['nl','int32']]
    spec['kernels'].append({'name':kernel,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_dmp_mf','resources':rs,'shapes':{x[0]:[['any_1d']] for x in rs},'parameters':p,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']})
    text+=f'\nkernel void {kernel}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &cfg [[buffer(1)]],uint i [[thread_position_in_grid]]){{uint n=uint(cfg.nc*(cfg.nl+1));if(i>=n)return;uint col=i/uint(cfg.nl+1),k=i%uint(cfg.nl+1);\n'
    for n in fields:
        j=profiles.index(n);extent='cfg.nl+1' if n in interfaces else 'cfg.nl';packed=f'r.packed[ulong({j})*n+i]';value=f'r.{n}[ulong(col)*({extent})+k]'
        text+=f'if(k<uint({extent})) '+(f'{value}={packed};\n' if scatter else f'{packed}={value};\n')
    text+='}\n'
(root/f'woof/metal/shaders/{name}.metal').write_text(re.sub(r'(?m)^[ \t]+$','',text));(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
