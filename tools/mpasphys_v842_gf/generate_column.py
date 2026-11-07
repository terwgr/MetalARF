import re
from pathlib import Path
from woof.hex.mpasphys import gf_mpas_v842 as native
from tools.mpasphys_v842_gf import lower_control as l
root=Path(__file__).resolve().parents[2];source=native._SCHEME_SOURCE
names=re.findall(r'__device__\s+(?:void|GFReal)\s+(\w+)\s*\([^;{]*\)\s*\{',source)
def structured(name):
    special={'cup_kbcon':l.lower_kbcon,'cup_up_aa0':l.lower_aa0,'cup_ktop':l.lower_ktop,'cup_up_moisture':l.lower_up_moisture,'cup_dd_edt':l.lower_dd_edt,'cup_forcing_ens_3d':l.lower_forcing,'cup_gf_sh':l.lower_shallow}
    if name in special:return special[name]()[0]
    if name=='cup_gf':return l.lower_wrapped_exit(name)[0]
    return l.lower_partial(name)[0]
def typed(name):
    original=structured(name);head,body=original.split('{',1)
    match=re.match(r'__device__\s+(void|GFReal)\s+\w+\s*\((.*)\)\s*$',head,re.S)
    if not match:raise ValueError('signature '+name)
    result,args=match.groups();params=[];types=[]
    for item in args.split(','):
        p=item.strip()
        if p.startswith('GFWorkspace &'):params.append('thread GFWorkspace &workspace')
        elif '*' in p:
            m=re.fullmatch(r'(?:GFReal|int|GFText|char|bool)\s*\*\s*(\w+)',p)
            if not m:raise ValueError(name+' pointer '+p)
            t='P'+str(len(types));types.append(t);params.append(t+' '+m.group(1))
        else:params.append(p.replace('long long','long'))
    template='template<'+','.join('typename '+t for t in types)+'> ' if types else ''
    signature=template+'static inline '+result+' '+name+'('+', '.join(params)+')'
    body=re.sub(r'^\s*(void|GFReal|int|GFText)\s*\*\s*(\w+)\s*;',lambda m:'\n  device '+m.group(1)+' * '+m.group(2)+';',body,flags=re.M)
    body=re.sub(r'\(void\s*\*\)\s*workspace\.allocate',r'(device void *) workspace.allocate',body)
    body=re.sub(r'\((GFReal|int|GFText)\s*\*\)\s*(D_\d+)',r'(device \1 *) \2',body)
    body=re.sub(r'\((GFReal|int)\s*\*\)\s*(\w+)',r'\2',body)
    body=re.sub(r'\(char\s*\*\)\s*\(char\s*\*\)\s*("[^"]*")',r'\1',body)
    body=re.sub(r'\(char\s*\*\)\s*("[^"]*")',r'\1',body)
    body=re.sub(r'\(char\s*\*\)\s*&ierrc',r'(device char *)&ierrc',body)
    body=re.sub(r'\bunsigned long long\b','ulong',body);body=re.sub(r'\blong long\b','long',body);body=re.sub(r'\bunsigned long\b','ulong',body)
    body=re.sub(r'refused \([^\n)]*\)\. Zero it:', 'refused. Zero it:', body)
    body=re.sub(r'\b(?:L_\d+|__label_\d+)\s*:\s*;','',body)
    if re.search(r'\bgoto\b',body):raise ValueError('remaining goto '+name)
    return signature,body
items=[typed(n) for n in names]
sat=(root/'woof/metal/shaders/hexv842_mpas_gf_satvap_f32.metal').read_text();pre=sat.split('static inline GFReal satvap(',1)[0]
pre=pre.replace('GFReal(float f):bits(as_type<uint>(f)){}','GFReal(float f):bits(as_type<uint>(f)){} explicit operator int() const {return int(as_type<float>(bits));} explicit operator int() const device {return int(as_type<float>(bits));}')
pre=pre.replace("gf_nan(soft_gfk_pow_bits(a.bits,b.bits),a.bits,0u)","gf_nan(soft_gfk_pow_bits(a.bits,b.bits),a.bits,b.bits)")
pre+='''
static inline bool operator>(GFReal a,GFReal b){return soft32_greater(a.bits,b.bits);}
static inline bool operator<=(GFReal a,GFReal b){return soft32_less_equal(a.bits,b.bits);}
static inline bool operator>=(GFReal a,GFReal b){return soft32_greater_equal(a.bits,b.bits);}
static inline bool operator==(GFReal a,GFReal b){return soft32_equal(a.bits,b.bits);}
static inline bool operator!=(GFReal a,GFReal b){return !soft32_equal(a.bits,b.bits);}
static inline GFReal gf_add_rhs(GFReal a,GFReal b){return gf_nan(soft32_add(a.bits,b.bits),b.bits,a.bits);}
static inline GFReal gf_abs(GFReal a){return GFReal::word(a.bits&0x7fffffffu);}
static inline GFReal gf_sqrt(GFReal a){return gf_nan(soft32_sqrt(a.bits),a.bits,0u);}
static inline GFReal gf_powi(GFReal a,int p){return (a*a)*a;}
static inline GFReal gf_nan(){return GFReal::word(0x7fc00000u);}
static inline GFReal gf_infinity(){return GFReal::word(0x7f800000u);}
template<class A,class B> static inline auto gf_max(A a,B b)->decltype(a+b){return a>b?decltype(a+b)(a):decltype(a+b)(b);}
template<class A,class B> static inline auto gf_min(A a,B b)->decltype(a+b){return a<b?decltype(a+b)(a):decltype(a+b)(b);}
template<class D,class S> static inline void gf_memmove(D d,S s,ulong n){for(ulong i=0;i<n;i++)d[i]=s[i];}
template<class T> static inline void gf_memset(device T* d,int v,ulong n){device char* p=(device char*)d;for(ulong i=0;i<n;i++)p[i]=char(v);}
template<class T> static inline void gf_memset(thread T* d,int v,ulong n){thread char* p=(thread char*)d;for(ulong i=0;i<n;i++)p[i]=char(v);}
template<class A,class B> static inline int gf_compare_string(long na,A a,long nb,B b){for(long i=0;i<max(na,nb);i++){int x=i<na?(uchar)a[i]:32;int y=i<nb?(uchar)b[i]:32;if(x!=y)return x-y;}return 0;}
static inline void gf_unreachable_random(thread GFReal*){__builtin_trap();}
typedef char GFText[50];
struct GFWorkspace {device char* data;ulong capacity,position,highwater;uint diagnostics;
 device void* allocate(ulong n){n=(n+15)&~15ul;if(position+n>capacity){__builtin_trap();return nullptr;}device void* result=data+position;position+=n;highwater=max(highwater,position);return result;}};
struct GFFrame {thread GFWorkspace &ws;ulong saved;GFFrame(thread GFWorkspace &w):ws(w),saved(w.position){} ~GFFrame(){ws.position=saved;}};
'''
protos='\n'.join(sig+';' for sig,_ in items)
defs='\n'.join(sig+' {'+body for sig,body in items)
calls=re.findall(r'gf_memmove\s*\(([^;]+)\);', defs)
if not calls or any(not re.search(r',\s*\"[^\"]*\"\s*,', c) for c in calls): raise ValueError('GF memmove specialization requires literal sources')
entry=native._ENTRY.split('#ifndef GF_HOST_WITNESS',1)[0]
entry=entry.replace('__device__ void gf_column','static inline void gf_column').replace('GFWorkspace &workspace','thread GFWorkspace &workspace')
entry=re.sub(r'\b(GFReal|int)\s*\*\s*(lev|surface|isurface|outlev|outsurface|outint)',r'device \1* \2',entry)
entry=re.sub(r'\b(GFReal|int)\s*\*\s*(control|constants)',r'thread \1* \2',entry)
name='hexv842_mpas_gf_column_f32'
kernel='''
#include "hexv842_mpas_gf_column_f32_bindings.metalh"
kernel void hexv842_mpas_gf_column_f32(constant Hexv842MpasGfColumnF32Resources &r [[buffer(0)]], constant Hexv842MpasGfColumnF32Params &p [[buffer(1)]], uint lane [[thread_position_in_grid]]) {
 if(lane!=0)return; uint col=uint(p.column);
 int control[4]={p.itimestep,p.deep,p.shallow,1};GFReal constants[4]={p.xlv,p.cp,p.gravity,p.rv};
 ulong cap=ulong(p.scratch_words)*4ul;
 GFWorkspace ws={(device char*)r.scratch,cap,0,0,0};
 gf_column(p.nz,(device GFReal*)(r.levels+col*15*p.nz),(device GFReal*)(r.surface+col*8),r.kpbl+col,(device GFReal*)(r.outlev+col*6*p.nz),(device GFReal*)(r.outsurface+col*6),r.outint+col*3,control,constants,ws);
 r.highwater[col]=uint(ws.highwater);
}
'''
text=pre+protos+'\n'+defs+'\n'+entry+kernel
(root/f'woof/metal/shaders/{name}.metal').write_text('\n'.join(line.rstrip() for line in text.splitlines())+'\n')
import json
resources=[['levels','float','read_write'],['surface','float','read_write'],['kpbl','int32','read_write'],['outlev','float','read_write'],['outsurface','float','read_write'],['outint','int32','read_write'],['scratch','uint32','read_write'],['highwater','uint32','read_write']]
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/gf_mpas_v842.py','cuda_symbol':'gf_column','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32'],['nz','int32'],['column','int32'],['scratch_words','int32'],['itimestep','int32'],['deep','int32'],['shallow','int32'],['xlv','float32'],['cp','float32'],['gravity','float32'],['rv','float32']],'threads_per_group':32,'layout_version':1,'arithmetic_policies':['default']}]}
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
print('GF source column generated',len(names),'procedures')
