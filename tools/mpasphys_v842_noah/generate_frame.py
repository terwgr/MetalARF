from pathlib import Path
import re, sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.noah_v842 import _CORE_CUDA, COLUMN_LAYOUTS
s=_CORE_CUDA
start=s.index('__device__ void noah_driver_column(')
end=s.index('// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207.',start)
driver=s[start:end]
head,body=driver.split('{',1)
first=body.index('int KTS = 1;')
last=body.index('XSDA_HFX = 0x0.0p+0f;')
aliases=body[:first]
decls=body[first:last]
fields=[]
for line in decls.splitlines():
 m=re.fullmatch(r'(float|int|bool) (\w+)(?:\[(\d+)\])?(?: = (.*?))?;(?: //.*)?',line)
 if not m: raise RuntimeError(line)
 fields.append(m.groups())
post_source=body[body.index('LAI = XLAI;'):body.rfind('} // module_sf_noahdrv.F:1590')]
live={n for t,n,a,v in fields if re.search(r'\b'+n+r'\b',post_source)}
struct='struct NoahDriverFrame {\n'+''.join(f'{t} {n}'+('[4]' if a else '')+';\n' for t,n,a,v in fields if n in live)+'};\n'
def refs(initialize):
 out=[]
 for t,n,a,v in fields:
  if n not in live:
   if initialize: out.append(f'{t} {n}'+('[4]' if a else '')+(f' = {v}' if v else '')+';')
   continue
  out.append(f'thread {t} '+(f'*{n}=f.{n};' if a else f'&{n}=f.{n};'))
  if initialize and v:out.append(f'{n} = {v};')
 return '\n'.join(out)+'\n'
pre=body[last:body.index('LAI = XLAI;')]
post=body[body.index('LAI = XLAI;'):body.rfind('} // module_sf_noahdrv.F:1590')]
for scheme,func in [('noah','noah_sflx'),('glacial','gl_sflx_glacial')]:
 decl=re.search(r'__device__ void '+func+r'\((.*?)\);',s,re.S).group(1)
 call=re.search(r'\b'+func+r'\(tables,I,J,([^;]+)\);',driver).group(0)
 args=call[call.index('(')+1:-2].split(',')
 params=decl.split(',');assert len(args)==len(params)
 names=[re.search(r'(\w+)\s*$',p).group(1) for p in params]
 actual=dict(zip(names,args)); existing=driver[:driver.index(call)]
 uninit={name for name,arg in actual.items() if arg not in ('tables','I','J','DT','ISURBAN','ISICE','USEMONALB','OPT_THCND','LUTYPE','SLTYPE','true') and not re.search(r'\b'+re.escape(arg)+r'\s*(?:\[[^]]+\])?\s*=',existing)}
 uninit.update(['SFCSPD','COSZ','PRCPRAIN','SOLARDIRECT','CM'])
 packed=[]; restored=[]
 for typ,ls,arr in [('float',COLUMN_LAYOUTS[scheme]['float'], 'land' if scheme=='noah' else 'glacier'),('int',COLUMN_LAYOUTS[scheme]['int'], 'landi' if scheme=='noah' else 'glacieri')]:
  for j,field in enumerate(ls):
   base=field.split('[')[0]; arg=actual[base]+field[len(base):]
   if arg=='DUMMY':continue
   if base not in uninit:packed.append(f'{arr}[c*{len(ls)}+{j}]={arg};')
   # I and J are call-by-value driver inputs, not persistent results.
   if arg not in ('I','J') and (arg.split('[')[0] in live or arg.split('[')[0] not in {n for _,n,_,_ in fields}):restored.append(f'{arg}={arr}[c*{len(ls)}+{j}];')
 replacement='\n'.join(packed)+'\nroute='+('1' if scheme=='noah' else '2')+'; return;'
 pre=pre.replace(call,replacement)
 pre=pre.replace('if(called)*called='+('1' if scheme=='noah' else '2')+';', '')
 if scheme=='noah':pre=pre.replace('if(tables.error)return; } // module_sf_noahdrv.F:1102','} // module_sf_noahdrv.F:1102')
 globals()[scheme+'_restore']='\n'.join(restored)
pre=pre.replace('label_50:; ; // module_sf_noahdrv.F:784','')
# Split exits each active branch at its source call. Inactive branches naturally finish.
pre += '}\n'  # closing outer land branch omitted by split
signature=head.replace('__device__ void noah_driver_column','static inline void frame_pre').replace('int I = 1, int J = 1, int* called = nullptr)', 'int I, int J, thread int &route, thread NoahDriverFrame &f, device float *land, device int *landi, device float *glacier, device int *glacieri, uint c)')
posthead=signature.replace('frame_pre','frame_post')
prep=signature+'{\n'+aliases+refs(True)+pre+'}\n'
postbody=posthead+'{\n'+aliases+refs(False)+'if(route==1){\n'+noah_restore+'\n}else if(route==2){\n'+glacial_restore+'\n}else{return;}\n'+post+'}\n'
# Source helper annotations are converted for Metal thread address space.
def metal(t):
 t=re.sub(r'\b(NoahTables|float|int|bool)\s*&',r'thread \1 &',t)
 t=re.sub(r'\b(const\s+)?(float|int|char)\s*\*',r'thread \1\2 *',t)
 return t.replace('device thread ', 'device ').replace('thread device ', 'device ').replace('auto& ', 'thread auto& ').replace('thread thread ', 'thread ')


from woof.hex.mpasphys.noah_v842 import _ENTRY_CUDA
entry=_ENTRY_CUDA.split('extern "C" __global__ void replay_driver(',1)[1].split('// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207.',1)[0]
setup=entry[entry.index('float TSK_arg='):entry.index('noah_driver_column(')]
call=entry[entry.index('noah_driver_column('):entry.index('\nx[8]=')]
writeback=entry[entry.index('x[8]='):entry.index('status[c]=tables.error;')]
call=call.replace('noah_driver_column(', 'FRAME_CALL(').replace('FGSN_2D_arg);','FGSN_2D_arg,1,1,route,f,land,landi,glacier,glacieri,c);')
header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define AD(a,b) mpas_add(a,b)
#define SU(a,b) mpas_sub(a,b)
#define MU(a,b) mpas_mul(a,b)
#define DV(a,b) cu_div(a,b)
static __attribute__((noinline)) float r_exp(float x){return gfk_exp(x);}
static __attribute__((noinline)) float r_pow(float x,float y){return gfk_pow(x,y);}
static inline float noah_powi(float a,int n){unsigned m=n<0?-n:n;float r=1;while(m){if(m&1)r=MU(r,a);m>>=1;if(m)a=MU(a,a);}return n<0?DV(1.0f,r):r;}
'''
tables=s[s.index('struct NoahTables {'):s.index('template<class T>')]
kern='''kernel void frame_kernel(device float *xs [[buffer(0)]], device int *ixs [[buffer(1)]], device const uint *ts [[buffer(2)]], device NoahDriverFrame *frames [[buffer(3)]], device int *routes [[buffer(4)]], device float *land [[buffer(5)]], device int *landi [[buffer(6)]], device float *glacier [[buffer(7)]], device int *glacieri [[buffer(8)]], device int *land_status [[buffer(9)]], device int *glacier_status [[buffer(10)]], constant int &mode [[buffer(11)]], constant int &n [[buffer(12)]], uint c [[thread_position_in_grid]]) {
if(c>=uint(n))return;
thread float x[79];thread int ix[7];for(int j=0;j<79;j++)x[j]=xs[c*79+j];for(int j=0;j<7;j++)ix[j]=ixs[c*7+j];
thread NoahTables tables;thread uint *t=reinterpret_cast<thread uint*>(&tables);for(uint j=0;j<sizeof(NoahTables)/4;j++)t[j]=ts[j];tables.error=0;
thread NoahDriverFrame f={};int route=0;
'''+setup+'''\nif(mode==0){\n'''+call.replace('FRAME_CALL','frame_pre')+'''\nframes[c]=f;routes[c]=route;
}else{route=routes[c];f=frames[c];if(route==1 && land_status[c])return;
'''+call.replace('FRAME_CALL','frame_post')+'''\n}
'''+writeback+'''\nfor(int j=0;j<79;j++)xs[c*79+j]=x[j];for(int j=0;j<7;j++)ixs[c*7+j]=ix[j];
}
'''

# Emit a private, unadmitted frame kernel. The two mode values bracket the
# existing land and glacial solvers; the caller owns route/status ordering.
import json
header += '#include "hexv842_mpas_noah_frame_f32_bindings.metalh"\n'
struct += 'static_assert(sizeof(NoahDriverFrame)==220, "Noah frame ABI");\n'
start=kern.index('kernel void frame_kernel(')
body_start=kern.index('if(c>=uint(n))return;')
kern = """kernel void hexv842_mpas_noah_frame_f32(
constant Hexv842MpasNoahFrameF32Resources &r [[buffer(0)]],
constant Hexv842MpasNoahFrameF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
device float *xs=r.floats; device int *ixs=r.integers; device const uint *ts=r.tables;
device NoahDriverFrame *frames=reinterpret_cast<device NoahDriverFrame*>(r.frames);
device int *routes=r.routes; device float *land=r.land; device int *landi=r.landi;
device float *glacier=r.glacier; device int *glacieri=r.glacieri;
device const int *land_status=r.land_status; device const int *glacier_status=r.glacier_status;
int mode=p.mode; int n=p.n;
"""+kern[body_start:]
kern=kern.replace('FGSN_2D_arg,1,1,route,f,', 'FGSN_2D_arg,r.locations[2*c],r.locations[2*c+1],route,f,')
kern=kern.replace('frames[c]=f;routes[c]=route;', 'frames[c]=f;routes[c]=route;r.status[c]=0;')
kern=kern.replace('if(route==1 && land_status[c])return;', 'if(route==1 && land_status[c]){r.status[c]=land_status[c];return;}\nif(route==2 && glacier_status[c]){r.status[c]=glacier_status[c];return;}')
(root/'woof/metal/shaders/hexv842_mpas_noah_frame_f32.metal').write_text(header+tables+'\n'+struct+metal(prep)+metal(postbody)+kern)
resources=[['floats','float','read_write'],['integers','int32','read_write'],['tables','uint32','read'],['frames','uint32','read_write'],['routes','int32','read_write'],['land','float','read_write'],['landi','int32','read_write'],['glacier','float','read_write'],['glacieri','int32','read_write'],['land_status','int32','read'],['glacier_status','int32','read'],['locations','int32','read'],['status','int32','read_write']]
spec={'kernels':[{'name':'hexv842_mpas_noah_frame_f32','module':'hexv842_mpas_noah_frame_f32','source':'woof/metal/shaders/hexv842_mpas_noah_frame_f32.metal','cuda_source':'woof/hex/mpasphys/noah_v842.py','cuda_symbol':'replay_driver_with_state','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32'],['mode','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(root/'woof/metal/signature_fragments/hexv842_mpas_noah_frame_f32.json').write_text(json.dumps(spec,indent=2)+'\n')
