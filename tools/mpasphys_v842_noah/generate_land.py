"""Generate the native Noah land kernel and its Metal ABI."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.noah_v842 import _CORE_CUDA
pat=re.compile(r'__device__\s+(?:__forceinline__\s+)?(?:void|float|int)\s+(\w+)\s*\(')
functions={}
for m in pat.finditer(_CORE_CUDA):
 depth=1;i=m.end()
 while depth:
  if _CORE_CUDA[i]=='(':depth+=1
  elif _CORE_CUDA[i]==')':depth-=1
  i+=1
 while _CORE_CUDA[i].isspace():i+=1
 if _CORE_CUDA[i]!='{':continue
 start=m.start();j=i+1;depth=1
 while depth:
  if _CORE_CUDA[j]=='{':depth+=1
  elif _CORE_CUDA[j]=='}':depth-=1
  j+=1
 functions[m.group(1)]=(start,j,_CORE_CUDA[i:j])
from woof.hex.mpasphys.noah_v842 import _ENTRY_CUDA,_POWI_RECURSIVE,_POWI_STATIC
seen=set();todo=['noah_sflx']
while todo:
 name=todo.pop()
 if name in seen:continue
 seen.add(name)
 if name in functions:todo.extend(set(re.findall(r'\b((?:noah|gl)_\w+)\s*\(',functions[name][2]))-seen)
s=_CORE_CUDA.split('extern "C" __device__ void noah_finish_module_locations')[0]
for name,(start,end,body) in sorted(functions.items(),key=lambda x:x[1][0],reverse=True):
 if name not in seen:s=s[:start]+s[end:]
s=s.replace("label_1001:; ; // module_sf_noahlsm.F:1531\nif ((!((NLOG < 10) && (KCOUNT == 0)))) { goto label_1002; } // module_sf_noahlsm.F:1532", "while ((NLOG < 10) && (KCOUNT == 0)) {")
s=s.replace("goto label_1001; // module_sf_noahlsm.F:1564\nlabel_1002:; ; // module_sf_noahlsm.F:1565", "}")
# The source checks category counts before its table reads. Only when a
# malformed table advertises more rows than its fixed ABI supplies do we
# refuse an otherwise unchecked upper index.
s=s.replace('CSOIL = CSOIL_DATA; // module_sf_noahlsm.F:2450',
            'if (SOILTYP>30 || VEGTYP>50 || SLOPETYP>30) { tables.error=-1; return; }\nCSOIL = CSOIL_DATA; // module_sf_noahlsm.F:2450')
s=s.replace(_POWI_RECURSIVE,_POWI_STATIC)
s=s.replace('__device__ __forceinline__','static inline').replace('__device__','static __attribute__((noinline))')
s=re.sub(r'\b(NoahTables|float|int|bool)\s*&',r'thread \1 &',s)
s=re.sub(r'\b(const\s+)?(float|int|char)\s*\*',r'thread \1\2 *',s)
s=s.replace('auto& ', 'thread auto& ')
body=_ENTRY_CUDA.split('extern "C" __global__ void replay_noah(',1)[1].split('extern "C" __global__ void replay_glacial(',1)[0]
body=body[body.index('// Four packed soil slots'):body.rfind('}')]
pre='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_libm64.metalh"
#include "hexv842_mpas_noah_land_f32_bindings.metalh"
#pragma clang fp contract(off)
#define AD(a,b) mpas_add(a,b)
#define SU(a,b) mpas_sub(a,b)
#define MU(a,b) mpas_mul(a,b)
#define DV(a,b) cu_div(a,b)
static __attribute__((noinline)) float r_log(float x){return gfk_log(x);}
static __attribute__((noinline)) float r_exp(float x){return gfk_exp(x);}
static __attribute__((noinline)) float r_pow(float x,float y){return gfk_pow(x,y);}
static __attribute__((noinline)) float r_log10(float x){return as_type<float>(soft64_to_f32_bits(soft64_div(soft64_plm_log(soft64_from_f32_bits(as_type<uint>(x))),0x40026bb1bbb55516ul)));}
'''
body=body.replace('if(ix[3]!=4){status[c]=426;return;}', 'if(ix[3]!=4){r.status[c]=426;return;}\nif(ix[5]<1 || ix[6]<1 || ix[7]<1){r.status[c]=-1;return;}')
body=body.replace('status[c]=tables.error;', 'r.status[c]=tables.error;')
kernel='''kernel void hexv842_mpas_noah_land_f32(
constant Hexv842MpasNoahLandF32Resources &r [[buffer(0)]],
constant Hexv842MpasNoahLandF32Params &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
if(c>=uint(p.n)) return;
thread float x[110]; thread int ix[14]; for(int j=0;j<110;j++) x[j]=r.floats[c*110+j]; for(int j=0;j<14;j++) ix[j]=r.integers[c*14+j];
thread NoahTables tables; thread uint *t=reinterpret_cast<thread uint*>(&tables); for(uint j=0;j<sizeof(NoahTables)/4;j++)t[j]=r.tables[j]; tables.error=0;
'''+body+'''
for(int j=0;j<110;j++)r.floats[c*110+j]=x[j]; for(int j=0;j<14;j++)r.integers[c*14+j]=ix[j];
}
'''
(ROOT/'woof/metal/shaders/hexv842_mpas_noah_land_f32.metal').write_text(pre+s+kernel)
resources=[['floats','float','read_write'],['integers','int32','read_write'],['tables','uint32','read'],['status','int32','read_write']]
spec={'kernels':[{'name':'hexv842_mpas_noah_land_f32','module':'hexv842_mpas_noah_land_f32','source':'woof/metal/shaders/hexv842_mpas_noah_land_f32.metal','cuda_source':'woof/hex/mpasphys/noah_v842.py','cuda_symbol':'replay_noah','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(ROOT/'woof/metal/signature_fragments/hexv842_mpas_noah_land_f32.json').write_text(json.dumps(spec,indent=2)+'\n')
