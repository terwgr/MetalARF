"""Generate the Metal land-use initializer from the native v842 transcription."""
from pathlib import Path
import json, re, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from woof.hex.mpasphys.noah_v842 import _LANDUSE_CUDA
body=_LANDUSE_CUDA.split('int season=',1)[1].rsplit('\n}',1)[0]
body='int season='+body
for i,name in enumerate(('ALBD','SLMO','SFEM','SFZ0','THERIN','SCFX','SFHC')):
    body=re.sub(r'ts->'+name+r'\[([^\]]+)\]',lambda m:f'r.tables[{1200*i}+({m.group(1)})]',body)
for old,new in (('AD','mpas_add'),('SU','mpas_sub'),('MU','mpas_mul'),('DV','cu_div')):
    body=re.sub(r'\b'+old+r'\(',new+'(',body)
body=body.replace('int is=(season-1)*100+category-1,ice=(season-1)*100+ix[3]-1;', '''if(category<1 || category>100 || ((x[2]>0.5f || x[3]>=x[13]) && (ix[3]<1 || ix[3]>100))) { r.status[c]=-1; return; }
int is=(season-1)*100+category-1,ice=(season-1)*100+ix[3]-1;''')
shader='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_noah_landuse_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_noah_landuse_f32(
    constant Hexv842MpasNoahLanduseF32Resources &r [[buffer(0)]],
    constant Hexv842MpasNoahLanduseF32Params &p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) {
    if(c>=uint(p.n)) return;
    device float *x=r.floats+c*14;
    device const int *ix=r.integers+c*9;
    r.status[c]=0;
    if(ix[4]) return;
'''+body+'\n}\n'
(ROOT/'woof/metal/shaders/hexv842_mpas_noah_landuse_f32.metal').write_text(shader)
resources=[['floats','float','read_write'],['integers','int32','read'],['tables','float','read'],['status','int32','read_write']]
spec={'kernels':[{'name':'hexv842_mpas_noah_landuse_f32','module':'hexv842_mpas_noah_landuse_f32','source':'woof/metal/shaders/hexv842_mpas_noah_landuse_f32.metal','cuda_source':'woof/hex/mpasphys/noah_v842.py','cuda_symbol':'replay_landuse','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':[['n','int32']],'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
(ROOT/'woof/metal/signature_fragments/hexv842_mpas_noah_landuse_f32.json').write_text(json.dumps(spec,indent=2)+'\n')
