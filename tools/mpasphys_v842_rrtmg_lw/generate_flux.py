from pathlib import Path
import json,re,sys
root=Path.cwd();sys.path.insert(0,str(root))
from woof.core.rrtmg_lw import _lw_coalesced_source
source=_lw_coalesced_source((root/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
for suffix,symbol in (('accum','rlw_rtrn_accum_coalesced'),('final','rlw_rtrn_final_coalesced')):
    m=re.search(r'extern "C" __global__ void '+symbol+r'\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    args=re.sub(r'//[^\n]*','',source[m.end():i].rsplit(')',1)[0]);resources=[];params=[]
    for arg in args.split(','):
        item=arg.strip()
        if '*' in item:
            match=re.fullmatch(r'(const\s+)?(float|int|unsigned char)\s*\*\s*__restrict__\s*(\w+)',item)
            if not match:raise ValueError(item)
            const,t,n=match.groups();resources.append([n,'float' if t=='float' else 'int32','read' if const else 'read_write'])
        else:
            t,n=item.split();params.append([n,'float32' if t=='float' else 'int32'])
    body=source[i+1:j-1].replace('long long','long').replace('long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;','long tid = long(gid);').replace('int col = blockIdx.x * blockDim.x + threadIdx.x;','int col = int(gid);')
    name='hexv842_mpas_rrtmg_lw_'+suffix+'_f32';typ='Hexv842MpasRrtmgLw'+suffix.title()+'F32'
    header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define NBNDLW 16
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
'''+f'#include "{name}_bindings.metalh"\n'
    header+=f'kernel void {name}(constant {typ}Resources &r [[buffer(0)]],constant {typ}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{\n'
    for n,t in params:header+=f' {"float" if t=="float32" else "int"} {n}=p.{n};\n'
    for n,t,access in resources:header+=f' device {"const " if access=="read" else ""}{"int" if t=="int32" else "float"}* {n}=r.{n};\n'
    (root/f'woof/metal/shaders/{name}.metal').write_text(header+body+'\n}\n')
    spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/core/kernels/rrtmg_lw_chain.cu','cuda_symbol':symbol,'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}]}
    (root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
