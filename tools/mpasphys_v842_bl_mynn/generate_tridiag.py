"""Generate native MYNN tridiagonal column solves from the selected body."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA
name='hexv842_mpas_bl_mynn_tridiag_f32'
header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_adapter_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_adapter_f32_bindings.metalh',name+'_bindings.metalh')
header+='\nstatic inline float mynn_neg(float a){return as_type<float>(as_type<uint>(a)^0x80000000u);}\n'
text=header;kernels=[]
for variant in (1,2,3):
    fn='f_tridiag'+('' if variant==1 else str(variant))
    match=re.search(r'__device__ __noinline__ void '+fn+r'\s*\(.*?\)\s*\{',MYNN_MODULE_CUDA,re.S);assert match
    i=match.end();j=i;depth=1
    while depth:
        if MYNN_MODULE_CUDA[j]=='{':depth+=1
        elif MYNN_MODULE_CUDA[j]=='}':depth-=1
        j+=1
    body=MYNN_MODULE_CUDA[i:j-1]
    body=re.sub(r'MynnFrame mynn_frame\(arena\);\n','',body)
    body=re.sub(r'\w+\.rebase\(\{1\}\);\n','',body)
    body=re.sub(r'MynnArray<float, 1> (q|cp|dp) = mynn_array<float, 1>\(arena, \{1\}, \{n\}\);\n','',body)
    assert 'Mynn' not in body
    names=['a','b','c','d']+(['x'] if variant!=1 else [])+(['q'] if variant==1 else ['cp','dp'] if variant==2 else [])
    accesses={'a':'read','b':'read_write' if variant==3 else 'read','c':'read_write' if variant==1 else 'read','d':'read' if variant==2 else 'read_write','x':'read_write','q':'read_write','cp':'read_write','dp':'read_write'}
    resources=[[n,'float',accesses[n]] for n in names];params=[['nc','int32'],['nl','int32']]
    kernel=f'hexv842_mpas_bl_mynn_tridiag{variant}_f32';typename=''.join(part.title() for part in kernel.split('_'))
    macros=''.join(f'#define {n}(i) r.{n}[(ulong(col)*nl)+((i)-1)]\n' for n in names)
    undef=''.join(f'#undef {n}\n' for n in names)
    text+=f'\nkernel void {kernel}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {{\nif(col>=uint(cfg.nc))return;int nl=cfg.nl;int '+('kte=nl;' if variant==3 else 'n=nl;')+'\n'+macros+body+'\n'+undef+'}\n'
    kernels.append({'name':kernel,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':fn,'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']})
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels':kernels},indent=2)+'\n')
