"""Generate strict open-boundary kernels from selected WOOF operations."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure, def_name, split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels

RADIATIVE=('open_u_radiative','open_v_radiative')
COPIES=('open_zero_gradient_x','open_zero_gradient_y',
        'open_zero_gradient_scalar_x','open_zero_gradient_scalar_y')
NAMES=RADIATIVE+COPIES
STEM='wrf_exact_openbc'


def retained_source():
    from woof.core.kernels import module_source
    source='#define GPUWM_WRF_EXACT 1\n'+module_source('openbc')
    expanded=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],
                            input=source,text=True,capture_output=True,check=True).stdout
    return closure(split_top(expanded),RADIATIVE)[0]


def extract_kernel(source,name):
    match=re.search(r'kernel void '+name+r'\s*\(',source)
    if not match: raise ValueError('missing '+name)
    start=source.index('{',match.start());depth=1;end=start+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[match.start():end]


def main():
    source='\n'.join(retained_source()); params=parse_kernels(source)
    registry={x['name']:x for x in json.loads((ROOT/'woof/metal/kernel_signatures.json').read_text())['kernels']}
    ordinary=(ROOT/'woof/metal/shaders/openbc_radiative.metal').read_text()
    signatures=[];defs=[];wrappers=[]
    for name in NAMES:
        sig=copy.deepcopy(registry[name]);sig.update(name='wrf_exact_'+name,module=STEM,
            source=f'woof/metal/shaders/{STEM}.metal',arithmetic_policies=['wrf-exact'])
        signatures.append(sig)
        if name in RADIATIVE:
            item=next(x for x in retained_source() if def_name(x)==name)
            item=re.sub(r'extern\s+"C"\s+__global__\s*','',item)
            pos=item.index('{');head,body=item[:pos].rstrip(),item[pos:]
            head=head[:-1]+', int t)\n'
            body,count=re.subn(r'int t = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;', '',body)
            if count != 1: raise ValueError('unexpected source index: '+name)
            item=head+body
            item=re.sub(r'\b__restrict__\b','',item)
            item=re.sub(r'\bconst real\s*\*','device const Exact32*',item)
            item=re.sub(r'\breal\s*\*','device Exact32*',item)
            item=re.sub(r'\breal\b','Exact32',item)
            item=re.sub(r'\bsize_t\b','ulong',item)
            for old,new in {'__fdiv_rn':'exact_div','fminf':'exact_min','fmaxf':'exact_max'}.items():
                item=re.sub(r'\b'+old+r'\b',new,item)
            if re.search(r'\b(?:__f\w+|blockIdx|threadIdx|real)\b',item): raise ValueError('untranslated '+name)
            defs.append(item)
            title=''.join(x.title() for x in sig['name'].split('_'))
            args=[]
            for p in params[name]:
                if p.pointer:
                    const='const ' if 'const' in p.text.split() else ''
                    args.append(f'reinterpret_cast<device {const}Exact32*>(r.{p.name})')
                else:
                    args.append(f'Exact32(q.{p.name})' if p.base=='real' else f'q.{p.name}')
            extent='q.nz*q.ny' if name=='open_u_radiative' else 'q.nz*q.nx'
            wrappers.append(f'kernel void {sig["name"]}(constant {title}Resources &r [[buffer(0)]], constant {title}Params &q [[buffer(1)]], uint t [[thread_position_in_grid]]) {{\n  if (t>=uint({extent})) return;\n  {STEM}_source::{name}('+', '.join(args)+', int(t));\n}')
        else:
            item=extract_kernel(ordinary,name)
            if name == 'open_zero_gradient_x':
                # The shared source uses the preceding vertical level, not row.
                item=item.replace('- ulong(nxf)', '- ulong(q.ny) * ulong(nxf)')
            item=item.replace('kernel void '+name,'kernel void '+sig['name'])
            title=''.join(x.title() for x in sig['name'].split('_'))
            old=''.join(x.title() for x in name.split('_'))
            item=item.replace(old+'Resources',title+'Resources').replace(old+'Params',title+'Params')
            for field,_dtype,access in sig['resources']:
                const='const ' if access=='read' else ''
                item=re.sub(r'\br\.'+field+r'\b',f'reinterpret_cast<device {const}Exact32*>(r.{field})',item)
            wrappers.append(item)
    headers=''.join(f'#include "wrf_exact_{name}_bindings.metalh"\n' for name in NAMES)
    shader=('// Generated from WOOF open-boundary operations; do not edit.\n'
            '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'+headers+'using namespace metal;\n'
            'static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }\n'
            f'namespace {STEM}_source {{\n'+'\n'.join(defs)+'\n}\n'+'\n'.join(wrappers)+'\n')
    (ROOT/f'woof/metal/shaders/{STEM}.metal').write_text('\n'.join(x.rstrip() for x in shader.splitlines())+'\n')
    (ROOT/f'woof/metal/signature_fragments/{STEM}.json').write_text(json.dumps({'kernels':signatures},indent=2)+'\n')

if __name__=='__main__': main()
