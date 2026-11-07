"""Generate classic RRTM strict SETCOEF."""
import copy
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=(ROOT/'woof/metal/shaders/rrtm_lw_family.metal').read_text()
    start=source.index('kernel void rrtm_lw_setcoef(')
    end=source.index('inline float rrtm_lw_interp4(',start)
    body=source[start:end]
    body=body.replace('rrtm_lw_setcoef','wrf_exact_rrtm_lw_setcoef')
    body=body.replace('RrtmLwSetcoef','WrfExactRrtmLwSetcoef')
    body=re.sub(r'\bfloat\b','Exact32',body)
    signatures=json.loads((ROOT/'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())
    entry=copy.deepcopy(next(x for x in signatures['kernels'] if x['name']=='rrtm_lw_setcoef'))
    for field,dtype,access in entry['resources']:
        if dtype != 'float': continue
        qualifier='const ' if access=='read' else ''
        body=re.sub(r'\br\.'+field+r'\b',f'reinterpret_cast<device {qualifier}Exact32*>(r.{field})',body)
    replacements={
        'log(':'exact_gfk_log(', 'exp(':'exact_gfk_exp(',
        'int(trunc(36.0f - 5.0f * (plog + 0.04f)))':'rrtm_trunc(36.0f - 5.0f * (plog + 0.04f))',
        'int(trunc(3.0f + tdiff / 15.0f))':'rrtm_trunc(3.0f + tdiff / 15.0f)',
        'int(trunc(3.0f + tdiff1 / 15.0f))':'rrtm_trunc(3.0f + tdiff1 / 15.0f)',
        'int(trunc(factor))':'rrtm_trunc(factor)',
        'Exact32(jt - 3)':'Exact32(float(jt - 3))',
        'Exact32(jt1 - 3)':'Exact32(float(jt1 - 3))',
        'Exact32(indself + 7)':'Exact32(float(indself + 7))',
        '(296.0f / 1013.0f)':'(Exact32(296.0f) / Exact32(1013.0f))'}
    for old,new in replacements.items():
        if old not in body: raise ValueError('missing '+old)
        body=body.replace(old,new)
    header='''// Generated from classic RRTM SETCOEF; do not edit.
#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_rrtm_lw_setcoef_bindings.metalh"
using namespace metal;
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline int rrtm_trunc(Exact32 x) { return int(trunc(as_type<float>(x.bits))); }
'''
    (ROOT/'woof/metal/shaders/wrf_exact_rrtm_setcoef.metal').write_text(header+body.rstrip()+'\n')
    entry.update(name='wrf_exact_rrtm_lw_setcoef',module='wrf_exact_rrtm_setcoef',
                 source='woof/metal/shaders/wrf_exact_rrtm_setcoef.metal',arithmetic_policies=['wrf-exact'])
    (ROOT/'woof/metal/signature_fragments/wrf_exact_rrtm_setcoef.json').write_text(json.dumps({'kernels':[entry]},indent=2)+'\n')
if __name__=='__main__':main()
