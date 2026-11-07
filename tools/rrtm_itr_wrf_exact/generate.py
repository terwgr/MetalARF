"""Generate strict classic RRTM GASABS Pade index."""
import copy,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=(ROOT/'woof/metal/shaders/rrtm_lw_family.metal').read_text()
    k0=source.index('kernel void rrtm_lw_make_itr(');k1=source.index('inline float rrtm_lw_o3_layer(',k0)
    body=source[k0:k1]
    body=body.replace('rrtm_lw_make_itr','wrf_exact_rrtm_lw_make_itr').replace('RrtmLwMakeItr','WrfExactRrtmLwMakeItr')
    body=re.sub(r'\bfloat\b','Exact32',body)
    entries=json.loads((ROOT/'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())['kernels']
    entry=copy.deepcopy(next(x for x in entries if x['name']=='rrtm_lw_make_itr'))
    for name,dtype,access in entry['resources']:
        if dtype!='float':continue
        const='const ' if access=='read' else ''
        body=re.sub(r'\br\.'+name+r'\b',f'reinterpret_cast<device {const}Exact32*>(r.{name})',body)
    body=body.replace('p.bpade','Exact32(p.bpade)')
    old='int(trunc(5000.0f*tf+0.5f))'
    if body.count(old)!=1:raise ValueError('missing Pade rounding')
    body=body.replace(old,'int(trunc(as_type<float>((5000.0f*tf+0.5f).bits)))')
    header='''// Generated from classic RRTM GASABS Pade index; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_make_itr_bindings.metalh"
using namespace metal;
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits,a.bits); }
'''
    (ROOT/'woof/metal/shaders/wrf_exact_rrtm_itr.metal').write_text(header+body.rstrip()+'\n')
    entry.update(name='wrf_exact_rrtm_lw_make_itr',module='wrf_exact_rrtm_itr',source='woof/metal/shaders/wrf_exact_rrtm_itr.metal',arithmetic_policies=['wrf-exact'])
    (ROOT/'woof/metal/signature_fragments/wrf_exact_rrtm_itr.json').write_text(json.dumps({'kernels':[entry]},indent=2)+'\n')
if __name__=='__main__':main()
