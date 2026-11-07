"""Generate strict classic RRTM GASABS bands 1 and 2."""
import copy,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=(ROOT/'woof/metal/shaders/rrtm_lw_family.metal').read_text()
    h0=source.index('inline float rrtm_lw_interp4(')
    h1=source.index('inline float rrtm_lw_interp8(',h0)
    k0=source.index('kernel void rrtm_lw_taumol_01_02(')
    k1=source.index('kernel void rrtm_lw_taumol_03(',k0)
    body=source[h0:h1]+source[k0:k1]
    body=body.replace('rrtm_lw_taumol_01_02','wrf_exact_rrtm_lw_taumol_01_02')
    body=body.replace('RrtmLwTaumol0102','WrfExactRrtmLwTaumol0102')
    body=re.sub(r'\bfloat\b','Exact32',body)
    signatures=json.loads((ROOT/'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())
    entry=copy.deepcopy(next(x for x in signatures['kernels'] if x['name']=='rrtm_lw_taumol_01_02'))
    for name,dtype,access in entry['resources']:
        if dtype!='float': continue
        const='const ' if access=='read' else ''
        body=re.sub(r'\br\.'+name+r'\b',f'reinterpret_cast<device {const}Exact32*>(r.{name})',body)
    expr='int(trunc(200.0f * fp2 + 0.5f))'
    if expr not in body:raise ValueError('unexpected index expression')
    body=body.replace(expr,'rrtm_trunc(200.0f * fp2 + 0.5f)')
    header='''// Generated from classic RRTM GASABS bands 1 and 2; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_taumol_01_02_bindings.metalh"
using namespace metal;
static inline int rrtm_trunc(Exact32 x) { return int(trunc(as_type<float>(x.bits))); }
'''
    (ROOT/'woof/metal/shaders/wrf_exact_rrtm_band12.metal').write_text(header+body.rstrip()+'\n')
    entry.update(name='wrf_exact_rrtm_lw_taumol_01_02',module='wrf_exact_rrtm_band12',
                 source='woof/metal/shaders/wrf_exact_rrtm_band12.metal',arithmetic_policies=['wrf-exact'])
    (ROOT/'woof/metal/signature_fragments/wrf_exact_rrtm_band12.json').write_text(json.dumps({'kernels':[entry]},indent=2)+'\n')
if __name__=='__main__':main()
