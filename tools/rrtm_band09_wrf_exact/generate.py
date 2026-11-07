"""Generate strict classic RRTM GASABS band 9."""
import copy,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=(ROOT/'woof/metal/shaders/rrtm_lw_family.metal').read_text()
    h0=source.index('inline float rrtm_lw_interp4(');h1=source.index('kernel void rrtm_lw_taumol_01_02(',h0)
    k0=source.index('kernel void rrtm_lw_taumol_09(');k1=source.index('kernel void rrtm_lw_rtrn_down(',k0)
    body=source[h0:h1]+source[k0:k1]
    body=body.replace('rrtm_lw_taumol_09','wrf_exact_rrtm_lw_taumol_09').replace('RrtmLwTaumol09','WrfExactRrtmLwTaumol09')
    old='r.selffac[i]*(s0+r.selffrac[i]*(s1-s0))'
    if body.count(old)!=1: raise ValueError('missing band 4–5 continuum')
    body=body.replace(old, '(s0+r.selffrac[i]*(s1-s0))')
    old='r.colh2o[i]*continuum'
    if body.count(old)!=1: raise ValueError('missing band 4–5 water continuum')
    body=body.replace(old, '(r.colh2o[i]*r.selffac[i])*continuum')
    old='if(r.layswtch[col]>=1 && lay+1>=r.layswtch[col])ioff=2*p.ng;'
    if body.count(old)!=1: raise ValueError('missing band 9 sticky switch')
    body=body.replace(old, 'if(r.layswtch[col]>=1 && lay+1>=r.layswtch[col] && (r.laylow[col]<1 || lay+1<r.laylow[col] || r.layswtch[col]>=r.laylow[col]))ioff=2*p.ng;')
    old='else if(js==9){jsl=10;fsl=0.0f;jfrac=8;ffrac=1.0f;}'
    if body.count(old)!=1: raise ValueError('missing band 9 terminal interval')
    body=body.replace(old,'else if(js==9){jsl=10;fsl=1.0f;jfrac=8;ffrac=1.0f;}')
    body=re.sub(r'\bfloat\b','Exact32',body)
    entries=json.loads((ROOT/'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())['kernels']
    entry=copy.deepcopy(next(x for x in entries if x['name']=='rrtm_lw_taumol_09'))
    for name,dtype,access in entry['resources']:
        if dtype!='float':continue
        const='const ' if access=='read' else ''
        body=re.sub(r'\br\.'+name+r'\b',f'reinterpret_cast<device {const}Exact32*>(r.{name})',body)
    replace={'int(trunc(mult))':'rrtm_trunc(mult)', 'floor(mult)':'rrtm_floor(mult)', 'int(trunc(fsl+0.5f))':'rrtm_trunc(fsl+0.5f)'}
    for old,new in replace.items():
        if old not in body:raise ValueError('missing '+old)
        body=body.replace(old,new)
    header='''// Generated from classic RRTM GASABS band 9; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_taumol_09_bindings.metalh"
using namespace metal;
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits,a.bits); }
static inline int rrtm_trunc(Exact32 x) { return int(trunc(as_type<float>(x.bits))); }
static inline Exact32 rrtm_floor(Exact32 x) { return Exact32(floor(as_type<float>(x.bits))); }
'''
    (ROOT/'woof/metal/shaders/wrf_exact_rrtm_band09.metal').write_text(header+body.rstrip()+'\n')
    entry.update(name='wrf_exact_rrtm_lw_taumol_09',module='wrf_exact_rrtm_band09',source='woof/metal/shaders/wrf_exact_rrtm_band09.metal',arithmetic_policies=['wrf-exact'])
    (ROOT/'woof/metal/signature_fragments/wrf_exact_rrtm_band09.json').write_text(json.dumps({'kernels':[entry]},indent=2)+'\n')
if __name__=='__main__':main()
