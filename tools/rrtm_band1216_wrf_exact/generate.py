"""Generate strict classic RRTM GASABS bands 12–16."""
import copy,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    source=(ROOT/'woof/metal/shaders/rrtm_lw_family.metal').read_text()
    h0=source.index('inline float rrtm_lw_interp4(');h1=source.index('kernel void rrtm_lw_taumol_01_02(',h0)
    k0=source.index('kernel void rrtm_lw_taumol_12_16(');k1=source.index('kernel void rrtm_lw_taumol_09(',k0)
    body=source[h0:h1]+source[k0:k1]
    body=body.replace('rrtm_lw_taumol_12_16','wrf_exact_rrtm_lw_taumol_12_16').replace('RrtmLwTaumol1216','WrfExactRrtmLwTaumol1216')
    old="""low=comb*rrtm_lw_interp8(r.tables+ma.x,p.nrowa,p.ng,g,lo0+js-1,
          lo1+js-1,9,fs,r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])
          +r.colh2o[i]*continuum;"""
    new="""low=comb*rrtm_lw_interp8(r.tables+ma.x,p.nrowa,p.ng,g,lo0+js-1,
          lo1+js-1,9,fs,r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])
          +(r.colh2o[i]*r.selffac[i])*(s0+r.selffrac[i]*(s1-s0));"""
    # Keep continuum bracket outside the branch so both source forms can use it.
    body=body.replace('continuum=r.selffac[i]*(s0+r.selffrac[i]*(s1-s0));', 'continuum=(s0+r.selffrac[i]*(s1-s0));')
    new=new.replace('(s0+r.selffrac[i]*(s1-s0))','continuum')
    if body.count(old)!=1: raise ValueError('missing mixed-band continuum')
    body=body.replace(old,new)
    old="""low=ga*rrtm_lw_interp4(r.tables+ma.x,p.nrowa,p.ng,g,lo0,lo1,
          r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])+r.colh2o[i]*continuum;"""
    new="""low=ga*(rrtm_lw_interp4(r.tables+ma.x,p.nrowa,p.ng,g,lo0,lo1,
          r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])+r.selffac[i]*continuum);"""
    if body.count(old)!=1: raise ValueError('missing band 14 continuum')
    body=body.replace(old,new)
    body=re.sub(r'\bfloat\b','Exact32',body)
    entries=json.loads((ROOT/'woof/metal/signature_fragments/rrtm_lw_family.json').read_text())['kernels']
    entry=copy.deepcopy(next(x for x in entries if x['name']=='rrtm_lw_taumol_12_16'))
    for name,dtype,access in entry['resources']:
        if dtype!='float':continue
        const='const ' if access=='read' else ''
        body=re.sub(r'\br\.'+name+r'\b',f'reinterpret_cast<device {const}Exact32*>(r.{name})',body)
    replace={'int(trunc(mult))':'rrtm_trunc(mult)', 'floor(mult)':'rrtm_floor(mult)', 'p.strrat':'Exact32(p.strrat)'}
    for old,new in replace.items():
        if old not in body:raise ValueError('missing '+old)
        body=body.replace(old,new)
    header='''// Generated from classic RRTM GASABS bands 12–16; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_taumol_12_16_bindings.metalh"
using namespace metal;
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline int rrtm_trunc(Exact32 x) { return int(trunc(as_type<float>(x.bits))); }
static inline Exact32 rrtm_floor(Exact32 x) { return Exact32(floor(as_type<float>(x.bits))); }
'''
    (ROOT/'woof/metal/shaders/wrf_exact_rrtm_band1216.metal').write_text(header+body.rstrip()+'\n')
    entry.update(name='wrf_exact_rrtm_lw_taumol_12_16',module='wrf_exact_rrtm_band1216',source='woof/metal/shaders/wrf_exact_rrtm_band1216.metal',arithmetic_policies=['wrf-exact'])
    (ROOT/'woof/metal/signature_fragments/wrf_exact_rrtm_band1216.json').write_text(json.dumps({'kernels':[entry]},indent=2)+'\n')
if __name__=='__main__':main()
