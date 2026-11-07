"""Shared source and table inventory for native LW gas-band generators."""
import re
from woof.hex.mpasphys.rrtmg_lw_v842 import _REDUCTIONS, _solver_source
from woof.core.rrtmg_lw import GPU_BAND_TABS, NGC


def band_source(band):
    """Return source macros, selected kernel body and its ordered table names."""
    if band not in GPU_BAND_TABS:
        raise ValueError('native LW gas band must be between 1 and 16')
    source=_solver_source()
    match=re.search(r'extern "C" __global__ void rlw_taugb'+str(band)+r'\s*\(',source)
    if match is None:
        raise ValueError('native LW gas band source is missing')
    start=source.index('{',match.end());end=start+1;depth=1
    while depth:
        if source[end]=='{':depth+=1
        elif source[end]=='}':depth-=1
        end+=1
    macros=source[source.index('#define SL_PAVEL 0'):source.index('// UNIVERSAL band-kernel signature')]
    return macros,source[start+1:end-1],tuple(GPU_BAND_TABS[band])


def band_table_sizes(band):
    """Return source reduction sizes in the native ordered table ABI."""
    sizes={}
    for b,dest,_,points,_,_ in _REDUCTIONS:
        if b!=band:continue
        name=dest.split('/',1)[1]
        name={'ka':'absa','kb':'absb'}.get(name,name)
        sizes[name]=points*int(NGC[band-1])
    return {name:sizes[name] for name in GPU_BAND_TABS[band]}
