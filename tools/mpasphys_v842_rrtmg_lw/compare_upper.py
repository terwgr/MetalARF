"""Compare upper LW gas bands to selected host-compiled source bodies."""
from pathlib import Path
from importlib.resources import files
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_rrtmg_lw.band_source import band_source,band_table_sizes
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source
from woof.hex.mpasphys.metal_rrtmg_lw_upper_v842 import band, BAND_TABLE_SIZES
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_kernel(number):
    source=_solver_source();m=re.search(r'extern "C" __global__ void rlw_taugb'+str(number)+r'\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    _,_,tables=band_source(number)
    macros=source[source.index('#define SL_PAVEL 0'):source.index('extern "C" __global__ void rlw_taugb1(')]
    chosen=source[m.start():j].replace('const float* const* __restrict__ tabs,',','.join('const float* __restrict__ in_'+n for n in tables)+',')
    for ix,n in enumerate(tables):chosen=chosen.replace(f'const float* {n} = tabs[{ix}];',f'const float* {n} = in_{n};')
    header='''#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
    if number in (9,12,13,15,16):
        def func(name):
            i=source.index('__device__ ',source.index(name)-30);j=source.index('{',i);k=j+1;depth=1
            while depth:
                if source[k]=='{':depth+=1
                elif source[k]=='}':depth-=1
                k+=1
            return source[i:k]
        a=source.index('#define KA3_0609');b=source.index('// tau_major sum',a)
        c=source.index('#define RLW1316_A3');d=source.index('// The specparm branch triplet',c)
        helpers=(files('woof.core.kernels').joinpath('glibc_flt32.cuh').read_text()+'\n#define rlw_pow gfk_pow\n'+func('rlw_pow4')+'\n'+source[a:b]+func('rlw_ninemajor_0609')+'\n'+source[c:d]+func('rlw_nine_major_b1316'))
    else:helpers=''
    return HostModule(f'mpas_v842_lw_band{number}',header+macros+helpers+chosen).get_function('rlw_taugb'+str(number))


def main():
    nc,nl=4,6;rng=np.random.default_rng(842)
    fs=rng.uniform(.01,.9,(33,nc,nl)).astype(np.float32);fs[0]=[900,600,200,80,30,5];fs[1]=2e23
    isv=np.ones((6,nc,nl),np.int32);isv[0]=[1,5,12,13,30,58];isv[1]=[1,2,4,1,2,4];isv[2]=[1,3,4,1,3,4];isv[3]=[1,3,9,1,3,9];isv[4]=[1,2,3,1,2,3];isv[5]=[1,3,18,1,3,18]
    fs[2,:,0]=[.000001,.3,1,1000];fs[3,:,0]=[10000,.3,.000001,15000];fs[5,:,0]=[1,3000,6000,10000]
    # Exact species interpolation boundaries 1/8 and 7/8.
    fs[2,:,1]=[1,7,1,7];fs[3,:,1]=[7,1,7,1];fs[5,:,1]=[7,1,7,1];fs[7,:,1]=[7,1,7,1];fs[14:26,:,1]=1
    # A positive subnormal species ratio exercises exact remainder.
    fs[2,0,2]=np.array(0x00010000,np.uint32).view(np.float32)
    laytrop=np.full(nc,3,np.int32);wx=rng.uniform(.01,.9,(nc,4,nl)).astype(np.float32);chi=np.ones(413,np.float32)
    oneminus=np.float32(.999999)
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        for number in map(int,sys.argv[1:]):
            sizes=band_table_sizes(number)
            assert sizes==BAND_TABLE_SIZES[number]
            for tiny in (False,True):
                tabs={n:rng.uniform(.001,.5,size).astype(np.float32) for n,size in sizes.items()}
                # Exercise direct and interpolated subnormal Planck coefficients.
                if tiny:
                    for name in ('fracrefa','fracrefb'):
                        if name in tabs:tabs[name][:]=np.array(0x00100000,np.uint32).view(np.float32)
                taug=rng.uniform(.01,.9,(nc,nl,140)).astype(np.float32);fracs=taug.copy()
                actual=xp.asarray(taug.copy());af=xp.asarray(fracs.copy())
                host_kernel(number)(((nc*nl+127)//128,),(128,),(np.int32(nc),np.int32(nl),laytrop,fs,isv,wx,chi,oneminus,*[tabs[n] for n in sizes],taug,fracs))
                if tiny:
                    mag=fracs.view(np.uint32)&np.uint32(0x7fffffff)
                    if not np.any((mag>0)&(mag<0x00800000)):raise AssertionError(f'band {number} did not produce subnormal fractions')
                band(number,*map(xp.asarray,(laytrop,fs,isv,wx,chi)),{n:xp.asarray(v) for n,v in tabs.items()},actual,af,oneminus=oneminus)
                for name,got,expected in (('taug',actual,taug),('fracs',af,fracs)):
                    result=got.to_numpy();bad=np.flatnonzero(result.view(np.uint32)!=expected.view(np.uint32))
                    if bad.size:raise AssertionError(f'band {number} {name}: {bad.size} differences; first {bad[0]}')
                print(f'band {number} tiny={tiny} native host source parity PASS',flush=True)

if __name__=='__main__':main()
