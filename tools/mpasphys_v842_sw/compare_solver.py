"""Compare the sunlit solver with independent captured Fortran SW columns."""
from pathlib import Path
import numpy as np
import sys
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables
from woof.hex.mpasphys.metal_rrtmg_sw_solver_v842 import solve_sunlit
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'shallow'
    filename,cases={'shallow':('fixtures_shallow.npz',('c00301','c00302','c00303')),'real5':('fixtures_real.npz',('c00001','c00003','c00004')),'real2':('fixtures_real.npz',('c00002','c00006'))}[mode]
    source=Path(__file__).resolve().parents[1]/'rrtmg_wrf461_oracle/sw_fixtures'/filename
    fixture=np.load(source);nc=len(cases);nl=int(fixture[cases[0]+'/entry/nlay'])
    def get(case,key):return fixture[case+'/entry/'+key]
    layers=np.zeros((15,nc,nl),np.float32)
    for index,name in enumerate(('play','tlay','h2ovmr','o3vmr','co2vmr','ch4vmr','n2ovmr','o2vmr')):
        for c,case in enumerate(cases):layers[index,c]=get(case,name)
    for index,name in ((12,'reicmcl'),(13,'relqmcl'),(14,'resnmcl')):
        for c,case in enumerate(cases):layers[index,c]=get(case,name)
    interfaces=np.empty((2,nc,nl+1),np.float32)
    for c,case in enumerate(cases):interfaces[0,c]=get(case,'plev');interfaces[1,c]=get(case,'tlev')
    mcica=np.empty((8,nc,nl,112),np.float32)
    for i,name in enumerate(('cldfmcl','ciwpmcl','clwpmcl','cswpmcl','taucmcl','ssacmcl','asmcmcl','fsfcmcl')):
        for c,case in enumerate(cases):mcica[i,c]=get(case,name).T
    cosine=np.array([get(c,'coszen') for c in cases],np.float32)
    albedo=np.array([get(c,'asdir') for c in cases],np.float32)
    scon=np.array([get(c,'scon') for c in cases],np.float32)
    if not np.all(scon==scon[0]):raise AssertionError('fixture needs a common solar constant')
    controls=np.array([0,1,0,scon[0]],np.float32)
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        tables=initialize_tables(ctx)
        out=solve_sunlit(tables=tables,layers=xp.asarray(layers),interfaces=xp.asarray(interfaces),cosine=xp.asarray(cosine),albedo=xp.asarray(albedo),controls=xp.asarray(controls),mcica=xp.asarray(mcica),cloudflag=int(get(cases[0],'inflgsw')),iceflag=int(get(cases[0],'iceflgsw')),_stage_probe=True)
        for name,group,ref in [('pdp','inatm','pdp'),('coldry','inatm','coldry'),('jp','setcoef','jp'),('colh2o','setcoef','colh2o'),('taug','taumol','taug'),('taur','taumol','taur'),('sflux','taumol','sfluxzen'),('zinc','rt','zincflx')]:
            key=cases[0]+'/'+group+'/'+ref
            if key in fixture:
                got=out['_debug'][name].to_numpy()[0];want=fixture[key]
                if got.shape==want.shape:
                    ix=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32));print('stage',name,'diff',len(ix),'max',np.max(np.abs(got.astype(np.float64)-want.astype(np.float64))),flush=True)
                    if name=='sflux': print('sflux examples',[(int(j),float(got[j]),float(want[j])) for j in ix[:12]],flush=True)
        for name,ref in [('zcd','zcd'),('zcu','zcu'),('zfd','zfd'),('zfu','zfu'),('ztdbt_nodel','ztdbt_nodel'),('ztdbtc_nodel','ztdbtc_nodel')]:
            available=[j for j,c in enumerate(cases) if c+'/rt/'+ref in fixture]
            if not available:continue
            got=out['_debug'][name].to_numpy()[available];want=np.stack([fixture[cases[j]+'/rt/'+ref] for j in available])
            bad=np.count_nonzero(got.view(np.uint32)!=want.view(np.uint32))
            bits=got.view(np.uint32)&np.uint32(0x7fffffff);sub=np.count_nonzero((bits>0)&(bits<0x00800000))
            print(f'intermediate {name}: {bad} differences, {sub} subnormal words',flush=True)
            if bad:raise AssertionError(f'independent Fortran {name} mismatch')
        for name,ref in [('swuflx','zbbfu'),('swdflx','zbbfd'),('swuflxc','zbbcu'),('swdflxc','zbbcd'),('swdkdir','zbbfddir')]:
            got=out[name].to_numpy();want=np.stack([fixture[c+'/spcout/'+ref] for c in cases])
            bad=np.count_nonzero(got.view(np.uint32)!=want.view(np.uint32))
            if bad:
                print(name,'mismatches',int(bad),'max abs',float(np.max(np.abs(got-want))),flush=True)
                raise AssertionError('independent Fortran solver mismatch')
    print(f'MPAS sunlit solver matches independent Fortran flux words for {nc} captured {mode} columns',flush=True)

if __name__=='__main__':main()
