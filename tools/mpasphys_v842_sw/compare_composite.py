"""Compare a full SW slot against host-compiled selected native source.

The host shim maps rounded PTX add/sub/mul to IEEE host operations. Its
accumulator uses the source's serial body, because the host shim cannot
emulate warp barriers. This is not a comparison against NVIDIA execution.
"""
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule, install
install()
import cupy as cp
from woof.core.rrtmg_sw import CudaSW, load_sw_tables, _pack_cuda_tables
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE, MPASShortwave, FIELD_NAMES, SCALAR_OUTPUTS
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables
from woof.hex.mpasphys.metal_rrtmg_sw_slot_v842 import MetalMpasShortwave
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def host_engine():
    tab=load_sw_tables();packed,defines=_pack_cuda_tables(tab)
    source=CUDA_SOURCE
    old='extern __shared__ real tile[];'
    assert source.count(old)==1
    source=source.replace(old,'real *tile = nullptr; if (level > nlayers) return;')
    old='pnicd + o1, pnifddir + o1, pnicddir + o1, true, tile);'
    assert source.count(old)==1
    source=source.replace(old,'pnicd + o1, pnifddir + o1, pnicddir + o1, false, tile);')
    # Serial accumulation reads the batch's level-major g-point layout.
    old=': (size_t)J + (size_t)(klev + 1) * iw;'
    assert source.count(old)==1
    source=source.replace(old, ': (size_t)J * RSW_NGPT + iw;')
    for op,intrinsic in (('add','__fadd_rn'),('sub','__fsub_rn'),('mul','__fmul_rn')):
        old=f'asm("{op}.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));'
        assert source.count(old)==1
        source=source.replace(old,f'r = {intrinsic}(a,b);')
    root=files('woof.core.kernels')
    source='#define __syncwarp() ((void)0)\n'+defines+root.joinpath('glibc_flt32.cuh').read_text()+root.joinpath('glibc_trig_flt32.cuh').read_text()+source
    module=HostModule('mpas_v842_sw_composite',source)
    class HostSW(MPASShortwave):
        def _k(self,name,shape=()):return module.get_function(name)
        def _check_kiss_pressure(self,play,icld,irng):
            if icld and np.any(play[:,0]*np.float32(100)<play[:,1]*np.float32(100)):raise ValueError('KISS pressure')
        def _wrapper_mcica(self,n,nl,cldovrlp,idcor,julday,layers,height,latitude):
            out=np.zeros((8,n,nl,112),np.float32);out[5]=1
            self._launch('mpas_sw_mcica',n,tuple(np.int32(v) for v in (n,nl,cldovrlp,idcor,julday,0,1))+(layers[0],layers[8],layers[9],layers[10],layers[11],height,latitude,layers[0],out))
            return out
        def rrtmg_sw_batched_device(self,*args,**kwargs):
            kwargs.pop('_stage_probe',None)
            return CudaSW.rrtmg_sw_batched_device(self,*args,**kwargs)
    e=HostSW.__new__(HostSW);e.cp=cp;e.tab=tab;e.tab_gpu=packed;e.ngb_gpu=np.asarray(tab.ngb,np.int32)
    e._laysolfr_spec_gpu=np.asarray(e._laysolfr_band_spec(),np.int32)
    e._scratch_by_stream={};e._chunk_by_nlayers={}
    return e


def verify_fortran(e):
    root=Path(__file__).resolve().parents[1]/'rrtmg_wrf461_oracle/sw_fixtures'
    for filename,cases in (('fixtures_shallow.npz',('c00301','c00302','c00303')),('fixtures_real.npz',('c00001','c00003','c00004')),('fixtures_real.npz',('c00002','c00006'))):
        fixture=np.load(root/filename);nc=len(cases)
        def get(c,key):return fixture[c+'/entry/'+key]
        def slab(key):return np.stack([get(c,key) for c in cases]).astype(np.float32)
        def scalar(key):return np.array([get(c,key) for c in cases],np.float32)
        def cloud(key):return np.stack([get(c,key).T for c in cases]).astype(np.float32)
        nlay=int(get(cases[0],'nlay'))
        out=e.rrtmg_sw_batched_device(nc,nlay,int(get(cases[0],'icld')),*[slab(k) for k in ('play','plev','tlay','tlev')],scalar('tsfc'),*[slab(k) for k in ('h2ovmr','o3vmr','co2vmr','ch4vmr','n2ovmr','o2vmr')],*[scalar(k) for k in ('asdir','asdif','aldir','aldif','coszen','adjes')],int(get(cases[0],'dyofyr')),scalar('scon'),*[int(get(cases[0],k)) for k in ('inflgsw','iceflgsw','liqflgsw')],*[cloud(k) for k in ('cldfmcl','taucmcl','ssacmcl','asmcmcl','fsfcmcl','ciwpmcl','clwpmcl','cswpmcl')],*[slab(k) for k in ('reicmcl','relqmcl','resnmcl')],column_chunk=nc,mcica_layout='column')
        for name,ref in (('swuflx','zbbfu'),('swdflx','zbbfd'),('swuflxc','zbbcu'),('swdflxc','zbbcd'),('swdkdir','zbbfddir')):
            expected=np.stack([fixture[c+'/spcout/'+ref] for c in cases])
            if not np.array_equal(out[name].view(np.uint32),expected.view(np.uint32)):raise AssertionError('host selected source differs from captured Fortran '+name+' '+filename)
        print(f'host selected source matches Fortran {filename}: {nc} columns',flush=True)


def main():
    nc,nz,nozn=5,6,2;rng=np.random.default_rng(842)
    fields=np.empty((12,nc,nz),np.float32)
    fields[0]=np.linspace(95000,20000,nz);fields[1]=np.linspace(280,230,nz);fields[2]=150;fields[3]=np.linspace(1,.7,nz)
    fields[4]=.003;fields[5]=.0001;fields[6]=.00003;fields[7]=.00002;fields[8]=rng.uniform(.1,.7,(nc,nz));fields[9]=.000012;fields[10]=.00003;fields[11]=.00005
    surf=np.zeros((7,nc),np.float32);surf[1]=[-180,-60,0,60,180];surf[2]=.2;surf[3]=280;surf[4]=[1,2,1,2,1];surf[6]=1
    pw=np.broadcast_to(np.linspace(100000,15000,nz+1,dtype=np.float32),(nc,nz+1)).copy();tw=np.broadcast_to(np.linspace(282,220,nz+1,dtype=np.float32),(nc,nz+1)).copy()
    pin=np.array([1,1100],np.float32);oz=np.full((nc,nozn),6e-6,np.float32);degrad=np.float32(np.pi/180)
    opts={'config_microp_re':True,'config_microp_scheme':'mp_thompson','config_radt_cld_overlap':'maximum_random','config_radt_cld_dcorrlen':'constant','config_o3climatology':True}
    e=host_engine();verify_fortran(e)
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);op=MetalMpasShortwave(ctx,initialize_tables(ctx))
        aliases=('pres_hyd_p','t_p','dz_p','pi_p','qvrad_p','qcrad_p','qirad_p','qsrad_p')
        prep={name:xp.asarray(fields[j].T.copy()) for j,name in enumerate(aliases)}
        prep['pres2_hyd_p']=xp.asarray(pw.T.copy());prep['t2_p']=xp.asarray(tw.T.copy())
        prep['qv_p']=xp.asarray(fields[4].T.copy());prep['nwfa_p']=xp.asarray(np.full((nz,nc),1e8,np.float32));prep['nifa_p']=xp.asarray(np.full((nz,nc),1e5,np.float32))
        diag={'cldfrac':xp.asarray(fields[8].T.copy()),'o3clim':xp.asarray(oz.T.copy()),'sfc_albedo':xp.asarray(surf[2])}
        diag.update({name:xp.asarray(fields[j].T.copy()) for j,name in enumerate(('re_cloud','re_ice','re_snow'),9)})
        diag.update({name:xp.zeros((nc,),np.float32) for name in SCALAR_OUTPUTS});diag['taod5502d']=xp.zeros((nc,),np.float32);diag['taod5503d']=xp.zeros((nz,nc),np.float32)
        surface={name:xp.asarray(surf[j]) for j,name in enumerate(('skintemp','xland','xice','snow'),3)}
        lat=surf[0]*degrad;lon=surf[1]*degrad
        lat[0]=np.float32(-1.0471976);lon[0]=np.float32(.7005143)
        pools=dict(prepared=prep,diag_physics=diag,sfc_input=surface,mesh={'latCell':xp.asarray(lat),'lonCell':xp.asarray(lon)},atm_input={'pin':xp.asarray(pin)},nCellsSolve=nc,nVertLevels=nz,constants={'degrad':degrad,'dpd':np.float32(360/365)},scalar_state={'dt_radtsw':900,'has_reqc':1,'has_reqi':1,'has_reqs':1},calendar={'curr_julday':180,'julday':180,'gmt':12},tend_physics={'rthratensw':xp.zeros((nz,nc),np.float32)})
        call=SimpleNamespace(slot='radiation_sw',scheme='rrtmg_sw',pools=pools,options=opts,time_seconds=0,backend='metal')
        times=np.empty(2,np.float32);solar=np.empty(2,np.float32)
        e._launch('mpas_sw_driver_times',1,(np.float32(0),np.float32(900),times))
        e._launch('mpas_sw_radconst',1,(np.float32(180),degrad,np.float32(360/365),solar))
        fieldmap={name:fields[j].T.copy() for j,name in enumerate(FIELD_NAMES)}
        sf={name:surf[j].copy() for j,name in enumerate(('skintemp','xland','xice','snow'),3)};sf['albedo']=surf[2].copy();sf['tsk']=sf.pop('skintemp')
        for aerosol in (False,True):
            opts['config_microp_scheme']='mp_thompson_aerosols' if aerosol else 'mp_thompson'
            cases=list(enumerate(('none','random','maximum_random','maximum','exponential','exponential_random')))
            cases.append((0,'none'))
            for case,(overlap,name) in enumerate(cases):
                cloud=fields[8].T.copy() if case<6 else np.zeros((nz,nc),np.float32)
                diag['cldfrac'][...]=xp.asarray(cloud);fieldmap['cldfra3d']=cloud
                opts['config_radt_cld_overlap']=name
                opts['config_radt_cld_dcorrlen']='latitude_varying' if overlap>=4 else 'constant'
                out=op.launch_slot(call)
                ref=e.radiation_sw_from_MPAS(fieldmap,sf,pw.T.copy(),tw.T.copy(),lat_cell=lat,lon_cell=lon,microp_scheme=opts['config_microp_scheme'],config_microp_re=True,nwfa=np.full((nz,nc),1e8,np.float32),nifa=np.full((nz,nc),1e5,np.float32),ht=np.zeros(nc,np.float32),aerosol_qv=fields[4].T.copy(),xtime=times[0],radt=times[1],declin=solar[0],solcon=solar[1],gmt=12,julday=180,degrad=degrad,icloud=1,cldovrlp=overlap,idcor=int(overlap>=4),o3input=2,pin=pin,o3clim=oz.T.copy(),has_reqc=1,has_reqi=1,has_reqs=1,column_chunk=nc)
                if not (np.any(ref['coszr']>0) and np.any(ref['coszr']<0) and np.any(ref['coszr']==0) and np.any(ref['gsw']>0)):
                    raise AssertionError('fixture must include active day, night and exact-zero columns')
                pairs=[(name,diag[name].to_numpy(),ref[name]) for name in SCALAR_OUTPUTS]
                pairs.append(('rthratensw',pools['tend_physics']['rthratensw'].to_numpy(),ref['rthratensw']))
                for j,name in enumerate(('swupflx','swupflxc','swdnflx','swdnflxc')):pairs.append((name,out['profiles'][j].to_numpy().T.copy(),ref[name]))
                if aerosol:pairs.extend((name,diag[name].to_numpy(),ref[name]) for name in ('taod5502d','taod5503d'))
                for name,a,b in pairs:
                    bad=np.flatnonzero(a.view(np.uint32)!=b.view(np.uint32))
                    if bad.size:raise AssertionError(f'aerosol={aerosol} overlap={overlap} {name}: {bad.size} mismatches; first={bad[0]} actual={a.flat[bad[0]]} expected={b.flat[bad[0]]}')
                print(f'composite aerosol={aerosol} overlap={overlap} clear={case==6} PASS',flush=True)

if __name__=='__main__':main()
