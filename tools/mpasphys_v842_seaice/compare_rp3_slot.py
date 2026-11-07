"""Corrected sea-ice source composition and slot replay."""
import ctypes
from types import SimpleNamespace
from importlib.resources import files
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.metal_mynn_word_probe import plain_arithmetic
from woof.hex.mpasphys.seaice_v842 import _CORE_SOURCE, FIELDS, _DIAG_FIELDS, _SFC_FIELDS, OPTIONAL_FIELDS, _SEA_FLUX_FIELDS
from woof.hex.mpasphys.metal_seaice_v842 import MetalMpasSeaice
from woof.hex.mpasphys.sf_mynn_v842 import _cuda_source as sf_source, FIELDS as SF_FIELDS
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main(nsoil):
    n=7
    header=plain_arithmetic(files('woof.core.kernels').joinpath('mynn_libm.cuh').read_text())
    source=f'#define NSOLD {nsoil}\ntypedef float real;\n'+header+_CORE_SOURCE
    host=HostModule('mpas_v842_seaice',source)
    data=np.zeros((len(FIELDS),n),np.float32)
    def put(name,value): data[FIELDS.index(name)]=np.asarray(value,np.float32)
    for name,value in dict(t3d=[255,265,270,272,275,280,268],qv3d=.002,p8w3d=99000,dz8w=30,glw=240,qgh=.003,gsw=100,snoalb2d=.8,xice=[0,.05,.2,.5,.8,1,1.1],rib=.1,chs=.015,emiss=.98,albedo=.4,z02d=.001,snow=15,tsk=265,snowc=.8,snowh2d=.06,cqs2=.014,qsfc=.002,cpm=1000,sr_input=.5,rainncv=.001,raincv=.002).items():put(name,value)
    for j,name in enumerate(('cpm_sea','chs_sea','chs2_sea','cqs2_sea','hfx_sea','lh_sea','qfx_sea','qgh_sea','qsfc_sea')):put(name, .01*(j+1))
    soil=np.broadcast_to(np.linspace(263,270,nsoil,dtype=np.float32)[:,None],(nsoil,n)).copy()
    pressure=np.stack((data[FIELDS.index('p8w3d')].copy(),np.full(n,97000,np.float32)))
    rows=[data[i] for i in range(len(FIELDS))];rows[FIELDS.index('p8w3d')]=pressure;rows[FIELDS.index('tslb')]=soil
    ptr=(ctypes.c_void_p*len(rows))(*(x.ctypes.data for x in rows)); raw=ctypes.cast(ptr,ctypes.c_void_p)
    err=np.zeros(n,np.int32);last=np.full(1,-1,np.int32);state=np.array([11,12],np.int32)
    def invoke(name,args): host.get_function(name)._fn(1,1,1,128,1,1,*args)
    original_skin=data[FIELDS.index('tsk')].copy()
    sst=np.linspace(275,291,n,dtype=np.float32)
    invoke('seaice_from_MPAS',(raw,n,np.float32(.02),np.float32(.65),np.float32(.001),np.float32(3),1,1))
    corrected=HostModule('mpas_v842_seaice_rp3_source',plain_arithmetic(sf_source(frozenset({'RP3'}))))
    sf=np.zeros((len(SF_FIELDS),n),np.float32); sf[SF_FIELDS.index('tsk')]=original_skin
    sea=np.zeros_like(sf); ice=data[FIELDS.index('xice')].copy()
    corrected.get_function('sf_mynn_from_mpas')((1,),(128,),(sf,sea,np.ones(n,np.float32),sst,ice,np.float32(3000),np.float32(.02),np.int32(1),np.int32(n)))
    active=(ice>=np.float32(.02)) & (ice<=np.float32(1))
    data[FIELDS.index('tsk'),active]=sf[SF_FIELDS.index('tsk'),active]
    data[FIELDS.index('tsk_sea'),active]=sea[SF_FIELDS.index('tsk'),active]
    invoke('seaice_noah',(raw,n,nsoil,np.float32(60),0,np.float32(.65),0,np.float32(3),0,np.float32(1e10),np.float32(.001),np.float32(.02),0,7,ctypes.c_void_p(err.ctypes.data),ctypes.c_void_p(last.ctypes.data)))
    invoke('seaice_module_state',(ctypes.c_void_p(last.ctypes.data),ctypes.c_void_p(state.ctypes.data),9,4))
    invoke('seaice_to_MPAS',(raw,n,np.float32(.02)))
    if np.any(err):raise AssertionError(f'host fatal {err}')
    # Reconstruct the identical starting values for device execution.
    initial=np.zeros_like(data)
    for name,value in dict(t3d=[255,265,270,272,275,280,268],qv3d=.002,p8w3d=99000,dz8w=30,glw=240,qgh=.003,gsw=100,snoalb2d=.8,xice=[0,.05,.2,.5,.8,1,1.1],rib=.1,chs=.015,emiss=.98,albedo=.4,z02d=.001,snow=15,tsk=265,snowc=.8,snowh2d=.06,cqs2=.014,qsfc=.002,cpm=1000,sr_input=.5,rainncv=.001,raincv=.002).items():initial[FIELDS.index(name)]=value
    for j,name in enumerate(('cpm_sea','chs_sea','chs2_sea','cqs2_sea','hfx_sea','lh_sea','qfx_sea','qgh_sea','qsfc_sea')):initial[FIELDS.index(name)]=.01*(j+1)
    with MetalContext(batch_kernel_calls=False) as ctx:
        orig=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=orig(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        d=xp.asarray(initial); s=xp.asarray(np.broadcast_to(np.linspace(263,270,nsoil,dtype=np.float32)[:,None],(nsoil,n)).copy());p=xp.asarray(pressure[1].copy());e=xp.zeros((n,),np.int32);ran=xp.zeros((n,),np.int32);st=xp.asarray(np.array([11,12],np.int32))
        par={k:np.int32(v) for k,v in dict(n=n,nsoil=nsoil,albedo_opt=0,thickness_opt=0,snowdepth_opt=0,frpcpn=0,optional_noah=7,microp=1,convection=1).items()}
        par.update({k:np.float32(v) for k,v in dict(dt=60,threshold=.02,albedo_default=.65,thickness_default=3,snowdepth_max=1e10,snowdepth_min=.001).items()})
        _launch('hexv842_seaice_from_f32',n,ctx,dict(fields=d),par)
        _launch('hexv842_mpas_seaice_rp3_f32',n,ctx,dict(fields=d,sst=xp.asarray(sst),skin=xp.asarray(original_skin),ice=xp.asarray(ice)),dict(n=np.int32(n),sea_index=np.int32(FIELDS.index('tsk_sea')),ice_index=np.int32(FIELDS.index('tsk')),threshold=np.float32(.02)))
        _launch('hexv842_seaice_column_f32',n,ctx,dict(fields=d,soil=s,p8w2=p,errors=e,ran=ran),par)
        _launch('hexv842_seaice_state_i32',1,ctx,dict(ran=ran,state=st),dict(n=np.int32(n),i_start=np.int32(9),j_index=np.int32(4)))
        _launch('hexv842_seaice_to_f32',n,ctx,dict(fields=d),par)
        actual=d.to_numpy(); bad=0
        for i,name in enumerate(FIELDS):
            if name in ('p8w3d','tslb'):continue
            mismatch=np.count_nonzero(actual[i].view(np.uint32)!=data[i].view(np.uint32));bad+=mismatch
            if mismatch:print(name,'mismatched',mismatch,actual[i],data[i])
        for name,got,ref in [('soil',s.to_numpy(),soil),('errors',e.to_numpy(),err),('state',st.to_numpy(),state)]:
            mismatch=np.count_nonzero(got.view(np.uint32)!=ref.view(np.uint32));bad+=mismatch
            if mismatch:print(name,'mismatched',mismatch,got,ref)
        print('active native sea-ice word differences:',bad,flush=True)
        if bad:raise AssertionError('sea-ice source mismatch')
        diag={src:xp.asarray(initial[FIELDS.index(dest)].copy()) for src,dest in _DIAG_FIELDS.items()}
        diag.update({name:xp.asarray(initial[FIELDS.index(name)].copy()) for name in OPTIONAL_FIELDS})
        diag.update(sr=xp.asarray(initial[FIELDS.index('sr_input')].copy()),rainncv=xp.asarray(initial[FIELDS.index('rainncv')].copy()),raincv=xp.asarray(initial[FIELDS.index('raincv')].copy()))
        surface={src:xp.asarray(initial[FIELDS.index(dest)].copy()) for src,dest in _SFC_FIELDS.items() if dest!='tslb'}
        surface['sst']=xp.asarray(sst)
        surface['tslb']=xp.asarray(np.broadcast_to(np.linspace(263,270,nsoil,dtype=np.float32)[:,None],(nsoil,n)).copy())
        shared={name:xp.asarray(np.stack((initial[FIELDS.index(name)],initial[FIELDS.index(name)]))) for name in ('t3d','qv3d','p8w3d','dz8w')}
        shared['p8w3d']=xp.asarray(np.stack((initial[FIELDS.index('p8w3d')],pressure[1])))
        shared.update({name:xp.asarray(initial[FIELDS.index(name)].copy()) for name in _SEA_FLUX_FIELDS})
        opts={'config_frac_seaice':True,'config_microp_scheme':'mp_thompson','config_convection_scheme':'cu_gf'}
        call=SimpleNamespace(slot='seaice',scheme='sf_noah',dt=60,options=opts,corrections=frozenset({'RP3'}),pools={'diag_physics':diag,'sfc_input':surface,'seaice_shared':shared})
        op=MetalMpasSeaice(ctx,module_locations=[11,12],i_start=9,j_index=4)
        saved=op.restart_state();op.launch_slot(call)
        for name in ('acsnom','acsnow','chs','chs2','cpm','cqs2','qsfc','qgh','hfx','qfx','lh','sfc_albedo','sfc_emiss','sfcrunoff','z0','znt','grdflx'):
            if not np.array_equal(diag[name].to_numpy().view(np.uint32),data[FIELDS.index(_DIAG_FIELDS[name])].view(np.uint32)):raise AssertionError('pool '+name)
        for name in ('snow','snowc','snowh','skintemp'):
            if not np.array_equal(surface[name].to_numpy().view(np.uint32),data[FIELDS.index(_SFC_FIELDS[name])].view(np.uint32)):raise AssertionError('pool '+name)
        if not np.array_equal(surface['tslb'].to_numpy().view(np.uint32),soil.view(np.uint32)):raise AssertionError('pool tslb')
        for name in OPTIONAL_FIELDS:
            if not np.array_equal(diag[name].to_numpy().view(np.uint32),data[FIELDS.index(name)].view(np.uint32)):raise AssertionError('pool '+name)
        if not np.array_equal(op.state.to_numpy(),state):raise AssertionError('module locations')
        op.restore_restart_state(saved)
        if not np.array_equal(op.state.to_numpy(),[11,12]):raise AssertionError('module location restore')
        # Replay a second active step from a pool and module-state checkpoint.
        checkpoint={('d',k):v.to_numpy().copy() for k,v in diag.items()}
        checkpoint.update({('s',k):v.to_numpy().copy() for k,v in surface.items()})
        checkpoint_state=op.restart_state()
        op.launch_slot(call)
        expected={('d',k):v.to_numpy().copy() for k,v in diag.items()}
        expected.update({('s',k):v.to_numpy().copy() for k,v in surface.items()})
        expected_state=op.restart_state()
        for (pool,name),value in checkpoint.items():
            (diag if pool=='d' else surface)[name][...]=xp.asarray(value)
        op.restore_restart_state(checkpoint_state)
        op.launch_slot(call)
        for (pool,name),value in expected.items():
            actual=(diag if pool=='d' else surface)[name].to_numpy()
            if not np.array_equal(actual.view(np.uint32),value.view(np.uint32)):raise AssertionError('restart replay '+name)
        if not np.array_equal(op.state.to_numpy(),expected_state['module_locations']):raise AssertionError('restart replay locations')
        # Source fatal 1: snow water equivalent exceeds snow depth.
        bad_snow=surface['snow'].to_numpy();bad_depth=surface['snowh'].to_numpy()
        bad_snow[1]=99.;bad_depth[1]=.001
        surface['snow'][...]=xp.asarray(bad_snow);surface['snowh'][...]=xp.asarray(bad_depth)
        before_state=op.restart_state();before_tsk=surface['skintemp'].to_numpy()
        try:op.launch_slot(call)
        except RuntimeError as ex:
            if 'fatal status 1' not in str(ex):raise
        else:raise AssertionError('missing fatal status')
        if not np.array_equal(before_tsk,surface['skintemp'].to_numpy()) or not np.array_equal(before_state['module_locations'],op.state.to_numpy()):raise AssertionError('fatal mutated pools or locations')
        opts['config_frac_seaice']=False
        for name in _SEA_FLUX_FIELDS:del shared[name]
        before=surface['skintemp'].to_numpy()
        try:op.launch_slot(call)
        except RuntimeError as ex:
            if 'unallocated ocean flux' not in str(ex):raise
        else:raise AssertionError('missing ocean flux guard')
        if not np.array_equal(before,surface['skintemp'].to_numpy()):raise AssertionError('guard mutated surface')
        # Even stale provided buffers cannot make a frac=false iced call defined.
        shared.update({name:xp.asarray(initial[FIELDS.index(name)].copy()) for name in _SEA_FLUX_FIELDS})
        try:op.launch_slot(call)
        except RuntimeError as ex:
            if 'fractional sea ice disabled' not in str(ex):raise
        else:raise AssertionError('disabled fraction accepted stale handoff')
        for name in _SEA_FLUX_FIELDS:del shared[name]
        surface['xice'][...]=xp.asarray(np.zeros(n,np.float32))
        prior=op.restart_state();op.launch_slot(call)
        if not np.array_equal(prior['module_locations'],op.state.to_numpy()):raise AssertionError('ice-free state changed')
        print('pool publication, replay, fatal state and missing-ocean guard pass',flush=True)

if __name__=='__main__':
    main(4)
