"""Compare active v842 GWDO against its host-compiled selected CUDA source."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.ysu_gwdo_v842 import kernel_source
from woof.hex.mpasphys.metal_gwdo_v842 import gwdo_slot
from woof.hex.mpasphys.driver_v842 import resolve_namelist
from woof.hex.mpasphys.framework_backend_v842 import FrameworkPhysics, registry_source
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.metal_pools_v842 import MetalPoolViews
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def run(nz):
    nc=7; rng=np.random.default_rng(842+nz)
    shape=(nz,nc); z=np.arange(nz,dtype=np.float32)[:,None]
    expand=lambda x: np.broadcast_to(x,shape).copy().astype(np.float32)
    profiles={'u_p':expand(8+z*.3)+rng.uniform(-1,1,shape).astype(np.float32),
        'v_p':expand(3+z*.2)+rng.uniform(-1,1,shape).astype(np.float32),
        't_p':expand(290-z*4), 'qv_p':expand(.008-z*.0001),
        'pres_hydd_p':expand(95000-z*7000), 'pi_p':expand(.99-z*.03),
        'zmid_p':expand(150+z*350)}
    profiles['pres2_hydd_p']=np.broadcast_to((98500-np.arange(nz+1,dtype=np.float32)[:,None]*7000),(nz+1,nc)).copy()
    old_u=rng.uniform(-.001,.001,shape).astype(np.float32)
    old_v=rng.uniform(-.001,.001,shape).astype(np.float32)
    density=rng.uniform(.7,1.3,nc).astype(np.float32)
    surface={'var2d':rng.uniform(100,600,nc).astype(np.float32),'con':rng.uniform(.1,.8,nc).astype(np.float32)}
    surface.update({n:rng.uniform(-.5,.5,nc).astype(np.float32) for n in ('oa1','oa2','oa3','oa4')})
    surface.update({n:rng.uniform(.1,.9,nc).astype(np.float32) for n in ('ol1','ol2','ol3','ol4')})
    constants={'gravity':np.float32(9.80616),'cp':np.float32(1004.5),'rd':np.float32(287.),
               'fv':np.float32(np.float32(461.6)/np.float32(287.)-np.float32(1)),
               'pi':np.float32(np.pi)}
    names=('rublten_out','rvblten_out','dtaux3d','dtauy3d','dusfcg','dvsfcg','rubldiff','rvbldiff')
    expected={n:np.empty((nc,) if n in ('dusfcg','dvsfcg') else shape,np.float32) for n in names}
    invalid=np.zeros(1,np.int32)
    kernel=HostModule(f'mpas_v842_gwdo_{nz}',kernel_source(nz)).get_function('bl_ysu_gwdo_v842_f32')
    kernel((1,),(64,),(nz,nc,np.float32(60),np.float32(12000),*constants.values(),
        *(profiles[n] for n in ('u_p','v_p','t_p','qv_p','pres_hydd_p','pres2_hydd_p','pi_p','zmid_p')),
        old_u,old_v,density,*(surface[n] for n in ('var2d','con','oa1','oa2','oa3','oa4','ol1','ol2','ol3','ol4')),
        *(expected[n] for n in names),invalid))
    if invalid[0]: raise AssertionError('host source rejected levels')
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        prepared={n:xp.asarray(v) for n,v in profiles.items()}
        text="&physics\nconfig_physics_suite='convection_permitting'\n/\n"
        dims=dict(nCells=nc,nEdges=9,nVertices=5,nVertLevels=nz,nSoilLevels=4,maxEdges=6,maxEdges2=12,vertexDegree=3)
        layout=RegistryLayout(registry_source(),resolve_namelist(text),text,dims)
        native=FrameworkPhysics(layout=layout)
        needed={'tend_physics':('rublten','rvblten'),'diag_physics':tuple(n for n in names if n not in ('rublten_out','rvblten_out'))}
        for pool,fields in needed.items():
            for name in fields:
                row=next(r for r in layout.fields if r['pool']==pool and r['pool_name']==name)
                host=np.full(row['shape'],-13,np.float32,order='F')
                if name=='rublten': host[...,:nc]=old_u
                if name=='rvblten': host[...,:nc]=old_v
                native.put_field(pool,name,host,nlevels=row['nlevels'])
        views=MetalPoolViews(native,layout,context)
        tendencies={n:views.field('tend_physics',n) for n in needed['tend_physics']}
        diag={n:views.field('diag_physics',n) for n in needed['diag_physics']}
        pools=dict(prepared=prepared,mesh={'meshDensity':xp.asarray(density)},sfc_input={n:xp.asarray(v) for n,v in surface.items()},tend_physics=tendencies,diag_physics=diag,dt_pbl=60,config_len_disp=12000,constants=constants)
        result=gwdo_slot(SimpleNamespace(scheme='bl_ysu_gwdo',pools=pools))
        for n,ref in expected.items():
            actual=result[n].to_numpy(); different=np.count_nonzero(actual.view(np.uint32)!=ref.view(np.uint32))
            if different: raise AssertionError(f'{n}, nlev={nz}: {different} words, max {np.nanmax(np.abs(actual-ref))}')
        for name,target in (('rublten_out','rublten'),('rvblten_out','rvblten')):
            if not np.array_equal(tendencies[target].to_numpy().view(np.uint32),expected[name].view(np.uint32)):
                raise AssertionError(f'tendency {target} not published')
        for n,value in diag.items():
            if not np.array_equal(value.to_numpy().view(np.uint32),expected[n].view(np.uint32)):
                raise AssertionError(f'diagnostic {n} not published')
        checkpoint=views.snapshot()
        for value in (*tendencies.values(),*diag.values()): value.fill(-99)
        views.restore(checkpoint)
        restored=views.snapshot()
        for name,value in checkpoint.items():
            if not np.array_equal(value.view(np.uint32),restored[name].view(np.uint32)):
                raise AssertionError(f'GWDO checkpoint restore differs: {name}')
        native.close()
    nonzero=np.count_nonzero(expected['dtaux3d'])
    if not nonzero: raise AssertionError('fixture exercised no drag')
    print(f'GWDO nlev={nz} matches host-selected source bitwise, active drag cells={nonzero}')

if __name__=='__main__':
    for nlev in (5,9): run(nlev)
