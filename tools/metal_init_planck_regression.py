import dataclasses,json,pathlib,sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from woof.metal.runtime import MetalContext
original=MetalContext._submit_kernel_batch
def sync(self,*args,**kwargs):
    result=original(self,*args,**kwargs)
    if result is not None: result.wait()
    return result
MetalContext._submit_kernel_batch=sync
from woof.metal.array import MetalArrayModule
from woof.metal.suites import configuration
from woof.core.grid import make_vertical_coord,make_base_state
from woof.core.state import init_at_rest
from woof.core.physics import initialize_physics
from woof.core.radiation_carriers import (SURFACE_RADIATION_POLICY_REQUIRED,
    CARRIER_SOURCE_DECLARED_CONSTANT,CARRIER_SOURCE_EXTERNAL_ARRAY)
from woof.core.dycore import run_steps
from woof.core import rrtmgp as r
from woof.metal.rrtmgp_gas_family import _metal_gas_tables,submit_interpolation,submit_planck_sources
from tests.physics_reference.npref import np_rrtmgp_planck_sources
module=MetalArrayModule()
try:
    def state_for(cfg):
        coord=make_vertical_coord(cfg.nz)
        base=make_base_state(coord,lambda z:300.+.003*np.asarray(z),cfg.p_surf,cfg.ztop)
        state=init_at_rest(cfg,coord,base,array_module=module)
        state.qv[...] = np.float32(.005)
        return state
    cfg=dataclasses.replace(configuration((0,0,2,0,0,'periodic')),
        surface_radiation_policy=SURFACE_RADIATION_POLICY_REQUIRED)
    state=state_for(cfg)
    driver=initialize_physics(state,cfg,swdown=500.,glw=350.,ivgtyp=1,isltyp=1)
    assert driver.carriers.source('glw')==CARRIER_SOURCE_DECLARED_CONSTANT
    assert driver.carriers.source('swdown')==CARRIER_SOURCE_DECLARED_CONSTANT
    for name,value in [('swdown',500.),('glw',350.),('ivgtyp',1),('isltyp',1)]:
        np.testing.assert_array_equal(driver.fields[name].to_numpy(),value)
    run_steps(state,cfg,3)
    assert np.isfinite(driver.fields['hfx'].to_numpy()).all()
    noah_hfx=float(driver.fields['hfx'].to_numpy().mean())
    array_state=state_for(cfg)
    array_driver=initialize_physics(array_state,cfg,
        swdown=np.full((cfg.ny,cfg.nx),500.,np.float32),
        glw=np.full((cfg.ny,cfg.nx),350.,np.float32),ivgtyp=1,isltyp=1)
    assert array_driver.carriers.source('glw')==CARRIER_SOURCE_EXTERNAL_ARRAY
    assert array_driver.carriers.source('swdown')==CARRIER_SOURCE_EXTERNAL_ARRAY
    cfg=configuration((28,0,0,0,0,'periodic'))
    aerosol=state_for(cfg)
    initialize_physics(aerosol,cfg,glw=300.)
    counts={name:float(getattr(aerosol,name).to_numpy().max()) for name in ['nwfa','nifa']}
    assert all(value>0 for value in counts.values()),counts
    cfg=configuration((0,5,0,0,0,'periodic'),sfclay_option=5)
    mynn=state_for(cfg)
    driver=initialize_physics(mynn,cfg,glw=300.)
    np.testing.assert_array_equal(driver.fields['ustm'].to_numpy(),driver.fields['ust'].to_numpy())
    assert driver.fields['ust'].to_numpy().max()>0
    # Drive the previously unused standalone kernel directly, without CuPy.
    tables=r.load_gas_tables('lw'); table=_metal_gas_tables(module.context,tables)
    play=np.array([[90000.,50000.,15000.]],np.float32)
    plev=np.array([[100000.,70000.,30000.,5000.]],np.float32)
    tlay=np.array([[295.,270.,230.]],np.float32)
    tlev=np.array([[300.,285.,250.,215.]],np.float32)
    tsfc=np.array([301.],np.float32)
    vmr=np.zeros((1,3,tables.ngas+1),np.float32)
    vmr[:,:,tables.gas_index['h2o']]=.005
    vmr[:,:,tables.gas_index['o3']]=5e-6
    meta={name:module.zeros(play.shape,np.float32) for name in ['iatm','jt','jp','ftemp','fpress']}
    u=module.asarray
    submit_interpolation(dict(play=u(play.reshape(-1)),tlay=u(tlay.reshape(-1)),press_ref=table['press_ref'],temp_ref=table['temp_ref'],
        **{name+'_out':value.view((3,)) for name,value in meta.items()}),press_ref_trop=tables.press_ref_trop,
        ncell=3,ntemp=tables.ntemp,npres=tables.npres)
    arrays=dict(play=u(play),tlay=u(tlay),tlev=u(tlev),tsfc=u(tsfc),vmr=u(vmr),
        **{name+'_meta':value for name,value in meta.items()})
    for name in ['temp_ref','vmr_ref','flavor','gpoint_flavor','gpoint_bands','planck_fraction','totplnk']:
        arrays[name]=table[name]
    arrays.update(lay_source=module.empty((1,3,tables.ngpt),np.float32),
        lev_source=module.empty((1,4,tables.ngpt),np.float32),sfc_source=module.empty((1,tables.ngpt),np.float32))
    submit_planck_sources(arrays,dict(ncol=1,nlay=3,ngpt=tables.ngpt,ngas=tables.ngas,
        ntemp=tables.ntemp,npres=tables.npres,neta=tables.neta,nband=tables.nband,
        nplanck=tables.totplnk.shape[0],natm=tables.vmr_ref.shape[0],nbound=tables.flavor.shape[1]))
    reference=np_rrtmgp_planck_sources(tables,play,plev,tlay,tlev,tsfc,vmr)
    errors={}
    for name in ['lay_source','lev_source','sfc_source']:
        got=arrays[name].to_numpy(); want=getattr(reference,name)
        # Established standalone Planck bound from test_rrtmgp.py.
        np.testing.assert_allclose(got,want,rtol=8e-5,atol=2e-7)
        errors[name]=float(np.max(np.abs(got-want)))
    assert 'cupy' not in sys.modules
    print(json.dumps(dict(noah_categories=[1,1],swdown=500.,glw=350.,noah_hfx_after_3_steps=noah_hfx,
        aerosols=counts,mynn_ustm_seed=True,standalone_planck_errors=errors,synchronous=True)))
finally: module.context.close()
