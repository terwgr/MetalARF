"""CPU control-flow contracts; numerical/device leaves are explicit probes.

No executed-Metal/CUDA parity is implied. Run this file alone, not GPU suites.
"""
from types import SimpleNamespace as NS
from datetime import datetime, timezone
import numpy as np
import pytest
import importlib
from contextlib import nullcontext
from dataclasses import replace
RunConfig = importlib.import_module("woof.config").RunConfig
physics = importlib.import_module("woof.core.physics")
dycore = importlib.import_module("woof.core.dycore")
MicrophysicsDiagnostics = importlib.import_module("woof.core.microphysics").MicrophysicsDiagnostics
microphysics_scratch_slots = importlib.import_module("woof.core.physics_inventory").microphysics_scratch_slots


class HostMetal(np.ndarray):
    """Boundary storage double, preserving canonical identity and prefix views."""
    def __new__(cls, value, context):
        obj = np.asarray(value).view(cls)
        obj.buffer = NS(context=context)
        obj.byte_offset = 0
        return obj

    def __array_finalize__(self, parent):
        self.buffer = getattr(parent, 'buffer', None)
        self.byte_offset = getattr(parent, 'byte_offset', 0)

    def view(self, shape=None, dtype=None, *, byte_offset=None):
        # Only the driver's contiguous Metal prefix-view API is modeled.
        if isinstance(shape, tuple):
            offset = self.byte_offset if byte_offset is None else byte_offset
            flat = np.asarray(self).reshape(-1)
            result = flat[(offset-self.byte_offset)//self.dtype.itemsize:]
            return HostMetal(result[:int(np.prod(shape))].reshape(shape),
                             self.buffer.context)
        return super().view(shape) if shape is not None else super().view()

    def to_numpy(self):
        return np.asarray(self).copy()


class HostModule:
    __woof_metal_array_module__ = True
    def __init__(self, context=None):
        self.context = context if context is not None else NS(defer_kernel_submissions=lambda *a: nullcontext())
    def asarray(self, value, dtype=np.float32, **kw):
        return HostMetal(np.array(value, dtype=dtype), self.context)
    def zeros(self, shape, dtype=np.float32):
        return self.asarray(np.zeros(shape, dtype), dtype)


class TinyState(NS):
    def scratch(self, shape, name, dtype=np.float32):
        if name not in self._scratch:
            self._scratch[name] = self._array_module.zeros(shape, dtype)
        return self._scratch[name]


def cfg(**kw):
    args = dict(nx=3, ny=3, nz=3, dx=1000., dy=1000., ztop=1000.,
                dt=15., run_seconds=90., moist=True, mp_physics=8,
                ra_lw_physics=0, ra_sw_physics=0)
    args.update(kw)
    return RunConfig(**args)


def state():
    xp = HostModule()
    s = TinyState(_array_module=xp, _metal_setup_state=True, _scratch={},
                  elapsed_seconds=0., p_top=10000., has_msf=False)
    for name in ('p', 'thp', 'thp0', 'qv', 'qc', 'qi', 'ni', 'qs', 'qg',
                 'qr', 'nr', 'h_diabatic', 'effc', 'effi', 'effs'):
        setattr(s, name, xp.zeros((3,3,3)))
    for name in ('mup','mup0','mub2d'):
        setattr(s, name, xp.zeros((3,3)))
    for name in ('u','u0','ru_t'):
        setattr(s, name, xp.zeros((3,3,4)))
    for name in ('v','v0','rv_t'):
        setattr(s, name, xp.zeros((3,4,3)))
    for name in ('w','w0','rw_t','phb','php'):
        setattr(s, name, xp.zeros((4,3,3)))
    for name in ('qv','qc','qr','qi','qs','qg','nr','ni'):
        setattr(s,name+'0',getattr(s,name).copy())
    s.thb = xp.asarray(np.full((3,3,3), 300.))
    s.rth_t = xp.zeros((3,3,3))
    s.c1h = s.c2h = s.c1f = s.c2f = xp.zeros((3,))
    return s


@pytest.fixture(autouse=True)
def cpu_boundary(monkeypatch):
    array = importlib.import_module("woof.metal.array")
    runtime = importlib.import_module("woof.metal.runtime")
    bandwidth_family = importlib.import_module("woof.metal.bandwidth_family")
    def forbidden(*a, **kw):
        raise AssertionError('unexpected real Metal device/launch')
    monkeypatch.setattr(runtime.MetalContext, '__init__', forbidden)
    monkeypatch.setattr(runtime.MetalContext, 'submit_kernel', forbidden)
    monkeypatch.setattr(runtime.MetalContext, 'buffer', forbidden)
    monkeypatch.setattr(dycore, 'get_kernel', forbidden)
    monkeypatch.setattr(importlib.import_module('woof.core.microphysics'), 'cp', None)
    monkeypatch.setattr(array, 'MetalArray', HostMetal)
    monkeypatch.setattr(array, 'MetalArrayModule', HostModule)
    monkeypatch.setattr(physics, 'cp', None)
    monkeypatch.setattr(dycore, 'cp', None)
    def add(src, dst):
        if not isinstance(src, HostMetal) or not isinstance(dst, HostMetal):
            return False
        dst[...] += src
        return True
    monkeypatch.setattr(bandwidth_family, 'add_array', add)
    monkeypatch.setattr(dycore, 'add_array', add)
    monkeypatch.setattr(dycore, 'prepare_add_arrays', lambda *a: None)


def initialized(s, c, **kw):
    return physics.initialize_physics(s, c, glw=250., swdown=400., **kw)


def noah_boundary(monkeypatch, delivered):
    monkeypatch.setattr(physics.noah_forcing, 'device_arrays', lambda *a: True)
    def forcing(f, **kw):
        f['sr'][...] = kw['sr_scheme']
        f['rainbl'][...] += kw['pending']
    monkeypatch.setattr(physics.noah_forcing, 'forcing', forcing)
    def clear(rain, pending):
        rain[...] = pending[...] = 0.
    monkeypatch.setattr(physics.noah_forcing, 'rain_clear', clear)
    monkeypatch.setattr(physics, 'launch_noah', lambda f,*a,**kw:
        delivered.append((f['rainbl'].copy(), f['sr'].copy(), kw['frpcpn'],
                          f.get('swdown',np.zeros((3,3))).copy(),
                          f.get('glw',np.zeros((3,3))).copy())))


def atmosphere(s):
    xp = s._array_module
    return {name: xp.asarray(np.full(shape, val)) for name, shape, val in (
        ('p_interface',(4,3,3),100000.), ('temperature',(3,3,3),280.),
        ('qv',(3,3,3),.01), ('dz',(3,3,3),50.))}


@pytest.mark.parametrize('mp,gfs', [(8,False), (10,True)])
def test_precipitation_multiple_calls_and_once(monkeypatch, mp, gfs):
    s=state(); d=physics.PhysicsDriver.__new__(physics.PhysicsDriver)
    d.state=s; d.mp_physics=mp; d._metal_gfs_profile=gfs
    d.microphysics_updates=0
    d.microphysics=MicrophysicsDiagnostics(**{
        name:s.scratch((3,3),slot) for name,slot in microphysics_scratch_slots(mp)})
    if gfs:
        morrison_family = importlib.import_module("woof.metal.morrison_family")
        monkeypatch.setattr(morrison_family,'submit_morrison_validate_diagnostics', lambda **kw:0)
        d._morrison_validation_status=s._array_module.zeros((1,),np.uint32)
        d._sr_roundoff_upper=1.
    d._pending_rainbl=s._array_module.zeros((3,3))
    d.fields={name:s._array_module.zeros((3,3)) for name in
        ('tsk','br','sfcprs','sfctmp','qv1','dz8w1','rib','sr','rainbl')}
    d.fields['rainbl'][...] = .125
    d.noah_params=None; d.bldt_seconds=30.; d.noah_usemonalb=False
    d.noah_rdlai2d=False; d.noah_opt_thcnd=1; d.urban_coupler=None
    delivered=[]; noah_boundary(monkeypatch, delivered)
    d.microphysics.sr[...] = .75
    wrong=replace(d.microphysics,rainncv=d.microphysics.rainncv.copy())
    with pytest.raises(ValueError,match='canonical'):
        d.accept_microphysics(wrong,dt=15.)
    for increment in (.25,.5):
        d.microphysics.rainncv[...] = increment
        d.accept_microphysics(d.microphysics,dt=15.)
    np.testing.assert_array_equal(d._pending_rainbl,.75)
    # Real scheduling predicate: skipped event must not consume pending rain.
    assert not physics._surface_pbl_step_due(3,2, .5, None)
    np.testing.assert_array_equal(d._pending_rainbl,.75)
    for step in (2,4):
        assert physics._surface_pbl_step_due(step,2,.5,None)
        d._run_noah(atmosphere(s),cfg(),step)
    np.testing.assert_array_equal(delivered[0][0],.875)
    np.testing.assert_array_equal(delivered[0][1],.75)
    assert delivered[0][2]
    np.testing.assert_array_equal(delivered[1][0],0.)
    np.testing.assert_array_equal(d.microphysics.rainncv,.5)
    assert d.microphysics_updates==2
    # Reversible mutation: missing numerical boundary accumulation is detected.
    bandwidth_family = importlib.import_module("woof.metal.bandwidth_family")
    monkeypatch.setattr(bandwidth_family,'add_array',lambda *a:True)
    d.accept_microphysics(d.microphysics,dt=15.)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(d._pending_rainbl,.5)


def test_mp_only_no_surface_should_accept(monkeypatch):
    s=state(); d=initialized(s,cfg())
    assert d._pending_rainbl is None
    d.microphysics.rainncv[...] = .25
    # Expected public mp-only contract (initialize_physics docstring).
    error=None
    try:
        d.accept_microphysics(d.microphysics,dt=15.)
    except NotImplementedError as observed:
        error=str(observed)
    assert error is None, error
    assert d.microphysics_updates==1


class RRTMGPRadiation:
    longwave=True; shortwave=True; validation_mode='fused'; publishes_olr=True
    def __init__(self):
        self.latitude_deg=np.zeros((3,3),np.float32)
        self.longitude_deg=self.latitude_deg.copy()
        self.start_time=datetime(2026,10,13,tzinfo=timezone.utc)


@pytest.mark.parametrize('swint',[0,1])
def test_initialized_radiation_carriers_and_swint(monkeypatch,swint):
    s=state(); c=cfg(ra_lw_physics=4,ra_sw_physics=4,swint_opt=swint,
                     sf_surface_physics=2,sf_sfclay_physics=1,bl_pbl_physics=1)
    d=initialized(s,c,radiation=RRTMGPRadiation())
    np.testing.assert_array_equal(d.fields['swdown'],400.)
    np.testing.assert_array_equal(d.fields['glw'],250.)
    seen=[]
    noah_boundary(monkeypatch,seen)
    d._run_noah(atmosphere(s),c,1)
    np.testing.assert_array_equal(seen[0][3],400.)
    np.testing.assert_array_equal(seen[0][4],250.)
    if swint:
        assert d.swint is not None, 'admitted swint_opt=1 must attach interpolation'
    else:
        assert d.swint is None


@pytest.mark.parametrize('km',[4,1])
def test_fixed_tendency_real_order_two_steps(monkeypatch,km):
    smag2d_family = importlib.import_module("woof.metal.smag2d_family")
    s=state(); c=cfg(km_opt=km, bl_pbl_physics=1,open_x=True)
    events=[]
    # Real specs, outer clear, km launcher, compute order and application.
    def deform(st,c,d11,d22,d12,**kw):
        events.append('deform')
        for arr in (d11,d22,d12): np.testing.assert_array_equal(arr,0.)
        for arr,value in zip((d11,d22,d12),(2.,3.,4.)): arr[...] = value
    def coefficients(st,c,d11,d22,d12,km,kh,**kw):
        events.append('K')
        np.testing.assert_array_equal(d11,2.)
        np.testing.assert_array_equal(d22,3.)
        np.testing.assert_array_equal(d12,4.)
        km[...] = 5.; kh[...] = 15.
    monkeypatch.setattr(smag2d_family,'submit_wrf_smag_deform',deform)
    monkeypatch.setattr(smag2d_family,'submit_wrf_smag2d_km',coefficients)
    monkeypatch.setattr(smag2d_family,'submit_wrf_smag_km_bc',lambda *a,**kw:events.append('Kbc'))
    def hd(st,c,f,k,out,**kw):
        events.append('hd'+kw['stagger'])
        np.testing.assert_array_equal(out,0.)
        if kw['stagger'] in ('x','y'):
            for arr,value in zip(kw['deformation'],(2.,3.,4.)):
                np.testing.assert_array_equal(arr,value)
        out[...] = 7.
    monkeypatch.setattr(dycore,'launch_wrf_smag2d_hd',hd)
    def strip(arr,c,width,**kw):
        events.append('strip')
        arr[..., :width] = 0.
        arr[..., -width:] = 0.
    monkeypatch.setattr(smag2d_family,'submit_zero_boundary_strips',strip)
    for step in range(2):
        for name in ('ru_t','rv_t','rw_t','rth_t'): getattr(s,name)[...] = 0.
        dycore.prepare_fixed_tendencies(s,c)
        for stage in range(3): dycore.add_fixed_dry_tendencies(s,c)
        for name in ('ru_t','rv_t','rw_t','rth_t'):
            expected=np.full(getattr(s,name).shape,21. if km==4 else 0.,np.float32)
            if km==4: expected[...,0]=expected[...,-1]=0.
            np.testing.assert_array_equal(getattr(s,name),expected)
    if km==4:
        assert events[:5]==['deform','K','Kbc','hdx','strip']
        assert events.count('deform')==2
        # Removal of the producer must break the fixed-value contract.
        monkeypatch.setattr(smag2d_family,'submit_wrf_smag_deform',lambda *a,**kw:None)
        with pytest.raises(AssertionError): dycore.prepare_fixed_tendencies(s,c)
    else: assert events==[]


def test_warm_network_pass_order_and_held_vapor(monkeypatch):
    """Real typed launcher, recorded numerical leaves; shader slot check is static."""
    from pathlib import Path
    warm=importlib.import_module('woof.metal.thompson_warm_frozen_source')
    runtime=importlib.import_module('woof.core.thompson_runtime')
    monkeypatch.setattr(warm,'MetalArray',HostMetal)
    # Constructor allocation at the launcher boundary needs (buffer, shape, dtype).
    original_new=HostMetal.__new__
    def allocate(cls,value,context,dtype=np.float32):
        if hasattr(value,'context'):
            return original_new(cls,np.zeros(context,dtype),value.context)
        return original_new(cls,value,context)
    monkeypatch.setattr(HostMetal,'__new__',staticmethod(allocate))
    xp=HostModule(); events=[]; seen=[]
    xp.context.buffer=lambda n:NS(context=xp.context,size=n)
    rate=[.125]
    def submit(name,arrays,params,**kw):
        events.append(name)
        if name.endswith('_limits'):
            arrays['cloud_state'][63]=rate[0]
        if name.endswith('_apply') or name.endswith('_finalize'):
            seen.append(float(arrays['cloud_state'][63]))
        if 'graupel_pow' in name and params.get('mode')==3:
            arrays['cloud_state'][45]=999.
    xp.context.submit_kernel=submit
    named={}
    logical={}
    for name,shape in warm._TABLE_SHAPES.items():
        key=warm._TABLE_LOGICAL_NAMES.get(name,name)
        named[key]=xp.zeros((int(np.prod(shape)),),np.float64)
        logical[key]=shape
    # Required accessor groups outside the warm launcher's selected tables.
    for key in ('tpi_ide','tps_iaus','tni_iaus','tpi_qcfz','tni_qcfz',
                *runtime.RAIN_FREEZING_TABLE_NAMES):
        named.setdefault(key,xp.zeros((1,),np.float64))
    owner=runtime.DeviceClassicTableSet.__new__(runtime.DeviceClassicTableSet)
    for key,val in dict(arrays=named,logical_shapes=logical,
                        identity_sha256='cpu-fixture',roundtrip_verified=True).items():
        object.__setattr__(owner,key,val)
    fields=[xp.zeros((1,)) for _ in range(11)]
    for value in (.125,.25):
        rate[0]=value
        warm.submit_warm_frozen_source_network(*fields,owner,15.)
    assert seen==[.125,.125,.25,.25]
    selected=[e.rsplit('network_',1)[-1] for e in events if 'source_network_' in e]
    assert selected==['melt','vapor','limits','apply','finalize']*2
    source=(Path(warm.__file__).parent/'shaders/thompson_warm_frozen_source.metal').read_text()
    assert 'state[63 * n + idx] = vapor_rate.b;' in source
    for name in ('apply','finalize'):
        body=source.split('kernel void thompson_warm_frozen_source_network_'+name+'(')[1].split('kernel void')[0]
        assert 'D64::fromBits(state[63 * n + idx])' in body
    # Resetting the held slot between apply/finalize violates this seam.
    real_submit=submit
    def early_reset(name,arrays,params,**kw):
        real_submit(name,arrays,params,**kw)
        if name.endswith('_apply'): arrays['cloud_state'][63]=0.
    xp.context.submit_kernel=early_reset
    seen.clear()
    warm.submit_warm_frozen_source_network(*fields,owner,15.)
    with pytest.raises(AssertionError): assert seen==[.25,.25]


def test_radiation_compute_due_skipped_day_night(monkeypatch):
    s=state(); c=cfg(ra_lw_physics=4,ra_sw_physics=4,radt_minutes=.5)
    adapter=RRTMGPRadiation(); calls=[]
    def radiation(self,**kw):
        now=kw['state'].elapsed_seconds; calls.append(now)
        xp=s._array_module
        return physics.RadiationResult(
            xp.zeros(s.p.shape),xp.zeros(s.p.shape),
            xp.asarray(np.full((3,3),600. if now<30 else 0.)),
            xp.asarray(np.full((3,3),300. if now<30 else 275.)),
            olr=xp.asarray(np.full((3,3),200.)))
    monkeypatch.setattr(RRTMGPRadiation,'__call__',radiation,raising=False)
    d=initialized(s,c,radiation=adapter)
    monkeypatch.setattr(physics,'_prepare_atmosphere',lambda st:atmosphere(st))
    monkeypatch.setattr(physics,'couple_column_tendencies',lambda *a,**kw:
        physics.PhysicsTendencies.zeros(s))
    observed=[]
    for now in (0.,15.,30.,45.):
        s.elapsed_seconds=now; d.compute(s,c)
        observed.append((float(d.fields['swdown'][0,0]),float(d.fields['glw'][0,0])))
    assert calls==[0.,30.]
    assert observed==[(600.,300.),(600.,300.),(0.,275.),(0.,275.)]
    assert d.call_counts['radiation']==2


def test_restart_real_manifest_restore_once(monkeypatch):
    restart=importlib.import_module('woof.io.restart')
    s=state(); c=cfg(sf_surface_physics=2,sf_sfclay_physics=1,bl_pbl_physics=1)
    d=initialized(s,c)
    for value in (.25,.5):
        d.microphysics.rainncv[...] = value
        d.accept_microphysics(d.microphysics,dt=15.)
    d.fields['rainbl'][...] = .125
    d.microphysics.sr[...] = .75
    stored={k:np.asarray(v).copy() for k,v in restart._driver_manifest(d).items() if v is not None}
    stored.update({'scratch/'+slot:np.asarray(s._scratch[slot]).copy()
                   for name,slot in microphysics_scratch_slots(8)})
    s2=state(); resumed=initialized(s2,c)
    for name,slot in microphysics_scratch_slots(8):
        s2.scratch((3,3),slot)[...] = stored['scratch/'+slot]
    carrier_payload=d.carriers.state()
    header={'config':{'bldt':c.bldt,'bl_pbl_physics':c.bl_pbl_physics},'driver':{
        'call_counts':d.call_counts,'ysu_nan_guard_fires':0,
        'microphysics_updates':2,'carriers':carrier_payload}}
    missing=dict(stored)
    missing.pop('driver/pending_rainbl')
    with pytest.raises(restart.RestartMismatchError,match='pending_rainbl'):
        restart._restore_driver(missing,header,s2,resumed,15.,s2._array_module.asarray,
                                restart.RESTART_FORMAT_VERSION)
    restart._restore_driver(stored,header,s2,resumed,15.,s2._array_module.asarray,
                            restart.RESTART_FORMAT_VERSION)
    np.testing.assert_array_equal(resumed._pending_rainbl,.75)
    np.testing.assert_array_equal(resumed.fields['swdown'],400.)
    np.testing.assert_array_equal(resumed.fields['glw'],250.)
    assert resumed.carriers.state()==carrier_payload
    for name,slot in microphysics_scratch_slots(8):
        assert getattr(resumed.microphysics,name) is s2._scratch[slot]
    delivered=[]; noah_boundary(monkeypatch,delivered)
    resumed._run_noah(atmosphere(s2),c,2)
    resumed._run_noah(atmosphere(s2),c,4)
    np.testing.assert_array_equal(delivered[0][0],.875)
    np.testing.assert_array_equal(delivered[1][0],0.)
