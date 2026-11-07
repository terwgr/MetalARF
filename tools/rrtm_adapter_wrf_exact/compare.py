"""Verify the real strict classic RRTM caller against source columns."""
from datetime import datetime
from pathlib import Path
import sys
from types import SimpleNamespace
import json
import os
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.core.rrtm import rrtm_default_trace_gases
from woof.core.rrtm_lw import RRTMLongwaveRadiation, RRTMDudhiaRadiation, _t8w_columns, rrtmlwrad_columns, rrtm_layer_count
from woof.core.dudhia import DudhiaShortwaveRadiation
from woof.core import constants
from woof.core.rrtmgp import cal_cldfra1, cal_cldfra2
from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from tools.smag_wrf_exact.compare import compare_words
nz, ny, nx = 5, 2, 3
ncol = ny * nx
icloud = int(os.environ.get("RRTM_PROBE_ICLOUD", "0"))
rng = np.random.default_rng(829)
shape = (nz, ny, nx)
interfaces = np.broadcast_to(np.linspace(100000, 1200, nz+1, dtype=np.float32)[:,None,None], (nz+1,ny,nx)).copy()
pressure = (interfaces[:-1] + interfaces[1:]) * np.float32(.5)
temperature = rng.uniform(250, 290, shape).astype(np.float32)
height = np.broadcast_to(np.arange(nz+1, dtype=np.float32)[:,None,None] * np.float32(600), (nz+1,ny,nx)).copy()
values = {'temperature': temperature, 'pressure': pressure, 'p_interface': interfaces,
          'z_interface': height, 'dz': np.full(shape, 600, np.float32),
          'qv': rng.uniform(.001,.02,shape).astype(np.float32),
          'qc': rng.uniform(0, .0004, shape).astype(np.float32), 'qi': rng.uniform(0, .0001, shape).astype(np.float32),
          'exner': rng.uniform(.65, 1, shape).astype(np.float32)}
values['qv'][0, 0, 0] = np.float32(-.001)
values['qc'][1, 0, 0] = np.float32(-.0002)
values['qi'][2, 0, 0] = np.float32(-.0001)
fields = {'tsk': np.full((ny,nx), 290, np.float32), 'emiss': np.full((ny,nx), .95, np.float32)}
fnm = np.full(nz, .4, np.float32); fnp = np.full(nz, .6, np.float32)
def columns(a): return np.ascontiguousarray(a.transpose(1,2,0)).reshape(ncol,a.shape[0])
gases = rrtm_default_trace_gases(2020)
t8w = _t8w_columns(np, columns(temperature), columns(height), fnm, fnp)
zeros = np.zeros((ncol,nz),np.float32)
f_qi, f_qs = legacy_cloud_fraction_flags(0)
if icloud == 1:
    cldfra = cal_cldfra1(columns(np.maximum(values['qv'],0)),columns(np.maximum(values['qc'],0)),
                         columns(np.maximum(values['qi'],0)),zeros,
                         columns(temperature),columns(pressure),f_qc=True,f_qi=f_qi,f_qs=f_qs)
elif icloud == 2:
    cldfra = cal_cldfra2(columns(values['qc']),columns(values['qi']),f_qc=True,f_qi=f_qi)
else:
    cldfra = zeros
if icloud == 1:
    assert np.any(cldfra > 0)
expected = rrtmlwrad_columns(t3d=columns(temperature), p3d=columns(pressure), pi3d=columns(values['exner']),
    dz8w=columns(values['dz']), qv=columns(values['qv']), qc=columns(values['qc']), qr=zeros, qi=columns(values['qi']), qs=zeros, qg=zeros,
    cldfra=cldfra, p8w=columns(interfaces), t8w=t8w, tsk=fields['tsk'].reshape(-1), emiss=fields['emiss'].reshape(-1),
    nlayers=rrtm_layer_count(nz,1200), icloud=icloud, r=constants.RD, g=constants.G,
    co2vmr=gases['co2'], n2ovmr=gases['n2o'], ch4vmr=gases['ch4'])
with MetalContext(batch_kernel_calls=False) as context:
    submit = context._submit_kernel_batch
    def sync(*a, **kw):
        result = submit(*a, **kw); result.wait(); return result
    context._submit_kernel_batch = sync
    xp=MetalArrayModule(context)
    dev = {k:xp.asarray(v) for k,v in values.items()}
    dev_fields = {k:xp.asarray(v) for k,v in fields.items()}
    adapter = RRTMLongwaveRadiation(datetime(2020,1,1),np.zeros((ny,nx),np.float32),
                                    np.zeros((ny,nx),np.float32),p_top=1200,icloud=icloud)
    state=SimpleNamespace(fnm=xp.asarray(fnm),fnp=xp.asarray(fnp),elapsed_seconds=0)
    got=adapter(atmosphere=dev, fields=dev_fields, state=state, cfg=SimpleNamespace(mp_physics=0, bl_pbl_physics=0, icloud_bl=0, dt=60))
    out = {'rthratenlw':expected['rthraten'].reshape(ny,nx,nz).transpose(2,0,1).copy(),
           'glw':expected['glw'].reshape(ny,nx),'olr':expected['olr'].reshape(ny,nx),
           'cldfra':cldfra.reshape(ny,nx,nz).transpose(2,0,1).copy()}
    report={name:compare_words(getattr(got,name).to_numpy(),want) for name,want in out.items()}
    if icloud == 1:
        from dataclasses import replace
        from woof.config import RunConfig
        from woof.core.physics import _metal_strict_physics_profile
        config=RunConfig(nx=nx,ny=ny,nz=nz,dx=1000.,dy=1000.,ztop=12000.,dt=60.,run_seconds=120.,
                         moist=True,ra_lw_physics=1,ra_sw_physics=1,icloud=1,
                         solar_generation='wrf_461',radt=1.,ra_sw_eclipse=0,
                         sf_sfclay_physics=0,sf_surface_physics=0)
        paired=RRTMDudhiaRadiation(datetime(2020,1,1),np.zeros((ny,nx),np.float32),
                                  np.zeros((ny,nx),np.float32),p_top=1200,icloud=1)
        assert _metal_strict_physics_profile(config,radiation=paired)
        assert not _metal_strict_physics_profile(config,radiation=adapter)
        assert not _metal_strict_physics_profile(replace(config,icloud=2),radiation=paired)
        for changes in ({'nested':True},{'terrain_opt':2},{'mp_physics':8},
                        {'sf_surface_physics':3},{'cu_physics':1}):
            assert not _metal_strict_physics_profile(replace(config,**changes),radiation=paired)
        state._metal_setup_state=True; state._array_module=xp
        dev_fields['albedo']=xp.asarray(np.full((ny,nx),.2,np.float32))
        dev_fields['glw']=xp.asarray(np.full((ny,nx),17,np.float32))
        sw=DudhiaShortwaveRadiation(datetime(2020,1,1),np.zeros((ny,nx),np.float32),
                                   np.zeros((ny,nx),np.float32),icloud=1)
        sw_result=sw(atmosphere=dev,fields=dev_fields,state=state,cfg=config)
        combined=paired(atmosphere=dev,fields=dev_fields,state=state,cfg=config)
        for name in ('rthratenlw','glw','olr','cldfra'):
            report['paired_'+name]=compare_words(getattr(combined,name).to_numpy(),out[name])
        for name in ('rthratensw','swdown','gsw','coszen'):
            report['paired_'+name]=compare_words(getattr(combined,name).to_numpy(),getattr(sw_result,name).to_numpy())
        assert paired.update_count == paired.longwave_adapter.update_count == 1
        assert paired.shortwave_adapter.update_count == sw.update_count == 1
        assert paired.restart_identity['longwave'] == paired.longwave_adapter.restart_identity
        assert paired.restart_identity['shortwave'] == paired.shortwave_adapter.restart_identity
        assert paired.spectrum_adapters == (paired.longwave_adapter, paired.shortwave_adapter)
        report['profile_and_lifecycle_checked']=True
    print(json.dumps(report,indent=2))
    assert not any(case['different_words'] for case in report.values() if isinstance(case,dict))
