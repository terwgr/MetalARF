"""Compare native transport, winds and publication on a public dycore candidate."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_physics_v841 import _MPAS_V842_SOURCE
from woof.hex.metal_driver_v841 import MetalDycoreV841
from woof.hex.mpasphys.metal_transport_v842 import source_baseline,handoff
from woof.hex.species_row import species_row_for_scheme

original=MetalDycoreV841.step_device_with_physics
captured=[]
# Reuse the real public stack's one-time sealed geometry, rather than
# reconstructing host geometry from the dycore's reduced device mesh.
from woof.hex import metal_arwen_physics_v841 as stack_module
open_stack=stack_module.open_metal_physics_stack_v841
geometries={}
def capture_stack(*args,**kwargs):
    stack=open_stack(*args,**kwargs)
    geometries[id(stack['driver'])]=stack['prep_geometry']
    return stack
stack_module.open_metal_physics_stack_v841=capture_stack

def step(driver,physics,*,transport_callback=None):
    def capture(current,carried,outer,raw):
        if raw is None:
            raise AssertionError('real dycore did not capture raw RK theta rate')
        names=species_row_for_scheme(driver.config.config_microp_scheme).names()
        nz,nc=outer.rho.shape; ns=len(names)
        nsolve=driver.regional_v841.n_cells_solve if driver.regional_v841 is not None else nc
        print(f'captured candidate levels={nz} columns={nc} solved={nsolve} species={ns}')
        dt=np.float32(driver.config.config_dt)
        before=outer.scalars.to_numpy(); source=physics.scalars.to_numpy(); rho=outer.rho.to_numpy()
        after=current.scalars.to_numpy(); theta=carried.theta_m.to_numpy(); rate=raw.to_numpy()
        expected_base=np.empty_like(before); expected_after=after.copy(); expected_rate=rate.copy(); expected_q=np.zeros_like(rho)
        host=HostModule('mpas_v842_transport',_MPAS_V842_SOURCE)
        host.get_function('mpas_transport_source_baseline_v842_f32')(((before.size+127)//128,),(128,),
            (nz,nc,nsolve,ns,dt,before,source,rho,expected_base))
        host.get_function('mpas_transport_handoff_v842_f32')(((nz*nc+127)//128,),(128,),
            (nz,nc,nsolve,ns,names.index('qv'),1,1,dt,np.float32(461.6),np.float32(287),
             np.float32(np.float32(461.6)/np.float32(287)),expected_base,expected_after,theta,expected_q,expected_rate))
        baseline=source_baseline(before=outer.scalars,source=physics.scalars,rho_old=outer.rho,dt=dt,n_cells_solve=nsolve)
        actual_after=current.scalars.copy(); actual_rate=raw.copy(); actual_q=driver.arrays.zeros((nz,nc),np.float32)
        handoff(before=baseline,after=actual_after,theta_m=carried.theta_m,scalar_names=names,
            rqvdynten=actual_q,rthdynten=actual_rate,convection_scheme='cu_grell_freitas',
            monotonic=True,dt=dt,n_cells_solve=nsolve)
        for name,expected,actual in (('baseline',expected_base,baseline),('scalars',expected_after,actual_after),
                                     ('water',expected_q,actual_q),('theta',expected_rate,actual_rate)):
            if not np.array_equal(expected.view(np.uint32),actual.to_numpy().view(np.uint32)):
                raise AssertionError(f'public dycore candidate transport differs from selected source: {name}')
        from tools.mpasphys_v842_metal.probe_candidate_session import exercise
        exercise(driver,current,carried,outer,raw,physics,names,expected_base,expected_after,expected_rate,expected_q,wind_geometry=geometries[id(driver)])
        captured.append(True)
        if transport_callback is not None: transport_callback(current,carried,outer,raw)
    return original(driver,physics,transport_callback=capture)

MetalDycoreV841.step_device_with_physics=step
from woof.cli import main
status=main()
if status not in (None,0): raise SystemExit(status)
if not captured: raise AssertionError('public forecast did not produce a physics candidate')
print(f'public dycore candidates matched selected MPAS transport source: {len(captured)}')
