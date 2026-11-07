"""Fresh-session file replay of the coupled fixture's injected dynamics.

The MPAS carrier, manager phase, native pool/service state and real scheme
services are exercised. Dycore variables and prescribed winds are fixture
inputs; this does not establish a public forecast or full source parity.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import json,pickle,subprocess,sys
import numpy as np
from woof.hex.mpasphys.metal_checkpoint_v842 import MetalCheckpointRoute
from woof.hex.mpasphys.checkpoint_v842 import CheckpointRoute
from woof.hex.mpasphys.moving_v842 import load_carrier,STATE_FIELDS,SAVED_FIELDS
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_tables
from woof.metal.runtime import MetalContext
from tools.mpasphys_v842_metal.probe_coupled_services import session_fixture,make_advance,same

ORIGIN='2024-06-28_12:00:00'
VALID='2024-06-28_12:05:00'

def identities(session):
    native=session.native
    mesh=NS(arrays={name:native.get_field('mesh',name) for name in ('latCell','lonCell','zgrid','zz')})
    vertical=NS(zgrid=mesh.arrays['zgrid'],dzw=native.get_field('mesh','dzu'))
    reference=NS(rho_base=native.get_field('diag','rho_base'),
        rho_theta_base=native.get_field('diag','rtheta_base'),
        pressure_base=native.get_field('diag','pressure_base'),
        exner_base=native.get_field('diag','exner_base'))
    grid=NS(vertical_grid=vertical);names=list(session.resident_preparation.names)
    return mesh,vertical,(mesh,grid,None,None,reference,None,names)

def make_route(session,carrier=None,receipt=None):
    route=MetalCheckpointRoute.from_route(CheckpointRoute())
    route.origin_time=ORIGIN;route.frame_from_cli=True;route.frame_seconds=300
    if carrier is not None:
        route.carrier=carrier;route.offset_seconds=carrier['time_seconds'];route.cli_restart=True
        route.restart_metadata=receipt['checkpoint']
    route.resolve_schedule(3600,300,[]);route.bind_inputs(identities(session)[2],session.config)
    return route

def dycore(session):
    p=session.pools
    # This fixture injects a stationary dycore rather than running the solver.
    rho=p.field('state','rho_zz',level=2);theta=p.field('state','theta_m',level=2)
    # Explicitly construct the stationary test candidate; it is not a dycore step.
    rho_theta=p.arrays.asarray(np.multiply(rho.to_numpy(),theta.to_numpy(),dtype=np.float32))
    state=NS(rho=rho,rho_theta=rho_theta,
             rho_u=p.field('state','u',level=2),rho_w=p.field('state','w',level=2),
             scalars=p.field('state','scalars',level=2))
    saved=NS(theta_m=p.field('state','theta_m',level=2),exner=p.field('diag','exner'),
        density_perturbation=p.field('diag','rho_p'),rho_theta_perturbation=p.field('diag','rtheta_p'),
        pressure_perturbation=p.field('diag','pressure_p'),normal_velocity=p.field('state','u',level=2),
        vertical_velocity=p.field('state','w',level=2))
    return NS(atmosphere=NS(state=state,saved=saved),stochastic=None,chemistry=None)

def run(context,packet,tables,path,phase):
    expected_path=path.with_suffix('.expected.pkl')
    if phase=='write':
        with session_fixture(context,packet,tables) as first:
            advance=make_advance(first);assert advance(1)
            mesh,vertical,_=identities(first);route=make_route(first)
            candidate=dycore(first)
            state_expected={name:getattr(candidate.atmosphere.state,name).to_numpy() for name in STATE_FIELDS}
            state_expected.update({name:getattr(candidate.atmosphere.saved,name).to_numpy() for name in SAVED_FIELDS})
            route.export(path,session=first,dycore=candidate,mesh=mesh,vertical=vertical,
                         names=first.resident_preparation.names,layout=first.layout,step=1,dt=300)
            assert not advance(2);assert advance(3);expected=first.transaction.checkpoint()
            with expected_path.open('wb') as stream:pickle.dump({'checkpoint':expected,'state':state_expected},stream,protocol=5)
    else:
        with expected_path.open('rb') as stream:reference=pickle.load(stream)
        receipt=json.loads(path.with_suffix(path.suffix+'.json').read_text());carrier=load_carrier(path)
        with session_fixture(context,packet,tables,restart=True,start=VALID) as resumed:
            # Verify and consume each sealed dycore carrier independently of
            # physics pools. The subsequent fixture dynamics is stationary.
            same(carrier['state'],reference['state'])
            for name,value in carrier['state'].items():
                same(resumed.pools.arrays.asarray(value).to_numpy(),reference['state'][name])
            route=make_route(resumed,carrier,receipt)
            route.restore_physics(resumed,resumed.layout,None,identities(resumed)[0])
            advance=make_advance(resumed);assert not advance(1);assert advance(2)
            actual=resumed.transaction.checkpoint()
            expected=reference['checkpoint'];same(actual['pools'],expected['pools'])
            for name in actual['services']:
                if name!='clock':same(actual['services'][name],expected['services'][name])
            same(actual['services']['clock']['text'],expected['services']['clock']['text'])
            assert resumed.native.clock_state.current_time.stamp()=='2024-06-28_12:15:00'
            print('coupled real-service fresh-process file resume matched numeric pools and non-clock service state; injected dynamics/winds',flush=True)

if __name__=='__main__':
    if len(sys.argv)==3:
        with TemporaryDirectory(prefix='mpas-coupled-file-') as temp:
            path=Path(temp)/'state.pkl'
            for phase in ('write','resume'):
                subprocess.run([sys.executable,__file__,sys.argv[1],sys.argv[2],str(path),phase],check=True)
    else:
        packet=json.loads(Path(sys.argv[1]).read_text())
        with MetalContext(batch_kernel_calls=False) as context:
            submit=context._submit_kernel_batch
            def sync(*args,**kwargs):
                result=submit(*args,**kwargs);result.wait();return result
            context._submit_kernel_batch=sync
            tables=load_tables({'table_directory':sys.argv[2]},context)
            run(context,packet,tables,Path(sys.argv[3]),sys.argv[4])
