"""Compare the bound native post-transport callback with selected source."""
from types import SimpleNamespace
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_physics_v841 import _MPAS_V842_SOURCE


def exercise(session, driver, held):
    p=session.pools; d=session.layout.dimensions
    nz,nc=d['nVertLevels'],d['nCellsSolve']; ns=len(session.resident_preparation.names)
    rng=np.random.default_rng(84291); xp=p.arrays
    before=rng.uniform(.001,.01,(ns,nz,nc)).astype(np.float32)
    after=rng.uniform(-.001,.02,before.shape).astype(np.float32)
    rho=rng.uniform(.7,1.5,(nz,nc)).astype(np.float32)
    theta=rng.uniform(260,300,(nz,nc)).astype(np.float32)
    rate=rng.uniform(-.002,.002,(nz,nc)).astype(np.float32)
    expected_base=np.empty_like(before); expected_after=after.copy()
    expected_q=np.empty_like(rho); expected_rate=rate.copy()
    host=HostModule('mpas_v842_transport',_MPAS_V842_SOURCE)
    host.get_function('mpas_transport_source_baseline_v842_f32')(((before.size+127)//128,),(128,),
        (nz,nc,nc,ns,np.float32(session.dt),before,held.scalars.to_numpy(),rho,expected_base))
    host.get_function('mpas_transport_handoff_v842_f32')(((nz*nc+127)//128,),(128,),
        (nz,nc,nc,ns,session.resident_preparation.names.index('qv'),0,1,np.float32(session.dt),
         np.float32(461.6),np.float32(287),np.float32(np.float32(461.6)/np.float32(287)),
         expected_base,expected_after,theta,expected_q,expected_rate))
    current=SimpleNamespace(scalars=xp.asarray(after.copy()),rho=xp.asarray(rho.copy()))
    outer=SimpleNamespace(scalars=xp.asarray(before.copy()),rho=xp.asarray(rho.copy()))
    carried=SimpleNamespace(theta_m=xp.asarray(theta.copy()))
    synchronized=[]
    def sync(actual,actual_carried):
        if actual is not current or actual_carried is not carried:
            raise AssertionError('transport synchronized a different candidate')
        synchronized.append(True)
    session.resident_transport.bind(current=current,carried=carried,outer=outer,rate=xp.asarray(rate.copy()),held=held,
        scalar_advection=True,monotonic=True,positive_definite=False,convection='off',
        scalar_names=session.resident_preparation.names,synchronize=sync)
    if d['nCells'] != nc:
        try:
            driver.finish_step(pools=session)
        except ValueError as error:
            if 'halo synchronization' not in str(error): raise
        else:
            raise AssertionError('unsupported transport halo was admitted')
        if synchronized:
            raise AssertionError('transport halo refusal mutated native state')
        session.resident_transport.restore_restart_state({})
        driver.hooks.external['post_transport_prepare']=lambda call:None
        driver.finish_step(pools=session)
        print('transport refuses unexchanged native halos before synchronization')
        return
    driver.finish_step(pools=session)
    if synchronized != [True] or session.resident_transport.pending is not None:
        raise AssertionError('post-transport callback did not synchronize and consume candidate')
    for name,expected,actual in (('baseline',expected_base,p.field('state','scalars',level=1)),
                                 ('after',expected_after,current.scalars),
                                 ('theta rate',expected_rate,p.field('tend_physics','rthdynten'))):
        if not np.array_equal(expected.view(np.uint32),actual.to_numpy().view(np.uint32)):
            raise AssertionError(f'real post-transport callback differs from selected source: {name}')
    expected_after=after.copy(); expected_q.fill(0); expected_rate=rate.copy()
    host.get_function('mpas_transport_handoff_v842_f32')(((nz*nc+127)//128,),(128,),
        (nz,nc,nc,ns,session.resident_preparation.names.index('qv'),1,1,np.float32(session.dt),
         np.float32(461.6),np.float32(287),np.float32(np.float32(461.6)/np.float32(287)),
         expected_base,expected_after,theta,expected_q,expected_rate))
    current.scalars=xp.asarray(after.copy()); synchronized.clear()
    session.resident_transport.bind(current=current,carried=carried,outer=outer,rate=xp.asarray(rate.copy()),held=held,
        scalar_advection=True,monotonic=True,positive_definite=False,convection='cu_grell_freitas',
        scalar_names=session.resident_preparation.names,synchronize=sync)
    hooks=session.bind_hooks(external={},save_fields=lambda:{},restore_fields=lambda x:None,after_call=lambda e,c:None)
    hooks.invoke('post_transport_prepare',None)
    if ('tend_physics','rqvdynten') not in session.native._fields:
        raise AssertionError('live fixture lacks native GF water tendency')
    for name,expected,actual in (('GF scalar',expected_after,current.scalars),
                                 ('GF water',expected_q,p.field('tend_physics','rqvdynten')),
                                 ('GF theta',expected_rate,p.field('tend_physics','rthdynten'))):
        if not np.array_equal(expected.view(np.uint32),actual.to_numpy().view(np.uint32)):
            raise AssertionError(f'GF post-transport differs from selected source: {name}')
    print('bound post-transport callback matches selected source')
