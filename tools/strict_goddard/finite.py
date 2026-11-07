"""Check strict Goddard finite-domain and unpublished-result refusals."""
from datetime import datetime,timezone
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from woof.core.radiation_composition import ColumnTableRadiation
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
assert ENABLED
with MetalContext() as context:
    xp=MetalArrayModule(context)
    array=lambda shape,value:xp.asarray(np.full(shape,value,np.float32))
    shape=(2,1,1); surface=(1,1)
    atmosphere={'temperature':array(shape,280),'pressure':array(shape,90000),
        'p_interface':array((3,1,1),90000),'z_interface':array((3,1,1),100),
        'dz':array(shape,100),'rho':array(shape,1),'exner':array(shape,.9)}
    fields={name:array(surface,value) for name,value in
            [('tsk',280),('emiss',.98),('albedo',.2),('glw',300)]}
    state=SimpleNamespace(_metal_setup_state=True,elapsed_seconds=0.,fnm=np.full(2,.5,np.float32),
        fnp=np.full(2,.5,np.float32),qv=array(shape,float('nan')),qc=None,qr=None,qi=None,qs=None,qg=None,qh=None)
    cfg=SimpleNamespace(mp_physics=0,bl_pbl_physics=0,icloud=0,icloud_bl=0,use_mp_re=1,
                        radt=3.,radt_minutes=3.,dt=1.)
    adapter=ColumnTableRadiation(datetime(2024,1,1,tzinfo=timezone.utc),np.zeros(surface,np.float32),np.zeros(surface,np.float32))
    with patch.object(context,'submit_kernel') as submit:
        try:adapter(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
        except ValueError as error:assert 'finite qv' in str(error)
        else:raise AssertionError('nonfinite input accepted')
        assert submit.call_count==0 and adapter.update_count==0
    state.qv=array(shape,.001)
    def nonfinite(*args,**kwargs):
        kwargs['heat_lw'].fill(np.nan)
        kwargs['heat_sw'].fill(0)
        kwargs['surface'].fill(0)
    with patch('woof.metal.goddard.solve_chunk',side_effect=nonfinite):
        try:adapter(atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
        except ValueError as error:assert 'nonfinite longwave heating' in str(error)
        else:raise AssertionError('nonfinite output published')
        assert adapter.update_count==0
print('strict Goddard refused nonfinite input before dispatch and output before publication')
