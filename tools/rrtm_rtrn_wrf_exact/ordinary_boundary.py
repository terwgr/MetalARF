"""Check ordinary RTRN's top boundary and model-top heating with poison."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from woof.core.rrtm_tables import load_rrtm_lw_tables, RRTM_BPADE
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.metal.rrtm_lw_family import submit_rtrn_down
ncol,nlay,ngpt=2,3,140
with MetalContext(batch_kernel_calls=False) as context:
    original=context._submit_kernel_batch
    def sync(*a,**kw):
        result=original(*a,**kw); result.wait(); return result
    context._submit_kernel_batch=sync
    xp=MetalArrayModule(context); table=load_rrtm_lw_tables(); stat=table.statics
    fields={'pfrac':xp.asarray(np.full((ncol,nlay,ngpt),.01,np.float32)),
      'pz':xp.asarray(np.broadcast_to(np.array([1000,700,400,100],np.float32),(ncol,nlay+1)).copy()),
      'tz':xp.asarray(np.full((ncol,nlay+1),260,np.float32)),
      'tavel':xp.asarray(np.full((ncol,nlay),260,np.float32)),
      'cldfrac':xp.zeros((ncol,nlay),np.float32),'taucloud':xp.zeros((ncol,nlay),np.float32),
      'itr':xp.zeros((ncol,nlay,ngpt),np.int32),'tbound':xp.asarray(np.full(ncol,290,np.float32)),
      'semiss':xp.asarray(np.ones((ncol,16),np.float32)),
      'totplnk':xp.asarray(stat['TOTPLNK'].reshape((181,16),order='F').copy()),
      'delwave':xp.asarray(stat['DELWAVE']),'tau_table':xp.asarray(table.tau),
      'tf_table':xp.asarray(table.tf),'trans_table':xp.asarray(table.trans),
      'ngb0':xp.asarray(np.asarray(table.ngb,np.int32)-1)}
    for name in ('bbu','bbutot','atot','abss'):fields[name]=xp.zeros((ncol,nlay,ngpt),np.float32)
    for name in ('down','clear_down'):fields[name]=xp.asarray(np.full((ncol,nlay+1,ngpt),np.nan,np.float32))
    submit_rtrn_down(**fields,ncol=ncol,nlay=nlay,bpade=float(RRTM_BPADE))
    for name in ('down','clear_down'):
        top=fields[name].to_numpy()[:,-1]
        assert np.all(top.view(np.uint32)==0), name
    from woof.metal.rrtm_lw_family import submit_rtrn_fluxes
    up=xp.zeros((ncol,nlay+1,ngpt),np.float32)
    heating=xp.zeros((nlay,ncol,1),np.float32)
    submit_rtrn_fluxes(pz=fields['pz'],up=up,down=fields['down'],
        exner=xp.asarray(np.ones((ncol,nlay),np.float32)),rthratenlw=heating,
        glw=xp.zeros((ncol,1),np.float32),olr=xp.zeros((ncol,1),np.float32),
        ncol=ncol,nmodel=nlay,nlay=nlay,ny=ncol,nx=1,
        heatfac=float(stat['HEATFAC'][0]),fluxfac=62831.855)
    assert np.all(np.isfinite(heating.to_numpy()[-1]))
    print('top boundaries are +0 and model-top heating is finite')
