"""Compare in-domain ozone operators and refuse the native undefined endpoint."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.radshared_v842 import CUDA_SOURCE,OzoneCalendarState,_monthly_date_weights
from woof.hex.mpasphys.metal_ozone_v842 import initialize_monthly,MetalOzoneMonthlyUpdate,interpolate_cam_time,interpolate_vertical
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()
    start=CUDA_SOURCE.index('extern "C" __global__ void ozone_latitude(')
    end=CUDA_SOURCE.index('// MPAS mpas_atmphys_driver_radiation_lw.F',start)
    host=HostModule('mpas_v842_ozone',header+'\n'+CUDA_SOURCE[start:end])
    latitude=host.get_function('ozone_latitude'); update=host.get_function('ozone_update'); time=host.get_function('ozone_time')
    nc,nl=7,5; rng=np.random.default_rng(2843)
    lat_oz=np.linspace(-90,90,64,dtype=np.float32)
    deg=np.array([-91,-83,-30,-.2,15,83,91],np.float32)
    lat=deg*np.float32(np.float32(3.1415926)/np.float32(180.))
    table=rng.uniform(.00001,.0001,(12,64,nl)).astype(np.float32)
    local=np.empty((nc,nl,12),np.float32)
    latitude((4,),(128,),(nc,nl,lat,lat_oz,table,local))
    expected=np.ascontiguousarray(local.transpose(2,1,0))
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        dlat=xp.asarray(lat); grid=xp.asarray(lat_oz); data=xp.asarray(table)
        monthly=xp.asarray(np.full((12,nl,nc),-4,np.float32))
        initialize_monthly(latitude=dlat,lat_ozone=grid,table=data,monthly=monthly)
        if not np.array_equal(monthly.to_numpy().view(np.uint32),expected.view(np.uint32)):
            raise AssertionError('ozone latitude differs from selected source')
        state=MetalOzoneMonthlyUpdate(); reference_state=OzoneCalendarState()
        output=xp.asarray(np.full((nl,nc),-11,np.float32))
        reference=np.full((nl,nc),-11,np.float32)
        for date in ('2024-02-28_00:00:00','2025-03-20_00:00:00'):
            m1,m2,d1,d2=_monthly_date_weights(date,reference_state)
            source=np.ascontiguousarray(expected[:,:,:nc-1].transpose(2,1,0))
            local_out=np.empty((nc-1,nl),np.float32)
            update((1,),(128,),((nc-1)*nl,m1,m2,d1,d2,source,local_out))
            reference[:,:nc-1]=local_out.T
            state.update(current_date=date,monthly=monthly,out=output,n_cells_solve=nc-1)
            if not np.array_equal(output.to_numpy().view(np.uint32),reference.view(np.uint32)):
                raise AssertionError('ozone monthly date update differs from selected source')
        saved=state.restart_state(); new=MetalOzoneMonthlyUpdate(); new.restore_restart_state(saved)
        if new.calendar.february_days!=29: raise AssertionError('ozone calendar state differs')
        for julian in (0.,14.999,15.,43.5,44.,74.,196.125,348.999,349.,364.,365.75):
            local_out=np.empty((nc-1,nl),np.float32)
            time((1,),(128,),((nc-1)*nl,np.float32(julian),source,local_out))
            reference[:,:nc-1]=local_out.T
            interpolate_cam_time(julian=julian,monthly=monthly,out=output,n_cells_solve=nc-1)
            if not np.array_equal(output.to_numpy().view(np.uint32),reference.view(np.uint32)):
                raise AssertionError(f'ozone CAM interpolation differs at day {julian}')
        pin=np.array([1,10,100,500,1000],np.float32)
        pmid=np.array([[1100,900,700,500,300,120,50],
                       [500,400,300,200,100,20,10],
                       [20,15,10,5,2,1,.5]],np.float32)
        ozone=reference.copy(); nz=pmid.shape[0]; nsolve=nc-1
        for pascal in (False,True):
            p=pmid*(np.float32(100) if pascal else np.float32(1))
            local_p=np.ascontiguousarray(p[:,:nsolve].T)
            local_o=np.ascontiguousarray(ozone[:,:nsolve].T)
            local_out=np.zeros((nsolve,nz),np.float32); upper=np.zeros(nsolve,np.int32); low=np.zeros(1,np.int32)
            for k in range(nz-1,-1,-1):
                host.get_function('ozone_min_upper')((1,),(128,),(nsolve,nl,upper,low))
                host.get_function('ozone_vertical_level')((1,),(128,),
                    (nsolve,nz,nl,k,int(pascal),local_p,pin,local_o,low,upper,local_out))
            out=xp.asarray(np.full((nz,nc),-17,np.float32))
            interpolate_vertical(pmid=xp.asarray(p),pin=xp.asarray(pin),ozone=xp.asarray(ozone),
                                 out=out,n_cells_solve=nsolve,pressure_in_pa=pascal)
            expected_vertical=np.full((nz,nc),-17,np.float32); expected_vertical[:,:nsolve]=local_out.T
            if not np.array_equal(out.to_numpy().view(np.uint32),expected_vertical.view(np.uint32)):
                raise AssertionError(f'ozone vertical interpolation differs (Pa={pascal})')
        endpoint=np.float32(lat_oz[-1]*np.float32(np.float32(3.1415926)/np.float32(180.)))
        if np.float32(endpoint/np.float32(np.float32(3.1415926)/np.float32(180.))) != lat_oz[-1]:
            raise AssertionError('endpoint fixture did not round to final latitude')
        endpoint_lat=xp.asarray(np.full(nc,endpoint,np.float32)); before=monthly.to_numpy()
        try: initialize_monthly(latitude=endpoint_lat,lat_ozone=grid,table=data,monthly=monthly)
        except ValueError as error:
            if 'undefined' not in str(error): raise
        else: raise AssertionError('undefined native endpoint was admitted')
        if not np.array_equal(monthly.to_numpy().view(np.uint32),before.view(np.uint32)):
            raise AssertionError('endpoint refusal mutated monthly ozone')
    print('in-domain ozone initialization/date update match selected source; endpoint refused before mutation')

if __name__=='__main__': main()
