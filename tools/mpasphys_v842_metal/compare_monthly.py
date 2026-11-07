"""Compare the MPAS monthly surface service with host-compiled selected source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.surface_update_v842 import _CUDA, MonthlyCalendar
from woof.hex.mpasphys.metal_surface_v842 import MetalMonthlySurface
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    code=_CUDA[_CUDA.index('__device__ float add('):_CUDA.index('// MPAS source: mpas_atmphys_update_surface.F:196-313.')]
    kernel=HostModule('mpas_v842_monthly',code).get_function('monthly')
    rng=np.random.default_rng(84231); nc=9; solved=7
    green=rng.uniform(0,1,(12,nc)).astype(np.float32)
    alb=rng.uniform(1,70,(12,nc)).astype(np.float32)
    land=np.array([0,1,1,0,1,1,0,1,1],np.int32)
    initial={n:rng.uniform(.1,.8,nc).astype(np.float32) for n in ('vegfra','shdmin','shdmax','sfc_albbck')}
    reference={n:v.copy() for n,v in initial.items()}
    calendar=MonthlyCalendar()
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        fields={'greenfrac':xp.asarray(green),'albedo12m':xp.asarray(alb),'landmask':xp.asarray(land),**{n:xp.asarray(v) for n,v in initial.items()}}
        service=MetalMonthlySurface()
        for date,use in [('2024-02-28_00:00:00',True),('2025-03-20_00:00:00',False),('2025-04-20_00:00:00',True)]:
            weights=calendar.weights(date)
            kernel((1,),(128,),(solved,*weights,np.asfortranarray(green),np.asfortranarray(alb),land,int(use),
                *(reference[n] for n in ('vegfra','shdmin','shdmax','sfc_albbck'))))
            service.update(current_date=date,config_sfc_albedo=use,fields=fields,n_cells_solve=solved)
            for n,v in reference.items():
                got=fields[n].to_numpy(); count=np.count_nonzero(got.view(np.uint32)!=v.view(np.uint32))
                if count: raise AssertionError(f'{date} {n}: {count} differing words')
            if service.restart_state() != {'february':calendar.february}: raise AssertionError('calendar differs')
        saved=service.restart_state()
        restored=MetalMonthlySurface(); restored.restore_restart_state(saved)
        if restored.calendar.weights('2026-03-20_00:00:00')!=calendar.weights('2026-03-20_00:00:00'):
            raise AssertionError('monthly calendar checkpoint phase differs')
    print('monthly surface source comparison passed; leap-February restart phase preserved')

if __name__=='__main__': main()
