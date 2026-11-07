"""Compare native shortwave sounding preparation against selected source."""
from importlib.resources import files
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import prepare_sounding
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    root=files('woof.core.kernels')
    arrays='\n'.join(re.search(r'__device__ const float '+n+r'\[\d+\]\s*=\s*\{[^}]*\};',CUDA_SOURCE).group() for n in ('MPAS_RETAB','MPAS_O3SUM','MPAS_O3WIN','MPAS_PPSUM','MPAS_PPWIN'))
    start=CUDA_SOURCE.index('__device__ float mpas_ice_radius(')
    end=CUDA_SOURCE.index('// MPAS module_ra_rrtmg_sw.F:5058-5150',start)
    source=root.joinpath('glibc_flt32.cuh').read_text()+root.joinpath('glibc_trig_flt32.cuh').read_text()+arrays+CUDA_SOURCE[start:end]
    kernel=HostModule('mpas_v842_sw_prep',source).get_function('mpas_sw_prep')
    nc,nz,nozn=7,5,59; rng=np.random.default_rng(842)
    fields=np.empty((12,nc,nz),np.float32)
    fields[0]=np.linspace(95000,25000,nz);fields[1]=np.linspace(280,230,nz)
    fields[2]=rng.uniform(70,200,(nc,nz));fields[3]=rng.uniform(.7,1,(nc,nz))
    for i in (4,5,6,7):fields[i]=rng.uniform(.000001,.004,(nc,nz))
    fields[8]=rng.uniform(0,1,(nc,nz))
    fields[8,0,0]=np.array(1,np.uint32).view(np.float32)
    for i in (5,6,7):fields[i,0,0]=np.array(0x10000,np.uint32).view(np.float32)
    for i in (9,10,11):fields[i]=rng.uniform(.000005,.00008,(nc,nz))
    fields[9,0,0]=fields[10,0,0]=0
    surface=np.zeros((7,nc),np.float32);surface[0]=np.linspace(-70,70,nc);surface[1]=np.linspace(-160,160,nc);surface[2]=.3;surface[3]=273;surface[4]=np.array([1,2,1,2,1,2,1]);surface[5]=np.linspace(0,1,nc);surface[6]=np.linspace(0,20,nc)
    pw=np.broadcast_to(np.linspace(100000,18000,nz+1,dtype=np.float32),(nc,nz+1)).copy()
    tw=np.broadcast_to(np.linspace(282,220,nz+1,dtype=np.float32),(nc,nz+1)).copy()
    pin=np.linspace(1,1100,nozn,dtype=np.float32);oz=rng.uniform(.00000001,.00001,(nc,nozn)).astype(np.float32)
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        for o3 in (0,2):
            for flags in ((0,0,0),(1,1,1)):
                for xtime in (0,1439.5,1440,2881.25):
                    controls=np.array([xtime,15,.2,1370],np.float32)
                    layers=np.empty((15,nc,nz+1),np.float32);interfaces=np.empty((2,nc,nz+2),np.float32);hgt=np.empty((nc,nz+1),np.float32);coszr=np.empty(nc,np.float32)
                    kernel((1,),(128,),(nc,nz,1,o3,*flags,nozn,*[np.float32(x) for x in (xtime,12,15,np.pi/180,.2)],fields,surface,pw,tw,pin,oz,layers,interfaces,hgt,coszr))
                    actual=prepare_sounding(fields=xp.asarray(fields),surface=xp.asarray(surface),pressure_interfaces=xp.asarray(pw),temperature_interfaces=xp.asarray(tw),pin=xp.asarray(pin),ozone=xp.asarray(oz),controls=xp.asarray(controls),gmt=12,degrad=np.pi/180,icloud=1,o3input=o3,has_reqc=flags[0],has_reqi=flags[1],has_reqs=flags[2])
                    for name,expected in [('layers',layers),('interfaces',interfaces),('hgt',hgt),('coszr',coszr)]:
                        got=actual[name].to_numpy();bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))
                        if bad:raise AssertionError(f'{name} o3={o3} flags={flags} time={xtime}: {bad} mismatches')
    print('SW sounding matches selected source for ozone, cloud-radius and time variants',flush=True)

if __name__=='__main__':main()
