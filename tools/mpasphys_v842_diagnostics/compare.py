"""Compare native diagnostic/EOS components and base-only restart publication."""
from types import SimpleNamespace as NS
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.diagnostics_v842 import native_diagnostic_source
from woof.hex.mpasphys.metal_diagnostics_v842 import output_diagnostics,initialize_thermodynamics
from woof.hex.cuda_physics_prep_v841 import RV_OVER_RD_F32
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def check(name,actual,expected):
    got=actual.to_numpy();bad=np.flatnonzero(got.view(np.uint32).reshape(-1)!=expected.view(np.uint32).reshape(-1))
    if bad.size:
        index=int(bad[0]);raise AssertionError(f'{name}: {bad.size} differences; first {index}, actual {got.view(np.uint32).flat[index]:08x}, expected {expected.view(np.uint32).flat[index]:08x}')
    return int(got.size)


def main():
    host=HostModule('mpas_v842_diagnostics',CUDA_FTZ_HELPERS+native_diagnostic_source())
    rng=np.random.default_rng(842);words=0
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        for shape in ((1,1),(7,9),(129,3)):
            count=int(np.prod(shape));f=lambda low,high:rng.uniform(low,high,shape).astype(np.float32)
            inputs=dict(theta_m=f(250,330),qv=f(-.001,.02),rho_zz=f(.2,1.5),zz=f(.5,1.5),pressure_base=f(1000,100000),pressure_p=f(-1000,1000))
            bits=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff],np.uint32)
            for i,name in enumerate(inputs):inputs[name].flat[:min(count,len(bits))]=np.roll(bits,i).view(np.float32)[:min(count,len(bits))]
            expected=[np.empty(shape,np.float32) for _ in range(3)]
            host.get_function('mpas_output_diagnostics_v842')(((count+127)//128,),(128,),(np.int32(count),RV_OVER_RD_F32,*inputs.values(),*expected))
            device={name:xp.asarray(value) for name,value in inputs.items()}
            atmosphere=NS(state=NS(rho=device['rho_zz'],scalars=xp.asarray(np.stack((inputs['qv'],np.ones(shape,np.float32))))),
                saved=NS(theta_m=device['theta_m'],pressure_perturbation=device['pressure_p']),
                vertical=NS(zz=device['zz']),reference=NS(pressure_base=device['pressure_base']))
            for name,actual,want in zip(('theta','rho','pressure'),output_diagnostics(atmosphere,('qv','qc')),expected):words+=check(name,actual,want)
            inputs=dict(zz=f(.5,1.5),rtheta_p=f(-2,2),rtheta_base=f(250,350))
            for name in ('zz','rtheta_base'):
                inputs[name].flat[:min(count,3)]=np.array([1,0x007fffff,0x00800000],np.uint32).view(np.float32)[:min(count,3)]
            inputs['rtheta_p'].flat[:min(count,3)]=np.array([1,0x80000001,0],np.uint32).view(np.float32)[:min(count,3)]
            seed={name:f(.1,.9) for name in ('exner','exner_base','pressure_p','pressure_base')}
            for base_only in (False,True):
                expected={name:value.copy() for name,value in seed.items()}
                temp=[expected[name].copy() if base_only and name in ('exner','pressure_p') else expected[name] for name in seed]
                rgas=np.float32(287);cp=np.float32(1004.5)
                host.get_function('mpas_cold_thermodynamics_v842')(((count+127)//128,),(128,),(np.int32(count),rgas,np.float32(1e5),np.float32(rgas/np.float32(cp-rgas)),*inputs.values(),*temp))
                device={name:xp.asarray(value) for name,value in {**inputs,**seed}.items()}
                atmosphere=NS(state=NS(rho=xp.empty(shape,np.float32)),vertical=NS(zz=device['zz']),
                    saved=NS(rho_theta_perturbation=device['rtheta_p'],exner=device['exner'],pressure_perturbation=device['pressure_p']),
                    reference=NS(rho_theta_base=device['rtheta_base'],exner_base=device['exner_base'],pressure_base=device['pressure_base']))
                initialize_thermodynamics(atmosphere,base_only=base_only)
                for name in seed:words+=check(name,device[name],expected[name])
            print('native diagnostic/EOS source match',shape,'cold and base-only; cumulative words',words,flush=True)
    print('Native diagnostic/EOS selected-source words',words,'different 0',flush=True)


if __name__=='__main__':main()
