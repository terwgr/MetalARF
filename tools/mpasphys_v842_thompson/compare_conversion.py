"""Compare native conversion/precipitation boundaries with literal selected source.

The actual Thompson column process and Registry slot are deliberately absent.
"""
import hashlib,re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import thompson_v842 as source
from woof.hex.mpasphys.resources_build import thompson_maths,_vendored_maths,MATHS
from woof.hex.mpasphys.driver_v842 import correct_source
from woof.hex.mpasphys.metal_thompson_v842 import launch_conversion,launch_precip
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
maths=thompson_maths(_vendored_maths())
assert hashlib.sha256(maths.encode()).hexdigest()==MATHS['mp_thompson'][1]
support=source._SUPPORT[:source._SUPPORT.index('// MPAS physics_mmm/mp_radar.F90:')]
conversion=source._conversion_source().replace('float** arrays,','unsigned long long* addresses,')
conversion=conversion.replace('    float* m_qv=arrays[0];','    float** arrays=reinterpret_cast<float**>(addresses);\n    float* m_qv=arrays[0];') if '    float* m_qv=arrays[0];' in conversion else conversion.replace('float* m_qv=arrays[0];','float** arrays=reinterpret_cast<float**>(addresses);\nfloat* m_qv=arrays[0];')
assert conversion.count('reinterpret_cast<float**>(addresses)')==2
host=HostModule('native_thompson_conversion',maths+support+conversion)
precip=source._PRECIP[:source._PRECIP.index('// MPAS mpas_atmphys_driver_microphysics.F:798-850')]
precip=precip.replace('float** work,float** diag,','unsigned long long* work_addresses,unsigned long long* diag_addresses,')
precip=precip.replace('    int col=','    float** work=reinterpret_cast<float**>(work_addresses);float** diag=reinterpret_cast<float**>(diag_addresses);\n    int col=',1)
precip_host={corrected:HostModule('native_thompson_precip_'+str(corrected),maths+support+correct_source(precip,source.CORRECT_MODE_SOURCE,('WB3',) if corrected else (),'thompson_v842')).get_function('mpas_thompson_precip') for corrected in (False,True)}
constants=dict(zip(source.CONVERSION_CONSTANTS,map(np.float32,(287.04,461.6,100000,287.04/717.5,461.6/287.04,9.81,10e-6,25e-6,25e-6))))
def pointers(values):return np.array([v.ctypes.data for v in values],np.uint64)
def invoke(kernel,count,args):kernel(((count+31)//32,),(32,),args)

def mapping(arrays):
    def group(prefix,names):return {name:arrays[prefix+name] for name in names}
    return (group('m_',source.MODEL_ARGUMENTS),group('g_',source.MESH_ARGUMENTS),
            group('d_',(*source.DIAG_ARGUMENTS,'surface_pressure','tend_sfc_pressure')),
            group('t_',source.TENDENCY_ARGUMENTS),group('f_',(*source.FIELD_ARGUMENTS,'rho','z')))

with MetalContext(batch_kernel_calls=False) as context:
    original=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=original(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context);rng=np.random.default_rng(842);total=0
    for nz in (2,9,129,256):
        nc=7;shape=(nz,nc)
        initial={name:rng.uniform(.001,.01,shape).astype(np.float32) for name in source._CONVERSION_ARRAYS}
        for name in ('d_surface_pressure','d_tend_sfc_pressure'):initial[name]=np.full(nc,98000,np.float32)
        initial['g_zgrid']=np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None]*100,(nz+1,nc)).copy()
        initial['m_w']=np.full((nz+1,nc),.1,np.float32)
        for name,value in [('m_theta_m',290),('m_rho_zz',1.15),('g_zz',.95),('d_exner',.98),('d_exner_base',.97),('d_pressure_base',90000),('d_pressure_p',100),('d_rtheta_base',300),('d_rtheta_p',10)]:initial[name].fill(value)
        for name in ('m_qc','m_qi','m_qg'):
            initial[name][0]=np.array([0,1,0x007fffff,0x00800000,2,0x30000000,0x31000000],np.uint32).view(np.float32)
        for top in sorted({1,nz//2,nz}):
            expected={name:value.copy() for name,value in initial.items()};actual={name:xp.asarray(value) for name,value in initial.items()}
            invoke(host.get_function('mpas_thompson_from'),nz*nc,(pointers(expected.values()),np.int32(nz),np.int32(nc),*constants.values()))
            launch_conversion('from',*mapping(actual),constants)
            for name in source._CONVERSION_ARRAYS:
                got=actual[name].to_numpy();assert np.array_equal(got.view(np.uint32),expected[name].view(np.uint32)),('from',nz,top,name)
            # Supply an explicit changed column result; this is not a microphysics solve.
            for name,delta in [('f_qv',-.00001),('f_qi',.00002),('f_nr',1),('f_th',.03)]:
                expected[name][...]=(expected[name]+np.float32(delta)).astype(np.float32)
                actual[name][...]=expected[name]
            invoke(host.get_function('mpas_thompson_to'),nc,(pointers(expected.values()),np.int32(nz),np.int32(nc),np.int32(top),np.float32(3),*constants.values()))
            launch_conversion('to',*mapping(actual),constants,mp_top_level=top,dt_dyn=3)
            for name in source._CONVERSION_ARRAYS:
                got=actual[name].to_numpy();assert np.array_equal(got.view(np.uint32),expected[name].view(np.uint32)),('to',nz,top,name)
            total+=sum(value.size for value in expected.values())*2
        print('conversion levels',nz,'top branches and all source words matched')
        fields={name:np.full(shape,value,np.float32) for name,value in [('qv',.01),('rho',1.15),('dz',100)]}
        fields['qv'][0,:3]=np.array([1,2,0x007fffff],np.uint32).view(np.float32)
        for corrected in (False,True):
            for bucket in (np.float32(0),np.float32(10),np.array(1,np.uint32).view(np.float32)):
                for acrain in (False,True):
                    work={name:np.ones(nc,np.float32) for name in source.SURFACE_ARGUMENTS[:7]}
                    diag={name:np.linspace(0,20,nc,dtype=np.float32) for name in source.PRECIP_ARGUMENTS};diag['i_rainnc']=np.zeros(nc,np.int32)
                    a_fields={name:xp.asarray(value) for name,value in fields.items()};a_work={name:xp.asarray(value) for name,value in work.items()};a_diag={name:xp.asarray(value) for name,value in diag.items()}
                    for phase in ('from','to'):
                        if phase=='to':
                            for index,name in enumerate(source.SURFACE_ARGUMENTS[:7]):
                                work[name][...]=np.arange(nc,dtype=np.float32)*np.float32(.5+index*.1);a_work[name][...]=work[name]
                            diag['rainnc'][...]=np.array([0,10,20,30,10,9,11],np.float32);a_diag['rainnc'][...]=diag['rainnc']
                        invoke(precip_host[corrected],nc,(*fields.values(),pointers(work.values()),pointers(diag[name] for name in source.PRECIP_ARGUMENTS),diag['i_rainnc'],np.int32(nz),np.int32(nc),np.int32(phase=='to'),np.int32(acrain),bucket))
                        launch_precip(phase,a_fields,a_work,a_diag,l_acrain=acrain,bucket=bucket,corrections=('WB3',) if corrected else ())
                        for group,device in ((work,a_work),(diag,a_diag)):
                            for name,value in group.items():
                                got=device[name].to_numpy();assert np.array_equal(got.view(np.uint32),value.view(np.uint32)),('precip',nz,phase,corrected,bucket,acrain,name)
        print('precipitation levels',nz,'WB3 off/on, reset/commit, buckets and counters matched')
    print('conversion compared words',total,'plus precipitation reset/commit cases; no full column/slot claim')
