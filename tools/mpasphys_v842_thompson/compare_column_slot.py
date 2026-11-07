"""Compare the diagnostics-off native Thompson slot with selected source."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.resources_build import thompson_maths,_vendored_maths
from woof.hex.mpasphys.driver_v842 import correct_source
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_raw_tables
from woof.hex.mpasphys.metal_thompson_column_v842 import launch_slot
from woof.hex.mpasphys.metal_registry_v842 import MetalSlotRegistry
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

nz,nc=4,2
radii=int(sys.argv[2]) if len(sys.argv)>2 else 1
if radii not in (0,1):raise ValueError('radius flag must be zero or one')
raw=load_raw_tables({'table_directory':sys.argv[1]}); tables={n:v.view(np.float32 if n=='tnccn_act' else np.float64) for n,v in raw.items()}
maths=thompson_maths(_vendored_maths()); support=native._SUPPORT[:native._SUPPORT.index('// MPAS physics_mmm/mp_radar.F90:')]
conv=native._conversion_source().replace('float** arrays,','unsigned long long* addresses,')
conv=conv.replace('float* m_qv=arrays[0];','float** arrays=reinterpret_cast<float**>(addresses);\nfloat* m_qv=arrays[0];')
assert conv.count('reinterpret_cast<float**>(addresses)')==2
convhost=HostModule('native_thompson_slot_conv',maths+support+conv)
prec=native._PRECIP[:native._PRECIP.index('// MPAS mpas_atmphys_driver_microphysics.F:798-850')]
prec=prec.replace('float** work,float** diag,','unsigned long long* work_addresses,unsigned long long* diag_addresses,')
prec=prec.replace('    int col=','    float** work=reinterpret_cast<float**>(work_addresses);float** diag=reinterpret_cast<float**>(diag_addresses);\n    int col=',1)
phost={b:HostModule('native_thompson_slot_prec_'+str(b),maths+support+correct_source(prec,native.CORRECT_MODE_SOURCE,('WB3',) if b else (),'thompson_v842')).get_function('mpas_thompson_precip') for b in (False,True)}
decl=','.join(f"const {'float' if n=='tnccn_act' else 'double'}* input_{n}" for n in native.TABLE_ARGUMENTS)
assign='\n'.join(f'm.{n}=input_{n};' for n in native.TABLE_ARGUMENTS)
source=f'#define THOMPSON_COLUMN_KMAX {nz}\n'+maths+'\n'+native._SUPPORT+native._COLUMN+native._DRIVER.replace('TABLE_DECLARATIONS',decl).replace('TABLE_ASSIGNMENTS',assign)
step=HostModule('native_thompson_slot_process',source).get_function('mpas_thompson_step')
constants=dict(zip(native.CONVERSION_CONSTANTS,map(np.float32,(287.04,461.6,100000,287.04/717.5,461.6/287.04,9.81,10e-6,25e-6,25e-6))))
def pointers(values):return np.array([v.ctypes.data for v in values],np.uint64)
def invoke(fn,n,args):fn(((n+31)//32,),(32,),args)
def mapping(arrays):
    def group(prefix,names):return {name:arrays[prefix+name] for name in names}
    return (group('m_',native.MODEL_ARGUMENTS),group('g_',native.MESH_ARGUMENTS),group('d_',(*native.DIAG_ARGUMENTS,'surface_pressure','tend_sfc_pressure')),group('t_',native.TENDENCY_ARGUMENTS),group('f_',(*native.FIELD_ARGUMENTS,'rho','z')))
initial={n:np.zeros((nz,nc),np.float32) for n in native._CONVERSION_ARRAYS}
initial['g_zgrid']=np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None]*500,(nz+1,nc)).copy()
for n in ('d_surface_pressure','d_tend_sfc_pressure'):initial[n]=np.full(nc,98000,np.float32)
for n,v in (('re_cloud',11e-6),('re_ice',22e-6),('re_snow',33e-6)):initial['f_'+n].fill(v)
for n,v in [('m_theta_m',290),('m_rho_zz',1.15),('g_zz',.95),('d_exner',.98),('d_exner_base',.97),('d_pressure_base',90000),('d_pressure_p',100),('d_rtheta_base',300),('d_rtheta_p',10),('m_qv',.006),('m_qc',.0001),('m_qr',.00008),('m_qi',.00001),('m_qs',.00001),('m_qg',.00001),('m_ni',900),('m_nr',2000),('m_w',.1)]:initial[n].fill(v)
initial['m_qv'][:,1]*=np.float32(.8)
def surface():
    x={n:np.zeros(nc,np.float32) for n in native.SURFACE_ARGUMENTS};x['ntc'][:]=100e6;return x
def precip():
    x={n:np.full(nc,2,np.float32) for n in native.PRECIP_ARGUMENTS};x['i_rainnc']=np.zeros(nc,np.int32);return x
with MetalContext(batch_kernel_calls=False) as context:
    xp=MetalArrayModule(context);device_tables={n:xp.asarray(v.view(np.float32) if n=='tnccn_act' else v) for n,v in raw.items()};device_tables['_file_sha256']=native.TABLE_FILE_SHA256.copy()
    for corrected in (False,True):
        expected={n:v.copy() for n,v in initial.items()};actual={n:xp.asarray(v) for n,v in initial.items()}
        work=surface();diag=precip();awork={n:xp.asarray(v) for n,v in work.items()};adiag={n:xp.asarray(v) for n,v in diag.items()}
        models,mesh,diagnostics,tendencies,fields=mapping(expected)
        am,ag,ad,at,af=mapping(actual)
        def hostprec(phase):
            invoke(phost[corrected],nc,(fields['qv'],fields['rho'],fields['dz'],pointers(work[n] for n in native.SURFACE_ARGUMENTS[:7]),pointers(diag[n] for n in native.PRECIP_ARGUMENTS),diag['i_rainnc'],np.int32(nz),np.int32(nc),np.int32(phase),np.int32(1),np.float32(1)))
        hostprec(0)
        invoke(convhost.get_function('mpas_thompson_from'),nz*nc,(pointers(expected.values()),np.int32(nz),np.int32(nc),*constants.values()))
        for _ in range(2):invoke(step,nc,tuple(fields[n] for n in native.FIELD_ARGUMENTS)+tuple(work[n] for n in native.SURFACE_ARGUMENTS)+(np.int32(nz),np.int32(nc),np.float32(1),np.int32(radii),np.int32(radii),np.int32(radii))+tuple(tables[n] for n in native.TABLE_ARGUMENTS))
        hostprec(1)
        invoke(convhost.get_function('mpas_thompson_to'),nc,(pointers(expected.values()),np.int32(nz),np.int32(nc),np.int32(2),np.float32(2),*constants.values()))
        p=dict(model=am,mesh=ag,diagnostics=ad,tendencies=at,fields=af,constants=constants,surface=awork,precip=adiag,tables=device_tables,dt_microp=np.float32(1),n_microp=2,mp_top_level=2,l_acrain=True,has_reqc=radii,has_reqi=radii,has_reqs=radii)
        registry=MetalSlotRegistry();registry.register_metal('microphysics','mp_thompson',launch_slot)
        call=SimpleNamespace(slot='microphysics',scheme='mp_thompson',backend='metal',flags=SimpleNamespace(diagnostics=False),pools={'thompson_v842':p},options={'config_bucket_rainnc':np.float32(1)},corrections=frozenset({'WB3'} if corrected else ()),dt=2)
        if not corrected:
            before={n:v.to_numpy().copy() for n,v in actual.items()}
            for flag in ('diagnostics','nest_look'):
                call.flags.diagnostics=flag=='diagnostics';p['nest_look']=True if flag=='nest_look' else None
                try:registry.run(call)
                except NotImplementedError:pass
                else:raise AssertionError(flag+' must be refused')
                for name,value in before.items():assert np.array_equal(actual[name].to_numpy().view(np.uint32),value.view(np.uint32)),name
            call.flags.diagnostics=False;p['nest_look']=None
        registry.run(call)
        bad=words=0
        for name,value in expected.items():
            got=actual[name].to_numpy();ix=np.flatnonzero(got.view(np.uint32)!=value.view(np.uint32));bad+=len(ix);words+=value.size
            if len(ix):print(name,'differences',len(ix),'first',[(int(i),hex(int(got.view(np.uint32).flat[i])),hex(int(value.view(np.uint32).flat[i]))) for i in ix[:3]],flush=True)
        for target,ref in ((awork,work),(adiag,diag)):
            for name,value in ref.items():
                got=target[name].to_numpy();ix=np.flatnonzero(got.view(np.uint32)!=value.view(np.uint32));bad+=len(ix);words+=value.size
                if len(ix):print(name,'differences',len(ix),flush=True)
        print('radii',radii,'WB3',corrected,'words',words,'differences',bad,flush=True)
        if bad:raise SystemExit(1)
