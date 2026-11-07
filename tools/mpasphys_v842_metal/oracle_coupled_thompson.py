"""Selected-host-source comparison for a live bound Thompson SlotCall.

Host transfers are solely for this oracle; scheme production stays on Metal.
This uses the native source and authenticated native tables, not NVIDIA code.
"""
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.resources_build import thompson_maths,_vendored_maths
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_raw_tables


class ThompsonOracle:
    def __init__(self, directory, nz):
        self.calls=0
        self.words=0
        raw=load_raw_tables({'table_directory':directory})
        self.tables={n:v.view(np.float32 if n=='tnccn_act' else np.float64) for n,v in raw.items()}
        maths=thompson_maths(_vendored_maths())
        support=native._SUPPORT[:native._SUPPORT.index('// MPAS physics_mmm/mp_radar.F90:')]
        conv=native._conversion_source().replace('float** arrays,','unsigned long long* addresses,')
        conv=conv.replace('float* m_qv=arrays[0];','float** arrays=reinterpret_cast<float**>(addresses);\nfloat* m_qv=arrays[0];')
        if conv.count('reinterpret_cast<float**>(addresses)')!=2:raise ValueError('native Thompson conversion source changed')
        module=HostModule('coupled_thompson_conversion',maths+support+conv)
        self.convert_from=module.get_function('mpas_thompson_from')
        self.convert_to=module.get_function('mpas_thompson_to')
        prec=native._PRECIP[:native._PRECIP.index('// MPAS mpas_atmphys_driver_microphysics.F:798-850')]
        prec=prec.replace('float** work,float** diag,','unsigned long long* work_addresses,unsigned long long* diag_addresses,')
        prec=prec.replace('    int col=','    float** work=reinterpret_cast<float**>(work_addresses);float** diag=reinterpret_cast<float**>(diag_addresses);\n    int col=',1)
        self.precip=HostModule('coupled_thompson_precip',maths+support+prec).get_function('mpas_thompson_precip')
        declaration=','.join(f"const {'float' if n=='tnccn_act' else 'double'}* input_{n}" for n in native.TABLE_ARGUMENTS)
        assignment='\n'.join(f'm.{n}=input_{n};' for n in native.TABLE_ARGUMENTS)
        source=f'#define THOMPSON_COLUMN_KMAX {nz}\n'+maths+'\n'+native._SUPPORT+native._COLUMN+native._DRIVER.replace('TABLE_DECLARATIONS',declaration).replace('TABLE_ASSIGNMENTS',assignment)
        self.step=HostModule('coupled_thompson_process',source).get_function('mpas_thompson_step')

    @staticmethod
    def _pointers(values):return np.array([v.ctypes.data for v in values],np.uint64)

    def compare(self,call,launch):
        p=call.pools['thompson_v842'];nz,nc=p['fields']['qv'].shape
        if 'WB3' in call.corrections:raise ValueError('coupled Thompson oracle covers native precipitation only')
        cached={};expected={}
        for group in ('model','mesh','diagnostics','tendencies','fields','surface','precip'):
            expected[group]={}
            for name,value in p[group].items():
                key=(id(value.buffer),value.byte_offset,value.shape,value.dtype.str)
                if key not in cached:cached[key]=value.to_numpy()
                expected[group][name]=cached[key]
        m,g,d,t,f,s,r=(expected[n] for n in ('model','mesh','diagnostics','tendencies','fields','surface','precip'))
        arrays=tuple(m[n] for n in native.MODEL_ARGUMENTS)+tuple(g[n] for n in native.MESH_ARGUMENTS)
        arrays+=tuple(d[n] for n in native.DIAG_ARGUMENTS)+tuple(t[n] for n in native.TENDENCY_ARGUMENTS)
        arrays+=tuple(f[n] for n in native.FIELD_ARGUMENTS)+tuple(f[n] for n in ('rho','z'))
        arrays+=tuple(d[n] for n in ('surface_pressure','tend_sfc_pressure'))
        constants=tuple(np.float32(p['constants'][n]) for n in native.CONVERSION_CONSTANTS)
        def invoke(fn,count,args):fn(((count+31)//32,),(32,),args)
        def precip(phase):
            invoke(self.precip,nc,(f['qv'],f['rho'],f['dz'],self._pointers(s[n] for n in native.SURFACE_ARGUMENTS[:7]),
                self._pointers(r[n] for n in native.PRECIP_ARGUMENTS),r['i_rainnc'],np.int32(nz),np.int32(nc),
                np.int32(phase),np.int32(p['l_acrain']),np.float32(call.options['config_bucket_rainnc'])))
        precip(0)
        invoke(self.convert_from,nz*nc,(self._pointers(arrays),np.int32(nz),np.int32(nc),*constants))
        for _ in range(p['n_microp']):
            invoke(self.step,nc,tuple(f[n] for n in native.FIELD_ARGUMENTS)+tuple(s[n] for n in native.SURFACE_ARGUMENTS)
                +(np.int32(nz),np.int32(nc),np.float32(p['dt_microp']))
                +tuple(np.int32(p[n]) for n in ('has_reqc','has_reqi','has_reqs'))
                +tuple(self.tables[n] for n in native.TABLE_ARGUMENTS))
        precip(1)
        invoke(self.convert_to,nc,(self._pointers(arrays),np.int32(nz),np.int32(nc),
                                  np.int32(p['mp_top_level']),np.float32(call.dt),*constants))
        launch(call)
        words=0
        for group in expected:
            for name,want in expected[group].items():
                got=p[group][name].to_numpy()
                if got.dtype!=want.dtype or got.shape!=want.shape or got.tobytes()!=want.tobytes():
                    raise AssertionError('coupled Thompson source differs: '+group+'.'+name)
                words+=want.size
        self.calls+=1;self.words+=words
        return words
