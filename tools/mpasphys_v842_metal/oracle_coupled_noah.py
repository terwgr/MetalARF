"""Selected-host-source Noah oracle for an actual coupled SlotCall.

Host copies serve only as a source comparator; the live service remains Metal.
The native driver uses the session's authenticated decoded table words and its
actual incoming four-word module-location state. No NVIDIA execution is implied.
"""
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_noah import compare_driver as driver
from woof.hex.mpasphys.noah_v842 import (_MPAS_IO_CUDA,_MPAS_INPUT_FIELDS,_MPAS_OUTPUT_FIELDS,
    _CORE_CUDA,_STATE_CUDA,_device_math_source)


class NoahOracle:
    def __init__(self):
        io=HostModule('coupled_noah_io',_device_math_source()+_MPAS_IO_CUDA)
        self.from_mpas=io.get_function('noah_from_mpas');self.to_mpas=io.get_function('noah_to_mpas')
        entry=driver.entry.replace('void replay_driver(', 'void replay_driver_with_state(').replace(
            'int* status,int n){','int* status,int n,const int* locations,int* calls){').replace(
            'FGSN_2D_arg);','FGSN_2D_arg,locations[2*c],locations[2*c+1],calls+c);')
        if 'calls+c);' not in entry:raise ValueError('native Noah driver entry changed')
        self.driver=HostModule('coupled_noah_driver',driver.header+driver.alias+driver.source+entry).get_function('replay_driver_with_state')
        source='extern "C" __device__ void noah_finish_module_locations('+_CORE_CUDA.split('extern "C" __device__ void noah_finish_module_locations(',1)[1]
        self.locations=HostModule('coupled_noah_locations',source+_STATE_CUDA).get_function('finish_module_locations')
        self.calls=0;self.words=0

    def compare(self,call,launch,service):
        p=call.pools;actual=p['fields'];n=int(p['count']);nv=actual['t'].shape[0];npres=actual['pres2_hyd'].shape[0]
        micro=call.options['config_microp_scheme']!='off';convection=call.options['config_convection_scheme']!='off'
        cache={};fields={}
        for name,kind in _MPAS_INPUT_FIELDS.items():
            enabled=micro if name in ('rainncv','sr') else convection
            value=actual['skintemp'] if kind=='optional_surface' and not enabled else actual[name]
            key=(id(value.buffer),value.byte_offset,value.shape,value.dtype.str)
            if key not in cache:
                a=value.to_numpy();cache[key]=np.ascontiguousarray(a.T) if a.ndim==2 else a
            fields[name]=cache[key]
        floats=np.empty((n,79),np.float32);integers=np.empty((n,7),np.int32);status=np.zeros(n,np.int32)
        params=(np.int32(nv),np.int32(npres),np.int32(0),np.int32(n),np.float32(p['dt_pbl']),
                np.float32(p['xice_threshold']),np.int32(micro),np.int32(convection),
                np.int32(p['isurban']),np.int32(p['isice']),np.int32(call.itimestep),np.int32(call.options['config_sfc_albedo']))
        self.from_mpas(((n+127)//128,),(128,),(floats,integers,*fields.values(),*params))
        positions=np.array([(i+1,1) for i in range(n)],np.int32);routes=np.zeros(n,np.int32)
        table=service.tables.to_numpy();state=service.module_locations.to_numpy()
        self.driver(((n+127)//128,),(128,),(floats,integers,table,status,np.int32(n),positions,routes))
        if np.any(status):raise RuntimeError('coupled Noah selected source returned fatal status '+str(status.tolist()))
        self.locations((1,),(1,),(positions,routes,state,np.int32(n)))
        self.to_mpas(((n+127)//128,),(128,),(floats,status,*(fields[name] for name in _MPAS_OUTPUT_FIELDS),np.int32(0),np.int32(n)))
        launch(call)
        for name in _MPAS_OUTPUT_FIELDS:
            got=actual[name].to_numpy();got=np.ascontiguousarray(got.T) if got.ndim==2 else got
            want=fields[name]
            if got.dtype!=want.dtype or got.shape!=want.shape or got.tobytes()!=want.tobytes():
                raise AssertionError('coupled Noah selected-source mismatch: '+name)
            self.words+=got.size
        got=service.module_locations.to_numpy()
        if got.tobytes()!=state.tobytes():raise AssertionError('coupled Noah module locations differ')
        self.words+=state.size;self.calls+=1
        return len(_MPAS_OUTPUT_FIELDS)
