"""Compare native-source pool/driver composition and direct restart carry."""
from pathlib import Path
import argparse,re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_noah import compare_driver as d
from woof.hex.mpasphys.noah_v842 import _MPAS_IO_CUDA,_MPAS_INPUT_FIELDS,_MPAS_OUTPUT_FIELDS,COLUMN_LAYOUTS,_device_math_source
from woof.hex.mpasphys.metal_noah_v842 import MetalMpasNoah
from types import SimpleNamespace
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);args=p.parse_args()
base=np.fromfile(args.fixture/'input.f32',np.float32); categories=np.fromfile(args.fixture/'input.i32',np.int32)
tables=np.fromfile(args.fixture/'tables.bin',np.uint32)
n=7;source=np.tile(base,(n,1));ints=np.tile(categories,(n,1));names=COLUMN_LAYOUTS['driver']['float'];inames=COLUMN_LAYOUTS['driver']['int']
source[1,names.index('SNOW')]=0;source[1,names.index('SNOWH')]=0;source[1,names.index('SNOWC')]=0;source[1,names.index('T3D[0]')]=290;source[1,names.index('TSK')]=291
source[2,names.index('T3D[0]')]=265;source[2,names.index('TSK')]=268
ints[3,inames.index('IVGTYP')]=ints[3,inames.index('ISICE')]
source[4,names.index('XICE')]=.5;source[5,names.index('XLAND')]=2;ints[6,inames.index('ISLTYP')]=31
fields={}
for name,kind in _MPAS_INPUT_FIELDS.items():
    width={'atmosphere':2,'pressure':2,'soil':4}.get(kind)
    fields[name]=np.zeros((n,width) if width else (n,),np.int32 if kind=='integer_surface' else np.float32)
start=_MPAS_IO_CUDA.split('void noah_from_mpas(',1)[1].split('\n}',1)[0]
for idx,name,sub in re.findall(r'x\[(\d+)\]=(\w+)\[(.*?)\];',start):
    if name not in fields:continue
    idx=int(idx)
    if fields[name].ndim==2:
        k=int(sub.rsplit('+',1)[1]);fields[name][:,k]=source[:,idx]
    else:fields[name][:]=source[:,idx]
fields['ivgtyp'][:]=ints[:,0];fields['isltyp'][:]=ints[:,1]
fields['gsw'][:]=source[:,14]*(np.float32(1)-source[:,21]);fields['rainncv'][:]=source[:,32];fields['raincv'][:]=0;fields['sr'][:]=source[:,51]
io=HostModule('noah_composed_io',_device_math_source()+_MPAS_IO_CUDA)
from_host=io.get_function('noah_from_mpas');to_host=io.get_function('noah_to_mpas')
driver=HostModule('noah_composed_driver',d.header+d.alias+d.source+d.entry).get_function('replay_driver')
params=(np.int32(2),np.int32(2),np.int32(0),np.int32(n),np.float32(base[33]),np.float32(base[72]),np.int32(1),np.int32(1),np.int32(categories[2]),np.int32(categories[3]),np.int32(categories[4]),np.int32(categories[5]))
options=dict(dt_pbl=float(base[33]),itimestep=int(categories[4]),isurban=int(categories[2]),isice=int(categories[3]),config_sfc_albedo=int(categories[5]),xice_threshold=float(base[72]),microphysics_enabled=True,convection_enabled=True)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    device={k:xp.asarray(v.T.copy() if v.ndim==2 else v) for k,v in fields.items()};state=xp.asarray(np.array([77,88,99,111],np.int32));table=xp.asarray(tables)
    slot=MetalMpasNoah(table,state)
    for step in range(2):
        x=np.empty((n,79),np.float32);ix=np.empty((n,7),np.int32);status=np.zeros(n,np.int32)
        from_host((1,),(128,),(x,ix,*fields.values(),*params))
        driver((1,),(128,),(x,ix,tables,status,np.int32(n)))
        to_host((1,),(128,),(x,status,*(fields[k] for k in _MPAS_OUTPUT_FIELDS),np.int32(0),np.int32(n)))
        pools=dict(fields=device,tables=table,level_first=True,start=0,count=n,dt_pbl=options['dt_pbl'],xice_threshold=options['xice_threshold'],isurban=options['isurban'],isice=options['isice'])
        call=SimpleNamespace(slot='land_surface',scheme='sf_noah',pools=pools,itimestep=options['itimestep'],options={'config_sfc_albedo':options['config_sfc_albedo'],'config_microp_scheme':'on','config_convection_scheme':'on'})
        try:
            slot.launch_slot(call)
            raise AssertionError('expected native fatal status')
        except RuntimeError as error:
            assert '2438' in str(error),error
        bad=[];total=0
        for name in _MPAS_OUTPUT_FIELDS:
            actual=device[name].to_numpy();actual=actual.T if actual.ndim==2 else actual;expected=fields[name];total+=expected.size
            if not np.array_equal(actual.view(np.uint32),expected.view(np.uint32)):bad.append(name)
        print('step',step,'pool words',total,'differing fields',bad,'status',status.tolist(),'locations',state.to_numpy().tolist())
        assert not bad and np.array_equal(state.to_numpy(),np.array([7,1,4,1],np.int32))
        if step==0:
            # A direct pool/state snapshot and restore, with no native session.
            saved=slot.restart_state()
            device={k:xp.asarray(v.to_numpy()) for k,v in device.items()};state=xp.asarray(np.zeros(4,np.int32))
            slot=MetalMpasNoah(table,state);slot.restore_restart_state(saved)
