"""Compare native land-use initialization with host-compiled selected source."""
from pathlib import Path
import argparse,json,re,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.noah_v842 import _LANDUSE_CUDA,table_dtype
from woof.hex.mpasphys.metal_noah_v842 import initialize_landuse,load_native_resources
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

source=_LANDUSE_CUDA[_LANDUSE_CUDA.index('extern "C" __global__'):]
source=source.replace('const LanduseTables* ts','const float* ts')
for j,name in enumerate(('ALBD','SLMO','SFEM','SFZ0','THERIN','SCFX','SFHC')):
    source=re.sub(r'ts->'+name+r'\[([^\]]+)\]',lambda m:f'ts[{j*1200}+({m.group(1)})]',source)
source='''#define AD(a,b) __fadd_rn(a,b)
#define SU(a,b) __fsub_rn(a,b)
#define MU(a,b) __fmul_rn(a,b)
#define DV(a,b) __fdiv_rn(a,b)
'''+source

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resources',type=Path)
    args=parser.parse_args()
    host=HostModule('mpas_v842_noah_landuse',source).get_function('replay_landuse')
    n=16; rng=np.random.default_rng(842)
    # Distinct values in every table slot expose season/category selection.
    table=rng.uniform(.02,.99,7*12*100).astype(np.float32)
    if args.resources:
        resources=json.loads(args.resources.read_text())
        payload=subprocess.check_output((resources['rust_decoder']['path'],resources['table_directory'],resources['landuse'],resources['soil_type']))
        table=np.frombuffer(payload,dtype=np.float32,offset=table_dtype().itemsize).copy()
    f=rng.uniform(.01,.9,(n,14)).astype(np.float32)
    f[:,0]=np.where(np.arange(n)%2, -30,30)
    f[:,2]=np.where(np.arange(n)%3==0,.7,.2)
    f[:,3]=np.linspace(0,1,n,dtype=np.float32);f[:,13]=.4
    ix=np.zeros((n,9),np.int32);ix[:,0]=np.arange(n)%10+1
    ix[:,1]=np.array([1,104,105,287,288,365,20,200]*2,np.int32)
    ix[:,2]=16;ix[:,3]=24;ix[:,4]=(np.arange(n)%7==0)
    for j in range(5,9):ix[:,j]=((np.arange(n)>> (j-5))&1)
    expected=f.copy();status=np.zeros(n,np.int32)
    host((1,),(128,),(expected,ix,table,status,np.int32(n)))
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def synchronous(*a,**k):
            result=submit(*a,**k);result.wait();return result
        context._submit_kernel_batch=synchronous;xp=MetalArrayModule(context)
        device_table=load_native_resources(resources,context)['landuse_tables'] if args.resources else xp.asarray(table)
        data=xp.asarray(f); output=initialize_landuse(data,xp.asarray(ix),device_table)
        actual=data.to_numpy(); got_status=output.to_numpy()
        bad=np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32))
        print('landuse selected source:',bad,'/',actual.size,'float words; status',got_status.tolist())
        assert bad==0 and np.array_equal(got_status,status)
        bad_ix=ix.copy();bad_ix[1,0]=101
        guarded=xp.asarray(f);guard_status=initialize_landuse(guarded,xp.asarray(bad_ix),device_table).to_numpy()
        assert guard_status[1]==-1 and np.array_equal(guarded.to_numpy()[1],f[1])
        print('out-of-domain category is explicitly refused before its table read')
if __name__=='__main__': main()
