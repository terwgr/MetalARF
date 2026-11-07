"""Compare Noah initialization against host-compiled selected CUDA source."""
from pathlib import Path
import argparse,json,re,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.noah_v842 import _CORE_CUDA,_ENTRY_CUDA,table_dtype
from woof.hex.mpasphys.metal_noah_v842 import initialize_soil,load_native_resources
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

# Keep the original source's control flow and arithmetic; remove unrelated
# bodies so the host compiler only needs the initialization call graph.
pat=re.compile(r'__device__\s+(?:__forceinline__\s+)?(?:void|float|int)\s+(\w+)\s*\(')
functions={}
for m in pat.finditer(_CORE_CUDA):
 depth=1;i=m.end()
 while depth:
  if _CORE_CUDA[i]=='(':depth+=1
  elif _CORE_CUDA[i]==')':depth-=1
  i+=1
 while _CORE_CUDA[i].isspace():i+=1
 if _CORE_CUDA[i]!='{':continue
 start=m.start();j=i+1;depth=1
 while depth:
  if _CORE_CUDA[j]=='{':depth+=1
  elif _CORE_CUDA[j]=='}':depth-=1
  j+=1
 functions[m.group(1)]=(start,j,_CORE_CUDA[i:j])
seen=set();todo=['noah_init_column']
while todo:
 name=todo.pop()
 if name in seen:continue
 seen.add(name)
 if name in functions:todo.extend(set(re.findall(r'\b((?:noah|gl)_\w+)\s*\(',functions[name][2]))-seen)
source=_CORE_CUDA.split('extern "C" __device__ void noah_finish_module_locations')[0]
for name,(start,end,_) in sorted(functions.items(),key=lambda p:p[1][0],reverse=True):
 if name not in seen:source=source[:start]+source[end:]
entry=_ENTRY_CUDA.split('extern "C" __global__ void replay_init(',1)[1].split('// MPAS v8.4.2 physics_wrf/module_sf_noahlsm.F:4526-4760.',1)[0]
entry='extern "C" __global__ void replay_init('+entry
entry=entry.replace('const NoahTables* ts','const unsigned int* ts').replace('NoahTables tables=*ts;', 'NoahTables tables=*(const NoahTables*)ts;')
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()
alias='''\n#define AD(a,b) FADD(a,b)
#define SU(a,b) FSUB(a,b)
#define MU(a,b) FMUL(a,b)
#define DV(a,b) FDIV(a,b)
#define r_log gfk_log
#define r_exp gfk_exp
#define r_pow gfk_pow
'''

def main():
    host=HostModule('mpas_v842_noah_init',header+alias+source+entry).get_function('replay_init')
    parser=argparse.ArgumentParser();parser.add_argument('--resources',type=Path)
    args=parser.parse_args()
    n=8;data=np.zeros((n,15),np.float32);ix=np.zeros((n,4),np.int32)
    data[:,:4]=np.array([250,270,273.2,285],np.float32)
    data[:,4:8]=np.array([.23,.28,.31,.35],np.float32)
    data[:,8:12]=np.array([.08,.13,.20,.30],np.float32)
    data[:,12]=.7;data[:,13]=np.arange(n,dtype=np.float32);data[:,14]=.03
    ix[:,0]=[1,2,3,4,5,6,7,8];ix[:,1]=[1,2,3,4,5,6,7,8];ix[:,2]=[0,0,0,0,1,1,0,0];ix[:,3]=[0,1,0,1,0,1,0,1]
    tab=np.zeros(1,dtype=table_dtype());tab['GRAVITY']=9.80616
    tab['MAXALB']=np.linspace(20,80,50,dtype=np.float32)
    tab['BB']=4.1;tab['MAXSMC']=.45;tab['SATPSI']=.3
    if args.resources:
        resources=json.loads(args.resources.read_text())
        payload=subprocess.check_output((resources['rust_decoder']['path'],resources['table_directory'],resources['landuse'],resources['soil_type']))
        words=np.frombuffer(payload,dtype=np.uint32,count=table_dtype().itemsize//4).copy()
    else:
        words=tab.view(np.uint32).reshape(-1)
    expected=data.copy();status=np.zeros(n,np.int32)
    host((1,),(128,),(expected,ix,words,status,np.int32(n)))
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*a,**k):
            result=submit(*a,**k);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        device_tables=load_native_resources(resources,context)['tables'] if args.resources else xp.asarray(words)
        actual=xp.asarray(data);got_status=initialize_soil(actual,xp.asarray(ix),device_tables).to_numpy()
        got=actual.to_numpy();bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))
        print('native initialization selected source:',bad,'/',got.size,'float words; status',got_status.tolist())
        assert bad==0 and np.array_equal(got_status,status)
        bad_ix=ix.copy();bad_ix[0,1]=31;bad_ix[1,1]=0
        refused=xp.asarray(data);ref=initialize_soil(refused,xp.asarray(bad_ix),device_tables).to_numpy()
        assert ref[0]==-1 and ref[1]==140 and np.array_equal(refused.to_numpy()[:2],data[:2])
        print('native lower soil refusal and undefined upper table index preserved')
if __name__=='__main__': main()
