"""Compare Noah glacier solver against host-compiled selected CUDA source."""
from pathlib import Path
import argparse,json,re,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.noah_v842 import _CORE_CUDA,_ENTRY_CUDA,COLUMN_LAYOUTS,table_dtype
from woof.hex.mpasphys.metal_noah_v842 import run_glacier_columns,load_native_resources
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
seen=set();todo=['gl_sflx_glacial']
while todo:
 name=todo.pop()
 if name in seen:continue
 seen.add(name)
 if name in functions:todo.extend(set(re.findall(r'\b((?:noah|gl)_\w+)\s*\(',functions[name][2]))-seen)
source=_CORE_CUDA.split('extern "C" __device__ void noah_finish_module_locations')[0]
for name,(start,end,_) in sorted(functions.items(),key=lambda p:p[1][0],reverse=True):
 if name not in seen:source=source[:start]+source[end:]
entry=_ENTRY_CUDA.split('extern "C" __global__ void replay_glacial(',1)[1].split('extern "C" __global__ void replay_driver(',1)[0]
entry='extern "C" __global__ void replay_glacial('+entry
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
    host=HostModule('mpas_v842_noah_init',header+alias+source+entry).get_function('replay_glacial')
    parser=argparse.ArgumentParser();parser.add_argument('--resources',type=Path)
    args=parser.parse_args()
    if not args.resources: raise ValueError('authenticated native resource packet is required')
    resources=json.loads(args.resources.read_text())
    payload=subprocess.check_output((resources['rust_decoder']['path'],resources['table_directory'],resources['landuse'],resources['soil_type']))
    words=np.frombuffer(payload,dtype=np.uint32,count=table_dtype().itemsize//4).copy()
    n=8;data=np.zeros((n,49),np.float32);ix=np.zeros((n,4),np.int32)
    names=COLUMN_LAYOUTS['glacial']['float'];ints=COLUMN_LAYOUTS['glacial']['int']
    for key,value in {'DT':60,'ZLVL':25,'LWDN':300,'SOLNET':400,'SFCPRS':95000,'SFCTMP':290,'Q2':.006,'TH2':294,'Q2SAT':.012,'DQSDT2':.0008,'ALB':.2,'SNOALB':.8,'TBOT':285,'Z0BRD':.1,'Z0':.1,'EMISSI':.96,'EMBRD':.96,'T1':292,'CH':.01,'RIBB':-.1}.items():data[:,names.index(key)]=value
    for key,value in {'IILOC':1,'JJLOC':1,'NSOIL':4,'ISICE':15}.items():ix[:,ints.index(key)]=value
    for k,depth in enumerate((.1,.3,.6,1.)):
        data[:,names.index(f'SLDPTH[{k}]')]=depth;data[:,names.index(f'STC[{k}]')]=285
    for c in range(n):
        if c&1:
            for key,value in {'SFCTMP':265,'TH2':267,'T1':268,'TBOT':270,'FFROZP':1}.items():data[c,names.index(key)]=value
            for k in range(4):data[c,names.index(f'STC[{k}]')]=269+k
        if c&2:data[c,names.index('SNEQV')]=.04;data[c,names.index('SNOWH')]=.2
        if c&4:data[c,names.index('PRCP')]=.001
    expected=data.copy();expected_i=ix.copy();status=np.zeros(n,np.int32)
    host((1,),(128,),(expected,expected_i,words,status,np.int32(n)))
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*a,**k):
            result=submit(*a,**k);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        device_tables=load_native_resources(resources,context)['tables']
        actual=xp.asarray(data);actual_i=xp.asarray(ix)
        got_status=run_glacier_columns(actual,actual_i,device_tables).to_numpy()
        got=actual.to_numpy();bad=np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32))
        bad_i=np.count_nonzero(actual_i.to_numpy()!=expected_i)
        print('native glacier selected source:',bad,'/',got.size,'float words;',bad_i,'integer differences; status',got_status.tolist())
        if bad:
            where=np.argwhere(got.view(np.uint32)!=expected.view(np.uint32))
            print([(int(c),names[int(k)],float(got[c,k]),float(expected[c,k])) for c,k in where[:20]])
        assert bad==0 and bad_i==0 and np.array_equal(got_status,status)
if __name__=='__main__': main()
