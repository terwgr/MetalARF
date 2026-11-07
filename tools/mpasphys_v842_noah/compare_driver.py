"""Compare Noah land solver against host-compiled selected CUDA source."""
from pathlib import Path
import argparse,json,re,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.noah_v842 import _CORE_CUDA,_ENTRY_CUDA,COLUMN_LAYOUTS,table_dtype,_device_math_source
from woof.hex.mpasphys.metal_noah_v842 import run_driver_columns
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
seen=set();todo=['noah_driver_column']
while todo:
 name=todo.pop()
 if name in seen:continue
 seen.add(name)
 if name in functions:todo.extend(set(re.findall(r'\b((?:noah|gl)_\w+)\s*\(',functions[name][2]))-seen)
source=_CORE_CUDA.split('extern "C" __device__ void noah_finish_module_locations')[0]
for name,(start,end,_) in sorted(functions.items(),key=lambda p:p[1][0],reverse=True):
 if name not in seen:source=source[:start]+source[end:]
entry=_ENTRY_CUDA.split('extern "C" __global__ void replay_driver(',1)[1].split('// MPAS v8.4.2 mpas_atmphys_lsm_noahinit.F:76-207.',1)[0]
entry='extern "C" __global__ void replay_driver('+entry
entry=entry.replace('const NoahTables* ts','const unsigned int* ts').replace('NoahTables tables=*ts;', 'NoahTables tables=*(const NoahTables*)ts;')
header=_device_math_source()
alias=""

def main():
    parser=argparse.ArgumentParser();parser.add_argument('fixture',type=Path);args=parser.parse_args()
    host=HostModule('mpas_v842_noah_driver',header+alias+source+entry).get_function('replay_driver')
    meta=json.loads((args.fixture/'fixture.json').read_text())
    import hashlib
    for name,digest in meta['files'].items():
        assert hashlib.sha256((args.fixture/name).read_bytes()).hexdigest()==digest
    data=np.tile(np.fromfile(args.fixture/'input.f32',np.float32),(7,1))
    ix=np.tile(np.fromfile(args.fixture/'input.i32',np.int32),(7,1))
    words=np.fromfile(args.fixture/'tables.bin',np.uint32)
    names=COLUMN_LAYOUTS['driver']['float'];ints=COLUMN_LAYOUTS['driver']['int']
    def setf(c,k,v):data[c,names.index(k)]=v
    def seti(c,k,v):ix[c,ints.index(k)]=v
    # Native captured snow, warm no-snow, colder snow, glacier, sea ice,
    # water and a native soil-category error respectively.
    setf(1,'SNOW',0);setf(1,'SNOWH',0);setf(1,'SNOWC',0);setf(1,'T3D[0]',290);setf(1,'TSK',291)
    setf(2,'T3D[0]',265);setf(2,'TSK',268)
    seti(3,'IVGTYP',int(ix[3,ints.index('ISICE')]))
    setf(4,'XICE',.5)
    setf(5,'XLAND',2)
    seti(6,'ISLTYP',31)
    n=len(data);expected=data.copy();expected_i=ix.copy();statuses=np.zeros(n,np.int32)
    host((1,),(128,),(expected,expected_i,words,statuses,np.int32(n)))
    with MetalContext(batch_kernel_calls=False) as context:
        submit=context._submit_kernel_batch
        def sync(*a,**k):
            result=submit(*a,**k);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        x=xp.asarray(data); i=xp.asarray(ix);t=xp.asarray(words)
        status,routes=run_driver_columns(x,i,t,xp.asarray(np.ones((n,2),np.int32)))
        got=x.to_numpy();got_i=i.to_numpy();got_status=status.to_numpy();got_routes=routes.to_numpy()
    bad=np.argwhere(got.view(np.uint32)!=expected.view(np.uint32))
    bad_i=np.argwhere(got_i!=expected_i)
    print(json.dumps({'float_words':int(got.size),'float_mismatches':[(int(c),names[int(k)],float(got[c,k]),float(expected[c,k])) for c,k in bad],
                      'integer_words':int(got_i.size),'integer_mismatches':bad_i.tolist(),
                      'status':got_status.tolist(),'expected_status':statuses.tolist(),'routes':got_routes.tolist()}))
    if bad.size or bad_i.size or not np.array_equal(got_status,statuses):raise SystemExit(1)
if __name__=='__main__':main()
