"""Compare captured native Fortran wrapper columns with the Metal wrapper."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.rrtmg_wrf461_oracle.lw_fixtures import read_fixture
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import load_tables,run_group
from woof.hex.mpasphys import metal_rrtmg_lw_v842 as lw
original_outputs=lw.outputs
captured={}
def capture_outputs(model,atmosphere,uf,df,uc,dc,*args,**kwargs):
    captured.update(zip(("uflx","dflx","uflxc","dflxc"),(uf,df,uc,dc)))
    return original_outputs(model,atmosphere,uf,df,uc,dc,*args,**kwargs)
lw.outputs=capture_outputs
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
fx=read_fixture(sys.argv[1]);bad=words=0;differences=[]
with MetalContext(batch_kernel_calls=False) as ctx:
    submit=ctx._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);tables=load_tables(ctx)
    for case in range(4):
        get=lambda key:fx[f'{case:03d}/{key}']
        flags=get('flags'); arrays=[get('model')[None],get('interfaces')[None],get('surface')[None],get('pin'),get('ozone')[None]]
        out=run_group(*[xp.asarray(np.ascontiguousarray(a)) for a in arrays],tables,icloud=int(flags[0]),overlap=int(flags[1]),idcor=int(flags[2]),julday=int(flags[3]),reqc=int(flags[4]),reqi=int(flags[5]),reqs=int(flags[6]),o3input=int(flags[7]))
        refs={'flux':get('flux')[None],'tendency':get('heating')[None],'radii':np.stack([get('rre_cloud'),get('rre_ice'),get('rre_snow')])[None]}
        for name in captured:
            actual=captured[name].to_numpy()[0];ref=get(name);neq=np.flatnonzero(actual.view(np.uint32)!=ref.view(np.uint32));bad+=len(neq);words+=ref.size;differences.extend((case,name,int(i),int(actual.view(np.uint32)[i]),int(ref.view(np.uint32)[i])) for i in neq);print(case,name,len(neq),'/',ref.size, [(int(i),float(actual[i]),float(ref[i])) for i in neq[:5]],flush=True)
        for name,ref in refs.items():
            actual=out[name].to_numpy();count=int(np.count_nonzero(actual.view(np.uint32)!=ref.view(np.uint32)));bad+=count;words+=ref.size
            print(case,name,count,'/',ref.size,flush=True)
            if count:
                for i in np.argwhere(actual.view(np.uint32)!=ref.view(np.uint32)):
                    ix=tuple(i);differences.append((case,name,int(np.ravel_multi_index(ix,ref.shape)),int(actual.view(np.uint32)[ix]),int(ref.view(np.uint32)[ix])));print('difference',ix,actual[ix],ref[ix],hex(int(actual.view(np.uint32)[ix])),hex(int(ref.view(np.uint32)[ix])),flush=True)
print('TOTAL',bad,'/',words,flush=True)
if len(sys.argv)>2 and sys.argv[2]=="--expect-observed":
    expected={(2,"dflx",27,0x42a4b41b,0x42a4b41a),(2,"dflxc",27,0x42a4b41b,0x42a4b41a),(2,"tendency",26,0xb80b8e80,0xb80b8e96),(2,"tendency",27,0xb7fd65e1,0xb7fd65b3)}
    if set(differences)!=expected:raise SystemExit(1)
    print("Observed Fortran differences match the recorded four words; this is not exact parity.")
elif bad:raise SystemExit(1)
