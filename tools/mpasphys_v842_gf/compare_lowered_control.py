"""Compare checked GF loop rewrites through the real source column entry."""
from pathlib import Path
import ctypes,os,subprocess,sys,tempfile
import numpy as np
from woof.hex.mpasphys import gf_mpas_v842 as native
from tools.mpasphys_v842_gf.lower_control import procedure,lower_partial,lower_kbcon,lower_aa0,lower_ktop,lower_up_moisture,lower_dd_edt,lower_forcing,lower_wrapped_exit,lower_shallow
import re

with tempfile.TemporaryDirectory(prefix='native-gf-control-') as directory:
    root=Path(directory);libs=[]
    for variant in ('original','lowered'):
        body=native._SCHEME_SOURCE
        if variant=='lowered':
            names=re.findall(r"__device__\s+(?:void|GFReal)\s+(\w+)\s*\([^;{]*\)\s*\{",body)
            total=0
            for name in names:
                before=procedure(name);after,count=(lower_kbcon() if name=="cup_kbcon" else lower_aa0() if name=="cup_up_aa0" else lower_ktop() if name=="cup_ktop" else lower_up_moisture() if name=="cup_up_moisture" else lower_dd_edt() if name=="cup_dd_edt" else lower_forcing() if name=="cup_forcing_ens_3d" else lower_shallow() if name=="cup_gf_sh" else lower_wrapped_exit(name)[:2] if name=="cup_gf" else lower_partial(name));assert body.count(before)==1;body=body.replace(before,after,1);total+=count
            remaining=re.findall(r"\bgoto\s+(\w+)\s*;",body)
            print("checked direct exits",total,"remaining jumps",len(remaining),flush=True)
            if remaining:raise ValueError("unlowered GF jumps: "+str(remaining[:10]))
        text='#define GF_HOST_WITNESS\n'+native._SUPPORT+body+native._ENTRY
        source=root/(variant+'.cpp');library=root/(variant+'.dylib');source.write_text(text)
        result=subprocess.run(['clang++','-std=c++17','-O1','-ffp-contract=off','-shared','-fPIC',str(source),'-o',str(library)],capture_output=True,text=True)
        if result.returncode:raise RuntimeError(variant+' compile: '+result.stderr[-5000:])
        libs.append(library)
    nz=32;levels=np.zeros((15,nz),np.float32);levels[0]=5;levels[1]=1;levels[3]=np.linspace(290,220,nz);levels[4]=np.linspace(.008,.0001,nz);levels[5]=np.linspace(100000,10000,nz);levels[6]=np.linspace(1,.55,nz);levels[7]=np.linspace(1.1,.2,nz);levels[8]=400;levels[9]=np.linspace(101000,10500,nz)
    surface=np.array([0,20,0,1,200,12000,1.4e8,60],np.float32);isurface=np.array([4],np.int32);control=np.array([1,0,8,1],np.int32);constants=np.array(native.MPAS_DRIVER_CONSTANTS,np.float32)
    if os.environ.get('GF_ACTIVE_CASE'):
        from tools.gf_wrf461_oracle import gf_regime_columns as fixture
        column=fixture.build_column(fixture.REGIMES[1],dx=3000,ishallow=1,ichoice=0,dt=60)
        nz=len(column.lev['t']);levels=np.empty((15,nz),np.float32)
        for i,name in enumerate(native.LEVEL_INPUTS):
            levels[i]=column.p8w[:-1] if name=='p8w' else column.lev['qv' if name=='q' else name]
        surface=np.array([column.sca['ht'],column.sca['hfx'],column.sca['qfx'],column.sca['xland'],200,3000,3000**2,column.dt],np.float32)
        isurface=np.array([column.kpbl],np.int32)

    results=[]
    for library in libs:
        fn=ctypes.CDLL(str(library)).gf_port_host;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*8;fn.restype=None
        ins=(levels.copy(),surface.copy(),isurface.copy(),np.zeros((6,nz),np.float32),np.zeros(6,np.float32),np.zeros(3,np.int32),control.copy(),constants.copy())
        fn(nz,*(x.ctypes.data for x in ins));results.append(ins)
    words=bad=0
    for a,b in zip(*results):
        x=a.view(np.uint32);y=b.view(np.uint32);ix=np.flatnonzero(x!=y);words+=x.size;bad+=len(ix)
        if len(ix):print('differences',[(int(i),hex(int(x.flat[i])),hex(int(y.flat[i]))) for i in ix[:10]])
    print('output nonzero counts',*[int(np.count_nonzero(a)) for a in results[0][3:6]],flush=True)
    print('native GF column source loop rewrite differences',bad,'/',words,flush=True)
    if bad:raise SystemExit(1)
