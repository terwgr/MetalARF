#!/usr/bin/env python3
"""Compare Goddard's five stages to the host-built CUDA operation graph."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
import host_backend
host_backend.CXX_FLAGS += ("-w",)
from host_backend import HostModule
from woof.core import kernels
from woof.core.radiation_tables import radiation_table_host_words
from woof.metal.array import MetalArrayModule
from woof.metal.goddard_family import _submit
from woof.metal.goddard_workspace import workspace_words


def fixture(nz,nc):
    rng=np.random.default_rng(5110)
    z=np.broadcast_to(np.linspace(0.,16000.,nz+1,dtype=np.float32)[:,None],(nz+1,nc)).copy()
    pm=(np.float32(100000.)*np.exp(-z/np.float32(7400.))).astype(np.float32)
    p=np.sqrt(pm[:-1]*pm[1:]).astype(np.float32)
    zm=np.float32(.5)*(z[:-1]+z[1:])
    t=np.maximum(np.float32(215.),np.float32(300.)-np.float32(.006)*zm)
    qv=(np.float32(.014)*np.exp(-zm/np.float32(2500.))).astype(np.float32)
    cf=rng.choice(np.array([0.,.15,.6,1.],np.float32),(nz,nc))
    rho=(np.float32(.622)*p)/(np.float32(287.04)*t*(qv+np.float32(.622)))
    mass=[rng.uniform(0.,2.e-4,(nz,nc)).astype(np.float32)*(cf>0.) for _ in range(6)]
    radii=[rng.uniform(2.,250.,(nz,nc)).astype(np.float32) for _ in range(6)]
    lat=np.linspace(-75.,75.,nc,dtype=np.float32)
    cosine=np.resize(np.array([.9,.05,-.2,0.,.5,.8],np.float32),nc)
    surface=[lat,cosine,np.full(nc,300.,np.float32),np.full(nc,.97,np.float32),np.full(nc,.2,np.float32)]
    exner=((p/np.float32(1.e5))**np.float32(287./1004.)).astype(np.float32)
    return dict(t=t,z=z,fnm=np.full(nz,.5,np.float32),fnp=np.full(nz,.5,np.float32)),[pm,t,p,qv,cf,np.diff(z,axis=0),rho,*mass,*radii],surface,exner


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--levels',type=int,default=10)
    parser.add_argument('--columns',type=int,default=6);parser.add_argument('--out',type=Path)
    args=parser.parse_args();nz,nc=args.levels,args.columns
    source=kernels.module_source_int_defines('goddard',(('GDD_MAX_LEVELS',nz),))
    host=HostModule('goddard',source)
    xp=MetalArrayModule();report=[]
    def run(name,named,params,outputs,count):
        names=[p.name for p in host.kernels[name]]
        original={k:np.ascontiguousarray(v) for k,v in named.items()}
        arrays={k:xp.asarray(v) for k,v in original.items()}
        values={**original,**params}
        host.get_function(name)(((count+127)//128,),(128,),tuple(values[k] for k in names))
        if name=='goddard_bands_columns':
            words=workspace_words(nz);arrays['workspace']=xp.empty((words*nc,),np.uint64)
            params=dict(params,workspace_words=words)
        _submit(name,arrays,params,count)
        diffs={}
        for k in outputs:
            expected=original[k];got=arrays[k].to_numpy().reshape(expected.shape)
            different=int(np.count_nonzero(got.view(np.uint32)!=expected.view(np.uint32)))
            if different:
                where=np.flatnonzero(got.view(np.uint32).reshape(-1)!=expected.view(np.uint32).reshape(-1))
                diffs[k]={'words':different,'first':int(where[0]),'host':int(expected.view(np.uint32).reshape(-1)[where[0]]),'metal':int(got.view(np.uint32).reshape(-1)[where[0]])}
        report.append(dict(stage=name,diffs=diffs))
        return {k:original[k] for k in outputs}
    try:
        inputs,planes,surface,exner=fixture(nz,nc)
        tw=run('goddard_interface_temperature',dict(inputs,tw=np.empty((nz+1,nc),np.float32)),dict(nz=nz,columns=nc),('tw',),(nz+1)*nc)['tw']
        raw=np.concatenate([a.reshape(-1) for a in planes]+[surface[0],surface[1],tw[0],*surface[2:],exner.reshape(-1)])
        prep=run('goddard_prepare_columns',dict(raw=raw,band=np.empty((5*nz+15)*nc,np.float32),flags=np.empty(3*nc,np.int32),dz=np.empty(nz*nc,np.float32),q=np.empty(6*nz*nc,np.float32),re=np.empty(6*nz*nc,np.float32)),dict(np=nz,columns=nc,julday=172),('band','flags','dz','q','re'),nc)
        optics=run('goddard_cloud_columns',dict(dz=prep['dz'],q=prep['q'],re=prep['re'],lut=radiation_table_host_words('goddard-cloud'),output=np.empty(63*nz*nc,np.float32)),dict(np=nz,columns=nc),('output',),nz*nc)['output']
        flux=run('goddard_bands_columns',dict(input=prep['band'],flags=prep['flags'],optics=optics,output=np.empty((6*(nz+1)+4)*nc,np.float32)),dict(np=nz,columns=nc),('output',),nc)['output']
        result=run('goddard_publish_columns',dict(band=prep['band'],flux=flux,raw=raw,output=np.empty((2*nz+15)*nc,np.float32)),dict(np=nz,columns=nc,solcon=float(np.float32(1361.))),('output',),nc)['output']
        print(json.dumps(report,indent=2))
        if args.out:args.out.write_text(json.dumps(report,indent=2)+'\n')
        return int(any(row['diffs'] for row in report))
    finally:xp.context.close()

if __name__=='__main__':raise SystemExit(main())
