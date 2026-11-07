"""Compare the real packed Goddard chunk caller to selected host CUDA stages."""
from pathlib import Path
import argparse
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
import host_backend
from tools.goddard_metal.column_probe import fixture
from woof.core import kernels
from woof.core.radiation_tables import radiation_table_host_words
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.goddard_family import interface_temperature, solve_chunk, column_chunk
from woof.wrf_exact import ENABLED
assert ENABLED
parser=argparse.ArgumentParser();parser.add_argument('--direct',action='store_true')
parser.add_argument('--levels',type=int,default=10)
parser.add_argument('--columns',type=int,default=6)
parser.add_argument('--chunk',type=int,default=3)
args=parser.parse_args();direct=args.direct
nz,nc=args.levels,args.columns
inputs,planes,surface,exner=fixture(nz,nc)
source=kernels.module_source_int_defines('goddard',(('GDD_MAX_LEVELS',nz),))
host=host_backend.HostModule('strict_goddard_composed',source)
def invoke(name, values, count):
    kernel=host.get_function(name)
    order=[p.name for p in host.kernels[name]]
    kernel(((count+127)//128,),(128,),[values[n] for n in order])

tw=np.empty((nz+1,nc),np.float32)
invoke('goddard_interface_temperature',dict(inputs,tw=tw,nz=np.int32(nz),columns=np.int32(nc)),tw.size)
heat_lw=np.empty((nz,nc),np.float32);heat_sw=np.empty_like(heat_lw)
sfc=np.empty((15,nc),np.float32)
field_names=('pw','temperature','pressure','qv','cldfra','dz','rho','qi','qc','qr','qs','qg','qh','re_ice','re_cloud','re_rain','re_snow','re_graupel','re_hail')
source_fields=dict(zip(field_names,planes))
source_fields.update(latitude=surface[0],cosine=surface[1],tw=tw,
                     tsk=surface[2],emiss=surface[3],albedo=surface[4],exner=exner)
with MetalContext() as context:
    xp=MetalArrayModule(context)
    dev={name:xp.asarray(value) for name,value in source_fields.items()}
    temp=interface_temperature(dev['temperature'],xp.asarray(inputs['z']),
        xp.asarray(inputs['fnm']),xp.asarray(inputs['fnp']),levels=nz,columns=nc)
    dev['tw']=temp
    a=xp.empty(heat_lw.shape,np.float32);b=xp.empty(heat_sw.shape,np.float32)
    c=xp.empty(sfc.shape,np.float32); table=radiation_table_host_words('goddard-cloud'); dt=xp.asarray(table)
    chunk=column_chunk(context,nz,nc,args.chunk)
    for first in range(0,nc,chunk):
        cols=min(chunk,nc-first); end=first+cols
        def pack(x):return np.ascontiguousarray(x[...,first:end]).reshape(-1)
        radii=[x if direct else (x*np.float32(1.e-6))*np.float32(1.e6) for x in planes[13:19]]
        raw=np.concatenate([pack(x) for x in planes[:13]]+[pack(x) for x in radii]+
            [pack(surface[0]),pack(surface[1]),pack(tw[0]),*[pack(x) for x in surface[2:]],pack(exner)])
        band=np.empty((5*nz+15)*cols,np.float32);flags=np.empty(3*cols,np.int32)
        dz=np.empty(nz*cols,np.float32);q=np.empty(6*nz*cols,np.float32);re=np.empty_like(q)
        common={'np':np.int32(nz),'columns':np.int32(cols)}
        invoke('goddard_prepare_columns',dict(raw=raw,band=band,flags=flags,dz=dz,q=q,re=re,**common,julday=np.int32(172)),cols)
        optics=np.empty(63*nz*cols,np.float32)
        invoke('goddard_cloud_columns',dict(dz=dz,q=q,re=re,lut=table,output=optics,**common),nz*cols)
        flux=np.empty((6*(nz+1)+4)*cols,np.float32)
        invoke('goddard_bands_columns',dict(input=band,flags=flags,optics=optics,output=flux,**common),cols)
        result=np.empty((2*nz+15)*cols,np.float32)
        invoke('goddard_publish_columns',dict(band=band,flux=flux,raw=raw,output=result,**common,solcon=np.float32(1361.)),cols)
        result=result.reshape(2*nz+15,cols)
        heat_lw[:,first:end]=result[:nz];heat_sw[:,first:end]=result[nz:2*nz];sfc[:,first:end]=result[2*nz:]
        solve_chunk(dev,dt,levels=nz,full_columns=nc,first=first,columns=cols,
                    julday=172,solcon=1361.,direct_radii=direct,heat_lw=a,heat_sw=b,surface=c)
    total=0
    for name,expected,value in [('tw',tw,temp),('longwave heating',heat_lw,a),
                                ('shortwave heating',heat_sw,b),('surface flux',sfc,c)]:
        actual=value.to_numpy();x=expected.view(np.uint32);y=actual.view(np.uint32)
        bad=np.flatnonzero(x.ravel()!=y.ravel());total+=x.size
        print(name,f'{len(bad)}/{x.size} word differences, nonzero={np.count_nonzero(expected)}')
        if len(bad):
            i=bad[0];print('first',i,hex(int(x.ravel()[i])),hex(int(y.ravel()[i])));raise SystemExit(1)
print(f'0/{total} words differ; selected CUDA stages compiled on host, actual Metal packed/chunk caller')
