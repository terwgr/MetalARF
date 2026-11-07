"""Word comparisons against host-built CUDA FDDA kernels; run on Metal."""
from pathlib import Path
import json
import numpy as np
from tools.thompson_real_column_parity import host_backend as host
from woof.metal import MetalArrayModule
from woof.metal.nudging import submit


def main():
    host._POINTER_DTYPES['cuFloatComplex']=np.complex64
    source=Path('woof/spectral_nudging/kernels.cu').read_text()
    source=source[:source.index('// Device-side timing')]
    source=source.replace('#include <cuComplex.h>', '''
struct cuFloatComplex { float x,y; };
inline cuFloatComplex make_cuFloatComplex(float x,float y) { return {x,y}; }
inline float cuCrealf(cuFloatComplex a) { return a.x; }
''')
    source='inline unsigned int atomicOr(unsigned int *p,unsigned int v) { unsigned int old=*p; *p|=v; return old; }\n'+source
    modules=[host.HostModule('nudging',source),host.HostModule('nudging_forecast',Path('woof/spectral_nudging/forecast.cu').read_text())]
    xp=MetalArrayModule()
    rng=np.random.default_rng(52)
    counts={}
    def check(name,arrays,params,size):
        module=modules[name in ('pack_fields','unpack_fields','wrf_weights')]
        kernel=module.get_function(name)
        ref={k:v.copy() for k,v in arrays.items()}
        values={k:xp.asarray(v) for k,v in arrays.items()}
        args=tuple(ref[p.name] if p.pointer else params[p.name] for p in kernel.params)
        kernel((1,),(1,),args) if name=="tick" else kernel(((size+127)//128,),(128,),args)
        mparams={k:v for k,v in params.items() if k not in ('end','ramp')}
        if name=='tick':mparams['n']=1
        if name in ('difference','relax'):
            values['timing']=xp.asarray(np.array([params['end'],params['ramp']],dtype=np.float64))
        submit(xp,name,values,mparams,size)
        for p in kernel.params:
            if not p.pointer or p.text.startswith('const '):continue
            actual=values[p.name].to_numpy()
            a=actual.view(np.uint32);b=ref[p.name].view(np.uint32)
            bad=np.count_nonzero(a!=b)
            if bad:
                indices=np.flatnonzero((a!=b).ravel())[:6]
                raise AssertionError(f'{name}.{p.name}: {bad}/{a.size} words differ: '+str([(int(i),hex(int(a.ravel()[i])),hex(int(b.ravel()[i]))) for i in indices]))
            counts[name]=counts.get(name,0)+a.size
        return ref
    nz,ny,nx=4,6,7;n=nz*ny*nx
    f=lambda shape: rng.uniform(-3.,8.,shape).astype('f')
    for repeat in range(3):
        for wrf in (0,1):
            for cadence in (1,3):
                for zero in (0,1):
                    for ramp in (-20.,0.,20.):
                        value=f((nz,ny,nx)); held=np.zeros_like(value)
                        for elapsed in (0.,10.,20.,30.,40.,60.,80.):
                            clock=np.array([elapsed,10.,0.,60.],dtype='d')
                            params=dict(n=n,wrf=wrf,cadence=cadence,end=60.,ramp=ramp,grid_nudging=wrf)
                            r=check('difference',dict(value=value,a=f(value.shape),b=f(value.shape),out=np.zeros(value.shape,dtype='F'),clock=clock,error=np.zeros(1,dtype='I')),params,n)
                            work=r['out']
                            check('truncate_modes',dict(spectrum=work.copy()),dict(n=n,nx=nx,ny=ny,cutoff_x=1,cutoff_y=2),n)
                            rp=dict(n=n,nx=nx,ny=ny,boundary=1,floor_m=0.,ramp_m=3000.,tau=5000.,wrf=wrf,cadence=cadence,fgdtzero=zero,end=60.,ramp=ramp)
                            r=check('relax',dict(value=value,filtered=work,height_agl=rng.uniform(0,5000,value.shape).astype('f'),pblh=np.full((ny,nx),1500.,dtype='f'),clock=clock,vertical_weight=rng.random(value.shape,dtype=np.float32),held=held),rp,n)
                            value,held=r['value'],r['held']
        # Invalid interval records an error and prevents tick and relaxation.
        check('difference',dict(value=f((n,)),a=f((n,)),b=f((n,)),out=np.zeros(n,dtype='F'),clock=np.array([70.,10.,0.,60.]),error=np.zeros(1,dtype='I')),dict(n=n,wrf=0,cadence=1,end=-1.,ramp=0.,grid_nudging=0),n)
        for error in (0,1):
            check('tick',dict(clock=np.array([.1,.2,0.,1.]),error=np.array([error],dtype='I'),count=np.array([17],dtype='Q')),{},1)
        for terrain in (0,1):
            for potential in (0,1):
                shape=(nz,ny,nx);face=(nz+1,ny,nx)
                phb=np.broadcast_to(np.arange(nz+1,dtype='f')[:,None,None]*9810.,face).copy() if terrain else np.arange(nz+1,dtype='f')*9810.
                thb=np.full(shape if terrain else (nz,),300.,dtype='f')
                arrays=dict(thp=f(shape),thb=thb,p=rng.uniform(20000,100000,shape).astype('f'),php=f(face),phb=phb,ht=f((ny,nx)),pbl=np.full((ny,nx),1500.,dtype='f'))
                for k,s in dict(temp=shape,oldt=shape,phi=face,oldphi=face,zu=(nz,ny,nx+1),zv=(nz,ny+1,nx),zt=shape,zf=face,bu=(ny,nx+1),bv=(ny+1,nx),bt=(ny,nx),bf=(ny,nx)).items():arrays[k]=np.zeros(s,dtype='f')
                params=dict(nz=nz,ny=ny,nx=nx,terrain=terrain,potential=potential)
                size=max(a.size for a in arrays.values())
                r=check('pack_fields',arrays,params,size)
                check('unpack_fields',dict(thp=arrays['thp'],php=arrays['php'],p=arrays['p'],temp=r['temp']+np.float32(.013),oldt=r['oldt'],phi=r['phi']+np.float32(.17),oldphi=r['oldphi']),dict(nz=nz,ny=ny,nx=nx,potential=potential),(nz+1)*ny*nx)
                for field in range(5):
                    for grid in (0,1):
                        for nopbl,zfac in ((0,0),(0,1),(1,0),(1,1)):
                            sx,sy=nx+(field==0),ny+(field==1)
                            a={k:arrays[k] for k in ('php','phb','ht','pbl')}
                            a['weight']=np.zeros((nz+(field==3),sy,sx),dtype='f')
                            check('wrf_weights',a,dict(nz=nz,ny=ny,nx=nx,terrain=terrain,field=field,grid_nudging=grid,no_pbl=nopbl,zfac=zfac,kz=2,dk=2),sx*sy)
    print(json.dumps(dict(differing_words=0,compared_words=counts,total=sum(counts.values()),repeats=3)))


if __name__=='__main__':main()
