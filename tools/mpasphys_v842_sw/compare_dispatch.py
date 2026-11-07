"""Exercise actual resident SW composition across mixed day/night columns."""
import numpy as np
import sys
from woof.hex.mpasphys.metal_rrtmg_sw_v842 import initialize_tables, run_shortwave
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    nc,nz,nozn=5,6,2;rng=np.random.default_rng(842)
    fields=np.empty((12,nc,nz),np.float32)
    fields[0]=np.linspace(95000,20000,nz);fields[1]=np.linspace(280,230,nz);fields[2]=150
    fields[3]=np.linspace(1,.7,nz);fields[4]=.003;fields[5]=.0001;fields[6]=.00003;fields[7]=.00002
    fields[8]=rng.uniform(.1,.7,(nc,nz));fields[9]=.000012;fields[10]=.00003;fields[11]=.00005
    surf=np.zeros((7,nc),np.float32);surf[1]=[-180,-60,0,60,180];surf[2]=.2;surf[3]=280;surf[4]=[1,2,1,2,1];surf[6]=1
    pw=np.broadcast_to(np.linspace(100000,15000,nz+1,dtype=np.float32),(nc,nz+1)).copy()
    tw=np.broadcast_to(np.linspace(282,220,nz+1,dtype=np.float32),(nc,nz+1)).copy()
    pin=np.array([1,1100],np.float32);oz=np.full((nc,nozn),6e-6,np.float32);controls=np.array([0,15,0,1368],np.float32)
    clear='--clear' in sys.argv
    if clear:fields[8]=0
    kwargs=dict(gmt=12,degrad=np.float32(np.pi/180),icloud=1,o3input=2,has_reqc=1,has_reqi=1,has_reqs=1,overlap=0 if clear else 2,decorrelation=0,julday=180)
    with MetalContext(batch_kernel_calls=False) as ctx:
        original=ctx._submit_kernel_batch
        def sync(*a,**kw):
            x=original(*a,**kw);x.wait();return x
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);tables=initialize_tables(ctx)
        def run(indices,with_aerosol):
            ix=np.asarray(indices)
            aerosol=None
            if with_aerosol:
                aerosol={name:xp.asarray(value) for name,value in dict(pressure=fields[0,ix,:].copy(),temperature=fields[1,ix,:].copy(),qv=fields[4,ix,:].copy(),dz=fields[2,ix,:].copy(),nwfa=np.full((len(ix),nz),1e8,np.float32),nifa=np.full((len(ix),nz),1e5,np.float32),xland=surf[4,ix].copy()).items()}
            return run_shortwave(tables=tables,fields=xp.asarray(fields[:,ix,:].copy()),surface=xp.asarray(surf[:,ix].copy()),pressure_interfaces=xp.asarray(pw[ix].copy()),temperature_interfaces=xp.asarray(tw[ix].copy()),pin=xp.asarray(pin),ozone=xp.asarray(oz[ix].copy()),controls=xp.asarray(controls),aerosol_inputs=aerosol,**kwargs)
        for with_aerosol in (False,True):
            mixed={n:x.to_numpy() for n,x in run(range(nc),with_aerosol).items()}
            if not np.any(mixed['scalars'][0]>0) or not np.any(mixed['scalars'][0]<=0):raise AssertionError('fixture must contain day and night')
            if not np.any(mixed['tendency']!=0):raise AssertionError('fixture has no active radiation')
            if clear and not np.array_equal(mixed['profiles'][0].view(np.uint32),mixed['profiles'][1].view(np.uint32)):raise AssertionError('clear-sky up flux differs from clear-sky diagnostic')
            if clear and not np.array_equal(mixed['profiles'][2].view(np.uint32),mixed['profiles'][3].view(np.uint32)):raise AssertionError('clear-sky down flux differs from clear-sky diagnostic')
            for c in range(nc):
                single={n:x.to_numpy() for n,x in run([c],with_aerosol).items()}
                rows=[('scalars',mixed['scalars'][:,c],single['scalars'][:,0]),('tendency',mixed['tendency'][c],single['tendency'][0]),('profiles',mixed['profiles'][:,c],single['profiles'][:,0])]
                if with_aerosol:rows.extend([('taod5503d',mixed['taod5503d'][c],single['taod5503d'][0]),('taod5502d',mixed['taod5502d'][c:c+1],single['taod5502d'])])
                for name,got,want in rows:
                    if not np.array_equal(got.view(np.uint32),want.view(np.uint32)):raise AssertionError(f'{name} batch differs at column {c}')
    print('native SW mixed-day batch matches isolated column runs, with active radiation',flush=True)

if __name__=='__main__':main()
