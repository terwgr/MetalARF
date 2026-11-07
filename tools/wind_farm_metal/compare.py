#!/usr/bin/env python3
"""Compare turbine Metal outputs with the CUDA source built without contraction."""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.metal.wind_farm import Kernel, MetalWindFarm, host, _Cost
from woof.core.wind_farm import WindFarm


def reference_source():
    source = (ROOT / 'woof/core/kernels/wind_farm.cu').read_text()
    start = source.index('extern "C" __global__ void wind_fitch_sparse')
    end = source.index('// Geometry rows:', start)
    # Sequential execution of the same 128-lane reduction graph: the host
    # launcher cannot emulate __syncthreads by calling threads in sequence.
    signature, body = source[start:end].split('{', 1)
    signature = signature.replace('wind_fitch_sparse', 'wind_fitch_serial')
    hub_start = body.index('   int z=0;')
    hub_end = body.index('\n  }', hub_start)
    hub = body[hub_start:hub_end]
    ec_start = body.index('  float ec=')
    ec_end = body.index('\n  if(k==0)', ec_start)
    ec = body[ec_start:ec_end]
    record_start = body.index('float* o=hub+10*a;')
    record_end = body.index('}', record_start)
    record = body[record_start:record_end]
    force_start = body.index('   float ct=hubnormalize')
    force_end = body.index('\n  }', force_start)
    force = body[force_start:force_end]
    for name in ('su', 'sv', 'st'):
        force = re.sub(r'\b'+name+r'\b',name+'[k]',force)
    serial = signature + '''{
 int c=cols[blockIdx.x];float sums[128],hs[7];
 float su[128]={},sv[128]={},st[128]={};
 for(int j=ptr[c];j<ptr[c+1];++j){
  int a=ids[j];float R=t[5*a+3],h=t[5*a+2]+zi[c];
''' + hub + '''
  for(int k=0;k<128;++k){int p=k*nc+c;float U=k<nz?hypotf(u[p],v[p]):0.f;
   float A=k<nz?diskprimitive(zi[p+nc]-h,R)-diskprimitive(zi[p]-h,R):0.f;
   sums[k]=A*U*U*U;}
  for(int stride=64;stride>0;stride>>=1)for(int k=0;k<stride;++k)sums[k]+=sums[k+stride];
''' + ec + '\n{' + record + '''}
  for(int k=0;k<nz;++k){int p=k*nc+c;float U=hypotf(u[p],v[p]);
   float A=diskprimitive(zi[p+nc]-h,R)-diskprimitive(zi[p]-h,R);
   if(A>0.f && t[5*a+4]!=0.f){
''' + force + '''}}
 }
 for(int k=0;k<nz;++k){int p=k*nc+c;
  du[p]+=su[k];dv[p]+=sv[k];qke[p]+=2.f*dt*st[k];if(adv!=qke)adv[p]+=2.f*dt*st[k];tke_rate[p]=st[k];}
}
'''
    return source[:start] + serial + source[end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=3)
    args = parser.parse_args()
    reference = HostModule('wind_farm', reference_source())
    rng = np.random.default_rng(42)
    compared = mismatch = 0
    def check(name, actual, expected):
        nonlocal compared, mismatch
        a,b = host(actual).view(np.uint32), np.asarray(expected).view(np.uint32)
        bad = a != b
        count = int(bad.sum());compared += a.size;mismatch += count
        if count:
            indices = np.flatnonzero(bad)[:6]
            print(name, 'mismatches', count, 'examples', [(int(i),hex(int(a.ravel()[i])),hex(int(b.ravel()[i]))) for i in indices])
    with MetalContext() as ctx, tempfile.TemporaryDirectory() as tmp:
        xp = MetalArrayModule(ctx)
        def launch(name, n, values, writes, block=128):
            expected = [v.copy() if isinstance(v,np.ndarray) else v for v in values]
            actual = [xp.asarray(v) if isinstance(v,np.ndarray) else v for v in values]
            if name == 'wind_fitch_sparse':
                grid=(n,);refname='wind_fitch_serial';refblock=(1,)
            else:
                grid=((n+127)//128,);refname=name;refblock=(128,)
            # Preserve aliased QKE carriers in both executions.
            for i in range(len(values)):
                for j in range(i):
                    if isinstance(values[i],np.ndarray) and values[i] is values[j]:
                        expected[i]=expected[j];actual[i]=actual[j]
            reference.get_function(refname)(grid,refblock,tuple(expected))
            Kernel(name,ctx)(grid,(block,),tuple(actual))
            if name == 'wind_wake':
                assert np.any(expected[-1] != values[2][:, 2, None]), 'wake fixture must shade an operating turbine'
            if name == 'wind_gad':
                assert np.any(expected[20] != 0), 'GAD fixture must exercise active rotor forces'
            for i in writes:
                check(name+':'+str(i),actual[i],expected[i])
            return [host(v) if isinstance(v,np.ndarray) else v for v in expected]
        for repeat in range(args.repeat):
            # Geographic setup, including dateline wrapping and spherical cells.
            ll=np.array([[0.2,179.9],[-.3,-179.8]],np.float32)
            lat=np.broadcast_to(np.linspace(-1,1,9,dtype=np.float32)[:,None],(9,9)).copy()
            lon=np.broadcast_to(np.linspace(179,181,9,dtype=np.float32)[None,:],(9,9)).copy()
            launch('wind_locate',2,[np.int32(2),np.int32(9),np.int32(9),ll,lat,lon,np.zeros((2,2),np.float32)],[6])
            launch('wind_locate_cells',2,[np.int32(2),np.int32(81),ll,lat.ravel()*np.float32(np.pi/180),lon.ravel()*np.float32(np.pi/180),np.zeros(2,np.int32)],[5])
            ll = rng.uniform([-70,-180],[70,180],(17,2)).astype(np.float32)
            lat = np.broadcast_to(np.linspace(-80,80,9,dtype=np.float32)[:,None],(9,9)).copy()
            lon = np.broadcast_to(np.linspace(-180,180,9,dtype=np.float32)[None,:],(9,9)).copy()
            launch('wind_locate',17,[np.int32(17),np.int32(9),np.int32(9),ll,lat,lon,np.zeros((17,2),np.float32)],[6])
            for nz in (40,130):
                nc,nt=9,3
                zi=np.broadcast_to(np.arange(nz+1,dtype=np.float32)[:,None]*np.float32(10),(nz+1,nc)).copy()
                u=rng.uniform(4,12,(nz,nc)).astype(np.float32);v=rng.uniform(-2,3,(nz,nc)).astype(np.float32)
                rho=rng.uniform(1,1.3,(nz,nc)).astype(np.float32)
                col=np.array([4,4,5],np.int32);ptr=np.array([0,0,0,0,0,2,3,3,3,3],np.int32);ids=np.arange(nt,dtype=np.int32)
                t=np.array([[100,100,100,60,1],[140,100,140,50,1],[200,100,100,60,0]],np.float32)
                meta=np.tile(np.array([0,3,0,2,3],np.int32),(nt,1))
                curves=np.array([[0,.8,.4,12,0],[15,.7,.3,15,2],[30,.2,.0,10,5]],np.float32)
                limits=np.tile(np.array([-1,-1,0,0,0],np.float32),(nt,1))
                limits[0] = [5,10,.05,.1,1.23]
                hub=np.zeros((nt,10),np.float32)
                sample=[np.int32(nt),np.int32(nc),np.int32(nz),col,t,meta,curves,limits,zi,u,v,rho,np.full_like(u,4),hub]
                h=launch('wind_sample',nt,sample,[13])[-1]
                wake=np.zeros((nt,7),np.float32)
                for superposition in (1,2,4):
                    w=launch('wind_wake',nt*7,[np.int32(nt),t,h,np.float32(.05),np.float32(20),np.float32(2),np.int32(superposition),wake],[7])[-1]
                    launch('wind_power',nt,[np.int32(nt),t,meta,curves,w,np.float32(2),h],[6])
                for scheme in (1,2,3):
                    for normalized in (0,1):
                        qke=np.ones_like(u);adv=qke if normalized else np.zeros_like(u)
                        src=[np.int32(scheme),np.int32(nc),np.int32(nz),np.int32(3),np.int32(3),ptr,ids,t,meta,curves,limits,h,w,zi,u,v,np.ones(nc,np.float32),np.float32(10000),np.float32(100),np.float32(.1),np.float32(.25),np.float32(1.7),np.float32(2),np.int32(normalized),np.zeros_like(u),np.zeros_like(u),qke,adv,np.zeros_like(u)]
                        launch('wind_source',nc*nz,src,[24,25,26,27,28])
                        if nz<=128:
                            sparse=[np.int32(nc),np.int32(nz),np.array([4,5],np.int32),ptr,ids,t,meta,curves,limits,zi,u,v,rho,np.ones(nc,np.float32),np.float32(10000),np.float32(.1),np.float32(.25),np.int32(normalized),np.zeros_like(u),np.zeros_like(u),qke,adv,hub,np.zeros_like(u)]
                            launch('wind_fitch_sparse',2,sparse,[18,19,20,21,22,23])
                blades=np.array([[.1,4,5,0],[1,2,0,0]],np.float32)
                polars=np.array([[-180,0,.1],[0,1,.01],[180,0,.1]],np.float32)
                gad=[np.int32(nc),np.int32(nz),np.int32(3),np.float32(100),np.float32(100),np.float32(100),ptr,ids,t,meta,curves,blades,polars,np.array([[0,3]],np.int32),h,zi,u,v,np.float32(1e-6),np.int32(100),np.zeros_like(u),np.zeros_like(u),np.zeros_like(u),np.zeros_like(u,dtype=np.int32)]
                launch('wind_gad',nc*nz,gad,[20,21,22,23])
        # Exercise the shared Python launch order and Metal setup, not only
        # individual kernels. Both QKE carriers advance across changing dt.
        table = Path(tmp) / 'farm.json'
        table.write_text(json.dumps(dict(schema='gpuwm-wind-farm.v1',
            types={'demo':dict(curve=[[0,.8,.4,12,0],[30,.7,.3,15,2]], blades=3,
                blade=[[.1,4,5,0],[1,2,0,0]], polars=[[[-180,0,.1],[0,1,.01],[180,0,.1]]])},
            turbines=[dict(id='a',farm_id='farm',type='demo',grid_x=4,grid_y=4,hub_height_m=100,rotor_diameter_m=120),
                      dict(id='b',farm_id='farm',type='demo',grid_x=5,grid_y=4,hub_height_m=120,rotor_diameter_m=100)])))
        def loader(module, name):
            if name == 'wind_fitch_sparse':
                return lambda grid,block,values: reference.get_function('wind_fitch_serial')(grid,(1,),values)
            return reference.get_function(name)
        shape=(40,9,9)
        atmosphere=dict(z_interface=np.broadcast_to(np.arange(41,dtype=np.float32)[:,None,None]*np.float32(10),(41,9,9)).copy(),
            u=rng.uniform(4,12,shape).astype(np.float32),v=rng.uniform(-2,3,shape).astype(np.float32),rho=np.full(shape,1.2,np.float32))
        for scheme in ('fitch','auto','ewp','jensen','gad','hex'):
            domain = atmosphere
            setup = {}
            shape = (40,9,9)
            if scheme == 'hex':
                from woof.wind_table import read_table
                data, digest = read_table(table)
                data = dict(data, turbines=[dict(t, grid_y=0) for t in data['turbines']])
                shape = (40,1,9)
                domain = {k: np.ascontiguousarray(v[:, :1]) for k,v in atmosphere.items()}
                setup = dict(cell_layout=True, prepared=(data,digest),
                             msf=np.linspace(.0001,.0002,9,dtype=np.float32).reshape(1,9))
            cfg=SimpleNamespace(wind_farm_table=table,wind_farm_scheme='fitch' if scheme=='hex' else scheme,dx=100.,dy=100.,dt=.1)
            h=WindFarm(cfg,shape,_array_module=np,_kernel_loader=loader,_cost=_Cost(),z_interface=domain['z_interface'], **setup)
            h.replay={}
            m=MetalWindFarm(cfg,shape,context=ctx,z_interface=domain['z_interface'], **setup)
            a={k:xp.asarray(v) for k,v in domain.items()}
            hf={'qke':np.ones(shape,np.float32),'qke_adv':np.full(shape,2,np.float32),'exch_m':np.full(shape,4,np.float32)}
            mf={k:xp.asarray(v) for k,v in hf.items()}
            hr={k:np.zeros(shape,np.float32) for k in ('du','dv')};mr={k:xp.asarray(v) for k,v in hr.items()}
            for dt in (.1,.2):
                if scheme == 'gad':
                    h.sample(domain,hf);m.sample(a,mf)
                else:
                    h.apply_mynn(domain,hf,hr,step_seconds=dt);m.apply_mynn(a,mf,mr,step_seconds=dt)
                for name in hr:check('driver:'+scheme+':'+name,mr[name],hr[name])
                for name in ('qke','qke_adv'):check('driver:'+scheme+':'+name,mf[name],hf[name])
                check('driver:'+scheme+':hub',m.hub,h.hub)
                fields=m.history_fields()
                check('history:'+scheme+':TURBINE_POWER',fields['TURBINE_POWER'],h.hub[:,7])
                check('history:'+scheme+':FARM_POWER',fields['FARM_POWER'],np.array([h.hub[:,7].astype(np.float64).sum()],np.float32))
                check('history:'+scheme+':POWER',fields['POWER'],np.bincount(h.col,weights=h.hub[:,7],minlength=h.nc).astype(np.float32).reshape(shape[1:]))
        # The complete-globe MYNN seam also needs CUDA's cold sea-ice optics.
        from datetime import datetime
        from woof.core.mpas_column_suite import _seed_ruc_fractional_seaice_optics
        from woof.core.mpas_column_suite_metal import _seed_ruc_fractional_seaice_optics_metal
        previous_cupy = sys.modules.get('cupy')
        sys.modules['cupy'] = np
        params = SimpleNamespace(dataset_identifier='MODIFIED_IGBP_MODIS_NOAH',
            vegetation=SimpleNamespace(name='MODI-RUC'), xice_threshold=.02)
        cfg = SimpleNamespace(sf_surface_physics=3, fractional_seaice=1)
        base = {name:rng.uniform(.1,.8,(1,1024)).astype(np.float32)
                for name in ('albbck','embck','albedo','emiss')}
        base['xice'] = np.linspace(0,1,1024,dtype=np.float32).reshape(1,-1)
        for month, missing in ((1,False),(7,False),(1,True),(7,True)):
            fields = {k:v.copy() for k,v in base.items()}
            if missing:
                fields['embck'].fill(.95)
            h = SimpleNamespace(fields=fields, ruc_params=params)
            m = SimpleNamespace(fields={k:xp.asarray(v) for k,v in fields.items()
                                       if not (missing and k=='embck')},
                ruc_params=params, state=SimpleNamespace(_array_module=xp))
            lat = rng.uniform(-90,90,1024).astype(np.float32)
            stamp = datetime(2026,month,1)
            _seed_ruc_fractional_seaice_optics(h,cfg,lat,stamp)
            _seed_ruc_fractional_seaice_optics_metal(m,cfg,lat,stamp)
            for name in base:
                check('cold_ice:'+name,m.fields[name],h.fields[name])
        if previous_cupy is None:
            del sys.modules['cupy']
        else:
            sys.modules['cupy'] = previous_cupy
        print(json.dumps(dict(compared_words=compared,mismatched_words=mismatch,repeats=args.repeat)))
        if mismatch:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
