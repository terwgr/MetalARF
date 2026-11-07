"""Literal-word check of production Metal diagnostic callers against retained CUDA source.

The CUDA source runs on the host; this does not establish NVIDIA-device parity.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace as State
import numpy as np
os.environ['GPUWM_WRF_EXACT'] = '1'
os.environ['WOOF_WRF_EXACT_DIAGNOSTICS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent / 'thompson_real_column_parity'))
import host_backend
from woof.core.kernels import module_source
source = '#define GPUWM_WRF_EXACT 1\n#define GPUWM_WRF_EXACT_D_DIAGNOSTICS 1\n' + module_source('diagnostics')
host_backend.CXX_FLAGS += ('-w',)
ref = host_backend.HostModule('strict_diagnostics', source)
from woof.core.diagnostics import update_diagnostics
from woof.metal.array import MetalArrayModule
xp = MetalArrayModule()
rng = np.random.default_rng(1352)
cases = words = subnormal = carried_differences = 0
for base3d in (False, True):
 for moist in (False, True):
  for hypso in (1, 2):
   for window in (None, (1, 1, 2, 3)):
    for carried in (False, True):
     nz, ny, nx = 5, 4, 7
     shape = (nz, ny, nx); plane = (ny, nx)
     profile = np.arange(nz, dtype=np.float32)
     base = (nz, ny, nx) if base3d else (nz,)
     bface = (nz + 1, ny, nx) if base3d else (nz + 1,)
     def broadcast(v, target):
      return np.broadcast_to(v.reshape((-1, 1, 1)) if len(target) == 3 else v, target).copy()
     h = dict(thp=rng.uniform(-3, 3, shape).astype(np.float32), php=rng.uniform(-.02, .02, (nz+1, ny, nx)).astype(np.float32),
              mup=rng.uniform(-50, 50, plane).astype(np.float32), mub=np.full(plane, 90000, np.float32),
              thb=np.full(base, 300, np.float32), phb=broadcast(np.arange(nz+1, dtype=np.float32)*12000,bface),
              alb=np.full(base, .8, np.float32), pb=broadcast(90000-profile*11000,base), dphbr=np.full(base, 42,np.float32),
              rdnw=np.full(nz, -nz,np.float32), c1h=np.ones(nz,np.float32), c2h=np.zeros(nz,np.float32),
              c3f=np.linspace(1,0,nz+1,dtype=np.float32), c4f=np.zeros(nz+1,np.float32),
              c3h=np.linspace(.9,.1,nz,dtype=np.float32), c4h=np.zeros(nz,np.float32),
              dc3f=np.full(nz,.2,np.float32),dc4f=np.zeros(nz,np.float32),
              qv=rng.uniform(.001,.02,shape).astype(np.float32))
     for n in ('thp','php','mup','qv'):
      h[n].flat[:4] = np.array([0, 0x80000000, 1, 0x80000001],np.uint32).view(np.float32)
     sums = np.float32(h['mub']+h['mup']); h['muts']=np.nextafter(sums,np.float32(np.inf))
     carried_differences += int(np.count_nonzero(h['muts'].view(np.uint32)!=sums.view(np.uint32)))
     initial = np.resize(np.array([0x3f123456,0x80000000,1,0x80000001],np.uint32).view(np.float32),shape)
     for n in ('p','al','alt','p_perturbation'):h[n]=initial.copy()
     args = {**h, 'p_top':np.float32(10000), 'hypso':np.int32(hypso),'moist':np.int32(moist),'base3d':np.int32(base3d),
             'nz':np.int32(nz),'ny':np.int32(ny),'nx':np.int32(nx)}
     j0,i0,nyw,nxw = (0,0,ny,nx) if window is None else window
     args.update(j0=np.int32(j0),i0=np.int32(i0),nyw=np.int32(nyw),nxw=np.int32(nxw))
     if not moist:args['qv']=h['thp']
     kernel=ref.get_function('calc_p_alpha_carried_muts' if carried else 'calc_p_alpha')
     s=State(_metal_setup_state=True,_array_module=xp,p_top=10000.)
     for n,a in h.items():setattr(s, {'mub':'mub2d','dphbr':'dphb_resid'}.get(n,n),xp.asarray(a.copy()))
     if not moist:s.qv=None
     if window is None:s.p_perturbation=None
     kernel(((nyw*nxw+127)//128,), (128,), tuple(args[p.name] for p in kernel.params))
     update_diagnostics(s,hypso,window=window,muts=s.muts if carried else None)
     for n in ('p','al','alt','p_perturbation'):
      got=getattr(s,n).to_numpy(); want=h[n]
      np.testing.assert_array_equal(got.view(np.uint32),want.view(np.uint32),err_msg=str((n,base3d,moist,hypso,window,carried)))
      bits=want.view(np.uint32);words+=bits.size;subnormal+=int(np.count_nonzero(((bits&0x7f800000)==0)&((bits&0x007fffff)!=0)))
     cases+=1
print(dict(cases=cases,words=words,subnormal_words=subnormal,carried_mass_differing_columns=carried_differences,mismatched_words=0))
xp.context.close()
