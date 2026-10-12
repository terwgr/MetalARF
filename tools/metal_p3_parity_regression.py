import pathlib,sys,argparse,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
parser=argparse.ArgumentParser();parser.add_argument('--output',type=pathlib.Path,required=True)
OUTPUT=parser.parse_args().output.resolve();OUTPUT.mkdir(parents=True,exist_ok=True)

import ast,io,subprocess,types,numpy as np
from woof.core import p3 as C
from woof.core.p3 import p3_init,mp_p3_wrapper_wrf
from woof.core.p3_device import SCRATCH_SLOTS,SEDW_SLOTS
from woof.metal.array import MetalArrayModule
from woof.metal.p3_constants import p3_constant_values
import woof.metal.p3_runtime as M
src=subprocess.check_output(['git','show','HEAD:tests/test_p3_cuda_gpu.py'],text=True); h=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='_tiny_column_case'); env={'np':np}; exec(compile(ast.Module(body=[h],type_ignores=[]),'<tracked-helper>','exec'),env)
def cpu(host,its,dt):
 names=('th','qv','qc','qr','nr','qi','qir','ni','qib','th_old','qv_old','pres','dz'); f={n:np.ascontiguousarray(host[n].T.copy(),np.float32) for n in names}; nc,nz=f['th'].shape
 s={n:np.zeros(nc,np.float32) for n in ('rainnc','rainncv','sr','snownc','snowncv')}; d={n:np.zeros((nc,nz),np.float32) for n in ('refl','re_cloud','re_ice','vmi','di','rhopo')}; aft={}; orig=C.p3_main
 def cap(*a,**kw):
  r=orig(*a,**kw); aft.update({n:np.array(a[i],copy=True) for n,i in (('qc',0),('nc',1),('qr',2),('nr',3),('th_old',4),('th',5),('qv_old',6),('qv',7),('qi',9),('qir',10),('ni',11),('qib',12),('nc',1),('ssat',13),('prt_liq',17),('prt_sol',18))}); return r
 C.p3_main=cap
 try:
  for it in its: mp_p3_wrapper_wrf(f['th'],f['qv'],f['qc'],f['qr'],f['nr'],f['qi'],f['qir'],f['ni'],f['qib'],f['th_old'],f['qv_old'],f['pres'],f['dz'],dt,it,s['rainnc'],s['rainncv'],s['sr'],s['snownc'],s['snowncv'],d['refl'],d['re_cloud'],d['re_ice'],d['vmi'],d['di'],d['rhopo'],runtime=p3_init())
 finally:C.p3_main=orig
 f.update(s); f.update(d); f.update({n:aft[n] for n in ('nc','ssat','prt_liq','prt_sol')}); f['re_cloud'] *= np.float32(1e6); f['re_ice'] *= np.float32(1e6); return f,aft
from woof.metal.runtime import MetalContext
_async_submit=MetalContext._submit_kernel_batch
def _synchronous_submit(self,*args,**kw):
 result=_async_submit(self,*args,**kw)
 if result is not None: result.wait()
 return result
MetalContext._submit_kernel_batch=_synchronous_submit
A_SHARED=MetalArrayModule()
def metal(host,its,dt):
 A=A_SHARED; up=A.asarray; nz,nc=host['th'].shape; sh=(nz,1,nc)
 def v(n):return up(np.ascontiguousarray(host[n].reshape(sh),np.float32))
 f={n:v(n) for n in ('qc','qr','nr','qi','qir','ni','qib','th_old','qv_old','qv','dz')}; f.update(theta=v('th'),pressure=v('pres'),nc=up(np.zeros(sh,np.float32)),ssat=up(np.zeros(sh,np.float32)))
 s={n:up(np.zeros(sh,np.float32)) for n in (*SCRATCH_SLOTS,*SEDW_SLOTS)}; s.update(vq=s['v_q'],vn=s['v_n'],fq=s['flux_q'],fn=s['flux_n'],fqir=s['flux_qir'],fbir=s['flux_bir'],temperature=up(np.zeros(sh,np.float32)),h_diabatic=up(np.zeros(sh,np.float32)),flags=up(np.zeros((2,1,nc),np.float32)))
 d={n:up(np.zeros(sh,np.float32)) for n in ('zdbz','effc','effi','vmi','di','rhopo')}; surf={n:up(np.zeros((1,nc),np.float32)) for n in ('prt_liq','prt_sol','rainnc','rainncv','sr','snownc','snowncv')}; ht=p3_init(); t=types.SimpleNamespace(**{n:up(np.ascontiguousarray(getattr(ht,n)).reshape(-1)) for n in ('itab','itabcoll','vn_table','vm_table','revap_table')}); t.metal_constants=up(p3_constant_values())
 aft={}; orig=M.submit_p3_kloopmain
 def cap(*a,**kw):
  r=orig(*a,**kw); ff=kw['fields']; ss=kw['scratch']; aft.update({n:np.array(ff[n].to_numpy(),copy=True) for n in ('qc','nc','qr','nr','th_old','theta','qv_old','qv','qi','qir','ni','qib')}); aft['temperature']=np.array(ss['temperature'].to_numpy(),copy=True); return r
 M.submit_p3_kloopmain=cap
 try:
  for it in its:M.run_p3_metal(f,s,d,surf,t,dt=dt,it=it)
 finally:M.submit_p3_kloopmain=orig
 out={n:np.array(f[k].to_numpy(),copy=True).reshape(nz,nc) for n,k in (('qc','qc'),('qr','qr'),('nr','nr'),('qi','qi'),('qir','qir'),('ni','ni'),('qib','qib'),('nc','nc'),('ssat','ssat'),('th','theta'),('qv','qv'),('th_old','th_old'),('qv_old','qv_old'))}; out.update({n:np.array(surf[n].to_numpy(),copy=True).reshape(nc) for n in ('rainnc','rainncv','sr','snownc','snowncv','prt_liq','prt_sol')})
 for n,k in (('refl','zdbz'),('re_cloud','effc'),('re_ice','effi'),('vmi','vmi'),('di','di'),('rhopo','rhopo')):
  x=np.array(d[k].to_numpy(),copy=True).reshape(nz,nc); out[n]=x
 return out,{n:np.asarray(x).reshape(nz,nc) for n,x in aft.items()}
def report(label,host,its,dt,focus):
 ref,ca=cpu(host,its,dt); got,ma=metal(host,its,dt); np.savez(OUTPUT/('p3-'+label+'-results.npz'),**{'cpu_'+n: (np.asarray(v).T if np.asarray(v).ndim==2 else v) for n,v in ref.items()},**{'metal_'+n:v for n,v in got.items()}); print(f'{label} profile shape={host["th"].shape} dt={dt} steps={its}')
 for n in ('th','qv','qc','qr','nr','qi','qir','ni','qib','nc','ssat','prt_liq','prt_sol','th_old','qv_old','rainnc','rainncv','sr','snownc','snowncv','refl','re_cloud','re_ice','vmi','di','rhopo'):
  a=np.asarray(ref[n],np.float64); b=np.asarray(got[n],np.float64)
  if a.ndim==2 and a.shape==b.T.shape:a=a.T
  assert np.isfinite(a).all() and np.isfinite(b).all(),n
  np.testing.assert_array_equal(np.asarray(b,np.float32).view(np.uint32),np.asarray(a,np.float32).view(np.uint32),err_msg=label+':'+n)
  delta=np.abs(b-a); ix=np.unravel_index(int(np.argmax(delta)),delta.shape); rel=delta/np.maximum(np.abs(a),1e-30)
  print(f' {n} abs={delta[ix]:.8g} rel={rel.max():.8g} ref={a[ix]:.8g} metal={b[ix]:.8g} idx={ix} magnitudes={np.max(np.abs(a)):.8g}/{np.max(np.abs(b)):.8g}')
 if label=='tiny':print(' effc k18 col2 micron CPU/Metal',float(ref['re_cloud'][2,18]),float(got['re_cloud'][18,2]))
host,_,_=env['_tiny_column_case'](24,8); report('tiny',host,[1,2,3],20.,[(18,2)])
raw=subprocess.check_output(['git','show','HEAD:tests/data/p3_cold_cloud_overflow_column.npz'])
with np.load(io.BytesIO(raw)) as z:cold={n:np.asarray(z[n],np.float32)[:,None] for n in ('th','qv','qc','qr','nr','qi','qir','ni','qib','th_old','qv_old','pres','dz')}
report('cold',cold,[284],60.,[(39,0),(47,0)])

A_SHARED.context.close()
