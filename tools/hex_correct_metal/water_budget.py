"""Exercise upstream's WSM6 budget columns on ordinary Metal in both modes.

Run from the repository root. This discriminates the conservative surface
return while checking that atmospheric and radius words do not change.
It is a budget comparison, not an independent CUDA-device parity proof.
"""
import ast
from pathlib import Path
import numpy as np
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.core.wsm6 import launch_wsm6
# Reuse upstream's discriminating budget columns, without invoking its test runner.
src=ast.parse(Path('tests/test_correct_mode_water_budget.py').read_text())
node=next(n for n in src.body if isinstance(n,ast.FunctionDef) and n.name=='_wsm6_columns')
namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'upstream-column-fixture','exec'),namespace)
with MetalContext(batch_kernel_calls=False) as context:
 submit=context._submit_kernel_batch
 def synchronous(*a,**kw):
  result=submit(*a,**kw);result.wait();return result
 context._submit_kernel_batch=synchronous;xp=MetalArrayModule(context)
 outcomes=[]
 for corrected in [False,True]:
  d=namespace['_wsm6_columns'](xp,ncol=16)
  def water():
   rho,dz=d['rho'].to_numpy().astype(np.float64),d['dz'].to_numpy().astype(np.float64)
   return sum((rho*dz*d[n].to_numpy().astype(np.float64)).sum(axis=0) for n in ('qv','qc','qr','qi','qs','qg'))
  before=water()
  launch_wsm6(d['theta'],d['qv'],d['qc'],d['qr'],d['qi'],d['qs'],d['qg'],d['rho'],d['pii'],d['pressure'],d['dz'],d['rainnc'],d['rainncv'],d['snownc'],d['snowncv'],d['graupelnc'],d['graupelncv'],d['sr'],60.,effc=d['effc'],effi=d['effi'],effs=d['effs'],hail_opt=0,conservative_surface=corrected)
  residual=(water()+d['rainncv'].to_numpy().astype(np.float64)-before).ravel()
  values={k:d[k].to_numpy() for k in d}
  outcomes.append((values,residual))
  print('conservative_surface',corrected,'maximum budget residual',np.abs(residual).max(),flush=True)
 assert np.abs(outcomes[0][1]).max()>1e-4
 assert np.abs(outcomes[1][1]).max()<5e-6
 names=('theta','qv','qc','qr','qi','qs','qg','effc','effi','effs')
 bad=sum(np.count_nonzero(outcomes[0][0][n].view(np.uint32)!=outcomes[1][0][n].view(np.uint32)) for n in names)
 assert bad==0
 print('Atmosphere and radii mode differences',bad,'/',sum(outcomes[0][0][n].size for n in names),'words')
