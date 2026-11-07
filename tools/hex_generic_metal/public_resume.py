"""Run midpoint/continuation for the fixed public generic-profile fixture."""
from pathlib import Path
import argparse, json, subprocess, sys
import numpy as np
from netCDF4 import Dataset
parser=argparse.ArgumentParser()
parser.add_argument('--case',type=Path,required=True)
parser.add_argument('--lbc',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--init-source',required=True)
args=parser.parse_args()
case,lbc,out=args.case,args.lbc,args.out
base=[sys.executable,'-m','woof','hex','forecast','--backend','metal','--mesh','gfs-ruc-cap',
 '--grid',str(case/'gfs-ruc-cap.grid.nc'),'--static',str(case/'gfs-ruc-cap.static.nc'),
 '--init',str(case/'gfs-ruc-cap.init.nc'),'--init-source',args.init_source,'--lbc-dir',str(lbc),
 '--physics-suite','arwen_mpas_column','--physics','icloud=2','--physics','config_gwdo_scheme=off',
 '--physics','config_convection_scheme=off','--physics-mode','exact','--dt','row',
 '--history-every-minutes','2','--hours','0.03333333333333333']
for label in ('part','resume'):
 cmd=base+['--out',str(out/label),'--checkpoint',str(out/(label+'.npz'))]
 if label=='resume':cmd+=['--restart',str(out/'part.npz')]
 subprocess.run(cmd,check=True,timeout=180,stdout=subprocess.DEVNULL)
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
with Dataset(out/'full/history.000002.nc') as a, Dataset(out/'resume/history.000002.nc') as b:
 assert set(a.variables)==set(b.variables)
 assert {k:a.getncattr(k) for k in a.ncattrs()}=={k:b.getncattr(k) for k in b.ncattrs()}
 bad=[]; words=0
 for name in a.variables:
  x=np.asarray(a[name][:]); y=np.asarray(b[name][:]);words+=x.size
  if not equal(x,y): bad.append(name)
 print('history',len(a.variables),'fields',words,'values; differences',bad)
 assert not bad
with np.load(out/'full.npz') as a, np.load(out/'resume.npz') as b:
 assert set(a.files)==set(b.files)
 bad=[]; words=0
 for name in a.files:
  if name=='metadata':continue
  x=a[name];y=b[name];words+=x.size
  if not equal(x,y):bad.append(name)
 m=json.loads(str(a['metadata'])); n=json.loads(str(b['metadata']))
 print('checkpoint',len(a.files)-1,'arrays',words,'values; differences',bad,'metadata equal',m==n)
 assert not bad and m==n
 identity=m['physics']['backend']['seam']['identity']
 assert identity['physics']['icloud']==2 and identity['physics']['cu_physics']==0
 assert identity['backend']=='metal' and len(identity['species'])==6
 print('generic identity: icloud=2, no cumulus, six species')
