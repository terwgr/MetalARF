"""Compare public continuous and resumed checkpoint and final history fields."""
import argparse,json
from pathlib import Path
import netCDF4
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--final',type=Path,required=True);p.add_argument('--resumed',type=Path,required=True);p.add_argument('--history',type=Path,required=True);p.add_argument('--resume-history',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def header(z):return json.loads(z['__gpuwm_restart_header__'].tobytes())
with np.load(a.first,allow_pickle=False) as first, np.load(a.final,allow_pickle=False) as final, np.load(a.resumed,allow_pickle=False) as resumed:
 h1,h2,hr=map(header,(first,final,resumed)); ks=set(final.files)-{'__gpuwm_restart_header__'}
 assert {k:v for k,v in h2.items() if k not in ('created','checkpoint_set_id')} == {k:v for k,v in hr.items() if k not in ('created','checkpoint_set_id')}
 assert ks==set(resumed.files)-{'__gpuwm_restart_header__'}
 diff={};words=0
 for key in sorted(ks):
  x,y=final[key],resumed[key]
  if x.shape!=y.shape or x.dtype!=y.dtype:diff[key]='shape or dtype';continue
  count=int(np.count_nonzero(x.view(np.uint8)!=y.view(np.uint8)));words+=x.size
  if count:diff[key]=count
 fields={k:{'nonzero':int(np.count_nonzero(final[k])),'different_bytes':int(np.count_nonzero(final[k].view(np.uint8)!=resumed[k].view(np.uint8)))} for k in ('fields/swddir','fields/swddif','fields/swddni','fields/swdown','driver/radiation_tendencies/rtheta') if k in ks}
 scalars=[{'marker':h['driver'].get('swdd_from_scheme'),'calls':h['driver']['call_counts']} for h in (h1,h2,hr)]
 assert scalars[0]['calls']['radiation']==scalars[1]['calls']['radiation']==scalars[2]['calls']['radiation']==1
 assert scalars[0]['marker']==scalars[1]['marker']==scalars[2]['marker']
 nc=[netCDF4.Dataset(path) for path in (a.history,a.resume_history)]
 try:
  assert set(nc[0].variables)==set(nc[1].variables)
  history={};history_words=0;history_count=len(nc[0].variables)
  for name in nc[0].variables:
   x=np.asarray(nc[0][name][:]);y=np.asarray(nc[1][name][:]);history_words+=x.size
   if x.shape!=y.shape or x.dtype!=y.dtype or not np.array_equal(x,y):history[name]='different'
 finally:
  for d in nc:d.close()
 report={'checkpoint_arrays':len(ks),'checkpoint_values':words,'checkpoint_differences':diff,'semantic_headers_equal':True,'history_variables':history_count,'history_values':history_words,'history_differences':history,'radiation_fields':fields,'step1_step2_resume':scalars}
 a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert not diff and not history
