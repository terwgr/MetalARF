"""Controlled CLI routing checks; no GPU allocation or full sweep."""
import contextlib, io, json, pathlib, signal, subprocess, sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import metal_combo_smoke as m
row = (0,0,0,0,0,'periodic')
results = []
for status,want in [('ok',0),('refused',0),('crash',1),('NaN',1)]:
    for mode in ['focused','matrix']:
        argv = ['smoke','--case',json.dumps(row),'--preflight-only'] if mode=='focused' else [
            'smoke','--microphysics','0','--pbl','0','--land-surface','0','--cumulus','0',
            '--radiation','0','--boundaries','periodic','--preflight-only','--jobs','1']
        with patch.object(sys,'argv',argv), patch.object(m,'result',return_value=(status,'controlled')), \
             contextlib.redirect_stdout(io.StringIO()):
            got = m.main()
        assert got==want,(mode,status,got,want)
        results.append(dict(mode=mode,status=status,exit=got))
    fake = SimpleNamespace(context=SimpleNamespace(close=lambda:None))
    with patch('woof.metal.array.MetalArrayModule',return_value=fake), \
         patch.object(m,'result',return_value=('ok','gates')), \
         patch.object(m,'execute',return_value=(status,'controlled')), \
         contextlib.redirect_stdout(io.StringIO()):
        got = m.run_single_context([('controlled',row,{})])
    assert got==want,('single-context',status,got,want)
    results.append(dict(mode='single-context',status=status,exit=got))
def deadline(*args,**kwargs): signal.raise_signal(signal.SIGALRM)
with patch('woof.metal.array.MetalArrayModule',return_value=fake), \
     patch.object(m,'result',side_effect=deadline), contextlib.redirect_stdout(io.StringIO()):
    assert m.run_single_context([('timeout',row,{})])==1
for mode in ['--pairwise','--suites']:
    with patch.object(sys,'argv',['smoke',mode]),patch.object(m,'pairwise_rows',return_value=[('one',row,{})]), \
         patch.object(m,'suite_rows',return_value=[('one',row,{})]), \
         patch.object(m,'_run_process_group',side_effect=subprocess.TimeoutExpired('controlled',285)), \
         contextlib.redirect_stdout(io.StringIO()):
        assert m.main()==124
results.append(dict(incomplete_exit=1,outer_timeout_exit=124))
print(json.dumps(results))
