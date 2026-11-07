"""Compare actual damping caller against the selected host-compiled source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.w_damp_wrf_exact.generate import retained_source
from tools.smag_wrf_exact.compare import compare_words


def make_inputs(subnormal):
    nz, ny, nx = 6, 5, 7
    rng = np.random.default_rng(724)
    sh = (nz + 1, ny, nx)
    w = rng.uniform(-2, 2, sh).astype(np.float32)
    ww = rng.uniform(-2, 2, sh).astype(np.float32)
    rw = rng.uniform(-1, 1, sh).astype(np.float32)
    if subnormal:
        rw *= np.float32(2**-128)
        w *= np.float32(2**-128)
    # Exact threshold neighbors distinguish strict > from >=.
    ww[1].flat[:8] = np.array([0, 1, -1, np.nextafter(np.float32(1), np.float32(0)),
                            np.nextafter(np.float32(1), np.float32(2)), .5,
                            np.nextafter(np.float32(.5), np.float32(0)),
                            np.nextafter(np.float32(.5), np.float32(1))], np.float32)
    return dict(rw_t=rw, ww=ww, w=w, mup=np.zeros((ny,nx),np.float32),
                mub2d=np.ones((ny,nx),np.float32), c1f=np.ones(nz+1,np.float32),
                c2f=np.zeros(nz+1,np.float32), rdnw=np.ones(nz,np.float32))


def run(arrays, xp, metal, implicit, enabled):
    from woof.core.dycore import apply_w_damping
    a = {key: xp.asarray(value.copy()) for key, value in arrays.items()}
    state = SimpleNamespace(**a, _metal_setup_state=metal)
    if metal: state._array_module=xp
    cfg = SimpleNamespace(nz=6, ny=5, nx=7, dt=1., w_damping=int(enabled),
                          zadvect_implicit=int(implicit), w_crit_cfl=.5)
    apply_w_damping(state, cfg, state.ww)
    return state.rw_t.to_numpy() if metal else np.asarray(state.rw_t).copy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED: raise ValueError('GPUWM_WRF_EXACT must be enabled before import')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source='\n'.join(retained_source())
    module=host_backend.HostModule('wrf_exact_w_damp',source)
    from woof.core import dycore
    original=dycore.get_kernel
    dycore.get_kernel=lambda unit,name,*args: (module.get_function(name)
        if unit == 'openbc' and name == 'w_damp' else original(unit,name,*args))
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':{}}
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        for sub in (False,True):
            arrays=make_inputs(sub)
            for implicit in (False,True):
                for enabled in (False,True):
                    expected=run(arrays,np,False,implicit,enabled)
                    actual=run(arrays,xp,True,implicit,enabled)
                    name=f'sub{int(sub)}_implicit{int(implicit)}_enabled{int(enabled)}'
                    report['cases'][name]=compare_words(actual,expected)
    report['words']=sum(x['words'] for x in report['cases'].values())
    report['different_words']=sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if report['different_words']: raise SystemExit(1)


if __name__ == '__main__': main()
