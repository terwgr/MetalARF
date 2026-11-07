"""Compare a continuous strict run with the real checkpoint restore path."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smag_wrf_exact.compare import compare_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--moist', action='store_true')
    parser.add_argument('--igw-sounding', action='store_true')
    parser.add_argument('--moist-adv-opt', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    from woof.config import RunConfig
    from woof.core import dycore
    from woof.core.grid import make_vertical_coord, make_base_state
    from woof.core.state import init_theta_perturbation, adopt_wrf_theta_carrier
    from woof.io.restart import write_restart, restore_restart
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    from woof.state_serialization_contract import WRF_MUTS_SLOT

    cfg = RunConfig(nx=8, ny=7, nz=5, dx=1000., dy=1000., ztop=5000., dt=1., run_seconds=2.,
                    moist=args.moist, moist_adv_opt=args.moist_adv_opt)
    coord = make_vertical_coord(cfg.nz)
    if args.igw_sounding:
        from woof.verify.cases.igw import sounding
    else:
        sounding = lambda z: np.full_like(np.asarray(z, dtype=np.float64), 300.)
    base = make_base_state(coord, sounding,
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    def perturb(x, z):
        return np.broadcast_to((.02 * np.cos(np.pi*x/(cfg.nx*cfg.dx)))[None,None,:],
                               (cfg.nz,cfg.ny,cfg.nx))
    fields = ('u','v','w','thp','php','mup','p','al','alt','p_perturbation')
    if args.moist:
        fields += ('qv', 'qc', 'qr')
    with MetalContext() as context, tempfile.TemporaryDirectory() as directory:
        context.batch_kernel_calls = False
        submit = context._submit_kernel_batch
        def sync(*a, **kw):
            result = submit(*a, **kw)
            result.wait()
            return result
        context._submit_kernel_batch = sync
        xp = MetalArrayModule(context)
        def fresh():
            state = init_theta_perturbation(cfg, coord, base, perturb, array_module=xp)
            if args.igw_sounding:
                adopt_wrf_theta_carrier(state)
            if args.moist:
                z, y, x = np.indices((cfg.nz, cfg.ny, cfg.nx))
                for name, amount in (('qv', .008), ('qc', .0003), ('qr', .0001)):
                    values = amount * (1. + .1*np.cos(2*np.pi*x/cfg.nx) + .05*np.sin(2*np.pi*y/cfg.ny))
                    getattr(state, name)[...] = xp.asarray(values, dtype=np.float32)
                dycore.update_diagnostics(state, cfg.hypsometric_opt)
            return state
        continuous, split, restored = fresh(), fresh(), fresh()
        dycore.step(continuous, cfg)
        dycore.step(continuous, cfg)
        dycore.step(split, cfg)
        path = Path(directory) / 'checkpoint.npz'
        write_restart(path, split, cfg)
        restore_restart(path, restored, cfg)
        dycore.step(restored, cfg)
        cases = {name: compare_words(getattr(continuous, name).to_numpy(),
                                     getattr(restored, name).to_numpy()) for name in fields}
        cases['muts'] = compare_words(continuous.existing_scratch(WRF_MUTS_SLOT).to_numpy(),
                                      restored.existing_scratch(WRF_MUTS_SLOT).to_numpy())
    result = {'cases': cases, 'words': sum(c['words'] for c in cases.values()),
              'different_words': sum(c['different_words'] for c in cases.values())}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if result['different_words']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
