"""Compare closed strict RTRN kernels with the source-ordered CPU path."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smag_wrf_exact.compare import compare_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from woof.core.rrtm_lw import rtrn_columns, _sequential_sum
    from woof.core.rrtm_tables import load_rrtm_lw_tables, RRTM_BPADE, RRTM_FLUXFAC
    from woof.metal.runtime import MetalContext
    from woof.metal.array import MetalArrayModule
    from woof.metal.rrtm_lw_family import submit_rtrn_transfer

    tables = load_rrtm_lw_tables()
    rng = np.random.default_rng(158)
    ncol, nlay, nmodel, ngpt = 4, 5, 3, 140
    inputs = {
        'tavel': rng.uniform(190, 310, (ncol, nlay)).astype(np.float32),
        'tz': rng.uniform(190, 310, (ncol, nlay + 1)).astype(np.float32),
        'pz': np.broadcast_to(np.linspace(1000, 10, nlay + 1, dtype=np.float32), (ncol, nlay + 1)).copy(),
        'cldfrac': np.array([[0, 0, 0, 0, 0], [.3, 0, .6, 0, 0], [0, .8, 0, 0, 0], [.4, 0, .6, 0, 0]], np.float32),
        'taucloud': rng.uniform(0, 2, (ncol, nlay)).astype(np.float32),
        'itr': rng.integers(0, 5001, (ncol, nlay, ngpt), dtype=np.int32),
        'pfrac': rng.uniform(.001, .1, (ncol, nlay, ngpt)).astype(np.float32),
        'tbound': np.array([290, 340, 270, 300], np.float32),
        'semiss': rng.uniform(.5, 1, (ncol, 16)).astype(np.float32),
    }
    # Preserve a subnormal radiance through the transfer, including clear and cloudy layers.
    inputs['pfrac'][3] = np.float32(2**-125)
    inputs['tz'][1, 1] = np.float32(340)
    inputs['tavel'][1, 1] = np.float32(340)
    down, up, htr, htrc = rtrn_columns(np, tables, **inputs)
    exner = rng.uniform(.7, 1, (ncol, nmodel)).astype(np.float32)
    expected = {'glw': down[:, 0], 'olr': up[:, nmodel],
                'rthratenlw': ((htr[:, :nmodel] / np.float32(86400)) / exner).T.copy(),
                'htr': htr, 'clear_htr': htrc}
    statics = tables.statics
    common = {'totplnk': statics['TOTPLNK'].reshape((181, 16), order='F').copy(),
              'delwave': statics['DELWAVE'], 'ngb0': np.asarray(tables.ngb, np.int32) - 1,
              'tau_table': tables.tau, 'tf_table': tables.tf, 'trans_table': tables.trans}
    report = {'cases': {}}
    with MetalContext() as context:
        context.batch_kernel_calls = False
        submit = context._submit_kernel_batch
        def sync(*a, **kw):
            result = submit(*a, **kw)
            result.wait()
            return result
        context._submit_kernel_batch = sync
        xp = MetalArrayModule(context)
        device = {k: xp.asarray(v) for k, v in {**inputs, **common, 'exner': exner}.items()}
        shape = (ncol, nlay, ngpt)
        for name in ('bbu', 'bbutot', 'atot', 'abss'):
            device[name] = xp.zeros(shape, np.float32)
        poison = np.full((ncol, nlay + 1, ngpt), np.nan, np.float32)
        for name in ('down', 'clear_down', 'up', 'clear_up'):
            device[name] = xp.asarray(poison)
        for name in ('clear_up_flux', 'clear_down_flux'):
            device[name] = xp.zeros((ncol, nlay + 1), np.float32)
        for name in ('htr', 'clear_htr'):
            device[name] = xp.asarray(np.full((ncol, nlay + 1), np.nan, np.float32))
        for name in ('glw', 'olr'):
            device[name] = xp.zeros((ncol, 1), np.float32)
        device['rthratenlw'] = xp.zeros((nmodel, ncol, 1), np.float32)
        def run(stage, names, params, grid):
            context.submit_kernel('wrf_exact_rrtm_lw_rtrn_' + stage,
                                  {name: device[name] for name in names.split()}, params,
                                  exact_policy=True, grid_size=grid, threads_per_group=128)
        run('down', 'pfrac pz tz tavel cldfrac taucloud itr tbound semiss totplnk delwave tau_table tf_table trans_table ngb0 bbu bbutot atot abss down clear_down',
            {'ncol': ncol, 'nlay': nlay, 'ngpt': ngpt, 'ncell': ncol * ngpt, 'bpade': float(RRTM_BPADE)}, ncol * ngpt)
        run('up', 'pfrac tavel cldfrac semiss tbound totplnk delwave ngb0 bbu bbutot atot abss taucloud down clear_down up clear_up',
            {'ncol': ncol, 'nlay': nlay, 'ngpt': ngpt, 'ncell': ncol * ngpt}, ncol * ngpt)
        run('fluxes', 'pz up down exner rthratenlw glw olr clear_up clear_down clear_up_flux clear_down_flux htr clear_htr',
            {'ncol': ncol, 'nmodel': nmodel, 'nlay': nlay, 'ny': ncol, 'nx': 1,
             'ncell': ncol * (nlay + 1), 'heatfac': float(statics['HEATFAC'][0]), 'fluxfac': float(RRTM_FLUXFAC)}, ncol * (nlay + 1))
        for name, want in expected.items():
            report['cases'][name] = compare_words(device[name].to_numpy().reshape(want.shape), want)
        down_radiance = device['down'].to_numpy()
        up_radiance = device['up'].to_numpy()
        for name, radiance, want in (('down_profile', down_radiance, down),
                                     ('up_profile', up_radiance, up)):
            got = (_sequential_sum(np, radiance) * np.float32(.5)) * RRTM_FLUXFAC
            report['cases'][name] = compare_words(got, want)
        for name, want in (('clear_down_flux', down), ('clear_up_flux', up)):
            report['cases'][name + '_clear_column'] = compare_words(device[name].to_numpy()[0], want[0])
        for name in ('down', 'clear_down'):
            assert np.all(device[name].to_numpy()[:, -1].view(np.uint32) == 0)
        sub = down_radiance[3].view(np.uint32) & np.uint32(0x7fffffff)
        report['subnormal_down_words'] = int(np.count_nonzero((sub != 0) & (sub < 0x00800000)))
        assert report['subnormal_down_words'] > 0
        transfer_inputs = {name: device[name] for name in (
            'pfrac', 'pz', 'tz', 'tavel', 'cldfrac', 'taucloud', 'itr', 'tbound',
            'semiss', 'totplnk', 'delwave', 'tau_table', 'tf_table', 'trans_table',
            'ngb0', 'exner', 'rthratenlw', 'glw', 'olr')}
        transfer_params = {'ncol': ncol, 'nmodel': nmodel, 'nlay': nlay, 'ny': ncol, 'nx': 1,
                           'heatfac': float(statics['HEATFAC'][0]),
                           'fluxfac': float(RRTM_FLUXFAC), 'bpade': float(RRTM_BPADE)}
        scratch = submit_rtrn_transfer(**transfer_inputs, **transfer_params)
        for name, want in expected.items():
            got = (scratch if name in scratch else device)[name].to_numpy().reshape(want.shape)
            report['cases']['typed_' + name] = compare_words(got, want)
        submitted = []
        original_submit = context.submit_kernel
        def observe(*a, **kw):
            submitted.append(True)
            return original_submit(*a, **kw)
        context.submit_kernel = observe
        bad_temperature = inputs['tz'].copy(); bad_temperature[0, 0] = np.float32(341)
        bad_index = inputs['itr'].copy(); bad_index[0, 0, 0] = 5001
        bad_pressure = inputs['pz'].copy(); bad_pressure[0, 1] = bad_pressure[0, 0]
        bad_cloud = inputs['taucloud'].copy(); bad_cloud[1, 0] = np.float32(np.inf)
        for name, bad in (('tz', bad_temperature), ('itr', bad_index),
                          ('pz', bad_pressure), ('taucloud', bad_cloud)):
            try:
                submit_rtrn_transfer(**{**transfer_inputs, name: xp.asarray(bad)}, **transfer_params)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid {name} accepted')
        assert not submitted
        report['invalid_refused_before_submission'] = True
    report['different_words'] = sum(case['different_words'] for case in report['cases'].values())
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report['different_words']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
