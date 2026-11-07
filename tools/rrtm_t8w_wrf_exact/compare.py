"""Compare classic RRTM interface temperatures with the shared source caller."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smag_wrf_exact.compare import compare_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise ValueError('GPUWM_WRF_EXACT must be enabled')
    from woof.core.rrtm_lw import _t8w_columns, RRTMLongwaveRadiation, RRTMDudhiaRadiation
    from types import SimpleNamespace
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    from woof.metal.rrtm_lw_family import submit_pack_top_down, submit_t8w
    rng = np.random.default_rng(901)
    report = {'cases': {}}
    with MetalContext() as context:
        context.batch_kernel_calls = False
        submit = context._submit_kernel_batch
        def sync(*a, **kw):
            submission = submit(*a, **kw)
            submission.wait()
            return submission
        context._submit_kernel_batch = sync
        xp = MetalArrayModule(context)
        # Both the standalone and paired caller must refuse before a driver
        # can publish fields or increment its cadence counter.
        adapter = object.__new__(RRTMLongwaveRadiation)
        adapter.update_count = 0
        adapter.icloud = 2
        paired = object.__new__(RRTMDudhiaRadiation)
        paired.longwave_adapter = adapter
        fields = {'sentinel': object()}
        atmosphere = {'temperature': xp.asarray(np.ones((3, 1, 1), np.float32))}
        for caller in (adapter, paired):
            try:
                caller(atmosphere=atmosphere, fields=fields,
                       state=SimpleNamespace(), cfg=SimpleNamespace())
            except NotImplementedError as error:
                assert 'strict Metal classic RRTM' in str(error)
            else:
                raise AssertionError('unported strict RRTM caller did not refuse')
            assert adapter.update_count == 0 and list(fields) == ['sentinel']
        adapter.icloud = 1
        for cfg in (SimpleNamespace(mp_physics=8), SimpleNamespace(bl_pbl_physics=1)):
            try:
                adapter(atmosphere=atmosphere, fields=fields,
                        state=SimpleNamespace(), cfg=cfg)
            except NotImplementedError:
                pass
            else:
                raise AssertionError('unproved strict RRTM cloud coupling was admitted')
            assert adapter.update_count == 0 and list(fields) == ['sentinel']
        report['direct_callers_refused_before_mutation'] = True
        for nz, ny, nx in ((3, 2, 3), (9, 3, 7)):
            z = np.arange(nz + 1, dtype=np.float32)[:, None, None]
            z = np.broadcast_to(z * z + 10 * z, (nz + 1, ny, nx)).copy()
            fnm = rng.uniform(.1, .9, nz).astype(np.float32)
            fnp = (np.float32(1) - fnm).astype(np.float32)
            for tiny in (False, True):
                t = rng.uniform(-2, 2, (nz, ny, nx)).astype(np.float32)
                t = t * np.float32(2**-128) if tiny else t + np.float32(280)
                bottom = np.ascontiguousarray(t.transpose(1, 2, 0)).reshape(ny*nx, nz)
                bottom_z = np.ascontiguousarray(z.transpose(1, 2, 0)).reshape(ny*nx, nz+1)
                expected = _t8w_columns(np, bottom, bottom_z, fnm, fnp)[:, ::-1].copy()
                packed_t = submit_pack_top_down(xp.asarray(t))
                packed_z = submit_pack_top_down(xp.asarray(z))
                actual = submit_t8w(packed_t, packed_z, xp.asarray(fnm), xp.asarray(fnp)).to_numpy()
                report['cases'][f'nz{nz}_tiny{int(tiny)}'] = compare_words(actual, expected)
                if tiny:
                    assert report['cases'][f'nz{nz}_tiny1']['subnormal_words'] > 0
    report['words'] = sum(x['words'] for x in report['cases'].values())
    report['different_words'] = sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if report['different_words']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
