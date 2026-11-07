"""Compare native RP3 temperature recovery with the selected source body."""
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.metal_mynn_word_probe import plain_arithmetic
from woof.hex.mpasphys.sf_mynn_v842 import _cuda_source, FIELDS as SF_FIELDS
from woof.hex.mpasphys.seaice_v842 import FIELDS
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    ice = np.array([0, .019, .02, .05, .099, .1, .15, .199, .2, .5, 1, 1.01], np.float32)
    skin = np.array([260, 260, 220, 230, 240, 240, 240, 250, 250, 270, 275, 280], np.float32)
    sst = np.array([280, 280, 275, 280, 285, 289, 280, 275, 274, 290, 280, 290], np.float32)
    n = ice.size
    packed = np.zeros((len(SF_FIELDS), n), np.float32)
    packed[SF_FIELDS.index('tsk')] = skin
    sea = np.zeros_like(packed)
    selected = HostModule('mpas_v842_seaice_rp3_source', plain_arithmetic(_cuda_source(frozenset({'RP3'}))))
    selected.get_function('sf_mynn_from_mpas')((1,), (128,), (packed, sea, np.ones(n, np.float32),
        sst, ice, np.float32(3000), np.float32(.02), np.int32(1), np.int32(n)))
    expected = np.stack((packed[SF_FIELDS.index('tsk')], sea[SF_FIELDS.index('tsk')]))
    with MetalContext(batch_kernel_calls=False) as context:
        submit = context._submit_kernel_batch
        def sync(*args, **kwargs):
            result = submit(*args, **kwargs); result.wait(); return result
        context._submit_kernel_batch = sync
        xp = MetalArrayModule(context)
        initial = np.zeros((len(FIELDS), n), np.float32)
        initial[FIELDS.index('tsk')] = skin
        initial[FIELDS.index('tsk_sea')] = skin
        fields = xp.asarray(initial)
        _launch('hexv842_mpas_seaice_rp3_f32', n, context,
            dict(fields=fields, sst=xp.asarray(sst), skin=xp.asarray(skin), ice=xp.asarray(ice)),
            dict(n=np.int32(n), sea_index=np.int32(FIELDS.index('tsk_sea')),
                 ice_index=np.int32(FIELDS.index('tsk')), threshold=np.float32(.02)))
        actual = fields.to_numpy()[[FIELDS.index('tsk'), FIELDS.index('tsk_sea')]]
    mismatches = np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))
    print(f'RP3 selected-source differences: {mismatches}/{actual.size}')
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
