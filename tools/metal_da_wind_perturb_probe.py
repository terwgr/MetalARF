"""Compare Metal wind application with the production host perturbation.

The provenance backend is NumPy because the draws and FFT run on the host.
The final wind addition runs on Metal.
"""
from types import SimpleNamespace
import numpy as np
from woof.da.perturb import PerturbationConfig, apply_perturbations, apply_metal_wind_perturbations
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def make_config(mode, taper, amplitude=1.):
    return PerturbationConfig.from_mapping({
        'dx_km': 1., 'dy_km': 1., 'rim_width': 3, 'rim_taper': taper,
        'wind_mode': mode, 'hypsometric_opt': 2,
        'fields': [{'name': name, 'amplitude': amplitude, 'length_scale_km': 2.}
                   for name in ('u', 'v')]})


def main():
    rng = np.random.default_rng(418)
    shapes = {'thp': (4, 16, 16), 'u': (4, 16, 17), 'v': (4, 17, 16)}
    fields = {name: rng.uniform(-2, 2, shape).astype(np.float32) for name,shape in shapes.items()}
    for name in ('u', 'v'):
        a = fields[name].view(np.uint32)
        a[:,0,0] = 0x80000000
        a[:,0,1] = 0x80000001
        a[:,-1,-1] = 0x7fc12345
        a[:,1,1] = 0x00000001
    checked_words = 0
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        for mode in ('rotational', 'independent'):
            for amplitude in (0., 1.):
                for taper in ('cosine', 'linear'):
                    cfg = make_config(mode,taper,amplitude)
                    reference = SimpleNamespace(**{name: a.copy() for name,a in fields.items()})
                    state = SimpleNamespace(_array_module=xp, **{name:xp.asarray(a) for name,a in fields.items()})
                    want_record = apply_perturbations(reference, 372, cfg)
                    got_record = apply_metal_wind_perturbations(state, 372, cfg)
                    assert got_record == want_record, (mode,taper,amplitude,'provenance')
                    for name in fields:
                        np.testing.assert_array_equal(state.__dict__[name].to_numpy().view(np.uint32),
                                                      reference.__dict__[name].view(np.uint32),
                                                      err_msg=str((mode,taper,amplitude,name)))
                        checked_words += fields[name].size
        cfg = make_config('rotational', 'cosine')
        for nz,ny,nx in ((0,4,4), (1,2,4), (1,4,2)):
            tiny = SimpleNamespace(_array_module=xp, thp=xp.zeros((nz,ny,nx),np.float32),
                                   u=xp.zeros((nz,ny,nx+1),np.float32),
                                   v=xp.zeros((nz,ny+1,nx),np.float32))
            assert apply_metal_wind_perturbations(tiny, 372, cfg) is None
        state = SimpleNamespace(_array_module=xp, **{name:xp.asarray(a) for name,a in fields.items()})
        before = {name:state.__dict__[name].to_numpy().view(np.uint32).copy() for name in fields}
        try:
            apply_metal_wind_perturbations(state, 1.5, cfg)
        except TypeError:
            pass
        else:
            raise AssertionError('invalid seed was accepted')
        for name in fields:
            np.testing.assert_array_equal(state.__dict__[name].to_numpy().view(np.uint32), before[name])
    print(f'0/{checked_words} different words; host draw provenance matched')

if __name__ == '__main__':
    main()
