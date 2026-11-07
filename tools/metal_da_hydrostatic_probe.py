"""Compare the staged Metal analysis operator against the Rust host operator."""
from types import SimpleNamespace

import numpy as np

from woof.da.host_analysis import hydrostatic
from woof.da.hydrostatic import HydrostaticError
from woof.metal.runtime import MetalContext


def fixture(levels):
    rng = np.random.default_rng(3137)
    shape = (levels, 5, 11)
    eta = np.linspace(1.0, 0.0, levels + 1)
    mid = 0.5 * (eta[:-1] + eta[1:])
    dry_mass = rng.uniform(80_000.0, 98_000.0, shape[1:])
    perturbation = rng.uniform(-310.0, 310.0, shape[1:])
    p_top = 5000.0
    state = SimpleNamespace(
        p=p_top + mid[:, None, None] * (dry_mass + perturbation),
        thb=np.linspace(280.0, 350.0, levels),
        qv=np.ones(shape), qr=np.ones(shape), mub2d=dry_mass, mup=perturbation,
        c1f=0.9 + eta * 0.1, c2f=1000.0 * (1.0 - eta),
        c1h=0.9 + mid * 0.1, c2h=1000.0 * (1.0 - mid),
        c3f=eta, c4f=500.0 * eta, c3h=mid, c4h=500.0 * mid,
        rdnw=-np.linspace(levels, levels + 1, levels),
        rdn=-np.linspace(levels, levels + 2, levels),
        dnw=np.diff(eta), p_top=p_top,
    )
    before = dict(p=state.p, thp=rng.uniform(-12.0, 12.0, shape),
                  qv=rng.uniform(1e-5, 0.023, shape), qr=rng.uniform(0.0, 0.0008, shape))
    after = {name: field.copy() for name, field in before.items() if name != "p"}
    after["thp"][..., 1:] += rng.uniform(-7.0, 7.0, shape[:-1] + (shape[-1] - 1,))
    after["qv"][..., 1:] *= 0.94
    after["qr"][..., 1:] *= 1.4
    mass = perturbation.copy()
    mass[:, 1:] += 137.5
    return state, before, after, mass


def main():
    checked = 0
    with MetalContext(batch_kernel_calls=False) as context:
        def compare(state, before, after, option, mass):
            nonlocal checked
            kwargs = dict(hypsometric_opt=option, mup_after=mass)
            reference = hydrostatic(state, before, after, **kwargs)
            got = hydrostatic(state, before, after, metal_context=context, **kwargs)
            for name, actual, expected in zip(("phi", "pressure"), got, reference):
                np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64),
                                              err_msg=str((state.p.shape, option, mass is not None, name)))
                assert np.all(actual[..., 0] == 0.0)
                assert np.any(actual[..., 1:] != 0.0)
                checked += expected.size
            return got

        for levels in (1, 55, 129):
            state, before, after, mass = fixture(levels)
            for option in (1, 2):
                for mup in (None, mass):
                    compare(state, before, after, option, mup)
        state, before, after, _ = fixture(9)
        state.c1h[:] = 0.0
        state.c2h[:] = 1e-310
        # Force several differently sized chunks using the real memory policy.
        context.recommended_working_set_size = 16 * 12_000
        phi, _ = compare(state, before, after, 1, None)
        subnormal = int(np.count_nonzero((np.abs(phi) < np.finfo(np.float64).tiny) & (phi != 0)))
        assert subnormal
        before["p"] = np.full(state.p.shape, -10_000.0)
        for device in (None, context):
            try:
                hydrostatic(state, before, after, hypsometric_opt=2, metal_context=device)
            except HydrostaticError as exc:
                assert "not finite and positive" in str(exc)
            else:
                raise AssertionError("invalid analysed pressure was accepted")
    print(f"Rust host analysis comparison matched {checked:,} FP64 words, including {subnormal} subnormal phi words")


if __name__ == "__main__":
    main()
