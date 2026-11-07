"""Strict regional Kessler submission for the admitted idealized profile."""
import numpy as np

def _submit(name, resources, params, count):
    first = next(iter(resources.values()))
    flat = {name: value.view((int(value.size),)) if value.ndim != 1 else value
            for name, value in resources.items()}
    return first.buffer.context.submit_kernel(name, flat, params,
        exact_policy=True, grid_size=int(count), threads_per_group=128)


def prepare(state, *, t, rho, pii, z, dz8w, z_interface):
    nz, ny, nx = state.p.shape
    if state.thb.shape not in ((nz,), (nz, 1, 1), (nz, ny, nx)):
        raise ValueError("strict Kessler thb has unsupported storage shape")
    if state.phb.shape not in ((nz + 1,), (nz + 1, 1, 1), (nz + 1, ny, nx)):
        raise ValueError("strict Kessler phb has unsupported storage shape")
    return _submit('strict_regional_kessler_prep',
        {'thb': state.thb, 'thp': state.thp, 'alt': state.alt,
         'pressure': state.p, 'phb': state.phb, 'php': state.php,
         't': t, 'rho': rho, 'pii': pii, 'z': z, 'dz': dz8w,
         'zi': z_interface, 'saved': state.h_diabatic},
        {'nz': nz, 'nc': ny * nx, 'thb3d': int(state.thb.shape == (nz, ny, nx)),
         'phb3d': int(state.phb.shape == (nz + 1, ny, nx))}, ny * nx)


def column(*, t, qv, qc, qr, rho, pii, z, dz8w, rainnc, rainncv, dt):
    nz, ny, nx = t.shape
    if not 2 <= nz <= 256:
        raise ValueError('strict Kessler requires 2 to 256 vertical levels')
    resources = dict(t=t, qv=qv, qc=qc, qr=qr, rho=rho, pii=pii, z=z,
                     dz8w=dz8w, rainnc=rainnc, rainncv=rainncv)
    return _submit('strict_regional_kessler', resources,
        {'dt': np.float32(dt), 'nz': nz, 'ny': ny, 'nx': nx}, ny * nx)


def finish(state, t, *, dt, mp_tend_lim, no_mp_heating):
    dt = np.float32(dt)
    limit = np.float32(np.float32(mp_tend_lim) * dt)
    return _submit('strict_regional_kessler_finish',
        {'thp': state.thp, 't': t, 'saved': state.h_diabatic},
        {'dt': dt, 'limit': limit, 'n': int(t.size),
         'no_heating': int(bool(no_mp_heating))}, t.size)
