"""Randomized physical convective columns shared by the host and Metal probes."""
import numpy as np

G = 9.81
RD = 287.04


def qsat(t, p):
    es = np.where(t > 273.16,
                  610.78 * np.exp(17.2693882 * (t - 273.16) / (t - 35.86)),
                  610.78 * np.exp(21.875 * (t - 273.16) / (t - 7.66)))
    return (0.622 * es / np.maximum(p - es, 1.0)).astype(np.float32)


def make_columns(nz, ncol, seed):
    rng = np.random.default_rng(seed)
    f = np.float32
    ts = rng.uniform(255., 305., ncol)
    gamma = rng.uniform(5.5, 9.0, ncol) * 1e-3
    conv = rng.random(ncol) < 0.45
    ts = np.where(conv, rng.uniform(297., 305., ncol), ts)
    gamma = np.where(conv, rng.uniform(7.0, 9.0, ncol) * 1e-3, gamma)
    inv = rng.random(ncol) < 0.3
    dz = np.empty((nz, ncol))
    for k in range(nz):
        dz[k] = (50. + 1500. * (k / max(nz - 1, 1)) ** 1.6) * rng.uniform(0.95, 1.05, ncol)
    zi = np.vstack([np.zeros(ncol), np.cumsum(dz, axis=0)])
    zm = 0.5 * (zi[1:] + zi[:-1])
    t = np.maximum(ts - gamma * zm, 212.)
    t = t + np.where(inv[None] & (zm > 900.) & (zm < 1300.), 3.5, 0.)
    p0 = rng.uniform(98000., 102000., ncol)
    p8w = np.empty((nz + 1, ncol))
    p8w[0] = p0
    for k in range(nz):
        p8w[k + 1] = p8w[k] * np.exp(-G * dz[k] / (RD * t[k]))
    pm = np.sqrt(p8w[1:] * p8w[:-1])
    rho = pm / (RD * t)
    rh0 = np.where(conv, rng.uniform(0.9, 1.0, ncol), rng.uniform(0.6, 1.0, ncol))
    scale = np.where(conv, rng.uniform(5000., 9000., ncol), rng.uniform(2500., 6000., ncol))
    rh = np.clip(rh0 * np.exp(-zm / scale) + rng.uniform(0., 0.08, (nz, ncol)), 0., 1.05)
    qv = (rh * qsat(t.astype(f), pm.astype(f))).astype(f)
    qc = np.where((rh > 0.97) & (t > 258.), rng.uniform(0., 6e-4, (nz, ncol)), 0.).astype(f)
    qi = np.where((rh > 0.95) & (t < 268.), rng.uniform(0., 4e-4, (nz, ncol)), 0.).astype(f)
    wcol = rng.uniform(-0.1, 1.8, ncol)
    w = np.empty((nz + 1, ncol))
    for k in range(nz + 1):
        w[k] = wcol * np.sin(np.pi * min(zi[k].mean() / 14000., 1.)) * 1.0
    w = (w + rng.normal(0., 0.02, w.shape)).astype(f)
    qvften = (rng.uniform(0., 2e-6, (nz, ncol)) * np.exp(-zm / 3000.)).astype(f)
    qvpbl = (rng.normal(0., 2e-8, (nz, ncol)) * np.exp(-zm / 1500.)).astype(f)
    u = rng.normal(5., 8., (nz, ncol)).astype(f)
    v = rng.normal(0., 6., (nz, ncol)).astype(f)
    exner = (pm / 1e5) ** (287.04 / 1004.)
    ptop = 5000.
    znu = ((pm[:, 0] - ptop) / (p8w[0, 0] - ptop))
    xland = rng.choice([1., 2.], ncol)
    qfx = rng.uniform(0., 4e-4, ncol)
    # Quiet columns: nothing to convect, every moisture field zero.
    quiet = rng.random(ncol) < 0.12
    for a in (qv, qc, qi, qvften, qvpbl):
        a[:, quiet] = 0.
    w[:, quiet] = 0.
    qfx[quiet] = 0.
    cases = dict(
        u3d=u, v3d=v, w=w, t3d=t.astype(f), qv3d=qv, qc3d=qc, qi3d=qi,
        pi3d=exner.astype(f), rho3d=rho.astype(f), qvften=qvften,
        qvpblten=qvpbl, dz8w=dz.astype(f), pcps=pm.astype(f), p8w=p8w.astype(f),
        xland=xland.astype(f), qfx=qfx.astype(f), znu=znu.astype(f))
    return {k: np.ascontiguousarray(v) for k, v in cases.items()}
