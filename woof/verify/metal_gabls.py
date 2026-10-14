"""Native Metal execution of the GABLS1 single-column reference experiment.

The column is intentionally not routed through the three-dimensional dycore:
GABLS1 scores the separate stable single-column protocol in
``sase_case_math.column_stable_cooling``.  Prognostics and physical constants
are packed as binary64 words and the Metal shader evaluates them through the
shared software binary64 implementation.
"""
from __future__ import annotations

import numpy as np


def _words(values):
    """Contiguous IEEE binary64 values as the shader's raw ulong storage."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64)).view(
        np.uint64)


def column_stable_cooling(geo_u, geo_v, fcor, nz, dz, theta_init, dt, steps,
                          theta_surface, cooling_rate, z0=0.1, e_init=None,
                          stable_dissipation=False,
                          additive_dissipation=False):
    """Run the exact CPU-reference GABLS column sequence on native Metal."""
    from woof.core import sase_math as sm
    from woof.core.sase_limits import E_MIN
    from woof.verify import sase_case_math as reference
    from woof.verify.metal_adapter import CaseBackend

    nz, steps = int(nz), int(steps)
    if not 2 <= nz <= 128:
        raise ValueError("GABLS Metal column supports 2..128 levels")
    if steps < 1:
        raise ValueError("GABLS Metal column requires at least one step")
    dz, dt = float(dz), float(dt)
    if not np.isfinite(dz) or dz <= 0.0 or not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("GABLS dz and dt must be positive and finite")
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    theta = np.asarray(theta_init, dtype=np.float64)
    if theta.shape != (nz,) or not np.all(np.isfinite(theta)):
        raise ValueError("theta_init must be a finite (nz,) profile")
    if e_init is None:
        e = np.full(nz, E_MIN, dtype=np.float64)
    else:
        e = np.maximum(np.asarray(e_init, dtype=np.float64), E_MIN)
    if e.shape != (nz,) or not np.all(np.isfinite(e)):
        raise ValueError("e_init must be a finite (nz,) profile")

    # Match the constants and expression source used by sase_case_math.  The
    # state remains FP64 throughout; MetalArray's FP64 support is storage only.
    constants = _words([
        dt, dz, fcor, geo_u, geo_v, z0, theta_surface, cooling_rate,
        sm.KARMAN, sm.G_ACCEL, sm.BLACKADAR_LAMBDA, sm.LS_COEF,
        sm.C_KV, sm.C_KS, sm.CKS_BLEND_EXP, sm.prandtl_blend(0.0),
        sm.C_E, sm.C_ES, sm.C_ED, sm.BL89_MIX_EXP,
        sm.SFC_WSPD_FLOOR, reference.SFC_SIMILARITY_BETA_M,
        reference.SFC_SIMILARITY_BETA_H,
        float(bool(stable_dissipation)), float(bool(additive_dissipation)),
        E_MIN,
    ])

    with CaseBackend("metal") as backend:
        context, xp = backend.context, backend.arrays
        u = xp.asarray(_words(np.full(nz, float(geo_u))), dtype=np.uint64)
        v = xp.asarray(_words(np.full(nz, float(geo_v))), dtype=np.uint64)
        th = xp.asarray(_words(theta), dtype=np.uint64)
        en = xp.asarray(_words(e), dtype=np.uint64)
        kv = xp.empty((nz,), dtype=np.uint64)
        kface = xp.empty((nz - 1,), dtype=np.uint64)
        zg = xp.asarray(_words(z), dtype=np.uint64)
        c = xp.asarray(constants, dtype=np.uint64)
        skin = xp.asarray(_words([theta_surface]), dtype=np.uint64)
        series = xp.empty((steps, 4), dtype=np.uint64)
        stress = xp.empty((nz - 1, 3), dtype=np.uint64)

        with context.defer_kernel_submissions("GABLS1 exact column"):
            for step_index in range(steps):
                context.submit_kernel(
                    "gabls_column_step",
                    {"u": u, "v": v, "theta": th, "e": en,
                     "kv": kv, "kface": kface, "z": zg, "c": c,
                     "skin": skin, "series": series},
                    {"nz": nz, "steps": steps,
                     "step_index": step_index},
                    exact_policy=False, grid_size=1, threads_per_group=1)
        context.submit_kernel(
            "gabls_column_finish",
            {"u": u, "v": v, "theta": th, "e": en, "kv": kv,
             "kface": kface, "z": zg, "c": c, "stress": stress},
            {"nz": nz}, exact_policy=False, grid_size=1,
            threads_per_group=1)

        # read() fences the asynchronous submissions before copying results.
        u_out = u.to_numpy().view(np.float64).copy()
        v_out = v.to_numpy().view(np.float64).copy()
        theta_out = th.to_numpy().view(np.float64).copy()
        e_out = en.to_numpy().view(np.float64).copy()
        kv_out = kv.to_numpy().view(np.float64).copy()
        skin_series = series.to_numpy().view(np.float64).reshape(steps, 4).T
        stress_out = stress.to_numpy().view(np.float64).reshape(nz - 1, 3)

    z_face = 0.5 * (z[1:] + z[:-1])
    return {"z": z, "u": u_out, "v": v_out, "theta": theta_out,
            "e": e_out, "kv": kv_out, "z_face": z_face,
            "uw": stress_out[:, 0], "vw": stress_out[:, 1],
            "wt": stress_out[:, 2], "ust": skin_series[0],
            "tstar": skin_series[1], "theta_s": skin_series[2],
            "hfx_kin": skin_series[3]}
