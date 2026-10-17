"""The MYNN PBL coupling layer: WRF's ``mynnedmf_wrapper_run``, on device.

``module_pbl_driver.F:1660-1721`` does **not** call ``mynn_bl_driver``.  It
calls ``mynnedmf_wrapper_run`` (``module_bl_mynn_wrapper.F``), and that
wrapper owns three things the column solver does not:

* the mixing-ratio to specific-humidity conversion on the way in
  (``:453-475``) and the reverse on the tendencies and the subgrid cloud
  water on the way out (``:587-607``);
* ``initflag = 1`` on ``itimestep == 1`` and 0 afterwards (``:1643-1647``),
  which is what selects ``mym_initialize`` over the carried state; and
* the per-``j``-row packing that woof replaces with a single
  level-major column chunk, since every column is independent.

Keeping it in its own module rather than inside :mod:`woof.core.physics`
means the unit conversions live next to the source anchor that justifies
them.  Losing either one is a silent O(qv) bias in the moisture tendency,
not a crash, which is the kind of defect that survives a smoke test.
"""

from __future__ import annotations

from collections.abc import Mapping

try:
    import cupy as cp
except ImportError:
    cp = None
import weakref

import numpy as np

from woof.core.kernels import get_kernel
from woof.core.mynn_pbl_gpu import (
    _flag_mask,
    _nonfinite,
    mynn_bl_driver_cuda,
    mynn_pbl_kernel,
)
from woof.core.mynn_pbl_scratch import (
    MYNN_PBL_COLUMN_CHUNK,
    MYNN_PBL_STAGE_LAYERS,
    MYNN_PBL_TENDENCY_FIELDS,
    MynnPblScratch,
    SLOT_STAGE_DX,
    SLOT_STAGE_LAYER,
    SLOT_ZERO_COLUMN,
    mynn_pbl_tendency_field_shapes,
    mynn_column_pieces,
    resolve_mynn_column_chunk,
)
from woof.core.state import DTYPE


from woof.core.physics_inventory import MYNN_PBL_STATE_3D  # noqa: F401  (one home; re-exported here)


#: Solver output name -> the ``PhysicsDriver.fields`` key it persists into.
_STATE_FIELD = {
    "qke": "qke", "tsq": "tsq", "qsq": "qsq", "cov": "cov",
    "el": "el_pbl", "sh": "sh3d", "sm": "sm3d",
    "qc_bl": "qc_bl", "qi_bl": "qi_bl", "cldfra_bl": "cldfra_bl",
}

from woof.core.physics_inventory import (  # noqa: F401  (one home; re-exported here)
    MYNN_PBL_DIAGNOSTICS_2D,
    MYNN_PBL_DIAGNOSTICS_INT_2D,
)


_TPB = 128
#: Full MYNN wrapper call requires the lowest five model levels.
VERTICAL_LEVEL_BOUNDS = (5, None)

#: WRF Registry packages whose moist state declares ``F_QS``.  The PBL
#: driver forwards that generated flag at ``module_pbl_driver.F:873-878``.
#:
#: ``28`` (Registry.EM_COMMON:3036, package ``thompsonaero``:
#: ``moist:qv,qc,qr,qi,qs,qg``) belongs here for exactly the reason 8 does,
#: and was missing while mp_physics=28 was being ported: with the flag false
#: ``module_bl_mynn.F:734`` and ``:876`` substitute ``sqs = 0``, so MYNN saw
#: no snow at all under the one Thompson variant whose package declares it,
#: and ``:1324`` skipped ``rqsblten``.  MEASURED on the committed WRF v4.6.1
#: MYNN driver oracle's ``snow_anvil`` column (max sqs 4.08e-05): the
#: substitution drove ``qi_bl`` from 5.4863e-07 to exactly 0 and moved
#: ``qc_bl``, ``cldfra_bl``, ``rqvblten``, ``rthblten`` and ``exch_h`` with
#: it.  ``woof/physics_registry_v2.json`` has published 28 as a
#: flag_qs-true selector since the scheme was registered; the two are bound
#: together by ``tests/test_physics_registry.py::
#: test_the_registry_flag_qs_contract_is_the_one_the_shipped_runtime_applies``.
#: 9 (Milbrandt-Yau) belongs here for the same Registry reason:
#: ``Registry.EM_COMMON:3025`` declares ``package milbrandt2mom
#: mp_physics==9 - moist:qv,qc,qr,qi,qs,qg,qh``, so F_QS is true and
#: MYNN must see the scheme's snow.
#:
#: ``mp_physics = 50`` (P3 one-category) is deliberately ABSENT, and it is
#: the one selector whose absence needs no apology: P3 carries a single ice
#: category and its Registry package is ``moist:qv,qc,qr,qi`` with no ``qs``
#: at all (``Registry.EM_COMMON:3038``), so WRF's ``F_QS`` is false and
#: MYNN's ``sqs = 0`` is the scheme's own answer rather than a withheld
#: field.  Contrast mp=28, which HAD snow and was wrongly excluded here.
#:
#: 16 (WDM6) belongs here on the same Registry reading:
#: ``Registry.EM_COMMON:3031`` declares ``package wdm6scheme
#: mp_physics==16 - moist:qv,qc,qr,qi,qs,qg;scalar:qnn,qnc,qnr``, so F_QS
#: is true and the scheme's snow is MYNN's to see.
MYNN_SNOW_MICROPHYSICS = frozenset((6, 8, 9, 10, 16, 18, 28))


def mynn_flag_qs(mp_physics: int) -> bool:
    """Return WRF v4.6.1's Registry-derived ``FLAG_QS`` for ``mp_physics``."""

    return int(mp_physics) in MYNN_SNOW_MICROPHYSICS


#: Layer inputs the driver reads straight out of ``atmosphere``, paired with
#: the key they arrive under.  ``w`` is WRF's staggered vertical velocity
#: indexed kts..kte -- the LOWER interface of each mass layer, not a layer
#: mean (``module_pbl_driver.F:1663`` binds ``w=w``).
_ATMOSPHERE_LAYERS = (
    ("dz", "dz"), ("u", "u"), ("v", "v"), ("th", "theta"),
    ("p", "pressure"), ("exner", "exner"), ("rho", "rho"),
    ("tk", "temperature"), ("qv", "qv"), ("qc", "qc"), ("qi", "qi"),
    ("qs", "qs"),
)
#: Per-column inputs that are contiguous prefixes of a ``(ny, nx)`` field and
#: therefore need no staging buffer at all: a chunk of one is a contiguous
#: slice.
_COLUMN_FIELDS = (
    ("xland", "xland"), ("ts", "tsk"), ("ps", "psfc"), ("ust", "ust"),
    ("hfx", "hfx"), ("qfx", "qfx"), ("wspd", "wspd"), ("pblh", "pblh"),
    ("rmol", "rmol"),
)
#: Solver output -> the returned A-grid tendency name.
_TENDENCY_OUT = (
    ("rublten", "du"), ("rvblten", "dv"), ("rthblten", "dtheta"),
    ("rqvblten", "dqv"), ("rqcblten", "dqc"), ("rqiblten", "dqi"),
)


#: Stage B (W4 full admission): solver qn name -> (state scalar attribute,
#: returned tendency name).  ``qnbca`` is deliberately absent: WRF's mp=28
#: Registry package declares no qnbca (F_QNBCA false there), so the runtime
#: feeds the driver a zero qnbca column -- the solve on an exactly-zero
#: column with an exactly-zero flux is exactly zero, which is bitwise the
#: same answer WRF's skipped solve leaves -- and returns no qnbca tendency.
MYNN_MIXSCALARS_QN = (
    ("qnc", "nc", "dnc"),
    ("qni", "ni", "dni"),
    ("qnwfa", "nwfa", "dnwfa"),
    ("qnifa", "nifa", "dnifa"),
)


def mynn_pbl_step(
    atmosphere: Mapping[str, cp.ndarray],
    fields: Mapping[str, cp.ndarray],
    *,
    w: cp.ndarray,
    dx: float,
    delt: float,
    itimestep: int,
    mp_physics: int,
    state=None,
    column_chunk: int | None = None,
    qn_scalars: Mapping[str, cp.ndarray] | None = None,
    scalar_pblmix: int = 0,
    spp_pbl: int = 0,
    pattern_spp_pbl=None,
    **options,
) -> dict[str, cp.ndarray]:
    """Run one MYNN PBL call and write its state back into ``fields``.

    Returns the A-grid tendency set (``du``/``dv``/``dtheta``/``dqv``/
    ``dqc``/``dqi``) in the same shape and units
    :func:`woof.core.physics.couple_ysu_tendencies` consumes, so the
    mass-coupling and C-grid interpolation stay one implementation.

    ``itimestep`` is WRF's, one-based: the first model step is 1 and selects
    the ``mym_initialize`` cold start.  ``options`` are the MYNN identity
    knobs, forwarded verbatim to the driver so the refusal for an unported
    branch comes from the routine that would have had to implement it.

    The domain is walked in equal pieces of at most ``column_chunk``
    columns (:func:`woof.core.mynn_pbl_scratch.resolve_mynn_column_chunk`,
    this process's settled width, by default -- the same width the
    preflight registry sized the shared arena with;
    :func:`woof.core.mynn_pbl_scratch.mynn_column_pieces` makes the pieces
    equal, so no call is a short remainder).
    Every MYNN kernel gives one CUDA thread one complete column and reads no
    neighbour, so the chunk boundary is not a seam: the split is bitwise
    identical to the single wide call at every width measured, and that is
    asserted in ``tests/test_mynn_pbl_scratch.py`` rather than assumed.  What
    it buys is a working set that no longer grows with ``ny * nx`` -- the
    whole reason a preflight could not price this scheme before.
    """

    if int(itimestep) < 1:
        raise ValueError("MYNN PBL itimestep is one-based and starts at 1")
    theta = atmosphere["theta"]
    nz, ny, nx = theta.shape
    if nz < VERTICAL_LEVEL_BOUNDS[0]:
        raise ValueError(
            f"MYNN PBL requires at least {VERTICAL_LEVEL_BOUNDS[0]} "
            "vertical levels")
    ncol = ny * nx
    from woof.core.spp_kernel_sources import spp_flag
    stochastic = spp_flag(spp_pbl, "spp_pbl")
    stochastic_flat = None
    try:
        from woof.metal.array import MetalArray
    except ImportError:
        MetalArray = ()
    if stochastic and isinstance(theta, MetalArray):
        if (not isinstance(pattern_spp_pbl, MetalArray)
                or pattern_spp_pbl.shape != (nz, ny, nx)
                or pattern_spp_pbl.dtype != DTYPE
                or not pattern_spp_pbl.flags.c_contiguous
                or pattern_spp_pbl.buffer.context is not theta.buffer.context):
            raise ValueError("MYNN SPP requires float32 pattern_spp_pbl[nz,ny,nx] in the atmosphere's Metal context")
        if not np.isfinite(pattern_spp_pbl.to_numpy()).all():
            raise ValueError("MYNN SPP pattern must be finite")
        stochastic_flat = pattern_spp_pbl.view((nz, ncol))
    elif stochastic:
        if (not isinstance(pattern_spp_pbl, cp.ndarray)
                or pattern_spp_pbl.shape != (nz, ny, nx)
                or pattern_spp_pbl.dtype != DTYPE
                or pattern_spp_pbl.device.id != theta.device.id):
            raise ValueError("MYNN SPP requires float32 pattern_spp_pbl[nz,ny,nx] on the atmosphere device")
        if not bool(cp.all(cp.isfinite(pattern_spp_pbl)).item()):
            raise ValueError("MYNN SPP pattern must be finite")
        stochastic_flat = pattern_spp_pbl.reshape(nz, ncol)
    if w.shape[0] < nz:
        raise ValueError("MYNN PBL needs w on the lower interface of each "
                         "layer, i.e. at least nz levels")

    if isinstance(theta, MetalArray):
        return _mynn_pbl_step_metal(
            atmosphere, fields, w=w, dx=dx, delt=delt,
            itimestep=itimestep, mp_physics=mp_physics, state=state,
            qn_scalars=qn_scalars, scalar_pblmix=scalar_pblmix,
            options=options, rstoch=stochastic_flat)

    chunk = (resolve_mynn_column_chunk(nz) if column_chunk is None
             else int(column_chunk))
    if chunk < 1:
        raise ValueError("MYNN PBL column_chunk must be a positive integer")
    chunk = min(chunk, ncol)

    if state is None:
        work = MynnPblScratch.standalone(chunk, nz)
        tendencies = {name: cp.zeros((nz, ny, nx), dtype=DTYPE)
                      for name in MYNN_PBL_TENDENCY_FIELDS}
    else:
        work = MynnPblScratch.from_state(state, chunk, nz,
                                        bl_mynn_version=options.get("bl_mynn_version", "wrf_461"))
        tendencies = {
            name: state.scratch(shape, slot) for name, (slot, shape) in zip(
                MYNN_PBL_TENDENCY_FIELDS,
                mynn_pbl_tendency_field_shapes(nz, ny, nx).items())
        }

    # Stage B (W4 full admission): the qn columns for the five stock
    # mixscalars solves.  Key-off runs never enter this block, stage no
    # extra arrays, and pass no extra driver keys -- bit-identity of the
    # mixscalars=0 path is by construction.  The per-chunk level-major
    # copies below are plain pool allocations, accepted behind the key
    # (the priced-scratch discipline covers the always-on path only).
    mixscalars_on = int(options.get("bl_mynn_mixscalars", 0)) == 1
    scalar_diffusion_on = scalar_pblmix == 1
    qn_source: dict[str, cp.ndarray] = {}
    qn_out: dict[str, cp.ndarray] = {}
    if mixscalars_on or scalar_diffusion_on:
        if qn_scalars is None:
            raise TypeError(
                "PBL scalar mixing requires qn_scalars (the mp=28 "
                "nc/ni/nwfa/nifa fields); the caller owns the presence "
                "check so the refusal names the missing state, not a "
                "KeyError inside a chunk loop")
        for solver_name, state_name, _ in MYNN_MIXSCALARS_QN:
            qn_source[solver_name] = (
                qn_scalars[state_name].reshape(nz, ncol))
        qn_out = {out_name: cp.zeros((nz, ny, nx), dtype=DTYPE)
                  for _, _, out_name in MYNN_MIXSCALARS_QN}
    if mixscalars_on:
        # bl_mynn_mixscalars=1 mixes all five number-concentration
        # scalars, so all five qn flags must be set.
        for flag in ("flag_qnc", "flag_qni", "flag_qnwfa", "flag_qnifa",
                     "flag_qnbca"):
            options.setdefault(flag, True)

    # Flat (nz, ncol) / (ncol,) views.  Reshape on a C-contiguous field is a
    # view, so nothing here is a copy.
    source = {name: atmosphere[key].reshape(nz, ncol)
              for name, key in _ATMOSPHERE_LAYERS}
    source["w"] = cp.ascontiguousarray(w[:nz]).reshape(nz, ncol)
    for solver_name, field_name in _STATE_FIELD.items():
        source[solver_name] = fields[field_name].reshape(nz, ncol)
    flat_columns = {name: fields[key].reshape(ncol)
                    for name, key in _COLUMN_FIELDS}
    flat_kpbl = fields["kpbl"].reshape(ncol)
    out_state = {name: fields[field].reshape(nz, ncol)
                 for name, field in _STATE_FIELD.items()}
    out_state["exch_h"] = fields["exch_h"].reshape(nz, ncol)
    out_state["exch_m"] = fields["exch_m"].reshape(nz, ncol)
    flat_tendencies = {name: array.reshape(nz, ncol)
                       for name, array in tendencies.items()}

    initflag = 1 if int(itimestep) == 1 else 0
    piece = mynn_column_pieces(ncol, chunk)
    if scalar_diffusion_on:
        from woof.core.mynn_scalar_mix_gpu import scalar_pblmix_columns_cuda
        scalar_work = cp.empty(piece * (5 * nz + 1), dtype=DTYPE)
    for lo in range(0, ncol, piece):
        hi = min(lo + piece, ncol)
        n = hi - lo
        stage = work.group(SLOT_STAGE_LAYER, MYNN_PBL_STAGE_LAYERS, (n, nz))
        for name in source:
            stage[name].T[...] = source[name][:, lo:hi]

        # --- module_bl_mynn_wrapper.F:453-475 mixing ratio -> specific -----
        count = n * nz
        blocks = (count + _TPB - 1) // _TPB
        get_kernel("mynn_pbl", "mynn_wrapper_to_specific")(
            (blocks,), (_TPB,),
            (stage["qv"], stage["qc"], stage["qi"], stage["qs"],
             stage["sqv"], stage["sqc"], stage["sqi"], stage["sqs"],
             np.int32(count)),
        )

        spacing = work.one(SLOT_STAGE_DX, (n,))
        spacing[...] = DTYPE(dx)
        zeros_column = work.one(SLOT_ZERO_COLUMN, (n,))
        values = {name: stage[name] for name in
                  ("dz", "u", "v", "w", "th", "sqv", "sqc", "sqi", "sqs",
                   "p", "exner", "rho", "tk")}
        if options.get("bl_mynn_version", "wrf_461") == "gsd_41":
            values.update({name: stage[name] for name in ("qv", "qc", "qi")})
        for solver_name in _STATE_FIELD:
            values[solver_name] = stage[solver_name]
        values["dx"] = spacing
        # woof has no ocean-current coupling; WRF's uncoupled path passes
        # the zero-initialised UOCE/VOCE arrays (Registry default 0.0).
        values["uoce"] = zeros_column
        values["voce"] = zeros_column
        for name in flat_columns:
            values[name] = flat_columns[name][lo:hi]
        values["kpbl"] = flat_kpbl[lo:hi]
        if stochastic:
            values["rstoch"] = cp.asfortranarray(stochastic_flat[:, lo:hi].T)
        if mixscalars_on:
            for solver_name, _, _ in MYNN_MIXSCALARS_QN:
                values[solver_name] = cp.asfortranarray(
                    qn_source[solver_name][:, lo:hi].T)
            # F_QNBCA is false in mp=28's Registry package; the zero
            # column solves to exactly zero (see MYNN_MIXSCALARS_QN).
            values["qnbca"] = cp.zeros((n, nz), dtype=DTYPE, order="F")

        out = mynn_bl_driver_cuda(
            values, initflag=initflag, delt=DTYPE(delt), scratch=work,
            flag_qs=mynn_flag_qs(mp_physics), spp_pbl=spp_pbl, **options,
        )

        # --- module_bl_mynn_wrapper.F:587-607 specific -> mixing ratio -----
        # gsd_41 already returns mixing-ratio tendencies from the updated
        # vapour and keeps its in-cloud QC_BL as written.
        mynn_pbl_kernel("mynn_wrapper_from_specific",
                        options.get("bl_mynn_version", "wrf_461"))(
            (blocks,), (_TPB,),
            (stage["sqv"], out["rqvblten"], out["rqcblten"],
             out["rqiblten"], out["qc_bl"], out["qi_bl"], np.int32(count)),
        )

        for name, target in out_state.items():
            target[:, lo:hi] = out[name].T
        for name in ("pblh", "rmol", "kpbl"):
            fields[name].reshape(ncol)[lo:hi] = out[name]
        for name in (*MYNN_PBL_DIAGNOSTICS_2D, *MYNN_PBL_DIAGNOSTICS_INT_2D):
            fields[name].reshape(ncol)[lo:hi] = out[name]
        for solver_name, tendency_name in _TENDENCY_OUT:
            flat_tendencies[tendency_name][:, lo:hi] = out[solver_name].T
        if mixscalars_on:
            for solver_name, _, out_name in MYNN_MIXSCALARS_QN:
                qn_out[out_name].reshape(nz, ncol)[:, lo:hi] = (
                    out[f"r{solver_name}blten"].T)
        if scalar_diffusion_on:
            # module_pbl_driver.F calls diff4d after MYNN. Its local solve
            # replaces, rather than adds to, any MYNN scalar plume rate.
            # Precipitating number scalars are excluded by WRF's diff4d.
            for solver_name, _, out_name in MYNN_MIXSCALARS_QN:
                solved, rate = scalar_pblmix_columns_cuda(
                    cp.asfortranarray(qn_source[solver_name][:, lo:hi].T),
                    stage["dz"], stage["rho"], out["exch_h"],
                    DTYPE(delt), scratch=scalar_work[:n * (5 * nz + 1)].reshape(
                        (n, 5 * nz + 1), order="F"))
                qn_out[out_name].reshape(nz, ncol)[:, lo:hi] = rate.T
                # The stream orders the copy before allocator reuse. Drop
                # both outputs before the next species builds its buffers.
                del solved, rate

    result = dict(tendencies)
    result.update(qn_out)
    return result


def _mynn_pbl_step_metal(atmosphere, fields, *, w, dx, delt, itimestep,
                         mp_physics, state, qn_scalars, scalar_pblmix,
                         options, rstoch=None):
    """MetalArray caller for the admitted MYNN momentum-only source chain."""
    from woof.metal.array import MetalArray
    from woof.metal.mynn_pbl_family import (
        submit_mynn_wrapper_from_specific, submit_mynn_wrapper_to_specific)
    from woof.core.mynn_pbl_gpu import mynn_bl_driver_metal

    if state is None:
        raise TypeError("Metal MYNN requires DomainState persistent scratch")
    bl_mynn_version = options.get("bl_mynn_version", "wrf_461")
    if bl_mynn_version not in ("wrf_461", "gsd_41"):
        raise NotImplementedError(
            f"Metal MYNN caller does not support option "
            f"bl_mynn_version={bl_mynn_version!r}")
    selected_mixlength = options.get(
        "bl_mynn_mixlength", 2 if bl_mynn_version == "gsd_41" else 1)
    allowed = {
        "restart": False, "cycling": False, "closure": 2.6,
        "bl_mynn_cloudpdf": 2, "bl_mynn_mixlength": 1,
        "bl_mynn_edmf": 1, "bl_mynn_edmf_mom": 1,
        "bl_mynn_edmf_tke": 0, "bl_mynn_mixscalars": 0,
        "bl_mynn_cloudmix": 1, "bl_mynn_mixqt": 0,
        "bl_mynn_output": 0, "bl_mynn_tkeadvect": False,
        "icloud_bl": 1, "tke_budget": 0,
        "mix_chem": False, "flag_qc": True, "flag_qi": True,
        "flag_qs": mynn_flag_qs(mp_physics), "flag_qnc": False,
        "flag_qni": False, "flag_qnwfa": False, "flag_qnifa": False,
        "flag_qnbca": False, "flag_ozone": False,
        "bl_mynn_version": bl_mynn_version,
        # The Metal caller currently implements the corrected GSD branch;
        # its two opt-in defect switches remain fail-closed.
        "bl_mynn_gsd41_unsquared_qtke": False,
        "bl_mynn_cloud_tendency_form": "wrf_461",
    }
    mixscalars_on = options.get("bl_mynn_mixscalars", 0) == 1
    if mixscalars_on:
        # bl_mynn_mixscalars=1 mixes all five number-concentration
        # scalars, so all five qn flags must be set.
        allowed.update({"bl_mynn_mixscalars": 1, "flag_qnc": True,
                        "flag_qni": True, "flag_qnwfa": True,
                        "flag_qnifa": True, "flag_qnbca": True})
        for flag in ("flag_qnc", "flag_qni", "flag_qnwfa", "flag_qnifa",
                     "flag_qnbca"):
            options.setdefault(flag, True)
    for name, value in options.items():
        if name == "bl_mynn_mixlength":
            if type(value) is not int or value not in (1, 2):
                raise NotImplementedError(
                    f"Metal MYNN caller does not support option {name}={value!r}")
            continue
        if name not in allowed or value != allowed[name] or type(value) is not type(allowed[name]):
            raise NotImplementedError(
                f"Metal MYNN caller does not support option {name}={value!r}")
    # scalar_pblmix=1 is WRF's diff4d local diffusion of the mp=28 number
    # scalars after MYNN, ported below; bl_mynn_mixscalars=1 is the MYNN
    # plume + diffusive mixing of the same scalars inside the driver.
    if scalar_pblmix not in (0, 1) or type(scalar_pblmix) is not int:
        raise NotImplementedError(
            f"Metal MYNN caller does not support scalar_pblmix={scalar_pblmix!r}")
    if (scalar_pblmix == 1 or mixscalars_on) and qn_scalars is None:
        raise TypeError("PBL scalar mixing requires qn_scalars (the mp=28 "
                        "nc/ni/nwfa/nifa fields)")
    if scalar_pblmix == 0 and not mixscalars_on and qn_scalars is not None:
        raise NotImplementedError(
            "Metal MYNN qn_scalars need bl_mynn_mixscalars=1 or "
            "scalar_pblmix=1")
    nz, ny, nx = atmosphere["theta"].shape
    ncol = ny * nx
    context = atmosphere["theta"].buffer.context

    def layer(array, name):
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous or array.size != nz * ncol
                or array.buffer.context is not context):
            raise TypeError(f"Metal MYNN {name} must be same-context FP32 (nz,ny,nx)")
        return array.view((nz, ncol))

    def scalar(array, name, dtype=np.float32):
        if (not isinstance(array, MetalArray) or array.dtype != np.dtype(dtype)
                or not array.flags.c_contiguous or array.size != ncol
                or array.buffer.context is not context):
            raise TypeError(f"Metal MYNN {name} must be same-context {np.dtype(dtype)} (ny,nx)")
        return array.view((ncol,))

    source = {name: layer(atmosphere[key], key) for name, key in _ATMOSPHERE_LAYERS}
    source["w"] = layer(w.view((nz, ny, nx)), "w")
    sq = {name: state.scratch((nz, ncol), f"mynn_pbl_metal_specific_{name}")
          for name in ("sqv", "sqc", "sqi", "sqs")}
    submit_mynn_wrapper_to_specific(context, source["qv"], source["qc"],
        source["qi"], source["qs"], sq["sqv"], sq["sqc"], sq["sqi"],
        sq["sqs"], experimental=True)
    source.update(sq)
    values = dict(source)
    for solver_name, field_name in _STATE_FIELD.items():
        values[solver_name] = layer(fields[field_name], field_name)
    for name, key in _COLUMN_FIELDS:
        values[name] = scalar(fields[key], key)
    values["kpbl"] = scalar(fields["kpbl"], "kpbl", dtype=np.int32)
    values["dx"] = state.scratch((ncol,), "mynn_pbl_metal_dx")
    values["dx"].fill(np.float32(dx))
    # The uncoupled WRF path binds zero ocean currents.
    values["uoce"] = state.scratch((ncol,), "mynn_pbl_metal_uoce")
    values["voce"] = state.scratch((ncol,), "mynn_pbl_metal_voce")
    values["uoce"].fill(np.float32(0)); values["voce"].fill(np.float32(0))
    if mixscalars_on:
        for solver_name, state_name, _ in MYNN_MIXSCALARS_QN:
            values[solver_name] = layer(qn_scalars[state_name], state_name)
    out = mynn_bl_driver_metal(values, initflag=1 if int(itimestep) == 1 else 0,
        delt=delt, state=state, flag_qs=mynn_flag_qs(mp_physics),
        bl_mynn_mixlength=selected_mixlength,
        bl_mynn_version=bl_mynn_version, rstoch=rstoch,
        bl_mynn_mixscalars=int(mixscalars_on))
    if bl_mynn_version == "wrf_461":
        submit_mynn_wrapper_from_specific(
            context, sq["sqv"], out["rqvblten"], out["rqcblten"],
            out["rqiblten"], out["qc_bl"], out["qi_bl"],
            experimental=True)
    for name, field in _STATE_FIELD.items():
        fields[field][...] = out[name].view(fields[field].shape)
    for name in ("pblh", "rmol", "kpbl"):
        fields[name][...] = out[name].view(fields[name].shape)
    for name in (*MYNN_PBL_DIAGNOSTICS_2D, *MYNN_PBL_DIAGNOSTICS_INT_2D):
        target = fields[name]
        value = out.get(name)
        if value is not None:
            target[...] = value.view(target.shape)
    for name in ("exch_h", "exch_m"):
        if name in fields:
            fields[name][...] = out[name].view(fields[name].shape)
    tendencies = {}
    for solver, target in _TENDENCY_OUT:
        tendency = state.scratch(
            (nz, ncol), f"mynn_pbl_metal_tendency_{target}")
        tendency[...] = out[solver]
        tendencies[target] = tendency.view((nz, ny, nx))
    if mixscalars_on:
        for solver_name, _, out_name in MYNN_MIXSCALARS_QN:
            rate = state.scratch((nz, ncol), f"mynn_pbl_metal_{out_name}")
            rate[...] = out[f"r{solver_name}blten"]
            tendencies[out_name] = rate.view((nz, ny, nx))
    if scalar_pblmix == 1:
        from woof.metal.mynn_pbl_family import submit_mynn_scalar_pblmix
        # module_pbl_driver.F calls diff4d after MYNN with MYNN's exch_h.
        # One whole-domain scratch, reused by the four species; it is the
        # size of four layer fields, like the rates it produces.
        work = state.scratch((4 * nz, ncol), "mynn_pbl_metal_scalar_pblmix_work")
        exch_h = layer(out["exch_h"].view((nz, ny, nx)), "exch_h")
        for _, state_name, out_name in MYNN_MIXSCALARS_QN:
            rate = state.scratch((nz, ncol), f"mynn_pbl_metal_{out_name}")
            submit_mynn_scalar_pblmix(
                context, layer(qn_scalars[state_name], state_name),
                source["dz"], source["rho"], exch_h, rate, work,
                dt=np.float32(delt), nz=nz, ncol=ncol)
            tendencies[out_name] = rate.view((nz, ny, nx))
    return tendencies


#: Validity words for :func:`validate_mynn_tendencies`, which has no
#: workspace to draw from because ``physics.py`` calls it on the returned
#: fields.  Built on first use rather than at import so a CPU-only test run
#: can import this module, and held for the life of the process: six int32 words,
#: allocated once per device and stream, never per step.
#:
#: KEYED BY DEVICE AND STREAM, so concurrent slabs never share writable words.  Held as a single array
#: it was allocated on whichever device ran MYNN first, and a second device
#: in the same process then reduced into it -- CuPy refuses that outright
#: ("The device where the array resides (0) is different from the current
#: device (1). Peer access is unavailable"), which is the polite version of
#: the same defect that kills RRTMGP's k-distribution with an illegal
#: address.  MEASURED on a dual-4090: with this held per process, the
#: ``full+MYNN`` and ``full+MYNN+Noah-MP`` rungs cannot run on two cards in
#: one process at all.
_VALIDITY_FLAGS: dict[tuple[int, int], cp.ndarray] = {}


def release_mynn_stream_scratch(*, device_id, stream):
    """Retire the completed member queue's mutable tendency validity words."""
    device_id = int(device_id)
    if int(cp.cuda.runtime.getDevice()) != device_id:
        raise ValueError("MYNN scratch release requires the owning CUDA device")
    stream.synchronize()
    key = (device_id, int(stream.ptr))
    present = key in _VALIDITY_FLAGS
    _VALIDITY_FLAGS.pop(key, None)
    return {"validity_flags": int(present)}


def _validity_flags() -> cp.ndarray:
    key = (int(cp.cuda.Device().id), int(cp.cuda.get_current_stream().ptr))
    flags = _VALIDITY_FLAGS.get(key)
    if flags is None:
        flags = cp.zeros(len(MYNN_PBL_TENDENCY_FIELDS), dtype=cp.int32)
        _VALIDITY_FLAGS[key] = flags
    return flags


#: Per-context holder for the grouped health scan's cached workspace.
_METAL_SCAN_HOLDERS = weakref.WeakKeyDictionary()


def _metal_first_nonfinite(context, names, arrays):
    """Name of the first field (in ``names`` order) holding a non-finite
    value, or None, from one grouped device scan."""
    from types import SimpleNamespace
    from woof.core.health import FieldRule, HealthField
    from woof.metal.array import MetalArrayModule
    from woof.metal.health_family import submit_validate_full_state

    holder = _METAL_SCAN_HOLDERS.get(context)
    if holder is None:
        holder = SimpleNamespace(
            _metal_setup_state=True,
            _array_module=MetalArrayModule(context))
        _METAL_SCAN_HOLDERS[context] = holder
    fields = tuple(
        HealthField(name=f"mynn_pbl.{name}", values=array,
                    rule=FieldRule("generic"))
        for name, array in zip(names, arrays))
    submission, result = submit_validate_full_state(holder, fields)
    if submission is None:
        return None
    submission.wait()
    _status, packed = (int(value) for value in result.to_numpy())
    if packed == (1 << 64) - 1:
        return None
    return names[packed >> 48]


def validate_mynn_tendencies(out: Mapping[str, cp.ndarray]) -> None:
    """Refuse a non-finite MYNN tendency before it reaches RK3.

    The YSU path has the same guard for the same reason: a NaN that enters
    the slow-tendency slot is unattributable one step later.

    ``bool(cp.isfinite(a).all())`` per field allocated a full ``(nz, ny,
    nx)`` boolean temporary and synchronised six times -- 105.8 MiB of
    per-step churn at the d04 nest, on arrays that had just been produced
    without any.  The reduction writes one int32 word per field and the six
    are read together; ``tests/test_mynn_pbl_scratch.py`` pins the two
    spellings against each other rather than assuming they agree.
    """

    names = tuple(out)
    if not names:
        return
    try:
        from woof.metal.array import MetalArray
    except ImportError:
        MetalArray = ()
    if isinstance(out[names[0]], MetalArray):
        arrays = [out[name] for name in names]
        context = arrays[0].buffer.context
        if all(isinstance(a, MetalArray) and a.dtype == np.dtype(np.float32)
               and a.flags.c_contiguous and a.buffer.context is context
               for a in arrays):
            # One grouped device scan; only its compact first-failure
            # record is read back.  The record's high bits are the index
            # of the first failing field, in ``names`` order.
            bad = _metal_first_nonfinite(context, names, arrays)
            if bad is not None:
                raise FloatingPointError(
                    f"MYNN PBL produced a non-finite {bad} tendency")
            return
        for name in names:
            if not np.isfinite(out[name].to_numpy()).all():
                raise FloatingPointError(
                    f"MYNN PBL produced a non-finite {name} tendency")
        return
    mask = _flag_mask(_nonfinite(), [out[name] for name in names],
                      _validity_flags())
    for name, nonfinite in zip(names, mask):
        if nonfinite:
            raise FloatingPointError(
                f"MYNN PBL produced a non-finite {name} tendency")


__all__ = [
    "MYNN_PBL_DIAGNOSTICS_2D",
    "MYNN_PBL_DIAGNOSTICS_INT_2D",
    "MYNN_SNOW_MICROPHYSICS",
    "MYNN_PBL_STATE_3D",
    "mynn_flag_qs",
    "mynn_pbl_step",
    "validate_mynn_tendencies",
]
