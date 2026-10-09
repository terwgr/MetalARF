"""Grell-Freitas cumulus adapter: the ``cu_physics=3`` engine callable.

The scheme itself is ``woof/globe/core/kernels/gf.cu`` -- the whole of WRF
v4.6.1's GFDRV per column, held at max_ulp 0 against the byte-frozen
oracle by ``tests/test_gf_gfdrv_cuda.py`` with fzu computed on the device.
This module is only the seam: it packs the engine's column state into the
kernel's input layout, launches ``gf_gfdrv_stage`` over every column, and
returns the driver's Task-1 :class:`~woof.globe.core.physics.CumulusResult`
(rates held until the next due call, RAINCV consumed once per due call --
GF has no NCA persistence; WRF recomputes it every STEPCU step, and the
importer pins ``cudt = 0`` for ``cu_physics=3`` so STEPCU is 1, exactly
WRF's usual GF configuration).

The kernel runs the SHIPPED identity: corrected k22 indexing
(``k22_wrf_faithful = 0``).  The WRF-faithful off-by-one is reachable only
through the parity suites, never from here; the measured ledger for the
difference is tests/test_gf_shallow_cuda.py::
test_the_corrected_k22_ledger_entry (three fixture cases move, all three
rejected either way, zero output words differ).

Recorded deviations of THIS seam (the kernel behind it is bitwise; these
are about what the engine can hand it today):

1. **Forcing tendencies** (CLOSED, task #243).  GFDRV's forced states
   (``tn``/``qo``) sum advective (RTHFTEN/RQVFTEN), radiative (RTHRATEN)
   and boundary-layer (RTHBLTEN/RQVBLTEN) forcing, and all four auxiliary
   lanes are read off the bound driver:
   ``gf_rthdynten``/``gf_rqvdynten`` (the integrator's advective rates --
   the ARW dycore exports them at RK stage 1 of every step into
   ``state.rthften``/``state.rqvften``, bound at driver construction; the
   MPAS driver computes its own exactly as native v8.4.1's
   mpas_atm_time_integration.F:6936 + :2789 construction) and
   ``gf_rthblten``/``gf_rqvblten`` (the PBL slot's raw rates, retained by
   ``PhysicsDriver._couple_pbl_slot`` under WRF's between-calls
   persistence contract -- YSU, MYJ, MYNN, Shin-Hong and SASE all fill
   the slot on the same terms, exactly as WRF's cumulus driver reads
   RTHBLTEN/RQVBLTEN without asking which scheme wrote them).  A lane
   whose driver attribute is ``None`` feeds zeros -- what a driverless
   adapter and the parity harnesses get.

   THE ADVECTIVE LANE IS PURE ADVECTION, deliberately.
   ``module_cumulus_driver.F:867`` pre-folds ``RTHRATEN + RTHBLTEN`` into
   ``RTHFTEN`` for G3SCHEME and NTIEDTKESCHEME only; GFSCHEME is not in
   that list because the kernel sums the three lanes itself at
   ``kernels/gf.cu:4428``.  Handing this adapter a pre-folded RTHFTEN
   would integrate the boundary layer and the sky twice.
2. **Convective momentum tendencies.**  The kernel computes GF's
   dudt/dvdt; :class:`CumulusResult` carries no momentum slots, so they
   are not yet coupled.  WRF couples them; MPAS-A v8.4.1 does NOT
   (its cu_grell_freitas call carries no rucuten/rvcuten), so for the
   MPAS seam this is native parity, not a gap.
3. **w on mass levels** is the KF-precedent average
   ``0.5*(w[k] + w[k+1])`` of the staggered field.
4. **dx** is per-column when the driver carries ``gf_dx_column``
   ([ny, nx], metres); the scalar ``cfg.dx`` otherwise.  WRF's own GFDRV
   takes dx(i,j), so the per-column feed is the WRF interface, not an
   extension.

Ice routing follows WRF's own ``F_QI`` gate: with a prognostic ``qi`` the
258 K split routes cold condensate to RQICUTEN; without one, everything
lands in RQCCUTEN (module_cu_gf_wrfdrv.F:810-840 with F_QI false), which
is exact here because the split routes one number to one of two slots.
"""

from __future__ import annotations

import numpy as np

from woof.core.kf import _model_clock_dt

DTYPE = np.float32

#: gf.cu's compiled level bound; nz above this recompiles via the
#: integer-define tier.
_GF_KMAX_DEFAULT = 40

# gf.cu input/output enum orders (gf_gfdrv_stage).  Must match the kernel;
# tools/gf_wrf461_oracle/gf_field_lists.py carries the same lists for the
# parity suites.
_IN_LEV = ("u", "v", "w", "t", "qv", "p", "pi", "rho", "dz8w", "p8w",
           "rthften", "rqvften", "rthraten", "rthblten", "rqvblten")
_OUT_LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt", "dvdt",
            "gdc", "gdc2",
            "outt", "outq", "outqc", "outu", "outv",
            "outts", "outqs", "outqcs")
_OUT_SCA = ("raincv", "pratec", "htop", "hbot", "xmb_shallow", "pret",
            "prets", "cuten", "cutens",
            # the deep closure reading (gf.cu GfClosureReading), diagnostic
            "xmb_request", "xmb_applied", "xmb_floor", "floor_fraction", "xf_quasi_equilibrium", "xf_omega",
            "xf_moisture_convergence", "xf_ecmwf", "xf_dicycle", "mconv",
            "mconv_den", "pr_ens7", "sig", "closure_n", "neg_check_factor",
            "heating_cap_k_day")
_OUT_ISCA = ("ktop_deep", "k22_shallow", "kbcon_shallow", "ktop_shallow",
             "kbcon", "ktop", "ierr_deep", "downdraft_dry_exit",
             "closure_family", "downdraft_massless_levels")

#: The per-column closure reading the seam exposes as
#: ``last_column_diagnostics["gf_<name>"]`` after every call: the deep
#: ensemble's requested cloud-base mass flux and the one applied (both
#: kg/m2/s), the four family requests (members 1, 4, 7 and 10 of the
#: sixteen), the moisture convergence they were built from, the neg_check
#: factor and the heating cap it held the column to, the family with the
#: largest request and the downdraft's massless levels.
CLOSURE_READING_SCALARS = _OUT_SCA[9:]
CLOSURE_READING_INTEGERS = _OUT_ISCA[6:]
#: closure_family codes.
CLOSURE_FAMILIES = {0: "none", 1: "quasi_equilibrium", 2: "omega",
                    3: "moisture_convergence", 4: "ecmwf"}


def _gf_module(nz: int, updraft_only_when_downdraft_dry: bool = False,
               resolved_convergence_closure: bool = False,
               resolved_convergence_floor_percent: int = 100):
    """The compiled kernel for ``nz`` levels; with
    ``updraft_only_when_downdraft_dry`` the deep arm keeps running, without
    a downdraft, in the columns WRF's GFDRV rejects because their downdraft
    cannot form (the ierr 7 exits and cup_dd_moisture's ierr 51 exit; gf.cu
    GF_UPDRAFT_ONLY_WHEN_DOWNDRAFT_DRY); with
    ``resolved_convergence_closure`` the deep arm takes its coarse-column
    form (gf.cu GF_RESOLVED_CONVERGENCE_CLOSURE: the heating cap yields to
    the column's resolved latent-heat supply, the downdraft sweeps continue
    past massless levels, a downdraft that cannot form leaves the updraft
    running).  Both off, the module is the oracle-graded kernel word for
    word."""
    defines = _gf_defines(nz, updraft_only_when_downdraft_dry,
                          resolved_convergence_closure,
                          resolved_convergence_floor_percent)
    if not defines:
        from woof.globe.core.kernels import load_module

        return load_module("gf")
    from woof.globe.core.kernels import load_module_int_defines

    return load_module_int_defines("gf", defines)


def _gf_defines(nz: int, updraft_only_when_downdraft_dry: bool = False,
                resolved_convergence_closure: bool = False,
                resolved_convergence_floor_percent: int = 100) -> tuple[tuple[str, int], ...]:
    """The integer defines :func:`_gf_module` compiles gf.cu with.  Every
    value is a positive integer, the only kind the define loader takes: a
    floor percent of 0 (no floor) is spelt GF_RESOLVED_CONVERGENCE_FLOOR_DISABLED
    1, because handing the loader GF_RESOLVED_CONVERGENCE_FLOOR_PERCENT 0
    raised in it and the documented no-floor setting never compiled."""
    defines: list[tuple[str, int]] = []
    if nz > _GF_KMAX_DEFAULT:
        defines.append(("GF_KMAX", int(nz)))
    if updraft_only_when_downdraft_dry:
        defines.append(("GF_UPDRAFT_ONLY_WHEN_DOWNDRAFT_DRY", 1))
    if resolved_convergence_closure:
        defines.append(("GF_RESOLVED_CONVERGENCE_CLOSURE", 1))
        percent = int(resolved_convergence_floor_percent)
        if percent == 0:
            defines.append(("GF_RESOLVED_CONVERGENCE_FLOOR_DISABLED", 1))
        elif percent != 100:
            defines.append(("GF_RESOLVED_CONVERGENCE_FLOOR_PERCENT", percent))
    return tuple(defines)


def gf_kernel_capacity(nz: int) -> int:
    """The ``GF_KMAX`` the module is compiled at for this ``nz``."""
    return _GF_KMAX_DEFAULT if nz <= _GF_KMAX_DEFAULT else int(nz)


# ---------------------------------------------------------------------------
# The per-thread column workspace
# ---------------------------------------------------------------------------
# gf.cu used to keep GFDRV's column arrays in the per-thread local frame, and
# CUDA prices a local frame at the card's RESIDENT-THREAD CAPACITY -- one
# per-context backing store of (frame - 1024) * SMs * maxThreadsPerSM, taken
# at first launch and never returned.  MEASURED on an RTX 5070 Ti (70
# SMs x 1,536, sm_120, NVRTC 13.3): the 22,416 B frame took 2,196.0 MiB while
# the kernel only ever had 384 threads/SM in flight.  The arrays now live in
# a global workspace this adapter allocates, sized to the threads ACTUALLY in
# flight, and the columns are launched in tiles of that size.
#
# These three must match gf.cu's GFWS_SLOT_COUNT_COL / GFWS_SLOT_COUNT_DRV /
# GFWS_SLOTS; tests/test_gf_workspace.py re-derives them from the .cu source
# and fails if either side moves alone.
GFWS_SLOT_COUNT_COL = 90
GFWS_SLOT_COUNT_DRV = 36
GFWS_SLOTS = GFWS_SLOT_COUNT_COL + GFWS_SLOT_COUNT_DRV

#: Launch block, and the tile's granularity.  gf.cu indexes the workspace by
#: the global thread id within a tile, so a tile is always a whole number of
#: blocks.
GF_BLOCK = 64

#: Blocks per SM the tile is sized for.  MEASURED, not assumed: at 100,000
#: columns on an RTX 5070 Ti (70 SMs) the kernel's wall is 31.07 ms at
#: 2 blocks/SM, 24.18 at 4, 24.33 at 6, 25.14 at 8 and 26.63 at 12, against
#: 19.33 ms for the pre-cut local-frame kernel.  4 is the plateau and the
#: cheapest point on it: 422 MiB of workspace where 12 would cost 1,266 MiB
#: and run slower.  The kernel's hardware occupancy at block=64 is 12
#: blocks/SM, so this is a throughput choice, not a residency limit, and the
#: query below only ever lowers it.
GF_TILE_BLOCKS_PER_SM = 4


def gf_workspace_floats(nz: int, columns: int) -> int:
    """Workspace floats for ``columns`` columns in flight at this ``nz``.

    Rounded up to whole blocks: gf.cu interleaves the workspace by LANE
    within a block, the way CUDA lays local memory out across a warp, so the
    unit of allocation is one block's region, not one column's.
    """
    blocks = (int(columns) + GF_BLOCK - 1) // GF_BLOCK
    return blocks * GFWS_SLOTS * (gf_kernel_capacity(nz) + 9) * GF_BLOCK


#: Columns packed and launched per pass.  The kernel's input and output
#: layouts are (columns, lanes, nz) blocks -- 15 input lanes and 16 output
#: lanes -- and they used to be built for the WHOLE batch at once.  MEASURED
#: on the T533 global batch (1,283,202 columns x 40 levels, RTX 5090 32 GB):
#: the input block was 3.07 GB, the output block 3.28 GB, and the output
#: allocation failed with 32.2 GB already live.  The kernel is one thread
#: per column and never reads across columns, so packing the columns in
#: chunks changes memory and nothing else: at this default the two blocks
#: are 0.65 GB at 40 levels.  ``GrellFreitas(column_chunk=...)`` overrides.
GF_COLUMN_CHUNK = 131_072


def gf_column_chunks(ncol: int, chunk: int) -> list[tuple[int, int]]:
    """``[lo, hi)`` column ranges covering ``ncol`` in pieces of ``chunk``."""
    ncol = int(ncol)
    chunk = int(chunk)
    if chunk < 1:
        raise ValueError(f"column_chunk must be positive, got {chunk}")
    return [(lo, min(lo + chunk, ncol)) for lo in range(0, ncol, chunk)]


def gf_tile_columns(fn, ncol: int) -> int:
    """Columns to keep in flight: enough to fill the card, and no more.

    The whole point of the workspace is that it is charged per thread IN
    FLIGHT where the local-memory backing store was charged per thread the
    card could ever hold.  Over-sizing the tile hands that 4x back.
    """
    import cupy as cp

    dev = cp.cuda.Device()
    sms = dev.attributes["MultiProcessorCount"]
    per_sm = GF_TILE_BLOCKS_PER_SM
    try:
        from cupy_backends.cuda.api import driver

        resident = driver.occupancyMaxActiveBlocksPerMultiprocessor(
            fn.kernel.ptr, GF_BLOCK, 0)
        # Never launch more blocks than the card can hold resident: those
        # columns would wait while their workspace slots stayed allocated.
        per_sm = min(per_sm, int(resident))
    except Exception:                              # noqa: BLE001
        pass                                       # keep the measured value
    per_sm = max(1, int(per_sm))
    tile = sms * per_sm * GF_BLOCK
    return int(min(int(ncol), tile))


class GrellFreitas:
    """Stateless ``cu_physics=3`` callable for :class:`PhysicsDriver`.

    The driver remains the sole cadence authority.  Each due call returns
    fresh rates (Task-1 contract: no ``nca_seconds``) and the RAINCV
    increment the kernel computed as ``pratec * dt``.
    """

    def __init__(self, *, column_chunk: int | None = None,
                 updraft_only_when_downdraft_dry: bool = False,
                 resolved_convergence_closure: bool = False,
                 resolved_convergence_floor_percent: int = 100):
        self._driver = None
        #: Deep convection continues without a downdraft where WRF's GFDRV
        #: would reject the column for a downdraft that cannot form (the
        #: saturated global column); False is WRF's word for word.
        self.updraft_only_when_downdraft_dry = bool(updraft_only_when_downdraft_dry)
        #: The coarse-column form of the deep arm (gf.cu
        #: GF_RESOLVED_CONVERGENCE_CLOSURE, a declared divergence from WRF
        #: for columns that carry deep convection as resolved forcing):
        #: the 300.01 K/day heating cap yields to the latent heat of the
        #: column's resolved moisture convergence, the downdraft sweeps
        #: continue past levels the downdraft reaches with no mass instead
        #: of exiting with ierr 51, and a downdraft that cannot form (ierr
        #: 7) leaves the updraft running.  False is WRF's word for word;
        #: the regional cu_physics=3 seam keeps False (its parity anchor),
        #: Arwen Global's native suite sets it (native_options).
        self.resolved_convergence_closure = bool(resolved_convergence_closure)
        #: Part (4) of that form: the mass flux of a convecting column is
        #: floored at this percent of the Kuo moisture-convergence member
        #: (the column precipitates at least that share of the moisture
        #: the grid converges into it); 100 is the member itself, 0 no
        #: floor.  Read only with resolved_convergence_closure.
        percent = int(resolved_convergence_floor_percent)
        if percent < 0:
            raise ValueError(
                f"resolved_convergence_floor_percent is {percent}: a negative floor "
                "never binds (the kernel takes the floor only where it exceeds the "
                "ensemble's mass flux), so the run would say floor and carry none")
        self.resolved_convergence_floor_percent = percent
        #: Per-column readings of the last call (``gf_ierr_deep``: the deep
        #: routine's exit code, 0 where deep convection ran;
        #: ``gf_downdraft_dry_exit``; and the closure reading, one
        #: ``gf_<name>`` array per CLOSURE_READING_SCALARS and
        #: CLOSURE_READING_INTEGERS entry); empty before the first call.
        self.last_column_diagnostics = {}
        chunk = GF_COLUMN_CHUNK if column_chunk is None else int(column_chunk)
        if chunk < 1:
            raise ValueError(f"column_chunk must be positive, got {chunk}")
        self.column_chunk = chunk

    def bind_driver(self, driver) -> None:
        """Optional driver hook: gives the adapter the held radiation rates.

        Mirrors the ``update_trigger_history`` optional-hook precedent: the
        driver calls it when present; a directly attached adapter without a
        driver simply runs with zero radiative forcing, which is also what
        a radiation-off configuration feeds.
        """
        self._driver = driver

    def release(self) -> int:
        """Drop the driver back-reference so a dropped driver can be freed.

        THE SAME CYCLE New Tiedtke had (0bea9490), with a different
        consequence.  PhysicsDriver holds ``cumulus_callable`` and the
        line above holds the driver back, so ``state.physics = None`` at a
        relocation frees neither and the whole driver graph stays resident
        -- refcounting cannot collect a cycle, and CPython's cyclic
        collector triggers on Python object counts rather than on the
        device bytes hanging off them.

        Unlike New Tiedtke this adapter is STATELESS: it caches no
        pipeline and owns no workspace, so there is nothing here to free
        and the return is 0.  What the cycle was pinning is the driver's
        own state, not the scheme's.  Measured across the receipt corpus,
        Grell-Freitas grew 35.04 MiB per relocation against New Tiedtke's
        61.20 and Kain-Fritsch's 1.28.

        Returns bytes freed BY THIS ADAPTER, which is exactly zero, so
        the relocation receipt's cumulus_workspace_bytes does not claim
        credit for memory the collector reclaims elsewhere.
        """
        self._driver = None
        return 0

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from woof.globe.core.physics import CumulusResult

        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def cols(a, lo, hi):
            # (nz, ny, nx) columns [lo, hi) -> (hi - lo, nz), C-contiguous
            return cp.ascontiguousarray(
                a.reshape(nz, ncol)[:, lo:hi].T, dtype=DTYPE)

        w_mass = DTYPE(0.5) * (state.w[:-1] + state.w[1:])
        if self._driver is not None:
            rthraten = self._driver.rthratenlw + self._driver.rthratensw
        else:
            rthraten = cp.zeros((nz, ny, nx), dtype=DTYPE)

        def held_lane(name):
            """Driver-held auxiliary forcing lane, None (zeros) when absent."""
            lane = (None if self._driver is None
                    else getattr(self._driver, name, None))
            if lane is None:
                return None
            if lane.shape != (nz, ny, nx):
                raise ValueError(
                    f"driver {name} must be [nz, ny, nx]={nz, ny, nx}, "
                    f"got {lane.shape}")
            return lane

        # Level lanes in gf.cu's input order; a None lane packs as zeros.
        lanes = {
            "u": atmosphere["u"],
            "v": atmosphere["v"],
            "w": w_mass,
            "t": atmosphere["temperature"],
            "qv": atmosphere["qv"],
            "p": atmosphere["pressure"],
            "pi": atmosphere["exner"],
            "rho": atmosphere["rho"],
            "dz8w": atmosphere["dz"],
            "p8w": atmosphere["p_interface"][:-1],
            "rthften": held_lane("gf_rthdynten"),
            "rqvften": held_lane("gf_rqvdynten"),
            "rthraten": rthraten,
            "rthblten": held_lane("gf_rthblten"),
            "rqvblten": held_lane("gf_rqvblten"),
        }

        # WRF hands the scheme its model DT (module_cumulus_driver.F:1028)
        # -- the outer clock when the case integrates internal substeps,
        # the same idiom as the KF adapter.
        clock_dt = DTYPE(_model_clock_dt(cfg))
        ht = state.ht.reshape(ncol)
        hfx = fields["hfx"].reshape(ncol)
        qfx = fields["qfx"].reshape(ncol)
        xland = fields["xland"].reshape(ncol)
        kpbl = fields["kpbl"].reshape(ncol)
        dx_column = (None if self._driver is None
                     else getattr(self._driver, "gf_dx_column", None))
        if dx_column is None:
            dx_flat = None
        else:
            if dx_column.shape != (ny, nx):
                raise ValueError(
                    f"driver gf_dx_column must be [ny, nx]={ny, nx}, "
                    f"got {dx_column.shape}")
            dx_flat = cp.asarray(dx_column, dtype=DTYPE).reshape(ncol)
        ishallow = np.int32(int(cfg.ishallow))
        clos_choice = np.int32(int(cfg.clos_choice))

        module = _gf_module(nz, self.updraft_only_when_downdraft_dry,
                            self.resolved_convergence_closure,
                            self.resolved_convergence_floor_percent)
        fn = module.get_function("gf_gfdrv_stage")
        block = GF_BLOCK
        # Columns are packed in chunks of `column_chunk` (GF_COLUMN_CHUNK:
        # the whole-batch blocks were 6.4 GB on the T533 global batch), and
        # within a chunk launched in tiles of `tile`.  The column arrays
        # live in a global workspace sized to the threads in flight, not in
        # the per-thread local frame (which CUDA prices at the card's whole
        # resident-thread capacity).  Every tile is launched with the SAME
        # one-thread-one-column mapping the kernel has always had -- the
        # slices below are contiguous views, so the kernel indexes them
        # exactly as before, and a column's result does not depend on which
        # chunk or tile carried it.
        chunks = gf_column_chunks(ncol, self.column_chunk)
        tile = gf_tile_columns(fn, max(hi - lo for lo, hi in chunks))
        ws = cp.empty(gf_workspace_floats(nz, tile), dtype=DTYPE)
        out_names = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten")
        out = {name: cp.empty((nz, ncol), dtype=DTYPE) for name in out_names}
        raincv = cp.empty((ncol,), dtype=DTYPE)
        ierr_deep = cp.empty((ncol,), dtype=cp.int32)
        downdraft_dry_exit = cp.empty((ncol,), dtype=cp.int32)
        reading_f = {name: cp.empty((ncol,), dtype=DTYPE)
                     for name in CLOSURE_READING_SCALARS}
        reading_i = {name: cp.empty((ncol,), dtype=cp.int32)
                     for name in CLOSURE_READING_INTEGERS}
        for lo, hi in chunks:
            n = hi - lo
            lvin = cp.empty((n, len(_IN_LEV), nz), dtype=DTYPE)
            for j, name in enumerate(_IN_LEV):
                lane = lanes[name]
                if lane is None:
                    lvin[:, j, :] = DTYPE(0.0)
                else:
                    lvin[:, j, :] = cols(lane, lo, hi)
            scin = cp.empty((n, 6), dtype=DTYPE)
            scin[:, 0] = ht[lo:hi]
            scin[:, 1] = hfx[lo:hi]
            scin[:, 2] = qfx[lo:hi]
            scin[:, 3] = xland[lo:hi]
            scin[:, 4] = clock_dt
            if dx_flat is None:
                scin[:, 5] = DTYPE(cfg.dx)
            else:
                scin[:, 5] = dx_flat[lo:hi]
            iin = cp.empty((n, 3), dtype=cp.int32)
            iin[:, 0] = kpbl[lo:hi]
            iin[:, 1] = ishallow
            iin[:, 2] = clos_choice
            lev = cp.zeros((n, len(_OUT_LEV), nz), dtype=DTYPE)
            sca = cp.zeros((n, len(_OUT_SCA)), dtype=DTYPE)
            isc = cp.zeros((n, len(_OUT_ISCA)), dtype=cp.int32)
            for t0 in range(0, n, tile):
                t1 = min(t0 + tile, n)
                fn((((t1 - t0) + block - 1) // block,), (block,),
                   (lvin[t0:t1], scin[t0:t1], iin[t0:t1],
                    lev[t0:t1], sca[t0:t1], isc[t0:t1], ws,
                    np.int32(0),      # SHIPPED default: corrected k22 indexing
                    np.int32(t1 - t0), np.int32(nz)))
            for name in out_names:
                out[name][:, lo:hi] = lev[:, _OUT_LEV.index(name), :].T
            raincv[lo:hi] = sca[:, _OUT_SCA.index("raincv")]
            ierr_deep[lo:hi] = isc[:, _OUT_ISCA.index("ierr_deep")]
            downdraft_dry_exit[lo:hi] = isc[:, _OUT_ISCA.index("downdraft_dry_exit")]
            for name, arr in reading_f.items():
                arr[lo:hi] = sca[:, _OUT_SCA.index(name)]
            for name, arr in reading_i.items():
                arr[lo:hi] = isc[:, _OUT_ISCA.index(name)]
            del lvin, scin, iin, lev, sca, isc
        del ws

        def grid(name):
            # (nz, ncol) -> (nz, ny, nx)
            return cp.ascontiguousarray(out[name].reshape(nz, ny, nx))

        rthcuten = grid("rthcuten")
        rqvcuten = grid("rqvcuten")
        rqc = grid("rqccuten")
        rqi = grid("rqicuten")
        if getattr(state, "qi", None) is None:
            # WRF's F_QI-false arm: the 258 K split routes ONE number to one
            # of two slots, so folding is exact addition against a zero.
            rqc = cp.ascontiguousarray(rqc + rqi)
            rqi = None
        rainc = cp.ascontiguousarray(raincv.reshape(ny, nx))
        # The deep routine's exit code per column (WRF's ierr; 0 where deep
        # convection ran) and the downdraft exit the updraft-only switch
        # overrode there (0 none, else 7 or 51), held for a caller that asks
        # why a column stayed dry or how many the switch rescued.  Not part
        # of the result contract: the driver's rates and rain are unchanged
        # by reading them.
        self.last_column_diagnostics = {
            "gf_ierr_deep": cp.ascontiguousarray(ierr_deep.reshape(ny, nx)),
            "gf_downdraft_dry_exit": cp.ascontiguousarray(
                downdraft_dry_exit.reshape(ny, nx)),
        }
        for name, arr in reading_f.items():
            self.last_column_diagnostics[f"gf_{name}"] = cp.ascontiguousarray(
                arr.reshape(ny, nx))
        for name, arr in reading_i.items():
            self.last_column_diagnostics[f"gf_{name}"] = cp.ascontiguousarray(
                arr.reshape(ny, nx))

        return CumulusResult(
            rthcuten=rthcuten, rqvcuten=rqvcuten, rqccuten=rqc,
            rqicuten=rqi, rainc=rainc)


__all__ = ["GrellFreitas", "CLOSURE_READING_SCALARS",
           "CLOSURE_READING_INTEGERS", "CLOSURE_FAMILIES"]
