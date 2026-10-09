"""Per-operator total-energy ledger: an instrument, never a source.

Around every operator of ``MoistHybridModel.step`` the observer marks the
atmosphere and the ledger books the PER-LEVEL NET CHANGE of the column
energy that operator made.  Per level ``k`` (area-weighted global means,
``grid.global_mean`` quadrature, column integrals ``f dp_k / g``):

* ``kinetic``   = 0.5 (u^2 + v^2) dp_k / g
* ``internal``  = cv T dp_k / g
* ``potential`` = R T dp_k / g, the level's share of the gravitational
  potential energy: ``int Phi dp / g = Phi_s ps / g + int R T dp / g`` by
  parts (budgets.py), so the level potential is booked as ``R T dp / g``
  and the surface term ``Phi_s ps / g`` as its own row (index ``nlev``);
* ``total``     = kinetic + internal + potential = (cp T + K) dp_k / g on
  a level and ``Phi_s ps / g`` on the surface row.

Summed over the rows, ``total`` is the hydrostatic primitive-equation
invariant ``int (cp T + K) dp / g + Phi_s ps / g`` (Kasahara 1974), the
same quantity budgets.py reports as ``total_dry_energy_j_m2``.

What the instrument is for, and what it must not do.  The Level 7 review
(2026-09-01) measured what a ledger of the POINTWISE-RECTIFIED column
kinetic-energy difference does: it admits ``max(delta, 0)`` per column,
so a reversible exchange (the semi-implicit map converting kinetic energy
to enthalpy in half the columns and back in the other half) is booked as
loss in every column where it went one way, and on the shipped smoke the
"numerical loss" over-credited the dycore's actual removal about seven
times (165x for the semi-implicit operator, 6.7x for diffusion).  This
ledger books the signed, per-level, area-integrated NET of total energy,
so a reversible operator nets ~0 over a period and a dissipative one
nets its analytic decay (tests/test_arwen_global_insitu_energy.py holds
both), and nothing here feeds a tendency: the marks are read-only and a
run with the ledger on is bit-identical to a run with it off.

Cost.  One mark synthesizes theta, vorticity, divergence and ln ps (four
spectral syntheses, no water species: the surface potential needs ps
only and the dry column energy needs T, not Tv) and reduces on the
device; the vector crosses to the host with the batched flush.  Measured
on the T255 40-level native suite (dt = 100 s, 60 steps, RTX 5070 Ti,
2026-09-02, two repeats each): marks at every step cost 0.093 s of a
1.795 s IMEX step (1.7935 / 1.7952 s against 1.7019 / 1.7013 s without
them, 5.4%) and 0.097 s of a 1.739 s split step (eleven marks:
1.7424 / 1.7346 s against 1.6288 / 1.6549 s, 5.9%), so the ledger samples
every ``energy_every`` steps (InsituOptions, default 50: 0.11% of the
run's steps).  The T533 per-step cost is not measured on a card: the
T533 leg allocated 28.7 GiB beside two other jobs on the 32 GiB card and
died in its first sampled step, and the 16 GiB card holds no T533 run
(its T255 native run alone holds 11.2 GiB through the steps, 12.6 GiB at
the close).

Spectral bands and hemispheres (2026-09-02).  Beside the per-level rows
every mark books the column kinetic energy of the wind BAND-FILTERED in
total spherical-harmonic degree (``BAND_EDGES``: n 1-20, 21-60, 61-120,
121-200, 201-truncation; ``spectral_bands`` clips the edges to the
truncation and drops the bands it does not reach) and split by
hemisphere, so an operator's kinetic-energy traffic is read per band and
per hemisphere, not only per level.  Definitions:

* ``column_kinetic_j_m2[band][hemisphere]`` = the hemisphere's share of
  the sphere-mean ``int 0.5 |v_b|^2 dp / g``, where ``v_b`` is the wind
  synthesised from the vorticity and divergence coefficients of the
  band's degrees alone (``VorticityDivergenceOperator.wind_from_vordiv``
  on the masked coefficients; every other degree is zero).  The
  hemisphere share is the same Gaussian quadrature as the global mean
  with the other hemisphere's rows weighted zero, so northern + southern
  is the sphere mean.
* ``level_kinetic_m2_s2[band][hemisphere]`` = the same for
  ``0.5 |v_b|^2`` on the model level whose full pressure on the
  reference column (``ps = LEVEL_BAND_REFERENCE_PS_PA``) lies nearest
  ``LEVEL_BAND_TARGET_PA`` (250 hPa, the level the spectrum instrument
  reads), without mass weighting.
* the ``total`` entry of each is the unfiltered wind's value, so
  ``total - sum(bands)`` is the cross-band term (zero over the sphere
  for a uniform ``dp`` by orthogonality; not zero over one hemisphere or
  under a varying ``dp``) and the reader can show the closure.
* ``level_spectral_m2_s2[band]`` = ``[rotational, divergent]`` sphere-mean
  kinetic energy per unit mass on the same reference level, read from the
  coefficients alone by Parseval (``insitu.spectra``: ``a^2 / (2 n (n+1))
  sum_m w_m |c_nm|^2 / (4 pi)`` per degree, summed over the band's
  degrees), no synthesis; on the dealiased grid ``rotational + divergent``
  of a band equals the band's ``level_kinetic_m2_s2`` northern + southern
  to roundoff (the cross term integrates to zero over the sphere), which
  the calibration holds as the cross-check of the two code paths.  This
  is the entry that says WHICH part of a band an operator moves: the
  balanced (rotational) cascade or the gravity-wave (divergent) part.
* ``level_spectral_by_degree_m2_s2`` = the same Parseval reading on the
  reference level at EVERY total degree 0..truncation, ``rotational`` and
  ``divergent`` vectors (2026-09-04): the band partition says which band
  an operator moves, this says which DEGREE inside it, so a shelf that
  sits on the last fifteen degrees of the truncation band is attributed
  to the operator that builds it and not to the band's mean.  Same
  arithmetic as the band entry (the band entry is this vector summed
  over the band's degrees; the calibration holds the two equal), no
  synthesis, 2 (truncation + 1) more floats per mark.

An operator that injects a known kinetic energy at one degree is read in
that degree's band and in no other, over the sphere to 1e-9 relative and
exactly zero elsewhere (the other bands' filtered winds are identically
zero); a pure rotation of the state about the polar axis and its mirror
in the equator read zero in every band (tests/test_arwen_global_insitu_
energy_bands.py holds both families in both directions).  Cost: one
band-filtered wind synthesis per band per mark, paid only on sampled
steps; unsampled steps pay nothing (the marks are the no-op they were).
"""
from __future__ import annotations

import time

import numpy as np

from ..constants import DRY_AIR_CP, DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2
from .budgets import global_mean_device
from .spectra import SpectralKineticEnergy

ENERGY_COMPONENTS = ("kinetic", "internal", "potential", "total")
#: Total-degree bands of the kinetic-energy booking; ``None`` closes the
#: last band at the truncation.  Degree 0 carries no kinetic energy and
#: degree 1 (the solid-body component of the wind) opens the first band
#: so the bands partition every degree the wind holds.
BAND_EDGES = ((1, 20), (21, 60), (61, 120), (121, 200), (201, None))
HEMISPHERES = ("northern", "southern")
#: The level booking reads the model level whose full pressure on the
#: reference column lies nearest this (the spectrum instrument's level).
LEVEL_BAND_TARGET_PA = 25_000.0
LEVEL_BAND_REFERENCE_PS_PA = 100_000.0


def spectral_bands(truncation: int) -> list[tuple[int, int]]:
    """``BAND_EDGES`` clipped to ``truncation``; bands it does not reach
    are dropped, so every band returned holds at least one degree."""
    t = int(truncation)
    if t < 1:
        raise ValueError("truncation must be at least 1 for a kinetic-energy band")
    bands: list[tuple[int, int]] = []
    for lo, hi in BAND_EDGES:
        top = t if hi is None else min(int(hi), t)
        if int(lo) > top:
            continue
        bands.append((int(lo), top))
    return bands


def band_label(lo: int, hi: int) -> str:
    return f"n{int(lo):03d}-{int(hi):03d}"


def reference_level_index(vertical, target_pa: float = LEVEL_BAND_TARGET_PA,
                          reference_ps_pa: float = LEVEL_BAND_REFERENCE_PS_PA) -> int:
    """Index of the model level whose full pressure on the reference
    column lies nearest ``target_pa``."""
    a = np.asarray(vertical.a_half_pa, dtype=np.float64)
    b = np.asarray(vertical.b_half, dtype=np.float64)
    p_half = a + b * float(reference_ps_pa)
    p_full = np.sqrt(p_half[:-1] * p_half[1:])
    return int(np.argmin(np.abs(p_full - float(target_pa))))
#: The operator names step() marks, in the order they act; the dynamics
#: stage contributes the split's three marks or the IMEX integrator's two
#: (the explicit tendency sum, then the implicit operator sum; imex.py).
START_MARK = "start"
STEP_OPERATORS = (
    "physics_first",
    "positivity_first",
    "semi_implicit_pre",
    "explicit_dynamics",
    "semi_implicit_post",
    "dynamics_explicit",
    "dynamics_implicit",
    # The semi-Lagrangian core's two: everything the trajectory, the
    # interpolation and the explicit part of the step did, then the one
    # arrival-point Helmholtz solve and the tracer mass fixer.
    "semilag_transport",
    "semilag_implicit",
    "diffusion",
    "mass_fixer",
    # The grid tracers' flux-form transport (2026-09-02): condensate moves
    # between columns here, so the latent and virtual-temperature terms of
    # the moist energy move with it.
    "tracer_transport",
    "physics_second",
    "positivity_second",
)


class OperatorEnergyLedger:
    """Device-side per-level energy marks around the operators of one
    step; ``drain`` hands the (name, vector) list to the flush."""

    def __init__(self, model):
        self.model = model
        transform = model.transform
        self.xp = transform.backend.xp
        self.nlev = int(model.nlev)
        self.weights = self.xp.asarray(
            transform.grid.quadrature_weights, dtype=self.xp.float64
        )
        # Hemisphere shares of the same quadrature: the other hemisphere's
        # rows weigh zero (an equator row, if the grid had one, is split).
        sin_lat = np.asarray(transform.grid.sin_lat, dtype=np.float64)
        weights = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
        north = np.where(sin_lat > 0.0, 1.0, np.where(sin_lat == 0.0, 0.5, 0.0))
        self.hemisphere_weights = {
            "northern": self.xp.asarray(weights * north, dtype=self.xp.float64),
            "southern": self.xp.asarray(weights * (1.0 - north), dtype=self.xp.float64),
        }
        self.truncation = int(transform.truncation)
        self.bands = spectral_bands(self.truncation)
        self.band_labels = [band_label(lo, hi) for lo, hi in self.bands]
        masks = np.zeros((len(self.bands), self.truncation + 1), dtype=np.float64)
        for index, (lo, hi) in enumerate(self.bands):
            masks[index, lo:hi + 1] = 1.0
        self.band_masks = self.xp.asarray(masks, dtype=transform.backend.float_dtype)
        self.level_index = reference_level_index(model.vertical)
        # Parseval on the reference level: KE per unit mass by degree is
        # a^2 / (2 n (n+1)) sum_m w_m |c_nm|^2 / (4 pi) (insitu.spectra).
        self.spectral = SpectralKineticEnergy(transform)
        a = np.asarray(model.vertical.a_half_pa, dtype=np.float64)
        b = np.asarray(model.vertical.b_half, dtype=np.float64)
        p_half = a + b * LEVEL_BAND_REFERENCE_PS_PA
        self.level_pressure_pa = float(np.sqrt(p_half[:-1] * p_half[1:])[self.level_index])
        self.active = False
        self.marks: list[tuple[str, object]] = []
        self.wall_s = 0.0

    @property
    def rows(self) -> int:
        return self.nlev + 1

    @property
    def base_width(self) -> int:
        return self.rows * len(ENERGY_COMPONENTS)

    @property
    def band_width(self) -> int:
        """The band block: (total + one per band) x (column, level) x
        (northern, southern), then (total + one per band) x (rotational,
        divergent) on the reference level by Parseval, then the level's
        rotational and divergent Parseval vectors by total degree
        (2 x (truncation + 1))."""
        return 6 * (len(self.bands) + 1) + 2 * (self.truncation + 1)

    @property
    def width(self) -> int:
        return self.base_width + self.band_width

    def describe_bands(self) -> dict[str, object]:
        return {
            "edges": [[int(lo), int(hi)] for lo, hi in self.bands],
            "labels": list(self.band_labels),
            "hemispheres": list(HEMISPHERES),
            "level_index": int(self.level_index),
            "level_reference_pressure_pa": float(self.level_pressure_pa),
            "column_units": "J/m2, hemisphere share of the sphere mean of int 0.5 |v_band|^2 dp/g",
            "level_units": "m2/s2, hemisphere share of the sphere mean of 0.5 |v_band|^2 on the level",
            "level_spectral_units": "m2/s2, sphere mean of 0.5 |v_rot|^2 and 0.5 |v_div|^2 on the level by Parseval",
            "level_spectral_by_degree_units": "m2/s2, the same Parseval reading per total degree 0..truncation, [rotational, divergent] vectors",
            "truncation": int(self.truncation),
        }

    def _hemisphere_shares(self, field):
        """``[northern, southern]`` shares of the sphere mean of a
        ``(nlat, nlon)`` field, one device vector of two."""
        xp = self.xp
        zonal = xp.mean(field, axis=-1, dtype=xp.float64)
        return xp.stack([
            0.5 * xp.sum(zonal * self.hemisphere_weights[name], dtype=xp.float64)
            for name in HEMISPHERES
        ])

    def _band_block(self, atmosphere, u, v, dp_g):
        """The band block of one mark: for the unfiltered wind and for
        every band-filtered wind, the column (mass-weighted) and the
        reference-level kinetic energy by hemisphere."""
        xp = self.xp
        k = self.level_index

        def entries(u_field, v_field):
            half_speed2 = 0.5 * (
                u_field.astype(xp.float64) ** 2 + v_field.astype(xp.float64) ** 2
            )
            column = self._hemisphere_shares(xp.sum(half_speed2 * dp_g, axis=0))
            level = self._hemisphere_shares(half_speed2[k])
            return column, level

        pieces = list(entries(u, v))
        for index in range(len(self.bands)):
            mask = self.band_masks[index][:, None]
            u_band, v_band = self.model.vector.wind_from_vordiv(
                atmosphere.vorticity * mask, atmosphere.divergence * mask
            )
            pieces.extend(entries(u_band, v_band))
            del u_band, v_band
        # Rotational / divergent on the reference level by Parseval: one
        # (truncation + 1) vector each, banded by the same masks.
        rot = self.spectral.by_degree(atmosphere.vorticity[k])
        div = self.spectral.by_degree(atmosphere.divergence[k])
        masks = self.band_masks.astype(xp.float64)
        pieces.append(xp.stack([xp.sum(rot), xp.sum(div)]))
        pieces.append(xp.stack([masks @ rot, masks @ div], axis=1).reshape(-1))
        # The same reading at every degree: the band entry is this vector
        # summed over the band's degrees.
        pieces.append(rot)
        pieces.append(div)
        return xp.concatenate(pieces)

    def _metal_measure(self, atmosphere, grid, context):
        """Compute the complete operator-energy vector on the resident context."""
        from ...metal.array import MetalArray, MetalArrayModule
        from ...metal.generated import kernel_specs

        metal = MetalArrayModule(context)
        nlat, nlon = self.model.transform.grid.shape
        nlevel = self.nlev
        ncell = nlat * nlon
        truncation = self.truncation
        norder = truncation + 1
        chunk_size = 1024
        nchunk = max(1, (ncell + chunk_size - 1) // chunk_size)

        def flat(value, label, dtype=np.float64):
            if isinstance(value, MetalArray):
                if value.buffer.context is not context:
                    raise ValueError(f"Global energy {label} belongs to another Metal context")
                if value.dtype != np.dtype(dtype):
                    raise TypeError(f"Global energy {label} must have dtype {np.dtype(dtype)}")
                source = value
            else:
                host = np.ascontiguousarray(np.asarray(value, dtype=dtype))
                source = metal.asarray(host)
            return source.view((source.size,), dtype=dtype)

        row = metal.empty((self.width,), dtype=np.float64)
        base_columns = metal.empty((self.base_width * ncell,), dtype=np.float64)
        base_partial = metal.empty((self.base_width * nchunk,), dtype=np.float64)
        base_resources = {
            "temperature": flat(grid["temperature"], "temperature"),
            "u": flat(grid["u"], "u"),
            "v": flat(grid["v"], "v"),
            "dp": flat(grid["dp"], "dp"),
            "ps": flat(grid["ps"], "ps"),
            "surface_geopotential": flat(self.model.surface_geopotential, "surface geopotential"),
            "weights": flat(self.weights, "quadrature weights"),
            "columns": base_columns,
        }
        context.submit_kernel(
            "global_insitu_energy_columns_fragment", base_resources,
            {"nlevel": nlevel, "nlat": nlat, "nlon": nlon},
            exact_policy=False, grid_size=ncell,
            threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_COLUMNS_FRAGMENT_THREADS_PER_GROUP,
        )
        context.submit_kernel(
            "global_insitu_energy_reduce_fragment",
            {"columns": base_columns, "partial": base_partial},
            {"nterm": self.base_width, "ncell": ncell, "nchunk": nchunk,
             "chunk_size": chunk_size},
            exact_policy=False, grid_size=self.base_width * nchunk,
            threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_REDUCE_FRAGMENT_THREADS_PER_GROUP,
        )
        context.submit_kernel(
            "global_insitu_energy_finalize_fragment",
            {"partial": base_partial, "row": row},
            {"nterm": self.base_width, "nchunk": nchunk, "row_offset": 0},
            exact_policy=False, grid_size=self.base_width,
            threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_FINALIZE_FRAGMENT_THREADS_PER_GROUP,
        )

        dp = flat(grid["dp"], "dp")
        weights = flat(self.weights, "quadrature weights")
        north = flat(self.hemisphere_weights["northern"], "northern weights")
        south = flat(self.hemisphere_weights["southern"], "southern weights")
        band_columns = metal.empty((4 * ncell,), dtype=np.float64)
        band_partial = metal.empty((4 * nchunk,), dtype=np.float64)

        def append_band_wind(component, u_field, v_field):
            context.submit_kernel(
                "global_insitu_energy_band_columns_fragment",
                {"u": flat(u_field, "band u"), "v": flat(v_field, "band v"),
                 "dp": dp, "weights": weights, "north": north, "south": south,
                 "columns": band_columns},
                {"nlevel": nlevel, "nlat": nlat, "nlon": nlon,
                 "level_index": self.level_index},
                exact_policy=False, grid_size=ncell,
                threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_BAND_COLUMNS_FRAGMENT_THREADS_PER_GROUP,
            )
            context.submit_kernel(
                "global_insitu_energy_reduce_fragment",
                {"columns": band_columns, "partial": band_partial},
                {"nterm": 4, "ncell": ncell, "nchunk": nchunk,
                 "chunk_size": chunk_size},
                exact_policy=False, grid_size=4 * nchunk,
                threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_REDUCE_FRAGMENT_THREADS_PER_GROUP,
            )
            context.submit_kernel(
                "global_insitu_energy_finalize_fragment",
                {"partial": band_partial, "row": row},
                {"nterm": 4, "nchunk": nchunk,
                 "row_offset": self.base_width + 4 * component},
                exact_policy=False, grid_size=4,
                threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_FINALIZE_FRAGMENT_THREADS_PER_GROUP,
            )

        append_band_wind(0, grid["u"], grid["v"])
        vorticity = atmosphere.vorticity
        divergence = atmosphere.divergence
        if (not isinstance(vorticity, MetalArray)
                or not isinstance(divergence, MetalArray)
                or vorticity.dtype != np.dtype(np.complex128)
                or divergence.dtype != np.dtype(np.complex128)
                or vorticity.buffer.context is not context
                or divergence.buffer.context is not context
                or vorticity.shape != divergence.shape):
            def describe(value):
                if isinstance(value, MetalArray):
                    return (f"MetalArray(dtype={value.dtype}, shape={value.shape}, "
                            f"same_context={value.buffer.context is context})")
                return type(value).__name__
            raise TypeError(
                "Global energy bands require resident complex128 "
                f"vorticity/divergence; got {describe(vorticity)} and "
                f"{describe(divergence)}"
            )
        nlevel_coeff, nterm, norder_coeff = map(int, vorticity.shape)
        if (nlevel_coeff != nlevel or nterm != truncation + 1
                or norder_coeff < nterm):
            raise ValueError("Global energy coefficient shape does not match the transform")
        masks = np.ascontiguousarray(np.asarray(self.band_masks, dtype=np.float64))
        for index in range(len(self.bands)):
            filtered_z = metal.empty((2 * vorticity.size,), dtype=np.float64)
            filtered_d = metal.empty((2 * divergence.size,), dtype=np.float64)
            mask = metal.asarray(np.ascontiguousarray(masks[index]))
            context.submit_kernel(
                "global_insitu_energy_mask_modes_fragment",
                {"vorticity": flat(vorticity, "vorticity", np.complex128).view(
                     (2 * vorticity.size,), dtype=np.float64),
                 "divergence": flat(divergence, "divergence", np.complex128).view(
                     (2 * divergence.size,), dtype=np.float64),
                 "mask": mask, "filtered_vorticity": filtered_z,
                 "filtered_divergence": filtered_d},
                {"nlevel": nlevel, "nterm": nterm, "norder": norder_coeff},
                exact_policy=False, grid_size=vorticity.size,
                threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_MASK_MODES_FRAGMENT_THREADS_PER_GROUP,
            )
            z_band = filtered_z.view(vorticity.shape, dtype=np.complex128)
            d_band = filtered_d.view(divergence.shape, dtype=np.complex128)
            u_band, v_band = self.model.vector.wind_from_vordiv(z_band, d_band)
            append_band_wind(index + 1, u_band, v_band)

        spectral_offset = self.base_width + 4 * (len(self.bands) + 1)
        context.submit_kernel(
            "global_insitu_energy_parseval_fragment",
            {"vorticity": flat(vorticity, "vorticity", np.complex128).view(
                 (2 * vorticity.size,), dtype=np.float64),
             "divergence": flat(divergence, "divergence", np.complex128).view(
                 (2 * divergence.size,), dtype=np.float64),
             "factor": flat(self.spectral._factor, "spectral factors"),
             "order_weight": flat(self.spectral._order_weight, "spectral order weights"),
             "masks": metal.asarray(masks.reshape(-1)), "row": row},
            {"nlevel": nlevel, "nterm": nterm, "norder": norder_coeff,
             "level_index": self.level_index, "nband": len(self.bands),
             "row_offset": spectral_offset},
            exact_policy=False,
            grid_size=2 * (len(self.bands) + 1) + 2 * nterm,
            threads_per_group=kernel_specs.GLOBAL_INSITU_ENERGY_PARSEVAL_FRAGMENT_THREADS_PER_GROUP,
        )
        return row

    def measure(self, atmosphere):
        """One device vector of ``rows * 4`` float64 entries: the per-level
        (kinetic, internal, potential, total) column energies and the
        surface row."""
        xp = self.xp
        g = self.model.grid_state(
            atmosphere, only=("temperature", "u", "v", "dp", "ps")
        )
        context = (getattr(self.model, "metal_diffusion_context", None)
                   or getattr(self.model.transform, "metal_context", None))
        if context is not None:
            return self._metal_measure(atmosphere, g, context)
        dp_g = g["dp"] / GRAVITY_M_S2
        kinetic = global_mean_device(
            xp, 0.5 * (g["u"] * g["u"] + g["v"] * g["v"]) * dp_g, self.weights
        )
        temperature_mass = global_mean_device(xp, g["temperature"] * dp_g, self.weights)
        internal = (DRY_AIR_CP - DRY_AIR_GAS_CONSTANT) * temperature_mass
        potential = DRY_AIR_GAS_CONSTANT * temperature_mass
        surface = global_mean_device(
            xp,
            self.model.surface_geopotential.astype(xp.float64) * g["ps"] / GRAVITY_M_S2,
            self.weights,
        )
        zero = xp.zeros((1,), dtype=xp.float64)
        columns = xp.stack([
            xp.concatenate([kinetic, zero]),
            xp.concatenate([internal, zero]),
            xp.concatenate([potential, surface[None]]),
            xp.concatenate([kinetic + internal + potential, surface[None]]),
        ], axis=1)
        bands = self._band_block(atmosphere, g["u"], g["v"], dp_g)
        return xp.concatenate([columns.reshape(-1), bands])

    def begin_step(self, atmosphere) -> None:
        self.marks = []
        if self.active:
            self.mark(START_MARK, atmosphere)

    def mark(self, name: str, atmosphere) -> None:
        if not self.active:
            return
        started = time.perf_counter()
        self.marks.append((name, self.measure(atmosphere)))
        self.wall_s += time.perf_counter() - started

    def drain(self) -> list[tuple[str, object]]:
        marks = self.marks
        self.marks = []
        return marks

    def _band_entry(self, block: np.ndarray) -> dict[str, object]:
        """One band block (``band_width`` values) as ``{column_kinetic_j_m2,
        level_kinetic_m2_s2}``, each ``{"total": [N, S], "bands": [[N, S],
        ...]}`` in band order, and ``level_spectral_m2_s2`` the same shape
        with ``[rotational, divergent]`` pairs."""
        count = len(self.bands) + 1
        block = np.asarray(block, dtype=np.float64)
        hemispheric = block[:4 * count].reshape(count, 2, 2)
        spectral = block[4 * count:6 * count].reshape(count, 2)
        t1 = self.truncation + 1
        by_degree = block[6 * count:6 * count + 2 * t1].reshape(2, t1)
        return {
            "column_kinetic_j_m2": {
                "total": [float(v) for v in hemispheric[0, 0]],
                "bands": [[float(v) for v in hemispheric[i + 1, 0]] for i in range(len(self.bands))],
            },
            "level_kinetic_m2_s2": {
                "total": [float(v) for v in hemispheric[0, 1]],
                "bands": [[float(v) for v in hemispheric[i + 1, 1]] for i in range(len(self.bands))],
            },
            "level_spectral_m2_s2": {
                "total": [float(v) for v in spectral[0]],
                "bands": [[float(v) for v in spectral[i + 1]] for i in range(len(self.bands))],
            },
            "level_spectral_by_degree_m2_s2": {
                "rotational": [float(v) for v in by_degree[0]],
                "divergent": [float(v) for v in by_degree[1]],
            },
        }

    def unpack(self, names: list[str], host: np.ndarray) -> dict[str, object]:
        """Host-side: the per-operator per-level NET of every component
        from the consecutive marks, plus the step's opening measure, and
        the per-band per-hemisphere kinetic nets (``band_net``) beside
        the opening band measure (``band_start``)."""
        rows = self.rows
        width = len(ENERGY_COMPONENTS)
        vectors = np.asarray(host, dtype=np.float64).reshape(len(names), self.width)
        measures = vectors[:, :self.base_width].reshape(len(names), rows, width)
        blocks = vectors[:, self.base_width:]
        net: dict[str, list[list[float]]] = {}
        band_net: dict[str, dict[str, object]] = {}
        for index in range(1, len(names)):
            delta = measures[index] - measures[index - 1]
            net[names[index]] = [[float(v) for v in row] for row in delta]
            band_net[names[index]] = self._band_entry(blocks[index] - blocks[index - 1])
        return {
            "components": list(ENERGY_COMPONENTS),
            "rows": rows,
            "operators": list(names[1:]),
            "start": [[float(v) for v in row] for row in measures[0]],
            "net": net,
            "column_total_net": {
                name: float(np.sum(measures[index, :, -1] - measures[index - 1, :, -1]))
                for index, name in enumerate(names) if index > 0
            },
            "bands": self.describe_bands(),
            "band_start": self._band_entry(blocks[0]),
            "band_net": band_net,
        }


__all__ = [
    "BAND_EDGES",
    "ENERGY_COMPONENTS",
    "HEMISPHERES",
    "LEVEL_BAND_TARGET_PA",
    "START_MARK",
    "STEP_OPERATORS",
    "OperatorEnergyLedger",
    "band_label",
    "reference_level_index",
    "spectral_bands",
]
