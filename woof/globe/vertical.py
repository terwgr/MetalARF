"""Hybrid A/B vertical coordinate and pressure-coordinate continuity."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constants import DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2


def _libm_pow(base, exponent) -> np.ndarray:
    """``base ** exponent`` element by element through the C library's pow.

    The coefficients built here feed the configuration identity, which
    checkpoints and receipts hash.  numpy 2.5's AVX-512 power and exp loops
    round differently in the last bit from the C library, so the same
    configuration hashed to a different identity on an AVX-512 Linux
    machine than on Windows or an older CPU.  Scalar pow and exp are the
    values every recorded identity was measured with.
    """
    b = np.broadcast_to(np.asarray(base, dtype=np.float64), np.broadcast_shapes(
        np.shape(base), np.shape(exponent)))
    e = np.broadcast_to(np.asarray(exponent, dtype=np.float64), b.shape)
    return np.array([math.pow(float(x), float(y)) for x, y in zip(b.ravel(), e.ravel())],
                    dtype=np.float64).reshape(b.shape)


def _libm_exp(values) -> np.ndarray:
    """``exp`` element by element through the C library, for the reason ``_libm_pow`` gives."""
    v = np.asarray(values, dtype=np.float64)
    return np.array([math.exp(float(x)) for x in v.ravel()], dtype=np.float64).reshape(v.shape)

#: Reference surface pressure the surface_stretched layout is designed at.
SURFACE_STRETCHED_REFERENCE_PS_PA = 101_325.0
#: Bottom layer thickness at the reference surface: 550 Pa puts the first
#: full level (geometric-mean pressure) 23 m AGL in the standard atmosphere.
SURFACE_STRETCHED_BOTTOM_LAYER_PA = 550.0
#: Fraction of the layers that stretch geometrically from the bottom; the
#: rest are uniform in ln p at the last stretched thickness.
SURFACE_STRETCHED_FRACTION = 0.7
#: B is exactly zero at and above this pressure: pure pressure levels.
SURFACE_STRETCHED_PURE_PRESSURE_ABOVE_PA = 10_000.0
#: Exponent of the hybrid weight B = x**e below the pure-pressure top; 1 is
#: sigma-like, larger detaches levels from terrain sooner.  1.2 keeps the
#: bottom layer 180 Pa thick over a 500 hPa surface and leaves B = 0.37
#: at 500 hPa.
SURFACE_STRETCHED_HYBRID_EXPONENT = 1.2
#: Largest adjacent-layer growth ratio the constructor accepts (audit VTW-6,
#: re-measured after DN-3: 64 m rms model-top error at ln ratio 1.94).
SURFACE_STRETCHED_MAXIMUM_GROWTH_RATIO = 2.0

#: The jet-refined layout (``HybridCoordinate.jet_refined``) is the
#: surface_stretched stack at this many levels with its layers between the
#: band top and the taper bottom re-laid; everything outside that span is
#: the base stack's own half levels, bit for bit.
JET_REFINED_BASE_NLEV = 40
#: The band is bounded above by the base half level nearest this pressure
#: (118.73 hPa on the 40-level stack) and below by a new half level at
#: exactly ``JET_REFINED_BAND_BOTTOM_PA``; inside it every layer is the
#: same thickness in pressure.
JET_REFINED_BAND_TOP_PA = 12_000.0
JET_REFINED_BAND_BOTTOM_PA = 40_000.0
#: Below the band the thickness grows geometrically over
#: ``JET_REFINED_TAPER_LAYERS`` layers to meet the base stack at its half
#: level nearest this pressure (504.19 hPa on the 40-level stack), so the
#: 23 hPa band layers never sit against a 60 hPa layer.
JET_REFINED_TAPER_BOTTOM_PA = 50_000.0
JET_REFINED_TAPER_LAYERS = 3
#: The thickest layer the band may hold: the level count a config names is
#: refused below the count that meets it, because a band that is coarser
#: than the reference analysis's own 25 hPa level spacing at 250 hPa
#: cannot carry the analysis's jet (the 40-level stack's 55 hPa layers
#: cost 1.85 m/s rmsve and -0.53 m/s of speed at 250 hPa before any
#: forecast, the representation floor of 2026-09-04).
JET_REFINED_BAND_MAXIMUM_LAYER_PA = 2_500.0
#: The pressures the scorecard's jet reading is taken between; describe()
#: reports the thickest layer of any stack in this span.
JET_BAND_REPORT_PA = (12_000.0, 40_000.0)

# ICAO standard atmosphere: (base height m, base temperature K, lapse K/m,
# base pressure Pa) from the surface to 71 km; above the last base the
# last layer's law continues.
_STANDARD_ATMOSPHERE_LAYERS = (
    (0.0, 288.15, -0.0065, 101_325.0),
    (11_000.0, 216.65, 0.0, 22_632.06),
    (20_000.0, 216.65, 0.001, 5_474.889),
    (32_000.0, 228.65, 0.0028, 868.0187),
    (47_000.0, 270.65, 0.0, 110.9063),
    (51_000.0, 270.65, -0.0028, 66.93887),
)


def standard_atmosphere_height_m(pressure_pa):
    """ICAO standard-atmosphere geopotential height (m) of a pressure (Pa)."""
    p = np.asarray(pressure_pa, dtype=np.float64)
    out = np.empty_like(p)
    bases = np.array([layer[3] for layer in _STANDARD_ATMOSPHERE_LAYERS])
    # Layer index: the last base whose pressure is >= p (pressure falls
    # with height); pressures above the surface base use the first layer.
    index = np.clip(np.searchsorted(-bases, -p, side="right") - 1, 0, bases.size - 1)
    for k, (z0, t0, lapse, p0) in enumerate(_STANDARD_ATMOSPHERE_LAYERS):
        mask = index == k
        if not np.any(mask):
            continue
        if lapse == 0.0:
            out[mask] = z0 + DRY_AIR_GAS_CONSTANT * t0 / GRAVITY_M_S2 * np.log(p0 / p[mask])
        else:
            exponent = -DRY_AIR_GAS_CONSTANT * lapse / GRAVITY_M_S2
            out[mask] = z0 + (t0 * (p[mask] / p0) ** exponent - t0) / lapse
    return out if out.ndim else float(out)


@dataclass(frozen=True)
class HybridCoordinate:
    """Hydrostatic hybrid pressure coordinate.

    Half-level pressure is ``p_half = A + B * ps``. Arrays are ordered from
    model top to the surface. The bottom identity is fixed at A=0, B=1.
    """

    a_half_pa: np.ndarray
    b_half: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.a_half_pa, dtype=np.float64)
        b = np.asarray(self.b_half, dtype=np.float64)
        if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size < 3:
            raise ValueError("hybrid A/B arrays must be equal 1-D arrays defining >=2 layers")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("hybrid A/B arrays contain non-finite values")
        if np.any(a < 0.0) or np.any((b < 0.0) | (b > 1.0)):
            raise ValueError("hybrid A must be nonnegative and B must lie in [0,1]")
        if abs(float(a[-1])) > 1.0e-10 or abs(float(b[-1]) - 1.0) > 1.0e-12:
            raise ValueError("bottom hybrid half level must satisfy A=0 and B=1")
        for ps in (50_000.0, 80_000.0, 105_000.0, 120_000.0):
            p = a + b * ps
            if np.any(np.diff(p) <= 0.0):
                raise ValueError(
                    f"hybrid pressure is not strictly increasing top-to-bottom at ps={ps:g} Pa"
                )
        object.__setattr__(self, "a_half_pa", a)
        object.__setattr__(self, "b_half", b)

    @property
    def nlev(self) -> int:
        return int(self.a_half_pa.size - 1)

    @property
    def delta_a(self) -> np.ndarray:
        return np.diff(self.a_half_pa)

    @property
    def delta_b(self) -> np.ndarray:
        return np.diff(self.b_half)

    @classmethod
    def pressure_blend(cls, nlev: int, p_top_pa: float = 100.0) -> "HybridCoordinate":
        """The original B = eta**1.7 blend, kept for identity-locked archives.

        At nlev=20 its bottom layer is 83.5 hPa thick and the lowest full
        level sits ~378 m AGL (audit 2026-09-01, DA-4 / task 1a), which is
        why it is no longer the default: every surface scheme assumes a
        lowest level tens of metres up.  Checkpoints and receipts hash the
        A/B arrays, so archives written under this grid stay readable only
        while it stays selectable.
        """
        n = int(nlev)
        if n < 2:
            raise ValueError("hybrid coordinate requires at least two layers")
        if not math.isfinite(p_top_pa) or not 0.0 < p_top_pa < 20_000.0:
            raise ValueError("p_top_pa must lie in (0,20000) Pa")
        # Quadratic B packs levels toward the lower atmosphere. A contributes
        # the top pressure and fades continuously to zero at the surface.
        eta = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
        b = _libm_pow(eta, 1.7)
        a = float(p_top_pa) * (1.0 - b)
        a[-1] = 0.0
        b[-1] = 1.0
        return cls(a, b)

    @classmethod
    def surface_stretched(
        cls,
        nlev: int,
        p_top_pa: float = 100.0,
        *,
        bottom_layer_pa: float = SURFACE_STRETCHED_BOTTOM_LAYER_PA,
        stretched_fraction: float = SURFACE_STRETCHED_FRACTION,
        pure_pressure_above_pa: float = SURFACE_STRETCHED_PURE_PRESSURE_ABOVE_PA,
        hybrid_exponent: float = SURFACE_STRETCHED_HYBRID_EXPONENT,
    ) -> "HybridCoordinate":
        """A surface-stretched hybrid grid with a ~23 m first full level.

        Half-level pressures are laid out at the reference surface
        ``SURFACE_STRETCHED_REFERENCE_PS_PA`` in zeta = ln(ps_ref / p):

        * the bottom ``round(stretched_fraction * nlev)`` layers grow
          geometrically in zeta from a ``bottom_layer_pa``-thick bottom
          layer (first full level 23 m AGL in the standard atmosphere) by a
          ratio ``r`` solved for below;
        * the remaining layers are uniform in zeta (uniform in ln p) at the
          last stretched thickness, so the stratosphere is resolved in
          scale heights rather than in pressure and the top layer's ln
          ratio is 0.37 at nlev=40 (the 20-level pressure_blend top layer
          was 1.96 and carried a measured +267 m geopotential error at the
          model top under the level-only half-layer integration, 34 m mean
          / 64 m rms under the DN-3 form; audit VTW-6).

        Pressure thickness therefore grows from the surface to a maximum in
        the mid troposphere (60 hPa near 390 hPa at nlev=40, where the
        20-level pressure_blend grid had 59 hPa) and thins in ln p toward
        ``p_top_pa``; at nlev=40 eleven full levels sit above 50 hPa.  The hybrid split is
        ``B = ((p - p_bt) / (ps_ref - p_bt)) ** hybrid_exponent`` above the
        surface and ``B = 0`` for ``p <= pure_pressure_above_pa`` (100 hPa),
        so the stratosphere is on pure pressure levels; ``A = p - B ps_ref``
        is nonnegative because ``B <= x <= p / ps_ref`` there.  Layers stay
        strictly increasing at ps down to 500 hPa because the surface
        term ``dB (ps_ref - ps)`` is at most ``hybrid_exponent * 0.562`` of
        the reference thickness.
        """
        n = int(nlev)
        if n < 2:
            raise ValueError("hybrid coordinate requires at least two layers")
        if not math.isfinite(p_top_pa) or not 0.0 < p_top_pa < pure_pressure_above_pa:
            raise ValueError(
                f"p_top_pa must lie in (0,{pure_pressure_above_pa:g}) Pa so the "
                "pure-pressure stratosphere holds at least the top layer"
            )
        ps_ref = SURFACE_STRETCHED_REFERENCE_PS_PA
        if not 0.0 < bottom_layer_pa < 0.1 * ps_ref:
            raise ValueError("bottom_layer_pa must lie in (0, 10% of the reference surface)")
        if not 0.0 < stretched_fraction <= 1.0:
            raise ValueError("stretched_fraction must lie in (0,1]")
        n_stretched = max(2, min(n, int(round(stretched_fraction * n))))
        n_uniform = n - n_stretched
        zeta_bottom = math.log(ps_ref / (ps_ref - float(bottom_layer_pa)))
        zeta_top = math.log(ps_ref / float(p_top_pa))

        def total(r: float) -> float:
            if abs(r - 1.0) < 1.0e-12:
                geometric = float(n_stretched)
            else:
                geometric = (r ** n_stretched - 1.0) / (r - 1.0)
            return zeta_bottom * (geometric + n_uniform * r ** (n_stretched - 1))

        if total(1.0) >= zeta_top:
            raise ValueError(
                f"{n} layers of at least {bottom_layer_pa:g} Pa already reach "
                f"p_top {p_top_pa:g} Pa without stretching; fewer layers or a "
                "thinner bottom layer are needed"
            )
        low, high = 1.0, 8.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            if total(mid) < zeta_top:
                low = mid
            else:
                high = mid
        ratio = 0.5 * (low + high)
        if ratio > SURFACE_STRETCHED_MAXIMUM_GROWTH_RATIO:
            # Refusal (audit VTW-6, re-measured after DN-3): a layer
            # spanning ~2 in ln p cannot hold the real profile's curvature.
            # The level-only half-layer integration carried +267 m mean /
            # 279 m rms model-top height error on the GDAS analysis at a
            # top-layer ln ratio of 1.94; the linear-in-ln-p half layer
            # (half_layer_thickness) still leaves +34 m mean / 64 m rms
            # there, with 54 m rms horizontal structure entering the
            # pressure-gradient force, against 2.7 m rms at the 40-level
            # default's 0.37.  A growth ratio above 2 between adjacent
            # layers puts every upper layer past that ratio.
            raise ValueError(
                f"surface_stretched needs more than {n} layers between a "
                f"{bottom_layer_pa:g} Pa bottom layer and p_top {p_top_pa:g} Pa: "
                f"adjacent layers would grow by {ratio:.2f}x (limit "
                f"{SURFACE_STRETCHED_MAXIMUM_GROWTH_RATIO:g}), past the ln-ratio "
                "at which the hydrostatic step mis-measures the model-top "
                "height by 64 m rms on the real analysis (audit VTW-6, "
                "re-measured after DN-3); select coordinate = "
                "\"pressure_blend\" for a grid this coarse"
            )
        thickness = zeta_bottom * _libm_pow(ratio, np.arange(n_stretched, dtype=np.float64))
        thickness = np.concatenate([
            thickness, np.full(n_uniform, thickness[-1], dtype=np.float64)
        ])
        zeta_half = np.concatenate([[0.0], np.cumsum(thickness)])
        zeta_half *= zeta_top / zeta_half[-1]
        p_ref = ps_ref * _libm_exp(-zeta_half[::-1])
        p_ref[0] = float(p_top_pa)
        p_ref[-1] = ps_ref

        p_bt = float(pure_pressure_above_pa)
        x = np.clip((p_ref - p_bt) / (ps_ref - p_bt), 0.0, 1.0)
        b = _libm_pow(x, float(hybrid_exponent))
        a = p_ref - b * ps_ref
        a = np.maximum(a, 0.0)
        a[-1] = 0.0
        b[-1] = 1.0
        return cls(a, b)

    @classmethod
    def jet_refined(
        cls,
        nlev: int = 48,
        p_top_pa: float = 100.0,
        *,
        base_nlev: int = JET_REFINED_BASE_NLEV,
        band_top_pa: float = JET_REFINED_BAND_TOP_PA,
        band_bottom_pa: float = JET_REFINED_BAND_BOTTOM_PA,
        taper_bottom_pa: float = JET_REFINED_TAPER_BOTTOM_PA,
        taper_layers: int = JET_REFINED_TAPER_LAYERS,
        pure_pressure_above_pa: float = SURFACE_STRETCHED_PURE_PRESSURE_ABOVE_PA,
        hybrid_exponent: float = SURFACE_STRETCHED_HYBRID_EXPONENT,
    ) -> "HybridCoordinate":
        """The surface_stretched stack with the jet band re-laid.

        Built from ``surface_stretched(base_nlev, p_top_pa)`` at the
        reference surface: every half level of the base stack at or
        above the base half level nearest ``band_top_pa`` (the
        stratosphere, 118.73 hPa and up on the 40-level stack) and every
        one at or below the base half level nearest ``taper_bottom_pa``
        (504.19 hPa down to the 550 Pa bottom layer) is kept with its A
        and B bit for bit, so the boundary layer, the surface schemes'
        first level and the pure-pressure stratosphere are the base
        stack's own.  Between them:

        * the BAND, from the kept top down to a new half level at exactly
          ``band_bottom_pa`` (400 hPa), holds ``nlev - base_nlev +
          replaced - taper_layers`` layers of one thickness in pressure
          (``replaced`` is the number of base layers the span held, seven
          on the 40-level stack); at the default 48 levels that is twelve
          layers of 23.44 hPa where the 40-level stack had 43.9 to 59.8
          hPa, with full levels at 129.9, 153.4, 176.9, 200.4, 223.9,
          247.4, 270.8, 294.2, 317.7, 341.2, 364.7 and 388.2 hPa;
        * the TAPER, ``taper_layers`` layers from ``band_bottom_pa`` to
          the kept bottom, grows geometrically from the band thickness
          so the first layer under the band is 1.21x the band layer and
          the last meets the kept 56.1 hPa layer at 1.35x (28.4, 34.3
          and 41.5 hPa at 48 levels), never a 2.5x step.

        The hybrid split of the new half levels is the base stack's own
        law, ``B = ((p - p_bt) / (ps_ref - p_bt)) ** hybrid_exponent``
        with ``A = p - B ps_ref``, so the band stays terrain-following
        to the same degree as the layers it replaces (B 0.01 to 0.19
        there) and the half levels stay strictly increasing at every
        surface pressure the class checks (a 23.44 hPa reference layer
        loses at most 12.4 hPa of thickness over a 500 hPa surface).

        The default count, 48, is the smallest at which the band layers
        meet ``JET_REFINED_BAND_MAXIMUM_LAYER_PA``; a smaller count is
        refused by name with the count that would meet it.  ``nlev`` is
        the whole stack's level count, the number every checkpoint,
        receipt and sizing line carries.
        """
        n = int(nlev)
        base = cls.surface_stretched(
            int(base_nlev), p_top_pa,
            pure_pressure_above_pa=pure_pressure_above_pa,
            hybrid_exponent=hybrid_exponent,
        )
        if not 0.0 < float(band_top_pa) < float(band_bottom_pa) < float(taper_bottom_pa):
            raise ValueError(
                "jet_refined needs band_top_pa < band_bottom_pa < taper_bottom_pa"
            )
        if int(taper_layers) < 1:
            raise ValueError("jet_refined needs at least one taper layer")
        ps_ref = SURFACE_STRETCHED_REFERENCE_PS_PA
        p_base = base.a_half_pa + base.b_half * ps_ref
        top_index = int(np.argmin(np.abs(p_base - float(band_top_pa))))
        bottom_index = int(np.argmin(np.abs(p_base - float(taper_bottom_pa))))
        replaced = bottom_index - top_index
        if replaced < 2:
            raise ValueError(
                f"jet_refined found only {replaced} base layer(s) between "
                f"{p_base[top_index] / 100.0:.2f} and "
                f"{p_base[bottom_index] / 100.0:.2f} hPa; the band and the "
                "taper need at least two to replace"
            )
        band_top = float(p_base[top_index])
        band_bottom = float(band_bottom_pa)
        taper_bottom = float(p_base[bottom_index])
        if not band_top < band_bottom < taper_bottom:
            raise ValueError(
                f"jet_refined band bottom {band_bottom / 100.0:.2f} hPa must lie "
                f"between the kept half levels {band_top / 100.0:.2f} and "
                f"{taper_bottom / 100.0:.2f} hPa"
            )
        n_band = n - int(base_nlev) + replaced - int(taper_layers)
        minimum_band = int(math.ceil(
            (band_bottom - band_top) / float(JET_REFINED_BAND_MAXIMUM_LAYER_PA) - 1.0e-9
        ))
        minimum_nlev = minimum_band + int(base_nlev) - replaced + int(taper_layers)
        if n_band < minimum_band:
            # Refusal: below this count the band is coarser than the
            # analysis's own 25 hPa level spacing at the jet, the very
            # thing the layout exists to resolve (floor of 2026-09-04:
            # 1.85 m/s rmsve and -0.53 m/s speed bias at 250 hPa on the
            # 40-level stack's 55 hPa layers).
            raise ValueError(
                f"jet_refined at {n} levels would lay {n_band} layer(s) of "
                f"{(band_bottom - band_top) / max(n_band, 1) / 100.0:.2f} hPa "
                f"between {band_top / 100.0:.2f} and {band_bottom / 100.0:.2f} "
                f"hPa, thicker than the {JET_REFINED_BAND_MAXIMUM_LAYER_PA / 100.0:g} "
                f"hPa the band exists to hold (the analysis's own level spacing "
                f"at the jet); nlev = {minimum_nlev} is the smallest count that "
                f"meets it, or select coordinate = \"surface_stretched\""
            )
        band_thickness = (band_bottom - band_top) / n_band
        band_half = band_top + band_thickness * np.arange(1, n_band + 1, dtype=np.float64)
        band_half[-1] = band_bottom
        # Taper: t_i = band_thickness * r**i, i = 1..taper_layers, summing to
        # the span; r solved by bisection (monotone in r).
        span = taper_bottom - band_bottom
        m = int(taper_layers)

        def taper_total(r: float) -> float:
            return band_thickness * sum(r ** i for i in range(1, m + 1))

        if taper_total(1.0) >= span:
            raise ValueError(
                f"jet_refined taper of {m} layer(s) of at least the band "
                f"thickness {band_thickness / 100.0:.2f} hPa already exceeds the "
                f"{span / 100.0:.2f} hPa between {band_bottom / 100.0:.2f} and "
                f"{taper_bottom / 100.0:.2f} hPa; fewer taper layers or a "
                "deeper taper are needed"
            )
        low, high = 1.0, 8.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            if taper_total(mid) < span:
                low = mid
            else:
                high = mid
        ratio = 0.5 * (low + high)
        taper_half = band_bottom + np.cumsum(
            band_thickness * _libm_pow(ratio, np.arange(1, m + 1, dtype=np.float64))
        )
        taper_half[-1] = taper_bottom
        new_half = np.concatenate([band_half[:-1], [band_bottom], taper_half[:-1]])
        p_bt = float(pure_pressure_above_pa)
        x = np.clip((new_half - p_bt) / (ps_ref - p_bt), 0.0, 1.0)
        b_new = _libm_pow(x, float(hybrid_exponent))
        a_new = np.maximum(new_half - b_new * ps_ref, 0.0)
        a = np.concatenate([
            base.a_half_pa[: top_index + 1], a_new, base.a_half_pa[bottom_index:],
        ])
        b = np.concatenate([
            base.b_half[: top_index + 1], b_new, base.b_half[bottom_index:],
        ])
        if a.size != n + 1:
            raise AssertionError(
                f"jet_refined laid {a.size - 1} layers for nlev = {n}"
            )
        return cls(a, b)

    def thickest_layer_pa_between(
        self, p_top_pa: float, p_bottom_pa: float,
        reference_surface_pa: float = SURFACE_STRETCHED_REFERENCE_PS_PA,
    ) -> float:
        """The thickest layer whose full level lies in ``[p_top_pa,
        p_bottom_pa]`` at the reference surface (0.0 if none does)."""
        ps = float(reference_surface_pa)
        p_half = self.a_half_pa + self.b_half * ps
        p_full = np.sqrt(p_half[:-1] * p_half[1:])
        inside = (p_full >= float(p_top_pa)) & (p_full <= float(p_bottom_pa))
        if not np.any(inside):
            return 0.0
        return float(np.max(np.diff(p_half)[inside]))

    def describe(
        self, reference_surface_pa: float = SURFACE_STRETCHED_REFERENCE_PS_PA
    ) -> dict[str, object]:
        """The grid a run states about itself: heights and a layer table.

        Heights are ICAO standard-atmosphere geopotential heights above a
        surface at ``reference_surface_pa`` (a description of the grid, not
        of any column of a run).
        """
        ps = float(reference_surface_pa)
        p_half = self.a_half_pa + self.b_half * ps
        p_full = np.sqrt(p_half[:-1] * p_half[1:])
        z_surface = standard_atmosphere_height_m(ps)
        z_full = standard_atmosphere_height_m(p_full) - z_surface
        z_half = standard_atmosphere_height_m(p_half) - z_surface
        layers = [
            {
                "level": int(k),
                "p_half_top_pa": float(p_half[k]),
                "p_half_bottom_pa": float(p_half[k + 1]),
                "p_full_pa": float(p_full[k]),
                "thickness_pa": float(p_half[k + 1] - p_half[k]),
                "ln_ratio": float(math.log(p_half[k + 1] / p_half[k])),
                "a_half_top_pa": float(self.a_half_pa[k]),
                "b_half_top": float(self.b_half[k]),
                "z_full_m_agl": float(z_full[k]),
                "z_half_top_m_agl": float(z_half[k]),
            }
            for k in range(self.nlev)
        ]
        return {
            "nlev": int(self.nlev),
            "p_top_pa": float(self.a_half_pa[0]),
            "reference_surface_pa": ps,
            "first_full_level_height_m": float(z_full[-1]),
            "bottom_layer_thickness_pa": float(p_half[-1] - p_half[-2]),
            "maximum_layer_thickness_pa": float(np.max(np.diff(p_half))),
            "top_layer_ln_ratio": float(math.log(p_half[1] / p_half[0])),
            "full_levels_above_50hpa": int(np.count_nonzero(p_full < 5_000.0)),
            "pure_pressure_levels": int(np.count_nonzero(self.b_half[:-1] == 0.0)),
            # The jet band the upper-air scorecard reads 250 hPa in: the
            # thickest layer whose full level lies between 120 and 400 hPa
            # (59.8 hPa on the 40-level surface_stretched stack, 23.4 hPa
            # on the 48-level jet_refined one).
            "jet_band_pa": [float(JET_BAND_REPORT_PA[0]), float(JET_BAND_REPORT_PA[1])],
            "jet_band_thickest_layer_pa": self.thickest_layer_pa_between(
                JET_BAND_REPORT_PA[0], JET_BAND_REPORT_PA[1], ps
            ),
            "full_levels_in_jet_band": int(np.count_nonzero(
                (p_full >= JET_BAND_REPORT_PA[0]) & (p_full <= JET_BAND_REPORT_PA[1])
            )),
            "height_reference": "ICAO standard atmosphere above the reference surface",
            "layers": layers,
        }

    def device_arrays(self, backend):
        return (
            backend.asarray(self.a_half_pa, dtype=backend.float_dtype),
            backend.asarray(self.b_half, dtype=backend.float_dtype),
        )

    def pressure(self, ps, backend, *, metal_context=None) -> dict[str, object]:
        xp = backend.xp
        surface = xp.asarray(ps, dtype=backend.float_dtype)
        if metal_context is not None and backend.name == "numpy":
            from woof.metal.array import MetalArrayModule
            from woof.metal.generated import kernel_specs

            surface_host = np.ascontiguousarray(surface, dtype=np.float64)
            if surface_host.ndim != 2:
                raise ValueError("Global Metal hybrid pressure expects a 2-D surface")
            nlevel = self.nlev
            ncolumn = int(surface_host.size)
            if (nlevel + 1) * ncolumn > 0x7FFFFFFF:
                raise ValueError("Global Metal hybrid pressure extent exceeds its int32 ABI")
            metal = MetalArrayModule(metal_context)
            resources = {
                "a_half": metal.asarray(np.ascontiguousarray(self.a_half_pa)),
                "b_half": metal.asarray(np.ascontiguousarray(self.b_half)),
                "surface_pressure": metal.asarray(surface_host.reshape(-1)),
                "pressure_half": metal.empty(
                    (nlevel + 1, ncolumn), dtype=np.float64
                ),
                "pressure_full": metal.empty(
                    (nlevel, ncolumn), dtype=np.float64
                ),
                "layer_thickness": metal.empty(
                    (nlevel, ncolumn), dtype=np.float64
                ),
            }
            metal_context.submit_kernel(
                "global_hybrid_pressure_setup_fragment", resources,
                {"nlevel": nlevel, "ncolumn": ncolumn},
                exact_policy=False, grid_size=(nlevel + 1) * ncolumn,
                threads_per_group=(
                    kernel_specs.GLOBAL_HYBRID_PRESSURE_SETUP_FRAGMENT_THREADS_PER_GROUP
                ),
            )
            p_half = resources["pressure_half"].to_numpy().reshape(
                nlevel + 1, *surface_host.shape
            )
            p_full = resources["pressure_full"].to_numpy().reshape(
                nlevel, *surface_host.shape
            )
            dp = resources["layer_thickness"].to_numpy().reshape(
                nlevel, *surface_host.shape
            )
            if np.any(p_half <= 0.0):
                raise FloatingPointError("hybrid half-level pressure must remain positive")
            if np.any(dp <= 0.0):
                raise FloatingPointError("hybrid layer pressure thickness must remain positive")
            ln_ratio = xp.log(p_half[1:] / p_half[:-1])
            return {
                "p_half": p_half, "p_full": p_full, "dp": dp,
                "ln_ratio": ln_ratio,
            }
        a, b = self.device_arrays(backend)
        p_half = a[:, None, None] + b[:, None, None] * surface[None]
        if bool(xp.any(p_half <= 0.0)):
            raise FloatingPointError("hybrid half-level pressure must remain positive")
        dp = p_half[1:] - p_half[:-1]
        if bool(xp.any(dp <= 0.0)):
            raise FloatingPointError("hybrid layer pressure thickness must remain positive")
        p_full = xp.sqrt(p_half[:-1] * p_half[1:])
        ln_ratio = xp.log(p_half[1:] / p_half[:-1])
        return {
            "p_half": p_half, "p_full": p_full, "dp": dp, "ln_ratio": ln_ratio,
        }

    @staticmethod
    def half_layer_thickness(
        virtual_temperature, p_full, ln_ratio, xp,
        *, gas_constant: float = DRY_AIR_GAS_CONSTANT,
    ):
        """``R * int_{p_k}^{p_{k+1/2}} Tv dlnp`` with Tv linear in ln p.

        The full level is the ln-p midpoint of its layer (``p_k =
        sqrt(p_{k-1/2} p_{k+1/2})``), so the full-layer integral ``R Tv_k
        ln_ratio_k`` is the midpoint rule, exact for Tv linear in ln p.  The
        half layer below the level is not: integrating it with the level
        temperature alone, ``R Tv_k ln_ratio_k / 2``, omits ``R (dTv/dlnp)
        ln_ratio_k^2 / 8``, and the terrain-following horizontal variation
        of that term was a 1.06 m/s per hour spurious top-level
        acceleration at rest over a 2 km mountain (audit 2026-09-01, DN-3;
        20-level pressure_blend grid, top layer 1.97 in ln p).  The
        gradient is the centred difference of Tv in ln p across the
        neighbouring levels (one-sided in the top and bottom layers), exact
        for a linear profile, so the geopotential of a column with Tv
        linear in ln p is exact at every full level and the midpoint
        pressure gradient ``R Tv_k grad(ln p_k)`` balances it pointwise.

        The Simmons-Burridge layer-mean pairing (``phi_k = phi_{k+1/2} + R
        Tv_k alpha_k`` against the layer-mean pressure gradient) was built
        and measured first on the same test: 1.36 m/s per hour, because it
        also carries one temperature per layer and is exact at rest only
        for an isothermal layer.
        """
        tv = virtual_temperature
        ln_full = xp.log(p_full)
        gradient = xp.empty_like(tv)
        gradient[1:-1] = (tv[2:] - tv[:-2]) / (ln_full[2:] - ln_full[:-2])
        gradient[0] = (tv[1] - tv[0]) / (ln_full[1] - ln_full[0])
        gradient[-1] = (tv[-1] - tv[-2]) / (ln_full[-1] - ln_full[-2])
        half = 0.5 * ln_ratio
        return float(gas_constant) * (tv * half + gradient * (half * half * 0.5))

    def hydrostatic_geopotential(
        self,
        virtual_temperature,
        surface_geopotential,
        p_half,
        *,
        gas_constant: float = DRY_AIR_GAS_CONSTANT,
    ):
        xp = np
        module = type(virtual_temperature).__module__.split(".", 1)[0]
        if module == "cupy":
            try:
                import cupy as xp  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "CuPy array supplied to hydrostatic_geopotential but CuPy is unavailable"
                ) from exc
        t = virtual_temperature
        phi = xp.empty_like(t)
        lower_phi = surface_geopotential
        ln_ratio = xp.log(p_half[1:] / p_half[:-1])
        p_full = xp.sqrt(p_half[:-1] * p_half[1:])
        half = self.half_layer_thickness(
            t, p_full, ln_ratio, xp, gas_constant=gas_constant
        )
        for k in range(self.nlev - 1, -1, -1):
            phi[k] = lower_phi + half[k]
            if k:
                lower_phi = lower_phi + float(gas_constant) * t[k] * ln_ratio[k]
        return phi

    def continuity(self, divergence_mass_flux, ps_t, backend, *,
                   metal_context=None):
        """Diagnose interface pressure velocity from layer mass continuity.

        ``divergence_mass_flux`` is ``div(dp*u, dp*v)`` in Pa/s. The top
        interface is zero. Exact algebra makes the bottom interface zero when
        ``ps_t = -sum(divergence_mass_flux)``; the returned residual measures
        arithmetic closure before the boundary is pinned.
        """
        xp = backend.xp
        divm = xp.asarray(divergence_mass_flux, dtype=backend.float_dtype)
        delta_b = backend.asarray(self.delta_b, dtype=backend.float_dtype)
        dp_t = delta_b[:, None, None] * ps_t[None]
        if metal_context is not None and backend.name == "numpy":
            from woof.metal.generated import kernel_specs
            if hasattr(kernel_specs, "GLOBAL_VERTICAL_CONTINUITY_FRAGMENT_RESOURCES"):
                from woof.metal.array import MetalArrayModule

                host_div = np.ascontiguousarray(
                    np.moveaxis(np.asarray(divm, dtype=np.float64), 0, -1)
                )
                host_dp_t = np.ascontiguousarray(
                    np.moveaxis(np.asarray(dp_t, dtype=np.float64), 0, -1)
                )
                ncolumn = int(np.prod(host_div.shape[:-1], dtype=np.int64))
                nlevel = int(host_div.shape[-1])
                metal = MetalArrayModule(metal_context)
                resources = {
                    "divergence_mass": metal.asarray(
                        host_div.reshape(ncolumn, nlevel)),
                    "pressure_tendency": metal.asarray(
                        host_dp_t.reshape(ncolumn, nlevel)),
                    "omega": metal.empty((ncolumn, nlevel + 1),
                                         dtype=np.float64),
                    "residual": metal.empty((ncolumn,), dtype=np.float64),
                }
                metal_context.submit_kernel(
                    "global_vertical_continuity_fragment", resources,
                    {"count": ncolumn, "nlevel": nlevel},
                    exact_policy=False, grid_size=ncolumn,
                    threads_per_group=(
                        kernel_specs.GLOBAL_VERTICAL_CONTINUITY_FRAGMENT_THREADS_PER_GROUP
                    ),
                )
                omega_host = resources["omega"].to_numpy().reshape(
                    *host_div.shape[:-1], nlevel + 1
                )
                omega_host = np.ascontiguousarray(
                    np.moveaxis(omega_host, -1, 0)
                )
                residual = resources["residual"].to_numpy().reshape(
                    host_div.shape[:-1]
                )
                return omega_host, residual
        omega = xp.zeros(
            (self.nlev + 1, *divm.shape[-2:]), dtype=backend.float_dtype
        )
        if backend.name == "cupy":
            # The recurrence omega[k+1] = omega[k] - dp_t[k] - divm[k] with
            # omega[0] = 0 is the prefix sum omega[j] = -sum_{k<j}(dp_t[k]
            # + divm[k]); one scan replaces three kernel launches per level.
            # Association differs from the loop only in fp last bits, so
            # this rides the cupy branch; the numpy loop below stays the
            # reference.
            omega[1:] = -xp.cumsum(dp_t + divm, axis=0)
        else:
            for k in range(self.nlev):
                omega[k + 1] = omega[k] - dp_t[k] - divm[k]
        residual = omega[-1].copy()
        omega[-1] = 0.0
        return omega, residual

    @staticmethod
    def mass_per_area(dp):
        return dp / GRAVITY_M_S2
