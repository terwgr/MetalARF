"""Member-batched point operators for the neutral variable vocabulary.

The v1 door's ``_ModelSpace`` evaluates one atmosphere at a set of points;
an ensemble of R members would pay R basis constructions for the same
points.  Here the members' coefficients are stacked on a leading axis and
``sample_scalar`` / ``sample_wind`` (which keep leading coefficient
dimensions) build the associated Legendre basis once per member chunk.
The arithmetic per member is the door's: surface pressure reduced to the
station through the lowest level's virtual temperature, temperature to
2 m at 6.5 K/km, the dewpoint of the lowest level's vapor at the station
pressure, the 10 m wind from the lowest level through the model's own
similarity diagnostic with the MEMBER's surface (skin, land fraction, soil
wetness, roughness); aloft rows read the member's profiles interpolated
in ln p to the report's level.

Members differ in their surface (Noah's soil and skin evolve per member),
so the 10 m reduction is per member; the terrain is one field.

Cost rule (measured on the T63 / T127 twin, 2026-09-06): the aloft wind
sampling (``sample_wind``, a vector synthesis of every level at every
point) is the operators' cost, and an analysis that evaluated each batch
on its own paid it once per batch and again per withheld split, about
fifty times per analysis.  :func:`evaluate_batches` therefore gathers the
rows of every batch into ONE evaluation per member set and scatters the
values back, and :meth:`MemberOperators.evaluate` computes only the
``variables`` asked for (the wind synthesis is skipped when no wind row
is present).

Where the members live on a card, the sampling runs there (the device
path of :mod:`woof.globe.spectral.sampling`): the coefficient stack of
EVERY member is one device array, the Legendre basis is built on the
device once per point chunk, and each order is one matmul over the whole
stack.  Measured on the case's real hourly tables (about 30,000 rows a
window, 100,000 motion-vector rows an hour thinned to the T127 grid) the
host path spent the whole cycle in numpy einsums (both cards at 0 %
utilisation for ten minutes without a first analysis, 2026-09-06); the
device path is what makes a real-data hour affordable.

Two families of rows, evaluated separately by :func:`evaluate_batches`:
the neutral point vocabulary above, and radio-occultation refractivity
(``refractivity_n``), whose row is anchored at its tangent HEIGHT
(``elevation_m``) with the retrieval's dry pressure in ``level_pa`` for
the localisation metric only.  The refractivity path samples the same
theta, vapour and ln ps stack on the card and runs the calibrated column
arithmetic of :mod:`woof.globe.obs_operators`
(:func:`~woof.globe.obs_operators.refractivity_at_heights_columns`)
over every row at once, so a refractivity row is a neutral row of this
package and needs no operator of its own; the door's
``VARIABLES_WITHOUT_OPERATOR`` refusal is the successive correction's
alone.

A motion vector's error is situation dependent (Forsythe and Saunders
2008): a height assignment that misses by ``sigma_p`` in a sheared layer
is a wind error of the shear across ``sigma_p``.  For rows whose
``measurement`` is ``amv_assigned_pressure`` the aloft path also reads,
per member, the wind change across ``sigma_p`` either side of the
assigned pressure (``wind_shear_m_s``, the pseudo-variable), and
:func:`evaluate_batches` stores its member mean on the batch
(``assignment_shear``) for the filter to fold into the row's error.
EVERY member is one device array, the packed Legendre basis of a point
chunk is one fused kernel, and the whole stack is contracted with one
GEMM per chunk and output.  Measured on the case's real hourly tables
(about 30,000 rows a window, 100,000 motion-vector rows an hour thinned
to the T127 grid) the host path spent the whole cycle in numpy einsums
(both cards at 0 % utilisation for ten minutes without a first analysis,
2026-09-06); the first device path then spent 53 s per control operator
call and 17 s per 32-member call in one small complex matmul per order
(three operator calls and six in-window member observations an hour,
about 230 s of a 560 s cycle on the RTX 5090).

Precision (``precision``): ``"state"`` (the default) samples a float32
state's coefficients with float32 GEMMs over 256-term blocks whose
partial sums are added in float64, the state's own precision (its
coefficients carry 6e-8 of relative quantisation; measured 4e-7 rms and
2e-6 maximum relative against the float64 contraction on the case's
T127 members, the transform's round-trip gate being 1e-6); a float64
state is sampled in float64.  ``"float64"`` forces the float64
contraction on any state (the host path's arithmetic to rounding, for the
bit-comparison).  The aloft profile arithmetic (the Exner factor, the
ln p interpolation, the dewpoint) runs where the samples are, per member,
with the host functions' operations in their order, and hands back
numpy; the surface rows' reductions stay on the host (a few thousand
points, and the 10 m wind reads the member's surface through the model's
own similarity diagnostic there).
"""
from __future__ import annotations

import numpy as np

from woof.globe.spectral.sampling import sample_scalar, sample_wind

from ..assimilate import (
    SURFACE_LAPSE_K_M,
    _anemometer_wind,
    _interp_ln_pressure,
    _sample_grid,
    _to_numpy_spectral,
)
from ..constants import (
    DRY_AIR_GAS_CONSTANT,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
)
from ..obs_operators import refractivity_at_heights_columns
from ..obs_table import VARIABLE_TABLE, ObsRow
from ..surface_energy import _BOLTON_A, _BOLTON_B, _BOLTON_C, _EPSILON, dewpoint_from_specific_humidity
from .observations import PointObs

__all__ = [
    "AMV_MEASUREMENT", "COLUMN_VARIABLES", "MemberOperators", "OPERATOR_PRECISIONS",
    "WIND_SHEAR_VARIABLE", "batches_from_rows", "evaluate_batches", "NEUTRAL_VARIABLES",
    "OPERATOR_VARIABLES",
]

#: How the operators contract a device-resident state: ``"state"`` at the
#: state's own precision (float32 GEMMs over blocks summed in float64 for a
#: float32 state, float64 for a float64 state), ``"float64"`` always in
#: float64.  The host path is complex128 under either.
OPERATOR_PRECISIONS = ("state", "float64")

#: The variables these operators evaluate.  The dewpoint is listed here
#: even where the observation table of the tree does not yet carry it (it
#: joined the table with the initial-state lane's moisture update): the
#: operator is the door's, the dewpoint of the lowest level's vapor at the
#: station pressure, and a table without the variable simply offers no
#: dewpoint rows.
OPERATOR_VARIABLES = (
    "surface_pressure_pa", "temperature_k", "dewpoint_k", "wind_u_m_s", "wind_v_m_s",
    "refractivity_n",
)
#: The point vocabulary the surface and aloft paths answer for; a
#: refractivity row is the other family (its own path, its own anchor).
COLUMN_VARIABLES = frozenset(OPERATOR_VARIABLES) - {"refractivity_n"}
#: The pseudo-variable of the height-assignment shear (never a report
#: variable; asked for beside the wind at motion-vector rows).
WIND_SHEAR_VARIABLE = "wind_shear_m_s"
#: The measurement label of a satellite motion vector row
#: (:data:`woof.globe.obs_table.MEASUREMENT_TABLE`).
AMV_MEASUREMENT = "amv_assigned_pressure"
#: A point at or beyond this latitude is the pole for the wind operator: a
#: wind vector there has no east or north, the gradient sampler refuses it,
#: and the operators hand back NaN so quality control can refuse the row by
#: name (``polar_wind``).  The case's tables carry the Amundsen-Scott
#: sounding at exactly 90 S.
POLE_LATITUDE_DEG = 90.0 - 1.0e-6
NEUTRAL_VARIABLES = tuple(dict.fromkeys((*VARIABLE_TABLE, *OPERATOR_VARIABLES)))


def _interp_ln_pressure_xp(xp, profile, ln_p, ln_target):
    """:func:`woof.globe.assimilate._interp_ln_pressure` with the
    array module named: the same operations in the same order, so the
    numpy call is bitwise the host function and the device call the same
    arithmetic on the card."""
    nlev = profile.shape[0]
    position = xp.sum(ln_p <= ln_target[None, :], axis=0)
    upper = xp.clip(position - 1, 0, nlev - 1)
    lower = xp.clip(position, 0, nlev - 1)
    take = xp.take_along_axis
    lp_upper = take(ln_p, upper[None, :], 0)[0]
    lp_lower = take(ln_p, lower[None, :], 0)[0]
    v_upper = take(profile, upper[None, :], 0)[0]
    v_lower = take(profile, lower[None, :], 0)[0]
    span = xp.where(lower == upper, 1.0, lp_lower - lp_upper)
    weight = xp.clip((ln_target - lp_upper) / span, 0.0, 1.0)
    return v_upper * (1.0 - weight) + v_lower * weight


def _dewpoint_xp(xp, q, p_pa):
    """:func:`woof.globe.surface_energy.dewpoint_from_specific_humidity`
    with the array module named (Bolton 1980, the same constants and
    operations)."""
    q = xp.clip(xp.asarray(q, dtype=xp.float64), 1.0e-9, None)
    e = q * p_pa / (_EPSILON + (1.0 - _EPSILON) * q)
    ln = xp.log(xp.maximum(e, 1.0e-6) / _BOLTON_A)
    return _BOLTON_C * ln / (_BOLTON_B - ln) + 273.15


def _unique_points(lat, lon):
    pairs = np.stack([np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique[:, 0].copy(), unique[:, 1].copy(), np.asarray(inverse).ravel()


class MemberOperators:
    """H(x_k) for every member at point rows.

    ``transform``, ``vertical`` and ``terrain_spectral`` (numpy complex,
    the surface geopotential's coefficients) are the ensemble's;
    ``soil_wetness_capacity`` the physics option the door's operator
    reads.  ``member_chunk`` bounds the host stack of coefficients one
    sampling call carries.
    """

    def __init__(self, transform, vertical, terrain_spectral, soil_wetness_capacity: float,
                 *, member_chunk: int = 8, precision: str = "state",
                 amv_height_assignment_sigma_pa: float = 0.0):
        if precision not in OPERATOR_PRECISIONS:
            raise ValueError(f"operator precision must be one of {OPERATOR_PRECISIONS}, got {precision!r}")
        self.transform = transform
        self.vertical = vertical
        self.terrain = np.asarray(terrain_spectral, dtype=np.complex128)
        self.soil_wetness_capacity = float(soil_wetness_capacity)
        self.member_chunk = max(1, int(member_chunk))
        self.amv_height_assignment_sigma_pa = float(amv_height_assignment_sigma_pa)
        self.precision = str(precision)
        self.a = np.asarray(vertical.a_half_pa)
        self.b = np.asarray(vertical.b_half)

    @classmethod
    def for_model(cls, model, transform, cfg, **kwargs) -> "MemberOperators":
        terrain = _to_numpy_spectral(transform.backend, transform.forward(model.surface_geopotential))
        return cls(transform, model.vertical, terrain, cfg.reference_physics.soil_wetness_capacity, **kwargs)

    def _point_pressures(self, ps: np.ndarray, xp=np) -> np.ndarray:
        """``(nlev, ...)`` full-level pressures at points with surface
        pressure ``ps`` (any trailing shape), on ``xp``."""
        shape = (self.a.size,) + (1,) * ps.ndim
        a = xp.asarray(self.a).reshape(shape)
        b = xp.asarray(self.b).reshape(shape)
        p_half = a + b * ps[None]
        return xp.sqrt(p_half[:-1] * p_half[1:])

    def _sample_precision(self, coeff) -> str:
        """The sampler precision for ``coeff``: float64 when asked for, else
        the coefficients' own (float32 for complex64)."""
        if self.precision == "float64":
            return "float64"
        dtype = np.dtype(getattr(coeff, "dtype", np.complex128))
        return "float32" if dtype == np.dtype(np.complex64) else "float64"

    @property
    def precision_record(self) -> dict[str, object]:
        """What the receipt says about the operators' arithmetic."""
        xp = self._xp
        return {
            "requested": self.precision,
            "path": "host" if xp is np else "device",
            "rule": (
                "host: complex128; device: float64 contraction, or under 'state' a float32 "
                "state's coefficients contracted by float32 GEMMs over 256-term blocks whose "
                "partial sums are added in float64 (the state's own precision)"
            ),
        }

    def _host(self, atmosphere, name):
        return _to_numpy_spectral(self.transform.backend, getattr(atmosphere, name))

    @property
    def _xp(self):
        """The array module the coefficient stacks are built with: the
        transform's backend module when it is a device module (the
        members' coefficients stay on the card and the point sampling runs
        there, one basis per point chunk for the whole stack), numpy
        otherwise."""
        xp = self.transform.backend.xp
        return np if xp is np else xp

    def _coeff(self, atmosphere, name):
        """A member's spectral field for the sampling stack: the device
        array itself on a device backend, the host complex128 copy on
        numpy."""
        if self._xp is np:
            return self._host(atmosphere, name)
        return getattr(atmosphere, name)

    def _chunked(self, members):
        """The member chunks one sampling call carries: every member at
        once on the device (one basis, one matmul per order for the whole
        stack), ``member_chunk`` at a time on the host."""
        step = len(members) if self._xp is not np else self.member_chunk
        step = max(1, step)
        return [(start, members[start:start + step]) for start in range(0, len(members), step)]

    def _wind_at(self, zeta, div, ulat, ulon, *, as_device: bool = False):
        """``sample_wind`` at the non-polar unique points, NaN at a pole;
        on the array module the coefficients live on when ``as_device``."""
        precision = self._sample_precision(zeta)
        xp = self._xp if as_device else np
        polar = np.abs(ulat) >= POLE_LATITUDE_DEG
        if not polar.any():
            return sample_wind(self.transform, zeta, div, ulat, ulon, precision=precision, as_device=as_device)
        keep = ~polar
        u, v = sample_wind(self.transform, zeta, div, ulat[keep], ulon[keep], precision=precision,
                           as_device=as_device)
        full_u = xp.full((*u.shape[:-1], ulat.size), np.nan)
        full_v = xp.full((*v.shape[:-1], ulat.size), np.nan)
        keep_idx = xp.asarray(np.nonzero(keep)[0])
        full_u[..., keep_idx] = u
        full_v[..., keep_idx] = v
        return full_u, full_v

    def _terrain_coeff(self):
        if self._xp is np:
            return self.terrain
        if getattr(self, "_terrain_device", None) is None:
            self._terrain_device = self._xp.asarray(self.terrain)
        return self._terrain_device

    def batch_operator(self, members, batch: PointObs) -> np.ndarray:
        """``(R, batch.count)`` H(x_k) of the batch's variable at the
        batch's rows: the ``operator`` every neutral batch carries."""
        values, _ = self.evaluate(
            members, batch.latitude_deg, batch.longitude_deg, batch.elevation_m, batch.level_pa(),
            variables=(batch.variable,))
        return values[batch.variable]

    def evaluate(self, members, latitude_deg, longitude_deg, elevation_m, level_pa, *, variables=None):
        """Every neutral variable's operator at the rows, per member.

        ``level_pa`` NaN marks a surface row.  ``variables`` names the
        operator variables to compute (``None``: the column vocabulary);
        the others stay NaN and their arithmetic is skipped.  Returns
        ``(values, ln_pressure)``: ``values[variable]`` is ``(R, n)`` (NaN
        where the variable has no operator at that row, surface pressure
        aloft), and ``ln_pressure`` ``(n,)`` is the row's vertical
        position for the localisation metric: ``ln(level_pa)`` aloft, the
        ensemble-mean ln of the model's surface pressure at the station
        for a surface row.  ``refractivity_n`` asks for the refractivity
        family (the rows' ``elevation_m`` are tangent heights, their
        ``level_pa`` the dry pressure); :data:`WIND_SHEAR_VARIABLE` asks
        for the height-assignment shear beside the wind at aloft rows.
        """
        lat = np.asarray(latitude_deg, dtype=np.float64)
        lon = np.asarray(longitude_deg, dtype=np.float64)
        elev = np.asarray(elevation_m, dtype=np.float64)
        level = np.asarray(level_pa, dtype=np.float64)
        wanted = frozenset(COLUMN_VARIABLES if variables is None else variables)
        n = lat.size
        r = len(members)
        out = {name: np.full((r, n), np.nan) for name in OPERATOR_VARIABLES}
        if WIND_SHEAR_VARIABLE in wanted:
            out[WIND_SHEAR_VARIABLE] = np.full((r, n), np.nan)
            wanted = wanted | {"wind_u_m_s", "wind_v_m_s"}
        ln_pressure = np.full(n, np.nan)
        surface = np.isnan(level)
        aloft = ~surface
        column = bool(wanted & COLUMN_VARIABLES)
        if column and surface.any():
            self._surface_rows(members, lat[surface], lon[surface], elev[surface], out, ln_pressure, surface, wanted)
        if column and aloft.any():
            self._aloft_rows(members, lat[aloft], lon[aloft], level[aloft], out, aloft, wanted)
        if "refractivity_n" in wanted and n:
            self._refractivity_rows(members, lat, lon, elev, out)
        ln_pressure[aloft] = np.log(level[aloft])
        return out, ln_pressure

    def _refractivity_rows(self, members, lat, lon, elev, out):
        """Refractivity at every row's tangent height, per member: the
        theta, vapour and ln ps stack sampled at the rows' distinct points
        (on the card where the members live), the full-level pressures
        from the hybrid coordinate, temperature through the Exner
        function, and the calibrated column arithmetic vectorised over
        the rows (``refractivity_at_heights_columns``: heights by the
        hypsometric integration with the model's gravity and gas constant,
        ln p linear in height inside the layer, T and q linear in ln p; a
        target outside the column's span is NaN, never extrapolated, and
        quality control refuses it by name)."""
        ulat, ulon, at = _unique_points(lat, lon)
        nlev = self.a.size - 1
        xp = self._xp
        terrain = np.asarray(sample_scalar(self.transform, self._terrain_coeff(), ulat, ulon), dtype=np.float64)
        phi_s = terrain[at]
        for start, chunk in self._chunked(members):
            stack = xp.stack([
                xp.concatenate([
                    self._coeff(m.atmosphere, "theta"),
                    self._coeff(m.atmosphere, "qv"),
                    self._coeff(m.atmosphere, "log_surface_pressure")[None],
                ]) for m in chunk
            ])                                                     # (Rc, 2 nlev + 1, n, m)
            sampled = np.asarray(sample_scalar(self.transform, stack, ulat, ulon), dtype=np.float64)  # (Rc, fields, U)
            del stack
            for c in range(len(chunk)):
                k = start + c
                ps = np.exp(sampled[c, -1])[at]                     # (n,)
                p_full = self._point_pressures(ps)                   # (nlev, n)
                theta = sampled[c, :nlev][:, at]
                t_prof = theta * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
                q_prof = np.maximum(sampled[c, nlev:2 * nlev][:, at], 0.0)
                values, _ = refractivity_at_heights_columns(
                    t_prof, q_prof, p_full, ps, phi_s, elev,
                    gravity=GRAVITY_M_S2, gas_constant=DRY_AIR_GAS_CONSTANT,
                )
                out["refractivity_n"][k] = values

    def _surface_rows(self, members, lat, lon, elev, out, ln_pressure, mask, wanted=frozenset(OPERATOR_VARIABLES)):
        ulat, ulon, at = _unique_points(lat, lon)
        xp = self._xp
        terrain_coeff = self._terrain_coeff()
        terrain = sample_scalar(self.transform, terrain_coeff, ulat, ulon,
                                precision=self._sample_precision(terrain_coeff))
        z_model = terrain / GRAVITY_M_S2
        r = len(members)
        ps_all = np.zeros((r, ulat.size))
        rows = np.nonzero(mask)[0]
        for start, chunk in self._chunked(members):
            stack = xp.stack([
                xp.stack([
                    self._coeff(m.atmosphere, "log_surface_pressure"),
                    self._coeff(m.atmosphere, "theta")[-1],
                    self._coeff(m.atmosphere, "qv")[-1],
                ]) for m in chunk
            ])                                                     # (Rc, 3, n, m)
            sampled = sample_scalar(self.transform, stack, ulat, ulon,
                                    precision=self._sample_precision(stack))  # (Rc, 3, U)
            del stack
            want_wind = bool(wanted & {"wind_u_m_s", "wind_v_m_s"})
            if want_wind:
                zeta = xp.stack([self._coeff(m.atmosphere, "vorticity")[-1:] for m in chunk])
                div = xp.stack([self._coeff(m.atmosphere, "divergence")[-1:] for m in chunk])
                u_low, v_low = self._wind_at(zeta, div, ulat, ulon)  # (Rc, 1, U) on the host, NaN at a pole
                del zeta, div
                u_low = u_low[:, 0]
                v_low = v_low[:, 0]
            for c, member in enumerate(chunk):
                k = start + c
                ps = np.exp(sampled[c, 0])
                p_full_low = self._point_pressures(ps)[-1]
                t_low = sampled[c, 1] * (p_full_low / REFERENCE_PRESSURE_PA) ** KAPPA
                qv_low = np.maximum(sampled[c, 2], 0.0)
                tv_low = t_low * (1.0 + 0.61 * qv_low)
                z_low_msl = z_model + (DRY_AIR_GAS_CONSTANT * tv_low / GRAVITY_M_S2) * np.log(ps / p_full_low)
                ps_all[k] = ps
                elev_u = elev  # per row, gathered below
                p_station = ps[at] * np.exp(
                    -GRAVITY_M_S2 * (elev_u - z_model[at]) / (DRY_AIR_GAS_CONSTANT * tv_low[at])
                )
                out["surface_pressure_pa"][k, rows] = p_station
                if "temperature_k" in wanted:
                    out["temperature_k"][k, rows] = t_low[at] + SURFACE_LAPSE_K_M * (
                        z_low_msl[at] - (elev_u + 2.0))
                if "dewpoint_k" in wanted:
                    out["dewpoint_k"][k, rows] = dewpoint_from_specific_humidity(qv_low[at], p_station)
                if not want_wind:
                    continue
                surface_state = member.surface
                to_numpy = self.transform.backend.to_numpy
                grid = self.transform.grid
                skin = np.asarray(to_numpy(surface_state.temperature_k), dtype=np.float64)
                land = np.asarray(to_numpy(surface_state.land_fraction), dtype=np.float64)
                wet = np.clip(
                    np.asarray(to_numpy(surface_state.soil_water_fraction[0]), dtype=np.float64)
                    / self.soil_wetness_capacity, 0.0, 1.0)
                rough = np.asarray(to_numpy(surface_state.roughness_m), dtype=np.float64)
                u10, v10 = _anemometer_wind(
                    u_low[c], v_low[c], t_low, qv_low, p_full_low, ps,
                    _sample_grid(skin, grid, ulat, ulon),
                    np.clip(_sample_grid(land, grid, ulat, ulon), 0.0, 1.0),
                    _sample_grid(wet, grid, ulat, ulon),
                    _sample_grid(rough, grid, ulat, ulon),
                )
                out["wind_u_m_s"][k, rows] = u10[at]
                out["wind_v_m_s"][k, rows] = v10[at]
        ln_pressure[rows] = np.log(ps_all.mean(axis=0))[at]

    def _aloft_rows(self, members, lat, lon, level, out, mask, wanted=frozenset(OPERATOR_VARIABLES)):
        ulat, ulon, at = _unique_points(lat, lon)
        nlev = self.a.size - 1
        rows = np.nonzero(mask)[0]
        r = len(members)
        want_t = "temperature_k" in wanted
        want_q = "dewpoint_k" in wanted
        want_wind = bool(wanted & {"wind_u_m_s", "wind_v_m_s"})
        want_shear = WIND_SHEAR_VARIABLE in wanted and self.amv_height_assignment_sigma_pa > 0.0
        sigma = self.amv_height_assignment_sigma_pa
        xp = self._xp
        on_device = xp is not np
        to_host = self.transform.backend.to_numpy if on_device else np.asarray
        # The row arithmetic runs where the samples are: the report levels
        # and the unique-point gather live there too.
        at_x = xp.asarray(at)
        level_x = xp.asarray(level, dtype=xp.float64)
        ln_level = xp.log(level_x)
        for start, chunk in self._chunked(members):
            parts = []
            if want_t:
                parts.append(lambda m: self._coeff(m.atmosphere, "theta"))
            if want_q:
                parts.append(lambda m: self._coeff(m.atmosphere, "qv"))
            parts.append(lambda m: self._coeff(m.atmosphere, "log_surface_pressure")[None])
            stack = xp.stack([xp.concatenate([part(m) for part in parts]) for m in chunk])
            sampled = sample_scalar(self.transform, stack, ulat, ulon,
                                    precision=self._sample_precision(stack), as_device=on_device)
            del stack                                              # (Rc, fields, U)
            if want_wind:
                zeta = xp.stack([self._coeff(m.atmosphere, "vorticity") for m in chunk])
                div = xp.stack([self._coeff(m.atmosphere, "divergence") for m in chunk])
                u_prof, v_prof = self._wind_at(zeta, div, ulat, ulon, as_device=on_device)
                del zeta, div                                      # (Rc, nlev, U), NaN at a pole
            for c in range(len(chunk)):
                k = start + c
                ps = xp.exp(sampled[c, -1])
                p_full = self._point_pressures(ps, xp)[:, at_x]
                ln_p = xp.log(p_full)
                offset = 0
                if want_t:
                    t_profile = sampled[c, offset:offset + nlev][:, at_x] * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
                    out["temperature_k"][k, rows] = to_host(_interp_ln_pressure_xp(xp, t_profile, ln_p, ln_level))
                    offset += nlev
                if want_q:
                    q_profile = xp.maximum(sampled[c, offset:offset + nlev][:, at_x], 0.0)
                    q_level = _interp_ln_pressure_xp(xp, q_profile, ln_p, ln_level)
                    out["dewpoint_k"][k, rows] = to_host(_dewpoint_xp(xp, q_level, level_x))
                    offset += nlev
                if want_wind:
                    u_at = u_prof[c][:, at_x]
                    v_at = v_prof[c][:, at_x]
                    out["wind_u_m_s"][k, rows] = to_host(_interp_ln_pressure_xp(xp, u_at, ln_p, ln_level))
                    out["wind_v_m_s"][k, rows] = to_host(_interp_ln_pressure_xp(xp, v_at, ln_p, ln_level))
                    if want_shear:
                        # The wind change across one height-assignment sigma
                        # either side of the assigned pressure, the interval
                        # clamped to the column (the top full level, the
                        # surface); half the vector difference is the wind
                        # error a one-sigma miss in height would make.  The
                        # same arithmetic as the wind, where the samples are.
                        p_hi = xp.minimum(level_x + sigma, ps[at_x])
                        p_lo = xp.maximum(level_x - sigma, p_full[0])
                        u_hi = _interp_ln_pressure_xp(xp, u_at, ln_p, xp.log(p_hi))
                        u_lo = _interp_ln_pressure_xp(xp, u_at, ln_p, xp.log(p_lo))
                        v_hi = _interp_ln_pressure_xp(xp, v_at, ln_p, xp.log(p_hi))
                        v_lo = _interp_ln_pressure_xp(xp, v_at, ln_p, xp.log(p_lo))
                        out[WIND_SHEAR_VARIABLE][k, rows] = to_host(0.5 * xp.hypot(u_hi - u_lo, v_hi - v_lo))


def evaluate_batches(operators: MemberOperators, states, batches, *, rows=None, target: str | None = "simulated"):
    """ONE evaluation of ``operators`` on ``states`` for the rows of every
    batch in ``batches`` (all of them, or per batch the index array
    ``rows[i]``), the values scattered back per batch.  ``target``
    ``"simulated"`` writes each batch's ``(R, n)`` member values (and the
    surface rows' ``ln_pressure``), ``"control_simulated"`` writes the
    ``(1, n)`` control values, ``None`` writes nothing and returns the list
    of ``(R, n_i)`` arrays.  Batches outside the neutral vocabulary are
    skipped (their operator is their own) and get ``None`` in the list.
    Why one call: the aloft wind synthesis is the operators' cost and is
    paid once per call, not once per batch and split."""
    out: list = [None] * len(batches)
    picks: list = [None] * len(batches)
    foreign: list[tuple[int, np.ndarray]] = []
    for i, batch in enumerate(batches):
        if batch.variable in OPERATOR_VARIABLES:
            continue
        if batch.operator is not None and getattr(batch.operator, "evaluates_states", False):
            idx = np.arange(batch.count) if rows is None else np.asarray(rows[i], dtype=int)
            foreign.append((i, idx))
    # A batch outside the neutral vocabulary whose operator declares
    # ``evaluates_states`` (a radiance stream's: it selects its transform by
    # the states' truncation) is evaluated here too, on the same rows and
    # into the same target, so a window observes its bins and the O-A pass
    # re-evaluates it like any other batch.
    for i, idx in foreign:
        batch = batches[i]
        if idx.size == 0:
            out[i] = np.zeros((len(states), 0))
            continue
        part = batch if idx.size == batch.count and np.array_equal(idx, np.arange(batch.count)) else batch.subset(idx)
        sim = np.asarray(batch.operator(states, part), dtype=np.float64)
        if sim.shape != (len(states), idx.size):
            raise ValueError(
                f"the operator of {batch.stream!r}/{batch.variable!r} handed back {sim.shape}, "
                f"not ({len(states)}, {idx.size})"
            )
        out[i] = sim
        if target == "simulated":
            if batch.simulated is None:
                batch.simulated = np.full((sim.shape[0], batch.count), np.nan)
            batch.simulated[:, idx] = sim
        elif target == "control_simulated":
            if batch.control_simulated is None:
                batch.control_simulated = np.full((1, batch.count), np.nan)
            batch.control_simulated[:, idx] = sim[:1]
    # Two families, one evaluation each: the column vocabulary and the
    # refractivity rows (their elevation is a tangent height, so they never
    # share a call with the surface and aloft paths).
    for family in ("column", "refractivity"):
        members_of = []
        lat, lon, elev, level = [], [], [], []
        variables = set()
        for i, batch in enumerate(batches):
            if batch.variable not in OPERATOR_VARIABLES:
                continue
            is_refractivity = batch.variable == "refractivity_n"
            if (family == "refractivity") != is_refractivity:
                continue
            idx = np.arange(batch.count) if rows is None else np.asarray(rows[i], dtype=int)
            picks[i] = idx
            if idx.size == 0:
                out[i] = np.zeros((len(states), 0))
                continue
            members_of.append(i)
            lat.append(batch.latitude_deg[idx])
            lon.append(batch.longitude_deg[idx])
            elev.append(batch.elevation_m[idx])
            level.append(batch.level_pa()[idx])
            variables.add(batch.variable)
            if (target == "simulated" and family == "column"
                    and batch.variable in ("wind_u_m_s", "wind_v_m_s")
                    and bool(np.any(batch.measurement[idx] == AMV_MEASUREMENT))):
                variables.add(WIND_SHEAR_VARIABLE)
        if not lat:
            continue
        values, ln_pressure = operators.evaluate(
            states, np.concatenate(lat), np.concatenate(lon), np.concatenate(elev), np.concatenate(level),
            variables=tuple(sorted(variables)))
        start = 0
        for i in members_of:
            batch = batches[i]
            idx = picks[i]
            sim = values[batch.variable][:, start:start + idx.size]
            lnp = ln_pressure[start:start + idx.size]
            shear = values.get(WIND_SHEAR_VARIABLE)
            shear = None if shear is None else shear[:, start:start + idx.size]
            start += idx.size
            out[i] = sim
            if target == "simulated":
                if batch.simulated is None:
                    batch.simulated = np.full((sim.shape[0], batch.count), np.nan)
                batch.simulated[:, idx] = sim
                surface = batch.surface[idx]
                batch.ln_pressure[idx[surface]] = lnp[surface]
                if shear is not None:
                    amv = batch.measurement[idx] == AMV_MEASUREMENT
                    if amv.any():
                        with np.errstate(invalid="ignore"):
                            batch.assignment_shear[idx[amv]] = np.nanmean(shear[:, amv], axis=0)
            elif target == "control_simulated":
                if batch.control_simulated is None:
                    batch.control_simulated = np.full((1, batch.count), np.nan)
                batch.control_simulated[:, idx] = sim[:1]
    return out


def batches_from_rows(rows: list[ObsRow], operators: MemberOperators, members) -> list[PointObs]:
    """The :class:`PointObs` batches of a row list, one per (source,
    variable), ``simulated`` and ``ln_pressure`` filled by ``operators``
    on ``members``; each batch's ``operator`` re-evaluates the same rows
    on another member list (the O-A pass)."""
    if not rows:
        return []
    lat = np.array([r.latitude_deg for r in rows])
    lon = np.array([r.longitude_deg for r in rows])
    elev = np.array([r.elevation_m for r in rows])
    level = np.array([np.nan if r.level_pa is None else r.level_pa for r in rows])
    is_ro = np.array([r.variable == "refractivity_n" for r in rows], dtype=bool)
    values = {name: np.full((len(members), len(rows)), np.nan) for name in OPERATOR_VARIABLES}
    ln_pressure = np.full(len(rows), np.nan)
    for family_mask, family_variables in (
            (~is_ro, tuple(sorted(COLUMN_VARIABLES))), (is_ro, ("refractivity_n",))):
        if not family_mask.any():
            continue
        part_values, part_lnp = operators.evaluate(
            members, lat[family_mask], lon[family_mask], elev[family_mask], level[family_mask],
            variables=family_variables)
        for name in family_variables:
            values[name][:, family_mask] = part_values[name]
        ln_pressure[family_mask] = part_lnp
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row.source, row.variable), []).append(index)
    batches = []
    for (source, variable), indices in groups.items():
        idx = np.asarray(indices)
        sim = values[variable][:, idx]
        finite = np.all(np.isfinite(sim), axis=0)
        if not finite.all():
            idx = idx[finite]
            sim = sim[:, finite]
        if idx.size == 0:
            continue
        batches.append(PointObs(
            stream=source, variable=variable,
            latitude_deg=lat[idx], longitude_deg=lon[idx],
            ln_pressure=ln_pressure[idx],
            surface=np.isnan(level[idx]),
            value=np.array([rows[i].value for i in idx]),
            error=np.array([rows[i].error for i in idx]),
            simulated=sim,
            elevation_m=elev[idx],
            identity=np.array([rows[i].identity_hash() for i in idx], dtype=object),
            valid_time=[rows[i].valid_time for i in idx],
            operator=operators.batch_operator,
            measurement=np.array([rows[i].measurement for i in idx], dtype=object),
        ))
    return batches
