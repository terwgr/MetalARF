"""The ATMS operator entry for the ensemble filter.

An operator entry is what the scorecard promotes when a channel's
clear-sky over-ocean O-B rmse after the bias correction is inside the
bar: the admitted channels, each with its observation error, its bias
coefficients and the day they were fitted on, its vertical position and
extent, the calibration receipt that preceded the numbers, and the
acceptance contract every operator ships with (measurement definition,
time, location, vertical coordinate, representativeness, bias treatment,
error correlations).  A channel outside the bar is refused with the term
it is closest to.  ``entry_from_scorecard`` returns None when no sounding
channel is admitted.

The filter side (:class:`woof.globe.da.observations.PointObs`):
one batch per satellite and admitted channel (:func:`stream_name`),
``variable`` :data:`VARIABLE`, one row per thinned clear-sky ocean cell,
the row's ``ln_pressure`` the channel's weighting-function centroid (for
thinning and the receipt) and its ``localisation_profile`` the channel's
weighting function convolved with the Gaspari-Cohn kernel of the vertical
length measured on the ensemble (:func:`channel_profile`,
:mod:`woof.globe.da.localisation`: the model-space placement the
filter reads at every level).  The batch's ``operator`` is
:class:`AtmsBatchOperator`: it synthesizes every state's column at the
row's position on the level set the O-B calibration was measured on (the
41 GDAS isobaric levels, linear in ln p from the state's hybrid levels,
held at the top and the surface), runs the radiative transfer on the
reference column and the tangent-linear transfer about it for the rest
(:data:`LINEARISATION_RULE`), and applies the channel's bias correction
with the state's own brightness temperature and the row's viewing
geometry as predictors.

Every row carries its cell's mean time, so a report at 12:08 is compared
with the member trajectory at 12:08 when the filter retains
observation-space trajectories through the window; the cell's own
identity (satellite, bin, ring, longitude index, channel) is the
assimilation-chain key.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .calibrate import STANDARD_LEVELS_PA, standard_column
from .channels import CHANNELS, TEMPERATURE_SOUNDING_CHANNELS
from .rte import Column, brightness_temperature, build_layers, temperature_jacobian, weighting_function

ENTRY_SCHEMA = "gpuwm-arwen-global-microwave-operator-entry-v1"
STREAM = "atms-clear-ocean"
VARIABLE = "brightness_temperature_k"
OPERATOR_PATH = "woof.globe.microwave.entry:AtmsBatchOperator"

#: The level set the member columns are interpolated onto: the GDAS
#: isobaric levels the O-B calibration was measured on.
OPERATOR_LEVELS_PA = STANDARD_LEVELS_PA


@dataclass(frozen=True)
class AcceptanceContract:
    measurement: str
    time: str
    location: str
    vertical_coordinate: str
    representativeness: str
    bias_treatment: str
    error_correlations: str
    assimilation_status: str


CONTRACT = AcceptanceContract(
    measurement=(
        "ATMS SDR antenna brightness temperature (IDPS, the remapped-to-scene product is not used), "
        "one thinned cell: the mean over the beams of one 0.25-degree cell in one time bin, clear sky "
        "over open ocean by the Grody 23.8/31.4 GHz liquid-water retrieval and the analysis cloud "
        "water path, |latitude| <= 60, zenith <= 60 degrees, at least three beams"
    ),
    time="the mean beam time of the cell (UTC, unix seconds); granule start, end and creation times in the fetch manifest",
    location="the mean beam latitude and longitude of the cell; the beam footprint is 2.2 degrees (about 32 km at nadir) for channels 3 to 16",
    vertical_coordinate=(
        "the row's vertical localisation is a profile: the channel's temperature Jacobian per unit ln p "
        "on the standard column at the row's zenith, convolved with the Gaspari-Cohn kernel of the "
        "vertical length measured on the ensemble at every window, read by the filter at every analysis "
        "level (model space); ln p of the weighting-function centroid names the row for thinning and the "
        "receipt, and vertical_cutoff_lnp carries the measured length. The member operator "
        "holds the member's top-level temperature above the model top (1 hPa on the default stack): "
        "1.8 percent of channel 13's weight and 8.3 percent of channel 14's lie above it on the "
        "standard column at nadir (29.6 percent of the refused channel 15's), so those channels "
        "read the held top temperature for that share"
    ),
    representativeness="the within-cell standard deviation of each channel is carried per cell as the spread read; cells above the per-channel spread limit are not scored and not offered",
    bias_treatment=(
        "per channel, O - B = a + b (B - mean_B) + c (sec z - 1), fitted on the even half of the day's "
        "clear-sky ocean cells in time order and scored on the odd half; the coefficients and the day are in the entry; "
        "the correction is applied to H(x) in the operator, never to the report; in the cycle the slope and scan "
        "terms move toward each window's own fit by rows / (rows + 2000) and the constant a stays the entry's "
        "(the anchor: a constant refitted on the control's own departures would absorb the control's drift)"
    ),
    error_correlations=(
        "channels are offered as separate batches with diagonal errors; inter-channel correlation of the "
        "residuals is not modelled (the Desroziers diagnostic is the place it will be measured); the error per "
        "channel is sqrt(NEdT^2 + rmse_after^2), never the NEdT alone"
    ),
    assimilation_status=(
        "assimilated by the ensemble filter through the atms stream (woof.globe.radiance_streams): "
        "clear-sky open-ocean cells thinned to the analysis grid and the observation bin, screened by the "
        "control background's cloud water and the retrieved liquid water, localised in model space through "
        "the channel's weighting-function profile with the vertical length measured on the ensemble each "
        "window, the slope and scan terms of the bias correction moved by the window's own departures with the "
        "constant anchored to the entry's fit; whether the stream is in the "
        "door's default set is the scorecard's verdict, recorded on the door page"
    ),
)


@dataclass
class ChannelEntry:
    channel: int
    centre_ghz: float
    nedt_k: float
    error_k: float
    bias_coefficients: dict[str, float]
    #: The weighting function's centroid pressure on the standard column.
    peak_pressure_pa: float
    vertical_cutoff_lnp: float
    rmse_after_k: float
    noise_floor_k: float
    score_cells: int


@dataclass
class OperatorEntry:
    name: str
    satellite: str
    day: str
    admitted_channels: list[int]
    channels: list[ChannelEntry]
    refused: dict[str, str]
    calibration_passes: dict[str, bool]
    analyses: list[dict]
    fitted_utc: str
    schema: str = ENTRY_SCHEMA
    stream: str = STREAM
    variable: str = VARIABLE
    operator: str = OPERATOR_PATH
    levels_pa: list[float] = field(default_factory=lambda: [float(p) for p in OPERATOR_LEVELS_PA])
    contract: AcceptanceContract = CONTRACT

    def channel_entry(self, number: int) -> ChannelEntry:
        for entry in self.channels:
            if entry.channel == number:
                return entry
        raise KeyError(f"channel {number} is not admitted by {self.name}")


def channel_vertical(number: int, zenith_deg: float = 0.0, *, coverage: float = 0.90) -> tuple[float, float]:
    """``(centre_pressure_pa, cutoff_lnp)`` of a channel on the standard
    column: the weight-weighted mean ln p of the atmospheric weighting
    function (its centroid; the discrete layer of largest weight is
    quantised by the refinement and ties between neighbouring channels)
    and the ln p distance from that centre that holds ``coverage`` of the
    weight (symmetric, the larger side)."""
    column = standard_column()
    p_hpa, weights, _ = weighting_function(column, number, zenith_deg)
    w = weights[:, 0]
    p = p_hpa[:, 0] * 100.0
    if w.sum() <= 0.0:
        raise ValueError(f"channel {number} carries no atmospheric weight on the standard column")
    ln_p = np.log(p)
    centre = float(np.sum(w * ln_p) / np.sum(w))
    order = np.argsort(ln_p)
    cumulative = np.cumsum(w[order]) / w.sum()
    lo = ln_p[order][np.searchsorted(cumulative, (1.0 - coverage) / 2.0)]
    hi = ln_p[order][min(np.searchsorted(cumulative, 1.0 - (1.0 - coverage) / 2.0), order.size - 1)]
    cutoff = max(abs(centre - lo), abs(hi - centre), 0.05)
    return float(np.exp(centre)), float(cutoff)


def channel_error_k(nedt_k: float, rmse_after_k: float, noise_floor_k: float | None = None,
                    beams: float | np.ndarray | None = None) -> float | np.ndarray:
    """The observation error of a thinned cell: the scored residual after
    correction with the scorecard's own cell noise taken out
    (``sqrt(max(rmse_after^2 - noise_floor^2, 0))``, the part above the
    noise) put back together with the cell's own noise, ``NEdT /
    sqrt(beams)`` when the cell's beam count is known and the scorecard's
    noise floor otherwise (then the error is the scored residual itself).

    Without a noise floor the rule is the one the entries of 2026-09-06
    were built with, ``sqrt(NEdT^2 + rmse_after^2)``.  What that rule got
    wrong: the NEdT is one beam's noise and the residual was scored on cell
    means of three beams and more (a noise floor of 0.10 to 0.15 K against
    a 0.5 K NEdT), so the cell's noise was counted once inside the residual
    and again, five times larger, beside it; on the run of record the
    Desroziers diagnosis read the assigned error of channels 6 to 11 at two
    to five times the diagnosed (0.52 to 1.24 K assigned, 0.06 to 0.37 K
    diagnosed), the sounding channels weighted a tenth of what their
    departures say."""
    if noise_floor_k is None:
        return float(np.sqrt(nedt_k ** 2 + rmse_after_k ** 2))
    above = np.sqrt(max(float(rmse_after_k) ** 2 - float(noise_floor_k) ** 2, 0.0))
    if beams is None:
        return float(np.sqrt(above ** 2 + float(noise_floor_k) ** 2))
    noise = float(nedt_k) / np.sqrt(np.maximum(np.asarray(beams, dtype=np.float64), 1.0))
    error = np.sqrt(above ** 2 + noise ** 2)
    return float(error) if np.ndim(error) == 0 else error


def entry_from_scorecard(document: dict, calibration: dict, *, satellite: str | None = None,
                         day: str | None = None) -> OperatorEntry | None:
    """Promote the scorecard's admitted sounding channels into an entry, or
    None when no sounding channel is inside the bar."""
    verdict = document["verdict"]
    admitted = [int(c) for c in verdict.get("admitted_channels", [])]
    if not admitted:
        return None
    thinned = document.get("provenance", {}).get("thinned", {})
    decoded = str(thinned.get("decoded_dir", ""))
    if satellite is None:
        satellite = next((code for code in ("noaa-20", "noaa-21") if code in decoded), "unknown")
    if day is None:
        span = document.get("provenance", {}).get("analysis_span") or ["", ""]
        day = span[0][:10] if span[0] else "unknown"
    by_channel = {int(c["channel"]): c for c in document["channels"]}
    channels = []
    for number in admitted:
        score = by_channel[number]
        peak_pa, cutoff = channel_vertical(number)
        rmse_after = float(score["geometry_corrected"]["rmse"])
        channels.append(ChannelEntry(
            channel=number,
            centre_ghz=float(score["centre_ghz"]),
            nedt_k=float(score["nedt_k"]),
            error_k=channel_error_k(float(score["nedt_k"]), rmse_after, float(score["noise_floor_k"])),
            bias_coefficients=dict(score["geometry_coefficients"]),
            peak_pressure_pa=peak_pa,
            vertical_cutoff_lnp=cutoff,
            rmse_after_k=rmse_after,
            noise_floor_k=float(score["noise_floor_k"]) if score.get("noise_floor_k") is not None else float("nan"),
            score_cells=int(score["score_cells"]),
        ))
    refused = {str(k): v for k, v in verdict.get("nearest_term", {}).items()}
    return OperatorEntry(
        name=f"{STREAM}:{satellite}:{day}",
        satellite=satellite,
        day=day,
        admitted_channels=admitted,
        channels=channels,
        refused=refused,
        calibration_passes=dict(calibration.get("passes", {})),
        analyses=list(document.get("provenance", {}).get("analyses", [])),
        fitted_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def write_entry(path: str | Path, entry: OperatorEntry | None) -> None:
    """The entry as JSON, or the refusal record when there is none."""
    if entry is None:
        payload = {"schema": ENTRY_SCHEMA, "operator_entry_ships": False,
                   "reason": "no temperature-sounding channel inside the bar"}
    else:
        payload = {"operator_entry_ships": True, **asdict(entry)}
    Path(path).write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")


def read_entry(path: str | Path) -> OperatorEntry | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != ENTRY_SCHEMA:
        raise ValueError(f"{path}: schema {payload.get('schema')!r} is not {ENTRY_SCHEMA}")
    if not payload.get("operator_entry_ships"):
        return None
    contract = AcceptanceContract(**payload.pop("contract"))
    channels = [ChannelEntry(**c) for c in payload.pop("channels")]
    payload.pop("operator_entry_ships")
    return OperatorEntry(channels=channels, contract=contract, **payload)


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if dataclasses.is_dataclass(value):
        return asdict(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


# --------------------------------------------------------------------------
# Member columns and the batch operator


def state_columns(states, transform, vertical, latitude_deg, longitude_deg, *,
                  levels_pa=OPERATOR_LEVELS_PA, member_chunk: int = 8) -> list[Column]:
    """One :class:`Column` batch per state at the points: temperature and
    specific humidity on ``levels_pa`` (linear in ln p from the state's
    full levels, held beyond the span), the state's surface pressure and
    its skin temperature (bilinear from the surface plane).  No 2 m
    temperature: the surface layer takes the lowest level.  ``transform``
    must be the states' own (their truncation); on a device backend the
    coefficient stack of every state is one device array and the point
    synthesis runs there (the path the point operators take)."""
    from woof.globe.spectral.sampling import sample_scalar

    from ..assimilate import _interp_ln_pressure, _sample_grid, _to_numpy_spectral
    from ..constants import KAPPA, REFERENCE_PRESSURE_PA

    lat = np.asarray(latitude_deg, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude_deg, dtype=np.float64).reshape(-1)
    n = lat.size
    a = np.asarray(vertical.a_half_pa, dtype=np.float64)
    b = np.asarray(vertical.b_half, dtype=np.float64)
    nlev = a.size - 1
    target = np.asarray(levels_pa, dtype=np.float64)
    ln_target = np.log(target)
    backend = transform.backend
    grid = transform.grid
    xp = backend.xp
    device = xp is not np

    def coeff(atmosphere, name):
        return getattr(atmosphere, name) if device else _to_numpy_spectral(backend, getattr(atmosphere, name))

    step = len(states) if device else max(1, int(member_chunk))
    columns: list[Column] = []
    for start in range(0, len(states), step):
        chunk = states[start:start + step]
        stack = xp.stack([
            xp.concatenate([
                coeff(m.atmosphere, "theta"),
                coeff(m.atmosphere, "qv"),
                coeff(m.atmosphere, "log_surface_pressure")[None],
            ]) for m in chunk
        ])                                                     # (Rc, 2 nlev + 1, n, m)
        sampled = np.asarray(sample_scalar(transform, stack, lat, lon))    # (Rc, 2 nlev + 1, n)
        del stack
        for c, member in enumerate(chunk):
            ps = np.exp(sampled[c, -1])
            p_half = a[:, None] + b[:, None] * ps[None, :]
            p_full = np.sqrt(p_half[:-1] * p_half[1:])         # (nlev, n)
            ln_p = np.log(p_full)
            t_full = sampled[c, :nlev] * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
            q_full = np.maximum(sampled[c, nlev:2 * nlev], 0.0)
            temperature = np.empty((target.size, n))
            humidity = np.empty((target.size, n))
            for k, lt in enumerate(ln_target):
                temperature[k] = _interp_ln_pressure(t_full, ln_p, np.full(n, lt))
                humidity[k] = _interp_ln_pressure(q_full, ln_p, np.full(n, lt))
            skin = np.asarray(backend.to_numpy(member.surface.temperature_k), dtype=np.float64)
            columns.append(Column(
                pressure_pa=target,
                temperature_k=temperature,
                specific_humidity=humidity,
                surface_pressure_pa=ps,
                skin_temperature_k=_sample_grid(skin, grid, lat, lon),
                air_temperature_2m_k=None,
            ))
    return columns


def member_columns(members, transform, vertical, latitude_deg, longitude_deg, *,
                   levels_pa=OPERATOR_LEVELS_PA, member_chunk: int = 8) -> list[Column]:
    """:func:`state_columns` under its first name."""
    return state_columns(members, transform, vertical, latitude_deg, longitude_deg,
                         levels_pa=levels_pa, member_chunk=member_chunk)


def mean_column(columns: list[Column]) -> Column:
    """The member-mean column of a batch of columns on one level set."""
    return Column(
        pressure_pa=columns[0].pressure_pa,
        temperature_k=np.mean([c.temperature_k for c in columns], axis=0),
        specific_humidity=np.mean([c.specific_humidity for c in columns], axis=0),
        surface_pressure_pa=np.mean([c.surface_pressure_pa for c in columns], axis=0),
        skin_temperature_k=np.mean([c.skin_temperature_k for c in columns], axis=0),
        air_temperature_2m_k=None,
    )


def make_identity(satellite: str, cell_bin: int, cell_j: int, cell_i: int, channel: int) -> str:
    return f"atms:{satellite}:{int(cell_bin)}:{int(cell_j)}:{int(cell_i)}:ch{int(channel):02d}"


def stream_name(satellite: str, channel: int) -> str:
    """One stream name per satellite and channel (``atms-clear-ocean:noaa-21:ch07``),
    so the receipt's O-B, O-A and Desroziers readings are per channel."""
    return f"{STREAM}:{satellite}:ch{int(channel):02d}"


def truncation_of(states) -> int:
    """The spectral truncation of a state list (from the coefficient shape)."""
    theta = states[0].atmosphere.theta
    shape = getattr(theta, "shape", None) or np.asarray(theta).shape
    return int(shape[-1]) - 1


LINEARISATION_RULE = (
    "the first evaluation of a row on a state set takes the full radiative transfer on the "
    "reference column (the members' mean column, or the single state's own) and keeps that "
    "column's layer temperatures, its temperature Jacobian per refined layer and its "
    "brightness temperature; every evaluation of the row after it on the same truncation "
    "(the members about their mean, the analysed members and the control analysis about "
    "their backgrounds) is the tangent-linear transfer about that reference, the absorption "
    "held; the residual of the linear form against the full transfer is measured on a "
    "subsample every window (linearisation_check) and carried here, per channel over every "
    "member evaluation of the window (the record of one call alone carried one channel)"
)


@dataclass
class AtmsBatchOperator:
    """``operator(states, batch) -> (R, n)`` for the filter: the states'
    columns at the batch rows, the radiative transfer for each row's
    channel at its viewing geometry, the entry's bias correction applied
    to H(x).  Geometry is looked up by row identity so any subset of a
    batch evaluates.  The transform is chosen by the STATES' truncation
    (``bind`` one per resolution: the ensemble's and the control's), so
    the same operator answers for the members, the control and the
    analysed copies (``evaluates_states``); a state list of several
    members is evaluated with the tangent-linear transfer about the
    member mean (:data:`LINEARISATION_RULE`), a single state with the
    full transfer.  ``coefficients`` are the live bias coefficients per
    channel (the entry's at construction, moved by the stream's day
    update and recorded)."""

    entry: OperatorEntry
    vertical: object
    #: identity -> (channel, zenith_deg, scan_angle_deg)
    geometry: dict[str, tuple[int, float, float]]
    transforms: dict[int, object] = field(default_factory=dict)
    coefficients: dict[int, dict[str, float]] | None = None
    member_chunk: int = 8
    linearise_members: bool = True
    #: The fraction of rows and members the full transfer is re-run on
    #: after a linearised evaluation, to measure the residual.
    check_rows: int = 64
    check_members: int = 2
    #: The window's linearisation record: every member evaluation of the
    #: window (one call per satellite and channel, the members' background
    #: and their analysis alike) adds its subsample residual, so the record
    #: the stream reads at the next window carries every channel.  What it
    #: prevents: one record per call, overwritten by the next, left the
    #: receipt with channel 14 alone on every window of the case day while
    #: the check had run on all eleven channels and thrown ten away.
    last_linearisation_check: dict | None = None
    _window_checks: list = field(default_factory=list, repr=False)
    calls: list = field(default_factory=list)
    #: (truncation, identity) -> (reference layer temperatures (nlay,), the
    #: Jacobian (nlay,), the reference brightness temperature): the full
    #: transfer's reading a later evaluation of the row linearises about.
    references: dict = field(default_factory=dict, repr=False)

    #: The flag :func:`woof.globe.da.operators.evaluate_batches` reads.
    evaluates_states = True

    def __post_init__(self) -> None:
        if self.coefficients is None:
            self.coefficients = {int(c.channel): dict(c.bias_coefficients) for c in self.entry.channels}

    def reset(self) -> None:
        """Forget the references (a new window's rows are new identities;
        the cache is cleared so it does not grow across the cycle)."""
        self.references.clear()
        self.calls.clear()
        # The finished window's record stays readable (the stream reads it
        # after the reset); the next call starts the new window's.
        self._window_checks = []

    def bind(self, transform) -> "AtmsBatchOperator":
        self.transforms[int(transform.truncation)] = transform
        return self

    def transform_for(self, states):
        t = truncation_of(states)
        found = self.transforms.get(t)
        if found is None:
            raise ValueError(
                f"the ATMS operator holds no transform at T{t} (bound: {sorted(self.transforms)}); bind the "
                "ensemble's and the control's transforms before the window opens"
            )
        return found

    def _rows(self, batch):
        identities = [str(i) for i in batch.identity]
        try:
            rows = [self.geometry[i] for i in identities]
        except KeyError as exc:
            raise ValueError(f"row {exc} of the batch carries no viewing geometry; the batch must come from "
                             "point_obs_from_cells") from exc
        channels = np.asarray([r[0] for r in rows], dtype=int)
        zenith = np.asarray([r[1] for r in rows], dtype=np.float64)
        scan = np.asarray([r[2] for r in rows], dtype=np.float64)
        return channels, zenith, scan

    def __call__(self, states, batch) -> np.ndarray:
        channels, zenith, scan = self._rows(batch)
        transform = self.transform_for(states)
        t = int(transform.truncation)
        columns = state_columns(states, transform, self.vertical, batch.latitude_deg,
                                batch.longitude_deg, member_chunk=self.member_chunk)
        out = np.empty((len(states), batch.count))
        identities = [str(i) for i in batch.identity]
        if not self.linearise_members:
            for k, column in enumerate(columns):
                out[k] = self.simulate_column(column, channels, zenith, scan)
            self.calls.append({"states": len(states), "rows": int(batch.count), "truncation": t, "mode": "full"})
            return out
        cached = np.array([(t, i) in self.references for i in identities], dtype=bool)
        raw = np.empty((len(states), batch.count))
        if (~cached).any():
            rows = np.flatnonzero(~cached)
            part = [_subcolumn(c, rows) for c in columns]
            reference = mean_column(part) if len(states) > 1 else part[0]
            layers_ref, jac, tb_ref = self._reference(reference, channels[rows], zenith[rows], scan[rows])
            for j, r in enumerate(rows):
                self.references[(t, identities[r])] = (layers_ref.t_k[:, j].copy(), jac[:, j].copy(), float(tb_ref[j]))
            if len(states) == 1:
                raw[0, rows] = tb_ref
            else:
                for k in range(len(states)):
                    t_k = build_layers(part[k]).t_k
                    raw[k, rows] = tb_ref + np.sum(jac * (t_k - layers_ref.t_k), axis=0)
        if cached.any():
            rows = np.flatnonzero(cached)
            t_ref = np.stack([self.references[(t, identities[r])][0] for r in rows], axis=1)
            jac = np.stack([self.references[(t, identities[r])][1] for r in rows], axis=1)
            tb_ref = np.array([self.references[(t, identities[r])][2] for r in rows])
            for k in range(len(states)):
                t_k = build_layers(_subcolumn(columns[k], rows)).t_k
                raw[k, rows] = tb_ref + np.sum(jac * (t_k - t_ref), axis=0)
        for number in np.unique(channels):
            rows = np.flatnonzero(channels == number)
            out[:, rows] = raw[:, rows] + self.bias(int(number), raw[:, rows], zenith[rows][None, :])
        mode = "full" if (len(states) == 1 and not cached.any()) else "linearised"
        if len(states) > 1:
            self._check(columns, channels, zenith, scan, raw)
        self.calls.append({"states": len(states), "rows": int(batch.count), "truncation": t, "mode": mode,
                           "rows_from_reference": int(cached.sum())})
        return out

    def _reference(self, column: Column, channels: np.ndarray, zenith: np.ndarray, scan: np.ndarray):
        """The full transfer on a reference column batch: its refined layers,
        the temperature Jacobian per layer ``(nlay, n)`` and the raw
        brightness temperature ``(n,)`` per row of the batch."""
        layers_ref = None
        jac_all = None
        tb_all = np.empty(column.ncol)
        for number in np.unique(channels):
            rows = np.flatnonzero(channels == number)
            part = _subcolumn(column, rows)
            layers, jac, tb = temperature_jacobian(part, (int(number),), zenith[rows], scan[rows])
            if layers_ref is None:
                layers_ref = build_layers(column)
                jac_all = np.zeros((layers_ref.t_k.shape[0], column.ncol))
            jac_all[:, rows] = jac[0]
            tb_all[rows] = tb[0]
        return layers_ref, jac_all, tb_all

    def _check(self, columns, channels, zenith, scan, raw) -> None:
        """The residual of the linear form on a subsample: the full
        transfer on a few members and rows against the values handed back."""
        n = raw.shape[1]
        r = len(columns)
        rng = np.random.default_rng(int(n) * 7919 + r)
        rows = np.sort(rng.choice(n, size=min(int(self.check_rows), n), replace=False))
        members = np.sort(rng.choice(r, size=min(int(self.check_members), r), replace=False))
        diff = []
        for k in members:
            full = self.simulate_column(_subcolumn(columns[k], rows), channels[rows], zenith[rows], scan[rows],
                                        correct=False)
            diff.append(full - raw[k, rows])
        diff = np.concatenate(diff) if diff else np.zeros(0)
        by_channel = {}
        if diff.size:
            ch_rows = np.tile(channels[rows], len(members))
            for number in np.unique(ch_rows):
                d = diff[ch_rows == number]
                by_channel[int(number)] = {"sum_sq_k2": float(np.sum(d ** 2)), "max_abs_k": float(np.max(np.abs(d))),
                                           "count": int(d.size)}
        self._window_checks.append({"members_checked": [int(k) for k in members], "rows_checked": int(rows.size),
                                    "by_channel": by_channel})
        self.last_linearisation_check = self._window_record()

    def _window_record(self) -> dict:
        """The window's linearisation record over every call so far: per
        channel the rms and the largest residual over all its calls, the
        overall rms over every pair checked, the calls and rows behind it."""
        merged: dict[int, dict] = {}
        total_sq = 0.0
        total_n = 0
        rows_checked = 0
        members: set[int] = set()
        for call in self._window_checks:
            rows_checked += int(call["rows_checked"])
            members.update(call["members_checked"])
            for number, stat in call["by_channel"].items():
                m = merged.setdefault(int(number), {"sum_sq_k2": 0.0, "max_abs_k": 0.0, "count": 0})
                m["sum_sq_k2"] += float(stat["sum_sq_k2"])
                m["max_abs_k"] = max(m["max_abs_k"], float(stat["max_abs_k"]))
                m["count"] += int(stat["count"])
                total_sq += float(stat["sum_sq_k2"])
                total_n += int(stat["count"])
        by_channel = {number: {"rms_k": float(np.sqrt(m["sum_sq_k2"] / m["count"])) if m["count"] else None,
                               "max_abs_k": m["max_abs_k"], "count": m["count"]}
                      for number, m in sorted(merged.items())}
        return {
            "rule": LINEARISATION_RULE,
            "calls": len(self._window_checks),
            "members_checked": sorted(members), "rows_checked": int(rows_checked),
            "rms_k": float(np.sqrt(total_sq / total_n)) if total_n else None,
            "max_abs_k": max((m["max_abs_k"] for m in merged.values()), default=None),
            "by_channel": by_channel,
        }

    def simulate_column(self, column: Column, channels: np.ndarray, zenith: np.ndarray,
                        scan: np.ndarray, *, correct: bool = True) -> np.ndarray:
        """H(x) per row of one state's column batch, the bias correction
        applied unless ``correct`` is False."""
        values = np.empty(column.ncol)
        for number in np.unique(channels):
            rows = np.flatnonzero(channels == number)
            part = _subcolumn(column, rows)
            tb = brightness_temperature(part, (int(number),), zenith[rows], scan[rows])[0]
            values[rows] = tb + (self.bias(int(number), tb, zenith[rows]) if correct else 0.0)
        return values

    def bias(self, number: int, background_k: np.ndarray, zenith_deg: np.ndarray) -> np.ndarray:
        c = self.coefficients[int(number)]
        sec = 1.0 / np.cos(np.deg2rad(np.asarray(zenith_deg, dtype=np.float64)))
        return c["a"] + c["b"] * (np.asarray(background_k) - c["mean_background_k"]) + c["c"] * (sec - 1.0)


def _subcolumn(column: Column, rows) -> Column:
    return Column(
        pressure_pa=column.pressure_pa,
        temperature_k=column.temperature_k[:, rows],
        specific_humidity=column.specific_humidity[:, rows],
        surface_pressure_pa=column.surface_pressure_pa[rows],
        skin_temperature_k=column.skin_temperature_k[rows],
        air_temperature_2m_k=None if column.air_temperature_2m_k is None else column.air_temperature_2m_k[rows],
    )


# --------------------------------------------------------------------------
# The vertical localisation profile of a channel

#: Zenith bins the channel profiles are tabulated at (degrees, the bin
#: centre nearest a row's zenith is read).
PROFILE_ZENITH_BINS_DEG = (0.0, 15.0, 30.0, 45.0, 60.0)


def channel_weighting_layers(number: int, zenith_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """``(layer_lnp, weights)`` of the channel's temperature Jacobian per
    refined layer on the standard column at ``zenith_deg`` (the layer
    weights per unit ln p, the measure the profile convolves)."""
    column = standard_column()
    layers, jac, _tb = temperature_jacobian(column, (int(number),), float(zenith_deg), min(float(zenith_deg), 50.0))
    lnp = np.log(layers.p_hpa[:, 0] * 100.0)
    thickness = np.asarray(layers.thickness_lnp)[:, 0]
    w = np.abs(jac[0, :, 0])
    density = np.where(thickness > 0.0, w / np.where(thickness > 0.0, thickness, 1.0), 0.0)
    return lnp, density


def channel_profile(number: int, zenith_deg: float, cutoff_lnp: float) -> np.ndarray:
    """``(M,)`` the localisation profile of a channel at a zenith on
    :data:`~woof.globe.da.observations.LOCALISATION_AXIS_LNP`
    (:func:`woof.globe.da.localisation.profile_from_weighting_function`)."""
    from ..da.localisation import profile_from_weighting_function

    lnp, weights = channel_weighting_layers(number, zenith_deg)
    return profile_from_weighting_function(lnp, weights, cutoff_lnp)


def point_obs_from_cells(entry: OperatorEntry, cells, *, transforms, vertical, channels=None,
                         satellite: str | None = None, cutoffs_lnp: dict[int, float] | None = None,
                         coefficients: dict[int, dict[str, float]] | None = None,
                         operator: AtmsBatchOperator | None = None) -> list:
    """One :class:`PointObs` per admitted channel from thinned cells
    (:class:`~woof.globe.microwave.atms_bridge.Thinned` or any
    object with its arrays).  The caller screens the cells first; every
    row offered here is taken as clear sky over ocean.  ``transforms`` are
    the transforms the operator answers for (the ensemble's and the
    control's); ``cutoffs_lnp`` the vertical localisation length per
    channel (the measured Gaspari-Cohn zero; the entry's
    ``vertical_cutoff_lnp`` when a channel is not named), from which every
    row's localisation profile is built at its zenith
    (:func:`channel_profile`); ``coefficients`` the live bias coefficients
    (the entry's when None).  ``simulated`` is left for the operator:
    ``batch.operator(states, batch)`` fills it.  One operator serves every
    batch (``operator`` when given, so a stream keeps its bias state)."""
    from ..da.observations import PointObs

    satellite = satellite or entry.satellite
    wanted = [int(c) for c in (channels or entry.admitted_channels) if int(c) in entry.admitted_channels]
    n = int(np.asarray(cells.lat_mean_deg).size)
    lat = np.asarray(cells.lat_mean_deg, dtype=np.float64)
    lon = np.mod(np.asarray(cells.lon_mean_deg, dtype=np.float64), 360.0)
    zenith = np.asarray(cells.zenith_mean_deg, dtype=np.float64)
    scan = np.asarray(cells.scan_angle_abs_mean_deg, dtype=np.float64)
    t = np.asarray(cells.time_mean_unix_s, dtype=np.float64)
    tb = np.asarray(cells.tb_mean_k, dtype=np.float64)
    valid_time = [dt.datetime.fromtimestamp(float(s), tz=dt.timezone.utc) for s in t]
    if operator is None:
        operator = AtmsBatchOperator(entry=entry, vertical=vertical, geometry={}, coefficients=coefficients)
    for transform in transforms:
        operator.bind(transform)
    geometry = operator.geometry
    bins = np.asarray(PROFILE_ZENITH_BINS_DEG)
    zbin = np.abs(zenith[:, None] - bins[None, :]).argmin(axis=1)
    # The cell's beam count, when the cells carry one: the row's error is
    # then the entry's residual above the noise with this cell's own noise.
    beams = getattr(cells, "count", None)
    beams = None if beams is None else np.asarray(beams, dtype=np.float64)
    batches = []
    for number in wanted:
        channel_entry = entry.channel_entry(number)
        cutoff = float((cutoffs_lnp or {}).get(number, channel_entry.vertical_cutoff_lnp))
        if beams is not None and np.isfinite(channel_entry.noise_floor_k) and channel_entry.noise_floor_k > 0.0:
            error = np.asarray(channel_error_k(channel_entry.nedt_k, channel_entry.rmse_after_k,
                                               channel_entry.noise_floor_k, beams), dtype=np.float64)
        else:
            error = np.full(n, channel_entry.error_k)
        profiles = {k: channel_profile(number, float(bins[k]), cutoff) for k in np.unique(zbin)}
        identity = np.array([
            make_identity(satellite, cells.cell_bin[i], cells.cell_j[i], cells.cell_i[i], number)
            for i in range(n)
        ], dtype=object)
        for i in range(n):
            geometry[str(identity[i])] = (number, float(zenith[i]), float(scan[i]))
        value = tb[:, number - 1]
        finite = np.isfinite(value)
        keep = np.flatnonzero(finite)
        if keep.size == 0:
            continue
        batches.append(PointObs(
            stream=stream_name(satellite, number), variable=entry.variable,
            latitude_deg=lat[keep], longitude_deg=lon[keep],
            ln_pressure=np.full(keep.size, np.log(channel_entry.peak_pressure_pa)),
            surface=np.zeros(keep.size, dtype=bool),
            value=value[keep],
            error=error[keep],
            identity=identity[keep],
            valid_time=[valid_time[i] for i in keep],
            vertical_cutoff_lnp=cutoff,
            localisation_profile=np.stack([profiles[int(zbin[i])] for i in keep]),
            operator=operator,
        ))
    return batches


__all__ = [
    "CONTRACT", "ENTRY_SCHEMA", "LINEARISATION_RULE", "OPERATOR_LEVELS_PA", "PROFILE_ZENITH_BINS_DEG",
    "STREAM", "VARIABLE",
    "AcceptanceContract", "AtmsBatchOperator", "ChannelEntry", "OperatorEntry",
    "channel_error_k", "channel_profile", "channel_vertical", "channel_weighting_layers",
    "entry_from_scorecard", "make_identity", "mean_column", "member_columns", "point_obs_from_cells",
    "read_entry", "state_columns", "stream_name", "truncation_of", "write_entry",
]
