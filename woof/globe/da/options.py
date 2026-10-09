"""Options of the WOOF global ensemble filter: the ensemble and the filter.

Two frozen dataclasses, both fully stated so a receipt can carry them by
:meth:`identity` and a door can build a configuration table from them.

:class:`EnsembleOptions` says what the resident ensemble IS: how many
members, at which truncation, how the initial perturbations are drawn
(spectral perturbations of the starting analysis scaled per level, plus
time-lagged differences of the analyses the caller hands over) and how
much additive inflation each cycle re-draws.

:class:`FilterOptions` says what one analysis DOES: which grid fields the
LETKF updates, the Gaspari-Cohn cutoffs (horizontal in kilometres, vertical
in ln p), the posterior relaxation, the observation-space quality control
(gross check against the background ensemble spread, thinning to one report
per grid cell), how the wind increment is balanced on its way into the
spectral state, and the cross-validation gate the v1 door established.
"""
from __future__ import annotations

import math
import dataclasses
from dataclasses import dataclass

#: The grid fields the filter may analyse.  ``u``/``v`` are the grid wind
#: (analysed back into vorticity and divergence through the vector
#: analysis; the default pair), ``psi``/``chi`` the grid streamfunction
#: and velocity potential (analysed back through the Laplacian; a
#: selectable arm with its loss named: on the T7 / T3 control twin with
#: perfect reports the potentials' analysis took the control's wind rmse
#: from 1.10 to 1.92 m/s over three analyses while the components' took it
#: down, because the LETKF's weights are solved column by column and the
#: wind is the derivative of the potential, so the weights' variation
#: from column to column differentiates into wind of the potential's own
#: size; the pointwise balance the potentials keep under an elementwise
#: covariance localisation, Kepert 2009, needs a model-space localisation
#: this filter does not run), ``theta`` the potential temperature, ``qv``
#: the vapor, ``lnps`` the log surface pressure (2-D).  The grid tracers
#: (condensate and number moments) are not in the list: no stream of the
#: first arm observes them and a station cannot tell the filter where a
#: cloud is; they stay each member's own.
ANALYSIS_FIELDS = ("u", "v", "psi", "chi", "theta", "qv", "lnps")
DEFAULT_ANALYSIS_FIELDS = ("u", "v", "theta", "qv", "lnps")

#: How the analysed wind increment enters the spectral state
#: (``assimilate.WIND_BALANCE_MODES``, the same two names): ``rotational``
#: keeps the vorticity part only, ``unconstrained`` applies vorticity and
#: divergence.  The v1 door keeps the rotational part because its scalar
#: spreading manufactured divergence; an ensemble increment is a linear
#: combination of member perturbations whose divergence is the members'
#: own, so which mode this filter wants is a measured question the global
#: OSSE answers (see the module doc).
WIND_BALANCE_MODES = ("rotational", "unconstrained")

RELAXATION_MODES = ("rtps", "rtpp")

#: Where the localised solve runs: ``auto`` (the members' own namespace,
#: the card when the model is on one), ``device`` (the card, refused by
#: name on a host backend), ``host`` (numpy on the host whatever the
#: members' namespace: the reference the device path is compared against,
#: the same code in the other array module).  Recorded in the receipt with
#: the path taken and its wall.
SOLVE_PATHS = ("auto", "device", "host")

#: How the point operators contract a device-resident state
#: (:data:`woof.globe.da.operators.OPERATOR_PRECISIONS`):
#: ``state`` at the state's own precision (float32 GEMMs over 256-term
#: blocks summed in float64 for a float32 state), ``float64`` always in
#: float64 (the host path's arithmetic to rounding).
OPERATOR_PRECISIONS = ("state", "float64")

#: How an increment enters the state: ``direct`` adds it at the analysis
#: instant; ``iau`` (incremental analysis update, the default since
#: 2026-09-06) stores it and the integration adds an equal portion before
#: each step of the window (the members over the next window, the control
#: over the window re-integrated from its start), so the atmosphere takes
#: the increment through its dynamics: the constant-weight window is a
#: low-pass filter on the increment's response whose first zero is at the
#: window length (a 3,600 s window passes a 2 h period at 0.64 and a 3 h
#: one at 0.83), and the first step after every analysis reads a
#: surface-pressure tendency 1.01 to 1.07 times the last one before it on
#: the record's arms against 1.44 for direct insertion of a planted 20 hPa
#: bump.
INCREMENT_APPLICATION_MODES = ("direct", "iau")

#: How the ensemble mean is recentred on the control analysis.
#: ``increment``: the ensemble-mean increment is replaced by the control's
#: increment truncated to the ensemble triangle (the members keep their own
#: terrain-consistent background and receive the control's analysis
#: increment), the default because the control's orography is not the
#: ensemble's and a truncated control STATE carries the finer terrain's
#: surface pressure and level structure onto the coarser grid (measured on
#: the T63 / T127 twin: the members' surface pressure rmse rose to 7 hPa
#: within the hour after every state recentring and the spread grew 1.4 to
#: 3.5 hPa in five cycles).  ``state``: the truncated control state
#: replaces the mean (the pre-measurement form, an experiment).
RECENTERING_MODES = ("increment", "state")

#: The ensemble weight of the hybrid covariance the door ships with.
#: Set by the graded six-cycle fresh arms of 2026-09-07: the value below
#: is the beta that graded best against the stations and the soundings
#: beside the GDAS cold start; 1.0 is the ensemble alone.
#: The package's own beta: the ensemble alone.  The fresh door's completed
#: system runs 0.75 (``woof.globe.da_door.DEFAULT_FRESH_HYBRID_BETA``,
#: 2026-09-06); a smoke or twin case on its own ladder keeps 1.0 unless it
#: names a table on that ladder.
DEFAULT_HYBRID_BETA = 1.0
#: How the initial (and additive) perturbations relate their mass fields to
#: their wind: ``linear`` derives the temperature and log surface pressure
#: from the drawn rotational wind through the linear balance equation and
#: the hydrostatic relation (the default since 2026-09-06, by measurement:
#: see :mod:`woof.globe.da.perturbations`); ``none`` draws the
#: three independently (the family before it, kept as the comparison arm).
PERTURBATION_BALANCE_MODES = ("linear", "none")

#: Which observation-error calibration the filter lays over the rows'
#: assigned errors: ``desroziers-2026-09-06`` is
#: :data:`woof.globe.da.observation_errors.DESROZIERS_ERROR_TABLE`;
#: ``None`` keeps every row's own error.
OBSERVATION_ERROR_CALIBRATIONS = ("desroziers-2026-09-06",)


def _coerce_option(raw, current, name: str):
    """A command-line string to the type of the field's current value."""
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if isinstance(current, bool):
        if text.lower() in ("true", "1", "on", "yes"):
            return True
        if text.lower() in ("false", "0", "off", "no"):
            return False
        raise ValueError(f"filter option {name} expects true or false, got {raw!r}")
    if isinstance(current, tuple):
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if text.lower() in ("none", "null"):
        return None
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(text))
    if isinstance(current, float) or current is None:
        return float(text)
    return text


def parse_filter_overrides(pairs) -> dict:
    """``["NAME=VALUE", ...]`` (the repeatable ``--filter-option``) to a
    mapping; a token without ``=`` is refused by name."""
    out: dict = {}
    for token in pairs or ():
        if "=" not in token:
            raise ValueError(f"--filter-option expects NAME=VALUE, got {token!r}")
        name, value = token.split("=", 1)
        out[name.strip()] = value.strip()
    return out


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")


@dataclass(frozen=True)
class EnsembleOptions:
    """What the resident ensemble is.

    members
        Ensemble size N.  32 by default; the filter refuses fewer than 3
        (two members have one perturbation direction and no covariance to
        speak of).  32 is the count the shared 32 GB card carries under a
        T255 control beside the other lanes' runs; the spread the filter
        needs comes from the family's amplitudes and the relaxation, not
        from a count the card cannot hold.
    truncation
        The ensemble's own spectral truncation (T127 by default, the
        deterministic member running at T255).  The ensemble grid is the
        dealiased Gaussian grid of that truncation.
    seed
        The one seed every random draw of the ensemble descends from
        (member k's initial perturbation of field f is drawn from a stream
        keyed on ``(seed, k, f)``, so a member is reproducible on its own).
    perturbation_*
        Amplitude of the initial spectral perturbations at the reference
        level, in the field's units: temperature (K, applied to theta
        through the local Exner function), wind (m/s, drawn as
        streamfunction and velocity potential so the wind is
        divergence-consistent), log surface pressure (dimensionless; 0.0008
        is about 0.8 hPa), vapor (relative, a fraction of the local vapor).
        The pressure and wind amplitudes of record (0.0008, 3.0 m/s) are
        the calibrated-spread criterion read on the 2026-09-01 case: the
        station and buoy pressure innovations leave a background error of
        60 to 80 Pa where the family before them spread 130 to 150, and the
        radiosonde wind innovations leave 3 m/s aloft where it spread 1.6.
        The vertical profile of each amplitude is the climatological table
        in :mod:`woof.globe.da.perturbations`.
    perturbation_max_degree
        Spectral degree above which the random perturbations carry no
        power (``None``: half the truncation).  The spectrum below it is
        red, ``n^perturbation_spectral_slope`` in power per degree.
    lagged_difference_weight
        Weight of the time-lagged analysis differences the caller hands
        over (``GlobalEnsemble.from_state(lagged_states=...)``): each
        difference, scaled to the random perturbations' amplitude, is added
        to a member with this weight and a random sign.  0 disables.
    additive_inflation_fraction
        Fraction of the initial perturbation amplitude re-drawn and added
        to every member after each analysis (additive inflation, Mitchell
        and Houtekamer 2000).  0 by default (amendment D: ONE inflation
        configuration first, RTPS; additions are tested one at a time
        because several adaptive mechanisms hide each other).
    perturbation_balance
        :data:`PERTURBATION_BALANCE_MODES`.  ``linear``: the temperature
        and the log surface pressure of a draw are the ones the linear
        balance equation and the hydrostatic relation give the drawn
        rotational wind, plus ``perturbation_unbalanced_fraction`` of the
        independent temperature and pressure draws; the vorticity draw is
        built on ``perturbation_vertical_modes`` smooth vertical modes
        instead of level by level.  ``none``: the three fields drawn
        independently.
    perturbation_unbalanced_fraction
        The share of the independent temperature and log-surface-pressure
        draws (``perturbation_temperature_k`` and
        ``perturbation_ln_surface_pressure`` times this) added to the
        balanced ones under ``linear``: the part of a forecast error the
        balance does not describe, and the tropics' mass spread.
    perturbation_vertical_modes
        The number of vertical modes (cosines in the column's normalised
        ln p, the barotropic mode first, amplitudes falling as
        ``(1 + m^2)^(-1/2)``) the balanced wind draw is built on under
        ``linear``; the draw is smooth in ln p so its thermal wind is
        bounded (an autoregression in level index or in ln p has an
        unbounded ln p derivative as the layers thin, and made a 1.5 K
        temperature spread out of a 1.5 hPa pressure one on the T127
        column).
    perturbation_peak_degree
        The total degree the balanced wind draw's kinetic energy per
        degree peaks at (rising as n^2 below it, falling as n^-3 above;
        12 is a 3,300 km wavelength).  The ``n^slope`` weights of the
        independent family put nine tenths of a vorticity draw's energy
        at n = 1, and the mass field a planetary wind balances is ten
        hectopascals of surface pressure.
    perturbation_internal_modes
        The number of jet-level internal vertical modes (squared sines in
        the normalised ln p between 70 hPa and the surface, every one zero
        with zero gradient at the surface and at the top; a low-level mode
        on the column below 500 hPa rides beside them) the balanced
        draw's second part is built on under ``linear``; that part
        carries wind and temperature aloft and no surface pressure, and
        is scaled so the whole draw's wind reaches
        ``perturbation_wind_m_s`` after the cosine-mode part has been
        scaled to ``perturbation_ln_surface_pressure``.  With 0 the
        family is the cosine modes alone, whose wind and temperature
        follow the pressure amplitude: on the record's T127 column 1.5
        hPa carried 0.39 K and 1.6 m/s where the innovations put the
        background error at 0.7 hPa, 0.8 K and 3 m/s.
    perturbation_internal_peak_degree
        The total degree the internal part's kinetic energy per degree
        peaks at (30, a 1,300 km wavelength, against the external part's
        ``perturbation_peak_degree``): the balanced temperature of a
        given shear grows with the horizontal scale, and at the external
        part's scale the internal modes carried 1.3 to 1.8 K through the
        troposphere for 3 m/s of wind on the T63 probe column.
    perturbation_tropical_wind_fraction
        The balanced wind draw's amplitude at the equator as a fraction of
        its amplitude at the poles (``sqrt(f^2 + (1 - f^2) sin^2 lat)``
        applied on the grid); 0.5 by default, the storm tracks against
        the tropics as the record's motion-vector and radiosonde
        innovations read them.
    """

    members: int = 32
    truncation: int = 127
    seed: int = 20260906
    perturbation_temperature_k: float = 1.0
    perturbation_wind_m_s: float = 3.0
    perturbation_ln_surface_pressure: float = 0.0008
    perturbation_vapor_relative: float = 0.10
    perturbation_max_degree: int | None = None
    perturbation_spectral_slope: float = -3.0
    lagged_difference_weight: float = 1.0
    additive_inflation_fraction: float = 0.0
    perturbation_balance: str = "linear"
    perturbation_unbalanced_fraction: float = 0.3
    perturbation_vertical_modes: int = 6
    perturbation_peak_degree: float = 12.0
    perturbation_internal_modes: int = 4
    perturbation_internal_peak_degree: float = 30.0
    perturbation_tropical_wind_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.perturbation_balance not in PERTURBATION_BALANCE_MODES:
            raise ValueError(f"perturbation_balance must be one of {PERTURBATION_BALANCE_MODES}")
        if isinstance(self.perturbation_internal_modes, bool) or int(self.perturbation_internal_modes) < 0:
            raise ValueError("perturbation_internal_modes must be a whole number of at least 0")
        tropical = float(self.perturbation_tropical_wind_fraction)
        if not math.isfinite(tropical) or not 0.0 < tropical <= 1.0:
            raise ValueError("perturbation_tropical_wind_fraction must lie in (0, 1]")
        _positive(float(self.perturbation_internal_peak_degree), "perturbation_internal_peak_degree")
        fraction = float(self.perturbation_unbalanced_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("perturbation_unbalanced_fraction must lie in [0, 1]")
        if isinstance(self.perturbation_vertical_modes, bool) or int(self.perturbation_vertical_modes) < 1:
            raise ValueError("perturbation_vertical_modes must be a positive whole number")
        _positive(float(self.perturbation_peak_degree), "perturbation_peak_degree")
        if isinstance(self.members, bool) or int(self.members) != self.members or self.members < 3:
            raise ValueError(
                "members must be a whole number of at least 3: two members "
                "span one perturbation direction and carry no covariance "
                "a filter can use"
            )
        if isinstance(self.truncation, bool) or int(self.truncation) != self.truncation or self.truncation < 1:
            raise ValueError("truncation must be a positive whole number")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a nonnegative whole number")
        for name in (
            "perturbation_temperature_k", "perturbation_wind_m_s",
            "perturbation_ln_surface_pressure", "perturbation_vapor_relative",
        ):
            _nonnegative(float(getattr(self, name)), name)
        if self.perturbation_max_degree is not None:
            degree = int(self.perturbation_max_degree)
            if degree < 1 or degree > self.truncation:
                raise ValueError(
                    "perturbation_max_degree must lie in [1, truncation]"
                )
        if not math.isfinite(self.perturbation_spectral_slope):
            raise ValueError("perturbation_spectral_slope must be finite")
        _nonnegative(float(self.lagged_difference_weight), "lagged_difference_weight")
        _nonnegative(float(self.additive_inflation_fraction), "additive_inflation_fraction")

    @property
    def max_degree(self) -> int:
        if self.perturbation_max_degree is None:
            return max(1, int(self.truncation) // 2)
        return int(self.perturbation_max_degree)

    def identity(self) -> dict[str, object]:
        return {
            "members": int(self.members),
            "truncation": int(self.truncation),
            "seed": int(self.seed),
            "perturbation_temperature_k": float(self.perturbation_temperature_k),
            "perturbation_wind_m_s": float(self.perturbation_wind_m_s),
            "perturbation_ln_surface_pressure": float(self.perturbation_ln_surface_pressure),
            "perturbation_vapor_relative": float(self.perturbation_vapor_relative),
            "perturbation_max_degree": int(self.max_degree),
            "perturbation_spectral_slope": float(self.perturbation_spectral_slope),
            "lagged_difference_weight": float(self.lagged_difference_weight),
            "additive_inflation_fraction": float(self.additive_inflation_fraction),
            "perturbation_balance": self.perturbation_balance,
            "perturbation_unbalanced_fraction": float(self.perturbation_unbalanced_fraction),
            "perturbation_vertical_modes": int(self.perturbation_vertical_modes),
            "perturbation_peak_degree": float(self.perturbation_peak_degree),
            "perturbation_internal_modes": int(self.perturbation_internal_modes),
            "perturbation_internal_peak_degree": float(self.perturbation_internal_peak_degree),
            "perturbation_tropical_wind_fraction": float(self.perturbation_tropical_wind_fraction),
        }


@dataclass(frozen=True)
class FilterOptions:
    """What one LETKF analysis does.

    analysis_fields
        Which grid fields are updated (:data:`ANALYSIS_FIELDS`; the
        default :data:`DEFAULT_ANALYSIS_FIELDS` analyses the wind as its
        components).
    horizontal_cutoff_km
        Gaspari-Cohn full-support radius on the sphere, kilometres: a report
        this far from an analysis column contributes exactly nothing there
        (the ``2c`` of Gaspari and Cohn, as :func:`woof.da.letkf.gaspari_cohn`
        names it).
    vertical_cutoff_lnp
        Gaspari-Cohn full-support separation in ln p between a report's
        pressure and the analysis level's, for an aloft temperature report.
    aloft_wind_vertical_cutoff_lnp, aloft_humidity_vertical_cutoff_lnp
        The same for an aloft wind report (a sounding level, a motion
        vector) and an aloft humidity report.
    surface_vertical_cutoff_lnp
        The same for a surface report of temperature or dewpoint (a 2 m
        sample of the boundary layer).
    surface_wind_vertical_cutoff_lnp
        The same for a 10 m wind report.
    pressure_vertical_cutoff_lnp
        The same for a surface-pressure report; ``None`` means no vertical
        localisation.  Every cutoff above is the Gaspari-Cohn support
        fitted to the mean vertical correlation the balanced ensemble
        carries for that class (the derivation leg ``woof global da
        localisation``, :mod:`woof.globe.da.localisation`; the
        values here are the 2026-09-06 derivation on the 2026-09-01 case,
        recorded in the module doc of that module).  The pressure cutoff
        was ``None`` ("a column's mass responds hydrostatically as a
        whole") until that derivation showed the unbalanced ensemble's
        pressure-to-column correlations were sampling noise above the
        boundary layer and a surface-pressure report was reaching the
        250 hPa wind through them.  The 6.00 it carries is the UPPER BOUND
        of the derivation's scan (``localisation.CUTOFF_BOUNDS_LNP``), not
        a value the profile chose: the balanced family's pressure signal
        is column-deep, the fit asked for more than the whole column, and
        at 6.00 a surface-pressure report still reaches 250 hPa at
        Gaspari-Cohn weight 0.72 and 100 hPa at 0.40, so the class is
        localised in name and reaches the column in fact; the receipt of
        the derivation says ``cutoff_at_bound: "upper"`` for it.
    refractivity_vertical_cutoff_lnp
        The same for a radio-occultation refractivity row (anchored at its
        tangent height; the retrieval's dry pressure is its position in
        the ln p metric).  Tighter than the sounding cutoff (1.0 against
        1.5: a half-width of 0.5 in ln p, about 3.5 km) because the
        retrieval resolves a few hundred metres and the door writes a row
        every 200 m, so the neighbouring rows carry the column's shape and
        a wide lens would smear the tropopause's refractivity kink across
        the levels either side of it.
    amv_height_assignment_sigma_pa
        The height-assignment uncertainty of a satellite motion vector
        (its ``measurement`` is ``amv_assigned_pressure``), in pascals:
        the operators read the member-mean wind change across this
        pressure interval either side of the assigned pressure and the
        filter adds it in quadrature to the row's assigned error (the
        situation-dependent error of Forsythe and Saunders 2008: a vector
        in a sheared layer is worth less than one in a barotropic layer,
        because a 100 hPa miss in its height is a wind error of the
        shear across those 100 hPa).  0 by default, the inflation off:
        it is selectable (10,000 Pa is the interval of record) because on
        the case day of record the assigned motion-vector error, 3.9 to
        4.0 m/s, already exceeded the Desroziers-diagnosed 2.6 m/s and the
        shear term took the consistency ratio from 0.45 to 0.21 while the
        package carrying it lost at the stations at 18 h.
    relaxation, rtps_alpha, prior_inflation
        The posterior relaxation of :mod:`woof.da.letkf` (RTPS by default,
        alpha 0.9) and Hunt's multiplicative prior inflation rho.
    background_check_sigmas
        Observation-space gross check: a report whose innovation exceeds
        this many standard deviations of ``sqrt(error^2 + spread_H^2)``,
        the spread being the background ensemble's spread in observation
        space, is rejected and counted.  4.0 rejects six in a hundred
        thousand well-calibrated innovations.
    thinning
        One report per analysis grid cell per stream and variable (per
        level bin for aloft reports): the report nearest the cell centre is
        kept.  What it prevents: a dense network's reports at one cell
        entering the local solve as hundreds of near-identical rows whose
        errors are treated as independent.
    max_local_obs
        Ceiling on the reports one analysis column gathers; the largest
        weights are kept and the count dropped is recorded.  3,000 since
        2026-09-06: at 400 the record's analyses dropped 24 to 35 million
        local pairs per cycle, a column over the station networks kept its
        nearest 80 cells and the effective horizontal localisation there
        was 300 km against the stated 1,200, so the weights varied from
        column to column at that scale and the increment they assembled
        carried a 500 hPa height field whose geostrophic wind was
        correlated 0.0 with the wind increment over CONUS against 0.3
        elsewhere (the record's 00Z increment).  A column inside the
        networks gathers 3,000 to 3,900 reports at 1,200 km on the case's
        tables, so the cap no longer sets the reach.
    memory_budget_mib
        The device budget the LETKF sizes its column chunks by (2,048 MiB
        since the cap rose; the sizing is pessimistic by a factor two).
    wind_balance
        :data:`WIND_BALANCE_MODES`.
    withheld_fraction, withheld_seed, gate_minimum_count
        The v1 door's cross-validation gate, unchanged: a seeded fraction of
        each gated variable's reports is judged, never analysed, and O-A
        rms must fall below O-B rms on them.
    maximum_age_s, future_tolerance_s
        The v1 door's age window.
    humidity_pressure_floor_pa
        A sounding's dewpoint rows above this pressure (lower pressure,
        higher up) are rejected by name (``humidity_above_floor``):
        sonde humidity does not read the dry upper troposphere and
        stratosphere and the model's dewpoint there is its vapor floor.
        ``None`` keeps every level.  Default 30,000 Pa.
    chunk_columns
        Latitude rings of analysis columns per gathered chunk (``None``:
        sized from ``memory_budget_mib`` once the local report count is
        known); the levels of a chunk are solved in batches sized from the
        same budget.
    solve_dtype, eigensolver
        As :class:`woof.da.letkf.LetkfConfig`.
    solve_path
        :data:`SOLVE_PATHS`: where the localised solve runs.
    operator_precision
        :data:`OPERATOR_PRECISIONS`: how the point operators contract a
        device-resident state.
    preserve_global_mean_pressure
        Add the constant to each member's ln ps that keeps its global-mean
        surface pressure at the background's (the v1 door's rule: the mass
        fixer owns that mean).
    transfer_taper_start_degree, transfer_taper_end_degree
        The smooth spectral taper the CONTROL increment carries on its way
        from the ensemble grid into the control's triangle (amendment C:
        the coarse ensemble corrects the scales it demonstrably represents
        and no others).  Weight one up to the start degree, a raised cosine
        to zero at the end degree, zero above.  ``None`` takes 0.6 and 1.0
        times the ensemble truncation; the OSSE calibrates them against
        the ensemble error spectrum and records the increment spectrum
        after every analysis.
    recentering_fraction
        How far the ensemble mean moves onto the control analysis at
        recentering: 1 replaces the mean (full recentring), 0 leaves it,
        a value between is partial recentring (operational precedent; an
        experiment of the OSSE).
    recentering_mode
        :data:`RECENTERING_MODES`.
    increment_application
        :data:`INCREMENT_APPLICATION_MODES`.
    iau_window_s
        The window the IAU distributes the increment over (the cycle
        length by default, 3600 s).
    desroziers_minimum_count
        Reports a (stream, variable) needs before its Desroziers ratios are
        judged in the statistical-consistency assessment.
    observation_error_calibration
        :data:`OBSERVATION_ERROR_CALIBRATIONS` or ``None``: the
        per-(stream, variable) observation-error table laid over the rows'
        assigned errors before quality control and the solve (the
        Desroziers estimates on the record; the receipt carries both).
    spread_ratio_band
        The band the ensemble spread over the Desroziers estimate of the
        background error (both in observation space, per stream and
        variable) is judged in by the statistical assessment.
    hybrid_beta
        The weight of the localised ensemble covariance in the hybrid
        background covariance ``beta B_ens + (1 - beta) B_static`` of
        amendment A (one positive-semidefinite representation, never
        several band analyses summed; :mod:`woof.globe.da.static_covariance`
        and the module doc of :mod:`woof.globe.da.letkf_point`).
        Below one the control's gain is solved in the augmented space of
        the members and ``static_samples`` draws of the static covariance
        table ``static_covariance`` names; a beta below one with no table
        is refused by name (an analysis that accepted it would run on the
        ensemble alone while its receipt claimed a hybrid).
    static_covariance
        The static covariance table: ``packaged`` (the table shipped under
        ``woof/data``, the lagged-forecast estimate of record), a path to
        a table written by ``woof global da static-covariance``, or
        ``None`` (no table; ``hybrid_beta`` must then be 1).
    static_samples
        The draws K from the static covariance per analysis (each analysis
        draws afresh from a stream keyed on ``static_seed`` and the cycle,
        so the draws' sampling noise does not repeat cycle to cycle).
    static_seed
        The seed those draws descend from.
    """

    analysis_fields: tuple[str, ...] = DEFAULT_ANALYSIS_FIELDS
    horizontal_cutoff_km: float = 1200.0
    vertical_cutoff_lnp: float = 3.43
    aloft_wind_vertical_cutoff_lnp: float = 5.07
    aloft_humidity_vertical_cutoff_lnp: float = 2.11
    surface_vertical_cutoff_lnp: float = 1.06
    surface_wind_vertical_cutoff_lnp: float = 0.70
    pressure_vertical_cutoff_lnp: float | None = 6.00
    refractivity_vertical_cutoff_lnp: float = 1.0
    amv_height_assignment_sigma_pa: float = 0.0
    relaxation: str = "rtps"
    rtps_alpha: float = 0.9
    prior_inflation: float = 1.0
    background_check_sigmas: float = 4.0
    thinning: bool = True
    max_local_obs: int = 3000
    wind_balance: str = "rotational"
    withheld_fraction: float = 0.1
    withheld_seed: int = 0
    gate_minimum_count: int = 50
    maximum_age_s: float = 5400.0
    future_tolerance_s: float = 600.0
    humidity_pressure_floor_pa: float | None = 30000.0
    chunk_columns: int | None = None
    memory_budget_mib: float = 2048.0
    solve_dtype: str = "float64"
    eigensolver: str = "auto"
    solve_path: str = "auto"
    operator_precision: str = "state"
    preserve_global_mean_pressure: bool = True
    transfer_taper_start_degree: int | None = None
    transfer_taper_end_degree: int | None = None
    recentering_fraction: float = 1.0
    recentering_mode: str = "increment"
    increment_application: str = "iau"
    iau_window_s: float = 3600.0
    desroziers_minimum_count: int = 30
    observation_error_calibration: str | None = "desroziers-2026-09-06"
    spread_ratio_band: tuple[float, float] = (0.5, 2.0)
    hybrid_beta: float = DEFAULT_HYBRID_BETA
    static_covariance: str | None = "packaged"
    static_samples: int = 64
    static_seed: int = 20260907

    def __post_init__(self) -> None:
        fields = tuple(self.analysis_fields)
        if not fields:
            raise ValueError("analysis_fields is empty: an analysis that updates nothing is a bug")
        unknown = [f for f in fields if f not in ANALYSIS_FIELDS]
        if unknown:
            raise ValueError(
                f"analysis_fields {unknown} are not analysable grid fields; "
                f"the fields are {list(ANALYSIS_FIELDS)}"
            )
        if len(set(fields)) != len(fields):
            raise ValueError(f"analysis_fields has duplicates: {fields}")
        if ("u" in fields) != ("v" in fields):
            raise ValueError(
                "u and v are analysed together or not at all: the wind "
                "increment is analysed into vorticity and divergence as a pair"
            )
        if ("psi" in fields) != ("chi" in fields):
            raise ValueError(
                "psi and chi are analysed together or not at all: the wind "
                "increment is analysed into vorticity and divergence as a pair"
            )
        if "psi" in fields and "u" in fields:
            raise ValueError(
                "the wind is analysed once, as psi and chi or as u and v, not as both"
            )
        object.__setattr__(self, "analysis_fields", fields)
        _positive(float(self.horizontal_cutoff_km), "horizontal_cutoff_km")
        _positive(float(self.vertical_cutoff_lnp), "vertical_cutoff_lnp")
        _positive(float(self.aloft_wind_vertical_cutoff_lnp), "aloft_wind_vertical_cutoff_lnp")
        _positive(float(self.aloft_humidity_vertical_cutoff_lnp), "aloft_humidity_vertical_cutoff_lnp")
        _positive(float(self.surface_vertical_cutoff_lnp), "surface_vertical_cutoff_lnp")
        _positive(float(self.surface_wind_vertical_cutoff_lnp), "surface_wind_vertical_cutoff_lnp")
        if self.pressure_vertical_cutoff_lnp is not None:
            _positive(float(self.pressure_vertical_cutoff_lnp), "pressure_vertical_cutoff_lnp")
        _positive(float(self.refractivity_vertical_cutoff_lnp), "refractivity_vertical_cutoff_lnp")
        _nonnegative(float(self.amv_height_assignment_sigma_pa), "amv_height_assignment_sigma_pa")
        if (self.observation_error_calibration is not None
                and self.observation_error_calibration not in OBSERVATION_ERROR_CALIBRATIONS):
            raise ValueError(
                f"observation_error_calibration must be None or one of {OBSERVATION_ERROR_CALIBRATIONS}"
            )
        band = tuple(float(v) for v in self.spread_ratio_band)
        if len(band) != 2 or not all(math.isfinite(v) for v in band) or not 0.0 < band[0] < band[1]:
            raise ValueError("spread_ratio_band must be (low, high) with 0 < low < high")
        object.__setattr__(self, "spread_ratio_band", band)
        if self.relaxation not in RELAXATION_MODES:
            raise ValueError(f"relaxation must be one of {RELAXATION_MODES}")
        alpha = float(self.rtps_alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("rtps_alpha must lie in [0, 1]")
        _positive(float(self.prior_inflation), "prior_inflation")
        _positive(float(self.background_check_sigmas), "background_check_sigmas")
        if isinstance(self.max_local_obs, bool) or int(self.max_local_obs) < 1:
            raise ValueError("max_local_obs must be a positive whole number")
        if self.wind_balance not in WIND_BALANCE_MODES:
            raise ValueError(f"wind_balance must be one of {WIND_BALANCE_MODES}")
        if not 0.0 < float(self.withheld_fraction) < 0.5:
            raise ValueError("withheld_fraction must lie in (0, 0.5)")
        if isinstance(self.withheld_seed, bool) or int(self.withheld_seed) < 0:
            raise ValueError("withheld_seed must be a nonnegative whole number")
        if int(self.gate_minimum_count) < 1:
            raise ValueError("gate_minimum_count must be >= 1")
        _positive(float(self.maximum_age_s), "maximum_age_s")
        _positive(float(self.future_tolerance_s), "future_tolerance_s")
        if self.chunk_columns is not None and int(self.chunk_columns) < 1:
            raise ValueError("chunk_columns must be >= 1 or None")
        _positive(float(self.memory_budget_mib), "memory_budget_mib")
        if self.solve_dtype not in ("float32", "float64"):
            raise ValueError("solve_dtype must be float32 or float64")
        if self.eigensolver not in ("auto", "jacobi", "library"):
            raise ValueError("eigensolver must be auto, jacobi or library")
        if self.solve_path not in SOLVE_PATHS:
            raise ValueError(f"solve_path must be one of {SOLVE_PATHS}")
        if self.operator_precision not in OPERATOR_PRECISIONS:
            raise ValueError(f"operator_precision must be one of {OPERATOR_PRECISIONS}")
        for name in ("transfer_taper_start_degree", "transfer_taper_end_degree"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) < 0):
                raise ValueError(f"{name} must be a nonnegative whole number or None")
        if (self.transfer_taper_start_degree is not None and self.transfer_taper_end_degree is not None
                and int(self.transfer_taper_end_degree) < int(self.transfer_taper_start_degree)):
            raise ValueError("transfer_taper_end_degree must be at or above transfer_taper_start_degree")
        fraction = float(self.recentering_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("recentering_fraction must lie in [0, 1]")
        if self.increment_application not in INCREMENT_APPLICATION_MODES:
            raise ValueError(f"increment_application must be one of {INCREMENT_APPLICATION_MODES}")
        if self.recentering_mode not in RECENTERING_MODES:
            raise ValueError(f"recentering_mode must be one of {RECENTERING_MODES}")
        _positive(float(self.iau_window_s), "iau_window_s")
        if isinstance(self.desroziers_minimum_count, bool) or int(self.desroziers_minimum_count) < 2:
            raise ValueError("desroziers_minimum_count must be a whole number of at least 2")
        beta = float(self.hybrid_beta)
        if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
            raise ValueError("hybrid_beta must lie in (0, 1]")
        if self.static_covariance is not None and not str(self.static_covariance).strip():
            raise ValueError("static_covariance must name a table (packaged or a path) or be None")
        if beta < 1.0 and self.static_covariance is None:
            raise ValueError(
                f"hybrid_beta {beta} asks for a static covariance's share of the gain and "
                "static_covariance names no table; an analysis that accepted it would run on the "
                "localised ensemble covariance alone while its receipt claimed a hybrid, so name the "
                "packaged table or a path"
            )
        if isinstance(self.static_samples, bool) or int(self.static_samples) < 1:
            raise ValueError("static_samples must be a positive whole number")
        if isinstance(self.static_seed, bool) or int(self.static_seed) < 0:
            raise ValueError("static_seed must be a nonnegative whole number")

    def taper_degrees(self, ensemble_truncation: int) -> tuple[int, int]:
        """``(start, end)`` of the transfer taper for an ensemble at
        ``ensemble_truncation``: the options' values, or 0.6 and 1.0 times
        the truncation."""
        t = int(ensemble_truncation)
        start = self.transfer_taper_start_degree
        end = self.transfer_taper_end_degree
        start = int(round(0.6 * t)) if start is None else int(start)
        end = t if end is None else int(end)
        return max(0, min(start, end)), max(start, end)

    def with_overrides(self, overrides) -> "FilterOptions":
        """This options object with the named fields replaced from a
        ``{name: value}`` mapping whose values may be command-line strings
        (``woof global da ... --filter-option NAME=VALUE``): a float field
        takes a float, an int field an int, a bool ``true``/``false``, an
        optional field ``none``, a tuple field a comma-separated list; a
        name this dataclass does not carry is refused with the field list,
        because a misspelt option that silently did nothing would be
        reported as a setting of the run.  The validation of
        ``__post_init__`` runs on the result."""
        if not overrides:
            return self
        fields = {f.name: f for f in dataclasses.fields(self)}
        values = {}
        for name, raw in dict(overrides).items():
            if name not in fields:
                raise ValueError(
                    f"unknown filter option {name!r}; FilterOptions carries {sorted(fields)}")
            values[name] = _coerce_option(raw, getattr(self, name), name)
        return dataclasses.replace(self, **values)

    def vertical_cutoff_for(self, variable: str, surface: bool) -> float | None:
        """The ln p cutoff a report of ``variable`` carries by class: the
        pressure cutoff for a surface-pressure report, the surface wind
        cutoff for a 10 m wind, the surface cutoff for a 2 m temperature
        or dewpoint, the aloft wind, humidity or temperature cutoff for a
        report aloft."""
        wind = variable in ("wind_u_m_s", "wind_v_m_s")
        if variable == "surface_pressure_pa":
            return self.pressure_vertical_cutoff_lnp
        if variable == "refractivity_n":
            return float(self.refractivity_vertical_cutoff_lnp)
        if surface:
            return float(self.surface_wind_vertical_cutoff_lnp if wind else self.surface_vertical_cutoff_lnp)
        if wind:
            return float(self.aloft_wind_vertical_cutoff_lnp)
        if variable == "dewpoint_k":
            return float(self.aloft_humidity_vertical_cutoff_lnp)
        return float(self.vertical_cutoff_lnp)

    def identity(self) -> dict[str, object]:
        return {
            "analysis_fields": list(self.analysis_fields),
            "horizontal_cutoff_km": float(self.horizontal_cutoff_km),
            "vertical_cutoff_lnp": float(self.vertical_cutoff_lnp),
            "aloft_wind_vertical_cutoff_lnp": float(self.aloft_wind_vertical_cutoff_lnp),
            "aloft_humidity_vertical_cutoff_lnp": float(self.aloft_humidity_vertical_cutoff_lnp),
            "surface_vertical_cutoff_lnp": float(self.surface_vertical_cutoff_lnp),
            "surface_wind_vertical_cutoff_lnp": float(self.surface_wind_vertical_cutoff_lnp),
            "pressure_vertical_cutoff_lnp": (
                None if self.pressure_vertical_cutoff_lnp is None
                else float(self.pressure_vertical_cutoff_lnp)
            ),
            "refractivity_vertical_cutoff_lnp": float(self.refractivity_vertical_cutoff_lnp),
            "amv_height_assignment_sigma_pa": float(self.amv_height_assignment_sigma_pa),
            "relaxation": self.relaxation,
            "rtps_alpha": float(self.rtps_alpha),
            "prior_inflation": float(self.prior_inflation),
            "background_check_sigmas": float(self.background_check_sigmas),
            "thinning": bool(self.thinning),
            "max_local_obs": int(self.max_local_obs),
            "wind_balance": self.wind_balance,
            "withheld_fraction": float(self.withheld_fraction),
            "withheld_seed": int(self.withheld_seed),
            "gate_minimum_count": int(self.gate_minimum_count),
            "maximum_age_s": float(self.maximum_age_s),
            "future_tolerance_s": float(self.future_tolerance_s),
            "humidity_pressure_floor_pa": None if self.humidity_pressure_floor_pa is None else float(self.humidity_pressure_floor_pa),
            "solve_dtype": self.solve_dtype,
            "eigensolver": self.eigensolver,
            "solve_path": self.solve_path,
            "operator_precision": self.operator_precision,
            "preserve_global_mean_pressure": bool(self.preserve_global_mean_pressure),
            "transfer_taper_start_degree": self.transfer_taper_start_degree,
            "transfer_taper_end_degree": self.transfer_taper_end_degree,
            "recentering_fraction": float(self.recentering_fraction),
            "recentering_mode": self.recentering_mode,
            "increment_application": self.increment_application,
            "iau_window_s": float(self.iau_window_s),
            "desroziers_minimum_count": int(self.desroziers_minimum_count),
            "observation_error_calibration": self.observation_error_calibration,
            "spread_ratio_band": [float(v) for v in self.spread_ratio_band],
            "hybrid_beta": float(self.hybrid_beta),
            "static_covariance": None if self.static_covariance is None else str(self.static_covariance),
            "static_samples": int(self.static_samples),
            "static_seed": int(self.static_seed),
        }


__all__ = [
    "parse_filter_overrides",
    "ANALYSIS_FIELDS",
    "DEFAULT_HYBRID_BETA",
    "DEFAULT_ANALYSIS_FIELDS",
    "INCREMENT_APPLICATION_MODES",
    "OBSERVATION_ERROR_CALIBRATIONS",
    "PERTURBATION_BALANCE_MODES",
    "OPERATOR_PRECISIONS",
    "RECENTERING_MODES",
    "RELAXATION_MODES",
    "SOLVE_PATHS",
    "WIND_BALANCE_MODES",
    "EnsembleOptions",
    "FilterOptions",
]
