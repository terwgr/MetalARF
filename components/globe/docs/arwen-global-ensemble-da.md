# WOOF global ensemble data assimilation: interface notes

Package `src/arwen_global/da/` (2026-09-06).
This page is the record of every interface decision the ensemble core
took, in the order it took them, so the door lane and the observation
lanes build against committed code and not against assumptions. Numbers
appear here only when they were measured; a section without numbers is
a design statement. The design amendments adopted on 2026-09-06 (A to K)
override the earlier decisions where the two disagree; section 2 marks
each such change.

## 1. The contract in one paragraph

A door builds the ensemble with `GlobalEnsemble.from_state` (a base state
at the ensemble truncation, N members perturbed around it) or reads one
back with `GlobalEnsemble.read`; steps it through the window with
`advance_to`, handing an `ObservationWindow` as the observer so every
report is compared with the member state at its own time bin; hands
`analyze_ensemble` the ensemble, the window's `PointObs` batches and a
`ControlBackground` (the T255 state, model, transform and config);
receives an `EnsembleAnalysis` whose `control_analysis` is the control's
own analysis (formed from the control's innovation through the ensemble
covariance, the increment tapered by degree and embedded in the control's
triangle), whose `ensemble` holds the analysed members (or, under the
incremental analysis update, the background with the increments pending),
and whose `report` is the receipt (O-B and O-A distributions per stream,
variable and region for the ensemble mean and the control, Desroziers
ratios, the four assessments); recentres the members on the control
analysis with `recenter(ensemble, control_analysis, det_transform,
fraction=options.recentering_fraction, mode=options.recentering_mode,
control_increment=result.control_increment_spectral,
mean_increment=result.mean_increment_spectral)` (by increment, decision
18). Options are two frozen
dataclasses, `EnsembleOptions` and `FilterOptions`, both carried into
every receipt by `identity()`. `apply_mean_increment` still exists and is
the OSSE's comparison experiment, not the analysis path.

## 2. Decisions

1. **Point observations, not gridded ones.** `woof.da.letkf.analyze`
   takes observations gridded on the model grid and gathers a fixed index
   stencil per gridpoint. On a Gaussian grid that stencil is wrong twice:
   longitude is periodic (the regional stencil has no wrap) and the
   zonal spacing shrinks to under a kilometre at the pole-most ring, so a
   1,200 km radius sized on the minimum spacing spans the whole ring. The
   global filter therefore gathers a ragged neighbour list per analysis
   column (geodesic distance, periodic longitude), pads it to the column
   chunk's widest count, and runs the same transform. What is shared with
   the regional module by import: `gaspari_cohn`, the eigensolver
   selection (`_resolve_eigensolver`, `_eigendecompose`, this project's
   Jacobi kernel on the device), the allocation-failure recognition, and
   the closed-form inactive point at rho = 1 (an increment beyond every
   cutoff is bitwise zero). What differs: the gather, the metric, and the
   vertical coordinate.
2. **Vertical metric is ln p.** A report's vertical position is its
   `ln_pressure`; an analysis gridpoint's is the column's `ln p_full` at
   that level; the 2-D field ln ps sits at `ln ps`. A surface report is
   placed at the model's ln ps at the station by the operator (its
   `ln_pressure` arrives NaN and is filled). Cutoffs are per report class
   (`FilterOptions.vertical_cutoff_for`): aloft 1.5, surface 2 m / 10 m
   0.6, surface pressure none (the column's mass responds hydrostatically
   as a whole). These are the stated starting values; the OSSE sweeps
   them.
3. **Analysis fields are grid fields**: `u`, `v`, `theta`, `qv` (3-D) and
   `lnps` (2-D), synthesised from every member's spectral state. The
   increments return to the spectral state through the transform's
   forward analysis (theta, qv, lnps) and the vector analysis (u, v into
   vorticity and divergence); the triangular truncation is the increment
   smoothing, the same route the v1 door takes. The grid tracers
   (condensate, number moments) are not analysed: no stream of the first
   arm observes them.
4. **Wind balance is a stated mode with the v1 door's two names.** The
   v1 door keeps the rotational part because its scalar spreading
   manufactured divergence; an ensemble increment is a linear combination
   of member perturbations whose divergence is the members' own. The mode
   is `FilterOptions.wind_balance`, default `rotational` until the global
   OSSE has measured both (the `wind_balance` sweep of section 4). Amendment
   D forbids a blanket mid-latitude balance in the tropics; the rotational
   mode is a blanket rule by construction and the receipt's physical
   assessment says so; the divergent share of every increment's kinetic
   energy is recorded whichever mode runs.
5. **Members share one model.** The transform tables, the vertical
   coordinate, the surface geopotential and the physics bridge are one
   object; members step sequentially. Safe because every step-to-step
   quantity of the native suite lives in the member's `physics_state`
   (restart is bit for bit). The native runtime caches the frozen-column
   mask from the first batch it sees; the members share the surface
   statics and the analysed sea ice (the filter perturbs neither), and
   `from_state` refuses a member set whose sea-ice or land planes differ.
   Conservation targets are per member and swapped onto the model before
   each member's step. `ensemble_config` re-cuts the deterministic config
   at the ensemble truncation with the native suite's `dx_m` scaled by
   the truncation ratio so the scale-aware closures see the grid they run
   on; `recut_config` does the same in either direction (the OSSE's nature
   run and control sit above the config's truncation).
6. **Initial ensemble**: spectral perturbations of the base state per
   field (streamfunction and velocity potential for the wind, theta
   through the local Exner function for temperature, ln ps, relative
   vapor), red spectrum `n^-3` in power to half the truncation, per-level
   amplitude from a climatological table, plus time-lagged differences of
   the analyses the caller hands over, scaled to the same amplitude with
   a random sign, weight `lagged_difference_weight`. Every seed, amplitude
   and lagged pair is in the provenance.
7. **Recentering**: the control analysis truncated to the ensemble
   truncation replaces the ensemble mean, each member keeping its own
   perturbation; surface, physics namespace and grid tracers stay each
   member's own. The control's tracers are never regridded onto the
   ensemble grid: the ensemble's base state comes from the same analysis
   ingest at its own truncation (`ensemble_config`), so its tracers,
   surface and physics are consistent with its grid from the start.
   **Amended (A):** `recenter` takes a `fraction` (1 replaces the mean,
   full recentring; a value in (0, 1) is the partial recentring with
   operational precedent), `FilterOptions.recentering_fraction`, default
   1.0 until the `recentering_fraction` sweep of section 4 says otherwise;
   with increments pending (IAU) the shift is folded into the pending
   increments so the window applies it.
8. **The control gets its own analysis (amendment A; replaces the earlier
   "deterministic update by the ensemble-mean increment").** For every
   assimilated row the control's innovation `d_H = y - H(x_H^b)` is formed
   on the T255 background by the same operators at the control's own
   truncation (`PointObs.control_simulated`, filled by `analyze_ensemble`
   for the neutral vocabulary or brought by the stream, or by the
   `ObservationWindow` at the report's own time), and the point LETKF
   returns beside the member increments the control increment `X_L Pa~ C
   d_H` at every gridpoint of the ensemble grid: the same localised
   weights and the same `Pa~ = ((K-1) I / rho + Yb^T R^-1 Yb)^-1`, the
   control's innovation in the place of the ensemble mean's. Rho and the
   relaxation act on the perturbations only, so the control increment is
   the localised ensemble gain applied to `d_H` (held to 1e-9 relative
   against the analytic single-report gain, and equal to the ensemble-mean
   increment to 1e-9 when the two innovations coincide). The ensemble-mean
   increment never touches the control. `apply_mean_increment` (the
   pre-amendment path) is kept for the OSSE's `transfer` family only.
9. **The transfer is a scientific component (amendment C).**
   `apply_control_increment` analyses the control increment from the
   ensemble grid into the ensemble triangle (theta, qv, ln ps through the
   forward analysis, the wind through the vector analysis under the
   balance rule), multiplies every degree by the transfer taper
   (`taper_weights`: one up to `transfer_taper_start_degree`, a raised
   cosine to zero at `transfer_taper_end_degree`, zero above; the defaults
   are 0.6 T and T of the ensemble truncation, stated values the OSSE
   calibrates against the ensemble error spectrum), embeds the tapered
   increment in the control's triangle with the degrees above the ensemble
   truncation exactly zero, adds it to the control background with the
   global-mean surface pressure kept, repairs the vapor and enforces the
   state. The increment spectrum is recorded by degree and by band
   (planetary n <= 20, resolved to 0.6 T, the taper band, above the
   ensemble truncation) before and after the taper for the temperature,
   ln ps and the wind's kinetic energy (`spectral_power_by_degree`,
   `wind_power_by_degree`, both Parseval-checked against the grid mean
   square), with the small-scale share above 0.6 T as the number a reader
   inspects after every localised grid-space analysis. The ensemble-mean
   increment's spectrum rides in the receipt the same way.
10. **Observations at their own times (amendment B).** `ObservationWindow`
    cuts the window `(start, end]` into `bin_s`-wide bins (a report before
    the window joins the first bin, after it the last) and observes each
    bin at the step it closes on: every report in the bin has happened by
    then, so the comparison is causal and a report is never more than one
    bin from its state. As `advance_to` passes through the window the
    observer fills the members' `simulated` (and, with `control=True`, the
    control's `control_simulated`) column by column; what is retained is
    the observation-space trajectory, never a state per step. A one-bin
    window reproduces the instantaneous operators bitwise (held by test);
    `finish` observes the rows an integration never reached and the record
    counts rows observed at their own time against rows observed at the
    end. The bin width is `GlobalOsseSetup.observation_time_bin_s` in the
    twin (the `observation_time_bin_s` sweep of section 4 is the
    sensitivity measurement); the door chooses its bin.
11. **Uncertainty maintenance is one configuration first (amendment D).**
    RTPS at alpha 0.9 is the one inflation; `additive_inflation_fraction`
    now defaults to 0 and is an addition tested on its own. The increment
    enters the state by `FilterOptions.increment_application`: `direct`
    (at the analysis instant) or `iau` (the incremental analysis update:
    each member's spectral increment is scheduled on the ensemble and
    `step_all` adds an equal portion before each of the window's steps,
    the receipt's O-A read from direct-insertion copies and labelled
    direct-equivalent; the control's IAU is the door's to wire from
    `control_increment_spectral`, the tapered increment at the ensemble
    truncation). The `increment_application` sweep of section 4 is the
    measurement the v1 door's unbalanced ln ps increment asked for. The
    balance and increment tests hold: the global-mean surface pressure is
    kept per state to 1e-12, the vapor's repair is recorded, the wind
    transforms round-trip through the vector analysis, and no balance is
    imposed by latitude.
12. **The scorecard rule is replaced (amendment G).** Engineering
    validity (ingest, quality control, operators on every accepted stream
    before and after, a finite localised solve, the increment applied, the
    control analysed when asked, the state enforced) is the only hard gate
    and the analysis `status`. O-B and O-A are distributions (count, bias,
    rms, std, quantiles p05 to p95) per stream, variable and region for
    the ensemble mean and for the control; the withheld split of the v1
    door is kept as a diagnostic (`gated`, `gate_passed`,
    `withheld_o_minus_a_below_o_minus_b`) and never drives the status; the
    Desroziers ratios per (stream, variable) carry their assumptions
    verbatim (`DESROZIERS_ASSUMPTIONS`); the receipt's `assessments`
    separate engineering validity, statistical consistency (Desroziers
    error-variance and innovation ratios inside [0.5, 2] on streams with at
    least `desroziers_minimum_count` rows), physical consistency (the mass
    offset the rule removed bounded at 1e-4 in ln ps, the divergent share
    of the wind increment, the repair, the control increment's small-scale
    share) and predictive value (deferred by name to the observation
    scorecard on the forecasts). The `gate_of_record` keys the door reads
    (`rule`, `failed`, `incomplete`, `passed`) stay, with the rule text
    now the engineering rule; the door should read `assessments`.
13. **Quality control order**: gross bounds and age window (the v1
    door's table), chain refusal, thinning to one report per ensemble
    grid cell per stream and variable (nearest the cell centre; per level
    bin aloft), background check against `sqrt(error^2 + spread_H^2)` at
    `background_check_sigmas`. Every count per stream by name.
14. **A batch's operator takes the batch.** `PointObs.operator(members,
    batch) -> (R, batch.count)`: the analysis subsets a batch (quality
    control, the withheld split) and re-evaluates the SUBSET on the
    analysed members for O-A. Every neutral batch is bound to the
    analysis's own operators at analysis time (a window built on the
    control's grid must not hand the members the control's operator); a
    foreign stream (radiance, refractivity) keeps its own and, when a
    control is given, must bring `control_simulated` or is refused by
    name. Surface rows carry `elevation_m`.
15. **The innovation is formed once.** `flatten_batches` computes the
    member mean of H(x) over the concatenated `(R, n)` array and every
    chunk gathers it (`FlatObs.simbar`); the innovation is `value -
    simbar` on those bits, and with `control=True` the control innovation
    `value - control_simulated` beside it. A report whose value IS that
    mean has an innovation of exactly zero and a mean weight vector of
    exactly zero.
16. **Member streams are keyed by a CRC of their purpose**, not Python's
    salted string hash: the same seed draws the same ensemble in every
    process.
17. **Device arrays stay on the device.** `ColumnGeometry` reads shapes
    from the arrays themselves; the T127 analysis on the RTX 5090 died on
    an implicit host conversion before this.
18. **Recentring is by increment (measured 2026-09-06).** The control's
    orography is not the ensemble's: on the T63 / T127 twin every
    recentring that replaced the ensemble mean with the control STATE
    truncated to T63 carried the T127 terrain's surface pressure onto the
    T63 grid, the members' surface pressure rmse against the truncated
    nature run rose from 2.9 to 7 hPa within the hour after each analysis
    and the pressure spread grew 1.4 to 3.5 hPa in five cycles while the
    control itself read 2.9 to 2.4 hPa. `recenter` therefore replaces the
    ensemble-mean INCREMENT with the control's increment at the ensemble
    truncation (`FilterOptions.recentering_mode = "increment"`, the
    default; the members keep their terrain-consistent background and
    receive the control's analysis increment, the tapered
    `control_increment_spectral` the analysis already formed), and the
    truncated-state form stays as `"state"` for the twin's
    `recentering_mode` sweep. The same rule holds the twin's ensemble base:
    its own spun-up cold start plus the truncated displacement, never the
    truncated control state. The door passes
    `mode=options.recentering_mode, control_increment=result.control_increment_spectral,
    mean_increment=result.mean_increment_spectral`.
19. **What the smoke grid cannot decide.** At T3 (72 columns, 16
    coefficients, six or twelve members) the 10 m wind rows sit at their
    own noise under both balance modes, and a noisy set whose prior mean
    already lies inside the report error can pull the global rmse UP while
    fitting the reports. The smoke tests judge the rmse with perfect
    observations; the noisy recovery, the wind-balance default, the bin
    width, the IAU verdict and the recentring fraction are the T63 twin's
    (section 4).
20. **The pull family's gate is the pooled normalised observation fit,
    and the analytic families' bars are rounding bars in the state
    dtype's epsilons (measured 2026-09-06).** The first T63 / T127 pull
    twin (perfect reports at the stated errors, three cycles) fit every
    stream but one: at the second analysis the control's 2 m temperature
    O-A rose 0.43 to 0.73 K on 257 rows while the sounding temperature
    fell 0.71 to 0.51 K on 2,957 rows and the pooled fit, the observation
    term of the cost `sum(count (rms / sigma)^2)` over every assimilated
    (stream, variable), fell 5,676 to 4,526; the analysis traded the
    stations against eleven times as many soundings, which is what
    amendment G says a least-squares analysis does and what the
    per-stream "O-A below O-B" rule graded as a failure. `pull` now passes
    when the control's grid rmse against the nature run falls at every
    analysis on temperature, wind and surface pressure and the pooled fit
    falls at every analysis (`pooled_normalised_fit`, recorded for the
    control and for the mean); the per-stream reading stays in the verdict
    as a diagnostic. The `single` and `agree` families read 1.0e-7 and
    8.7e-7 relative on float32 members on the card against bars of 1e-9
    and 1e-10 written for the test world's host doubles, so a rounding
    residual read as a failed calibration: the bars are now
    `SINGLE_BAR_EPSILONS` (128) and `AGREE_BAR_EPSILONS` (256) machine
    epsilons of the state dtype, four times the worse residual measured in
    either dtype (single 0.84 float32 epsilons on the T63 device ensemble
    and 25 float64 epsilons on the test world, agree 7.3 and 44); a wrong
    localisation shape, error variance or gain sign shows at 1e-2 relative,
    five orders above the bar, and the receipt states the dtype, the
    epsilon and the bar it was graded against.
21. **The hybrid covariance has its interface and no body (amendment A,
    release 2).** `FilterOptions.hybrid_beta` is the weight of the localised
    ensemble covariance in `beta B_ens + (1 - beta) B_static`, one
    positive-semidefinite representation; it ships at 1.0, rides in every
    receipt through `identity()`, and any value below one is refused by
    name because no static covariance is built in release 1 and an analysis
    that accepted the value would run on the ensemble covariance alone
    while its receipt claimed a hybrid. The door carries the knob from the
    start; the static term, its calibration and the beta sweep are
    release-2 work.
22. **Recentring keeps every member's global-mean surface pressure
    (refuted and fixed 2026-09-06).** The control's increment reaches
    `recenter` as the tapered increment BEFORE the control's own mass rule
    added its constant, while the ensemble-mean increment it replaces
    carries the members' constants, so the shift of decision 18 taken raw
    carried a global-mean ln ps of -0.4e-4 to -1.6e-4 into every member at
    every analysis (measured on the seed-777 T63 / T127 twin: -1.64e-4,
    -0.78e-4, -0.82e-4, -0.78e-4, -0.73e-4, -0.42e-4 over six analyses,
    -51 Pa cumulative; on the lane's twin of record the control's offsets
    read 1.0e-4 to 2.2e-4 per analysis, 10 to 22 Pa each) and the members'
    conservation epoch then held the moved value. `recenter` now applies
    the same rule every increment application uses, per member (the
    constant that keeps the member's global-mean surface pressure, folded
    into the pending increments under IAU), records the offsets and the
    members' mean surface pressure before and after (`preserve_global_mean_pressure`,
    default True; the raw form is the twin's `recenter_preserve_mass=false`
    experiment arm). The perturbations move by the difference of the
    members' constants only, below float32 rounding of ln ps.
23. **The spread is area-weighted, and the pressure score of a coarse mean
    against a finer truth is the reduced pressure (refuted and fixed
    2026-09-06).** `GlobalEnsemble.spread` was the equal-weight mean of the
    pointwise spread over the Gaussian gridpoints, set beside an
    area-weighted rmse in one ratio; it now area-weights (levels averaged)
    and the twin records the equal-weight form beside it
    (`spread_equal_weight`; at T63 the two differ by 2 to 4 percent on
    temperature and under 1 percent on wind, so the lane's temperature and
    wind ratios stand within that). The twin's `surface_pressure_pa` rmse
    of the ensemble mean against the truth truncated to the ensemble
    triangle is the two orographies' difference before it is an error
    (590 Pa at T63 against T127 on seed 777, where the control on its own
    grid reads 255), so the lane's "spread over rmse 0.30 on pressure,
    short" was that floor in the denominator; `score` now adds `mslp_pa`,
    each member's and the truth's surface pressure reduced to sea level
    through its own lowest-level virtual temperature and the terrain its
    pressure sits on (the truth's: the control terrain truncated to the
    ensemble triangle), and the verdict's pressure ratio is that row (0.61
    on the seed-777 pull twin, 0.78 on the recovery twin, inside the band).
24. **The door's operators reduce through the spectral terrain, which is
    the grid terrain on the native configs and not on the smoke config
    (measured 2026-09-06).** `MemberOperators` sample the terrain from the
    spectral projection of the model's grid surface geopotential (the v1
    door's `_ModelSpace` arithmetic). On the T63 and T127 native configs
    the grid terrain is band-limited at the truncation, so the two agree
    to 1.2 and 2.0 mm at the worst point (0.13 and 0.17 mm area rms, 0.01
    and 0.02 Pa of station pressure at the worst point): no station-height
    bias there. On the smoke config the terrain is not band-limited (4.3
    m of 22 m at T3, 22 Pa of station pressure at one column), which is
    where an independent closed-form test first read a 22 Pa gap. The test
    in `tests/test_arwen_global_da_refutation.py` holds the reduction
    arithmetic to 1e-9 against the operator's own reference and asserts
    the smoke-config gap exists, so a change to either the smoke terrain
    or the shared operator flips it knowingly; the native-config numbers
    are `r-terrain-gap-t63-t127.json` in the refutation's evidence folder.

## 3. What the door lane can call today

Everything in section 1 has a body and a test
(`tests/test_arwen_global_da_letkf_point.py`,
`tests/test_arwen_global_da_ensemble.py`,
`tests/test_arwen_global_da_amendments.py`; 33 tests on the desktop under
`GPUWM_NO_LOCAL_GPU=1`). The door builds the ensemble model from
`ensemble_config(det_cfg, EnsembleOptions(...))` with
`runner.build_model_and_cold_state`, then per window:

```python
ens = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)   # or GlobalEnsemble.read(...)
ops = MemberOperators.for_model(model, transform, ecfg)
batches = batches_unevaluated(rows, ops)                                 # woof.globe.da.window
window = ObservationWindow(batches, start_s=t0, end_s=t1, epoch=epoch, bin_s=600.0, dt_s=ecfg.dt_s)
ens.advance_to(t1, ecfg.dt_s, observer=lambda e, t: window.observe(e.members, t, ops))
# the door's own deterministic advance calls, after each step,
#   window.observe([det_state], t, det_ops, control=True)
window.finish(ens.members, ops); window.finish([det_state], det_ops, control=True)
control = ControlBackground(det_state, det_model, det_transform, det_cfg)
result = analyze_ensemble(ens, window.batches, FilterOptions(...), analysis_time=moment, control=control)
det_analysis = result.control_analysis                                    # the T255 analysis
recenter(ens, det_analysis, det_transform, fraction=options.recentering_fraction,
         mode=options.recentering_mode, control_increment=result.control_increment_spectral,
         mean_increment=result.mean_increment_spectral)
ens.write(outdir)                                                          # members + manifest
```

`result.report` is the receipt (per stream, per variable, per region O-B
and O-A distributions on the assimilated and the withheld rows for the
ensemble mean and the control; the Desroziers ratios; the four
assessments; spread before and after; the increment summaries and their
spectra by band; the LETKF's counts and timings; the rejections by name;
`window.record()` is the door's to add). `python -m
woof.globe.da.osse` runs the dual-resolution twin, the families
and the sweeps; `python -m woof.globe.da.measure` reads the card;
`tools/arwen_global_ensemble_osse_chart.py` charts a twin or a sweep.

What changed for the door since its 5c4bebb4b wiring: pass `control=`
instead of calling `apply_mean_increment`; take `result.control_analysis`;
pass the recentring fraction and mode with the two increments (decision
18); read `assessments` (the withheld gate no longer sets the status); build the batches through
`batches_unevaluated` and an `ObservationWindow` when the cycle's reports
carry times inside the window (a one-bin window is the instantaneous
form).

## 4. Measurements

RTX 5090, the native suite (RRTMGP, sfclay, Noah, YSU, Grell-
Freitas, Morrison), 40 levels, GDAS 2026-08-31 00Z cold start, tree
64ffc5c53 plus the dx_m scaling, 2026-09-06 01:18 UTC
(`woof.globe.da.measure`):

| ensemble | dt | members | pool per member at construction | member arrays after the first step (spectral / tracers / surface / physics) | pool live after 3 to 6 steps | wall per member-step (mean after the first) |
|---|---|---|---|---|---|---|
| T63 L40 | 200 s | 8 | 29.9 MB | 73.0 MB (5.3 / 29.5 / 2.2 / 36.0) | 0.65 GB | 0.0786 s |
| T127 L40 | 100 s | 8 | 119.8 MB | 291.8 MB (21.1 / 118.0 / 8.8 / 143.9) | 2.55 GB | 0.1106 s |
| T127 L40 | 100 s | 32 | 141.5 MB | 291.8 MB | 9.68 GB (8.70 GiB of member arrays) | 0.1126 s |

Reading (amendment J): "32 x 50 MB" is the spectral state alone; the
complete restart state of a T127 member is 292 MB once the native suite's
step-to-step quantities (144 MB) and the grid tracers (118 MB) exist, so a
32-member T127 ensemble holds 8.7 GiB of member arrays and 9.0 GiB live
beside a shared workspace of about 0.25 GB at rest; the T255 control adds
its own 8.5 GiB process footprint. Throughput: 0.113 s per member-step at
T127 with dt 100 s is 4.1 s per member-hour and 130 s per ensemble-hour at
32 members (36 steps); T63 at dt 200 s is 1.4 s per member-hour. The model
build is 5 s, the 32-member construction 1.8 s.

Twins (the dual-resolution OSSE, `python -m woof.globe.da.osse`):
on the T3 / T7 smoke twin (six members, 1,428 reports per cycle, bins of
20 s, three cycles) the control temperature rmse against the nature run
falls 1.317 to 1.174 to 1.112 to 1.071 K and the control analysis beats
the mean-increment arm at every cycle (1.174 against 1.181, 1.112 against
1.120, 1.071 against 1.079 K).

The T63 / T127 recovery twin of record (RTX 5090 shared with another
lane's run, 2026-09-06 02:27 UTC; T127 nature run and control at dt 100 s,
T63 ensemble of 16 at dt 200 s, the 2,390 CONUS ASOS stations and 625
IGRA2 sites of the case with their own levels plus 400 synthetic motion
vectors per hour, 28,198 reports per cycle drawn at their own 600 s bins,
13,700 assimilated after thinning to one report per T63 cell and 1,530
withheld, six hourly cycles, recentring by increment): the control's
temperature rmse against the nature run 1.734 K displaced, then 1.578,
1.425, 1.359, 1.320, 1.285, 1.256 K after the six analyses (every analysis
reduced it); surface pressure 280 to 241 Pa; vapor 1.36 to 0.86 g/kg; the
wind 2.567 to 2.509 at the first analysis and then 2.368, 2.403, 2.462,
2.538, 2.620 m/s (the control's wind error grows 0.08 m/s per hour between
analyses and each analysis removes 0.02; the rotational rule discards the
divergent half of the analysed wind increment, 0.44 to 0.54 of its kinetic
energy per cycle; the `wind_balance` sweep is the measurement). Ensemble
spread over the mean's rmse at the last cycle 0.51 (T), 0.56 (u), 0.58
(ps), inside the stated band at its low edge. Desroziers error-variance
ratios 1.2 to 2.4 and innovation ratios 1.1 to 2.1 on the streams (the
assigned errors and the spread together under-read the residuals, so the
statistical assessment is flagged), the mass-preserving ln ps offset 1.4e-4
(flagged against the 1e-4 bound). The LETKF solve is 2.3 s on the card (9
chunks of 11 rings, 400 local reports the cap, 320,000 dropped by it); the
analysis wall 18 to 36 s, of it 16 s the O-A operators on the host; the
ensemble's hour 63 to 157 s (16 members plus six observing bins) and the
control's 10 to 26 s, so T_cycle at T63 / 16 reads 90 to 200 s under a
shared card. The same twin with recentring on the truncated STATE (the
pre-measurement default) gave the control the same numbers (1.255 K,
240 Pa, 2.623 m/s) but dragged the ensemble mean by 1.22 K rms of theta
every cycle and left the members' surface pressure 6.9 hPa from the
truncated truth before every analysis (2.6 to 2.9 hPa after); by increment
the shift is 0.12 K and the mean's pressure reads a steady 5.9 hPa against
the truncated truth, which is the T63 against T127 orography difference
and not a filter reading (the members' pressure is judged at the stations:
O-B 405, O-A 337 Pa on the assimilated rows at the last cycle, the
control's 263 and 143 Pa). The ensemble pressure spread grows 1.4 to 3.5
hPa over the six cycles in both modes. The transfer twin (the same setup, a second control trajectory receiving
the ensemble-MEAN increment, the pre-amendment path) reads the control's
own analysis ahead from the second cycle on: temperature 1.578 / 1.575,
1.425 / 1.427, 1.359 / 1.368, 1.320 / 1.339, 1.285 / 1.314, 1.256 / 1.299 K
(control / mean-increment arm), wind 2.620 against 2.714 m/s and surface
pressure 241 against 267 Pa at the last cycle; amendment A is measured, not
argued. The recentring-mode sweep (three cycles each) reads the control
indifferent to the mode (1.359 against 1.356 K, 275 against 276 Pa, 2.403
against 2.408 m/s after the third analysis) while the members' pressure
against the truncated truth reads a steady 585 to 595 Pa by increment and
585, 265, 693, 285, 694, 276 Pa (before and after each analysis) by state,
the same spread in both (144 to 179 Pa, 0.81 to 0.70 K): the mode is a
consistency property of the members, not a control score, which is what
decision 18 claims.

The T127 / T255 twin (the production shape: T255 nature run and control
at dt 50 s, T127 ensemble of 16 at dt 100 s, the same network, four
hourly cycles, 16,300 reports assimilated and 1,800 withheld per cycle,
2026-09-06 02:57 UTC on the shared card): the control's temperature 1.770
K displaced, then 1.536, 1.321, 1.254, 1.230 K; its wind 4.317 to 3.825,
3.609, 3.636, 3.695 m/s (the analysis takes 0.2 to 0.5 m/s per cycle here,
the sounding wind O-B 5.3 to O-A 3.8 m/s at the first analysis); its
vapor 1.36 to 0.85 g/kg; its surface pressure 202 to 194, 248, 352, 440
Pa, GROWING between analyses while the CONUS stations read the control at
157 to 284 Pa O-B and 103 to 142 Pa O-A: the displaced control's pressure
drifts where the twin's network observes nothing (the stations are
CONUS-only, the soundings and motion vectors do not hold the mass field),
a coverage reading of the synthetic network and not a filter defect. The
spread is short: spread over rmse 0.59 (T), 0.40 (u), 0.43 (ps) at the
fourth analysis, below the band for wind and pressure, with Desroziers
innovation ratios 1.5 to 3.3 on the streams; RTPS at 0.9 alone
under-disperses this twin and the additive or a larger relaxation is the
next single change to test (amendment D). Wall: the LETKF 5.6 to 7.5 s on
the card (39 chunks of 5 rings, 2,700 local reports at the widest column,
2.6 million dropped by the 400 cap), the analysis 78 to 158 s of which 72
s is the O-A operators on the host, the ensemble hour 80 to 87 s (0.145 s
per member-step shared) plus its six observing bins, the control hour 50
s: T_cycle at T127 / 16 reads 215 s per hour on the shared card, the host
operators the lever. The verdict passed (temperature and wind below the
displaced start, every analysis reducing the control temperature). The
T127 / T255 transfer twin reads the control's own analysis ahead of the
mean-increment arm on temperature at every cycle after the first (1.321 /
1.330, 1.254 / 1.271, 1.230 / 1.255 K), level on the wind (3.695 against
3.688 m/s, inside the cycle-to-cycle noise) and on pressure 440 against
440 Pa at the fourth analysis. Charts, tables and JSON are recorded with
the measurement (2026-09-06, with captions); an earlier copy of the same
set is without the 32-member twin and the families.

The wind-balance sweep (T63 / T127, three cycles each) keeps the
rotational rule: after the third analysis the control reads 1.359 K,
u 2.403 and v 1.998 m/s, 275 Pa under `rotational` against 1.365 K,
2.410 and 2.000 m/s, 292 Pa under `unconstrained`; the divergent half of
the analysed wind increment adds imbalance the pressure pays for (17 Pa)
and buys the wind nothing, so `wind_balance` stays `rotational` by
measurement (decision 4).

The increment-application sweep (the members' increment direct at the
analysis instant against the incremental analysis update spread over the
next hour's steps; the control takes its increment directly in both arms,
its IAU being the door's to wire) reads the control the same after three
cycles (1.359 against 1.363 K, 2.403 against 2.404 m/s, 275 against 275
Pa) and the members' background at the third analysis the same
(1.458 against 1.461 K of mean temperature rmse, pressure spread 179
against 182 Pa): at T63 with hourly windows the members' application
mode does not change the covariance the control is analysed with, and the
v1 door's pressure-increment shock is a control question the twin's
control cannot show while it inserts directly. Direct stays the default;
`iau` is the tested option.

The bin-width sweep (amendment B; the synthetic reports drawn at times
spread through each hour and observed, truth and members alike, at the
step their bin closes on) reads the control after three cycles at
1.355 K, 2.403 m/s, 273 Pa with one 3,600 s bin (every report compared
with the analysis-time state), 1.359 K, 2.403 m/s, 275 Pa with 600 s
bins and 1.359 K, 2.404 m/s, 276 Pa with 200 s bins: at T63 with hourly
windows the bin width moves the control by 4 mK and 2 Pa, inside
the cycle-to-cycle noise, so 600 s stays the stated default and the cost
of comparing a report with a state up to one bin away is measured, not
assumed.

The recentring-fraction sweep (full against half recentring of the
members' mean increment onto the control's, three cycles) reads the
control the same (1.359 against 1.359 K, 2.403 against 2.404 m/s, 275
against 275 Pa) and the members' background at the third analysis
1.458 against 1.452 K of mean temperature rmse with the same spread
(0.707 against 0.707 K): by increment the two fractions differ only in
how much of the control's increment the members receive, and at this
cadence the control cannot tell. Full recentring stays the default; the
partial form is the tested option with operational precedent.

The 32-member T127 / T255 twin (the design's member count at the
production shape: T255 nature run and control at dt 50 s, 32 members at dt
100 s, the same network, three hourly cycles, 16,321 reports assimilated
and 1,813 withheld per cycle, 2026-09-06 04:18 UTC under a 20 GiB pool
reservation on an otherwise idle card): the control's temperature 1.770 K
displaced, then 1.529, 1.308, 1.235 K (16 members read 1.536, 1.321, 1.254
K at the same three analyses: 7, 13 and 19 mK better at 32); its wind
4.317 to 3.786, 3.555, 3.563 m/s; its surface pressure 202 to 198, 253,
351 Pa, growing between analyses where the CONUS network observes nothing,
as at 16 members; spread over rmse 0.59 (T), 0.41 (u), 0.39 (ps) at the
third analysis against 0.59, 0.40, 0.40 at 16 members at the same
analysis: doubling the members does not move the spread ratio, so the
deficit is the inflation configuration's, not the sample's, and the next
single change (amendment D) is a larger relaxation or the additive term,
tested alone. The control increment carries 0.94 of its temperature power
at n <= 20 and 1.4e-4 of it in the taper band before the taper (0.8e-4
after), zero above T127 by construction; the rotational rule discards 0.26
of the analysed wind increment's kinetic energy at this shape (0.44 to
0.54 at T63); the vapor repair rescales at most 0.53 of one column at the
third analysis with 7e-7 kg/m2 unfillable. Wall (amendment J; the phases
run in sequence, no overlap): the 32 members' hour 158 to 171 s (0.143 s
per member-step, 36 steps, plus the six observing bins), the control's
hour 51 s, the analysis 149 to 150 s of which the O-A operators on the
host are 124 s and the LETKF 25 s (94 chunks of 2 rings, 2.1 million
active points, 2,695 local reports at the widest column, 2.7 million
dropped by the 400 cap); the first analysis 281 s with the background
operators' 131 s inside it. T_cycle at T127 / 32 reads 358 to 370 s per
hour against 215 s at 16 members: the ensemble's hour and the LETKF scale
with the member count, the host operators do not, and the host operators
remain the lever (124 s of 150). Member arrays 8.70 GiB (292 MB each) as
the measure run read them; an issued 24 h forecast of the control at 51 s
per hour adds 20 minutes to the schedule.

The calibration families on the card (T63 ensemble of 16 on the RTX 5090,
2026-09-06 04:45 UTC, tree 2931c7d52): `single` (one planted report
against the analytic localised gain `w P_xy d / (w P_yy + sigma^2)` at
every gridpoint of every analysis field) reads a worst relative difference
of 1.01e-7, 0.84 float32 epsilons against the bar of 128, and exactly zero
beyond the cutoff; `agree` (the 2,390 stations and 625 soundings reporting
the ensemble-mean H(x) itself) reads an innovation of exactly zero and a
worst relative mean increment of 8.7e-7, 7.3 float32 epsilons against the
bar of 256; at T127 (16 members) the same two families read 1.07e-7 (0.89
epsilons) and 9.2e-7 (7.75 epsilons), so the residual does not grow with
the truncation. The `pull` twin (T63 / T127, perfect reports at the stated
errors, three hourly cycles) takes the control's temperature 1.734 to
1.571, 1.417, 1.349 K, its wind 2.567 to 2.500, 2.390 to 2.349 and 2.412
to 2.381 m/s across the three analyses and its surface pressure 280 to
264, 280 to 278 and 280 to 272 Pa, every analysis reducing all three; the
pooled normalised fit falls 20,217 to 4,414, 5,676 to 4,526 and 7,267 to
4,187 (ratios 0.22, 0.80, 0.58) while the 2 m temperature O-A rises above
its O-B at the second and third analyses (0.43 to 0.73 K, 0.55 to 0.78 K)
as the analysis trades 257 station rows against 2,957 sounding rows
(decision 20); spread over rmse 0.495 (T, a hair under the band's low
edge of 0.5), 0.61 (u), 0.30 (ps) at the third analysis, the pressure
spread short where the network observes no pressure outside CONUS. Passed under the amended gate; under the per-stream rule it
had read failed on the one stream.

The refutation (2026-09-06 12:24 to 13:30 UTC, an RTX 5090 shared with
another job, tree 7e6be02cf plus the refutation's fixes, seed 777, a
start that was never run; the other card held another 7.4 GiB job at 100
percent for the whole window and carried nothing of it). The analytic
families on the T63 ensemble of 16: `single` 8.33e-8 relative (0.70
float32 epsilons, bitwise zero beyond the cutoff, 3,808 active points),
`agree` an innovation of exactly zero and 7.94e-7 relative (6.66
epsilons) over 520,167 active points: the lane's 1.01e-7 and 8.7e-7 hold on
a second start. The `pull` twin (T63 / T127, three cycles) takes the
control's temperature 1.768 to 1.557, 1.354, 1.288 K, its wind 2.524 to
2.404, 2.287, 2.315 m/s and its surface pressure 254.7 to 242.9, 260.9,
287.6 Pa (rising between analyses, 265 to 292, each analysis reducing it),
the pooled fit 0.22, 0.68, 0.82: passed, and the 2 m temperature stream
moves away at the second and third analyses on this start too (the
control's O-B to O-A 0.559 to 0.761 and 0.443 to 0.764 K while the sounding
temperature falls 0.806 to 0.557 and 0.557 to 0.483 K); the
`vertical_cutoff_lnp` sweep of the pull family (the aloft cutoff 1.5 against
0.8 and 0.4 in ln p) leaves that reading in place (0.585 to 0.758 K at 0.8,
0.581 to 0.750 K at 0.4) and reads the control worse as the cutoff
tightens (1.288 against 1.299 and 1.326 K after the third analysis), so
the reach of the sounding rows to the surface is not the cause and 1.5
stands; the ensemble MEAN's own fit of the same 2 m rows stays flat
(1.265 to 1.276 K) while the control's worsens, and the 10 m wind moves
the same way (0.39 to 0.74 m/s on the control): the coarse ensemble's
surface covariance applied to a fine control whose surface error is
already small pushes the control's surface diagnostics past the truth,
a representativeness gap of the transfer at the surface (amendment C),
not a least-squares trade in the ensemble's own space; named for the
door and observation lanes, not fixed here. The
`recovery` twin (T63 / T127, six cycles, the mass rule of decision 22):
the control's temperature 1.768 to 1.568, 1.368, 1.304, 1.250, 1.208,
1.172 K (every analysis reducing it), wind 2.524 to 2.408, 2.298, 2.333,
2.367, 2.414, 2.449 m/s (below the start after six, growing 0.04 m/s per
hour between analyses as on the lane's start), surface pressure 254.7 to
241.6, 264.8, 291.7, 342.3, 377.3, 407.2 Pa growing between analyses while
the CONUS stations read the control 147 to 231 Pa O-B and 118 to 172 Pa
O-A (the lane's coverage reading, here at T63 / T127 as well); the mean's
temperature 1.899 to 1.283 K, its reduced pressure 264 to 445 Pa; spread
over rmse 0.52 (T), 0.61 (u), 0.78 (mslp) with the equal-weight spread
reading 0.54 and 0.59: passed. The recentring mass sweep on the same
start: the raw shift moved the members' global-mean surface pressure by
-16, -24, -32, -40, -47 and -51 Pa cumulative over the six analyses (the
raw shift's global-mean ln ps -1.64e-4, -0.78e-4, -0.80e-4, -0.80e-4,
-0.77e-4, -0.36e-4), the mean's reduced-pressure rmse reading 448.1
against 444.9 Pa kept at the sixth analysis (the bias in quadrature) and
the control identical to four digits in both arms (the control never
receives the members' mass). The report-cap sweep (`max_local_obs` 400
against 3000, three cycles; 316,000 to 323,000 rows dropped per analysis
at 400, none at 3000 with the widest column padded to 1,215): the
control's temperature 1.5678 / 1.5678, 1.3678 / 1.3678, 1.3039 / 1.3038
K, wind 2.4077 / 2.4080, 2.2983 / 2.2987, 2.3331 / 2.3329 m/s, pressure
241.6 / 241.6, 264.8 / 264.8, 291.7 / 291.5 Pa, the LETKF solve 1.4 to
1.7 against 11.5 to 12.0 s: at T63 the cap selects the far reports out and
the near ones carry the information, so the cap taken by horizontal weight
before the level loop does not starve the upper levels here (T127, where
the widest column gathers 2,700, is not measured). Two processes on the
same seed (the cap-400 arm and the recovery twin) agree on 33 of 33
after-analysis rmse entries bitwise over three cycles. The card measure
(`woof.globe.da.measure`, eight members): T63 72.96 MB per member
after the first step and 0.0787 to 0.0789 s per member-step on the quiet
steps (0.155 to 0.179 s on the steps the other run shared), pool 652.7
MB; T127 291.8 MB and 0.113 s, pool 2.55 GB: the lane's rows hold to the
byte and to 2 percent of the wall. The terrain gap of decision 24: 1.2 mm
(T63) and 2.0 mm (T127) at the worst point. The T127 / T255 twin on the same start (16 members, four
cycles, the production shape): the control's temperature 1.686 to 1.573,
1.396, 1.343, 1.338 K (every analysis reducing it), wind 5.263 to 4.864,
4.424, 4.335, 4.197 m/s, vapor 1.13 to 0.81 g/kg: passed; its surface
pressure 364.5 to 388.1, 409.6, 504.2, 697.0 Pa, the first two analyses
RAISING the control's global pressure error (364.5 to 388.1 and 404.7 to
409.6 Pa) while the CONUS stations read it 391 to 105 and 180 to 188 Pa
(O-B to O-A), and the error growing 40 to 200 Pa per hour between
analyses where the network holds no pressure; the mean's reduced
pressure 399 to 718 Pa with a spread of 147 to 222 Pa, spread over rmse
0.53 (T), 0.37 (u), 0.31 (mslp), the equal-weight spread reading 0.54
and 0.36: the lane's pressure and wind spread deficit at this shape
stands on a second start, and the reduced-pressure ratio agrees with
the raw one here (0.31 against 0.28) because a 700 Pa error dwarfs the
orography floor. The 2 m temperature stream moves away on the control
at every analysis after the first (1.432 to 1.470, 1.254 to 1.384, 1.346
to 1.413 K) as at T63 / T127. Wall on the shared card: the members' hour
79 to 87 s, the control's 51 s, the analysis 78 s (the host O-A operators
72 s, the LETKF 5.5 s, 39 chunks of 5 rings, 2,693 local reports at the
widest column, 2.65 million dropped by the cap), T_cycle 208 to 216 s
against the lane's 215 s.

