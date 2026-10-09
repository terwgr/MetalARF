# WOOF global: a fresh global analysis, and a forecast from it

`woof global da` is the data-assimilation door of the experimental global
model. It makes an analysis of the atmosphere from public observations
whenever you want one, records where every number came from and what
information it was allowed to use, and hands you the checkpoint a forecast
starts from. This page is the path a user follows; the door's own report
and receipt carry the numbers.

EXPERIMENTAL, like the rest of `woof global`: a research door whose
configuration surface can move between releases.

## The one command

```
woof global da fresh arwen_global_t255_quickstart \
    --outdir out/fresh --stream iem-asos --filter letkf --members 32
```

The base TOML names the core the analysis and its forecast run on: a base
that omits `[time] integrator` runs the shipped default, the semi-Lagrangian
core at 300 s with its own drain and gather (the configs of record under
`configs/verify/` are bare for that reason), and the quickstart base above
runs that core too, at 300 s and 288 steps a day. A base that names
`imex_ssp3` runs the Eulerian core at its rule step (90 s at T255). The
quickstart's reference suite has not been run for a forecast day under the
semi-Lagrangian core on a card (2026-09-07); the native suite has, three
times from the bare door on two cards, every gate green.

What happens, in order, and what each step leaves behind in `--outdir`:

1. **The information cutoff.** `fresh` hands back the latest analysis
   constructible from information available by a declared cutoff
   (`--cutoff-utc`, default now). The newest observation hour is the
   cutoff minus `--observation-latency-s` (default 3600), floored to the
   hour; `--until-utc` names it directly. Every fetched object carries the
   window it covers, the instant it was first on disk here, the
   publication time where the source states one, and a latency class
   (`fast` within an hour of the window's end, `replay` within a day,
   `retrospective` beyond, `unverified` for a table already on disk whose
   arrival nobody measured; the observation streams module's vocabulary).
   Every row of a `gpuwm-obs.table.v2` table (the Rust observation doors'
   format) carries its measurement definition, nominal, published and
   received times and revision; a row received after the cutoff is
   dropped and counted, a row without a receipt time is kept and counted
   as latency-unverified. The receipt says which the run was and how many
   rows the cutoff turned away.
2. **The analysis.** The newest published GDAS cycle (00, 06, 12 or 18 UTC,
   about seven hours behind real time) is fetched through the tree's fetch
   door (`woof fetch --source gdas --mode full-file` underneath, the Rust
   `rw_fetch` when it is built) into `analysis/`, with
   `fetch-manifest.json` recording URL, bytes and SHA-256. Pass
   `--analysis-cycle 2026-09-01T00:00:00Z` for a named cycle, or
   `--analysis-grib PATH --analysis-cycle ...` for an object already on
   disk (digested, never re-fetched).
3. **The run configuration.** `fresh-config.toml` is the base TOML with
   `[initial] analysis_grib` pointed at the fetched object and
   `[time] duration_s` set to the cycle span plus `--forecast-hours`
   (default 24). Nothing else in the base changes; its hash is the
   lineage's config identity from here on.
4. **`init`.** The control state is the analysis on the model grid (the
   cold start, written as `da-initial-state.npz`); with `--filter letkf`
   the members are built from the ensemble configuration's own cold start
   at `--ensemble-truncation` (default 127) and written under `ensemble/`;
   the manifest `da-ensemble.json` names every checkpoint, unless a
   manifest already sits in the output.
5. **`cycle`.** Hourly from the analysis instant to the newest observation
   hour. When a window `(hour - 1 h, hour]` OPENS its streams are fetched
   into `fetch/` with a manifest per window and the reports are assigned
   to time bins; as the control steps through the window the filter takes
   its observation-space equivalents at the bin instants
   (`--observation-bin-s`); at the hour the members catch up the same way,
   the reports are quality controlled, the control is analysed, the
   members are analysed and recentred, the analysis checkpoint
   `arwen_global_analysis_stepNNNNNNNN.npz` and its
   `assimilation-report-stepNNNNNNNN.json` are written, and the model
   steps on. The receipt of the run (`arwen-global-receipt.json`) carries
   the `cycle` record.
6. **The hand-back.** `da-receipt.json` names the analysis checkpoint (path,
   digest, step) and prints the forecast command:

```
woof global da forecast out/fresh/fresh-config.toml \
    --analysis out/fresh/arwen_global_analysis_step00000432.npz \
    --outdir out/fresh/forecast
```

The forecast runs to the derived duration and writes the ordinary hourly
checkpoints; `woof global export` and `woof render` read them as they
read any run's.

Without a card, rehearse the whole path on the smoke configuration
(analytic initial state, nothing fetched, `--start-utc` names the instant
model time zero stands for):

```
woof global da fresh arwen_global_moist_smoke \
    --outdir out/fresh-smoke --stream local-tables:paths=obs.csv \
    --start-utc 2026-08-31T11:59:40Z --until-utc 2026-08-31T12:00:20Z \
    --interval-s 20 --forecast-hours 0.00555555555556 --filter letkf --members 3 \
    --ensemble-truncation 3 --observation-bin-s 10
```

(`--forecast-hours` is 20 s here, spelled to the digit: the derived
duration has to be a whole number of the smoke core's 10 s step, and
`0.0056` is 20.16 s, which the configuration refuses by name. Every
report of the table sits at 12:00:00, the first window's end, so the
second window carries the background: see the window section.)

## Streams

A stream is a public observation source the door fetches for a time
window on its own and records: URL or path, bytes, SHA-256, the wall the
fetch took, how far behind real time the window's end was when the bytes
arrived, and the latency class that reads from it. The table is
`woof.globe.da_streams.STREAM_TABLE`; a stream is spelled `NAME`
or `NAME:key=value;key=value`.

| stream | what it fetches | decoder (obs-table entry) | options |
|---|---|---|---|
| `iem-asos` | ASOS/METAR surface reports worldwide from the Iowa Environmental Mesonet archive through `rw_asos`, the Rust surface front door: 2 m temperature and dewpoint, 10 m wind, altimeter | `iem-asos-csv` | `networks=IA_ASOS,IL_ASOS` (default every network the tree's surface-network table lists), `bbox=W,S,E,N`, `slack_minutes=60` |
| `local-tables` | tables already in the obs-table vocabulary, on disk or at a URL: the neutral `gpuwm-obs.table.v2` files the Rust observation doors write (and the ten-column v1) (`rw_igra2`, `rw_amv`, `rw_ndbc`, `rw_asos table`, `rw_gnssro`), the radiosonde level table `tools/arwen_global_igra2_levels_csv.py` writes, a METAR cache | whatever header the table carries | `paths=A.csv,B.csv.gz` |

`--obs PATH` is the older spelling of a local table decoded once for every
cycle; it still works beside `--stream`.

Adding a stream is a table entry with a fetcher and an obs-table decoder
entry, never a code path. The streams of the observation programme (the
radiosonde archive and real-time feed, GOES ABI derived motion winds
through `rw-sat`, GNSS radio occultation, NDBC buoys, the WIS2 caches)
register here as they land; their tables already read through
`local-tables` today. The external analysis is NOT a stream (see the
anchor below). Aircraft (MADIS) needs an account and is reported, not
fetched.

## The window and the reports' own times

An hourly cycle is a window, not an instant. With `--observation-bin-s B`
the window `(t0, t1]` is cut into bins of `B` seconds (a whole multiple of
the model step and of the ensemble step, dividing the interval; the last
bin ends at the analysis instant), every report is assigned to the bin
instant nearest its valid time, and the filter evaluates its operators at
that instant along the trajectory: a report at 12:08 is compared with the
12:08 state (the observation-space value is kept, never the full state).
Reports of the last bin are compared at the analysis instant. Without the
option every report is compared at the analysis instant, and the report
says so.

A report whose time lies OUTSIDE the window is not analysed at this
window's instant: one earlier than `t0` belongs to an earlier window (the
neighbours the thinning kept then carried its information into the
state, and it has no trajectory in this window to be compared at its own
time), one later than `t1` to the next. The report counts them
(`rejections.outside_window`). Before this rule (the refutation of
2026-09-06) the rows one cycle thinned away were analysed again at the
next cycle's instant, twenty seconds after their own time on the smoke
case and up to an hour and a half in an hourly cycle, because the chain
holds only the rows an analysis used. A window with no admissible report
at all is CARRIED: the background is written as the hour's checkpoint,
the members keep their own forecast, the report reads `status carried`
with the reason (`carried_reason`) and the receipt counts the hour as
incomplete rather than dying on it.

The report's `observation_times` block carries the bin, the rows per
instant, the largest offset between a report and the instant it was
compared at, and the binning SENSITIVITY: the rms difference between the
analysis-instant equivalent and the binned equivalent of the same reports
per stream and variable, which is what an analysis-instant comparison
would have mistaken for an innovation. After the update, the analysis
equivalent of a binned report is the LINEARISED one (the binned
background equivalent plus the increment in observation space at the
analysis instant) and it is labelled so in the card
(`o_minus_a_label`).

## The filter

`--filter` names how the increment is formed. Two filters:

- `successive-correction` (the default): the deterministic v1 door, a
  data-density normalised successive correction against one background,
  reports compared at their own height and at the analysis instant, the
  wind increment applied as its rotational part, the pressure increment
  preserving the global mean (see the assimilation section of
  `ARWEN_GLOBAL.md`).
- `letkf`: the dual-resolution ensemble filter of the ensemble package
  (`src/arwen_global/da/`, its own interface notes in
  `docs/arwen-global-ensemble-da.md`) under the design amendments of
  2026-09-06: N members (`--members`, at least 3; 32 by default when the
  package's defaults stand) at their own truncation
  (`--ensemble-truncation`, default 127, at or below the control's)
  resident in one process sharing one model, stepped by the package at
  their own time step under per-member conservation targets. One
  analysis, in order:
  1. the members catch up with the control, observing the bins on the way;
  2. the batches are built on the members at the analysis instant, the
     control's `H(x_H^b)` is evaluated with the same operator arithmetic on
     the control, and the binned equivalents replace the analysis-instant
     ones where a bin was observed;
  3. quality control (gross bounds, age window, chain refusal, thinning
     to one report per ensemble cell, a background check against the
     ensemble spread) and the withheld split, the package's own, so the
     control and the members analyse the same rows;
  4. **the control's own analysis**: for every report the control's
     innovation `y - H(x_H^b)` goes through the local ensemble transform
     on the background members (the mean weight vector applied to the
     ensemble perturbations; Gaspari-Cohn on the sphere in kilometres and
     in ln p), the increment is tapered per total degree, embedded in the
     control's spectral triangle (degrees above the ensemble truncation
     exactly zero) and added; the global-mean surface pressure is kept
     and the vapor repaired;
  5. the members' own LETKF analysis (the perturbation update and the
     ensemble's own mean update, RTPS); the ensemble-mean increment is
     RECORDED beside the control's (`mean_increment_transfer`: both rms
     per field and the rms of their difference) and never applied to the
     control unless `--control-increment ensemble-mean` names the
     comparison experiment;
  6. the anchor on the control analysis, when configured;
  7. the members recentred on the control analysis: by INCREMENT by
     default (`--recentre-mode increment`, the package's measured
     decision: the ensemble-mean increment is replaced by the control's
     increment, anchor included, restricted to the ensemble triangle and
     carrying the mass rule's constant, so the members keep their own
     terrain-consistent background and the ensemble's global-mean
     surface pressure (without the constant the ensemble mean fell by
     the control's raw increment mean every cycle, 78 Pa of METAR
     pressure O-B bias between the members and the control after 24
     hourly cycles); recentring
     on the truncated control STATE, `--recentre-mode state`, carried a
     finer orography's surface pressure onto the coarser grid, 7 hPa at
     T63 under T127), fully (`--recentre-fraction 1`, the default) or
     partially (a fraction in (0, 1), operational precedent, an
     experiment);
  8. the card on the control.

  Inflation: RTPS (alpha 0.9) is the one mechanism by default; additive
  inflation is off unless `--additive-inflation FRACTION` names it,
  because several adaptive mechanisms hide each other (a broken
  observation error looks like a spread deficit). The hybrid covariance
  (an ensemble weight beta and a static term) is declared in the control
  options and ships at beta 1 only; another value is refused by name.

The door never substitutes one filter for another silently: a manifest
built by one filter refuses a cycle that asks for the other by name.

### The transfer between the two resolutions

The coarse ensemble may correct the scales it demonstrably represents and
no others. The control increment carries a smooth spectral taper (the
ensemble package's): weight one at and below `--taper-full-degree`
(default 0.6 of the ensemble truncation), zero at and above
`--taper-zero-degree` (default the truncation), a raised cosine between;
the T255 background keeps every degree above the ensemble truncation
untouched. The report's `control` block records the taper weights by
degree and the increment's power by band (planetary, resolved to 0.6 T,
the taper band, above the ensemble truncation) before and after the
taper with the small-scale share, because spatially varying local
weights manufacture small-scale structure no observation supports and the
spectrum is where that shows. The taper's
degrees are the stated starting values; calibrating them against the
ensemble error spectrum and forecast sensitivity is the OSSE's
measurement, recorded when it is taken. The transfer is tested in the
model's native variables in both directions
(`tests/test_arwen_global_da_control.py`).

### Inserting the increment

`--increment-application direct` (the default) inserts the increment at
the analysis instant. `--increment-application iau` re-integrates the
window from its start adding the increment in equal parts at every step
(the incremental analysis update) and hands back the state that arrives
at the analysis instant, which carries the analysis's chain; the members
keep direct insertion; the cost is a second integration of the window,
priced in the budget as `iau_reintegration_s`. Both modes are read by the
same instrument: the rms surface-pressure tendency of the first step
after the applied analysis against the last step before it
(`physical_consistency.surface_pressure_tendency_rms_pa_s` in the
receipt's per-cycle assessments), a ratio well above one being the
imbalance the increment inserted.

### The anchor

`--anchor PATH[:key=value;...]` holds the largest scales of the control
toward an external analysis as a weak low-pass constraint, after the
observation increment: `x <- x + w L (x_ext - x)` with `L` one at and
below `full_degree` (default 30), zero at and above `zero_degree` (default
40, a wavelength of about 1,000 km), raised-cosine between, and `w` the
`weight` (default 0.1 per analysis); degree 0 of ln ps is excluded (the
mass fixer owns the global mean). `PATH` is a checkpoint of this
configuration (`valid_utc=` required) or a GRIB analysis (`valid_utc=`
required, `mapping=` optional; decoded through the tree's own analysis
ingest at this truncation). An anchor whose valid time lies more than
`max_age_s` (default 10800) from the analysis instant is not applied and
the record says why. The report's `anchor` block records the source, its
valid and availability times, the affected band, the weight, the power of
the departure inside the band and the increment it produced, separately
from the observation increment. An external analysis already contains
many of the observations the cycle assimilates; offered as
pseudo-observations it would double-count them, which is why it is a
constraint on selected scales and never a stream, and why the door never
claims it makes the cycle unable to drift.

## The DA scorecard

Every analysis report carries a scorecard: for every stream (the row's
source), every variable and every region, the number of rows, the mean,
rms and quantiles of observation minus background (O-B) and observation
minus analysis (O-A), the fraction of rows the analysis moved closer, the
assimilated and the withheld rows separately, the largest time offset,
and a consistency reading. Four assessments are separated:

- **engineering validity**: did ingest, quality control, operators,
  analysis and O-A run for every stream that offered rows. The ONLY hard
  gate: a stream whose rows never reached the operators or has no O-A
  reads INCOMPLETE and the analysis is incomplete.
- **statistical consistency**: the Desroziers estimate of the observation
  error, `sigma_o^2 ~ E[d_oa d_ob]`, against the assigned error (the
  assumptions ride with it: a linear analysis with the optimal gain and
  independent, unbiased errors; the estimate reads the assigned and the
  background error together), the innovation variance against
  `sigma_o^2 + spread_H^2` where the ensemble spread in observation space
  is known, a plausibility band of 0.5 to 2 on the ratios, and whether
  O-A rms fell below O-B rms. Readings, never a gate: an analysis need
  not move closer to every stream (a background of 0 with reports +1 and
  -3 at equal weights analyses to -2/3 and moves away from the +1
  report; a system that fits every report is overfitting), and
  observation errors are not retuned until these diagnostics look right.
- **physical consistency**: the mass-preserving offset, the vapor repair,
  the wind balance, the increment spectrum after the taper, the anchor's
  increment, the recentring shift, the spread before and after, and the
  cycle's surface-pressure tendency reading.
- **predictive value**: the withheld rows' O-B and O-A here (the
  cross-validation reading; for `letkf` the package's own withheld gate on
  the members rides beside it, labelled as a reading), and the
  observation scorecard on the forecast, which is the number of record.

The judgement is made on every row of the stream the operators evaluated,
in the `global` region, against the CONTROL analysis the door hands back;
for `letkf` every offered row that passed the table's own quality control
(gross bounds, age, duplicates), through the control's operators, so a
row the package thinned away still reads, and the package's own
per-stream receipt on the members rides beside it as `ensemble_scorecard`.
The regions are `global`, `nh_extratropics` (20N and north), `tropics`
(20S to 20N), `sh_extratropics` and `conus` (24 to 50N, 125 to 66W).

The cycle door prints the table after every analysis (n, O-B, O-A, the
fraction moved closer, the Desroziers ratio, whether O-A fell below O-B,
the engineering verdict) and the receipt stacks the cards per cycle
(`scorecards`: for every stream and variable, how many cycles were
complete, in how many O-A fell below O-B and which cycles it did not,
with every cycle's O-B and O-A rms and Desroziers ratio in order) and the
four assessments per cycle (`assessments`).
`python -m woof.globe.da_scorecard show REPORT.json` renders one
report's card; `calibrate` plants the ten synthetic families the
instrument is held to in both directions (the conflicting-report family
and the Desroziers recovery of planted errors 1 and 2 among them).

The scorecard is not the observation scorecard
(`woof.globe.obs_scorecard`), which reads a forecast against the
stations and soundings and is the number of record for forecast skill.
This one says whether the analysis ran and how it sits against the
reports it was offered.

The observation streams' `local-tables` route reads a table at a URL as
well as on disk; the file it lands is named by the digest and the URL's
last path segment with the query dropped (`321be4fecc4292d3-asos.py` for
the IEM archive's CSV service), because a query string is the request,
not the file.

## The control twin

Before any real number, the door's control path is exercised on the model
itself: `python -m woof.globe.da_twin --config CONFIG --outdir DIR
--control-truncation T --ensemble-truncation t --members N --cycles C
--family recovery|perfect|agree [--increment-source control|ensemble-mean]`
runs a nature run at the control truncation, starts the control displaced
from the truth with the ensemble built around it at the lower truncation,
draws synthetic reports of every stream from the truth (stations on the
terrain, soundings at the mandatory levels, motion vectors at four
levels), cycles through the door's own filter interface and scores the
control's grid rmse against the truth before and after every analysis.
Four families, both directions: `agree` (reports equal to the control's
own H(x) move it by rounding only, bounded at 1e-10 of the field),
`perfect` (noise-free reports pull the control to the truth at every
analysis on temperature and wind), `recovery` (noisy reports; the rmse
falls from the displaced start to the last analysis) and `mirror`
(reports drawn from the truth reflected through the control, `2 c - t`,
the same network and errors as `perfect` pointing the other way: the rmse
must RISE at every analysis on temperature and wind, so a filter that
damped every increment could not pass `perfect` weakly and hide; on the
smoke case it takes the temperature rmse from 1.317 to 2.695 K over three
analyses where `perfect` takes it to 0.913, the square of the increment
adding in both directions and the cross term flipping sign). The comparison arm
of the design review, `--increment-source ensemble-mean`, runs on the
same truth, network and seed so its curve lies beside the control path's;
the verdict never says which is better, the curves do. On the smoke
configuration at T7 over T3 with six members on the CPU
(`tests/test_arwen_global_da_twin.py`): the agreeing family changes the
control by 2e-16 of the field, perfect reports take the temperature rmse
from 1.317 to 0.913 K over three analyses, and the two recovery arms
finish within 0.001 K of each other on temperature (0.950 against 0.949)
and 0.002 m/s on wind, the control path 4 Pa lower on surface pressure
(99 against 102 Pa); toy scales. On a card (an RTX 5090 shared with another job, 2026-09-06): the T127 L40 native-physics control over
a 16-member T63 ensemble with 1,200 stations, 120 soundings and 600
motion vectors per hourly cycle (11,040 rows, about 9,000 assimilated)
reads the agreeing family at exactly zero change on temperature, wind and
surface pressure (the vapor 9.6e-6 relative, the positivity repair's clip
of a float32 state, recorded beside the bar), and perfect reports take
the control's temperature rmse from 1.734 to 1.257 K over four analyses,
the wind from 2.567 to 2.414 m/s and the surface pressure from 280 to
125 Pa, every analysis improving; the ensemble spread in temperature 0.81
to 0.70 K against a control error of 1.7 to 1.3 K (spread over rmse 0.47
to 0.55, under-dispersive at 16 members). The recorded verdict of that
agreeing run reads `passed false` under the bar of its day (1e-10 on
every field, the vapor included); the per-field record it carries
(`control_change_relative`: temperature, u, v and surface pressure
exactly 0, vapor 9.6e-6 and 7.6e-6) is what this page cites, and the bar
was moved to the analysed fields with the vapor recorded beside them.
What did NOT hold on the card: the `recovery` family on the control path
(noisy reports, 16 members, four hourly cycles) died in the physics one
forecast step after its first analysis, three times out of three
(2026-09-06 02:39, 02:46 and 02:54 UTC): the analysis read
`pass` with the surface pressure inside the gross bounds (51,144 to
107,231 Pa against the truth's 51,280 to 107,324) and the first step
after it drove one column to 1,096.98 hPa, past the RRTMGP pressure
table's 1,096.6 hPa edge; the comparison arm (`--increment-source
ensemble-mean`, the same truth, network and seed) ran its four cycles
(temperature rmse 1.734 to 1.303 K, wind 2.567 to 2.529, pressure 280 to
192 Pa) with a smaller first increment (217 against 165 Pa rmse after the
first analysis). So the control path is demonstrated on the card with
perfect reports and with agreeing reports, and NOT with noisy reports:
the noisy control increment's magnitude and balance at T127 over T63 is
the open defect of this path, the analysed-state bound at the analysis
instant does not catch it (the overshoot happens in the forecast step),
and until it is measured and fixed the ensemble filter is selectable, not
the default. The integration then measured the death's cause on the
same twin (2026-09-06): the T63 truncation's orography undershoots below
sea level at the Andes' Pacific foot, so every T63 member carries 109.2 to
109.7 kPa there by construction and sits within a few hundred pascals of
the radiation tables' 109,663 Pa ceiling before any analysis (one seed
starts a member above it and dies with no analysis at all); the twin now
records the displaced start's headroom and refuses a start above the
ceiling by name, an analysed member above it is named in the report, and
the real arms below ran six and twenty-four hourly cycles of noisy real
reports on T127 members under the T255 control with the incremental
analysis update. So the noisy control path is demonstrated on the real
case, and the T63 twin is not the calibration of record for it. The wall of one hourly cycle:
the members' advance 41 s, the LETKF solve 2 s, the members' O-A
operators 12 s, the control's operators for the binning sensitivity and
the card 15 s, the control's own hour of forecast 11 s; the resident
footprint 73 MB per T63 member (1.09 GiB for 16) beside 1.67 GB of shared
workspace.

## The first real arm, and what ships

Measured on the case (2026-09-06, the merged system): `woof global da fresh` from the GDAS
2026-08-31 18Z analysis, six hourly letkf cycles to 2026-09-01 00Z with the METAR, IGRA2, NDBC and
GOES derived-motion-wind tables of record, 32 T127 members under the T255 control, 600 s bins, the
control's incremental analysis update, no anchor; 71,000 to 78,000 reports assimilated per analysis,
every cycle under 0.17 of real time on the RTX 5090, the analysis handed back 65 minutes after the
first byte. Against the CONUS stations the handed-back analysis reads a 2 m dewpoint bias of +0.01 K
(rmse 2.57) where the GDAS analysis of the same instant reads -2.61 K (3.92), sea-level pressure
2.20 against 2.56 hPa rmse, 2 m temperature 1.75 against 1.90 K; at the soundings it is the GDAS
analysis's equal on 500 hPa height and worse on temperature and wind (W250 4.48 against 3.98 m/s).
The 24 h forecast from it keeps the dewpoint gain (3.94 against the cold start's 4.60 K rmse at 24 h)
and loses sea-level pressure (3.39 against 2.41 hPa, a -2.7 hPa bias: the added boundary-layer vapor
rains out, the initial-state lane's finding on its moisture update reproduced), 10 m wind at 18 h,
500 hPa height at 12 h and the sounding winds. By the rule of the program (not worse than the GDAS
cold start by more than 0.03 K, 0.03 hPa, 0.1 m/s, 1 m on any score, better on two) the ensemble
system ships SELECTABLE: the fresh door's default initial state stays the GDAS cold start and its
default filter the successive correction, and

```
woof global da fresh BASE.toml --outdir DIR --analysis-cycle 2026-08-31T18:00:00Z     --until-utc 2026-09-01T00:00:00Z --stream local-tables:paths=...     --filter letkf --members 32 --ensemble-truncation 127 --observation-bin-s 600     --increment-application iau
```

is the configuration the numbers above belong to. The same system cycled for 24 hours (the second
arm, to 2026-09-01 18Z) does not drift: the METAR surface-pressure O-B rms rises from 121 to 173 Pa
over the first twelve analyses and returns to 144 Pa over the next twelve with the O-A between 72
and 89 Pa throughout, and the 18Z analysis after a day of cycling reads closer to the 18Z stations
than the GDAS 18Z analysis on temperature (2.43 against 2.56 K rmse), dewpoint (2.78 against 3.79 K)
and sea-level pressure (1.94 against 2.58 hPa, a -0.9 hPa bias where the v1 door's cycle read -3.3);
its 500 hPa height bias at the soundings is 0.0 m six hours later where every cold-started column
carries -6 to -8 m, and the 250 hPa wind stays its weakest row. The full scorecards, the per-cycle
cards and the increment maps are in the integration report of 2026-09-06.

The same door graded once more on the merged tree after the six refutations were folded in (the tree
the owner carries): `fresh` from the GDAS 2026-09-01 18Z analysis, six hourly cycles with the seven
hourly tables of record to 2026-09-02 00Z (the latest hour the streams cover; a window needs its own
hour's table and the previous one, each table spanning half an hour either side of its hour), 73,000 to
100,000 reports per analysis, 592 s of analysis budget per cycle (0.176 of real time at the worst),
4,128 s from the first byte to the handed-back analysis, then the 24 h forecast beside a T255 cold start
from the GDAS 2026-09-02 00Z analysis of the SAME initialisation time. Against the CONUS stations (the
2026-09-02 18Z and 2026-09-03 00Z sets fetched for the grade) and the IGRA2 levels: better than the cold
start on 2 m dewpoint (2.96 and 3.53 against 3.91 and 4.32 K rmse at 18 and 24 h) and on 2 m temperature
(2.69 and 2.77 against 2.84 and 3.13 K), worse beyond the rule on sea-level pressure (2.63 and 3.33
against 2.39 and 2.62 hPa, the forecast's bias reaching -2.5 hPa by 24 h), on 10 m wind (2.01 and 2.04
against 1.74 and 1.91 m/s), on 500 hPa height at 12 h (16.90 against 14.24 m) and on every sounding
temperature and wind row; the handed-back analysis beats the GDAS analysis of its own instant at the
stations (2 m temperature 2.10 against 2.35 K, dewpoint 2.60 against 4.10, sea-level pressure 2.44
against 2.79 hPa) and not at the soundings (500 hPa height 15.67 against 14.33 m). The verdict above
stands: selectable, the same losses by stream.

## The wall budget

Every cycle's receipt row carries `budget`: the wall of the forecast steps
since the previous analysis, the stream fetch, the background's identity,
the analysis (with the members' advance, the control solve, the ensemble
analysis and the IAU re-integration priced separately) and the checkpoint
publish, against the interval, as a fraction of real time. The cycle door
prints it after every analysis (`wall 44.1 s of a 3600 s interval (0.012
of real time)`) and the receipt summarises it (`mean_wall_s`,
`max_wall_s`, `max_real_time_fraction`, `every_cycle_keeps_up`). A cycle
above 1.0 of real time is falling behind the clock it is meant to follow.

## Where the analysis runs

The letkf analysis runs on the card the members live on, with nothing
set: the point operators, the local report gather, the Gaspari-Cohn
weights and the observation-error scaling, the batched
eigendecomposition and the increment assembly, for the members and for
the control through the same path. `--letkf-solve-path {auto,device,host}`
(default `auto`: the members' own array namespace, the card when the model
is on one) selects it; `host` is the numpy reference the device path is
compared against, the same code in the other array module, kept for the
bit comparison and for a machine without a card. `--operator-precision
{state,float64}` (default `state`) says how the point operators contract a
device-resident state: a float32 state's coefficients through float32
GEMMs over 256-term blocks whose partial sums are added in float64, the
state's own precision; `float64` forces the float64 contraction, the host
path's arithmetic to rounding. Every analysis report's `letkf` block names
the path taken (`path`), the eigensolver, the number of column chunks and
level batches, the widest local report count and the count dropped at
the cap, and its wall (`wall_seconds`, split into gather, setup, solve and
finish); the `operators` block names the operators' path and precision
and the rule they follow. The receipt's form did not change.

The solve is batched over columns AND levels: a chunk of latitude rings
gathers its reports once, its columns are taken in sub-chunks, and every
plane of a sub-chunk (the 40 levels of the 3-D fields and the surface
plane the 2-D field sits at) enters one eigendecomposition batch of the
project's Jacobi kernel, so the card factors thousands of 32 x 32
matrices per launch instead of one plane's 768. A column with more
reports inside its lens than the cap keeps the largest weights, ties
broken by report index, so the choice at the cap is the same on either
path (before that rule the two paths' increments differed by up to 0.41
of a field's maximum on the case, because 91 percent of the case's
reports share a position with another and tie in horizontal weight).

MEASURED 2026-09-07 on the case of record (the GDAS 2026-08-31 18Z
analysis, six hourly letkf cycles to 2026-09-01 00Z, the four hourly
tables, 32 T127 members under the T255 control at `imex_ssp3` and 50 s,
600 s bins, the control's incremental analysis update; 71,000 to 78,000
reports per analysis; an RTX 5090 shared with other jobs), the
same configuration before the change (the DA system's arm of record, run
on the DA system lane's tree 7c47e835b, that lane's last tree before the
door and ensemble refutations were merged into it, the whole reaching the
owner as 4ead780a4; its operator and solve code is unchanged from there to
the tree this lane started from, d3fb08e5e) and after it (the lane tree
c539b331f), means of the six cycles:

| component | before (s) | after (s) |
|---|---|---|
| members' hour of forecast, with the in-window H(x) | 294.5 | 283.8 |
| control IAU re-integration | 40.3 | 41.6 |
| control H(x) for the card | 49.6 | 2.5 |
| control operators (innovations) | 50.0 | 2.8 |
| members' O-A operators | 53.6 | 2.4 |
| LETKF localised solve | 30.2 | 16.6 |
| everything else (QC, apply, recentre, inflation, receipt) | 2.4 | 2.1 |
| the analysis step (sum of the above) | 520.6 | 351.9 |
| the analysis proper (the step less the members' forecast and the IAU re-integration) | 185.8 | 26.4 |
| the whole hourly cycle (control forecast, fetch, analysis step, checkpoint) | 556.8 | 390.5 |

Six cycles took 3,886.5 s before and 2,608.7 s after; the worst cycle
607.8 s (0.169 of real time) before and 424.1 s (0.118) after. The
LETKF solve's own wall on the card read 15.6 to 17.4 s per cycle (path
`device`, 915 level batches in 60 chunks, 17 Jacobi sweeps at most). The
hour of member forecasts is now 0.73 of the cycle: the analysis is no
longer where the time goes.

The device path against the host path on the first analysis of that run
(75,385 reports; the captured solve inputs replayed on both paths in one
process, MEASURED 2026-09-07 on the RTX 5090 and again on an RTX
5070 Ti with the same result to every printed digit): the ensemble
increments differ by at most 1.8e-5 m/s in u and v, 2.1e-4 K in theta,
1.4e-8 kg/kg in qv and 1.8e-5 in ln ps, which is 2.3e-6 of the increment
field's maximum for u, v and qv, 3.6e-5 for theta and 2.7e-3 for ln ps
(about twenty units of float32 roundoff on a ln ps of 11.5, the state's own
precision); the rms of the difference is 2.0e-3 of the field's rms for ln
ps and at most 1.9e-5 for the rest; the control increment differs by at
most 1.0e-7 of its field maximum in u and v and by 3e-9 or less in theta,
qv and ln ps. The receipt values the two paths print (the increment rms
per field, the prior and posterior spread) agree to six significant
figures or better; the active column and point counts are equal; the
widest local report count reads 3,294 against 3,292 and the count dropped
at the cap 24,964,427 against 24,964,408: the two paths' lens counts differ
by the reports within about 5e-4 of the cutoff (under 700 m of 1,200 km),
where the weight function, which vanishes as the fourth power of the
distance to the cutoff, is the polynomial's rounding noise (1e-16 to
1e-12) and reads 0.0 on one array module and 5e-16 on the other at the
last ulp of the distance; measured on the case, 280 of the 73,728 columns
differ by one or two such reports, and a report with that weight moves
nothing the increment can show. The replayed solve took 16.7 s
on the RTX 5090 (20.9 s a second time with three other jobs on the
card) and 32.9 s on the RTX 5070 Ti; the host path 316 s on a 16-core host
at eight threads and 740 s on a loaded 24-core host. The difference maps
and the wall charts are in the evidence gallery of 2026-09-06.

What this measurement did not do: the six cycles ran on the Eulerian
control of the configuration of record (the same configuration as the
before run, so the walls compare like with like); the same six cycles on
the semi-Lagrangian core at 300 s were started three times and did not
finish (the members re-cut to 600 s by the Eulerian rule failed the
trajectory gate in the third hour and then the Lipschitz gate at 0.7689
against 0.75, fixed by keeping a coarser truncation's step under a
semi-Lagrangian integrator; at 300 s a member's surface pressure passed
the radiation tables' 109,663 Pa ceiling in the second hour, 109,764 Pa,
and the cycle stopped by name). The letkf cycle on the semi-Lagrangian
members is therefore not demonstrated here.

## Lineage, receipts and the information cutoff

An analysis checkpoint carries its own history in the physics metadata
(`assimilation_history`): one link per analysis with the instant, the
background checkpoint's digest, the report counts, the filter and the
streams that fed it, pruned to what a later cycle could still be offered;
a report the chain holds is never assimilated again. The report's
`lineage` and the DA receipt's repeat the last link and the chain length;
the ensemble manifest carries the same summary. Every receipt
(`da-receipt.json`, schema `gpuwm.arwen-global-da/v1`) is self-hashed and
names the door that wrote it, the config hash, the analysis checkpoint
handed back with its digest, the scorecards, the assessments, the budget,
the settings (bin, insertion, anchor, control options, additive
inflation) and the causal record (`causal`: the information cutoff, the
latency classes of every fetched object, and whether the run was fast
hourly, a delayed replay, or a historical case whose arrival times are
unverified).

Exit codes: 0 when the leg completed (status `pass`, or `incomplete` with
the streams named), 1 when the cycle's own gates failed or no analysis
could be handed back, 2 for a refusal (a missing file, an output that
exists without `--overwrite`, a stream or filter the table does not carry,
a fresh with nothing to cycle yet, a cutoff in the future).

## The legs one by one

```
woof global da init CONFIG --outdir DIR [--from-checkpoint CKPT] [--members N] [--filter F]
woof global da cycle CONFIG --outdir DIR --cycles N [--stream S]... [--obs T]... \
    [--interval-s 3600] [--start-utc T] [--ensemble DIR/da-ensemble.json] [--until-s S] \
    [--observation-bin-s B] [--increment-application direct|iau] [--anchor SPEC]
woof global da analyze CONFIG CKPT --obs T --out DIR [--analysis-time T]
woof global da fresh BASE.toml --outdir DIR [--stream S]... [--analysis-cycle T] \
    [--until-utc T] [--cutoff-utc T] [--forecast-hours H]
woof global da forecast CONFIG --analysis CKPT --outdir DIR [--until-s S]
```

Every option is listed with its help in `CLI-OPTIONS.md` under
`woof global da`. The same five legs answer as
`woof global da ...` with the same arguments.

## Interface record

The door builds against the ensemble package and the observation streams
through code, and the decisions are recorded here so a later reader knows
what was agreed and where:

- the filter interface is `woof.globe.da_filter`
  (`resident_states`, `begin_window`, `observe`, `analyse`, `init`,
  `attach`, `manifest`): the door steps the control through its model
  between analyses, tells the filter when a window opens (with the rows
  it fetched for it) and after every step, hands the filter the window's
  reports with the control's model, transform and state, and takes back
  the control analysis and a report in the door's shape (`status`,
  `gate_of_record`, `variables`, `scorecard`, `assessments`, `lineage`,
  the totals and rejections);
- the control's analysis is the ensemble package's (`analyze_ensemble`
  with a `ControlBackground`: one local solve returns the members'
  increments and the control's from the control's own innovations, the
  control increment tapered by degree and embedded; the package's
  interface notes, decision 8 as amended, are the contract). The door's
  `woof.globe.da_control` carries its spelling of the settings
  (`ControlOptions`, laid over the package's `FilterOptions`) and the
  comparison record; the door built a second-solve control analysis
  first, against the package's then-committed primitives, and retired it
  when the package landed the one-solve form;
- the window's streams are fetched when the window OPENS and the analysis
  window is the trailing `(t - interval, t]`, because the trajectory has
  to know where and when the reports are before it passes them; the
  observation lane's `analysis_window` is centred and belongs to its
  table-cutting tool, not to the door's real-time cycle;
- the window is the package's `ObservationWindow` (batches built when
  the window opens, observed column by column at the end of the bin a
  report falls in, the control through `control=True`); the door chooses
  the bin (`--observation-bin-s`, the whole window when unset), finishes
  both trajectories at the analysis instant, hands rows offered inside
  the window but not batched (a table given after the window opened) as
  batches the package evaluates at the instant, refuses rows outside the
  window by count (a report is compared in the window it falls in and
  nowhere else; `rejections.outside_window` and `observation_times.
  rows_outside_window`; before the 2026-09-06 refutation they were
  analysed at the instant, and the rows one cycle thinned away came back
  the next; only a cycle with no window takes the rows inside the
  package's age window at the instant), carries a window with nothing
  admissible by name, and measures the binning sensitivity on the control
  once per cycle;
- two manifests, layered: the door's `da-ensemble.json` records the filter,
  the control checkpoint, the member checkpoints, the lineage, the
  `EnsembleOptions`, `FilterOptions` and `ControlOptions` the members were
  built under, and for letkf points at the package's own
  `arwen-global-ensemble.json` under `ensemble/`; `attach` rebuilds the
  filter from them, the door's shared settings (the withheld gate, the
  age window, the wind balance, the recentring fraction and the taper
  degrees) laid over the filter's and an explicitly named
  additive-inflation fraction laid over the ensemble's; the members keep
  direct insertion (the package's own incremental option stays at its
  default) while `--increment-application iau` governs the control;
- the letkf report's `status` is engineering validity; the package's
  withheld gate on the members rides in `predictive_value` and never
  carries the background or withdraws a variable (the partial rule stays
  the successive correction's own);
- the anchor is not a stream: `STREAM_TABLE` never carries an
  analysis-pseudo-observation entry, and the observation lane's
  `analysis-pseudo` stream is superseded by `--anchor` (amendment E);
- a stream is a `da_streams.STREAM_TABLE` entry whose `fetch(window_start,
  window_end, out_dir)` returns `FetchRecord`s (with first receipt time,
  publication time where known and latency class) for files an obs-table
  decoder entry reads; the door decodes a file once (by digest) however
  many windows re-record it; the neutral `gpuwm-obs.table.v2` files the
  Rust observation doors write (and the ten-column v1) read through
  `local-tables` unchanged, their `received_time` feeding the row-level
  information cutoff and the latency vocabulary shared with
  `woof.globe.obs_streams`;
- the observation vocabulary is `obs_table.VARIABLE_TABLE` (surface
  pressure, temperature, u and v wind, dewpoint; the observation lane
  adds refractivity, whose operator is a `PointObs.operator` a stream
  supplies, refused by name until it does) and the scorecard groups by
  the row's `source`; an aloft row carries its pressure level;
- the analysis checkpoint is the runner's checkpoint (schema v3) with the
  chain in the physics metadata; `init` writes the control state under
  its own name (`da-initial-state.npz`) so the cycle that restarts from
  it does not have to overwrite it; with `iau` the handed-back state is
  the re-integrated one and carries the same chain;
- the mass convention of the pressure increment is stated in the report
  (`mass_preservation` for the successive correction, the control
  record's `mass_preserving_log_offset` for letkf): the global mean of
  surface pressure is preserved, so a pressure bias every station shares
  is removed rather than analysed, and the amount removed is recorded.
