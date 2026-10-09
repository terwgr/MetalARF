# ABI brightness temperature as a forward operator for WOOF global

`src/arwen_global/abi_operator.py` (the SimSat measurement, the entries, the
four assessments), `abi_reference.py` (the columns, the reference, the
scorecard), `abi_fast_model.py` (the fast model's table and trainer),
`abi_radiance_operator.py` (the stream for the ensemble filter),
`tools/crtm_reference/` (the CRTM driver), `tools/rustwx/crates/rw-goes/src/radiance.rs`
(`rw_goes bt`, `colocate`, `quicklook`) and `forward.rs` (`rw_goes forward`, the
operator). Doors: `woof global abi-score`, `abi-reference`, `abi-fast-model`.
Tests: `tests/test_arwen_global_abi_operator.py`, `test_arwen_global_abi_reference.py`,
`test_arwen_global_abi_fast_model.py`, the crate's own.

Status, 2026-09-06: **the clear-sky operator ships for both bands over water;
land is reported by term.** The operator is `rw_goes forward`, a Rust
fast transmittance model trained on CRTM (the numerical reference of the
design's amendment H) and marching the emission radiative transfer with
GOES-19's own Planck constants. Against the real GOES-19 18:00Z full disk on
the GDAS 18Z analysis, graded on the blocks the stream hands the filter (the
stream QC of section 2: at least 12 both-clear pixels, at least half the
block both-clear, zenith 60 degrees or less; one block one row, no pixel
weighting): band 13 over water 7,417 blocks, bias +0.005 K, rmse 0.555 K,
0.554 K after the linear correction (gate 1.5 K); band 8 over water -0.63 K,
1.39 K, 1.214 K after. Over land both bands miss the gate (3.06 K and
1.83 K after) and the reference misses it by the same amount (3.05 and
1.84 K), so the land term is the analysed skin and the analysed upper-level
humidity, not the operator. The first grade of this lane was pixel-weighted
over every both-clear block, cloud edges included (0.607 and 1.234 K after
the correction, 15,753 blocks); that record rides beside the filter-facing
one in the entries, and the refutation that replaced it is in section 4.
SimSat's own renders, the first measurement of this lane, remain the
imagery path and the record of where its optics stand: +2.7 K (band 13) and
+7.4 K (band 8) warm against CRTM on the same columns.

## 1. What the operator is

A clear-sky column operator: for one model column (40 layers of temperature
and vapor on the hybrid coordinate, the surface pressure, the skin, the
surface class) and one satellite zenith angle it returns the ABI band 13
(10.3 um window) or band 8 (6.2 um upper water vapor) brightness
temperature, the emissivity it used, the radiance, and central-difference
Jacobians in every layer's temperature and vapor and in the skin.

* **Transmittance**: per model layer a nadir optical depth from named
  features of the column (`abi_fast_model.FEATURE_NAMES`: the layer's
  thickness, vapor path `u`, vapor pressure proxy, ln p, temperature, the
  slant vapor path from the top of the atmosphere to the layer middle, the
  path-weighted temperature and pressure above, ln sec zenith, and the
  neighbouring layers' vapor and temperature). Band 13 is linear in eleven
  products of them (dry and continuum terms add); band 8 is
  `exp(wet polynomial of 35 terms) + dry (4 terms)`, fitted by damped
  Gauss-Newton on the log residual so the emitting layers are fitted to
  relative accuracy. Every feature is held inside the range the training
  set spanned (per layer), so a column outside the envelope extrapolates
  no further than its edge.
* **Radiative transfer**: the emission march CRTM's clear-sky solver
  takes, reproduced from CRTM's own layer optical depths to 0.005 K rms in
  band 13 and 0.033 K in band 8 (the remainder is the stratosphere above
  1 hPa that CRTM adds and the operator does not carry): upwelling layer
  emission on the slant path, the surface's emission and its specular
  reflection of the downwelling along the same path.
* **Source function**: the band-corrected Planck relation of the Level 1b
  granule itself, `L = fk1 / (exp(fk2 / (bc1 + bc2 T)) - 1)`, with GOES-19's
  constants (band 13: fk1 10860.40, fk2 1395.19, bc1 0.07481, bc2 0.99975;
  band 8: 50451.5, 2327.96, 1.71389, 0.99631), so the instrument the
  operator is compared with is the instrument it inverts.
* **Surface emissivity**: per column from the table, water 0.981 (band 13)
  and 0.960 (band 8), the Nalli model's mean on the case; land by the
  model's own MODIFIED_IGBP_MODIS_NOAH class from CRTM's IGBP table (16 of
  20 classes seen on the case, the others take the land mean); snow-covered
  land and sea ice by their own values. Or, per column, the emissivity the
  columns stream carries (`--emis-mode 1`).
* **Jacobians**: central differences, 0.1 K in a layer's temperature, 1 %
  of a layer's vapor, 0.1 K in the skin; against CRTM's K-matrix on the
  27,713 case columns the skin Jacobian differs by 0.0036 rms (of 0.63),
  the layer temperature Jacobian by 0.0009 (of 0.018) in band 13 and 0.0064
  (of 0.064) in band 8, the vapor Jacobian in its relative form
  (dTb/dln q) by 0.02 (of 0.29) and 0.07 (of 0.54).

The table (`gpuwm-da.abi-fast-model.v1`) carries the coordinate it was
trained on (a_half, b_half, their sha256), the feature definitions, the
per-layer coefficients and clip ranges, the Planck constants with their
granule, the emissivity table, the reference's provenance (CRTM version and
commit, coefficient file hashes, the tarball's identity), the training split
and the validation numbers, and the reference's Jacobian summary (the
weighting function) for the acceptance contract. The Rust refuses a column
set on another coordinate, naming both.

## 2. The doors

```
# the reference leg: columns under the both-clear blocks, then CRTM, then the scorecard
woof global abi-reference columns --blocks 13=band13-blocks.csv --blocks 8=band08-blocks.csv \
    --config CASE.toml --checkpoint analysis.npz --tapes "tapes/*/wrfout_d01_*" --valid 2026-09-01_18:00:00 \
    --emissivity 13=0.99 --emissivity 8=0.99 --out columns.bin           # .json and .npz sidecars beside it
tools/crtm_reference/abi_crtm_reference abi_g16 COEFFS/ columns.bin g16_own.bin 0 IGBP.IRland.EmisCoeff.bin 8 13
woof global abi-reference score --blocks 13=x --blocks 8=x --columns columns.bin.npz \
    --run g16_own=g16_own.bin --run g16_simsat=g16_simsat.bin --primary g16_own --simsat-emissivity-run g16_simsat --out score/

# the operator: train once per coordinate, run on any columns stream, score like the reference
woof global abi-fast-model train --columns columns.bin.npz --reference g16_own.bin \
    --pack 13=band13.goespack --pack 8=band08.goespack --form 13=linear --form 8=two_term --provenance ref.json --out table.json
rw_goes forward --columns columns.bin --table table.json --out fast.bin --emis-mode 0 --threads 8   # or: woof global abi-fast-model forward
woof global abi-reference score ... --run fast_own=fast.bin --run g16_own=g16_own.bin --primary fast_own \
    --table table.json --reference-run g16_own --out score-fast/
    # writes operator-entries.json (what the scorecard admits, graded on the stream's population, the Jacobians and the
    # brightness temperatures compared with the NAMED reference run) and the four assessments; --reference-run is required
    # with --table, because an entry compared with an arbitrary other run says nothing about the reference

# the SimSat measurement (imagery on the exact ABI lattice, the first arm of this lane)
woof global abi-score CONFIG CHECKPOINT --start-date ... --goes-rad 13=... --goes-rad 8=... --goes-acm ... --out OUT
```

Exit codes: `abi-reference score` with `--table` exits 0 when at least one
class is admitted, else 1; without it 0 when the primary run passes the gate
on every band; a class that measured nothing reads INCOMPLETE, never PASS.

### Interface decisions (for the ensemble and door lanes built beside this one)

* **Columns stream** `gpuwm-da.abi-columns.v2` (`abi_reference.write_columns`,
  read by the Fortran reference and the Rust operator alike): little-endian,
  int32 magic `ABIC`, version 2, n, nlay, n_user_chan; float64 a_half_pa and
  b_half (nlay+1 each, top first); int32 user channels; float64 lat, lon,
  zenith, land_fraction, skin, psfc_hpa, wind10, snowh, seaice (n each); int32
  climatology, land_type (n each); float64 p_half (n, nlay+1), p_full, T,
  q [g/kg], O3 [ppmv] (n, nlay each), top first; float64 emissivity (n,
  n_user_chan). A JSON sidecar carries the coordinate sha256 and the
  consistency checks.
* **Operator output** `gpuwm-da.abi-crtm.v1` (the reference driver and
  `rw_goes forward` write the same stream; `abi_reference.read_crtm_output`
  reads it, refusing a length that does not match its header): int32 magic
  `ABIR`, version 1, n, nlay, nchan, emis_mode, refused; int32 channels;
  float64 bt, emissivity, radiance, jac_tskin, surface_planck [chan][n];
  jac_t, jac_q [K per g/kg], layer_od (nadir) [chan][n][nlay].
* **Table** `gpuwm-da.abi-fast-model.v1` (`abi_fast_model.write_table`,
  above). `rw_goes --abi` names `gpuwm-da.abi-forward.v1` (the forward
  receipt) beside the pack schemas; `woof/obs/frontdoor.py` pins it.
* **The stream QC** (`abi_reference.STREAM_QC`, one place): a block becomes
  an observation only with at least 12 both-clear pixels, at least half of
  its paired pixels both-clear, and a zenith of 60 degrees or less. The
  clear-fraction gate is the cloud-edge gate: on the case the blocks that
  are 1 to 15 percent clear under the ACM mask read the cloud the mask
  missed (sim minus obs +8 to +12 K over the Peru stratocumulus deck), not
  the operator, and with them the water class's unweighted residual was
  0.85 K where the pixel-weighted number read 0.61 K. The entries are graded
  on exactly this population, and a batch built under another QC is refused
  by name.
* **The stream for the filter** (`abi_radiance_operator`): stream
  `goes-abi-l1b-bt`, variable `brightness_temperature_k`, one `PointObs`
  batch per band from the block table (`superobs_from_blocks`: the both-clear
  block means under the stream QC, identity `abi<band>:<x>:<y>`, the block's
  time inside the scan from its row, the rejections counted by reason). With
  the band's operator entry and a land fraction per block (an array aligned
  with the table, or a callable of latitude and longitude, refused when
  missing) the batch keeps only the classes the entry admits, carries the
  entry's per-class observation error, and returns the entry's per-row
  linear correction in its extras; `register_batch` binds the zenith angles
  and the correction to the operator by row identity, and
  `AbiRadianceOperator` returns `intercept + slope * bt` for those rows so
  the filter's O-B is in the frame the entry was graded in (the receipt
  counts the rows corrected). `operator = AbiRadianceOperator` (members to
  columns by spectral sampling at the points, `rw_goes forward`, back to
  `(R, n)`). Release-1 vertical placement: the reference weighting function's
  median peak (947 hPa band 13, 357 hPa band 8) with the band's own cutoff,
  labelled as such; the Jacobians ride in the output for the model-space
  localisation of release 2. The module imports the ensemble lane's
  `woof.globe.da.observations.PointObs` lazily; the stream tests run
  against that contract where it is merged and against a stand-in with the
  same fields elsewhere.
* **Entries** `gpuwm-da.abi-operator-entries.v1` (`abi_operator.fast_operator_entries`):
  per band and per surface class the verdict (ADMITTED, NOT ADMITTED,
  INCOMPLETE) judged on the `filter_facing` row (the stream QC population,
  unweighted; a row that measured nothing reads INCOMPLETE), with that row's
  correction `obs = a + b sim` and residual as the observation error the
  admitted class carries, its residual shape after the correction (skew,
  excess kurtosis, the fraction beyond two and three sigma against the
  Gaussian's, the correlation with zenith and latitude), the pixel-weighted
  statistics over every both-clear block beside it as the record of the
  whole scan, the reference-minus-operator brightness temperature per class,
  the transmittance model's validation (labelled as the transmittance model
  alone, at the reference's own emissivity), the Jacobian agreement with the
  NAMED reference run (`reference_run`; refused without one), the stream QC,
  and the acceptance contract (measurement, time, location, vertical
  coordinate, representativeness, bias treatment, error correlations).
* **Four assessments** (`abi_operator.four_assessments`) in the score
  receipt: engineering validity (every run evaluated every column),
  statistical consistency (at least one class admitted on the stream's
  population; the per-class residual shapes ride in the facts, and the facts
  say what is not tested: Desroziers and the spread need an analysis and an
  ensemble), physical consistency and predictive value NOT MEASURED here (an
  operator changes no state; both are assessed where the analysis applies
  its increment).
* Earlier decisions stand: the simulated plane sidecar
  `gpuwm-da.simsat-plane.v1`, the pack `gpuwm-obs.goes-bt.v1`, the
  colocation record `gpuwm-da.abi-colocation.v1` and its block table (the
  superobservation vector), the colocation by lattice index and never by
  interpolation.

## 3. The reference (CRTM v3.1.1 on the analysed columns)

Built from JCSDA's public repository (tag v3.1.1, commit
bd262087) with gfortran 15.2.0, coefficients streamed from the public
`fix_REL-3.1.1.2.tgz` (7.85 GB, hashed; only the ABI, IR emissivity and
cloud/aerosol files kept; `tools/crtm_reference/README.md` has every hash).
GOES-19 has a public SpcCoeff but no TauCoeff, so the reference runs the
GOES-16 pair; the GOES-18 pair differs by -0.045 K (band 13) and -0.21 K
(band 8) on the case, the sensor term, folded into the measured correction.

The columns: the both-clear blocks of the colocation at zenith 70 or less
(27,713 blocks), the analysed atmosphere at the nearest 0.25-degree tape
point (PB, theta, vapor; the half levels rebuilt from the model's own a/b at
the tape's surface pressure agree with the tape's full levels to 1.1e-7
relative), the surface from the checkpoint's own planes at the nearest
Gaussian node (land fraction agrees with the tape's land mask on 99.3 % of
blocks; the checkpoint skin against the regridded tape skin reads 0.01 K mean,
1.1 K rms, the bilinear regrid), US-standard ozone, CRTM's climatology by
latitude and season above the model top.

The reference's own two-direction calibration (`reference-calibration.json`):
an identity column set reproduces every brightness temperature exactly;
skin +2 K reads +1.21 K in band 13 and the K-matrix predicts 1.207 (rms
difference 0.0036 K), skin -2 K reads -1.20 K; upper vapor x1.5 above
500 hPa reads -2.93 K in band 8 and x1/1.5 +2.91 K (the linear K-matrix
prediction -3.60 and +2.40, the nonlinearity of a strong band), band 13
-0.14 and +0.07 K. `RTSolution%Layer_Optical_Depth` is the nadir optical
depth of the user layers (the plain emission march with it times the secant
reproduces CRTM's BT to 0.005 and 0.033 K rms; without the secant it misses
by +1.2 and +5.1 K), which is what the fast model is trained on.

### The clear-sky gap by term (both-clear blocks, zenith 60 or less)

Pixel-weighted block statistics, simulated minus observed; "after" is the
rmse after the least-squares `obs = a + b sim`.

| band | class | n blocks | SimSat vs obs | CRTM vs obs | SimSat vs CRTM at 0.99 | emissivity term (0.99 vs CRTM surface) |
|---|---|---:|---|---|---|---|
| 13 | water | 15,753 | +2.93, rmse 3.13, after 0.97 | +0.16, 0.60, after 0.58 | +2.66, 2.85, after 0.74 | +0.10 |
| 13 | land | 8,107 | +5.66, 7.00, after 3.43 | +1.04, 3.58, after 3.14 | +3.61, 3.96, after 1.48 | +1.01 |
| 8 | water | 15,753 | +6.74, 7.11, after 1.48 | -0.61, 1.39, after 1.22 | +7.35, 7.52, after 0.73 | 0.00 |
| 8 | land | 8,107 | +7.79, 8.25, after 2.00 | -0.24, 1.98, after 1.92 | +8.02, 8.17, after 0.59 | 0.00 |

So: over water the analysed SST and lower troposphere reproduce the real
band-13 radiances to 0.6 K rms with CRTM, and SimSat's +2.7 K is its own
window absorption and source function (the continuum too weak and the
single-wavelength Planck), not the state. Over land the reference itself
misses by +1.0 K with a 3.1 K residual and slope 0.77: the analysed midday
skin (the GDAS surface temperature at 18Z over the Americas) is the term, and
the gray emissivity adds another +1.0 K to SimSat's. In band 8 the reference
reads the analysed upper-level humidity against the instrument at 1.2 K
(water) to 1.9 K (land) after the fit, a real diagnostic of the GDAS moisture
field on the T255 grid, and SimSat's +7.4 K is entirely its weighting
function (the gray coefficient puts the emission level too low).

## 4. The fast model against the reference and against the instrument

Trained on the 13,638 even-quarter-degree columns, validated on the 14,075
odd ones (the table's `validation`, against the reference's own optical
depths through the same march, so the number is the transmittance model
alone): band 13 rms 0.081 K (bias -0.009, p99 0.25, max 0.45; by zenith
0.05 to 0.15 K); band 8 rms 0.207 K (bias 0.005, p99 0.59, max 10.1 K in one
column at the envelope's edge; 0.15 K at low zenith, 0.32 K at 60 to 70).
The neighbour-layer features are what brought band 8 from 1.55 K (local
features only) to 0.68 K (with the slant path above) to 0.21 K: CRTM's ODPS
interpolates the profile onto its own levels, so a layer's optical depth
depends on its neighbours' vapor.

The operator as shipped (the table's emissivity by class, not the
reference's per-column value) against the reference's brightness
temperatures on the same columns: band 13 0.22 K rms over all 27,713 columns
(0.15 K at zenith 60 or less; +0.26 K bias at 60 to 70, outside the admitted
band; 158 columns beyond 1 K, the largest 1.7 K), band 8 0.21 K (0.17 K at
zenith 60 or less; 30 columns beyond 1 K, one edge column at 10.2 K). The
difference between the 0.08 K of the transmittance validation and the
0.22 K of the operator in band 13 is the emissivity table: the reference's
sea-water emissivity varies with wind and angle (standard deviation 0.012)
and the table carries its mean. Away from each column's own angle the
operator follows the reference's zenith dependence to 0.14 K (every column
at 10 degrees: the reference warms by 1.36 K, the operator by 1.23 K; at 55
degrees -1.17 against -1.20 K).

`rw_goes forward` on the 27,713 columns, both bands, with Jacobians: 2.8 s
on 8 threads. Against the instrument, on the population the stream hands
the filter (`operator-entries.json`, `score-fast/`; the stream QC of
section 2, one block one row, unweighted):

| band | class | n | bias | rmse | fit | after | verdict |
|---|---|---:|---:|---:|---|---:|---|
| 13 | water | 7,417 | +0.005 | 0.555 | -2.822 + 1.0096 sim | **0.554** | ADMITTED |
| 13 | land | 5,311 | +1.188 | 3.567 | 72.162 + 0.7570 sim | 3.056 | NOT ADMITTED (the analysed skin) |
| 8 | water | 7,417 | -0.633 | 1.394 | 12.529 + 0.9505 sim | **1.214** | ADMITTED |
| 8 | land | 5,311 | -0.246 | 1.893 | 15.606 + 0.9373 sim | 1.834 | NOT ADMITTED (the analysed humidity) |

The reference on the same population: band 13 water 0.528 K after the fit,
land 3.052; band 8 water 1.202, land 1.844. The pixel-weighted statistics
over every both-clear block (15,753 water, 8,107 land, zenith 60 or less)
read 0.607 and 3.143 K (band 13), 1.234 and 1.909 K (band 8) after the fit,
and ride in the entries as `pixel_weighted_all_blocks`.

**The refutation that replaced the first grade.** The first entries were
graded pixel-weighted over every both-clear block, cloud edges included, and
their observation error (0.607 K) was not the error of the rows the stream
hands the filter: unweighted, at the stream's own 12-pixel threshold, the
band 13 water residual after the correction was 0.70 K (0.85 K with the
1-pixel blocks in), its bias +0.22 K, its skew -7.9 and excess kurtosis
270, and the worst residuals were blocks 1 to 15 percent clear under the
ACM mask over the Peru stratocumulus deck (sim minus obs +8 to +12 K, the
cloud the mask missed). The clear-fraction gate removes them (6,751 water
blocks below half clear, 1,585 below 12 pixels), the bias goes to +0.005 K
and the residual to 0.554 K, with a split-half cross-fit of 0.58 / 0.63 K
showing the in-sample fit is not optimistic. The tails that remain are
reported, not hidden: 1.3 percent of band 13 water residuals beyond three
sigma against the Gaussian's 0.27 (excess kurtosis 26), 1.0 percent in band
8 (kurtosis 7), and a correlation of the band 8 residual with latitude of
0.24 and with the column vapor of 0.27 (the analysed upper humidity, the
worst rows fully clear blocks off Brazil at 20 S, 39 W where the analysis
reads 8 to 9 K too warm in band 8). The first entries also compared the
operator's Jacobians with the first other run of the score (the operator
itself under the columns' emissivity), not with the reference; the entries
now name the reference run and refuse to be built without one.

The operator's own two-direction calibration through `rw_goes forward` on
the planted column sets (`fast-calibration.json`): identity exactly zero in
both bands; skin +2 K reads +1.210 K in band 13 (reference 1.211, rms
difference 0.007) and -2 K reads -1.203; band 8 0.0009 K; upper vapor x1.5
reads -2.77 K in band 8 (reference -2.93, rms difference 0.21), x1/1.5
+2.88 K (reference +2.91), every column with the right sign; band 13 -0.13
and +0.08 K (reference -0.14 and +0.07).

What is admitted: both bands, the water class, zenith 60 or less, under
the stream QC, with the measured linear correction and the after-correction
residual as the observation error (0.55 K and 1.21 K). Land waits on the
state: a skin that the analysis does not carry well at 18Z over land is not
the operator's to fix (a skin sink variable or a land-surface analysis is
the door's release-2 item), and the band-8 land residual is the analysed
humidity over land against a 52 km column.

## 5. SimSat, the first measurement (GDAS 2026-09-01 18Z, GOES-19 18:00Z full disk)

Inputs, fetched 2026-09-06 00:29 UTC from `noaa-goes19.s3.amazonaws.com`
(the `noaa-goes16` bucket carries no 2026 full-disk scans; GOES-19 has been
GOES-East since April 2025; manifest with URL, bytes, SHA-256, ETag and wall):

| granule | bytes | sha256 (first 16) | fetch |
|---|---:|---|---:|
| OR_ABI-L1b-RadF-M6C13_G19_s20262441800203_e20262441809523_c20262441809550.nc | 26,424,776 | fbc1c0eff97f7d3e | 2.26 s |
| OR_ABI-L1b-RadF-M6C08_G19_s20262441800203_e20262441809511_c20262441809558.nc | 19,626,327 | a80163e639b3fe74 | 1.86 s |
| OR_ABI-L2-ACMF-M6_G19_s20262441800203_e20262441809511_c20262441810223.nc | 26,693,487 | 311f4f28acec7d3e | 4.67 s |
| OR_ABI-L2-CMIPF-M6C13_G19_s20262441800203_e20262441809523_c20262441809575.nc | 24,428,865 | 1d7c04910592ada1 | 3.87 s |

The state: a one-step cold start of the T255 native configuration of record
with the analysis file and the 18Z start (step-0 checkpoint, RTX 5070 Ti, 36 s),
exported to four tiles at 0.25 degrees. **The L1b decode against NOAA's own
inversion**: over 23,046,036 pixels this pack's band-13 brightness
temperature minus the CMIP `CMI` reads bias -0.0005 K, rmse 0.0177 K,
largest 0.031 K (the CMIP quantization). Clear-sky mask: 8,572,294 clear,
14,474,032 cloudy. Colocation: 21,676,836 pairs, 38,168 blocks of 24 pixels.
The analysis carries no condensate (the pgrb2 mapping has none), so the
cloudy classes on this state measure the absence of model cloud; the
cloudy-sky measurement is the forecast arm below.

SimSat, both-clear, zenith 60 or less (FM4 response in band 13, the 6.2 um
centre in band 8): band 13 bias +4.04 K, rmse 5.30, 2.84 after the linear
correction (water +3.14 / 3.37 / 1.16; land +5.64 / 6.97 / 3.49; no zenith
dependence); band 8 +7.17, 7.62, 1.79 after (water and land alike, the
correlation 0.96). Both FAIL the 1.5 K gate. The gap by term as first named
from the renders alone (the window continuum too weak; the land skin and the
gray emissivity 0.99; the band-8 weighting function too low) is confirmed
and quantified by the reference in section 3. Planted read-backs on the nw
tile: identity exactly zero; skin +/-2 K reads +/-1.48 K in band 13; upper
vapor x1.5 reads -3.81 K in band 8; six of six.

The 18 h forecast of the 00Z control (with cloud) scored through the same
door: both-clear band 13 +5.08 K (land +7.50, the model's own afternoon
skin 1.9 K warmer than the analysed; water +2.05), band 8 +6.68; the cloudy
classes are cloud placement (the two disagreement classes hold 5.7 of 17.2
million pairs, +17.3 and -12.5 K) and inside both-cloudy the correlation is
0.53 at a 24 K rmse; a cloudy-sky entry waits on the clear-sky one, which now
exists.

### Assumptions of the SimSat path

See `abi_operator.ASSUMPTIONS`; every abi-score receipt carries them. A 250 m
brick to 19.75 km; GOES-R ellipsoid navigation with the march on the 6370 km
sphere; Planck through the FM4 band-13 response or at the 6.2 um centre; gray
cloud absorption per hydrometeor class at fixed effective radii;
`kappa rho_std(z) q_v` water-vapor absorption (5.0e-3 and 3.0 m^2 kg^-1) on a
standard density; gray surface emissivity 0.99; no scattering, no CO2, no
ozone, no subcolumn cloud fraction, no instrument spatial response.

### Bookkeeping (amendment F)

`rw_goes bt` writes a provenance row into every pack (`BtMeta.provenance`,
read back by `woof.obs.goes_pack`): the granule's `dataset_name` and `id`
(its version identity), `date_created` (the publication instant), the
coverage window, `production_site`, `production_environment`,
`production_data_source`, `platform_ID`, `processing_level`, the first
receipt time given as `--received-utc` (the fetch manifest's wall, threaded
through `abi-score --goes-received BAND=ISO`), and the row-time model
verbatim. On the case granule: created 2026-09-01T18:09:55Z, 3 s after the
scan's end (18:09:52.3); the object store's Last-Modified from the fetch
manifest is 18:10:08 GMT, so the radiance was public 13 s after publication
and 16 s after the scan ended, the fast-cycle class; our receipt
(2026-09-06T00:29:18Z) is a historical fetch and the batch's extras say
"first receipt recorded" with that time, which is not a live latency.
`superobs_from_pack` builds the batch from the pack's own scan window and
returns the provenance and the latency class in its extras.

### Assumptions of the operator path

Clear sky only (the both-clear class by the ACM mask and the model's own
condensate); plane-parallel columns at the block centre; no aerosol; the
stratosphere above 1 hPa not carried (0.03 K in band 8); the reference's
GOES-16 transmittance coefficients under GOES-19 Planck constants (the
sensor term inside the correction); the emissivity by class from the
reference's tables; the scan treated as one instant (the analysis is valid
18:00, the disk scans 18:00:20 to 18:09:52; the stream carries a per-block
time from the scan row, about 26 s uncertainty, for the window bookkeeping
of amendment B).

## 6. What ships, what does not

Ships default-on: `rw_goes bt`, `colocate`, `quicklook`, `forward` (the
`--abi` marker carries `gpuwm-da.abi-forward.v1`, pinned in
`woof/obs/frontdoor.py`); the columns door, the reference scorer, the
trainer and the forward door; the fast-model table for the 40-level
surface_stretched coordinate as trained on the case (its provenance inside);
the operator entries for band 13 and band 8 over water, graded on the
stream's population; the stream module with the stream QC, the entry-driven
class admission and the correction applied by the operator; the four
assessments in the score receipt; the calibrations both ways for the
reference and the operator (the lane's two families and three more: a
whole-column temperature change of +/-1 K reads +0.397 / -0.405 K in band
13 and +/-0.996 K in band 8 through the reference and +0.394 / -0.396 and
+/-0.978 K through the operator; a low-level moistening x1.5 below 700 hPa
reads -2.02 K in band 13 and -0.002 K in band 8 through the reference,
-1.91 and -0.002 K through the operator, the drying +1.17 / +1.11 K; every
column with the right sign). Reported, not admitted: the land class of both
bands, by term. Not shipped: a cloudy-sky entry; SimSat as the operator
(its numbers and the request to its optics stand in section 5).

## 7. Unfinished

* Model-space vertical localisation on the operator's Jacobians (amendment
  H) is the ensemble lane's; release 1 places rows at the weighting
  function's median peak and says so.
* The land class: a skin sink variable or a land-surface analysis in the
  door; until then land blocks are not admitted and the entries say why.
* The table is per coordinate: the jet-refined 48-level layout needs its own
  training run (columns, CRTM, `abi-fast-model train`; minutes).
* The stream's latency class on a live fetch: the pack's provenance row
  records the granule's publication (`date_created`) and this system's first
  receipt, and on this case the receipt is a historical fetch (four days
  later), so the live latency is read from the object store's own
  Last-Modified instead (below); a live `rw_goes fetch` of the L1b prefixes
  (not yet listed by `list`) would measure it directly.
* `rw_goes list` and `fetch` do not yet list the `ABI-L1b-Rad` prefixes; the
  four granules were fetched by URL with a manifest.
* SimSat's window continuum and band-8 coefficient: the numbers in sections
  3 and 5 are the request to its optics; the operator that ships does not
  wait on it.
* The operator's sea-water emissivity is the reference's case mean; the
  0.22 K it costs against the reference in band 13 (0.15 K inside the
  admitted zenith band) is the emissivity's wind and angle dependence, a
  per-column model (the Nalli form) being the remedy; the band 13 window
  form carries no zenith feature, and its bias against the reference runs
  from +0.02 K at nadir to +0.26 K at 60 to 70 degrees.
* The residual tails (1.0 to 1.3 percent beyond three sigma in the admitted
  classes) are reported in the entries; a gross check in the filter reads
  them against the entry's error, and a within-block homogeneity test (the
  block's own brightness-temperature spread, which the block table does not
  yet carry) is the named tightening.
* The stream's land fraction per block comes from the caller (the model's
  own plane sampled at the block centres); the block table itself carries
  no surface class.
