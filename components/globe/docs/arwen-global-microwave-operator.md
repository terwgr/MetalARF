# ATMS microwave brightness temperature as a forward operator for WOOF global

`src/arwen_global/microwave/` (package), `tools/rustwx/crates/rw-atms` (the
HDF5 decoder and the thinning), door `woof global microwave`
(also `woof global microwave ...` and `woof global
microwave ...`), tests `tests/test_arwen_global_microwave.py` and
`tests/test_arwen_global_microwave_entry.py`, evidence charts
`tools/arwen_global_microwave_evidence.py`.

Status, 2026-09-06: **measured; the operator entry ships for channels 4 to
14.** On one day of NOAA-21 ATMS (2026-09-01, 12,206 clear-sky open-ocean
cells against GDAS analysis columns) the temperature-sounding channels 4 to
14 read an out-of-sample O-B rmse of 0.14 to 0.97 K after the entry's bias
correction, inside the 1 K bar; channel 15 (57.29 GHz, the 4.5 MHz
sidebands, about 2 hPa) reads 1.14 K against a 0.54 K noise floor and is
refused with its term named below. CRTM v2.3.0 on 800 of the same columns
agrees with the operator to 0.01 to 0.20 K in channels 7 to 12 and itself
reads 1.12 K in channel 15. The calibration holds in both directions, on
the two families the design named and on three more (a surface-step
column against its closed form, the nadir transmittance squared predicting
the 60 degree reading, superposition under frozen opacity), and the
radiative transfer agrees with an independent 200-sub-layer integration to
0.02 K and with the rw_atms decode of a granule read back through h5py.
The numbers above are the re-read after the refutation pass found the
absorption being handed the total pressure where P.676 takes the dry-air
pressure (the term and its size are in the divergences below). The
entry is registered for the ensemble filter but is not in the door's
default stream set: radiances are localised in model space in release 2,
and until then a cycle assimilates them only when asked.

## What it is

The ATMS instrument on NOAA-20 and NOAA-21 measures 22 microwave channels;
channels 4 to 15 sit on the 50 to 58 GHz oxygen band and each reads the
temperature of one layer from the lower troposphere (channel 4) to about
2 hPa (channel 15). The leg does what the design's item 5b asked, measured
first:

1. **Fetch** one day of Sensor Data Records (the calibrated brightness
   temperatures, one HDF5 granule per 32 s of scan, 12 scans of 96 beams)
   with their geolocation granules from the public NOAA JPSS bucket
   (`noaa-nesdis-n21-pds`, anonymous HTTPS), hash every byte, and record the
   manifest (URL, size, SHA-256 per file, per-hour volume, the ground
   segment's processing latency and the bucket's publication latency read
   from the file names and the object dates).
2. **Decode** every granule pair through `rw_atms`, the Rust bridge and the
   only reader of the files (inventory, flat arrays with scale and offset
   applied, the 1024-byte IDPS user block skipped), and **thin** the beams
   onto latitude rings x longitudes x time bins in Rust: one cell is the mean
   brightness temperature per channel over the beams that fell in it, with
   the beam count, the within-cell standard deviation, the mean viewing
   geometry and the mean time.
3. **Model** the clear-sky brightness temperature of a column: ITU-R
   P.676-13 clear-air absorption (oxygen with first-order line mixing, water
   vapour, the dry continuum; a public Rosenkranz-form model transcribed from
   the published tables), specular ocean emissivity from the Meissner-Wentz
   (2004) dielectric constant with the ATMS quasi-polarization mix, and the
   plane-parallel radiative transfer with the exact Planck function
   (Rayleigh-Jeans selectable, the difference stated in the calibration).
4. **Score** the day's clear-sky over-ocean cells against GDAS analysis
   columns (the 0.25-degree pgrb2 analyses through the Rust mapped engine,
   interpolated in time to each cell's instant), per channel, before and
   after a family of linear bias corrections, with the noise floor.
5. **Promote** the admitted channels into an operator entry with its
   acceptance contract, and hand the filter one `PointObs` batch per channel
   whose `operator` evaluates every ensemble member.

Every data-path step is Rust: the S3 listing and download are orchestration
with hashing, the HDF5 decode and the beam colocation are `rw_atms`, the
GRIB analysis decode is the mapped engine. Python does the column arithmetic
(the radiative transfer is numpy like every other observation operator of
the global model) and the bookkeeping.

## The door

```
woof global microwave fetch --satellite noaa-21 --day 2026-09-01 --out ATMS
woof global microwave decode --fetched ATMS/noaa-21/2026-09-01 --out DECODED
woof global microwave thin --decoded DECODED --out THINNED --latitudes 0.25 --nlon 1440 \
    --origin 2026-09-01T00:00:00Z --bin-s 3600
woof global microwave columns --out COLS gdas.t00z.pgrb2.0p25.f000 gdas.t06z.pgrb2.0p25.f000 ...
woof global microwave calibrate --out microwave-calibration.json
woof global microwave score --thinned THINNED --columns COLS/*.npz --out SCORE [--bar-k 1.0] [--max-cells N]
python tools/arwen_global_microwave_evidence.py SCORE EVIDENCE
```

`score` writes `microwave-scorecard.json` (every stage of the screen, every
channel's statistics and diagnostics, the verdict), `microwave-calibration.json`
(the two-direction receipt, re-run every time), `microwave-operator-entry.json`
(the entry or the refusal) and `microwave-cells.npz` (the per-cell record).
Its exit status is 0 only when every sounding channel is inside the bar, so
on the case day it exits 1 while the entry ships for eleven channels; the
entry file is the thing to read. `--max-cells` subsamples the candidates and
is recorded as a probe, never the reading of record.

## The fetch record (NOAA-21, 2026-09-01)

Bucket `noaa-nesdis-n21-pds`, prefixes `ATMS-SDR/2026/09/01/` and
`ATMS-SDR-GEO/2026/09/01/`; listed 2026-09-06 00:24 UTC in 4.3 s; 2,701
granule pairs (5,402 files), none unpaired, none failed; 711,076,380 bytes in
373.8 s with 8 workers (1.9 MB/s effective through the desktop's link);
SHA-256 per file in the manifest and in `SHA256SUMS`. Per hour the day is
112 or 113 granules and 29.4 to 29.9 MB (16.3 to 16.5 MB of SDR, 13.0 to
13.4 MB of geolocation). Latency behind real time, read from the granule
end time against the file's creation stamp (processing) and the bucket
object's date (publication): the publication latency over the 5,402 files
has a minimum of 188 s, a median of 1,391 s, a 90th percentile of 2,439 s
and a maximum of 3,568 s. **Latency class:** with a median of 23 minutes
and a tail to 59 minutes, ATMS is a fast-cycle stream for an hourly
analysis whose information cutoff sits at least an hour behind the
observation hour, and a delayed-replay stream for anything tighter; the
manifest carries every granule's own numbers so the cutoff decides per
granule, not per stream.

## The calibration (both directions, before any real number)

`microwave-calibration.json`, `calibrate.run()`, held by the tests:

* An isothermal column over a black surface at 250 K reads 250 K in every
  channel to 1.7e-13 K; over a grey surface (emissivity 0.6) at 280 K it
  reads the analytic `B(TB) = B(T) - (1 - e) t_s^2 (B(T) - B(T_cmb))` to
  1.1e-13 K.
* The Rayleigh-Jeans weights of every channel sum with the surface
  transmittance to one within 4.4e-16.
* A +0.5 K layer planted at the level pair where a channel's weighting
  function peaks reads back as the pair's weight times 0.5 K, to 1.4e-7 K
  under frozen opacity (8.4e-4 K for channel 4, whose pair is the lowest
  and whose straddling sub-layer blends with the unmoved 2 m temperature)
  and within 0.0091 K with the absorption's own temperature dependence;
  channels whose weight at that pair is below 1e-3 move by less than
  0.0007 K; and the planted channel is the loudest sounding channel for
  every channel but 5, which peaks in the surface layer of the tropical
  column where channel 4 carries more weight than channel 5 does anywhere
  (rank 2, recorded). The peak is the layer of largest weight per unit ln p:
  the first receipt took the largest weight per sub-layer, which the level
  spacing quantises (channels 8 and 9 both read 143 hPa; per unit ln p they
  peak at 243 and 155 hPa), planted some channels a pair away from their
  peak and read ranks up to 4 there. The receipt carries the density peak,
  the per-sub-layer maximum and the centroid side by side.
* Three more families, held by the refutation pass and recorded in its
  report: an isothermal atmosphere at 250 K over a black surface at 290 K
  reads its closed form `B(TB) = B(T_a)(1 - t_s) + B(T_s) t_s` with the
  transmittance from an independent 20,000-layer integration to 1.8e-3 K
  at 0, 30 and 60 degrees (the residual is the four-fold layering); the
  nadir transmittance squared predicts the 60 degree reading to 3e-13 K
  (plane-parallel sec z, no dependence on the absorption); and under frozen
  opacity in the Rayleigh-Jeans limit the operator is linear in the column
  to 6e-14 K (a half-and-half mixture of two columns reads the mean of
  their readings).
* The absorption is handed the dry-air pressure `p - e` and the vapour
  pressure `e` of each sub-layer, as P.676 defines its arguments (the
  pyrtlib comparison above was made that way from the first). The first
  record run handed it the total pressure: on the record day's columns
  that read 0.75, 0.69 and 0.35 K warm in channels 3, 4 and 5 and 0.42 K in
  channel 16 (1.3, 1.1 and 0.6 K at the wettest columns), under 0.06 K in
  channels 6 to 15, an error that grew with the path length and was
  absorbed by the scan term of the geometry correction (channel 4's
  `c` fell from +2.00 to +0.47 K with the fix). Against pyrtlib's R22
  models on twelve of the columns over a black surface the operator reads
  within 0.1 K in the windows, 0.3 to 0.6 K cold in channels 4 to 7 and
  0.1 to 0.7 K warm in channels 9 to 14 (the two line models), and 1.2 K
  apart in channel 15.
* A column against itself moves nothing (bitwise zero).
* The Planck-exact minus Rayleigh-Jeans term per channel on the standard
  column is stated in the receipt (0.013 to 0.036 K in channels 1 to 4,
  0.088 K in channel 16, under 1e-3 K in channels 6 to 15).
* The P.676-13 absorption agrees with pyrtlib 1.2.0 (R98 and R22 oxygen,
  R98 and R22SD water vapour, N2 R18) within 5 percent from the surface to
  10 hPa on the sounding frequencies; the 2 hPa row (channels 14 and 15)
  diverges by 12 to 26 percent where the pressure width meets the P.676
  width floor and the Zeeman splitting neither model carries (the recorded
  comparison of 2026-09-06).
* The surface slab: over the ocean the surface sits 10 to 20 hPa below the
  1000 hPa analysis level; the column is extended to a fixed 1100 hPa bottom
  along the lowest layer's ln p slope and each column's surface trims it.
  Without the slab the operator read 1.39 K cold in channel 4 and 0.70 K in
  channel 5 at 1010 hPa (2.74 and 1.37 K at 1020 hPa), nothing above
  channel 7. The 2 m temperature blend of the straddling sub-layer is worth
  under 0.005 K in every channel.

## The reading of record

12,206 clear-sky open-ocean cells (from 2,127,759 cells on the day; 132,714
geometry candidates inside 60 degrees latitude with three beams and zenith
under 60 degrees; 99,076 over open ocean; 53,610 with analysis cloud water
at or under 0.01 kg/m2; 12,265 with a retrieved liquid water path at or
under 0.02 kg/m2 and a positive 23.8 minus 31.4 GHz difference; 12,206
inside the analysis span with finite radiances), 3.2 beams per cell on
average. Every corrected rmse is out of sample: the correction is fitted on
the even half of the cells in time order and scored on the odd half (6,103
cells per channel; 5,994 for channel 14 and 4,536 for channel 15 after the
per-channel spread limit). The correction of record is the geometry model
`O - B = a + b (B - mean_B) + c (sec z - 1)`, the one the member operator can
evaluate; the wind model adds `d W10` with the analysis 10 m wind as a
diagnostic. The noise floor is the rms over the scored cells of the
within-cell spread over the square root of the beam count.

| ch | GHz | raw bias | raw rmse | linear in B | geometry (record) | + wind | noise floor | d, K per m/s | verdict |
|---:|----:|---------:|---------:|------------:|------------------:|-------:|------------:|-------------:|---------|
| 1 | 23.8 | +3.51 | 3.98 | 1.83 | 1.83 | 1.64 | 0.28 | +0.42 | window |
| 2 | 31.4 | +3.72 | 3.93 | 1.21 | 1.21 | 0.99 | 0.22 | +0.36 | window |
| 3 | 50.3 | +4.13 | 4.36 | 1.37 | 1.36 | 0.98 | 0.22 | +0.49 | window control |
| 4 | 51.76 | +2.05 | 2.28 | 0.97 | **0.97** | 0.63 | 0.15 | +0.39 | admitted (0.03 K inside) |
| 5 | 52.8 | +1.08 | 1.19 | 0.48 | **0.47** | 0.29 | 0.10 | +0.19 | admitted |
| 6 | 53.6 | +0.31 | 0.37 | 0.19 | **0.18** | 0.16 | 0.10 | +0.05 | admitted |
| 7 | 54.4 | +0.18 | 0.23 | 0.14 | **0.14** | 0.14 | 0.10 | 0.00 | admitted |
| 8 | 54.94 | -0.48 | 0.51 | 0.17 | **0.17** | 0.17 | 0.10 | +0.01 | admitted |
| 9 | 55.5 | -0.56 | 0.61 | 0.23 | **0.23** | 0.23 | 0.10 | +0.01 | admitted |
| 10 | 57.29 | -0.52 | 0.61 | 0.32 | **0.32** | 0.32 | 0.14 | +0.01 | admitted |
| 11 | 57.29 | -0.27 | 0.40 | 0.30 | **0.30** | 0.30 | 0.19 | 0.00 | admitted |
| 12 | 57.29 | +0.00 | 0.33 | 0.33 | **0.33** | 0.33 | 0.21 | 0.00 | admitted |
| 13 | 57.29 | -0.75 | 0.88 | 0.46 | **0.46** | 0.46 | 0.30 | +0.01 | admitted |
| 14 | 57.29 | -0.87 | 1.11 | 0.67 | **0.67** | 0.67 | 0.42 | +0.01 | admitted |
| 15 | 57.29 | +0.80 | 1.49 | 1.14 | **1.14** | 1.13 | 0.54 | -0.04 | refused: upper term |
| 16 | 88.2 | +4.61 | 5.11 | 2.17 | 2.17 | 2.11 | 0.39 | +0.24 | window |
| 17 | 165.5 | +0.99 | 1.67 | 1.32 | 1.32 | 1.29 | 0.27 | +0.13 | window |
| 18 to 22 | 183.31 | +0.25 to +0.46 | 0.89 to 1.55 | 0.85 to 1.50 | 0.85 to 1.50 | same | 0.18 to 0.29 | under 0.02 | water vapour, not graded |

Before the dry-pressure fix the same cells read raw biases of +3.59, +3.44,
+3.39, +1.37, +0.73 and +4.19 K in channels 1 to 5 and 16, channel 4 read
0.98 K after the geometry correction and channel 5 0.49 K; channels 8 to 15
are the same to 0.001 K in every column of the table.

Wall: 255 s for the day on the CPU host alone (232 s of it the operator on
12,206 columns, 53 columns per second single-threaded; 329 s when the
record was re-read beside the test suites), 0.3 s of column sampling after
the two-stage screen. The record reproduces exactly in a fresh sync of the
tree (every per-channel number to 0.0 K), and the per-channel statistics
recompute from the stored cells with the shared scorer to 2e-7 K.

Channel 4 sits at the bar within its sampling uncertainty and the design's
rule is the point reading: a block bootstrap over 60 contiguous blocks of
the scored half (the correction held at the fitted coefficients) puts the
95 percent interval at 0.86 to 1.10 K with 29 percent of resamples above
1 K; fitted on the odd half and scored on the even it reads 0.96 K; the
tropical band 0 to 30 N reads 1.14 K and the 00 to 06Z window 1.08 K, the
other bands and windows 0.79 to 0.98 K. Its entry error (1.09 K) covers
the interval. Channel 15 is above the bar in every resample (1.08 to
1.19 K) and every band (1.04 to 1.42 K).

### CRTM as the numerical reference

CRTM v2.3.0 (JCSDA, built on the CPU host from the public `crtm_v2.3.0`
source with gfortran 15; the `atms_n20` ODPS transmittance and SpcCoeff
coefficients and FASTEM-6 ocean emissivity from `fix_REL-2.3.0_emc`, the
NOAA-21 coefficients not being in that set and the channel plan being the
same) was run on 800 of the scored columns drawn at random, the same
levels, surface pressure, skin, analysis wind and viewing geometry, through
a 90-line Fortran driver (`CRTM_Forward`, clear sky, sea water, salinity
35). Tree operator minus CRTM, mean and standard deviation over the 800
columns, with the slope of that difference against the analysis wind:

| ch | tree minus CRTM | slope, K per m/s | O-B tree (bias/std) | O-B CRTM (bias/std) |
|---:|----------------:|-----------------:|--------------------:|--------------------:|
| 1 | -2.12 +- 1.07 | -0.49 | +3.59 / 1.88 | +1.47 / 1.65 |
| 2 | -3.28 +- 1.21 | -0.41 | +3.79 / 1.28 | +0.51 / 1.01 |
| 3 | -3.05 +- 1.37 | -0.58 | +4.18 / 1.43 | +1.14 / 1.03 |
| 4 | -2.19 +- 0.92 | -0.43 | +2.08 / 1.00 | -0.10 / 0.68 |
| 5 | -0.90 +- 0.38 | -0.18 | +1.09 / 0.50 | +0.20 / 0.30 |
| 6 | -0.17 +- 0.07 | -0.03 | +0.32 / 0.19 | +0.15 / 0.17 |
| 7 | +0.03 +- 0.04 | 0.00 | +0.17 / 0.14 | +0.21 / 0.16 |
| 8 | +0.01 +- 0.04 | 0.00 | -0.48 / 0.16 | -0.47 / 0.17 |
| 9 | -0.10 +- 0.03 | 0.00 | -0.56 / 0.22 | -0.66 / 0.23 |
| 10 | -0.20 +- 0.08 | 0.00 | -0.51 / 0.32 | -0.70 / 0.33 |
| 11 | +0.00 +- 0.02 | 0.00 | -0.26 / 0.28 | -0.26 / 0.29 |
| 12 | +0.11 +- 0.02 | 0.00 | +0.01 / 0.35 | +0.12 / 0.35 |
| 13 | +0.32 +- 0.04 | 0.00 | -0.75 / 0.44 | -0.43 / 0.43 |
| 14 | +0.50 +- 0.16 | 0.00 | -0.86 / 0.71 | -0.36 / 0.66 |
| 15 | -1.18 +- 0.49 | +0.01 | +0.79 / 1.25 | -0.38 / 1.12 |
| 16 | -3.64 +- 1.24 | -0.37 | +4.71 / 2.20 | +1.07 / 2.13 |

(The tree column is the re-read after the dry-pressure fix on the exported
columns, which carry no 2 m temperature; that blend is under 0.012 K. The
CRTM column is the driver's output, reproduced byte for byte by a second
run.) Reading: in channels 7 to 12 the two models agree to 0.01 to 0.20 K
in the mean with 0.02 to 0.08 K of scatter, so the absorption, the layering
and the radiative transfer of this operator are the reference's to within a
tenth of a kelvin where the surface does not enter. In channels 1 to 5 and
16 the whole difference is the emissivity: it is proportional to the wind
(-0.18 to -0.58 K per m/s) because FASTEM-6 carries the wind-roughened,
foam-covered sea and the specular model does not (the dry-pressure fix
cooled this operator's surface-seeing channels by 0.4 to 0.7 K and so
widened the gap to FASTEM-6 by that much; the wind slope did not move);
with FASTEM-6 the channel 4 O-B scatter is 0.68 K against this operator's
1.00 on the same columns and channel 5's 0.30 against 0.50. In channels 13 to 15 the
models part by +0.3, +0.5 and -1.2 K where the P.676 line widths meet
their floor and the Zeeman splitting differs between them; CRTM's own O-B
scatter in channel 15 is 1.12 K against this operator's 1.25 on the same
800 columns, so the bar there is beyond the reference model too at this
cell size and the term is shared between the 2 hPa analysis and the beam
noise. The driver, its input columns, its output and the coefficient
digests are recorded with the measurement; CRTM itself is not vendored.

### The terms, by channel

* **Channels 1, 2, 3, 16 (windows), and 4 and 5 through their surface
  transmittance (0.45 and 0.25 at 30 degrees).** The raw bias of +3.5 to
  +4.6 K in the windows is the specular ocean emissivity: no wind
  roughness, no foam, both of which raise the emissivity by a few
  hundredths at these frequencies, so the modelled surface is too cold
  under wind. The wind diagnostic reads the term: +0.42, +0.36, +0.49 and
  +0.24 K per m/s of analysis 10 m wind for channels 1, 2, 3 and 16, +0.39
  for channel 4 and +0.19 for channel 5; CRTM's FASTEM-6 carries the same
  slopes with the opposite sign against this operator. Channel 4 is
  admitted on its geometry reading (0.97 K, 0.03 K inside the bar) with the
  roughness term named as its dominant residual (0.63 K with the wind
  predictor, 0.68 K under FASTEM-6); its assigned error, 1.09 K, covers the
  wind-dependent part. The remedy is a wind-roughened emissivity model with
  published coefficients (FASTEM or the Meissner-Wentz 2012 wind-induced
  emissivity), not a wind predictor in the bias correction, because a
  predictor the operator cannot evaluate at a member is not part of H(x).
* **Channels 6 to 9.** Small biases (+0.3 to -0.6 K) that the constant term
  removes; the residual of 0.14 to 0.23 K is 1.4 to 2.3 times the noise
  floor and agrees with CRTM to 0.1 K.
* **Channels 10 to 14.** Biases of -0.3 to -0.9 K and residuals of 0.30 to
  0.67 K against noise floors of 0.14 to 0.42 K: the residual above noise is
  0.23 to 0.52 K and grows with height, the stratospheric analysis and the
  absorption above 10 hPa; CRTM reads the same scatter (0.33 to 0.66 K).
* **Channel 15.** 1.14 K after the correction against a noise floor of
  0.54 K (about 1.0 K per beam, 3.2 beams per cell): 1.0 K remains above the
  noise. The largest background slope of any channel (b = +0.28) says the
  operator or the analysis scales the 2 hPa layer wrongly. Two terms are
  named and neither is modelled: the Zeeman splitting of the 57 GHz oxygen
  lines by the geomagnetic field, which at 2 hPa is comparable to the
  pressure broadening and moves the 4.5 MHz sidebands by tenths of a kelvin
  depending on the field's strength and orientation along the path, and the
  GDAS analysis above 5 hPa, which the pgrb2 product carries on levels to
  0.01 hPa but which no conventional observation constrains. CRTM reads
  1.12 K on the same columns. Channel 15 is 0.14 K outside the bar; a
  larger cell (more beams, a lower noise floor) is the first lever and a
  Zeeman term the second.
* **Channels 18 to 22** are water-vapour channels and are not graded by this
  leg (they are not temperature-sounding channels); their residuals of 0.85
  to 1.50 K after correction are recorded for the humidity leg that follows.

## The operator entry

`microwave-operator-entry.json`, schema
`gpuwm-arwen-global-microwave-operator-entry-v1`, name
`atms-clear-ocean:noaa-21:2026-09-01`, stream `atms-clear-ocean`, variable
`brightness_temperature_k`, admitted channels 4 to 14. Per channel: the
error `sqrt(NEdT^2 + rmse_after^2)` (1.09 K for channel 4, 0.52 to 0.82 K
for channels 5 to 10, 1.24 to 2.49 K for 11 to 14 where the specified NEdT
dominates), the geometry coefficients with the mean background they are
centred on, the weighting-function centroid pressure on the standard column
(556 hPa for channel 4 down to 3.9 hPa for channel 14) and the ln p distance
about it that holds 90 percent of the weight (1.2 to 1.8), the corrected
rmse and the noise floor it was read against, and the cell count. The
member operator holds the member's top-level temperature above the model
top (1 hPa on the default stack): 1.8 percent of channel 13's weight and
8.3 percent of channel 14's lie above it on the standard column at nadir
(29.6 percent of channel 15's), and the contract says so. The entry
carries the calibration verdicts and the digests of the analyses the
correction was fitted against.

### Acceptance contract

* **Measurement.** ATMS SDR antenna brightness temperature (the IDPS
  product; the remapped-to-scene product is not used), one thinned cell: the
  mean over the beams that fell in one 0.25-degree cell in one time bin,
  clear sky over open ocean by the Grody (2001) 23.8/31.4 GHz liquid-water
  retrieval and the analysis cloud water path, |latitude| <= 60, zenith <=
  60 degrees, at least three beams.
* **Time.** The mean beam time of the cell (UTC); granule start, end and
  creation instants in the fetch manifest. A row at 12:08 is compared with
  the member trajectory at 12:08 when the filter keeps observation-space
  trajectories through the window.
* **Location.** The mean beam latitude and longitude; the beam footprint is
  2.2 degrees (about 32 km at nadir) for channels 3 to 16.
* **Vertical coordinate.** ln p of the channel's weighting-function
  centroid on the standard column, with `vertical_cutoff_lnp` the extent
  holding 90 percent of the weight. This is the interim point placement of
  a radiance in a point filter; the release-2 form localises in model space
  and the entry says it is not that.
* **Representativeness.** The within-cell standard deviation per channel
  rides with every cell; cells above the per-channel spread limit (3 K for
  the windows, 1.5 K for the sounding channels) are not scored and not
  offered.
* **Bias treatment.** Per channel `a + b (B - mean_B) + c (sec z - 1)`,
  fitted on the even half of the day's cells in time order, scored on the
  odd half, applied to H(x) in the operator and never to the report; the
  coefficients carry the day they were fitted on. A channel that passes
  only with the analysis wind as a predictor is surface-limited and not
  admitted.
* **Error correlations.** Channels are offered as separate batches with
  diagonal errors; inter-channel residual correlation is not modelled and
  is what the Desroziers diagnostic will measure first. Observation errors
  are not retuned until those diagnostics look right.
* **Assimilation status.** Release 1: measured and registered; not in the
  door's default stream set until radiances are localised in model space.

### Interface decisions (for the ensemble, door and observation lanes)

* A filled batch is a `woof.globe.da.observations.PointObs`
  (stream `atms-clear-ocean`, variable `brightness_temperature_k`, one batch
  per admitted channel, `surface` false on every row, `ln_pressure` the
  channel's centroid, `vertical_cutoff_lnp` the channel's extent, `error`
  the entry's per-channel error, `valid_time` the cell instants). Row
  identity is `atms:<satellite>:<bin>:<ring>:<lon index>:chNN`, so the
  assimilation chain refuses a cell a posterior already used and any subset
  of a batch still evaluates.
* `batch.operator(members, batch) -> (R, n)` is
  `entry.AtmsBatchOperator`: for each member it synthesizes theta, vapour
  and ln ps at the rows through `sample_scalar`, converts to temperature on
  the member's hybrid full levels, interpolates linearly in ln p onto the 41
  GDAS isobaric levels the O-B calibration was measured on (held at the
  member's top and surface), takes the skin from the member's surface plane,
  runs the same radiative transfer at each row's zenith and scan angle, and
  adds the entry's bias for that channel with the member's own brightness
  temperature as the predictor. No 2 m temperature enters (measured worth
  under 0.005 K in every channel on the standard column).
* The layer set of the radiative transfer is a function of the level set
  alone (top extension to 1 Pa, bottom slab to 1100 hPa), so a column reads
  the same kelvin in any batch to 1e-9 K (numpy's reduction order over a
  different trailing dimension moves the last bit); the filter's O-A on a
  subset therefore matches the O-B pass.
* The entry is built by `entry.entry_from_scorecard(document, calibration)`
  from the scorecard's admitted channels and written beside it; the door
  reads it back with `entry.read_entry` and builds batches with
  `entry.point_obs_from_cells(entry, thinned_cells, transform=...,
  vertical=...)` after the scorecard's `candidate_mask` and `screen_cells`
  have selected the clear-sky ocean cells for the window. Registering the
  stream in the door's `STREAM_TABLE` is a factory that runs `fetch`,
  `decode` and `thin` for the window and returns those batches; it is not
  in the default set (see the status).
* Held by tests on a three-member CPU ensemble (T3, four levels, the moist
  smoke configuration): identical members read bitwise the same; a member
  whose theta is scaled by 1 percent reads channel 7 warmer by 0.7 to 1.3
  percent of its brightness temperature; a planted bias model is applied to
  the kelvin; a batch built from cells fills `(R, n)` finite values and any
  subset of it evaluates to the same rows within 1e-9 K.

## What ships, what does not

Ships default-on: the fetcher with its manifest, `rw_atms` (inventory,
decode, thin), the channel plan, the absorption, the emissivity, the
radiative transfer with the surface slab, the GDAS column mapping
(`woof/authorities/rw-wps-gdas-pgrb2-0p25-microwave-columns.mapping.json`),
the two-stage scorecard with its predictor family and noise floor, the
calibration receipt, the operator entry for channels 4 to 14 with its
contract and member operator, the `microwave` door on the module CLI and on
`woof global`, the evidence tool.

Not shipped: an admitted entry for channel 15 (upper term, 0.14 K outside;
CRTM reads the same); channels 1 to 3 and 16 to 22 as filter observations
(the windows carry the emissivity term, the water-vapour channels wait on
the humidity leg); the stream in the door's default set (release 2,
model-space vertical localisation); a cloudy-sky or over-land operator;
NOAA-20 (the same code path, a second day of measurement when a second
satellite is wanted); a wind-roughened emissivity (the named lever for
channels 4 and 5, 0.3 K of scatter each by the CRTM reading); a Zeeman
term; CRTM inside the tree (built beside it as the reference, not
vendored).

## Divergences and assumptions, stated

* P.676-13 carries Tretyakov (2005) oxygen line parameters and the
  Rosenkranz (1998) water-vapour continuum through a 1780 GHz pseudo-line;
  it does not carry speed-dependent line shapes nor Zeeman splitting beyond
  a width floor. Both are named terms, measured above against pyrtlib and
  CRTM. Its pressure argument is the dry-air pressure and each sub-layer
  hands it `p - e`; the first record run handed it the total pressure and
  read up to 1.1 K warm in channel 4 (the calibration section has the term
  by channel), corrected before the numbers in this note were read.
* Specular Fresnel emissivity from the Meissner-Wentz 2004 dielectric at
  35 psu; no roughness, no foam, no salinity field. The term is measured
  above (0.2 to 0.49 K per m/s of wind on the surface-seeing channels, and
  the same against FASTEM-6).
* Plane-parallel geometry with sec(zenith); the spherical correction (0.7
  degrees at 60 degrees zenith and 50 km) is not applied.
* GDAS columns on 41 isobaric levels refined four times in ln p with an
  isothermal extension to 1 Pa and the surface slab to 1100 hPa (sixteen
  sub-layers) trimmed at each column's surface pressure; the straddling
  sub-layer blends its temperature with the 2 m temperature; time
  interpolation linear between six-hourly analyses at each cell's instant.
* The GDAS analysis total cloud cover record carries no data points in the
  pgrb2 f000 file, so the analysis side of the clear-sky screen is the cloud
  water path alone; the observation side is the Grody retrieval.
* Bias coefficients fitted on one day and one satellite; the entry carries
  that day and expects a re-fit when the satellite, the analysis source or
  the season changes.
