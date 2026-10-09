# WOOF global: effective resolution, the instrument of record

Rebuilt 2026-09-01 on spherical-harmonic total-degree spectra, after an
independent audit (`ARWEN-GLOBAL-CORE-AUDIT-2026-09-01`) reproduced six
defects in the 10-row FFT-band instrument that produced every number in the
earlier editions of this document. Findings SP-1 to SP-6 are stated in
plain language below with the magnitudes the audit measured. Every reading
made with that instrument is kept in "Superseded readings" and is marked
**superseded (band-FFT instrument, 2026-08-30..09-01), to be re-measured**.
None of those numbers has been converted to this instrument's conventions
and none has been replaced by an estimate.

## What the instrument measures

`woof.verify.harness.dynamics_spectrum` (door:
`python -m woof.verify.harness run dynamics.spectrum --config <run.toml>
--checkpoints <step*.npz...> [--minimum-hours 12]`).

**The spectrum.** Kinetic energy per unit mass by total degree `n`, from
the checkpoint's own vorticity and divergence coefficients:

    KE_n = a^2 / (2 n (n+1)) * (1/4 pi) * sum_m w_m (|zeta_nm|^2 + |D_nm|^2),
    w_0 = 1, w_m = 2

under the transform's complex-orthonormal convention, rotational and
divergent parts kept separately, so that `sum_n KE_n` equals the
sphere-mean of `0.5 (u^2 + v^2)`. Parseval is proven numerically:
relative error 7.7e-15 at T127 (5.4e-15 in the test suite at T31),
wind -> vorticity/divergence -> wind round trip 1.1e-11 relative.

**One wavelength convention**, stated once in the code and here:

    isotropic: lambda(n) = 2 pi a / sqrt(n (n+1)),  k_n = sqrt(n (n+1)) / a.

T63 -> 630.4 km, T127 -> 314.0 km, T533 -> 75.0 km, T719 -> 55.6 km. The
"dx at 45 degrees" and "truncation wavelength at 45 degrees" conventions of
the retired instrument are gone; nothing here is quoted in multiples of dx.
What the retired flagship claims read under the isotropic convention is in
"Truncation scale under the isotropic convention" below.

**Regions.** `global` is the coefficients' own spectrum. `northern` and
`southern` remove degrees below the synoptic window (minus one), multiply
the synthesised wind by a smooth window (1 poleward of 30 degrees, cosine
squared taper to 0 at the equator, 0 in the other hemisphere), re-analyse
it and divide by the sphere-mean of the squared window, so the spectrum
sums to the window-weighted mean KE of the hemisphere's high-passed wind.
Measured leakage kernel of a single degree through the window (T127): 42 %
stays at the degree, 9.7 % lies outside +-1, 4.8 % outside +-2, 1.0 %
outside +-5, 0.05 % outside +-10. The +-1 mixing is inherent to any
hemispheric window (it is the parity flip of a half-sphere function); it
smooths the spectrum over two degrees and does not move a departure
reading measurably (see the calibration). Without the high-pass,
planetary-degree energy leaking through the window inflated the synoptic
degrees of an isotropic k^-3 field by 1.4-2.3x and steepened the
hemispheric fit by 0.6; with it, hemispheric and global fits agree to
0.005 +- 0.26 in slope over 16 seeds.

**Two readings, each labelled by what it compares against.**

- `self_relative`: departure from the power law fitted (weighted least
  squares, weights = mode count 2n+1) to the subject's own synoptic degrees,
  isotropic wavelength 1200-3000 km (n = 13-32 on any truncation). This is
  the dissipation-range reading, where the model stops following the trend
  it set itself. It never consults an observation and is never called
  "observed". Its window is 1200-3000 km because the retired 2000-6000 km
  window straddles the spectral peak of a real atmosphere (GFS f024 250 hPa
  NH, 2026-08-31 00Z: E_7 = 8.4, E_13 = 8.2, E_20 = 1.8 m^2 s^-2) and fitted
  a -2.3 slope there whose extrapolation read 104 / 1128 / 1456 km on the
  three regions of one product; the 1200 km lower edge (rather than 1000)
  halves the bias of a 900 km cutoff (0.88 -> 0.92x) at the cost of
  single-shot noise (7.6 -> 11.5 % of truth at 400 km).
- `absolute`: departure from an observed reference in physical units, the
  Lindborg (1999, J. Fluid Mech. 388) fit to the Nastrom-Gage (1985) GASP
  aircraft spectra,

      E(k) = d1 k^(-5/3) + d2 k^(-3),  d1 = 9.1e-4 m^(4/3) s^-2,
      d2 = 3.0e-10 m^2 s^-2,  k in rad/m,

  read as the isotropic two-dimensional kinetic-energy spectrum (the
  usage of Skamarock et al. 2014, JAS 71, against spherical-harmonic
  spectra) and converted to a total-degree spectrum by

      E_n = E(k_n) * dk/dn,  dk/dn = (2n+1) / (2 a sqrt(n(n+1))),

  the energy the continuous spectrum holds between the shells of adjacent
  degrees (the sum over n = 13..255 equals the integral of E(k) between the
  half-degree shells to 0.5 %). The two terms are equal at 455 km. The
  aircraft data are upper-tropospheric, so this reading is anchored ONLY
  on levels whose mean pressure lies in 200-300 hPa; elsewhere it is
  reported `not_anchored` with no wavelength. Beside the criterion reading
  the instrument reports the retained fraction `KE_n / E_n` at 2000, 1000,
  500 and 250 km.

  This is the first reading in this document's history that compares the
  subject against an observed AMPLITUDE. Every number the retired
  instrument published was self-anchored (SP-1).

  Consistency of the two-dimensional reading of the fit: a non-divergent
  synthetic field with `KE_n = E_n` has along-latitude ring spectra of u
  (longitudinal) and v (transverse) that match the isotropic
  two-dimensional relations `E11(k1) = 4/pi int E(k) k2^2/k^3 dk2`,
  `E22(k1) = 4/pi int E(k) k1^2/k^3 dk2` (integrated over the degrees the
  field holds) to within the +-15 % Legendre interference of a single
  latitude, and averaged over |lat| < 60 degrees (T127, 12 seeds) to
  measured/analytic E11 = 1.10 / 1.10 / 0.93 / 0.94 / 0.94 / 1.01 and E22 =
  1.24 / 1.03 / 0.93 / 0.92 / 0.86 / 1.03 at m = 8 / 13 / 20 / 30 / 40 / 60;
  the transverse/longitudinal ratio sits between the k^-3 value 3 and the
  k^-5/3 value 5/3 as it must.
  The adversarial calibration of 2026-09-01 re-derived the conversion
  independently (a plane check of the 1-D relations at p = 3 and p = 5/3,
  the equator great-circle spectra of a Lindborg-shaped field, and the
  retained fraction of a field whose along-track spectrum IS Lindborg) and
  passed it (D1-D4). An earlier edition of this paragraph also claimed
  that a one-dimensional reading of the same constants "would have put a
  global analysis at 15-30 % of the observed atmosphere"; the verifier
  could not reproduce it, it followed from nothing measured, and it is
  deleted rather than re-derived: no product has yet been read on its
  native grid, so there is no measured retained fraction to state.

**The criterion**, identical for both readings: the largest wavelength at
which the spectrum (log-smoothed over three adjacent degrees, edges
averaged over the two degrees that exist, no zero padding) falls below
50 % of the reference AND stays below it at every finer degree examined,
AND holds for at least 3 consecutive degrees (the smoother's footprint),
AND reaches 20 % of the reference somewhere inside that range (the
confirmation depth). The self-relative reading examines degrees finer
than its fit window; the absolute reading examines from the first
synoptic degree. When no such departure exists inside the examined range
the reading is `status = "unresolved"` and carries the smallest wavelength
examined and the ratio there; a departure with too little support or too
little depth is `unresolved` with the count or the depth in its flag; when
the spectrum is already below the reference at the first examined degree
it is `status = "below_reference_throughout"`. Two more refusals belong to
the self-relative reading only: a synoptic window whose weighted RMS
log-residual from the fitted line exceeds 0.35 is not a power law and the
reading is `status = "unfitted"` with the residual; a hemispheric fit
whose slope lies more than 0.6 from the global fit of the same subject is
`unresolved` with the flag "hemispheric fit out of family". A gridded
product analysed from a regular lat-lon grid is examined only outside
its trust band (the last 4 grid lengths, below); a departure that exists
only inside the band is `status = "inside_product_trust_band"` with the
wavelength withheld. No status but `resolved` carries a wavelength:
`wavelength_km` is `None` otherwise. Every departure, resolved or
refused, carries `departure_log_slope`, the slope of `log(E_n /
reference)` against `log k_n` over its supported degrees: the rolloff
sharpness a reader compares against the shape table below. The retired
instrument returned the grid-scale bin when nothing resolved and read it
on 20/20 no-cutoff controls. This one, before the support and
confirmation gates, read 0/60 GLOBAL no-cutoff controls as resolved but
17/120 HEMISPHERIC ones (F1 of the adversarial calibration, below);
with the gates it reads 0/120 and 0/600.

**Gridded products** (GFS, GDAS, IFS, AIFS regular lat-lon files; door
`tools/external_ke_stick.py`). Each meridian is continued around its great
circle through the antipodal column (wind components change sign across
the pole), sampled uniformly in colatitude at `2 (nlat-1)` points and
trigonometrically interpolated onto the Gaussian latitudes (exact for
fields band-limited below the row count), and the wind is analysed at the
matched truncation `T = nlon/2 - 1` on the linear Gaussian grid whose
columns are the product's own (0.25 deg -> T719 on 720 x 1440; 0.5 deg ->
T359; 1 deg -> T179), so the truncation is the only filter. No bilinear
regrid to a coarse grid happens anywhere; products without pole rows or
with uneven spacing are refused with the reason. Measured recovery of a
band-limited synthetic wind through the route: max relative error of KE_n
2.4e-6 (median 5.4e-8) at 1 deg / T179, 1.1e-6 at 2 deg / T89 (the
residual is the 1e-4 deg pole limit of the synthetic, not the route). A
0.25 deg product analyses at T719 in 119 s on the CPU. The reading is of
the PRODUCT on its grid.

**The product trust band.** A public product is interpolated by its
post-processor from a finer native grid with no low-pass (NCEP's 0.25
deg pgrb2 comes from the 13 km Gaussian grid, a 2.13:1 ratio), so the
sub-grid variance folds into the product's last grid lengths and the
interpolation kernel attenuates them. The instrument therefore examines a
regular product only at wavelengths of at least 4 grid lengths (grid
length = `a * dlat`, the product's uniform meridional spacing: 27.8 km
and a 111 km band at 0.25 deg; `PRODUCT_TRUST_GRID_LENGTHS`), and a
crossing inside the band is `inside_product_trust_band` with its
position in grid lengths in the flag and no wavelength; the retained
fraction inside the band is withheld and a product whose band reaches
the fit window is refused. Measured (`tools/dynamics_spectrum_calibration.py
--sections trust_band`; `CALIBRATION["product_trust_band"]`): a wind with
the Lindborg spectrum filled to T191 on the Gaussian rows of the T191
linear grid, bilinearly interpolated onto a 2 deg product (T89, 2.1:1)
and a 2.5 deg one (T71, 2.6:1); the same field band-limited to T179 on a
1 deg regular grid decimated to 2 deg (coincident nodes, pure foldover);
and the field band-limited to T89 with no interpolation as the route
control. Recovered/planted `KE_n` at the degree nearest each multiple of
the coarse grid length:

| source | 2 | 2.5 | 3 | 3.5 | 4 | 5 | 6 | 8 | 10 grid lengths |
|---|---|---|---|---|---|---|---|---|---|
| bilinear from the Gaussian grid, 2 deg (2.1:1), Lindborg | 1.187 | 0.928 | 0.938 | 0.940 | 0.941 | 0.960 | 0.971 | 0.982 | 0.985 |
| bilinear, 2.5 deg (2.6:1), Lindborg | 1.466 | 1.069 | 0.976 | 0.985 | 0.977 | 0.941 | 0.953 | 0.988 | 0.998 |
| decimated 1 deg -> 2 deg, Lindborg | 1.700 | 1.158 | 1.101 | 1.046 | 1.029 | 1.014 | 1.006 | 0.997 | 0.997 |
| bilinear, 2 deg, k^-3 | 1.122 | 0.891 | 0.915 | 0.928 | 0.934 | 0.957 | 0.968 | 0.981 | 0.986 |
| decimated, k^-3 | 1.598 | 1.098 | 1.064 | 1.024 | 1.017 | 1.007 | 1.003 | 0.998 | 0.998 |
| route control (band-limited, no interpolation) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The error first exceeds 10 % at 2.8-3.3 grid lengths (bilinear 2 deg,
decimation) and 5 % at 4.1-5.1; the sign change (foldover excess against
kernel attenuation) is inside 3 grid lengths. At 4 grid lengths what
remains is a smooth 2-3 % excess (decimation) or a 6 % attenuation
(bilinear, 4-6 % out to about 5 grid lengths), which moves a 50 %
crossing by `ln(0.94) / departure_log_slope`: under 1.3 % in wavelength
for the departures the calibration reads (log-slope -4.7 and steeper).
On the 2 deg synthetic product (band 889.6 km) a 600 km cutoff reads
`inside_product_trust_band` on all six region x reading cells and a
1000 km one is confirmed only inside the band (the absolute departure
begins at 4.8-4.9 grid lengths with 8-9 degrees outside the band, but
its minimum ratio outside is 0.34-0.36, reached only inside). The
bilinear attenuation of 6 % at the band edge is recorded, not corrected.

**Native Gaussian grids** (door `tools/native_ke_stick.py <atmf.nc>
--label ... --out ...`). A field already on Gaussian rows (the GFS/GDAS
NetCDF on the T1534 linear grid, 3072 x 1536, 127 hybrid levels) goes
through `spectrum_from_gaussian` (`measure_product(latitude_grid=
"gaussian")`): the rows are checked against the Gauss-Legendre nodes of
their own count (tolerance 1e-4 deg: float32 rounding of the nodes
measures 3.6e-6 to 3.8e-6 deg at 64-1536 rows, a regular 0.25 deg grid
departs from the 721-node set by 0.191 deg), at least `T + 1` rows are
required for the `T = nlon/2 - 1` the columns imply (3072 -> T1535), and
the field is analysed in place: no interpolation, no trust band. Rows
that are not Gaussian are refused with the measured departure; the two
routes never guess which kind of axis they hold. Measured recovery on
the T63 linear grid (64 x 128, rows reversed and columns rolled as a
file might hold them): `KE_n` within 4.8e-13 relative of the planted
spectrum. The tool takes the wind at 250 hPa from `ugrd`/`vgrd` by
linear interpolation in ln p between the bracketing layers, with the
layer-bottom pressure `phalf[0]` plus the top-down cumulative `dpres`
(levels ordered by `pfull`; units read from the file, refused when
unstated) checked against `pressfc` and refused above 1 % (float32
rounding of a 127-term sum is ~1e-5, so a percent is a units or
level-order error). The reading is of the wind on the grid the model
wrote: for an FV3-based GFS that is the write component's remap of the
cubed-sphere state at like resolution, which the instrument does not
undo and says so in its report. The real file's row departure and
surface-pressure closure, measured 2026-09-02 on all seven GFS/GDAS
files below (instrument tree `2e9e5532b`): the rows depart from the
Gauss-Legendre nodes of their own count by 8.5e-14 deg against the 1e-4
tolerance, and the `pressfc` closure of the top-down cumulative `dpres`
is 0.338 % to 0.345 % against the 1 % refusal. Both are in "Measured
readings" below.
At T1535 the numpy analysis needs ~29 GB per Legendre table; the tool's
`--backend cupy` runs it on a card.

**Gates** (`measure_run`): `ensemble_size` >= 8 hemispheric samples
(single-shot spread below); `every_sample_resolved` = 1.0 (the ensemble
mean is not formed from a stand-in; an unresolved, unfitted or
unconfirmed sample fails the gate instead, and the report lists the
statuses); `synoptic_slope_in_family` |slope + 3| <= 0.6 (the self-relative
reading anchors on that fit; slope recovery on the k^-3 family below);
`effective_resolution_near_truncation` mean self-relative / lambda(T)
<= 2.0. This core's value under this instrument, measured 2026-09-01/02
on T255 (the 48 h control and both arms of the 24 h A/B, instrument tree
`2e9e5532b`): the gate carries NO number. `every_sample_resolved` reads
0.0 on all three reports, so the ensemble mean the ratio needs is never
formed, the reported value is NaN and the gate fails on that rather than
on a wide ratio. `ensemble_size` (52, 20, 20) and
`synoptic_slope_in_family` (0.0032, 0.057, 0.054) pass on all three. The
readings are in "Measured readings" below.

## Calibration of record (2026-09-01, `tools/dynamics_spectrum_calibration.py`)

T127. Planted truth = the 50 %-power wavelength of the filter
`exp(-(k_n c)^4)`, `c = (ln 2)^(1/4) lambda_truth / 2 pi`, applied to the
coefficients. Values are reading/truth; `+-` is the single-shot standard
deviation over seeds. Bases: `k^-3` (a pure power law, on which the
self-relative truth is exact) and `lindborg` (the reference itself, on
which the absolute truth is exact).

Self-relative reading, k^-3 base:

| family | region | 400 km | 600 km | 900 km |
|---|---|---|---|---|
| isotropic (30 seeds, noise kept) | global | 0.986 +- 0.102 | 0.970 +- 0.065 | 0.916 +- 0.038 |
| isotropic | northern | 1.055 +- 0.190 | 1.003 +- 0.094 | 0.922 +- 0.053 |
| isotropic | southern | 1.021 +- 0.155 | 0.984 +- 0.107 | 0.921 +- 0.061 |
| isotropic, rescaled to the base exactly | global | 0.986 | 0.974 | 0.917 |
| isotropic, rescaled | northern | 0.976 | 0.963 | 0.917 |
| zonally coherent (u = W(lat) f(lon), rescaled) | global | 0.986 | 0.974 | 0.917 |
| zonally coherent | northern | 0.976 | 0.960 | 0.917 |
| real product (GFS 0.25 deg f024 250 hPa, T719 -> T127, rescaled) | global | 0.986 | 0.974 | 0.917 |
| real product | northern | 0.976 | 0.960 | 0.917 |
| real checkpoint (T127 baroclinic wave, hour 12, rescaled) | northern | 0.986 | 0.974 | 0.917 |

Unresolved: 0 of 30 seeds at every truth and region. Fitted synoptic slope
on the k^-3 base: -3.02 +- 0.18 (global), -2.94 +- 0.24 (northern), -3.01
+- 0.28 (southern).

Absolute reading, Lindborg base:

| family | region | 400 km | 600 km | 900 km |
|---|---|---|---|---|
| isotropic (30 seeds) | global | 0.989 +- 0.015 | 0.990 +- 0.020 | 0.985 +- 0.025 |
| isotropic | northern | 0.987 +- 0.021 | 0.996 +- 0.035 | 0.987 +- 0.043 |
| isotropic | southern | 0.993 +- 0.027 | 0.989 +- 0.023 | 0.993 +- 0.041 |
| isotropic, rescaled | global | 0.996 | 0.988 | 1.000 |
| zonally coherent | northern | 0.996 | 0.988 | 0.978 |
| real product | northern | 0.996 | 0.988 | 0.978 |
| real checkpoint | northern | 0.996 | 0.988 | 0.978 |

**Family agreement.** Global readings are identical across families: the
instrument is a function of the total-degree spectrum only, which is what
retires audit SP-3 (2.6x between row-independent and row-identical
fields). Hemispheric readings agree within 1.5 % across the rescaled
isotropic, zonally coherent, real-product and real-checkpoint families
(0.960-0.974 at 600 km), the residual being the window's leakage acting
on different latitudinal structures.

**Known biases, recorded, not corrected.** The self-relative reading is
biased fine by 3-8 % for cutoffs within ~1.5x of the 1200 km fit edge (a
900 km filter already removes 20 % of the power at 1200 km, steepening
the fit) and its single-shot noise grows with the extrapolation distance
(10 % at 400 km); an ensemble of >= 8 is the gate for that reason. On a
Lindborg-shaped field the self-relative reading is 0.88-0.92x the filter's
50 % point, because the reference's k^-5/3 mesoscale term sits above the
synoptic fit; on a pure k^-3 field the absolute reading is 1.10-1.35x it,
because such a field is already below the reference's mesoscale before the
filter acts. A real atmosphere is neither base; both numbers are what the
criterion returns for what it is given.

**Controls.** No-cutoff fields, global: 0/60 read as resolved. Scaling a
field by 0.25x or 4x: moved the self-relative reading 0/60 times and the
absolute reading 60/60 times (audit SP-1 retired). The hemispheric
no-cutoff control is the subject of the next section.

## The departure gates, measured (2026-09-01, adversarial calibration)

An independent calibration of the rebuilt instrument, on its own synthetic
constructions, found four defects in the readings above. Each is stated
with its magnitude, then what closes it and what that costs. All numbers
are from `tools/dynamics_spectrum_calibration.py --sections gates residual
slope_band coherence` at T127 (record: `CALIBRATION` in the module).

**F1: hemispheric false positives.** On 120 hemispheric k^-3 fields with
NO cutoff (60 seeds, both hemispheres) the self-relative reading returned
"resolved" 17 times (NH 12, SH 5; global 0/120), at 314-474 km, with
1-44 degrees of support. Mechanism: the hemispheric fit over 20 degrees of
a windowed half-sphere has a slope spread of 0.26 (global 0.18) and the
reference is extrapolated from n ~ 22 to n = 127, so a fit 0.4 shallower
than the field moves the reference at the truncation by 2x and a k^-3
tail sits at half of it. The false positives fitted -2.61 +- 0.10 against
a population of -2.99 +- 0.26. Extended to 300 seeds: 79/600 (13 %), at
314-651 km, support 1-67, and the ratio inside the departure never below
0.211.

**F2: no minimum support.** A pure k^-3 spectrum with only its LAST
degree x0.05 read "resolved" at 316.5 km against a truncation wavelength
of 314.0 km, the most flattering value the reading can take.

**F3: shape dependence of the self-relative reading** (recorded, not
corrected - it is a property of a self-anchored criterion; `CALIBRATION
["rolloff_shape"]`, tool section `shapes`). The reading is calibrated on
the quartic rolloff `exp(-ln2 x^4)`, `x = k lambda_truth / 2 pi`; a
gentler rolloff leaks into the 1200-3000 km fit window and steepens the
fit, so the reading is finer than truth. On the exact k^-3 base, with the
gates on, reading/truth at 400 / 600 / 900 km is:

| rolloff | self-relative | fit slope | `departure_log_slope` |
|---|---|---|---|
| quartic `exp(-ln2 x^4)` | 0.986 / 0.974 / 0.917 | -3.01 / -3.04 / -3.22 | -4.7 / -13.0 / -43 |
| `1/(1+x^8)` | 0.996 / 0.988 / 0.978 | -3.00 / -3.00 / -3.08 | -5.7 / -7.2 / -7.6 |
| Gaussian `exp(-ln2 x^2)` | unresolved / 0.829 / 0.735 | -3.07 / -3.16 / -3.37 | -1.9 / -3.1 / -5.5 |
| exponential `exp(-ln2 x)` | unresolved / unresolved / unresolved | -3.16 / -3.23 / -3.35 | - / - / -1.4 |

The audit's raw-criterion values (gates off) were 0.898 / 0.829 / 0.735
for the Gaussian and unresolved / unresolved / 0.438 for the exponential;
the confirmation gate now refuses the Gaussian at 400 km and the
exponential everywhere at T127, because those rolloffs never reach 20 %
of the reference before the truncation. The absolute reading on the
Lindborg base is 0.996 / 0.988 / 1.000 for the quartic and `1/(1+x^8)`
shapes, unresolved / 0.988 / 1.000 for the Gaussian and unresolved /
unresolved / 0.978 for the exponential (the audit's raw values were 0.996
/ 0.988 / 1.000 for every shape); where it resolves it is shape-free. On
the isotropic family (10 seeds) the Gaussian reads 0.829 +- 0.090 (600
km) and 0.748 +- 0.072 (900 km) globally, with `departure_log_slope`
-3.14 +- 0.19 against the quartic's -12.99 +- 1.05 and the hyper-8's
-7.19 +- 0.08 at the same truth; wave packets rescaled to k^-3 read the
exact-base values. What the instrument does about it: every reading
carries `departure_log_slope`, which is truth-dependent and is read
against this table at the same wavelength, never against a threshold;
the self-relative reading is quoted as "the 50 % departure from the
fitted synoptic law", never as "the filter's half-power point"; nothing
is corrected.

**F4: no fit-residual gate.** A field of zonally coherent rows (one phase
set for every row of a 25-65 degree band) fits a synoptic slope of -3.08
- inside the family gate - while KE_n / fit runs 0.61 -> 3.54 -> 1.61
across n = 10-40, and read "resolved" at 942 km on every seed with nothing
planted.

**What closes F1, F2 and F4.** Three gates, each measured (F3 above is
recorded, not closed):

- Minimum support 3 degrees (F2). A one-degree drop reads support 2 after
  the three-degree smoothing. Support alone does NOT close F1: 0/120
  needs 45 degrees and 0/600 needs 68 (nothing finer than 660 km would be
  readable at T127, and 45 degrees is the whole examined range at T63).
- Confirmation depth 0.20 (F1). Inside the supported range the ratio must
  reach 20 % of the reference. False positives left of 600 by depth: 15
  at 0.35, 6 at 0.30, 2 at 0.25, 0 at 0.20 (the two lowest reached 0.211
  and 0.227); of the audit's 120: 1 at 0.35, 0 at 0.30 and below. A
  planted quartic cutoff reaches 0.000 at 600 and 900 km and 0.04-0.37
  (mean 0.15) at 400 km, so the gate refuses 9/30 global, 5/30 northern
  and 6/30 southern single samples at 400 km (1.27x the truncation
  wavelength) and none at 600 or 900 km; the survivors read 1.03 +- 0.09
  / 1.09 +- 0.19 / 1.07 +- 0.14 of truth. No absolute reading is refused
  (its planted depth is 0.05-0.07 at 400 km). The false-positive tail is
  continuous: 0/600 bounds the per-sample rate near 0.3 %, it does not
  make it zero, and the ensemble gate remains the wall.
- Fit residual limit 0.35 (F4). Weighted RMS log-residual of the synoptic
  fit: isotropic k^-3 0.170 +- 0.038, max 0.299 over 600 hemispheric
  samples and 0.180 +- 0.032, max 0.289 over 300 global; wave packets
  with and without a 250 km front max 0.269 (30 samples); the coherent
  rows 0.97 globally and 0.53-0.57 in the hemispheres (min 0.526). The
  limit is 4.8 sigma above the isotropic mean, 1.17x the largest
  power-law residual and 0.67x the smallest coherent-rows residual. The
  residual of a real product, measured 2026-09-02 on the native grid
  (five GFS `atmf012` cycles at T1535, 250 hPa, instrument tree
  `2e9e5532b`): 0.141 to 0.316 over the fifteen region readings, mean +-
  SEM 0.22 +- 0.03 global, 0.26 +- 0.02 northern, 0.21 +- 0.03 southern;
  0.170 to 0.300 on the two GDAS analyses. Nothing reached the limit, so
  no real reading has yet been refused by it.

**The slope band, measured and found not to be the mechanism.** Requiring
the hemispheric slope within a band of the global slope of the same
subject was the proposed fix for F1. Measured: the 79 false positives'
differences from their global fit are 0.28 +- 0.16 (0.01..0.64), inside
the legitimate distribution (isotropic 0.017 +- 0.219, -0.58..0.64 over
600 samples; packets rescaled to k^-3 globally 0.00 +- 0.51, -0.68..0.58
over 24). A band that removes them (0.2) rejects 53 % of legitimate
hemispheres and still leaves 12 false positives of 47 at support 3. The
band is set at 0.6 (the width of the synoptic-family gate): it rejects
2.8 % of the legitimate isotropic + packet samples, catches 3/47 false
positives, and its job is the hemisphere genuinely out of family with its
globe (the meridionally white family reads +0.65).

**Family agreement with only the global base controlled** (six seeds; each
family rescaled in one global pass so its global spectrum is the base,
hemispheres carrying the family's own latitudinal structure through the
window; cell = family x hemisphere x truth; tolerance 0.10 of the
isotropic mean). Families: coherent rows (identical / correlated over
1000 km / independent phases), wave packets, packets + front.

| reading | gates off | with the gates |
|---|---|---|
| self-relative (k^-3 base) | 9/30 cells outside tolerance | 7 outside, 8 refused on every sample |
| absolute (Lindborg base) | 3/30 | 3/30 |

The 8 refused cells are the identical-rows family everywhere (unfitted:
correct, its window is not a power law) and two independent-rows cells.
The 7 remaining self-relative failures are the correlated and independent
rows at 400/600 km on 1/6 surviving samples each, and the packet families
at 400 km NH (3/6 and 2/6 survivors at 1.12-1.19x vs 0.98x). The three
absolute failures are unchanged by the gates: correlated rows NH 400 km
(1.13x vs 0.99x) and the correlated / independent rows SH 600 km (0.86 /
0.88x vs 0.98x), a -0.10 to -0.12 that the window path reads on
meridionally decorrelated structure. The packet families, the one
physically shaped family, pass all 30 absolute cells (0.95-1.05x) and every
self-relative cell at 600 and 900 km. Recorded, not corrected: a
hemispheric absolute reading on a field with white meridional phases is
0.9x its global one.

**What a single hemispheric self-relative reading is worth, stated.** At
T127 a departure finer than ~1.5x the truncation wavelength on one
hemisphere is within the fit's own extrapolation noise; the instrument
now refuses such a reading unless the spectrum drops to a fifth of the
reference, and the number of record is the ensemble mean over >= 8
hemispheric samples that all resolved. Single-hemisphere readings quoted
before these gates (the GFS northern-hemisphere 300 km below) are
withdrawn.

## First readings under this instrument

GFS 0.25 deg f024 of 2026-08-31 00Z, 250 hPa, analysed at T719
(lambda(T) = 55.6 km), one cycle:

| region | slope (1200-3000 km) | self-relative | absolute | retained fraction 2000 / 1000 / 500 / 250 km |
|---|---|---|---|---|
| global | -2.83 | unresolved (0.51 at 55.6 km) | unresolved (ratio 0.51 at 55.6 km) | 1.35 / 1.38 / 1.18 / 0.81 |
| northern | -2.33 | 300 km | 56.7 km (ratio 0.42 at 55.6 km) | 1.20 / 1.30 / 1.70 / 1.04 |
| southern | -2.94 | unresolved | 56.1 km (ratio 0.40 at 55.6 km) | 3.14 / 2.00 / 1.02 / 0.78 |

**Withdrawn (2026-09-01), for two reasons.** First, the northern
self-relative 300 km is a single-hemisphere reading made before the
support and confirmation gates, on a -2.33 fit; it is exactly the class
of reading F1 showed to be within the fit's noise, and it is not restated.
Second, the 0.25 degree pgrb2 product is interpolated by the
post-processor from the 13 km native Gaussian grid with no low-pass, so
sub-grid variance aliases into the band near the product truncation: the
absolute crossings at 56-57 km (two grid lengths of the product) are
foldover, not resolution (measured above: 1.12-1.70x the planted power at
two grid lengths), and the IFS open-data product is smoothed the other
way. Under the instrument as it now stands those crossings lie inside
the 111 km trust band and would be reported `inside_product_trust_band`
with no wavelength. What survives of this table is the retained fraction
at 2000 / 1000 / 500 km, which the band does not reach (250 km is 9 grid
lengths and survives too; the tool now withholds only entries inside the
band). The reading of record for a real analysis is the native-grid one
(`tools/native_ke_stick.py` on the GFS/GDAS NetCDF, T1534 linear Gaussian
grid, 127 hybrid levels, analysed in place at T1535), measured 2026-09-02
on five GFS `atmf012` cycles and two GDAS `atmanl` analyses and stated
with its SEM in "Measured readings" below. GFS now has the five cycles a
scoreboard needs; GDAS has two analyses and is not one yet.

The WOOF global core has been measured at T255 (52 km) and only there.
The T255 readings, the T533 arm that produced none, and the T63 GDAS 48 h
run that has still not been re-read are in "Measured readings" below.

## Measured readings (2026-09-01 and 2026-09-02)

Every reading in this section was made with the instrument tree whose
`SYNCED-COMMIT.txt` reads `2e9e5532b`, except the public-product set at the
end, which was made with the earlier `9e39efbc7`. The measuring checkouts
carry no `.git`, so there is no `git describe` for them; the identity of
record is that commit file, and `woof/verify/harness/dynamics_spectrum.py`,
`tools/native_ke_stick.py`, `tools/external_ke_stick.py` and
`tools/arwen_global_ke_stick.py` at `2e9e5532b` are byte-identical to this
document's own tree. Recorded as a gap: a report's `provenance` block holds
only the absolute reference's constants and the calibration record, so the
tree that produced a JSON is not IN the JSON and is carried here from the
measuring checkout instead.

Every wavelength below is the instrument's own isotropic convention,
`lambda(n) = 2 pi a / sqrt(n(n+1))`, the `wavelength_convention` key each
report states: `lambda(T)` is 26.07 km at T1535, 55.64 km at T719, 75.04 km
at T533 and 156.68 km at T255. Nothing is quoted as a multiple of dx.

### Native Gaussian-grid GFS and GDAS, 250 hPa (measured 2026-09-02)

Door `tools/native_ke_stick.py --level-pa 25000`, numpy float64, one file
per report, on the T1534 linear Gaussian NetCDF (3072 x 1536, 127 hybrid
levels) analysed in place at T1535. `regrid` reads "none" and there is no
product trust band: nothing was interpolated. Reports (the `--out` path
takes whatever name it is given and always holds JSON):
`gfs.20260830.t18z.atmf012`, `gfs.20260831.t00z.atmf012`,
`gfs.20260831.t12z.atmf012`, `gfs.20260831.t18z.atmf012`,
`gfs.20260901.t00z.atmf012`, `gdas.20260901.t00z.atmanl` and
`gdas.20260830.t18z.atmanl`. All seven the fetch chain named exist; none
was skipped.

**The route checks the tool reports.** On every one of the seven files: row
departure from the Gauss-Legendre nodes 8.5e-14 deg (tolerance 1e-4 deg);
`pressfc` closure of the top-down cumulative `dpres` 0.338 % to 0.345 %
(refusal above 1 %); levels 58 to 65 of 127 bracket 25000 Pa; maximum wind
speed 89.4 to 103.3 m/s. Wall clock 122 to 146 s per file on the CPU.

GFS `atmf012`, five cycles (2026-08-30 18Z, 08-31 00Z, 08-31 12Z, 08-31 18Z,
09-01 00Z), mean +- SEM over the five:

| region | slope (1200-3000 km) | self-relative | absolute | retained fraction 2000 / 1000 / 500 / 250 km |
|---|---|---|---|---|
| global | -2.82 +- 0.03 | unresolved 5/5 | 56.5 +- 0.6 km | 1.16 +- 0.14 / 1.61 +- 0.08 / 1.04 +- 0.05 / 0.88 +- 0.02 |
| northern | -2.53 +- 0.05 | resolved 5/5, 38.4 to 74.3 km | 58.7 +- 0.6 km | 1.46 +- 0.20 / 1.84 +- 0.18 / 1.60 +- 0.06 / 1.12 +- 0.04 |
| southern | -2.93 +- 0.03 | unresolved 5/5 | 58.2 +- 1.6 km | 2.67 +- 0.30 / 1.93 +- 0.07 / 1.03 +- 0.06 / 0.70 +- 0.03 |

The absolute reading resolved on 15 of 15 region readings and every one is
anchored (250 hPa is inside the 200-300 hPa band); its
`departure_log_slope` is -3.34 to -4.76 with 810 to 915 degrees of support,
and `KE_n / E_n` at `lambda(T)` is 0.02 to 0.04. **No self-relative number
of record is formed for GFS**: the ten hemispheric samples clear
`ensemble_size`, but only the five northern ones resolved, so
`every_sample_resolved` is 0.5 and the rule stated above ("the number of
record is the ensemble mean over >= 8 hemispheric samples that all
resolved") refuses the mean. The five northern values are listed as a range
for that reason, not as a reading.

GDAS `atmanl`, two analyses. Two is not a scoreboard, so both are given
whole rather than averaged:

| analysis | region | slope | self-relative | absolute | retained fraction 2000 / 1000 / 500 / 250 km |
|---|---|---|---|---|---|
| 2026-08-30 18Z | global | -2.70 | 34.98 km | 55.41 km | 0.86 / 1.60 / 0.90 / 0.97 |
| 2026-08-30 18Z | northern | -2.45 | 62.89 km | 64.83 km | 2.48 / 2.38 / 1.43 / 1.24 |
| 2026-08-30 18Z | southern | -2.78 | 34.23 km | 51.49 km | 1.44 / 1.84 / 1.13 / 0.80 |
| 2026-09-01 00Z | global | -2.82 | unresolved | 55.64 km | 1.38 / 1.53 / 1.24 / 1.07 |
| 2026-09-01 00Z | northern | -2.32 | 244.84 km | 57.23 km | 1.18 / 1.50 / 1.81 / 1.38 |
| 2026-09-01 00Z | southern | -3.05 | unresolved (support 1) | 61.73 km | 3.18 / 2.02 / 1.10 / 0.84 |

The 2026-09-01 00Z northern self-relative 244.84 km sits on a -2.32 fit
with a `departure_log_slope` of -1.36; it is a single hemispheric reading
of exactly the class F1 measured, and it is recorded, not quoted as a
resolution.

These are readings of the wind on the grid the model wrote. For an
FV3-based GFS that grid is the write component's remap of the cubed-sphere
state at like resolution, which this instrument does not undo.

### The WOOF global core at T255, 52 km (measured 2026-09-01 and 2026-09-02)

Door `tools/arwen_global_ke_stick.py` (`measure_run`) on the run's own
checkpoints. T255, `lambda(T) = 156.68 km`, 40 levels, native physics
suite, dt 50 s, initialised from the GDAS analysis of 2026-09-01 00Z, read
at 250 and 500 hPa. One instrument tree (`2e9e5532b`) read all three
reports below, including the archived control, so they are comparable.

**Control, hours 12 to 48 every 3 h** (13 frames, 52 hemispheric samples;
report `after48nat-northern-25000`). The half-variance scale reads **none**:
every self-relative and every absolute sample is `unresolved`, at both
levels and in both hemispheres. The spectrum never falls below half of the
observed reference inside the resolved range. Gates: `ensemble_size` 52
PASS, `every_sample_resolved` 0.0 FAIL, `synoptic_slope_in_family` 0.0032
PASS, `effective_resolution_near_truncation` NaN FAIL for want of a
resolved ensemble. Mean fitted synoptic slope -3.003.

Retained fraction `KE_n / E_n` over the 13 anchored 250 hPa samples, mean
+- population sd, with the ratio at `lambda(T)` in the last column:

| region | 1000 km | 500 km | 250 km | at 156.68 km | slope |
|---|---|---|---|---|---|
| global | 0.90 +- 0.23 | 0.72 +- 0.08 | 1.15 +- 0.27 | 1.10 +- 0.38 | -3.00 |
| northern | 1.20 +- 0.50 | 0.94 +- 0.16 | 1.40 +- 0.30 | 1.24 +- 0.40 | -2.81 |
| southern | 0.99 +- 0.17 | 0.73 +- 0.25 | 0.72 +- 0.29 | 0.57 +- 0.22 | -3.20 |

Divergent share of `KE_n` by degree band, same samples (the 201-400 band is
clipped at the T255 truncation; a mostly divergent tail is not the balanced
cascade):

| level | region | n 2-20 | 21-60 | 61-120 | 121-200 | 201-255 |
|---|---|---|---|---|---|---|
| 250 hPa | global | 0.02 | 0.09 | 0.31 | 0.56 | 0.71 |
| 250 hPa | northern | 0.03 | 0.07 | 0.25 | 0.53 | 0.71 |
| 250 hPa | southern | 0.03 | 0.05 | 0.26 | 0.59 | 0.77 |
| 500 hPa | global | 0.01 | 0.07 | 0.27 | 0.43 | 0.53 |
| 500 hPa | northern | 0.01 | 0.06 | 0.27 | 0.45 | 0.55 |
| 500 hPa | southern | 0.01 | 0.04 | 0.20 | 0.38 | 0.51 |

**The semi-implicit A/B, 24 h frames** (hours 12, 15, 18, 21, 24; 250 and
500 hPa; 20 hemispheric samples per arm). The control report is
`after48nat-f024-northern-25000`, the archived arm re-read by this
instrument tree; the treatment report is
`siab48-vm-f024-northern-25000`, the vertical-mode scheme. Both are 24 h
readings.

- Control: absolute resolved on 0 of the 10 anchored hemispheric samples,
  self-relative unresolved on every sample. Mean fitted synoptic slope
  -2.943.
- Treatment: absolute resolved on 4 of the 5 southern frames, at 410.6 km
  (h12), 394.4 km (h15), 419.2 km (h18) and 457.5 km (h24); h21 unresolved.
  Mean 420.4 km, sd 26.8 km, SEM 13.4 km over the four. Northern 0 of 5.
  Self-relative unresolved on every sample. Mean fitted synoptic slope
  -2.946.

The four southern departures carry `departure_log_slope` -0.70 to -1.01
with 155 to 169 degrees of support, and their minimum ratio inside the
departure is 0.171 to 0.194 against the 0.20 confirmation depth, so the
depth gate is met with little margin. The retained fractions say where the
difference is: at 250 hPa the treatment holds 0.60 +- 0.27 of the reference
at 250 km southern against the control's 0.70 +- 0.29, and 0.43 +- 0.22 at
`lambda(T)` against 1.07 to 1.21 globally and northern in the control; the
divergent share of the 121-200 band falls from 0.55 to 0.44 southern and
from 0.53 to 0.38 northern. Both arms still fail `every_sample_resolved`
and both carry NaN for `effective_resolution_near_truncation`.

Pending at the time of writing: the 48 h reading of the treatment arm and
the same-tree control arm (`scheme = "external"`), which were still
running. Nothing is stated for either.

### T533, 25 km: not measured (2026-09-02)

The production-resolution arm produced no spectrum, and the reason is
device memory, not the instrument. Both launches died before a usable
checkpoint: the first at its first radiation call, where RRTMGP cloud
optics with a 12,500-column chunk needed about 22 GB beside the 10.41 GiB
T533 dycore state; the second, with the radiation chunk cut to 5,000
columns, at its first cumulus call, `cupy.cuda.memory.OutOfMemoryError`
allocating 3,284,997,120 bytes with 32,231,514,112 bytes already allocated
inside the Grell-Freitas driver, `woof/globe/core/gf.py`. The run left one
checkpoint at hour 0 and the door refused it by the spin-up rule: "no
samples: every checkpoint is younger than 12.0 h". The Grell-Freitas
column chunking that bounds that allocation landed afterwards at
`a74805618` and is in this document's tree;
no T533 run has been read since. The T533 reading is **not measured** under
the current build, for that reason.

An earlier T533 24 h run was read on 2026-09-01 with the earlier build
`9e39efbc7` (reports `run-t533-<region>-<level>.json`, hours 12, 15 and 18,
three frames at 250 and 500 hPa, 12 hemispheric samples). It resolved
nothing: every self-relative and every absolute sample is `unresolved`, and
three of its four gates fail, `synoptic_slope_in_family` at 0.602 against
the 0.6 limit as well as `every_sample_resolved` and the truncation ratio.
What it does record is a tail far above the observed reference: `KE_n / E_n`
at `lambda(T) = 75.04 km` is 5.4 +- 2.4 global, 4.0 +- 1.0 northern and
4.0 +- 2.4 southern over the three anchored 250 hPa frames, against 2.0 +-
0.5, 1.9 +- 0.3 and 1.7 +- 0.4 at 250 km. That build predates the support,
confirmation and residual gates and the product trust band, and three
frames is not an ensemble, so this is recorded, not a reading of record.

### T63: still to be re-measured

The T63 GDAS 48 h core, whose superseded 560 +- 23 km reading is below, has
not been read under this instrument. It stays **to be re-measured**.

### Public 0.25 degree products (read 2026-09-01, earlier instrument build)

Twenty-six reports from `tools/external_ke_stick.py` on the 0.25 degree
GRIB2 products exist: GFS f024 (`gfs.20260830.t00z.f024`,
`gfs.20260830.t12z.f024`, `gfs.20260831.t00z.f024`, `gfs.20260831.t12z.f024`,
`gfs.20260901.t00z.f024`), IFS open data f024 (`ifs.20260830.t00z.0p25.f024`,
`ifs.20260831.t00z.0p25.f024`, `ifs.20260901.t00z.0p25.f024`), AIFS single
f024 (`aifs-single.20260830.t00z.0p25.f024`,
`aifs-single.20260831.t00z.0p25.f024`,
`aifs-single.20260901.t00z.0p25.f024`) and two GDAS f000 analyses
(`gdas.20260830.t18z.f000`, `gdas.20260901.t00z.f000`), each written twice
under a `-northern` and a `-southern` name that hold identical three-region
content, so the 26 files are 13 distinct readings. Twelve more reports in
the same folder are core runs read from their own checkpoints, not
products: `run-t533-*` (above), `run-before48nat-*` and `run-b384nat-*`.

They were read with the earlier build `9e39efbc7`. Its reports carry no
`product_grid_length_km`, no `product_trust_band_km`, no
`fit_rms_log_residual` and no departure `support`: that build predates the
product trust band and the support, confirmation and residual gates. Their
departure wavelengths are therefore **not restated here as readings of
record** under the instrument as it now stands, for the same two reasons
the GFS table above them was withdrawn. What they establish is that IFS and
AIFS readings exist to be re-read, and that unlike the GFS and GDAS
crossings at 55 to 65 km most of them are nowhere near the 111 km trust
band of a 0.25 degree product, so they are not foldover and would survive
it. Where they resolved: AIFS absolute 340.7 to 473.8 km (7 of 9 region
readings) and AIFS self-relative 468.2 to 661.7 km (5 of 9); IFS absolute
132.3 to 220.6 km (7 of 9) and IFS self-relative 78.6 to 398.3 km (5 of 9),
of which two, 78.6 km and 109.5 km, would fall inside the band and be
withheld. Re-reading the three IFS and three AIFS cycles on the current
build is the next thing this scoreboard needs.

## Truncation scale under the isotropic convention

The retired document said "truncation scale" without naming a convention,
and the convention it used was the zonal wavelength of `m = T` at 45
degrees latitude. The audit (SP-5) recomputed both conventions and both
flagship claims. Its numbers, quoted:

- T63 (nlon 192): dx at 45 deg 147.4 km; zonal truncation at 45 deg
  `2 pi a cos45 / T` = 449.3 km = 3.05x dx; the published 560 km reads
  560/449.3 = 1.246 and 560/147.4 = 3.797 there (the document's 449, 3.05,
  1.25, 3.80). Isotropic `2 pi a / T` = 635.4 km, and **560/635.4 = 0.88**.
- T533 (nlon 1602): dx at 45 deg 17.7 km; zonal truncation at 45 deg
  53.1 km = 3.01x dx; the published 70 km reads 70/53.1 = 1.318 there (the
  document's 1.3). Isotropic `2 pi a / T` = 75.1 km, and
  **70/75.1 = 0.93**.

Under the isotropic convention both readings sit BELOW truncation, so the
"1.25x truncation" and "1.3x truncation" claims are statements about the
zonal-at-45-degrees convention only. The audit's `2 pi a / T` differs from
this instrument's `2 pi a / sqrt(n(n+1))` at n = T by 0.8 % at T63 and
0.09 % at T533. The 560 km and 70 km readings themselves are superseded
(band-FFT instrument) and are not restated as claims under either
convention: they are to be re-measured.

## The audit findings, in plain language

Six findings, all confirmed by the audit's own reproduction, against the
retired 10-row FFT-band path (now `band_fft_diagnostic`, secondary, never
used by the instrument).

**SP-1 (minor). The "observed" reference never consulted an observed
amplitude.** It was the Nastrom-Gage shape (k^-3 shallowing to k^-5/3
below 500 km) with its amplitude least-squares anchored on the subject's
own 2000-6000 km bins. Measured: scaling both wind planes by 0.25x and by
4x left the reading bit-identical (312.8 km, slope -3.059 on the auditor's
row-independent synthetic; 608.7 km observed-reading, 595.9 km self-fit
reading, slope -3.145 on an isotropic 2-D synthetic), with the reference
amplitude tracking the field's own exactly. A model with half the real
atmosphere's synoptic energy read the same effective resolution.
Consequence for this document: "holds >= 50 % of observed variance" and
"absolute variance sits under 50 % of observed" were not what the code
measured. Retired by the `absolute` reading, which is in physical units:
the same 0.25x/4x scaling now moves it 60/60 times.

**SP-2 (major). No departure found returned the grid-scale bin, unflagged.**
`return float(wavelength_km[-1])` with no status. The fallback value at
the T63 band geometry is 296.4 km (`192 * 147.425 / 95.5`, 95 retained
bins). Measured on the repo's own calibration family: a 400 km cutoff hit
the fallback on 20/20 seeds (296 +- 0 km), a 600 km cutoff read 313 +- 7
km (0.48x the 658 km truth) with 2/20 seeds falling through, and a
no-cutoff control returned exactly 296.4 km on all 20 seeds, so "no
departure" and "departure at the last bin" were the same returned number
and neither gate could separate them (both pass more easily on the
fallback). The audit's tempering, recorded: on a genuinely isotropic 2-D
field with the same rolloff there were 0 fallback hits in 60 runs, and
neither headline reading is the fallback value (T63 560 km vs 296.4 km;
T533 70 km vs a ~35 km last bin). Retired by the explicit statuses
(`resolved` / `unresolved` / `below_reference_throughout`) and a
`wavelength_km` of `None` unless resolved.

**SP-3 (major). The reading moved 2.6x with the meridional coherence of
the band.** Same 1-D target spectrum, 20 seeds, T63 band geometry:
row-independent planes read 296 / 313 / 492 km for 400 / 600 / 900 km
cutoffs (0.68 / 0.48 / 0.50x the 438 / 658 / 986 km truths), row-identical
planes read 773 / 809 / 949 km (1.76 / 1.23 / 0.96x), and a genuinely
isotropic 2-D field with the same rolloff read 378 / 526 / 728 km (0.86 /
0.80 / 0.74x), between the two extremes. Mechanism: `spectral.py` forms
the radial index from bin-unit magnitudes with `span = max(ny, nx) = 192`,
so with 10 rows the ky quantum is 19.2 radial units and bin counts jump
from 2 per bin to 24 / 14 / 10 at bin 19 (1452 km); every bin at or beyond
that mixes low-kx power into high-|k| annuli. `ke/reference` at the 658 km
truth was 6.69 with independent rows and 0.17 with identical rows. The
"common-mode across band geometries" calibration key is contradicted by
this directly, and absolute km were comparable only between subjects of
the same meridional structure. Retired: the total-degree spectrum has no
band and no radial binning, and global readings are identical across all
four calibration families.

**SP-4 (minor). One dx at 45 degrees was applied to both FFT axes.**
At T63 the 35-55 degree band holds 10 rows with mean dlat 1.865 deg, so
dy = 207.4 km against dx45 = 147.4 km, `dy/dx45 = 1.407`, and the j = 1
meridional mode's true wavelength of 2074 km was binned at 1452 km. Any
meridionally dominated departure was reported at 1/1.407 = 0.71x its true
wavelength. The audit refuted the sub-claim that the Hann leak entered the
fit window: the window was selected on the instrument's own wavelength
axis, where the leak sits at 1452 km and is excluded regardless. Retired
with the band.

**SP-5 (note). "Truncation scale" was the zonal wavelength at 45 degrees.**
See the section above for the audit's recomputation and the 0.88x / 0.93x
isotropic readings.

**SP-6 (note). The tail smoothing zero-padded.** `np.convolve(mode='same')`
pads `log ke` with zeros, pulling the last two smoothed bins toward
`log(1) = 0`, and the 5-bin persistence guard makes the last declarable
index `len-5 = 90`, whose window reads those two padded bins. Measured
magnitude: re-running 120 measurements (400 / 600 / 900 km cutoffs x 20
seeds, row-independent) with edge-replicating padding instead reproduced
every reading identically (296 +- 0, 313 +- 7, 492 +- 77 km; the same
declaring bins; the same 20/20, 2/20 and 0/20 fallback counts). The bias
existed and changed nothing. The rebuilt criterion smooths over three
degrees with the edges averaged over the degrees that exist.

## Superseded readings (band-FFT instrument, 2026-08-30..09-01)

Every number in this section is **superseded (band-FFT instrument,
2026-08-30..09-01), to be re-measured**. They are kept so they can be
recognised when met in older reports, evidence folders and commit
messages. Nothing here has been converted to the isotropic convention or
replaced by an estimate.

**Which reading each was.** The retired instrument computed two departure
values per plane: `effective_vs_observed_km`, against the shape-anchored
Nastrom-Gage curve, and `effective_vs_self_fit_km`, against the band's own
fitted power law. Every kilometre figure published in this document, in
the gates and in the campaign reports is `effective_vs_observed_km`. Under
today's labels that is NEITHER reading: its amplitude came from the
subject (so it is not `absolute`) and its slope was the fixed -3/-5/3
shape rather than the subject's own fit (so it is not `self_relative`).
`effective_vs_self_fit_km`, the closest ancestor of today's
`self_relative`, was computed but never published.

Core, T63 GDAS 48 h (hours 12-48 every 3 h, NH and SH 35-55 deg bands, 250
and 500 hPa, n = 48):

- 560 +- 23 km (SEM), quoted as 3.80x dx and as 1.25x the 449 km
  zonal-at-45-deg truncation.
- By population: 250 hPa NH 550 +- 28; 500 hPa NH 465 +- 51; 500 hPa SH
  462 +- 38; 250 hPa SH 764 +- 190.
- Claim of record then: "the core holds >= 50 % of observed variance to
  ~1.25x its truncation scale" (SP-1: the comparison was self-anchored;
  SP-5: the truncation scale was the zonal one at 45 deg).

Cross-model f024 public 0.25 deg products, single cycle: GFS 347, IFS 370,
GDPS 375, AIFS 446, GDAS analysis f000 446 km; single-shot noise quoted as
+-100-130 km; product-pipeline damping at those scales < 1 %; our own
state pushed through the same 0.25 deg treatment 492 km.

Multi-cycle scoreboard: GFS f024, 8 cycles, 32 readings, all-band 432 +-
44 km (250 NH 420 +- 33; 250 SH 768 +- 92; 500 NH 326 +- 21; 500 SH 215
+- 9); IFS f024, 3 cycles, all-band 443 +- 82 km; IFS 250 SH 835 +- 195.
Native-grid context: GFS atmf012 287-446 km at 500/250 NH, native GDAS
analysis 237-584 km.

T533, dt = 50 run, h12-21, n = 16: 500 SH 139 -> 70 km, locked at 70 km
for three consecutive checkpoints and quoted as 1.3x truncation (53.1 km,
zonal at 45 deg); 500 NH ~200-235 km; 250 NH ~450 km (584 km at the
jet-spike hour); 250 SH ~440 km (1029 km at h21); pooled 339 +- 64 km,
against the fused dt = 40 run's 342 +- 60 km; gates then read ensemble
PASS, synoptic_slope_in_family PASS (0.39), effective/truncation FAIL
(pooled 6.44).

T533 first reading, h12 + h15, n = 8: 500 SH 69-74 km (quoted 1.3x
truncation); 500 NH 161-207 km; 250 NH/SH ~440-453 km; pooled 286 +- 62 km
= 5.4x truncation. The statement made there that the 250 bands' "absolute
variance sits under 50 % of observed" was made with the self-anchored
curve (SP-1); no absolute comparison was performed.

Instrument calibration then claimed: reads 0.69-0.75x of truth for
Gaussian-family cutoffs and 0.60x for gentle exponential rolloffs,
"common-mode across band geometries"; no false cutoffs on no-cutoff
controls (bias 1.000, 40 seeds); slope bias -0.38 on the 10-row T63 band;
single-shot noise +-100-130 km. The audit measured 0.48-0.68x
(row-independent) against 0.96-1.76x (row-identical) and 0.74-0.86x
(isotropic 2-D), and the no-cutoff controls it ran returned the fallback
bin on 20/20 seeds.

Fitted synoptic slopes from the same band planes are superseded on the
same terms (SP-3 and SP-4 act on the binned radial spectrum the slope is
fitted to, and the instrument carried a stated -0.38 band bias): pooled
-4.25 on the v3 48 h run; GFS f024 anchor -2.95 / -2.86 / -2.14 / -2.49
and v3 -3.16 / -4.02 / -4.41 / -5.43 at 250 NH / 250 SH / 500 NH / 500 SH;
GDAS f000 -4.14 NH / -4.45 SH at 250 hPa; 250 SH f024 GDPS -2.35, IFS
-1.96; hour-0 -4.83 / -4.22 (SH) and -2.26 / -2.89 (NH); hybrid vs
true-isobaric vs source at 500 NH -2.89 / -3.42 / -3.21; A/B arm h24 500
NH -3.82 (pbl800), -3.75 (diff6), -3.77 (v5).

**What does not rest on the departure reading.** Band kinetic energy and
its decay rate are plain variances of the band's wind, not radial-bin
departures, so SP-1 to SP-4 and SP-6 do not act on them: the 0-24 h 500 SH
band-EKE decay 0.513 -> 0.498/d (pbl800) and 0.513 -> 0.501/d (diff6)
against a GFS reference of 0.25-0.28/d, and the arm conclusion that both
second-order sinks are real but not controlling, are carried by that
evidence. The same holds for the other campaign findings that never used
the departure value: the downscale-transfer attribution of the SH tail
(n = 11-20 draining while n = 46-63 grows with physics and diffusion off),
the Betts-Miller and vector-transform acquittals, the bilinear-init
aliasing observation, the day-2 jet CFL record, and the spin-up rule
(checkpoints younger than 12 h measure the analysis, not the model) which
the door keeps as `--minimum-hours`. Their slope NUMBERS are superseded
above; their arm-to-arm conclusions are not restated as measurements of
this instrument until re-read.

## The companion instrument: energy-tendency attribution

`woof.verify.harness.energy_tendency` (door: `python -m
woof.verify.harness run dynamics.energy_tendency --config <run.toml>
--checkpoints <step*.npz...>`) answers the question the spectrum raises:
which operator puts energy into, or takes it out of, each band of total
degree at each sampled level. Rebuilt 2026-09-01 after the same audit
reproduced three defects in it (EN-1 to EN-3).

**What it measures.** One `model.step` is re-walked operator by operator
in the model's own order with the model's own methods: physics (dt/2),
lid absorber (dt/2), positivity repair, adiabatic core (the run's
integrator plus the semi-implicit apply) at dt, hyperdiffusion at dt,
mass fixer, physics (dt/2), lid absorber (dt/2), positivity repair, water
fixer. The change of kinetic energy per total degree (W/kg) and of theta
variance per degree (K^2/s, the proxy the available-potential-energy
closure consumes) across EACH application is booked to that operator, at
the state the model handed it. The arms therefore telescope to the step's
actual energy change; the `residual` row is the difference, and it is
read per band (quarters of 1..T, the top tenth n >= 0.9 T, and all) per
sampled level, for kinetic energy and theta variance separately. The
re-walk's final state is bit-identical to `model.step`'s on every
calibration family (relative L-inf 0.0), which is what makes the closure a
check of the arms rather than of the arithmetic.

**EN-1.** The retired design applied every operator in isolation to the
checkpoint state and lumped their O(dt) non-commutation into the residual;
its gate was one global kinetic sum. On the calibration states that sum
read 1.9e-4 / 0.0165 / 0.0293 while the top level's worst band read 1.3e-3
/ 4.47 / 5.53, and theta variance was never gated. On the audit's own T21
run the global KE closure read 0.0001 against a quarter-4 closure of
0.0228, a factor of 228, with a theta closure of 0.0063 that no gate saw.
The gates are now `attribution_closure_kinetic` and
`attribution_closure_theta`, worst cell over bands x sampled levels.

**Sampled levels (EN-3).** The model top (level 0), the deepest level whose
mean `p_full` lies inside the absorber (`sponge_base_pa`, 5000 Pa by
default), and the nearest levels to 250 / 500 / 850 hPa. The retired
default was the three pressure levels only; on the shipped 20-level
p_top = 100 Pa stack the sponge holds levels 0-2 (267 / 1222 / 2919 Pa)
and none of the three, so the `lid_absorber` arm read exactly 0.0 on every
production stack and the earlier "0.0104-0.0176" closure on seven T63 GDAS
checkpoints was recorded with the arm dead. Pointed at the levels where
the sponge acts, the audit measured the arm at 44 % of the total and a
closure of 0.1669 against the 0.05 gate then in force. Sampled where it
acts, the absorber is 0.99 / 1.29 / 1.42 of the top level's kinetic change
on the three families and 0.49 at the sponge base of the 20-level stack,
momentum only (its theta row is exactly zero).

Why the retired absorber arm did not close: on frozen winds the absorber
applied twice at dt/2 equals once at dt to roundoff (tested), but the
model's two halves bracket a dynamics sub-step that supplies 2.4x the top
level's net change, and at the top ring dt x rate = 0.33 (900 s
relaxation, 300 s step), so the anomaly the second half acts on is not the
one the first half saw. Evaluated once at dt on the checkpoint winds the
arm differed from the placed arm by 27 % of the arm on both T21 families
(14 % at the sponge base); the placed arm carries no such term.

**EN-2, named and bounded.** The retired hyperdiffusion arm was the
analytic factor map on the checkpoint state; the model applies the
operator after the dynamics sub-step. The audit measured that placement
error at 60 % on the worst degree and 14 % in sum|.| at T21, while it was
3.4e-4 of the residual it was lumped into. The placed arm is now the
model's own `_apply_diffusion` at the post-dynamics state (it agrees with
the analytic map there to 3.3e-12 / 7.7e-13 / 8.6e-13 relative over
draining degrees), and the departure of the checkpoint-state map from it
is carried as the row `hyperdiffusion_checkpoint_state_mismatch` with its
shares in the receipt. Measured: 2.5e-3 / 0.172 / 0.194 of the
hyperdiffusion arm (worst single degree 0.010 / 0.96 / 1.16 of the arm),
1.0e-4 / 4.9e-8 / 5.4e-8 of the step, and 0.53 / 3.0e-6 / 1.9e-6 of the
retired design's residual on the T3 smoke / T21 8-level / T21 20-level
states.

**Calibration of record (2026-09-01, `tools/energy_tendency_calibration.py`,
numpy float64).** Families: the shipped T3 smoke config (4 levels, dt 10 s,
level 0 at 978 Pa inside the sponge); T21, 8 levels, dt 300 s, 12 steps;
T21, the shipped 20-level geometry, dt 300 s, 12 steps. Worst per-band,
per-level closure with the faithful arms: kinetic 3.2e-16 / 8.5e-16 /
2.5e-15, theta 2.9e-16 / 2.1e-16 / 1.9e-16. Each retired arm planted back
in, worst kinetic cell:

| planted wrong arm | T3 smoke | T21 8-level | T21 20-level |
|---|---|---|---|
| Euler dynamics (state + dt rhs) | 0.29 | 2.46 | 3.05 |
| checkpoint-state hyperdiffusion map | 6.5e-4 | 0.124 | 0.070 |
| absorber arm dropped | 1.01 | 10.1 | 12.6 |
| absorber once at dt on checkpoint winds | 6.2e-4 | 2.99 | 3.70 |
| physics once at dt on the checkpoint | 1.8e-6 | 3.4e-3 | 2.2e-5 |

Theta variance: Euler dynamics 0.13 / 0.30 / 0.60, single-application
physics 7.9e-3 / 0.049 / 0.119. The closure limit is 1e-8 for both gates:
4e6 above the faithful worst and 180x below the weakest planted defect
(single-application physics on the dt = 10 s smoke state, where physics
is linear over the step). Reference-suite one-at-a-time interaction
residual: kinetic 3.8e-7 / 3.7e-5 / 0.012, theta 4.6e-4 / 1.2e-4 / 2.6e-6.

**Status of the earlier reading.** The retired instrument's first reading
on T63 GDAS checkpoints ("250 hPa tail: hyperdiffusion drains 2.1x the
cascade's supply", 2026-09-01) cannot be re-read here: no checkpoints of
that run are on this machine, and the CPU cannot produce them in the
run. What can be said from the calibration is that in the 250 hPa top
tenth the retired arms left 2.7 % of the band's change unexplained on both
T21 families and their hyperdiffusion arm carried a 17-19 % placement
error against the arm itself, so a 2.1x ratio measured with them is
uncertain at roughly that level and does not survive as a measurement of
this instrument until it is re-read on fresh T63 checkpoints; it is
**to be measured**. The 0.0104-0.0176 closure quoted with it was taken
with the absorber arm dead (EN-3) and is retired on the same terms.

## Sources anchoring the expectation

Nastrom & Gage 1985 (JAS 42), the aircraft spectra. Lindborg 1999 (JFM
388), the two-term fit and its constants. Skamarock 2004 (MWR 132),
effective resolution as departure from the observed spectrum, ~7 dx for
finite-difference cores. Abdalla, Isaksen, Janssen & Wedi 2013 (ECMWF
Newsletter 137), spectral IFS "effective useful resolution" (>= 50 %
variance criterion) ~3-5x grid. Skamarock et al. 2014 (JAS 71), MPAS
spectra by spherical wavenumber against the Lindborg fit, ~6 dx. Augier &
Lindborg 2013 (JAS 70), Eulerian spectral AFES holds k^-5/3 to near
truncation. Bonavita 2023 (arXiv:2309.08473), Pangu-class ML effective
resolution 500-700 km, cited beside the superseded AIFS reading of 446 km.
