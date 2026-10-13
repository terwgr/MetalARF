# Metal HRRR/Noah shared-input execution and Thompson coefficients

The transferred CUDA consistency case selects Thompson mp8, YSU, MM5 surface
layer, Noah, RTE-RRTMGP, `alb_sol=1`, `diff_6th_form="noaa_wrf39"`, and the
NOAA upper-wind limiter with `damp_opt=3`. It is a 40 × 40 × 49, 3 km domain
forced by HRRR from 2024-07-10 18Z through 2024-07-11 06Z.

On release 1.0.3 plus Coriolis PR #23, the public Metal route accepted this
configuration but failed before advancing its first model step. The repairs
preserve the prepared input bytes and every experiment selector:

- Allocate the persistent sun-angle albedo pair for Noah as well as RUC,
  using the same initial copies and MODIS guard as the shared CUDA initializer.
- Run HRRR edge diffusion through typed device padding and crop/add kernels.
  Reuse the preflight-priced workspace, low halo of three cells, high halo of
  four, original staggerings, and existing diffusion kernel arithmetic.
- Correct a preexisting one-cell shift in diffusion's cell/corner mass-face
  averages. Donor-center fluxes retain their original positions.
- Port the descending saved-wind limiter and place it after W/PHI advancement
  and before boundary-frame handling in the immutable acoustic topology.
- Refresh the saved horizontal mass fluxes with existing Metal coupling kernels
  after limiting; do not recompute saved Omega or the acoustic mean fluxes.

- Correct four classic/aerosol cold/warm liquid-saturation helpers whose seventh-order coefficient was transcribed ten times too large. In the classic cold source, the wrong coefficient opened the ice-nucleation gate in clear air.

The shaders, typed ABI, generated registry and rebuilt default/exact libraries
travel together. No CPU forecast fallback or memory-guard override is used.

## Verification

After rebuilding the bundle, the bounded checks are reproducible with:

```bash
python tools/metal_diff6_edge_experiment.py --out diff6-edge-check.json
python tools/metal_upper_wind_experiment.py --out upper-wind-check.json
python tools/metal_thompson_clear_air_experiment.py --table-root /path/to/tables --out thompson-clear-air-check.json
```

The diffusion check covers mass, U, V and W with/without the terrain taper,
compared with the existing float64 NumPy mirror on edge-padded inputs
(rtol 3e-6, atol 5e-5). Each of eight cases repeats three times with identical
device outputs. The limiter's 12 existing compiled Fortran fixtures repeat
three times; U/V match the fixture words exactly.

One public `woof go --backend metal` run completed 43,200 model seconds,
1,827 outer steps and 49 histories using the sealed CUDA inputs. Both final
health checks passed. The paired histories share valid times, grids, shapes,
units and inventories; all 3,919 numeric field/time comparisons are finite.
Forecast plus I/O took 597.22 s on an M2 8-core GPU / 16 GB, versus the CUDA
reference's 119.64 s on RTX 2000 Ada. These are observed machine timings,
not a controlled benchmark. A later +1e-6 K initial-theta sensitivity run used
the same original build, completed all 49 histories and passed health. It
took 577.02 s for forecast plus I/O. No full forecast has used the coefficient
correction; both existing Metal trajectories retain the original defect.

Input hashes, bundle/source identities, bounded results, trajectory summaries,
and peak locations are retained in
`tools/metal_hrrr_shared_case/validation.json`.

## Numerical limits

This repair establishes execution of the case; it does not establish exact
CUDA or general WRF parity. At +12 h, T2 RMSE is 0.0915 K and U/V RMSE is
0.1013/0.1260 m/s. Across all output times, local maxima reach 1.1845 K for T2
and 2.7219/2.3707 m/s for U/V. Both rain totals remain zero.

Those large maxima are interior, not boundary-frame outliers. Removing five
native rows/faces on every side leaves the listed maxima unchanged. The main
wind and T2 peaks sit 7–9 cells in; pressure's peak is 15 cells in. Archived
top-three-level speeds remain below 16.9 m/s, versus the limiter's 110 m/s
activation threshold; 15-minute histories cannot prove every intermediate
substep's activation status.

The +1e-6 K sensitivity run stays close to baseline Metal at the main error
columns. Across all 49 times, pooled U/U10 RMSE is 0.00275/0.00170 m/s,
versus Metal–CUDA 0.1174/0.1173 m/s; T2 RMSE is 0.00389 K versus 0.0889 K.
Individual sensitivity maxima can still be large (PSFC 9.20 Pa, PBLH 1,134 m),
so one perturbation is not a statistical noise envelope or parity criterion.

At +15 minutes, Metal has cloud ice and snow while CUDA has neither; the
liquid cloud appears later, at +45 minutes. Zero surface rain does not imply
only condensation is active. Identical-input replay found a concrete upstream
error: CUDA's RSLF c7 is `0.379534310e-11f`, while four Metal helpers expanded
it to `0.0000000000379534310f`, ten times too large. The wrong saturation
opens the classic cold-source nucleation gate in clear air. On 143 selected
initial clear-air states, a single original kernel launch changes temperature
and vapour (max 0.523 K / 1.85e-4 kg/kg); the correct CUDA predicate returns
without changing them. An ice-supersaturated control also makes spurious ice.
After fixing all four copies, all 146 replay cases keep their physical input
words exactly; the expected snow-velocity boost reset is excluded.

This verifies the classic cold-source clear-air fix, not all aerosol or
warm-source regimes. Fifteen-minute histories cannot identify every first
within-step operation or establish that this defect accounts for every later
peak. The corrected forecast has not been integrated, so CUDA forecast parity
is still unmeasured. The initial P/TH2 ULP differences remain an earlier
observed diagnostic rounding difference. Both original full trajectories and
the sensitivity experiment are preserved as evidence of the pre-fix behavior.
