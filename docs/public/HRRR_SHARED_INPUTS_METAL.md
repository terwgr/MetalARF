# Metal HRRR/Noah shared-input execution

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

The shaders, typed ABI, generated registry and rebuilt default/exact libraries
travel together. No CPU forecast fallback or memory-guard override is used.

## Verification

After rebuilding the bundle, the bounded checks are reproducible with:

```bash
python tools/metal_diff6_edge_experiment.py --out diff6-edge-check.json
python tools/metal_upper_wind_experiment.py --out upper-wind-check.json
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
not a controlled benchmark. There was no additional full forecast for the
subsequent diagnostic investigation.

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

The saved histories show differing cloud formation and radiative fluxes early
in the run, followed by an offset in evening surface/PBL transitions. For
example, at an interior column (i=31,j=27, zero-based), PBLH is 12.3 m on Metal
and 2,172.6 m on CUDA at +6.5 h; CUDA's collapse follows in the next frame.
This is a diagnostic lead, not proof of a remaining kernel defect. Initial
perturbation growth, microphysics thresholds and radiation sampling must be
separated by identical-input component replay before attributing the remaining
trajectory differences to an implementation error.

## Reusing a transfer bundle

Treat its experiment, namelists, prepared bytes and Thompson tables as sealed.
With an installed environment from this source:

```bash
export WOOF_THOMPSON_TABLE_ROOT="$CASE/tables"
woof go "$CASE/prepared/experiment.toml" --prepared-root "$CASE/prepared" \
  --wps-namelist "$CASE/prepared/namelist.wps" --backend metal \
  --outdir "$OUTPUT" --run-stamp off --products none
woof render --diff "$OUTPUT/chain/run/wrfout" "$CASE/cuda/chain/run/wrfout" \
  --products t2,wind10,precip --out "$DIFF" --run-stamp off
```

The two `wrfout` directories avoid discovery of `ready/wrfout*.json` receipt
files as histories. No change to forecast inputs is needed.
