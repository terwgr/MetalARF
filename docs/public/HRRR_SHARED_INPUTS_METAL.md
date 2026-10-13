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
took 577.02 s for forecast plus I/O. A subsequent corrected run also completed 49 histories and health PASS,
with forecast plus I/O 621.10 s. The original and noise trajectories remain
separate from the corrected trajectory.

Input hashes, bundle/source identities, bounded results, trajectory summaries,
and peak locations are retained in
`tools/metal_hrrr_shared_case/validation.json`.

## Numerical limits after the coefficient correction

The corrected 43,200-second forecast at commit
`3a8ca4aa9f8db044d3cf301a7d683f4b441fb395` completes 1,827 steps, 49
histories and health PASS. All 3,919 numeric variable/time checks against CUDA
are finite and matching in shapes, dtypes, units and valid times.

The correction removes QICE differences at all saved times and leaves only a
tiny QSNOW residual (max 1.61e-8 kg/kg). Pooled temperature RMSE drops from
0.11456 to 0.02750 K. Pooled U/U10/T2 RMSE falls from
0.11741/0.11734/0.08892 to 0.06726/0.07925/0.05895, but remains
24.5/46.5/15.1 times the recorded original-build perturbation spread. QCLOUD
RMSE remains 121 times that spread. One positive uniform perturbation on the
old build is not a statistical or corrected-build sensitivity ensemble.
Corrected CUDA forecast parity is not established.

The remaining low-level column still develops appreciable vapour differences
before saved liquid cloud: at (28,15,19), +15/+30 min deltas are
7.65e-5/2.95e-4 kg/kg. Radiation and surface forcing there stay close until
cloud appears at +45 min. Therefore the high-altitude cold-source defect was
not the sole trigger of the main liquid-cloud/radiation/PBL chain. Histories
alone cannot identify the first remaining within-step backend operation.

A bounded capture stopped at the first production YSU call, replaying seven
exact FP32 input columns against the existing float64 NumPy mirror. Maximum
vapour-tendency difference is 1.38e-10 kg/kg/s, PBLH differs by at most 9 mm,
and kpbl agrees. This exposes no large first-call YSU error on common inputs;
it does not establish CUDA bitwise parity or clear later thresholds, transport
and producer/input coupling. No subsequent full forecast was launched.

## Numeric-literal census

Run the source-only tool without CUDA or GPU dependencies:

```bash
python tools/audit_backend_numeric_literals.py --root . --out literal-audit
```

The census covers all `.cu`/`.cuh` files under `woof` and all Metal shader/header
files, including unused schemes and globals. It normalizes f-suffixed literals
to float32, retains possible software-FP64 words, matches named functions and
source-verified wrapper/helper mappings first, and marks heuristic/ambiguous
pairings. Missing values, multiplicities and unmatched scopes remain visible.
Its historical-source positive control catches all four original c7 errors.

The current census covers 552 files and 63,005 floating-literal occurrences,
plus 5,875 possible software-FP64 words. It pairs 1,144 Metal scopes, with
218 ambiguous pairings, and retains 2,263 unmatched Metal / 1,654 unmatched
CUDA scopes. It flags 1,937 CUDA-only values, 1,073 Metal-only values and 23
factor-of-ten candidates. Expression folds, unrelated constants, wrong helper
pairings and scheme/formulation variants create legitimate review candidates;
a flag is not a proven defect. In particular the MYNN mixing-length variants
need configuration-specific mapping/replay, not silent acceptance of their
coefficient differences. This case uses YSU.

The census/flags are complete, but semantic validation of every flagged value
and unmatched scope is not. It does not evaluate every expression or macro,
preprocess every configuration, validate operators/control flow, or establish
that every pairing represents identical physics. No further coefficient was
changed solely because of a heuristic flag. Compact receipts and all paired
flags are in `tools/metal_hrrr_shared_case/`.
