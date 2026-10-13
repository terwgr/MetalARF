# Metal HRRR/Noah execution, Thompson coefficients and scalar order

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
and producer/input coupling. The next bounded investigation and its single corrected forecast are recorded below.

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

All 23 factor-of-ten flags were subsequently reviewed against their actual expressions: 0 confirmed defects, 23 dismissed, 0 unresolved factor-of-ten pairings. Each dismissal is recorded in `tools/metal_hrrr_shared_case/LITERAL_FACTOR_REVIEW.md`. One-sided values and unmatched scopes were not reviewed. The census does not establish semantic equivalence of every flagged value or unmatched scope. It does not evaluate every expression or macro,
preprocess every configuration, validate operators/control flow, or establish
that every pairing represents identical physics. No further coefficient was
changed solely because of a heuristic flag. Compact receipts and all paired
flags are in `tools/metal_hrrr_shared_case/`.

## First remaining Qv operator defect: vertical order

The +15-minute Qv error occupies interior patches from the surface through the PBL top, strongest around levels 18–19. The trim5 core holds 88.2% of squared error in 56.25% of horizontal cells; core/rim RMSE is 1.03149e-5/4.27061e-6 kg/kg. Gradient correlations are modest and terrain correlation weak; the map suggests transport rather than a narrow edge strip but does not uniquely identify an operator.

This case requests `v_sca_adv_order=5`. In `_advance_scalars_stage_metal`, both provisional unlimited and final PD calls omitted the selector and used the launcher default 3. The shared PD dispatcher also dropped the selector, and the Metal PD shader lacked the fifth-order path. The fix forwards the configured order through every layer, adds the CUDA vertical stencil/degradation ladder, and adds the order5 semi-Lagrangian low flux for signed Courant magnitudes above one. The typed ABI and rebuilt bundle are included; the donor-budget limiter remains unchanged.

The first actual Qv advection call was captured before completing step 1 on unchanged prepared inputs. On identical full-domain inputs, original production versus the requested float64 order5 mirror has tendency RMSE 9.22465e-4 and maximum 1.59007e-2. Production versus the order3 mirror instead has RMSE 6.83837e-8. After repair, production actually passes order5 and RMSE is 6.76095e-8, maximum 1.01605e-6. The wrong-order error is roughly 13,600 times this float32 scale. These are dry-mass-coupled tendencies, not kg/kg increments.

Eight bounded checks cover unlimited and PD/limiter at orders3/5 on the captured domain, plus signed vertical Courant controls ±2.25/±3.75, against existing float64 mirrors. All pass epsilon-scaled bounds. Reproduce with `python tools/metal_scalar_order_experiment.py --inputs scalar-step1-inputs.npz --out scalar-order-verification.json` using the retained local capture. No broader test suite was run. The operator search stopped on this defect; scalar diffusion and Noah flux were not replayed afterward.

Exactly one subsequent full 12-hour forecast ran at `cfca154b06b9283fe8769a9ec9dd09e3954def4e`: 1827 steps, 49 histories, health PASS, and 3,919 finite matching numeric field/time checks. Input identity equals the previous run. Forecast plus I/O took 878.36 s. No new noise-floor forecast ran.

Pooled RMSE over all 49 times and all grid cells:

| Field | Previous c7-corrected | Scalar-order corrected | Recorded original-build perturbation |
|---|---:|---:|---:|
| PSFC | 0.441809 | 0.448684 | 0.34045 |
| U | 0.0672561 | 0.0678365 | 0.00274668 |
| V | 0.0845924 | 0.0847601 | 0.00298216 |
| U10 | 0.0792542 | 0.0802005 | 0.00170286 |
| V10 | 0.0895839 | 0.0892339 | 0.00188512 |
| T2 | 0.0589461 | 0.0572595 | 0.00389278 |
| T | 0.0274978 | 0.0273408 | 0.00375904 |
| W | 0.0152933 | 0.015462 | 0.00206457 |
| QVAPOR | 4.15226e-05 | 3.9791e-05 | 1.00618e-06 |
| QCLOUD | 4.4719e-07 | 6.02648e-07 | 3.68883e-09 |
| QICE | 0 | 0 | 3.89586e-08 |
| QSNOW | 1.55741e-11 | 8.63799e-12 | 2.05244e-08 |
| PBLH | 41.8677 | 42.2362 | 5.20838 |
| SWDOWN | 6.31529 | 7.22196 | 0.635553 |
| TSK | 0.0935666 | 0.101527 | 0.0108409 |

Units are Pa for PSFC, m/s for winds, K for temperatures, kg/kg for mixing ratios, m for PBLH and W/m² for SWDOWN. T is stored potential-temperature perturbation. The remaining forecast errors still substantially exceed the recorded perturbation spread; parity is not established. The early-cloud chain persists, and some fields worsen after restoring the requested order. A confirmed selector defect is therefore fixed without claiming that it accounts for all remaining Qv drift. The search stops here as requested. Full comparison and bounded receipts are in `tools/metal_hrrr_shared_case/scalar_order_fix.json`; the local report retains per-level maps, peak locations and masked metrics.


## Subsequent wind/config investigation (no forecast)

A source-reviewed audit enumerates every option in the sealed experiment and namelist plus the omitted horizontal momentum-order default: 226 rows, 220 honored (including preparation, pinned and inactive controls) and 6 stock-WRF I/O declarations ignored on both backends. No additional active Metal selector defect was confirmed. See `tools/metal_hrrr_shared_case/CONFIG_HONORING_AUDIT.md` for the full value/path/honored table and JSON for resolved runtime values. This is configuration-flow evidence, not numerical kernel equivalence.

At +15 minutes U/V RMS is .00905/.02367 m/s. U error is concentrated in the surface/PBL (77.1% of squared error in levels0–19); V is dominated by upper levels35–48 (84.9%). V's maximum .47461 m/s is at native staggered (3,16,48) in the relaxation zone, but its broad model-top error survives trim5: k48 RMSE .15022→.14262 m/s. U's maximum .16217 is at (15,15,0), interior. The divergence-proxy correlations are modest and do not identify an operator.

U/V/Qv are initially identical, then all differ in the first +15-minute history while cloud, ice and snow remain zero in both. Thus winds precede saved condensate differences; the cadence cannot order wind versus Qv. The upper V structure cannot be explained solely by the later low-level cloud/PBL chain. Its first within-step source remains unresolved.

No forecast or numerical edit followed this audit. `tools/dump_backend_step1.py` supplies exact before/after points on either backend, all three RK stages and all seven acoustic substeps, then stops before a second outer step. Invocation, boundary list and validation limits are in `tools/metal_hrrr_shared_case/STEP1_DUMP_POINTS.md`. Only syntax/help/list-point execution was checked; the script has not been GPU-executed on either backend. The requested investigation stops here.

## Actual first-step capture: V-advection extent

The next task ran one actual Metal15s outer step, all3RK stages and7acoustic substeps, retaining468before/after snapshots. Seven initial prognostic fields, expressed in history conventions (T=thp+thb−300), match the CUDA t0 history byte-for-byte. The CUDA machine was inaccessible by user confirmation, so no executed CUDA dump or globally first-divergence claim is made.

An identical-input replay found that `_launch_metal_advection` requested78,400V threads instead of80,360: it used mass-grid ny40, while the V shader decodes41staggered rows. The entire top V layer and part of the preceding layer received no advection increment. CUDA's launcher uses ny+1. The fix uses `q.size`; U/W extents are preserved and no shader/ABI rebuild is needed.

For the captured order5 call, float64-reference RMS/max error falls from2.7354/79.4507 to0.00018238/0.002453 in coupled-tendency units, approximately15,000×RMS reduction. All1,560eligible top-level V points now receive their increments. Order3 also passes the bounded reference replay. The older diff6 mirror needed the directional map multiplier already present in both backends; it was corrected only in the local replay adapter, not the model.

One model defect was fixed and verified, then the investigation stopped. No full forecast ran, so the preceding forecast table is unchanged and the forecast impact remains unmeasured. Reproduce with `tools/metal_v_advection_extent_experiment.py` and the retained local FP32 capture. Compact evidence is in `tools/metal_hrrr_shared_case/step1_v_extent.json`. The dump wrapper also now records scalar/config metadata and accepts its intentional CLI stop only with a valid completion receipt and final event.

The rain investigation independently found and committed the same launch correction in [PR26](https://github.com/terwgr/MetalARF/pull/26), stacked on this PR. Numerical-fix ownership stays there; this PR adds only the independent dry replay tool/receipt and dump-wrapper corrections. The temporary local launch edit used for the passing dry replay is not duplicated in this PR.
