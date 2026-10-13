# Rainy CUDA parity root causes

The unchanged Flint rainy case exposes five confirmed Metal defects. The fixes are stacked on the complete dry-case compatibility work in PR 25. They affect staggered V dispatch, the surface/PBL gas ratio, snow riming, warm-source latent heating and the Smagorinsky mixing pipeline.

## Missing V advection points

`_launch_metal_advection` submitted `nz * ny * nx` points for V, although its target has `(nz, ny+1, nx)` points. The shader itself correctly maps a flattened index over the staggered target. For the 49 by 40 by 40 case, 78,400 threads were launched instead of 80,360. The last 1,960 outputs therefore retained their incoming tendency: all of level 48 and rows 33 through 40 of level 47. This explains the separate upper-level V anomaly in the dry comparison.

Use the actual target size for all momentum launches. U and W retain their complete target extents; V now includes every staggered row. This applies to both third and fifth vertical orders, without changing stencil arithmetic or selectors.

Three captured rainy RK-stage calls were replayed on identical FP32 field/flux/metric inputs against the existing float64 advection mirror. The comparison is the post-operator output against the incoming tendency plus reference increment, so it includes ordinary FP32 stencil cancellation and output rounding.

| Stage | Order | Before order 5 RMSE | Corrected RMSE | Corrected top-level RMSE | Repeated byte-identical |
|---|---:|---:|---:|---:|---|
| 1 | 3 | 2.25756 | 0.00149508 | 1.49769e-05 | True |
| 1 | 5 | 2.25756 | 0.00149527 | 1.49769e-05 | True |
| 2 | 3 | 2.26413 | 0.00157817 | 1.49089e-05 | True |
| 2 | 5 | 2.26413 | 0.00157849 | 1.49089e-05 | True |
| 3 | 3 | 2.19051 | 0.0015586 | 1.53438e-05 | True |
| 3 | 5 | 2.19051 | 0.00155863 | 1.53438e-05 | True |

These are coupled tendency units, not forecast wind speeds. No source inputs or configuration selectors were changed. The pre-fix large residual was confined to the exact omitted dispatch band.

## Incorrect gas ratio

SFCLAY and YSU hard-coded EP2 as 0.6217157244682312. Their CUDA authority derives EP2 from RD/RV = 287/461.6, which rounds to FP32 0.6217504143714905. The Metal coefficient was 0.0055824% too small. Derive it from each shader’s RD and RV constants instead of retaining a mismatched decimal.

On the first actual full-grid SFCLAY inputs, saturation-humidity RMSE against the float64 mirror fell from 7.559e-7 to 4.682e-9 kg/kg; surface saturation-humidity RMSE fell from 7.575e-7 to 4.642e-9. All three repetitions were byte-identical. YSU uses this ratio in its cloudy top-down mixing arm. The sampled YSU first-call vapor tendencies were already close to the reference and changed only slightly; this coefficient correction is not claimed to explain the entire rain trajectory.

## Evidence and reproduction

`v-launch-replay.json` and `ep2-replay.json` contain before/after replay metrics. `tools/rain_surface_replay.py` captures first production surface-layer, Noah and YSU calls; `tools/rain_dynamics_replay.py` captures the first step’s actual advection and Coriolis calls across all three RK stages. Both accept `--capture-root DIR` followed by the ordinary `woof go` arguments. They synchronize readbacks and stop early without altering arithmetic or selectors.

Use the prepared inputs and Thompson tables from the transferred CUDA case. The full forecast command remains:

```sh
export WOOF_THOMPSON_TABLE_ROOT="$CASE/tables"
woof go "$CASE/prepared/experiment.toml" --prepared-root "$CASE/prepared" \
  --wps-namelist "$CASE/prepared/namelist.wps" --backend metal \
  --outdir "$RESULT" --run-stamp off --products none
```

The Metal shaders, generated registry hashes and prebuilt bundle were rebuilt together. The first full rerun after the dispatch and EP2 corrections completed all 1,827 steps with finite output. It reduced pooled U/V error by 5.7%/4.5%, but rain error by only 0.45%; those repairs did not contain the rainy trajectory discrepancy. This motivated the identical-input Thompson component investigation below.


## Snow/cloud collision Stokes number

The cold-cloud Metal shader divided the collision Stokes numerator by viscosity times melted snow diameter. CUDA divides by **9 times** viscosity times diameter. The missing factor made the Stokes number nine times too large, increasing collision efficiency, cloud-water riming and its graupel conversion. Restore the denominator, preserving CUDA's widened FP32 pi and double division for the snow riming cap. The same audit corrected the H-M coefficient word from 350,017,104 to the source's 350,000,000.

Capture the actual first-call FP32 fields and scratch inputs for 56 production columns, including the strongest initial hydrometeors. Replay each component independently against the unchanged CUDA source compiled with the existing CPU host shim. This oracle uses host libm and disables floating-point contraction, so it establishes translation defects rather than exact NVIDIA arithmetic parity.

With identical cold inputs, cloud-water maximum error falls from 2.87403e-5 to 2.52021e-9 kg/kg; graupel maximum error falls from 2.68434e-5 to 1.45519e-11 kg/kg. Temperature maximum error falls from 0.00949097 K to zero. Three Metal repetitions are byte-identical. Condensation and subsequent stages already agree closely on their own identical inputs; their larger whole-adapter discrepancies were inherited from the cold stage.

## Warm-source vapor heating scratch lifetime

The split warm network saves the admitted vapor rate in slot 63. Its graupel diameter power passes reuse slot 45, which the final temperature update incorrectly read. The water species update correctly read slot 63, so vapor changed without its matching latent heating/cooling. Read the preserved slot in the final heat update. This affects snow/graupel sublimation and deposition above freezing, including columns with no entry graupel.

On the captured identical warm inputs, the pre-fix maximum temperature error is 0.00115967 K. After correction, every sampled temperature is byte-identical to the CUDA-source host result in all three repetitions. The mass fields retain their previous close agreement.

`tools/rain_microphysics_capture.py --capture-root DIR go ...` captures these per-component inputs through the public forecast entrypoint. `tools/rain_thompson_replay.py` takes `--capture-root`, `--tables`, `--output` and `--backend host|metal`; run host first, then Metal with `--reference-root` pointing to the host output. `thompson-component-replay.json` records the before/after errors and repeated-run agreement.

## Full forecast reassessment after the Thompson fixes

The second unchanged-input forecast completed all 1,827 steps and 49 matching history frames. All 3,919 numeric field/time checks are finite with matching schemas, dtypes, dimensions and units. Prepared-input/configuration identities match the transferred CUDA run.

Final rain mean improves from 3.510141 to 3.423540 mm versus CUDA 3.382034 mm: the excess falls from 3.79% to 1.23%. Final rain RMSE falls from 0.273736 to 0.136745 mm, and pooled rain RMSE from 0.216966 to 0.110635 mm. Maximum T2 error falls from 2.625 to 1.032 K. Final U/V RMSE falls to 0.00910/0.01110 m/s.

Parity is still not established. Pooled V10 RMSE remains 0.452496 m/s, Qv RMSE 5.19385e-5 kg/kg and T2 RMSE 0.272551 K. These transient gaps motivate later-call surface/PBL replay and a rainy-case perturbation control; the small dry-case perturbation spread cannot establish a rainy numerical tolerance. `forecast-iteration-2.json` retains the trajectory metrics.

## Missing Smagorinsky deformation and boundary stages

The Metal `launch_wrf_smag2d_km` branch allocated three reused deformation-buffer prefixes, then launched the coefficient consumer without launching `wrf_smag_deform`. CUDA explicitly launches deformation first, coefficients second and the physical-boundary coefficient reset third. The Metal branch also omitted that final reset.

At the captured first dry production call, every Metal deformation and K coefficient was zero. The identical-input CUDA-source host calculation has K_m maximum 401.946411 m²/s and K_h maximum 1205.839233 m²/s. On later steps the borrowed tensor storage contains carrying tendencies from the previous step, so omission can also feed stale non-deformation data into the closure. This affects the same active km_opt=4/diff_opt=2 pipeline used by the rainy case.

Submit deformation before its coefficient consumer and restore the same boundary reset as CUDA and the other admitted Metal closures. The corrected native replay fills 70,756 nonzero coefficients. K_m RMSE falls from 74.91297 to 2.99481e-6 m²/s and K_h from 224.73891 to 9.52379e-6 m²/s. Deformation maximum errors are below 4.4e-10 s⁻¹. Three native repetitions are byte-identical; all physical boundary coefficient rows are exactly zero.

The source host harness requires launcher-only renaming of its tx/ty thread loops because those names shadow pointer parameters in this module; no model source is changed by that diagnostic adaptation. `tools/rain_smag_replay.py --capture-root DIR --config CONFIG --output OUT --backend host|metal` reproduces the coefficient comparison from `tools/dump_backend_step1.py` captures; Metal additionally takes `--reference-root` for the host outputs. The host result remains a CPU CUDA-source transcription, not executed NVIDIA output.

The rainy perturbation control was stopped before completion after this deterministic defect was found. Its partial history is not used to establish a tolerance. A third unchanged-input full rainy forecast follows the repaired mixing pipeline.

## Third full forecast: mixing pipeline repaired

All 49 paired histories and 3,919 numeric checks pass finiteness/schema/time verification again, on commit 0f909c471. Final rain mean is 3.388779 mm versus CUDA 3.382034 mm (+0.1994%); maxima are 17.322386 versus 17.329737 mm. Final rain RMSE is 0.087357 mm (original 0.273736), maximum final difference 0.480037 mm and correlation 0.999598. Pooled rain RMSE is 0.068359 mm; pooled snow RMSE falls to 1.82840e-7 kg/kg (original 2.70184e-6).

Transient gaps remain: pooled T2 0.271741 K, V10 0.451910 m/s, Qv 5.14307e-5 kg/kg; maxima 1.04410 K, 2.51467 m/s and 0.000742807 kg/kg respectively. Final U/V RMSE is 0.00619/0.00855 m/s. Rainfall bias is now small, but this does not establish full trajectory parity.

The transferred CUDA certification capsule verifies that the local CUDA-source text hashes match 29 of 30 examined numerical modules. The openbc difference is confined to removal of the CFL histogram diagnostic; its damping and boundary numerical formulas are unchanged. The module authority comes from the actual run's recorded source hashes, with its release commit ea80021978f86287acb8cdaf21063076d3bf3be0 available locally.

Additional coverage: first-call YSU mass winds, theta and qv/qc/qi inputs match the initial CUDA history exactly; the first-call and three-hour YSU host replays have exact PBL height and level indices. At three hours the recorded timestep is 24 seconds (the forecast's unchanged adaptive clock), and the sampled dv/dtheta outputs match the source host byte-for-byte. The dry capture's PBL face coupling and its three RK momentum additions are also byte-identical to the FP32 source expressions. Remaining boundary/coupling diagnosis continues; no numerical-parity claim is made.

The first six rainy lateral-boundary relaxation calls (u, v, theta, geopotential, mass and qv) are byte-identical to the recorded CUDA-source host calculation on the same inputs. The remaining trajectory gap needs paired actual CUDA/Metal intermediate states to establish the first backend divergence; a one-step CUDA handoff and matching Metal capture are prepared.
