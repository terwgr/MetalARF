# Rainy CUDA parity root causes

The unchanged Flint rainy case exposes seven confirmed Metal discrepancies. The fixes are stacked on the complete dry-case compatibility work in PR 25. They affect staggered V dispatch, the surface/PBL gas ratio, snow riming, warm-source latent heating, the Smagorinsky mixing pipeline, precipitation delivery to Noah and the surface/PBL contraction policy. The trajectory effect of the last correction is assessed separately below.

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

At the captured first dry production call, every Metal deformation and K coefficient was zero. The identical-input CUDA-source host calculation has K_m maximum 401.946411 m²/s and K_h maximum 1205.839233 m²/s. The normal caller zeroes the carrying buffers before computing the time-t mixing tendencies, so the omitted deformation launch suppresses mixing. The earlier explanation about stale carrying tendencies was incorrect for this production route. This affects the same active km_opt=4/diff_opt=2 pipeline used by the rainy case.

Submit deformation before its coefficient consumer and restore the same boundary reset as CUDA and the other admitted Metal closures. The corrected native replay fills 70,756 nonzero coefficients. K_m RMSE falls from 74.91297 to 2.99481e-6 m²/s and K_h from 224.73891 to 9.52379e-6 m²/s. Deformation maximum errors are below 4.4e-10 s⁻¹. Three native repetitions are byte-identical; all physical boundary coefficient rows are exactly zero.

The source host harness requires launcher-only renaming of its tx/ty thread loops because those names shadow pointer parameters in this module; no model source is changed by that diagnostic adaptation. `tools/rain_smag_replay.py --capture-root DIR --config CONFIG --output OUT --backend host|metal` reproduces the coefficient comparison from `tools/dump_backend_step1.py` captures; Metal additionally takes `--reference-root` for the host outputs. The host result remains a CPU CUDA-source transcription, not executed NVIDIA output.

The rainy perturbation control was stopped before completion after this deterministic defect was found. Its partial history is not used to establish a tolerance. A third unchanged-input full rainy forecast follows the repaired mixing pipeline.

## Third full forecast: mixing pipeline repaired

All 49 paired histories and 3,919 numeric checks pass finiteness/schema/time verification again, on commit 0f909c471. Final rain mean is 3.388779 mm versus CUDA 3.382034 mm (+0.1994%); maxima are 17.322386 versus 17.329737 mm. Final rain RMSE is 0.087357 mm (original 0.273736), maximum final difference 0.480037 mm and correlation 0.999598. Pooled rain RMSE is 0.068359 mm; pooled snow RMSE falls to 1.82840e-7 kg/kg (original 2.70184e-6).

Transient gaps remain: pooled T2 0.271741 K, V10 0.451910 m/s, Qv 5.14307e-5 kg/kg; maxima 1.04410 K, 2.51467 m/s and 0.000742807 kg/kg respectively. Final U/V RMSE is 0.00619/0.00855 m/s. Rainfall bias is now small, but this does not establish full trajectory parity.

The transferred CUDA certification capsule verifies that the local CUDA-source text hashes match 29 of 30 examined numerical modules. The openbc difference is confined to removal of the CFL histogram diagnostic; its damping and boundary numerical formulas are unchanged. The module authority comes from the actual run's recorded source hashes, with its release commit ea80021978f86287acb8cdaf21063076d3bf3be0 available locally.

Additional coverage: first-call YSU mass winds, theta and qv/qc/qi inputs match the initial CUDA history exactly; the first-call and three-hour YSU host replays have exact PBL height and level indices. At three hours the recorded timestep is 24 seconds (the forecast's unchanged adaptive clock), and the sampled dv/dtheta outputs match the source host byte-for-byte. The dry capture's PBL face coupling and its three RK momentum additions are also byte-identical to the FP32 source expressions. Remaining boundary/coupling diagnosis continues; no numerical-parity claim is made.

The first six rainy lateral-boundary relaxation calls (u, v, theta, geopotential, mass and qv) are byte-identical to the recorded CUDA-source host calculation on the same inputs. A one-step CUDA handoff and matching Metal capture are prepared for further diagnosis when that machine is available. The skipped rain handoff below is independently reproducible locally.

## MP8 rainfall skipped at the Noah handoff

At revision 0f909c47, the no-CuPy Metal `accept_microphysics` branch validates the canonical Thompson diagnostics, but adds `RAINNCV` to `_pending_rainbl` only inside `_metal_gfs_profile`. The MP8/YSU/MM5/Noah configuration has that flag false, so the branch returns without recording any Thompson precipitation for the land surface. Its published cumulative RAINNC can still be nonzero while Noah receives no microphysics rainfall.

`MicrophysicsDiagnostics.rainncv` aliases the canonical `mp_rainncv` scratch buffer, the per-call increment in mm (kg m⁻²). `dycore.step` calls `apply_microphysics` after RK3, then calls `state.physics.accept_microphysics` exactly once with that result and the actual step dt. The driver initializes `_pending_rainbl` to zero. Each post-RK acceptance must add that increment once to `_pending_rainbl`, independently of the GFS-specific Morrison validation. At the next due `_run_noah`, the native forcing kernel adds the pending total to `fields['rainbl']`, preserving externally supplied rainfall. Noah consumes that amount (its column kernel computes precipitation rate as `rainbl / dt`). After Noah and any urban coupling return, the native clearing kernel zeros both RAINBL and the pending accumulator. The canonical microphysics increment is not cleared by this handoff.

Move accumulation outside the GFS guard. `tests/test_metal_noah_rain_handoff.py` exercises real native Metal accumulation, Noah forcing and clearing with `cp=None`, MP8 and `_metal_gfs_profile=False`; only the Noah column integration is replaced by a probe at its real entrypoint. It verifies two accepted increments accumulate, the next Noah call receives their sum plus external rainfall exactly once, both carriers clear, a second Noah call receives zero despite unchanged canonical RAINNCV, and a later new increment is delivered once. Three independent cycles use the usual batched runtime. Before the repair it fails with all six pending cells zero instead of the expected total; after the repair it passes all three cycles.

## Fourth full forecast: rainfall reaches Noah

The unchanged 12-hour run at bfbd69ecdda5d408f91bc6bd441e1684ddc9dd76 passes all 1,827 steps, 49 paired history times and 3,919 finite/schema/dimensions/dtype/units checks. All nine recorded input/configuration SHA256 identities match CUDA. `forecast-iteration-4.json` retains the measured results.

| Metric | Before rain handoff fix | After rain handoff fix |
|---|---:|---:|
| Final RAINNC RMSE (mm) | 0.0873569 | 0.00661366 |
| Pooled RAINNC RMSE (mm) | 0.0683594 | 0.00514182 |
| Pooled T2 RMSE (K) | 0.271741 | 0.0171141 |
| Pooled V10 RMSE (m/s) | 0.451910 | 0.0310900 |
| Pooled QVAPOR RMSE (kg/kg) | 5.14307e-5 | 1.47184e-6 |
| Pooled PBLH RMSE (m) | 268.352 | 31.5418 |

Metal final rainfall mean is 3.381841878 mm versus CUDA 3.382034306 mm (-0.005690%); maxima are 17.339205 versus 17.329737 mm. Final rain correlation is 0.999997662 and maximum pointwise difference 0.073783 mm. Restoring rainfall delivery explains most of the previous broad surface wind, temperature and humidity drift.

Full trajectory parity remains unestablished. Local peak differences include T2 0.605957 K, V10 1.52243 m/s, Qv 0.000203218 kg/kg and SWDOWN 84.0853 W/m². PBL height differs by 1,102.46 m at +1.5 h, (i=7,j=5): Metal 9.330 m versus CUDA 1,111.786 m. V10 peaks at +2.5 h, (i=26,j=19). These residual roots are not established, and the unavailable CUDA machine prevents a new paired intermediate-state capture at present. No tolerance or exact-parity claim is inferred from the small final errors. Pooled metrics equally weight all saved native-grid values, including initialization.

## Remaining peaks: stability feedback and contraction policy

The large PBL discrepancies already appear at +15 min in a few columns. `tools/rain_peak_capture.py` observes seven selected columns at every real SFCLAY, Noah and YSU call through +2.5 h: 384 calls per operator. Readbacks do not change inputs, selectors or arithmetic. All 890 paired history field checks from the instrumented prefix are byte-identical to the ordinary fourth run.

On these same inputs, the old Metal YSU produces exactly the same PBL height and level as the unchanged CUDA-source CPU build with contraction disabled, at all 384 calls. Direct SFCLAY V10 differences are at most 2.861e-6 m/s; Noah TSK differences are at most 6.104e-5 K. Thus the 1.10 km and 1.52 m/s trajectory peaks are not a same-input missing branch or direct wind-conversion error in these kernels.

At the +1.5 h peak, Metal's SFCLAY BR is -3.03011e-6 and its integrated PSIM correction is zero; the CUDA history's PSIM is -1.37059e-4, indicating the weak stable branch. YSU selects an unstable critical Richardson number of 0 when BR <= 0, while its stable shallow-column fallback uses 0.25. The near-zero surface-stability difference therefore chooses different diagnostic regimes: about 9 m versus 1,112 m. Surface fluxes and prior PBL mixing feed the next call, magnifying small arithmetic/state differences into intermittent height and near-surface wind differences. This identifies the amplification mechanism; it does not establish the first differing NVIDIA instruction.

The transferred CUDA certification capsule records `['-std=c++17']` for SFCLAY, Noah and YSU, with no contraction override. [NVRTC documents `--fmad=true` as the default](https://docs.nvidia.com/cuda/nvrtc/index.html). All three Metal shaders instead disabled contraction explicitly. Permit contraction inside those ordinary numerical kernels, while keeping CuPy driver operator boundaries separate and Noah's explicit RN primitive bodies uncontracted. No threshold, constant or model selector is changed.

`tools/rain_peak_host_replay.py --contract fast` builds a bounded contraction counterfactual from unchanged CUDA source; host libm and compiler choices mean it is still not NVIDIA execution. Only CUDA occupancy annotations are removed for CPU compilation, and Noah's explicit RN leaves are kept as noinline operations to prohibit fusion across those results. `tools/rain_peak_metal_replay.py` checks selected identical inputs three times. Corrected YSU PBL heights match that contracted source exactly in all six selected calls. The two largest selected dv discrepancies fall from 8.712e-6/1.269e-5 to 5.960e-8/5.309e-8 in tendency units. Noah's selected maximum TSK error falls from 6.104e-5 to 3.052e-5 K. Every native repetition is identical. `peak-surface-contraction-replay.json` preserves the full operator comparisons. A fifth unchanged-input full forecast assesses the trajectory effect rather than assuming those small operator repairs resolve every peak.

The coordinated CPU contract audit additionally exposed a regression in the rain handoff repair for admitted MP-only configurations: they have `_pending_rainbl=None` because no land-surface consumer exists. Skip accumulation only in that no-carrier case while still accepting canonical diagnostics and counting the update. The audit's focused reproduction fails before this guard and passes after it; its MP8 and MP10 delivery controls also pass (3 passed total). The native handoff regression now includes MP-only acceptance across its three cycles. The separate SWINT1 admission defect is inactive here: the transferred experiment resolves `swint_opt=0`.

## Fifth forecast: contraction is insufficient to contain the peaks

The unchanged run at 26b9582dd7f62eec4c344d7c2e88ce8cb0957bbc passes all 49 paired histories and 3,919 numeric field/time checks, with matching recorded input/configuration identities. Final rain RMSE improves to 0.00533919 mm (previous 0.00661366), pooled rain RMSE to 0.00418533 mm and pooled V10 to 0.0287591 m/s (previous 0.0310900). However, pooled PBL height is essentially unchanged at 31.7740 m (previous 31.5418). Its peak remains 1,087.68 m at the same +1.5 h (i=7,j=5) column. The original V10 peak decreases to about 0.846 m/s, but another column has a larger 1.72180 m/s error at +2.5 h (i=24,j=33). T2 pooled RMSE is 0.0176955 K. `forecast-iteration-5.json` retains these results. The source-policy correction is not claimed to resolve the remaining trajectory problem.

## Pressure-power accuracy and the near-neutral stability cliff

An unchanged CUDA-source one-column probe using the actual +1.5 h input reproduces the height jump when only BR changes sign. BR=-3.03011e-6 gives hpbl=9.33035 m, kpbl=1 and dv at the lowest level +0.00281916 m/s/s. BR=+3.03011e-6 gives hpbl=1,111.49963 m, kpbl=16 and dv=-0.00345414 m/s/s. The observed CUDA height is 1,111.786 m. This demonstrates the threshold amplification mechanism and its momentum feedback without modifying any source threshold or claiming to reproduce a complete NVIDIA trajectory.

The two existing Metal Exner paths are mutually consistent, but both have a systematic native-power low bias against the rounded mathematical value. On all 78,400 actual initial EOS pressures, 57,857 Exner words differ: 51,179 are one ULP low and 6,678 are two ULPs low. A temporary GPU primitive probe, compiled with the production safe/precise flags, separates normalization from power: runtime division, constant division and software RN division are identical to the FP32 reference on all 9,440 sampled pressures. Native and `precise::pow` still have the same low bias. The shared software binary64 power, rounded back to FP32, has no differing forward or inverse result on those samples, across three native repetitions. Thus division was excluded from this repair.

Use `pressure_power.metalh` for the sensitive pressure-to-temperature powers in physics preparation, microphysics preparation, SFCLAY, Noah, YSU cloudy-top temperature and Noah SFCDIAGS. The normalization and returned field remain FP32; only the transcendental evaluation uses the existing GPU software binary64 implementation. No host numerical fallback, physical constant, threshold, configuration or new floating-point field is introduced. This corrects mathematical accuracy; it is not a claim of bit-identical NVIDIA powf implementation.

The production forward Exner replay now has zero differing words against the rounded reference on all 78,400 inputs, with physics and microphysics still byte-identical in three runs. `tools/rain_pressure_power_replay.py` reproduces the full forward check and sampled forward/inverse comparison through the real production helper. It uses an unused context-required calc_coefs placeholder only in the temporary primitive library, never as a forecast operator. The selected real SFCLAY HFX mean RMSE against the contracted CPU-source replay falls from 0.00186646 to 0.000160109 W/m²; Noah HFX from 0.000762654 to 0.000158722 W/m². All three native repetitions agree. `pressure-power-replay.json` retains the measured accuracy and source-operator evidence. A sixth unchanged-input full run assesses whether this thermodynamic correction contains the near-neutral trajectory discrepancies.
