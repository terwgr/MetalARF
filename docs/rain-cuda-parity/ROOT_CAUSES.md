# Rainy CUDA parity root causes

The unchanged Flint rainy case exposes four confirmed Metal defects. The fixes are stacked on the complete dry-case compatibility work in PR 25. They affect staggered V dispatch, the surface/PBL gas ratio, snow riming and warm-source latent heating.

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
