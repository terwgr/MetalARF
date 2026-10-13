# Rainy CUDA parity root causes

The unchanged Flint rainy case exposes two confirmed Metal defects. The fixes are stacked on the complete dry-case compatibility work in PR 25. The larger defect is an incomplete dispatch of V momentum advection. The second is an incorrect surface/PBL gas ratio.

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

The Metal shaders, generated registry hashes and prebuilt bundle were rebuilt together. A full rainy rerun and trajectory reassessment follow the confirmed operator repairs.

