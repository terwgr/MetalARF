# Bounded MetalARF handoff audit

Baseline: `bfbd69ecdda5d408f91bc6bd441e1684ddc9dd76` (immutable detached checkout).
Source: exact-SHA checkout from `https://github.com/terwgr/MetalARF.git`.
The audit was isolated from active working trees and running forecasts.
Publication base: PR26 branch `fix/rain-cuda-parity`, verified open at
`15faa65b4ea87f210c05a9b45bb1313bf7fa86b9` on publication preparation.
Its additional commit contains reports only; the numerical/source audit remains
pinned to the baseline above.

Read baseline AGENTS.md; no AGENTS.local.md, .agents, or .codex at this revision.
Repository instructions prefer NumPy/algorithm contracts to CUDA parity;
user's CPU-only restriction supersedes suggested native smoke/CLI forecasts.
No GPU, shader build, CUDA execution, physics edit, service installation,
or global environment change was performed. The only authorized publication
is this test/document draft PR; no merge is included.

## Confirmed defects

1. **MP-only no-CuPy acceptance fails (control boundary, outside actual Noah case).**
   Baseline above; `woof/core/physics.py`, `_initialize_metal_sfclay_only:2764-2769`
   and `accept_microphysics:3365-3369`. MP8, CU0, surface0, PBL0, sfclay0 is
   admitted and initialized by the real public initialization function. It
   allocates `_pending_rainbl=None`, then acceptance unconditionally calls
   `add_array(RAINNCV,None)` and raises `NotImplementedError: Metal Noah rain
   accumulation has no supported FP32 layout`. Public initialize_physics
   docstring explicitly promises an MP-only driver for canonical diagnostics.
   Minimal reproduction: `test_mp_only_no_surface_should_accept`, canonical
   FP32 RAINNCV=.25 mm. Expected: accept diagnostics/increment update counter
   without a surface consumer. Observed: exception, counter not incremented.
   Smallest proposed repair: guard only pending accumulation when no pending
   carrier exists; retain canonical validation and successful acceptance count.
   Do not claim this breaks the current surface2 Noah configuration.

2. **Requested SWINT1 is admitted and silently bypassed in no-CuPy initialization.**
   Baseline above; `woof/core/physics.py`, `initialize_physics:8544-8565`,
   `_initialize_metal_sfclay_without_cupy:8196-8206` and constructor `:2841`.
   Shared allocation `:9059-9076`/attachment `:9221-9243` are unreachable through
   that early return. Runtime `_run_radiation:3670`/`compute:7042` guards on
   `self.swint is not None`, so both fit and between-call evaluation disappear.
   `_metal_admission_refusals:928-965`/RRTMGP predicates admit this selection;
   config's SWINT validator `woof/config.py:2327-2387` permits 1 with SW4.
   Minimal reproduction: real initialize_physics with MP8/CU0/YSU1/MM5 sfclay1/
   Noah2/LW4/SW4/RTE-RRTMGP and swint_opt=1, supplied geometry and supported
   adapter properties; `test_initialized_radiation_carriers_and_swint[1]`.
   Expected: interpolation attached with its persistent fields. Observed:
   `driver.swint is None`. SWINT0 control passes. Smallest proposed repair:
   share SWINT state allocation/attachment across both initializer doors (or
   explicitly reject 1 pending support, rather than silently ignore it).
   This is a selector-honoring defect, not executed numerical evidence or a
   demonstrated cause of a forecast discrepancy. The defect does not apply to SWINT0.

## Five contracts and traces

| Family | Producer → consumer; units/storage/authority | Predicates, cadence and reset | Execution boundary |
|---|---|---|---|
| Precipitation | post-RK microphysics → `accept_microphysics` → `_run_noah`; RAINNCV and pending/RAINBL in mm, SR dimensionless. MP diagnostics alias `state.scratch` through `microphysics_scratch_slots`; pending separately owned. Noah receives canonical current scheme SR, not a snowfall-weighted pending SR. Authority: diagnostics contract, Noah forcing docstring, WRF Noah caller comments. | `dycore:4658` accepts after RK; `physics:3265-3373` no-CuPy ownership/layout validation; MP10/GFS optional validator then common accumulation. Noah `:5119-5230` uses scheme-SR membership, adds external RAINBL + pending before launch and clears both after consumer/urban. `compute:7052-7215` uses mandatory step1 and modulo STEPBL; skipped surface calls retain pending. CU0 bypasses cumulus; CU1 clock adds its own rain before later MP. | Real accept and Noah dispatch, exact canonical aliases, 2 increments, skipped/due scheduling predicate, two consumption calls, SR=.75, external=.125, MP8 and MP10 legacy GFS guard; Morrison validator and Noah forcing/kernel are numerical probes. Native existing test is retained, not executed. |
| Moisture/heating | classic Thompson adapter `_apply_thompson:618-881` prepares temperature in K and qv kg/kg; cold/warm source kernels mutate both; final adapter converts T/Exner to theta and h_diabatic K/s with configured limiter. Warm launcher `thompson_warm_frozen_source.py:302-374`: melt → vapor → limits → apply → mode3 graupel power → finalize, same `cloud_state` allocation throughout. Shader limits `:1200` holds vapor mass rate in slot63; apply `:1231` and finalize `:1317` read63. Vapor volumetric mass rate becomes qv rate using inverse density and dt; heat uses latent heat, moist cp, inverse density and dt. | MP8 warm entry mask controls all warm passes; cold kernel has its own in-place coupled heat/moisture update. Slots45 (rate/power reuse), 59-62 (fusion rates), 63 (held vapor) are private per-call FP64 bit storage. Slot63 is written by limits before consumption, untouched by intervening mode3 powers, then allocation dies after final submit. Entry-mask early returns do not require a skipped cell's prior scratch to be valid. No shared driver vapor tendency is mistaken for K/s. | Real typed launcher and actual table-owner/type/layout validation, 2 independent calls, fixed known slot63 markers, record all submissions, static actual shader read/write check. Numerical shaders are recording probes: neither rates nor temperature values are computed/validated. Top-level full Thompson adapter is source-inspected, execution-unverified; leaf probes cannot establish physics parity. |
| Radiation/surface | initialize supplied SWDOWN/GLW W/m² into host fields then same-context surface storage (`physics:7813-7814`). `compute:7012-7049` runs due radiation first; `_run_radiation:3590-3749` validates/copies result, couples held K/s heating, updates persistent flux destinations and stamps producer age/provenance. Carrier contract is authoritative before LSM consumption (`:7167-7190`). | SW/LW4 RTE adapter admission checked; first step mandatory then modulo STEPRA offset1. SWINT0 holds last radiative flux between due calls, including held daylight until next nighttime call. SWINT1 should fit/evaluate at each current sun but is bypassed (defect above). External forcing `set_forcing:7340+` has own provenance. GLW-off declared forcing is not falsely relabeled radiation-made. | Real full initialize, direct real Noah consumer of supplied 400/250 forcing, SWINT0/1 selection, real compute/_run_radiation at 0/15/30/45 s with synthetic daylight/night numerical adapter and skipped steps. Numeric coupling and atmosphere preparation are numerical probes. Sun geometry/interpolation algebra, actual RRTMGP and due surface/PBL integration are not executed. |
| Turbulence | `dycore.step:4442` prepare once, stage loop `:4495` applies held dry tendencies; `_prepare_fixed_tendencies_impl:2390-2485` zeroes carrying buffers before producers. `_compute_wrf_smag_tendencies:2237-2342` → real `launch_wrf_smag2d_km:1313-1357` deformation → K → coefficient BC → u/v temporary stresses → boundary strips → copy back → w/scalars. K m²/s; tensor strain s⁻¹; held dry tendencies mass-coupled per second per `_smag2d_specs`/map-factor authority. | km4 diff_opt2 with YSU1 (vertical stress addition skipped under active PBL); km1/diff6-off is no-mixing control. D11/D22/D12 borrow prefixes of smag_rw/ru/rv until both u/v readers complete, then those buffers become final tendencies. Prepare clears again each outer step; dry fixed tendencies are reused over 3 stages. `prepare_fixed_tendencies:2380` defers submissions around actual implementation; strip reset precedes application. | Real prepare, specs, prefix views, K launcher, compute ordering, strip dispatch and dry application; numerical deformation/K/HD/BC/strip leaves stubbed. Two outer steps and three applications each, known tensors/K/held values and physical open-strip zero expectations. No numerical strain, stress, map-factor, vertical/PBL mixing or whole step executed. |
| Restart | `io/restart._driver_manifest:3405-3510` persists pending, held radiation rates, surface numeric fields, tendencies, and requires MP canonical scratch aliases. Canonical MP cumulative/current/SR scratch persisted by carried-scratch manifest; `_restore_driver:6828-7006` restores fields/pending in place, reconstructs diagnostics against scratch, restores counters and carrier provenance/age with `_restore_carriers:1686`. | Preserve pending unconsumed mm + external RAINBL separately to avoid double counting. Preserve radiative carriers and age across skipped calls. SWINT coefficients would live in fields while interpolation object is rebuilt; that rebuild is currently missing on no-CuPy door. Private Thompson source work, deformation/K/held dycore work are rebuilt according to restart/preflight lifetimes at the next outer-step prepare. | Real driver manifest → CPU stored copies → actual driver restore; fixture preloads canonical scratch as outer restore normally does. Noah consumes restored .875 mm once then zero; MP aliases and carrier metadata preserved; missing pending member rejected. No checkpoint-file I/O, outer validation, full-state/time activation, exact forecast equivalence or restored SWINT1 execution claimed. |

All readers/writers searched for pending/RAINBL/SR, surface category carriers,
SWDOWN/GLW/SWINT, smag buffers, and warm cloud-state slot63. Alternate paths
examined include no-CuPy early returns, GFS validation guard, Noah mosaic,
RUC/Noah-MP, cumulus pending additions, externally set forcing, PBL-off vertical
mixing, diff_opt1, km2/3 and diff6 branches, shared CUDA initialization and
restart restore. Those alternate numerical branches were not blanket-tested.

## Already fixed / remaining candidates

- Rain repair at this commit visibly moves common accumulation outside GFS guard.
  CPU MP8/MP10 probes pass. Existing `tests/test_metal_noah_rain_handoff.py` is
  GPU-marked, imports MetalContext, and covers native MP8 multi-call delivery,
  clearing and repeat cycles; it does not cover MP-only admission, CPU-only
  fallback, actual scheduled compute, MP10 guard, SR assertions or restart.
- Smag deformation is visibly produced before coefficients at this baseline,
  and the real outer caller clears borrowed buffers each step. The supported
  earlier mechanism is suppressed mixing if the producer is absent; the tests
  do not demonstrate stale prior-step tensors in the normal caller.
- Warm vapor scratch protection is present (limits writes63; apply/finalize
  read63, powers reuse45). No missing producer or early-reset defect found in
  this seam at the frozen revision; shader numerics remain unexecuted.
- Out-of-selected-case source candidates only: no-CuPy return before
  `physics:3480-3489` categorical surface_rainncv/snowncv/graupelncv/hailncv
  updates; RUC Metal branch `:5251-5285` does not copy canonical SR whereas
  ordinary branch `:5286` does, and `ruc_metal_runtime:128-129` binds surface SR
  to frzfrac. Aggregate precipitation may reconstruct categories. Neither is
  labeled total precipitation loss or a current Noah-case defect. Future
  focused RUC/Noah-MP category/SR fixture needed; no full port audit conducted.

## Verification and exact commands

Environment: Python 3.14.7, NumPy 2.5.3, pytest 9.1.1 from an existing
local environment. The system interpreter lacked NumPy, another available
environment lacked pytest; no dependency installation was needed or performed.
Sandbox DNS initially blocked exact-SHA access; task-local fetch succeeded.
No remaining source-access blocker.

Run from the frozen checkout:

```sh
PYTHONPATH=. GPUWM_NO_LOCAL_GPU=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -u -c \
'from woof.core import physics, dycore; import pytest; raise SystemExit(pytest.main(["tests/test_metal_handoff_cpu_audit.py", "-q", "--tb=short"]))'
```

Preimports select this checkout via PYTHONPATH; pytest's tree identity, GPU,
no-silent-skip and no-silent-deselection gates remain active. New tests force
physics/dycore/microphysics cp=None; real MetalContext construction, allocation
and launches and unexpected dycore CUDA lookup fail immediately. Host storage
and explicit numerical kernel probes are the only replacements. No production
source copied into rewritten drivers. Assertions use exact fixed FP32 values;
no tunable tolerances. Expected physical numerical results are not inferred
from these synthetic kernel outputs.

Final result: **8 passed, 2 failed, 0 skipped, 2 warnings; exit status 1**,
160.09 seconds. The failures are exactly
`test_mp_only_no_surface_should_accept` and
`test_initialized_radiation_carriers_and_swint[1]`, reproducing the two defects
above. No fixture/infrastructure failures remain. The warnings are existing
invalid-escape SyntaxWarnings from source AST inspection. The 8 passes include
MP8/MP10 rain, SWINT0 initialization/Noah forcing, km4/km1 two-step ordering,
warm-source pass/scratch trace, daylight/night radiation scheduling, and restart
round-trip. All three reversible mutation detections passed; missing restart
pending member was rejected as expected.

Mutation checks are reversible monkeypatches inside the tests: missing rain
addition breaks the pending=.5 assertion; missing Smag deformation breaks the
fixed tensors before coefficient use; early warm slot63 reset breaks final
marker delivery. Restart missing pending member fails closed. No production or
shared source mutation and no gate weakening. These establish fixture sensitivity,
not numerical correctness of substituted kernels. Earlier development runs
were interrupted for slow AST/import inspection or showed fixture-only errors;
only the final frozen-patch run is the delivery count. Direct fixture diagnostics
were supplementary, not reported as pytest suite passes.

## Deliverables and stopping boundary

Modified tracked production files: **none**.
Added PR files: `tests/test_metal_handoff_cpu_audit.py`,
`tools/handoff_audit/REPORT.md`. Test output is quoted above; raw local logs and
review patch are supporting workspace artifacts and are not committed.

Priorities: (1) separately repair or explicitly refuse SWINT1 and verify actual
interpolation with native/reference geometry; (2) separately guard MP-only
pending accumulation; (3) run native numerical/reference checks on these seams
when device access is available, preserving current forecast; (4) later test
RUC/Noah-MP categorical/SR candidates and outer restart timing/file validation.
Do not infer whole-forecast or CUDA parity from this CPU seam audit. Stop here;
no sensitivity ensembles, sixth interface family, YSU threshold diagnosis or
production fixes included.
