# Dual-backend first-step dumps

The configuration audit found no new active selector defect in this case. No new forecast or numerical fix followed it. The earliest histories show surface/PBL U error and a separate broad model-top V error before any saved condensate; wind and Qv already differ together at +15 minutes, so their ordering is unresolved.

Use the same fixed source and sealed inputs on both machines:

```bash
python tools/dump_backend_step1.py --list-points
WOOF_THOMPSON_TABLE_ROOT=/path/to/tables python tools/dump_backend_step1.py \
  --backend metal --experiment /path/to/prepared/experiment.toml \
  --prepared-root /path/to/prepared --wps-namelist /path/to/prepared/namelist.wps \
  --outdir /path/to/step1-metal
# On the CUDA machine, use --backend cuda and --outdir /path/to/step1-cuda.
```

The tool stops after exactly one outer step, before another step. It uses the production CLI/prepared restore rather than a constructed synthetic state; no physics selector, run-length or input override is applied. It records all **three** RK stages, including the two provisional stages and final PD stage, and all seven acoustic substeps for `time_step_sound=4`.

Before/after points:

| Phase | Exact dump boundaries |
|---|---|
| Initial state and physics | Restored state; EOS; radiation; surface layer; Noah; YSU; PBL mass/map/face coupling; composed and held physics tendencies |
| Time-t diffusion | Smag deformation, coefficient and coefficient boundary; each momentum/scalar stress call; scalar metric flux/divergence; each edge-diffusion species; diffusion flux divergence and completed held tendencies |
| Every RK stage | Coupled ru/rv and Omega; theta/U/V/W advection separately; PGF; buoyancy; geopotential; Coriolis/curvature; held physics/heating/diffusion additions; w-damping; specified/relaxation tendencies |
| Every acoustic stage/substep | Seed UV and column; implicit coefficients; external divergence filter; UV; mass/theta; W/phi; saved upper-wind limiter; boundary frame; sumflux zero/accumulate/mean; saved momentum flux refresh; UV/column fold |
| Moisture and final state | Species advection; PD flux construction; PD donor-budget renormalization; moisture stage update; boundary values and W ghosts; terrain surface W; microphysics/zero-out; completed step |

Function wrappers catch all imported aliases. CUDA raw-kernel proxies catch the acoustic closures. Metal batches are diagnostically submitted as ordered singletons to expose those same boundaries. These readbacks intentionally synchronize and change timing, without changing arithmetic or selector values. Kernel scalar parameters are recorded (CUDA positional scalars; Metal parameter ABI bytes). Common full state, relevant scratch, physics fields and function inputs appear in every chronological event; immutable content-addressed `.npy` blobs avoid writing unchanged buffers repeatedly.

Compare common state/scratch arrays at equal point, stage, substep and occurrence, then compare the operator increments on matching inputs. CUDA positional kernel arguments require the source's argument-name mapping; they are not named Metal resources. A missing point is not a zero. First locate a large increment discrepancy, then replay that operator on common inputs; different incoming states alone do not implicate its kernel.

Validated here by syntax and `--list-points` execution only. No GPU diagnostic step or forecast was run, and the CUDA execution path is not locally exercised. Keep raw captures local; no input/state data is committed to this PR.
