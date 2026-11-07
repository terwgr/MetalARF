# Metal coverage

MetalARF runs WOOF 1.1.5 on Apple Silicon with a prebuilt Metal bundle.
Coverage is configuration-specific; admission checks reject unsupported combinations.
Selectable arithmetic policies remain available; full WRF-exact forecasts are refused.

## Regional

- Public commands: prepared `woof go`, WRF input, simulation, streaming, downscaling, warm starts, fixed/moving nests, ensembles and restart continuation.
- Dynamics: staggered-grid RK/acoustic steps, advection, positive-definite moisture, Coriolis, diffusion, specified/periodic/open boundaries and rational boundary time laws.
- Mixing: coordinate mixing, constant-K, 3-D Smagorinsky, prognostic TKE and TKE budget diagnostics under the shared configuration rules.
- Microphysics: Kessler, WSM3/5/6/6R/7, WDM6, Lin, Ferrier-Aligo, Morrison, Thompson, aerosol Thompson, NSSL, P3, TEMPO and ordinary FAST_SBM under their scheme-specific limits.
- Capacity: classic `wrf_44` Thompson and NSSL sedimentation support at most 256 levels; TEMPO supports at most 512 levels.
- Microphysics coupling: conservative WSM6/WDM6 surface return, cumulus-fed Thompson rain, negative-vapour repair, hydrometeor handover and mixed-scheme nest number carry.
- PBL: MYNN generations and mixing lengths, YSU generations, Shin-Hong, UW and CCPP sa-TKE-EDMF under their supported selectors.
- Cumulus: Kain-Fritsch, Grell-Freitas, Grell 3D, original/new Tiedtke and New SAS under their source and column restrictions.
- Surface: revised MM5 and MYNN surface layers, Noah, RUC, Noah-MP, SLUCM, BEP+BEM, Noah mosaic SLUCM and source-initialized lakes.
- Ocean: mixed-layer and 3DPWP models, prognostic sea ice, ocean-owned heat exchange and restart state.
- Radiation: valid LW/SW combinations of IDs 0, 1, 4, 5 and 90, including Goddard, classic RRTM, legacy RRTMG and RRTMGP with ordinary `icloud=2`.
- Radiation capacity: modern forecast profiles and longwave Planck sources require at most 128 total model/top layers; the isolated shortwave solver uses budgeted workspace.
- Solar/drag: WRF 4.8 solar geometry and eclipse obscuration, internal terrain drag and `gwd_opt=3` with supported SPP controls.
- Wind farms: resident-domain Fitch, EWP, Jensen and GAD with MYNN TKE coupling and turbine/farm power history.
- Fire: coupled SFIRE (`ifire=2`), ideal fire and restart continuation.
- DA: WRF restart initialization, grid/spectral nudging, perturbations, latent heating, hydrostatic rebalance and IAU, including supported prepared nest trees; some preparation and constraints run on the host.
- Stochastic physics: SPPT, SKEBS and SPP, including supported tile forecasts with host-staged pattern transforms.
- Diagnostics: HAILCAST generations, lightning 1/2/3/11, GSD and convective diagnostics, precipitation buckets, whole-domain CFL and scalar PBL mixing.
- Outputs: standard history/restarts, NetCDF compression, Icechunk from compressed history, native S/C/X simulated radar and supported ensemble products.

## Hex and Global

- Hex dynamics: resident global/limited-area meshes, monotonic RRFS transport, supported local-time-step schedules, history and checkpoint continuation on one device.
- Hex column providers: WSM6, Thompson/aerosol Thompson and MYNN/RUC/GSL, including supported synthesized MYNN generations and mixing lengths.
- Hex generic plan: WSM6/YSU/revised-MM5/Noah-MP/RRTMG 4/4 with `icloud=2`, convection off and gravity-wave drag off.
- Native MPAS physics: bounded regional convection-permitting routes with source-authenticated tables, supported physics selectors and checkpoint identity.
- NOAA RRFS: `hrrrv5` on regional and closed zoom meshes with default monotonic scalar transport, native initialization without regional LBCs on closed meshes, and fresh-process checkpoint continuation.
- Native TEMPO history: reflectivity and relative humidity, stream-selected hourly radar maxima and pin-selected diagnostic resets through explicit state/physics stream decks.
- Hex wind farms: complete-globe Fitch through MYNN with turbine/farm power history.
- Hex outputs: shared `--output-format`, `--output-config` and `--icechunk-store`, including Icechunk history.
- Hex UGWP terrain statistics: built on the host when the Rust generator is absent (macOS), from 30-arc-second terrain.
- Hex initialization: supported native correct-mode Thompson/aerosol Thompson/WSM6 starts retain source hydrometeors and publish initial radar fields.
- Global: retained `woof global` forecast commands with `--metal`, spectral transforms, physics, diagnostics and restart support under existing admission rules.

## Remaining refusals and limits

- Full WRF-exact forecasts and `GPUWM_NUMERICS=strict`: complete dynamics/physics arithmetic contracts are unavailable; selected strict component/profile paths have separate admission gates.
- Strict physics: Thompson, complete NSSL, RUC sea-ice/classic surface option 91 and unsupported radiation/mixing/terrain combinations remain gated.
- Strict Goddard: at most 128 model levels; strict RRTM and revised MM5 reject inputs outside their supported finite arithmetic contracts.
- Strict standalone WSM6: flat periodic mixed-phase profiles, physics otherwise off, positive-definite moisture, 5–80 levels and timestep at most 120 seconds.
- Original Tiedtke: referees before glibc 2.41 are refused because their tangent implementation is unavailable.
- FAST_SBM: full WRF-exact mode and polar radar diagnostics remain refused.
- Fire: bulk smoke, firebrand spotting and `fmoist_run` without a surface package remain CUDA only.
- Ensemble products: `cyclone_products` and `sounding_points` remain CUDA only.
- Tiled CFL reduction remains CUDA only; supported tile forecasts retain their existing mode/store constraints.
- Regional wind farms require resident whole-domain ownership; culled regional Hex turbine ownership is unavailable.
- Native MPAS/RRFS: unsupported suites, selectors, correction combinations, unresolved ownership and multi-device requests refuse before stepping.
- Native MPAS output: full isobaric and convective diagnostic stream decks remain refused; use explicit supported state/physics history fields.
- Hex generic plans other than the listed plan remain closed; built-in providers reject source tables and `woof hex pair` lacks a Metal source-capable provider.
- Nested radar tendency forcing and upstream multi-slot/4-D nested IAU remain unavailable.
- Restart configuration and carried-state identity must match; older held-shortwave checkpoints without a producer marker require a known zero radiation-call count.
- Icechunk variable-length strings remain refused.
