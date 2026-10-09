# Command reference

Every door of `woof global`, with every option, generated from the parser by
`tools/build_cli_reference.py`.  Nothing on this page is typed by hand; run
the tool after changing an option and commit what it writes.

`woof global <command> --help` prints the same text at the terminal.

**48 commands.**

- `woof global pins`
- `woof global physics-manifest`
- `woof global doctor`
- `woof global fetch-doors`
- `woof global fetch-analysis`
- `woof global obs fetch`
- `woof global obs subscribe`
- `woof global obs hours`
- `woof global obs summary`
- `woof global obs streams`
- `woof global obs anchors`
- `woof global run-plan`
- `woof global sources`
- `woof global transform-check`
- `woof global run`
- `woof global statics`
- `woof global assimilate`
- `woof global cycle`
- `woof global da init`
- `woof global da cycle`
- `woof global da analyze`
- `woof global da fresh`
- `woof global da forecast`
- `woof global da static-covariance`
- `woof global da localisation`
- `woof global inspect`
- `woof global check-receipt`
- `woof global export-parent`
- `woof global inspect-export`
- `woof global export`
- `woof global microwave`
- `woof global abi-score`
- `woof global abi-reference`
- `woof global abi-fast-model`
- `woof global migrate-level4-checkpoint`
- `woof global check-migration`
- `woof global make-regional-target`
- `woof global inspect-regional-target`
- `woof global translate-regional-frame`
- `woof global inspect-regional-frame`
- `woof global make-parent-series`
- `woof global inspect-parent-series`
- `woof global native-qualify`
- `woof global check-native-evidence`
- `woof global check-native-candidate`
- `woof global go`
- `woof global render`
- `woof global configs`

---

## `woof global pins`

## `woof global physics-manifest`

## `woof global doctor`

| option | what it does | default |
|---|---|---|
| `--json` | machine-readable report on stdout instead of the human table |  |

## `woof global fetch-doors`

| option | what it does | default |
|---|---|---|
| `--from` `DIR_OR_ZIP` | stage from a local bundle archive or a directory of built doors instead of downloading, under identical verification |  |
| `--dest` `DIR` | stage somewhere other than the default companion door directory (woof global doctor prints the resolved path) |  |
| `--list` | print what would be staged, and from where, without staging it |  |

## `woof global fetch-analysis`

| option | what it does | default |
|---|---|---|
| `--out` `OUT` | directory the analysis object and its manifest land in | `data/gdas-analysis` |
| `--cycle` `YYYY-MM-DDTHH` | the GDAS cycle to fetch; without it, the newest published one |  |
| `--engine` `{auto,rust,python}` | which transport the engine's fetch route uses | `auto` |
| `--quiet` | do not print the transport's own progress lines |  |

## `woof global obs`

### `woof global obs fetch`

| option | what it does | default |
|---|---|---|
| `--stream` `{awc-metar,cdaac-ro,gnss-ro,goes-dmw,iem-metar,igra2,madis-aircraft,ndbc,nws-upper-air-text,wis2}` | _(the parser declares no help text for this option)_ |  |
| `--start` `START` | window start, ISO-8601 UTC |  |
| `--end` `END` | window end, ISO-8601 UTC |  |
| `--out` `OUT` | directory; the stream lands under <out>/<stream>/ |  |
| `--networks` `NETWORKS` | iem-metar: comma-separated IEM networks, or 'all' (default) |  |
| `--stations` `STATIONS` | igra2 or ndbc: comma-separated station ids (default all) |  |
| `--satellites` `SATELLITES` | goes-dmw: comma-separated, default G18,G19 |  |
| `--seconds` `SECONDS` | wis2: how long to subscribe (default 600) |  |
| `--missions` `MISSIONS` | cdaac-ro: comma-separated missions (default cosmic2,paz,kompsat5) |  |
| `--archive` `ARCHIVE` | wis2: a subscriber's archive directory to decode for the window instead of subscribing now |  |

### `woof global obs subscribe`

| option | what it does | default |
|---|---|---|
| `--stream` `{wis2}` | the stream to subscribe to; only doors that read a live feed are listed |  |
| `--out` `OUT` | directory; the archive lands under <out>/<stream>/archive/ |  |
| `--seconds` `SECONDS` | how long to stay subscribed (default 600) | `600` |

### `woof global obs hours`

| option | what it does | default |
|---|---|---|
| `--tables` `TABLES` | neutral tables (v2 or v1) |  |
| `--start` `START` | first analysis instant, ISO-8601 UTC |  |
| `--end` `END` | last analysis instant, ISO-8601 UTC |  |
| `--cycle-s` `CYCLE_S` | cycle length in seconds | `3600` |
| `--cutoff` `CUTOFF` | information cutoff: an ISO-8601 instant, or 'analysis' for each hour's own instant; rows received later are dropped and counted, rows without a receipt time are kept and counted |  |
| `--thin-grid` `THIN_GRID` | NLAT,NLON: keep one row per source, variable, cell and ln p layer, the one nearest the analysis instant |  |
| `--out` `OUT` | directory for the hourly tables and hours.json |  |

### `woof global obs summary`

| option | what it does | default |
|---|---|---|
| `--dir` `DIR` | the directory `fetch` wrote the streams under |  |

### `woof global obs streams`

### `woof global obs anchors`

## `woof global run-plan`

| option | what it does | default |
|---|---|---|
| `plan` (positional) | a gpuwm.run-plan.v1 document: which route to execute, which config to execute it with, and where the outputs land |  |
| `--resolve` | print the fully resolved configuration plus every automatic resolution as one JSON document, and run nothing |  |
| `--estimate` | print this plan's VRAM estimate and output-checkpoint counts as one JSON document, and run nothing |  |
| `--catalog` | print the renderer's product catalog as one JSON document -- what may be put in the render_products run option, and which of those a global tape can draw -- and run nothing; needs no plan |  |
| `--sources` | print the source registry as one JSON document -- every source this model initializes from or scores against, and whether the installed engine carries its authority mapping -- and run nothing; needs no plan |  |
| `--physics-profiles` | print the physics menu as one JSON document -- the reference suite and the native suite, which shipped experiments bind each, and what the INSTALLED engine can actually run -- and run nothing; needs no plan |  |
| `--probe` | print this machine's device inventory and readiness as one JSON document; needs no plan. The device inventory is NVML only and creates no CUDA context |  |
| `--no-readiness` | with --probe, report the device inventory only: the NVML-only half, safe to poll on a card that is busy |  |

## `woof global sources`

| option | what it does | default |
|---|---|---|
| `source` (positional) | print ONE row in full, named by its registry id or any alias it declares (omit for the listing) |  |
| `--json` | emit the registry document instead of the table -- the gpuwm.run-plan.sources.v1 schema, narrowed to the one row when ID is given |  |

## `woof global transform-check`

| option | what it does | default |
|---|---|---|
| `--truncation` `TRUNCATION` | _(the parser declares no help text for this option)_ | `15` |
| `--backend` `{numpy,cupy}` | _(the parser declares no help text for this option)_ | `numpy` |
| `--precision` `{float32,float64}` | _(the parser declares no help text for this option)_ | `float64` |
| `--dealias-factor` `DEALIAS_FACTOR` | _(the parser declares no help text for this option)_ | `1.5` |

## `woof global run`

| option | what it does | default |
|---|---|---|
| `config` (positional) | WOOF global experiment TOML (truncation, vertical coordinate, physics suite, gates, and [time] integrator: 'sl_si', the two-time-level semi-Lagrangian core, is the default at every truncation and steps 300 s when dt_s is omitted, with its own drain and gather (order 16 at 720 s, the six-point gather, off-centring 0.55) when those are omitted (at equal cost it grades five scorecard rows better, four worse and nine level against the Eulerian core, better at the surface on temperature and wind and worse on dewpoint and aloft, the rows named on the door page, at about a quarter of the wall per forecast day); 'imex_ssp3', the Eulerian IMEX pair, is selectable by name, its step bounded by the wind and set by a rule when dt_s is omitted (the largest whole step keeping the strongest analysis day on disk under 0.70 of the CFL gate: 90 s at T255, 60 s at T383, 40 s at T533), its ten-step identity pinned) |  |
| `--outdir` `OUTDIR` | where the checkpoints and the self-hashed receipt are written (default out/arwen-global) | `out/arwen-global` |
| `--restart` `RESTART` | continue from this checkpoint instead of the cold state; the config hash it carries has to be this config's |  |
| `--overwrite` | replace this run's own files in --outdir; without it an existing output is refused rather than half-rewritten |  |
| `--until-s` `UNTIL_S` | stop the integration at this model time in seconds from the run's start instead of the config's duration_s (a cycling segment: the next segment restarts from the checkpoint this one ends on); the config identity is unchanged and the receipt records the segment end |  |
| `--profile-steps` `PROFILE_STEPS` | time every operator of this many steps (after --profile-warmup unprofiled ones) on the host and on the device stream, write profile.json beside the receipt and print the table; the run itself is unchanged (default 0, off) | `0` |
| `--profile-warmup` `PROFILE_WARMUP` | steps left unprofiled before the profiled window (default 2: table builds, kernel compiles and the allocator's first shapes are not a step's steady cost) | `2` |
| `--latitude-bands` `N` | how many latitude bands grid space is streamed through (config [memory].latitude_bands, default 0 = the sizer chooses): 1 is the resident run, and above one every grid-space operator of the step runs a band at a time while spectral space stays whole. The Legendre contraction keeps K = N = nlat at every band count, so the band count changes no arithmetic and enters no identity: a banded run shares a config hash, a checkpoint lineage and a receipt with the resident run and reproduces its checkpoints byte for byte. It buys capacity, not speed: a truncation the card cannot hold resident runs, and one that fits gains nothing. NOTHING HAS TO BE SET: with no memory flag and no [memory] table the sizer reads the card first and chooses this and the host tier together, and the receipt's 'sizer' block says what it chose and why |  |
| `--host-spill` `{auto,on,off}` | whether the persistent grid state -- the ten grid tracers, the surface reservoirs and the native physics namespace -- lives in pinned host memory instead of on the card (config [memory].host_spill, default 'auto'). It reaches the card only as the copy its consumer builds anyway, so what stops existing is the original standing beside that copy for the life of the run. 'auto' parks the minimum the predicted peak needs, coldest slice first; 'on' parks all three; 'off' keeps every slice on the card. Memory only, same bits: a spilled run reproduces the resident run's checkpoints byte for byte and shares its config hash. WHAT IT BUYS, MEASURED 2026-09-06 at the ten-step probe of record with nothing set: T383 L40 (34.7 km) starts, fits and reports on a 16 GB RTX 5070 Ti, which it does not do resident at any radiation chunk, and T533 L40 (25.0 km) on a 32 GB RTX 5090, which ran out of memory at step 1 before this work. T533 on a 16 GB card, and T799 on either, are refused at the door by name |  |
| `--cards` `P` | how many GPUs share the band schedule (config [memory].cards, default 1). Grid space is partitioned by latitude band and spectral space is replicated; in the default 'gather' exchange the waist rows cross the wire and the Legendre contraction still runs at K = N = nlat on every card, so two cards return the single-card answer bit for bit. Every rank needs --card-rank and the same --card-addresses list |  |
| `--card-exchange` `{gather,partial}` | 'gather' (default) ships the Fourier waist's latitude rows and keeps the contraction whole, which is bit-identical to one card and enters no identity; 'partial' ships partial Legendre sums and adds them in rank order, which is a third of the bytes and a CHANGE OF ARITHMETIC that carries its own pin and joins the config hash |  |
| `--card-axis` `{band,order}` | which decomposition a run above one card uses (config [memory].card_axis, default 'band'). 'band' partitions grid space by latitude and gathers waist rows -- the SPEED axis, and the one a model run uses. 'order' partitions the Legendre orders and gathers coefficient columns, holding one card's fraction of the table -- the CAPACITY axis past the whole-table wall, proven bit-identical at the transform and refused for a model run until a truncation that needs it has a card that fits it |  |
| `--card-agreement` `{refuse,record}` | what a gather run above one card does when its cards return different bits for a contraction the step presents (config [memory].card_agreement, default 'refuse'). 'refuse' stops the run by name at that contraction, before the waist it feeds is assembled from both cards' rows. 'record' carries on, lists every disagreeing shape in the receipt and FAILS the run's two_card_contractions_agree gate row: a timing device for a pair of unlike cards whose bits do not agree, and never a run of record |  |
| `--card-rank` `R` | this process's rank, 0-based, inside --cards |  |
| `--card-addresses` `H:P,H:P` | the rendezvous host:port of every rank IN RANK ORDER, comma separated. Use the address of the fast interconnect between the cards, not the management network |  |
| `--card-transport` `{auto,tcp,nccl}` | 'auto' (default) takes NCCL where it imports and TCP otherwise; the two measured within 1 percent of each other on this link |  |
| `--card-weights` `W,W` | one relative card speed per rank, in rank order, overriding the per-band cost profile measured at run start. The assignment is outside the arithmetic (gate BIT-6), so this changes who computes a row and not what the row is |  |
| `--card-halo-rows` `N` | latitude rows exchanged across a card boundary for the meridional sweep's deep halo (default 16). It must cover 2n for the step's sub-step count n; a step that needs more is refused by name rather than swept from stale neighbour rows |  |
| `--spectral-chunk` `SPECTRAL_CHUNK` | widest field stack one transform call carries (config [memory].spectral_chunk, default 6): a smaller chunk bounds the complex Fourier temporaries a wide stack materializes and costs one more pass of the per-order loop; it is the Legendre GEMM's M dimension and moves bits at a narrow vertical ladder, so it carries its own config identity |  |
| `--synthesis-memo` `{on,off}` | serve repeated syntheses of one state within a step from a memo (default on); off recomputes every synthesis, which is the memory-tightest form and the same bits: a ten-step T255 native A/B is byte-identical across all 126 checkpoint arrays, so the two share one config hash and one lineage |  |
| `--legendre-band` `LEGENDRE_BAND` | orders per packed Legendre band (config [memory].legendre_band, default 32): the strided-batched GEMM's batch count on cupy, and ARITHMETIC there - a band other than 32 changes the analysis of a single plane by about 4e-06 at T255 float32 and takes a ten-step run to a different state, so it carries its own config and transform identity and a checkpoint written under it will not restart under another |  |
| `--streaming` `{on,off}` | hold no Legendre table and regenerate the basis one band of orders at a time (default off): the entry point for a truncation whose tables do not fit, at 37.7x a resident synthesis and 15,144x a resident analysis (MEASURED T383) |  |
| `--device-allocator` `{default,slab,async}` | which allocator the run spends its device bytes through (config [memory].device_allocator, default 'default'): 'default' is the CuPy pool the process already carries, which held the least over what the run had live and cost the least wall on both shapes measured on an RTX 5070 Ti 2026-09-06; 'slab' takes one contiguous arena before the first model byte and extends it, cutting it with an exact-fit coalescing free list, so what the card holds is the arena and no pool's binned free blocks; 'async' installs the driver's cudaMallocAsync pool. Memory only, same bits: a ten-step checkpoint gate is byte-identical under each (T85, 3 checkpoints, 45 arrays) |  |

## `woof global statics`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML whose grid (truncation, nlat, nlon, dealias) and [statics] table select the build |  |
| `--out` `OUT` | cache file to write; the default is the path the run itself reads, <cache_dir>/arwen-global-statics-T<t>-<nlat>x<nlon>-<geog_data_res>.npz (a different --out is not found by the run unless [statics] cache_dir names its directory) |  |
| `--overwrite` | replace an existing cache and its sidecar |  |
| `--sector-degrees` `SECTOR_DEGREES` | longitude width of one build sector (default 30; bounds the source window read at once) | `30.0` |
| `--geog-root` `GEOG_ROOT` | the WPS_GEOG archive to build from, overriding [statics] geog_root for this build only. Without it the config's own value is used, and without that the engine's archive search answers. A flag is here because the archive is a property of the MACHINE and the config is a property of the EXPERIMENT: pinning a machine path inside a shipped experiment is how a config stops working on the next machine |  |

## `woof global assimilate`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML the checkpoint was run under, or a shipped experiment's bare name |  |
| `--latitude-bands` `N` | how many latitude bands grid space is streamed through (config [memory].latitude_bands, default 0 = the sizer chooses): 1 is the resident run, and above one every grid-space operator of the step runs a band at a time while spectral space stays whole. The Legendre contraction keeps K = N = nlat at every band count, so the band count changes no arithmetic and enters no identity: a banded run shares a config hash, a checkpoint lineage and a receipt with the resident run and reproduces its checkpoints byte for byte. It buys capacity, not speed: a truncation the card cannot hold resident runs, and one that fits gains nothing. NOTHING HAS TO BE SET: with no memory flag and no [memory] table the sizer reads the card first and chooses this and the host tier together, and the receipt's 'sizer' block says what it chose and why |  |
| `--host-spill` `{auto,on,off}` | whether the persistent grid state -- the ten grid tracers, the surface reservoirs and the native physics namespace -- lives in pinned host memory instead of on the card (config [memory].host_spill, default 'auto'). It reaches the card only as the copy its consumer builds anyway, so what stops existing is the original standing beside that copy for the life of the run. 'auto' parks the minimum the predicted peak needs, coldest slice first; 'on' parks all three; 'off' keeps every slice on the card. Memory only, same bits: a spilled run reproduces the resident run's checkpoints byte for byte and shares its config hash. WHAT IT BUYS, MEASURED 2026-09-06 at the ten-step probe of record with nothing set: T383 L40 (34.7 km) starts, fits and reports on a 16 GB RTX 5070 Ti, which it does not do resident at any radiation chunk, and T533 L40 (25.0 km) on a 32 GB RTX 5090, which ran out of memory at step 1 before this work. T533 on a 16 GB card, and T799 on either, are refused at the door by name |  |
| `--cards` `P` | how many GPUs share the band schedule (config [memory].cards, default 1). Grid space is partitioned by latitude band and spectral space is replicated; in the default 'gather' exchange the waist rows cross the wire and the Legendre contraction still runs at K = N = nlat on every card, so two cards return the single-card answer bit for bit. Every rank needs --card-rank and the same --card-addresses list |  |
| `--card-exchange` `{gather,partial}` | 'gather' (default) ships the Fourier waist's latitude rows and keeps the contraction whole, which is bit-identical to one card and enters no identity; 'partial' ships partial Legendre sums and adds them in rank order, which is a third of the bytes and a CHANGE OF ARITHMETIC that carries its own pin and joins the config hash |  |
| `--card-axis` `{band,order}` | which decomposition a run above one card uses (config [memory].card_axis, default 'band'). 'band' partitions grid space by latitude and gathers waist rows -- the SPEED axis, and the one a model run uses. 'order' partitions the Legendre orders and gathers coefficient columns, holding one card's fraction of the table -- the CAPACITY axis past the whole-table wall, proven bit-identical at the transform and refused for a model run until a truncation that needs it has a card that fits it |  |
| `--card-agreement` `{refuse,record}` | what a gather run above one card does when its cards return different bits for a contraction the step presents (config [memory].card_agreement, default 'refuse'). 'refuse' stops the run by name at that contraction, before the waist it feeds is assembled from both cards' rows. 'record' carries on, lists every disagreeing shape in the receipt and FAILS the run's two_card_contractions_agree gate row: a timing device for a pair of unlike cards whose bits do not agree, and never a run of record |  |
| `--card-rank` `R` | this process's rank, 0-based, inside --cards |  |
| `--card-addresses` `H:P,H:P` | the rendezvous host:port of every rank IN RANK ORDER, comma separated. Use the address of the fast interconnect between the cards, not the management network |  |
| `--card-transport` `{auto,tcp,nccl}` | 'auto' (default) takes NCCL where it imports and TCP otherwise; the two measured within 1 percent of each other on this link |  |
| `--card-weights` `W,W` | one relative card speed per rank, in rank order, overriding the per-band cost profile measured at run start. The assignment is outside the arithmetic (gate BIT-6), so this changes who computes a row and not what the row is |  |
| `--card-halo-rows` `N` | latitude rows exchanged across a card boundary for the meridional sweep's deep halo (default 16). It must cover 2n for the step's sub-step count n; a step that needs more is refused by name rather than swept from stale neighbour rows |  |
| `--spectral-chunk` `SPECTRAL_CHUNK` | widest field stack one transform call carries (config [memory].spectral_chunk, default 6): a smaller chunk bounds the complex Fourier temporaries a wide stack materializes and costs one more pass of the per-order loop; it is the Legendre GEMM's M dimension and moves bits at a narrow vertical ladder, so it carries its own config identity |  |
| `--synthesis-memo` `{on,off}` | serve repeated syntheses of one state within a step from a memo (default on); off recomputes every synthesis, which is the memory-tightest form and the same bits: a ten-step T255 native A/B is byte-identical across all 126 checkpoint arrays, so the two share one config hash and one lineage |  |
| `--legendre-band` `LEGENDRE_BAND` | orders per packed Legendre band (config [memory].legendre_band, default 32): the strided-batched GEMM's batch count on cupy, and ARITHMETIC there - a band other than 32 changes the analysis of a single plane by about 4e-06 at T255 float32 and takes a ten-step run to a different state, so it carries its own config and transform identity and a checkpoint written under it will not restart under another |  |
| `--streaming` `{on,off}` | hold no Legendre table and regenerate the basis one band of orders at a time (default off): the entry point for a truncation whose tables do not fit, at 37.7x a resident synthesis and 15,144x a resident analysis (MEASURED T383) |  |
| `--device-allocator` `{default,slab,async}` | which allocator the run spends its device bytes through (config [memory].device_allocator, default 'default'): 'default' is the CuPy pool the process already carries, which held the least over what the run had live and cost the least wall on both shapes measured on an RTX 5070 Ti 2026-09-06; 'slab' takes one contiguous arena before the first model byte and extends it, cutting it with an exact-fit coalescing free list, so what the card holds is the arena and no pool's binned free blocks; 'async' installs the driver's cudaMallocAsync pool. Memory only, same bits: a ten-step checkpoint gate is byte-identical under each (T85, 3 checkpoints, 45 arrays) |  |
| `checkpoint` (positional) | background state to analyse |  |
| `--obs` `PATH_OR_URL` | observation CSV stream (URL or local path, gzip ok); repeatable |  |
| `--length-scale-km` `LENGTH_SCALE_KM` | horizontal influence radius of one report (default 300) |  |
| `--max-age-minutes` `MAX_AGE_MINUTES` | discard reports older than this before the analysis time (default 90) |  |
| `--elevation-limit-m` `ELEVATION_LIMIT_M` | discard a surface report whose station elevation differs from the model's by more than this (default 500) |  |
| `--wind-balance` `{rotational,unconstrained}` | how the wind increment enters the state: rotational (default) applies its streamfunction part only; unconstrained also applies the divergence the scalar spreading produced, which is gravity-wave energy, kept for measuring against the default |  |
| `--moisture-update` `{on,off}` | analyse dewpoint reports into specific humidity (default off): the dewpoint innovation spread with its own vertical localization, the vapor re-derived at every level, capped at saturation and floored at zero; selectable because on the graded 24 h cycle it won the 2 m dewpoint (18 h bias -2.76 to -0.97 K) and lost sea-level pressure (2.70 to 4.19 hPa rmse) beyond the admission rule; off leaves water untouched as the v1 door did |  |
| `--humidity-decay-height-m` `HUMIDITY_DECAY_HEIGHT_M` | e-folding height above the surface of a surface dewpoint report's moisture increment (default 1500) |  |
| `--out` `OUT` | checkpoint to write the analysis to; the o-minus-b/o-minus-a report is written beside it |  |
| `--analysis-time` `ANALYSIS_TIME` | ISO-8601 analysis instant; default is the newest decoded report |  |
| `--overwrite` | replace an existing analysis checkpoint and report |  |

## `woof global cycle`

| option | what it does | default |
|---|---|---|
| `config` (positional) | WOOF global experiment TOML; its duration_s is the end of the whole run, cycle and forecast |  |
| `--latitude-bands` `N` | how many latitude bands grid space is streamed through (config [memory].latitude_bands, default 0 = the sizer chooses): 1 is the resident run, and above one every grid-space operator of the step runs a band at a time while spectral space stays whole. The Legendre contraction keeps K = N = nlat at every band count, so the band count changes no arithmetic and enters no identity: a banded run shares a config hash, a checkpoint lineage and a receipt with the resident run and reproduces its checkpoints byte for byte. It buys capacity, not speed: a truncation the card cannot hold resident runs, and one that fits gains nothing. NOTHING HAS TO BE SET: with no memory flag and no [memory] table the sizer reads the card first and chooses this and the host tier together, and the receipt's 'sizer' block says what it chose and why |  |
| `--host-spill` `{auto,on,off}` | whether the persistent grid state -- the ten grid tracers, the surface reservoirs and the native physics namespace -- lives in pinned host memory instead of on the card (config [memory].host_spill, default 'auto'). It reaches the card only as the copy its consumer builds anyway, so what stops existing is the original standing beside that copy for the life of the run. 'auto' parks the minimum the predicted peak needs, coldest slice first; 'on' parks all three; 'off' keeps every slice on the card. Memory only, same bits: a spilled run reproduces the resident run's checkpoints byte for byte and shares its config hash. WHAT IT BUYS, MEASURED 2026-09-06 at the ten-step probe of record with nothing set: T383 L40 (34.7 km) starts, fits and reports on a 16 GB RTX 5070 Ti, which it does not do resident at any radiation chunk, and T533 L40 (25.0 km) on a 32 GB RTX 5090, which ran out of memory at step 1 before this work. T533 on a 16 GB card, and T799 on either, are refused at the door by name |  |
| `--cards` `P` | how many GPUs share the band schedule (config [memory].cards, default 1). Grid space is partitioned by latitude band and spectral space is replicated; in the default 'gather' exchange the waist rows cross the wire and the Legendre contraction still runs at K = N = nlat on every card, so two cards return the single-card answer bit for bit. Every rank needs --card-rank and the same --card-addresses list |  |
| `--card-exchange` `{gather,partial}` | 'gather' (default) ships the Fourier waist's latitude rows and keeps the contraction whole, which is bit-identical to one card and enters no identity; 'partial' ships partial Legendre sums and adds them in rank order, which is a third of the bytes and a CHANGE OF ARITHMETIC that carries its own pin and joins the config hash |  |
| `--card-axis` `{band,order}` | which decomposition a run above one card uses (config [memory].card_axis, default 'band'). 'band' partitions grid space by latitude and gathers waist rows -- the SPEED axis, and the one a model run uses. 'order' partitions the Legendre orders and gathers coefficient columns, holding one card's fraction of the table -- the CAPACITY axis past the whole-table wall, proven bit-identical at the transform and refused for a model run until a truncation that needs it has a card that fits it |  |
| `--card-agreement` `{refuse,record}` | what a gather run above one card does when its cards return different bits for a contraction the step presents (config [memory].card_agreement, default 'refuse'). 'refuse' stops the run by name at that contraction, before the waist it feeds is assembled from both cards' rows. 'record' carries on, lists every disagreeing shape in the receipt and FAILS the run's two_card_contractions_agree gate row: a timing device for a pair of unlike cards whose bits do not agree, and never a run of record |  |
| `--card-rank` `R` | this process's rank, 0-based, inside --cards |  |
| `--card-addresses` `H:P,H:P` | the rendezvous host:port of every rank IN RANK ORDER, comma separated. Use the address of the fast interconnect between the cards, not the management network |  |
| `--card-transport` `{auto,tcp,nccl}` | 'auto' (default) takes NCCL where it imports and TCP otherwise; the two measured within 1 percent of each other on this link |  |
| `--card-weights` `W,W` | one relative card speed per rank, in rank order, overriding the per-band cost profile measured at run start. The assignment is outside the arithmetic (gate BIT-6), so this changes who computes a row and not what the row is |  |
| `--card-halo-rows` `N` | latitude rows exchanged across a card boundary for the meridional sweep's deep halo (default 16). It must cover 2n for the step's sub-step count n; a step that needs more is refused by name rather than swept from stale neighbour rows |  |
| `--spectral-chunk` `SPECTRAL_CHUNK` | widest field stack one transform call carries (config [memory].spectral_chunk, default 6): a smaller chunk bounds the complex Fourier temporaries a wide stack materializes and costs one more pass of the per-order loop; it is the Legendre GEMM's M dimension and moves bits at a narrow vertical ladder, so it carries its own config identity |  |
| `--synthesis-memo` `{on,off}` | serve repeated syntheses of one state within a step from a memo (default on); off recomputes every synthesis, which is the memory-tightest form and the same bits: a ten-step T255 native A/B is byte-identical across all 126 checkpoint arrays, so the two share one config hash and one lineage |  |
| `--legendre-band` `LEGENDRE_BAND` | orders per packed Legendre band (config [memory].legendre_band, default 32): the strided-batched GEMM's batch count on cupy, and ARITHMETIC there - a band other than 32 changes the analysis of a single plane by about 4e-06 at T255 float32 and takes a ten-step run to a different state, so it carries its own config and transform identity and a checkpoint written under it will not restart under another |  |
| `--streaming` `{on,off}` | hold no Legendre table and regenerate the basis one band of orders at a time (default off): the entry point for a truncation whose tables do not fit, at 37.7x a resident synthesis and 15,144x a resident analysis (MEASURED T383) |  |
| `--device-allocator` `{default,slab,async}` | which allocator the run spends its device bytes through (config [memory].device_allocator, default 'default'): 'default' is the CuPy pool the process already carries, which held the least over what the run had live and cost the least wall on both shapes measured on an RTX 5070 Ti 2026-09-06; 'slab' takes one contiguous arena before the first model byte and extends it, cutting it with an exact-fit coalescing free list, so what the card holds is the arena and no pool's binned free blocks; 'async' installs the driver's cudaMallocAsync pool. Memory only, same bits: a ten-step checkpoint gate is byte-identical under each (T85, 3 checkpoints, 45 arrays) |  |
| `--obs` `PATH_OR_URL` | observation CSV stream (URL or local path, gzip ok); repeatable |  |
| `--length-scale-km` `LENGTH_SCALE_KM` | horizontal influence radius of one report (default 300) |  |
| `--max-age-minutes` `MAX_AGE_MINUTES` | discard reports older than this before the analysis time (default 90) |  |
| `--elevation-limit-m` `ELEVATION_LIMIT_M` | discard a surface report whose station elevation differs from the model's by more than this (default 500) |  |
| `--wind-balance` `{rotational,unconstrained}` | how the wind increment enters the state: rotational (default) applies its streamfunction part only; unconstrained also applies the divergence the scalar spreading produced, which is gravity-wave energy, kept for measuring against the default |  |
| `--moisture-update` `{on,off}` | analyse dewpoint reports into specific humidity (default off): the dewpoint innovation spread with its own vertical localization, the vapor re-derived at every level, capped at saturation and floored at zero; selectable because on the graded 24 h cycle it won the 2 m dewpoint (18 h bias -2.76 to -0.97 K) and lost sea-level pressure (2.70 to 4.19 hPa rmse) beyond the admission rule; off leaves water untouched as the v1 door did |  |
| `--humidity-decay-height-m` `HUMIDITY_DECAY_HEIGHT_M` | e-folding height above the surface of a surface dewpoint report's moisture increment (default 1500) |  |
| `--outdir` `OUTDIR` | where the hourly checkpoints, each cycle's analysis checkpoint (arwen_global_analysis_step*.npz) and report (assimilation-report-step*.json), and the receipt are written |  |
| `--cycles` `CYCLES` | how many analyses to form, one every --interval-s of model time from the run's start; refused when they do not fit before the end of the run |  |
| `--interval-s` `INTERVAL_S` | model time between analyses in seconds (default 3600); must be a whole number of steps | `3600.0` |
| `--start-utc` `START_UTC` | ISO-8601 instant model time zero stands for, so each analysis time is this plus its model time; default is the config's physics start_time_utc, refused when neither exists |  |
| `--until-s` `UNTIL_S` | stop the integration at this model time in seconds from the run's start instead of the config's duration_s (equal to the last analysis time for a cycle without a forecast leg) |  |
| `--restart` `RESTART` | continue from this checkpoint instead of the cold state (an interrupted cycle resumes from its last analysis or hourly checkpoint); the config hash it carries has to be this config's |  |
| `--keep-backgrounds` | also write each analysis hour's background checkpoint; by default only the analysis is written there, the background's identity riding in the report and the chain |  |
| `--partial-analyses` `{on,off}` | when some variables fail the gate of record (default on): withdraw their reports and analyse the hour again with the rest, which have to pass on their own, carrying only the failed variables as the background; off carries the whole background whenever any variable fails, as the per-segment chain did | `on` |
| `--overwrite` | replace this cycle's own files in --outdir; without it an existing output is refused rather than half-rewritten |  |

## `woof global da`

### `woof global da init`

| option | what it does | default |
|---|---|---|
| `config` (positional) | WOOF global experiment TOML, or a shipped experiment's bare name |  |
| `--outdir` `OUTDIR` | where the deterministic checkpoint, the ensemble manifest and the DA receipt are written |  |
| `--from-checkpoint` `FROM_CHECKPOINT` | the analysis checkpoint the ensemble is built from; default is the config's cold start written as step 0 into --outdir |  |
| `--members` `MEMBERS` | ensemble members (default 1: the deterministic filter carries one; more need --filter letkf) | `1` |
| `--analysis-time` `ANALYSIS_TIME` | ISO-8601 instant the analysis stands for, recorded in the manifest |  |
| `--filter` `{successive-correction,letkf}` | the analysis filter (default letkf on fresh since 2026-09-06, successive-correction on init and cycle: the deterministic v1 door); letkf is the dual-resolution ensemble filter (N members at their own truncation resident in one process; the control analysed from its own innovations through the ensemble covariance, the members recentred on it) |  |
| `--ensemble-truncation` `ENSEMBLE_TRUNCATION` | the ensemble members' spectral truncation for --filter letkf (default 127; at or below the deterministic truncation) |  |
| `--control-increment` `{control,ensemble-mean}` | letkf: how the control is updated (default control: its own innovations through the ensemble covariance; ensemble-mean is the comparison experiment, the ensemble-mean increment embedded) | `control` |
| `--recentre-fraction` `RECENTRE_FRACTION` | letkf: how far the ensemble mean moves to the control analysis restricted to the ensemble truncation (default 1.0, full; a fraction is partial recentring) | `1.0` |
| `--recentre-mode` `{increment,state}` | letkf: how the members follow the control analysis (default increment: the ensemble-mean increment is replaced by the control's, anchor included, at the ensemble truncation, so the members keep their own terrain-consistent background; state: the control analysis truncated to the ensemble truncation replaces the ensemble mean, which carries a finer orography's surface pressure onto the coarser grid) | `increment` |
| `--taper-full-degree` `TAPER_FULL_DEGREE` | letkf: the total degree up to which the control increment keeps full weight (default 0.6 of the ensemble truncation) |  |
| `--taper-zero-degree` `TAPER_ZERO_DEGREE` | letkf: the total degree at and above which the control increment is zero (default the ensemble truncation) |  |
| `--additive-inflation` `FRACTION` | letkf: additive inflation as a fraction of the initial perturbation amplitude re-drawn after every analysis (default off: RTPS is the one inflation mechanism) |  |
| `--filter-option` `NAME=VALUE` | letkf: one FilterOptions field set by name (repeatable), for example amv_height_assignment_sigma_pa=10000 or refractivity_vertical_cutoff_lnp=1.5; an unknown name is refused with the field list; the effective options ride in every analysis report and the ensemble manifest |  |
| `--hybrid-beta` `BETA` | letkf: the ensemble weight of the hybrid covariance beta B_ens + (1 - beta) B_static the control's gain is built from, in (0, 1] (default 0.75 on fresh, the completed system's, and 1 on init and cycle; 1 is the ensemble alone); below one the static covariance table is sampled into the localised solve |  |
| `--static-covariance` `TABLE` | letkf: the static covariance table for --hybrid-beta below one: 'packaged' (default, the lagged-forecast estimate shipped with the package), a path written by `woof global da static-covariance`, or 'none' (beta 1 only) | `packaged` |
| `--static-samples` `K` | letkf: draws from the static covariance per analysis for the augmented solve (default the package's 64) |  |
| `--letkf-solve-path` `{auto,device,host}` | letkf: where the localised solve runs (default auto: the members' own namespace, the card when the model is on one; host is the numpy reference the device path is compared against, the same code in the other array module); the receipt names the path taken and its wall |  |
| `--operator-precision` `{state,float64}` | letkf: how the point operators contract a device-resident state (default state: a float32 state's coefficients through float32 GEMMs over 256-term blocks summed in float64, the state's own precision; float64 forces the float64 contraction, the host path's arithmetic to rounding) |  |
| `--overwrite` | replace an existing manifest and cold-start checkpoint in --outdir |  |

### `woof global da cycle`

| option | what it does | default |
|---|---|---|
| `config` (positional) | WOOF global experiment TOML; its duration_s is the end of the whole run, cycle and forecast |  |
| `--obs` `PATH_OR_URL` | observation table (URL or local path, gzip ok) in the obs-table vocabulary, decoded once for every cycle; repeatable |  |
| `--stream` `NAME[:key=value;...]` | observation stream fetched for every analysis window and recorded (URL or path, bytes, SHA-256, latency behind real time): iem-asos[:networks=IA_ASOS,IL_ASOS;bbox=W,S,E,N], local-tables:paths=A.csv,B.csv, atms[:satellites=noaa-20,noaa-21;cache=DIR] and goes-abi[:satellites=G19;bands=13,8;cache=DIR] (the radiance streams, letkf only); repeatable; the table is woof.globe.da_streams.STREAM_TABLE; fresh with no --stream and no --obs runs the shipped roster (da_door.DEFAULT_FRESH_STREAMS), and naming any stream replaces it |  |
| `--length-scale-km` `LENGTH_SCALE_KM` | horizontal influence radius of one report (default 300) |  |
| `--max-age-minutes` `MAX_AGE_MINUTES` | discard reports older than this before the analysis time (default 90) |  |
| `--elevation-limit-m` `ELEVATION_LIMIT_M` | discard a surface report whose station elevation differs from the model's by more than this (default 500) |  |
| `--wind-balance` `{rotational,unconstrained}` | how the wind increment enters the state: rotational (default) applies its streamfunction part only; unconstrained also applies the divergence the scalar spreading produced, which is gravity-wave energy, kept for measuring against the default |  |
| `--moisture-update` `{on,off}` | analyse dewpoint reports into specific humidity (default off): the dewpoint innovation spread with its own vertical localization, the vapor re-derived at every level, capped at saturation and floored at zero; selectable because on the graded 24 h cycle it won the 2 m dewpoint (18 h bias -2.76 to -0.97 K) and lost sea-level pressure (2.70 to 4.19 hPa rmse) beyond the admission rule; off leaves water untouched as the v1 door did |  |
| `--humidity-decay-height-m` `HUMIDITY_DECAY_HEIGHT_M` | e-folding height above the surface of a surface dewpoint report's moisture increment (default 1500) |  |
| `--outdir` `OUTDIR` | where the hourly checkpoints, each cycle's analysis checkpoint and report, the fetch manifests (fetch/), the ensemble manifest, the run receipt and the DA receipt are written |  |
| `--cycles` `CYCLES` | how many analyses to form, one every --interval-s of model time from the run's start |  |
| `--interval-s` `INTERVAL_S` | model time between analyses in seconds (default 3600) | `3600.0` |
| `--start-utc` `START_UTC` | ISO-8601 instant model time zero stands for; default the config's physics start_time_utc, refused when neither exists |  |
| `--until-s` `UNTIL_S` | stop the integration at this model time (default the config's duration_s; equal to the last analysis time for a cycle without a forecast leg) |  |
| `--restart` `RESTART` | continue from this checkpoint instead of the cold state |  |
| `--ensemble` `ENSEMBLE` | the ensemble manifest (da-ensemble.json) to cycle from; its deterministic checkpoint is the restart unless --restart names one |  |
| `--filter` `{successive-correction,letkf}` | the analysis filter (default letkf on fresh since 2026-09-06, successive-correction on init and cycle: the deterministic v1 door); letkf is the dual-resolution ensemble filter (N members at their own truncation resident in one process; the control analysed from its own innovations through the ensemble covariance, the members recentred on it) |  |
| `--ensemble-truncation` `ENSEMBLE_TRUNCATION` | the ensemble members' spectral truncation for --filter letkf (default 127; at or below the deterministic truncation) |  |
| `--control-increment` `{control,ensemble-mean}` | letkf: how the control is updated (default control: its own innovations through the ensemble covariance; ensemble-mean is the comparison experiment, the ensemble-mean increment embedded) | `control` |
| `--recentre-fraction` `RECENTRE_FRACTION` | letkf: how far the ensemble mean moves to the control analysis restricted to the ensemble truncation (default 1.0, full; a fraction is partial recentring) | `1.0` |
| `--recentre-mode` `{increment,state}` | letkf: how the members follow the control analysis (default increment: the ensemble-mean increment is replaced by the control's, anchor included, at the ensemble truncation, so the members keep their own terrain-consistent background; state: the control analysis truncated to the ensemble truncation replaces the ensemble mean, which carries a finer orography's surface pressure onto the coarser grid) | `increment` |
| `--taper-full-degree` `TAPER_FULL_DEGREE` | letkf: the total degree up to which the control increment keeps full weight (default 0.6 of the ensemble truncation) |  |
| `--taper-zero-degree` `TAPER_ZERO_DEGREE` | letkf: the total degree at and above which the control increment is zero (default the ensemble truncation) |  |
| `--additive-inflation` `FRACTION` | letkf: additive inflation as a fraction of the initial perturbation amplitude re-drawn after every analysis (default off: RTPS is the one inflation mechanism) |  |
| `--filter-option` `NAME=VALUE` | letkf: one FilterOptions field set by name (repeatable), for example amv_height_assignment_sigma_pa=10000 or refractivity_vertical_cutoff_lnp=1.5; an unknown name is refused with the field list; the effective options ride in every analysis report and the ensemble manifest |  |
| `--hybrid-beta` `BETA` | letkf: the ensemble weight of the hybrid covariance beta B_ens + (1 - beta) B_static the control's gain is built from, in (0, 1] (default 0.75 on fresh, the completed system's, and 1 on init and cycle; 1 is the ensemble alone); below one the static covariance table is sampled into the localised solve |  |
| `--static-covariance` `TABLE` | letkf: the static covariance table for --hybrid-beta below one: 'packaged' (default, the lagged-forecast estimate shipped with the package), a path written by `woof global da static-covariance`, or 'none' (beta 1 only) | `packaged` |
| `--static-samples` `K` | letkf: draws from the static covariance per analysis for the augmented solve (default the package's 64) |  |
| `--letkf-solve-path` `{auto,device,host}` | letkf: where the localised solve runs (default auto: the members' own namespace, the card when the model is on one; host is the numpy reference the device path is compared against, the same code in the other array module); the receipt names the path taken and its wall |  |
| `--operator-precision` `{state,float64}` | letkf: how the point operators contract a device-resident state (default state: a float32 state's coefficients through float32 GEMMs over 256-term blocks summed in float64, the state's own precision; float64 forces the float64 contraction, the host path's arithmetic to rounding) |  |
| `--observation-bin-s` `OBSERVATION_BIN_S` | compare every report with the state at the bin instant nearest its valid time, bins this many seconds wide (a whole multiple of the model step, dividing the interval); default: at the analysis instant on cycle, and on fresh under letkf 600 s or the smallest multiple of the step above it that divides the interval |  |
| `--increment-application` `{direct,iau}` | iau (default): the window re-integrated from its start with the increment added in equal parts at every step (the members take theirs over the next window); direct: the increment inserted at the analysis instant | `iau` |
| `--anchor` `PATH[:key=value;...]` | the external analysis as a weak low-pass constraint on the control: a checkpoint of this config or a GRIB analysis, with valid_utc=..., weight=0.1, full_degree=30, zero_degree=40, max_age_s=10800, fields=theta,log_surface_pressure,... |  |
| `--keep-backgrounds` | also write each analysis hour's background checkpoint |  |
| `--partial-analyses` `{on,off}` | when some variables fail the gate of record (default on): withdraw them and analyse the hour again with the rest | `on` |
| `--overwrite` | replace this cycle's own files in --outdir |  |

### `woof global da analyze`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML the checkpoint was run under, or a shipped experiment's bare name |  |
| `checkpoint` (positional) | background state to analyse |  |
| `--obs` `PATH_OR_URL` | observation CSV stream (URL or local path, gzip ok); repeatable |  |
| `--length-scale-km` `LENGTH_SCALE_KM` | horizontal influence radius of one report (default 300) |  |
| `--max-age-minutes` `MAX_AGE_MINUTES` | discard reports older than this before the analysis time (default 90) |  |
| `--elevation-limit-m` `ELEVATION_LIMIT_M` | discard a surface report whose station elevation differs from the model's by more than this (default 500) |  |
| `--wind-balance` `{rotational,unconstrained}` | how the wind increment enters the state: rotational (default) applies its streamfunction part only; unconstrained also applies the divergence the scalar spreading produced, which is gravity-wave energy, kept for measuring against the default |  |
| `--moisture-update` `{on,off}` | analyse dewpoint reports into specific humidity (default off): the dewpoint innovation spread with its own vertical localization, the vapor re-derived at every level, capped at saturation and floored at zero; selectable because on the graded 24 h cycle it won the 2 m dewpoint (18 h bias -2.76 to -0.97 K) and lost sea-level pressure (2.70 to 4.19 hPa rmse) beyond the admission rule; off leaves water untouched as the v1 door did |  |
| `--humidity-decay-height-m` `HUMIDITY_DECAY_HEIGHT_M` | e-folding height above the surface of a surface dewpoint report's moisture increment (default 1500) |  |
| `--out` `OUT` | directory for the analysis checkpoint, the report (with its scorecard) and the DA receipt |  |
| `--analysis-time` `ANALYSIS_TIME` | ISO-8601 analysis instant; default is the newest decoded report |  |
| `--filter` `{successive-correction,letkf}` | the analysis filter (default letkf on fresh since 2026-09-06, successive-correction on init and cycle: the deterministic v1 door); letkf is the dual-resolution ensemble filter (N members at their own truncation resident in one process; the control analysed from its own innovations through the ensemble covariance, the members recentred on it) |  |
| `--control-increment` `{control,ensemble-mean}` | letkf: how the control is updated (default control: its own innovations through the ensemble covariance; ensemble-mean is the comparison experiment, the ensemble-mean increment embedded) | `control` |
| `--recentre-fraction` `RECENTRE_FRACTION` | letkf: how far the ensemble mean moves to the control analysis restricted to the ensemble truncation (default 1.0, full; a fraction is partial recentring) | `1.0` |
| `--recentre-mode` `{increment,state}` | letkf: how the members follow the control analysis (default increment: the ensemble-mean increment is replaced by the control's, anchor included, at the ensemble truncation, so the members keep their own terrain-consistent background; state: the control analysis truncated to the ensemble truncation replaces the ensemble mean, which carries a finer orography's surface pressure onto the coarser grid) | `increment` |
| `--taper-full-degree` `TAPER_FULL_DEGREE` | letkf: the total degree up to which the control increment keeps full weight (default 0.6 of the ensemble truncation) |  |
| `--taper-zero-degree` `TAPER_ZERO_DEGREE` | letkf: the total degree at and above which the control increment is zero (default the ensemble truncation) |  |
| `--additive-inflation` `FRACTION` | letkf: additive inflation as a fraction of the initial perturbation amplitude re-drawn after every analysis (default off: RTPS is the one inflation mechanism) |  |
| `--filter-option` `NAME=VALUE` | letkf: one FilterOptions field set by name (repeatable), for example amv_height_assignment_sigma_pa=10000 or refractivity_vertical_cutoff_lnp=1.5; an unknown name is refused with the field list; the effective options ride in every analysis report and the ensemble manifest |  |
| `--hybrid-beta` `BETA` | letkf: the ensemble weight of the hybrid covariance beta B_ens + (1 - beta) B_static the control's gain is built from, in (0, 1] (default 0.75 on fresh, the completed system's, and 1 on init and cycle; 1 is the ensemble alone); below one the static covariance table is sampled into the localised solve |  |
| `--static-covariance` `TABLE` | letkf: the static covariance table for --hybrid-beta below one: 'packaged' (default, the lagged-forecast estimate shipped with the package), a path written by `woof global da static-covariance`, or 'none' (beta 1 only) | `packaged` |
| `--static-samples` `K` | letkf: draws from the static covariance per analysis for the augmented solve (default the package's 64) |  |
| `--letkf-solve-path` `{auto,device,host}` | letkf: where the localised solve runs (default auto: the members' own namespace, the card when the model is on one; host is the numpy reference the device path is compared against, the same code in the other array module); the receipt names the path taken and its wall |  |
| `--operator-precision` `{state,float64}` | letkf: how the point operators contract a device-resident state (default state: a float32 state's coefficients through float32 GEMMs over 256-term blocks summed in float64, the state's own precision; float64 forces the float64 contraction, the host path's arithmetic to rounding) |  |
| `--overwrite` | replace an existing analysis checkpoint and report |  |

### `woof global da fresh`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the BASE experiment TOML; fresh derives the run config from it (fresh-config.toml in --outdir: the fetched analysis as the initial state, duration = cycle span + --forecast-hours) |  |
| `--obs` `PATH_OR_URL` | observation table (URL or local path, gzip ok) in the obs-table vocabulary, decoded once for every cycle; repeatable |  |
| `--stream` `NAME[:key=value;...]` | observation stream fetched for every analysis window and recorded (URL or path, bytes, SHA-256, latency behind real time): iem-asos[:networks=IA_ASOS,IL_ASOS;bbox=W,S,E,N], local-tables:paths=A.csv,B.csv, atms[:satellites=noaa-20,noaa-21;cache=DIR] and goes-abi[:satellites=G19;bands=13,8;cache=DIR] (the radiance streams, letkf only); repeatable; the table is woof.globe.da_streams.STREAM_TABLE; fresh with no --stream and no --obs runs the shipped roster (da_door.DEFAULT_FRESH_STREAMS), and naming any stream replaces it |  |
| `--length-scale-km` `LENGTH_SCALE_KM` | horizontal influence radius of one report (default 300) |  |
| `--max-age-minutes` `MAX_AGE_MINUTES` | discard reports older than this before the analysis time (default 90) |  |
| `--elevation-limit-m` `ELEVATION_LIMIT_M` | discard a surface report whose station elevation differs from the model's by more than this (default 500) |  |
| `--wind-balance` `{rotational,unconstrained}` | how the wind increment enters the state: rotational (default) applies its streamfunction part only; unconstrained also applies the divergence the scalar spreading produced, which is gravity-wave energy, kept for measuring against the default |  |
| `--moisture-update` `{on,off}` | analyse dewpoint reports into specific humidity (default off): the dewpoint innovation spread with its own vertical localization, the vapor re-derived at every level, capped at saturation and floored at zero; selectable because on the graded 24 h cycle it won the 2 m dewpoint (18 h bias -2.76 to -0.97 K) and lost sea-level pressure (2.70 to 4.19 hPa rmse) beyond the admission rule; off leaves water untouched as the v1 door did |  |
| `--humidity-decay-height-m` `HUMIDITY_DECAY_HEIGHT_M` | e-folding height above the surface of a surface dewpoint report's moisture increment (default 1500) |  |
| `--outdir` `OUTDIR` | where the analysis fetch (analysis/), the stream fetches (fetch/), the cycle and the receipts are written |  |
| `--analysis-grib` `ANALYSIS_GRIB` | an analysis GRIB already on disk (with --analysis-cycle) instead of fetching the newest GDAS cycle |  |
| `--analysis-cycle` `ANALYSIS_CYCLE` | ISO-8601 instant of the analysis cycle to fetch or of --analysis-grib; default the newest published GDAS cycle |  |
| `--start-utc` `START_UTC` | for an analytic base config (a smoke or OSSE case, nothing fetched): the instant model time zero stands for |  |
| `--until-utc` `UNTIL_UTC` | the newest observation hour to cycle to (ISO-8601); default the current hour minus --observation-latency-s, floored to the hour |  |
| `--observation-latency-s` `OBSERVATION_LATENCY_S` | how far behind real time the observation streams are complete (default 3600) | `3600.0` |
| `--forecast-hours` `FORECAST_HOURS` | the forecast length the derived config allows after the last analysis (default 24) | `24.0` |
| `--interval-s` `INTERVAL_S` | model time between analyses in seconds (default 3600) | `3600.0` |
| `--members` `MEMBERS` | ensemble members when fresh has to init (default 32 under letkf, 1 under successive-correction) |  |
| `--filter` `{successive-correction,letkf}` | the analysis filter (default letkf on fresh since 2026-09-06, successive-correction on init and cycle: the deterministic v1 door); letkf is the dual-resolution ensemble filter (N members at their own truncation resident in one process; the control analysed from its own innovations through the ensemble covariance, the members recentred on it) |  |
| `--ensemble-truncation` `ENSEMBLE_TRUNCATION` | the ensemble members' spectral truncation for --filter letkf (default 127; at or below the deterministic truncation) |  |
| `--control-increment` `{control,ensemble-mean}` | letkf: how the control is updated (default control: its own innovations through the ensemble covariance; ensemble-mean is the comparison experiment, the ensemble-mean increment embedded) | `control` |
| `--recentre-fraction` `RECENTRE_FRACTION` | letkf: how far the ensemble mean moves to the control analysis restricted to the ensemble truncation (default 1.0, full; a fraction is partial recentring) | `1.0` |
| `--recentre-mode` `{increment,state}` | letkf: how the members follow the control analysis (default increment: the ensemble-mean increment is replaced by the control's, anchor included, at the ensemble truncation, so the members keep their own terrain-consistent background; state: the control analysis truncated to the ensemble truncation replaces the ensemble mean, which carries a finer orography's surface pressure onto the coarser grid) | `increment` |
| `--taper-full-degree` `TAPER_FULL_DEGREE` | letkf: the total degree up to which the control increment keeps full weight (default 0.6 of the ensemble truncation) |  |
| `--taper-zero-degree` `TAPER_ZERO_DEGREE` | letkf: the total degree at and above which the control increment is zero (default the ensemble truncation) |  |
| `--additive-inflation` `FRACTION` | letkf: additive inflation as a fraction of the initial perturbation amplitude re-drawn after every analysis (default off: RTPS is the one inflation mechanism) |  |
| `--filter-option` `NAME=VALUE` | letkf: one FilterOptions field set by name (repeatable), for example amv_height_assignment_sigma_pa=10000 or refractivity_vertical_cutoff_lnp=1.5; an unknown name is refused with the field list; the effective options ride in every analysis report and the ensemble manifest |  |
| `--hybrid-beta` `BETA` | letkf: the ensemble weight of the hybrid covariance beta B_ens + (1 - beta) B_static the control's gain is built from, in (0, 1] (default 0.75 on fresh, the completed system's, and 1 on init and cycle; 1 is the ensemble alone); below one the static covariance table is sampled into the localised solve |  |
| `--static-covariance` `TABLE` | letkf: the static covariance table for --hybrid-beta below one: 'packaged' (default, the lagged-forecast estimate shipped with the package), a path written by `woof global da static-covariance`, or 'none' (beta 1 only) | `packaged` |
| `--static-samples` `K` | letkf: draws from the static covariance per analysis for the augmented solve (default the package's 64) |  |
| `--letkf-solve-path` `{auto,device,host}` | letkf: where the localised solve runs (default auto: the members' own namespace, the card when the model is on one; host is the numpy reference the device path is compared against, the same code in the other array module); the receipt names the path taken and its wall |  |
| `--operator-precision` `{state,float64}` | letkf: how the point operators contract a device-resident state (default state: a float32 state's coefficients through float32 GEMMs over 256-term blocks summed in float64, the state's own precision; float64 forces the float64 contraction, the host path's arithmetic to rounding) |  |
| `--observation-bin-s` `OBSERVATION_BIN_S` | compare every report with the state at the bin instant nearest its valid time, bins this many seconds wide (a whole multiple of the model step, dividing the interval); default: at the analysis instant on cycle, and on fresh under letkf 600 s or the smallest multiple of the step above it that divides the interval |  |
| `--increment-application` `{direct,iau}` | iau (default): the window re-integrated from its start with the increment added in equal parts at every step (the members take theirs over the next window); direct: the increment inserted at the analysis instant | `iau` |
| `--anchor` `PATH[:key=value;...]` | the external analysis as a weak low-pass constraint on the control: a checkpoint of this config or a GRIB analysis, with valid_utc=..., weight=0.1, full_degree=30, zero_degree=40, max_age_s=10800, fields=theta,log_surface_pressure,... |  |
| `--cutoff-utc` `CUTOFF_UTC` | the information cutoff (ISO-8601): the analysis handed back is the latest constructible from information available by then (default now); the newest observation hour is the cutoff minus --observation-latency-s, floored |  |
| `--fetch-engine` `{auto,rust,python}` | the fetch engine for the GDAS analysis (default auto: rw_fetch when built) | `auto` |
| `--overwrite` | replace the derived config, the ensemble and the cycle in --outdir |  |
| `--keep-backgrounds` | also write each analysis hour's background checkpoint (the free forecast to the instant, beside the analysis handed back), so the increment can be read from the two files; what is analysed does not change |  |

### `woof global da forecast`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML the analysis was formed under (fresh writes fresh-config.toml) |  |
| `--analysis` `ANALYSIS` | the analysis checkpoint to start from (the DA receipt's analysis_checkpoint) |  |
| `--outdir` `OUTDIR` | where the forecast checkpoints and receipts are written |  |
| `--until-s` `UNTIL_S` | stop the forecast at this model time from the run's start (default the config's duration_s) |  |
| `--overwrite` | replace this run's own files in --outdir |  |

### `woof global da static-covariance`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML the forecasts ran under, or a shipped experiment's bare name (the transform and the level ladder the table is estimated on) |  |
| `--pair` `LATER,EARLIER` | one lagged pair: the checkpoint of the longer forecast and the checkpoint of the shorter one valid at the same instant (f024,f012); repeatable, at least two pairs |  |
| `--out` `OUT` | where the table (static-covariance.npz), its receipt and its charts are written |  |
| `--version` `VERSION` | the table's version label carried in the receipt (default the sample's date range) |  |
| `--ridge` `RIDGE` | the relative ridge of the balance regressions (default 1e-3) | `0.001` |
| `--no-charts` | skip the variance-spectrum and correlation charts |  |
| `--backend` `{numpy,cupy}` | the array backend the transform runs on (default the config's; numpy estimates on a host without a card) |  |
| `--precision` `{float32,float64}` | the transform precision of the estimation (default float64, whatever the run's precision) | `float64` |
| `--overwrite` | replace an existing table in --out |  |

### `woof global da localisation`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML the ensemble was built under, or a shipped experiment's bare name (its vertical coordinate) |  |
| `--ensemble` `ENSEMBLE` | the ensemble store (the directory holding the member checkpoints and their manifest, or the manifest itself) |  |
| `--out` `OUT` | where the derivation receipt is written (JSON) |  |
| `--step` `STEP` | read the members at this step instead of the manifest's |  |

## `woof global inspect`

| option | what it does | default |
|---|---|---|
| `checkpoint` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global check-receipt`

| option | what it does | default |
|---|---|---|
| `receipt` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global export-parent`

| option | what it does | default |
|---|---|---|
| `config` (positional) | _(the parser declares no help text for this option)_ |  |
| `checkpoint` (positional) | _(the parser declares no help text for this option)_ |  |
| `output` (positional) | _(the parser declares no help text for this option)_ |  |
| `--nlat` `NLAT` | _(the parser declares no help text for this option)_ |  |
| `--nlon` `NLON` | _(the parser declares no help text for this option)_ |  |
| `--overwrite` | _(the parser declares no help text for this option)_ |  |

## `woof global inspect-export`

| option | what it does | default |
|---|---|---|
| `input` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global export`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML the checkpoints were run under, or a shipped experiment's bare name |  |
| `checkpoints` (positional) | checkpoints to export, one tape each, in time order |  |
| `--outdir` `OUTDIR` | where the tapes are written |  |
| `--nlat` `NLAT` | latitude points of the regular output grid (default 360) | `360` |
| `--nlon` `NLON` | longitude points of the regular output grid (default 720) | `720` |
| `--start-date` `START_DATE` | analysis valid time as YYYY-MM-DD_HH:MM:SS; checkpoint times offset from it |  |
| `--overwrite` | replace tapes that already exist |  |
| `--bbox` `('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX')` | crop the tape to a lat/lon window (degrees, lon in -180..180) |  |

## `woof global microwave`

| option | what it does | default |
|---|---|---|
| `microwave_args` (positional) | the microwave door's own subcommand and arguments: fetch, decode, thin, columns, calibrate, score (each with --help) |  |

## `woof global abi-score`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML the checkpoint was run under, or a shipped experiment's bare name |  |
| `checkpoint` (positional) | the checkpoint whose state is rendered (an analysis or a forecast) |  |
| `--start-date` `START_DATE` | analysis valid time as YYYY-MM-DD_HH:MM:SS; the checkpoint time offsets from it |  |
| `--out` `OUT` | where tapes, planes, packs, tables, charts and the receipt go |  |
| `--goes-rad` `BAND=FILE` | an ABI-L1b-Rad granule of the scan for one band, e.g. 13=OR_ABI-L1b-RadF-M6C13_...nc; repeatable |  |
| `--goes-acm` `GOES_ACM` | the ABI-L2-ACM clear-sky mask of the same scan (without it no pixel is obs-clear or obs-cloudy) |  |
| `--goes-cmip` `BAND=FILE` | an ABI-L2-CMIP granule of the same band and scan, cross-checking the L1b inversion; repeatable | `[]` |
| `--tile` `LABEL,LATMIN,LATMAX,LONMIN,LONMAX` | a render tile (every corner must be on the visible disk); default: the four GOES-East tiles of abi_operator.DEFAULT_TILES |  |
| `--bands` `BANDS` | ABI bands to score, comma-separated (default 13,8) | `13,8` |
| `--rw-goes` `RW_GOES` | the rw_goes front door (else WOOF_RW_GOES, the tree, PATH) |  |
| `--simsat-cli` `SIMSAT_CLI` | simsat-render-ir, for the condensate mask that classes simulated columns clear or cloudy |  |
| `--threads` `THREADS` | SimSat render threads (default: rayon's) |  |
| `--block` `BLOCK` | block side in lattice pixels for the block table (default 24) | `24` |
| `--zenith-max` `ZENITH_MAX` | drop pairs beyond this satellite zenith angle (deg) |  |
| `--calibrate` | also plant skin and upper-vapor changes in the checkpoint and read them back in both bands |  |
| `--goes-received` `BAND=ISO` | when this system first held the band's radiance granule (the fetch manifest's fetched_at), recorded in the pack's provenance row; repeatable | `[]` |
| `--no-charts` | skip the matplotlib analysis charts |  |
| `--no-reuse-tapes` | export every tape again |  |
| `--nlat` `NLAT` | latitude points of the tape grid (default 720) | `720` |
| `--nlon` `NLON` | longitude points of the tape grid (default 1440) | `1440` |

## `woof global abi-reference`

| option | what it does | default |
|---|---|---|
| `mode` (positional) | columns: write the gpuwm-da.abi-columns.v1 stream; score: read CRTM outputs and score |  |
| `--blocks` `BAND=CSV` | the block table rw_goes colocate wrote for a band (band13-blocks.csv); repeatable |  |
| `--out` `OUT` | columns: the stream to write (its .json and .npz sidecars beside it); score: the directory |  |
| `--zenith-max` `ZENITH_MAX` | columns: keep blocks to this zenith (deg) | `70.0` |
| `--config` `CONFIG` | columns: the experiment TOML, or a shipped experiment's bare name (the vertical coordinate) |  |
| `--checkpoint` `CHECKPOINT` | columns: the checkpoint whose surface planes are read |  |
| `--tapes` `TAPES` | columns: glob of the export tapes the blocks were rendered from |  |
| `--valid` `VALID` | columns: the analysis valid time YYYY-MM-DD_HH:MM:SS (season) |  |
| `--emissivity` `BAND=VALUE` | columns: the per-band surface emissivity carried for the reference's user-emissivity run (default the operator's own 0.99 for every band asked) |  |
| `--columns` `COLUMNS` | score: the columns .npz the columns mode wrote |  |
| `--run` `NAME=OUT.bin` | score: a CRTM output stream by name; repeatable |  |
| `--primary` `PRIMARY` | score: the run with CRTM's own surface models (the reference) |  |
| `--simsat-emissivity-run` `SIMSAT_EMISSIVITY_RUN` | score: the run at the operator's emissivity (the absorption term reads against it) |  |
| `--zenith-gate` `ZENITH_GATE` | score: the gate's zenith bound (deg) | `60.0` |
| `--gate-k` `GATE_K` | score: the gate (default the operator's 1.5 K) |  |
| `--no-charts` | score: skip the analysis charts |  |
| `--table` `TABLE` | score: the fast-model table of the primary run; with it the door writes the operator entries the scorecard admits (operator-entries.json) and the four assessments |  |
| `--reference-run` `REFERENCE_RUN` | score: with --table, the run that is the numerical reference (CRTM with its own surface models); the entries' Jacobian agreement and brightness-temperature difference are measured against it |  |

## `woof global abi-fast-model`

| option | what it does | default |
|---|---|---|
| `mode` (positional) | train: fit the table from a reference run; forward: run rw_goes forward on a columns stream |  |
| `--columns` `COLUMNS` | train: the columns .npz the abi-reference columns door wrote; forward: the columns .bin stream |  |
| `--out` `OUT` | train: the table JSON; forward: the output stream |  |
| `--reference` `REFERENCE` | train: the reference run (gpuwm-da.abi-crtm.v1) whose layer optical depths are fitted |  |
| `--pack` `BAND=GOESPACK` | train: the gpuwm-obs.goes-bt.v1 pack whose Planck row the band uses; repeatable |  |
| `--planck` `BAND=fk1,fk2,bc1,bc2` | train: Planck constants given directly (instead of --pack); repeatable |  |
| `--form` `BAND=linear|two_term` | train: the layer model form per band (default 13=linear, others two_term) |  |
| `--provenance` `PROVENANCE` | train: a JSON file of reference provenance (CRTM version, coefficient hashes) copied into the table |  |
| `--satellite` `SATELLITE` | train: the instrument the Planck rows belong to (default G19) | `G19` |
| `--table` `TABLE` | forward: the coefficient table |  |
| `--rw-goes` `RW_GOES` | forward: the rw_goes front door |  |
| `--emis-mode` `EMIS_MODE` | forward: 0 the table's emissivity, 1 the columns' own | `0` |
| `--bands` `BANDS` | forward: bands to evaluate, comma-separated (default: the table's) |  |
| `--threads` `THREADS` | forward: worker threads |  |

## `woof global migrate-level4-checkpoint`

| option | what it does | default |
|---|---|---|
| `config` (positional) | target run config, or the name of a shipped experiment |  |
| `input` (positional) | _(the parser declares no help text for this option)_ |  |
| `output` (positional) | _(the parser declares no help text for this option)_ |  |
| `--receipt` `RECEIPT` | _(the parser declares no help text for this option)_ |  |
| `--allow-native-zero-moments` | _(the parser declares no help text for this option)_ |  |
| `--overwrite` | _(the parser declares no help text for this option)_ |  |

## `woof global check-migration`

| option | what it does | default |
|---|---|---|
| `receipt` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global make-regional-target`

| option | what it does | default |
|---|---|---|
| `input` (positional) | _(the parser declares no help text for this option)_ |  |
| `output` (positional) | _(the parser declares no help text for this option)_ |  |
| `--name` `NAME` | _(the parser declares no help text for this option)_ |  |
| `--grid-id` `GRID_ID` | _(the parser declares no help text for this option)_ |  |
| `--source-identity-json` `SOURCE_IDENTITY_JSON` | _(the parser declares no help text for this option)_ |  |
| `--overwrite` | _(the parser declares no help text for this option)_ |  |

## `woof global inspect-regional-target`

| option | what it does | default |
|---|---|---|
| `input` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global translate-regional-frame`

| option | what it does | default |
|---|---|---|
| `parent` (positional) | _(the parser declares no help text for this option)_ |  |
| `target` (positional) | _(the parser declares no help text for this option)_ |  |
| `output` (positional) | _(the parser declares no help text for this option)_ |  |
| `--overwrite` | _(the parser declares no help text for this option)_ |  |

## `woof global inspect-regional-frame`

| option | what it does | default |
|---|---|---|
| `input` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global make-parent-series`

| option | what it does | default |
|---|---|---|
| `target` (positional) | _(the parser declares no help text for this option)_ |  |
| `output` (positional) | _(the parser declares no help text for this option)_ |  |
| `frames` (positional) | _(the parser declares no help text for this option)_ |  |
| `--overwrite` | _(the parser declares no help text for this option)_ |  |

## `woof global inspect-parent-series`

| option | what it does | default |
|---|---|---|
| `input` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global native-qualify`

| option | what it does | default |
|---|---|---|
| `config` (positional) | _(the parser declares no help text for this option)_ |  |
| `--outdir` `OUTDIR` | _(the parser declares no help text for this option)_ |  |
| `--overwrite` | replace this door's own artifacts in --outdir (native-device-evidence.json, native-contract-candidate.json, continuous/, resumed/); other files are left untouched |  |

## `woof global check-native-evidence`

| option | what it does | default |
|---|---|---|
| `input` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global check-native-candidate`

| option | what it does | default |
|---|---|---|
| `input` (positional) | _(the parser declares no help text for this option)_ |  |

## `woof global go`

| option | what it does | default |
|---|---|---|
| `config` (positional) | the experiment TOML, or the name of a shipped experiment |  |
| `--outdir` `OUTDIR` | the run directory; checkpoints, receipt, status.json and the log go here, and pictures under <outdir>/pictures |  |
| `--start-date` `START_DATE` | analysis valid time as YYYY-MM-DD_HH:MM:SS for the render stage; without it the forecast still runs and the render stage is skipped out loud, because a tape with no valid time is a tape nobody can place in time |  |
| `--products` `LIST` | comma-separated products for the render stage |  |
| `--geog-root` `GEOG_ROOT` | the WPS_GEOG archive for the statics stage; without it the config's own [statics] geog_root is used |  |
| `--no-statics` | skip the statics stage even for a real planet |  |
| `--no-render` | stop after the forecast |  |
| `--overwrite` | replace an existing run directory's artifacts |  |

## `woof global render`

| option | what it does | default |
|---|---|---|
| `--config` `CONFIG` | the experiment TOML the checkpoints were run under, or the name of a shipped experiment. Optional: a run directory written by `run` or `go` carries a copy of its own config, and that copy is used when this flag is absent |  |
| `inputs` (positional) | checkpoints in time order, run directories (every arwen_global_step*.npz inside, sorted), or wrfout tapes that are already exported |  |
| `--outdir` `OUTDIR` | where the pictures go, laid out <outdir>/<domain>/<product>/<valid-day>/ |  |
| `--start-date` `START_DATE` | analysis valid time as YYYY-MM-DD_HH:MM:SS; checkpoint times offset from it |  |
| `--products` `LIST` | comma-separated products, or 'all' (default: 2m_temperature,mslp_10m_winds,10m_wind_speed_and_direction,total_qpf) | `2m_temperature,mslp_10m_winds,10m_wind_speed_and_direction,total_qpf` |
| `--size` `WxH` | picture size in pixels (default 1600x1000) | `1600x1000` |
| `--nlat` `NLAT` | latitude points of the export grid (default 360) | `360` |
| `--nlon` `NLON` | longitude points of the export grid (default 720) | `720` |
| `--bbox` `('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX')` | crop to a lat/lon window (degrees, lon in -180..180) |  |
| `--tapes-dir` `TAPES_DIR` | where the intermediate tapes are written (default: a directory under --outdir that is removed when the pictures are drawn) |  |
| `--keep-tapes` | keep the exported tapes and their receipt |  |
| `--overwrite` | replace tapes and pictures that already exist |  |

## `woof global configs`

| option | what it does | default |
|---|---|---|
| `--paths` | print full paths instead of names |  |
