# Changelog

## 0.1.3 (unreleased)

### Fixed

- `woof global doctor` exits 0 on a correct install against the published
  woof 2.8.0.  It graded `preflight.measured_free_vram_bytes` and
  `surface_bias.interpolate_to_tape`, which that engine does not carry, as
  gaps and exited 1, though no documented command reaches either.  Both now
  print as optional notes; calling either still refuses by name.
- A Rust door the installed engine's own bundle declares is the engine's
  door.  `doctor` grades it against the engine's pins and names the
  engine's `fetch-bridges` as its remedy, and `fetch-doors` leaves it to the
  engine.  On an install whose engine bundle carries every door this model
  runs on, `doctor` graded the engine's `rw_asos` and `rw_goes` against this
  package's pins for a different build, reported the other six as not
  staged, and failed with eight gaps; there `fetch-doors` now says it has
  nothing to stage and succeeds.  The contract literal still has to be in
  the bytes, so an engine build older than the contract is still a gap.
- `fetch-doors` downloads from the repository the installed distribution's
  own metadata names, not from an address written into the source.
- An open-water column starts on a water temperature.  Open water holds its
  starting skin temperature for the whole forecast (there is no ocean model;
  an inland lake now follows its own heat budget, below), and the cold start
  took that skin from a plain bilinear regrid of the analysis skin
  temperature, which mixes in the analysis's land points wherever the run's
  land fraction calls a column water and the analysis calls part of its
  stencil land.  On the GDAS 2026-09-30 12Z analysis the north basin of Lake
  Turkana (4.45 N, 36.09 E on the T255 grid) started at 321.1 K, the
  afternoon desert around the lake, where the analysis's own lake point read
  300.8 K.  Held for the forecast, that column evaporated about 1.2e-3
  kg/m2/s, near 3,000 W/m2 of latent heat, day and night, and emptied its
  500 kg/m2 surface reservoir at hour 117.3 of a 120 h T255 forecast, which
  ended with "native physics water closure exceeds the explicit surface
  reservoir".  The water books were exact; the flux they booked was wrong.
  Open-water columns now take the analysis skin over the analysis's own
  water points only: the bilinear weights renormalised over the stencil's
  water points, and a stencil with none reads the analysis water values
  grown outward over its land, the rule metgrid applies to SST.  Land and
  sea-ice columns keep the plain regrid.  On that analysis the rule changed
  1,784 of the 164,018 open-water columns by more than 1 K, and the run
  receipt records it under `initial.provenance.open_water_skin`.
- The land surface never receives convective rain the atmosphere did not
  lose.  Grell-Freitas's deep arm reports more rain than its own tendencies
  take out of the column: 1.27 times over the raining columns of WRF
  v4.6.1's oracle fixture, and 1.77 times over the globe and 2.28 times over
  land in the first day of the T255 GDAS 2026-09-30 12Z forecast.  Noah was
  forced with the reported number, so its soil, canopy and runoff gained
  water that never left the atmosphere, and the surface reservoir paid for
  it: in a 240 h forecast from that analysis the reservoir of a convective
  column in Colombia fell to 124 kg/m2 as the scheme reported 660 kg/m2 of
  rain and removed 288, about 1.6 kg/m2/h, enough to stop a forecast of
  about two weeks with the same refusal as the lake.  The land bucket now
  takes the reported rain only up to the water the call removed, and `RAINC`
  carries that same number, so the maps and the water budget's evaporation
  read the convective rain that left the atmosphere rather than the scheme's
  larger report (the scheme reported 660 kg/m2 on that Colombian column and
  removed 288).  What a call reported beyond the removed water is kept as its
  diagnostic `cumulus_rain_withheld_kg_m2`, beside
  `cumulus_rain_reported_kg_m2` and `cumulus_water_removed_kg_m2`.  The
  atmosphere is unchanged.  On the GDAS 2026-09-25 12Z case the lowest
  reservoir at 120 h rises from 267 to 399 kg/m2; the global 2 m temperature
  and 500 hPa height errors against the GDAS analyses at 24, 72 and 120 h (a
  diagnostic, not the verification of record) change by less than 0.01 K
  and 0.5 m.
  Against observations on that case, the verification of record (one
  T255 case, 1 degree boxes), the four-day precipitation total over 60 S
  to 60 N falls from 1.43 to 1.06 times CMORPH's gauge-adjusted total, over
  land from 1.45 to 1.05 times the CPC gauge analysis, and over the
  contiguous United States from 1.59 to 1.17 times the MRMS multi-sensor
  total, and the equitable threat score at 10 and 25 mm rises against all
  three.  At about 2,700 to 2,950 surface stations (METAR, 12Z, 24 to
  120 h) the 2 m temperature error rises by 0.002 to 0.022 K, the 2 m
  dewpoint error moves by -0.034 to +0.195 K (the largest at 96 h, where
  the dry bias grows from 1.83 to 1.98 K) and the sea-level pressure error
  by +0.002 to +0.224 hPa; against about 470 to 540 radiosondes (12 to
  72 h) the 500 hPa height error falls by 0.02 to 0.18 m and the 850 hPa
  temperature error by 0.002 to 0.015 K.
- An inland lake's water temperature follows its own heat budget.  Open
  water held its starting skin temperature for the whole forecast.  On the
  ocean that is the analysis SST; on an inland lake it is the analysis's
  lake temperature, which is weakly observed.  GDAS carried Lake Tana (11.9
  N, 37.5 E) at 304.8 to 305.1 K at all eight cycles of 2026-09-29 and
  2026-09-30, 9.2 K above the 2 m air over it, and held for a 240 h T255
  forecast that column evaporated about 22 kg/m2 a day, several times
  published estimates of the lake's open-water evaporation (about 4 to 5 mm
  a day); the warm lakes were the columns whose surface reservoirs would
  next reach "native physics water closure exceeds the explicit surface
  reservoir", about 23 days out.  A lake column (WRF's LAKEMASK rule: open
  water whose lake share in the statics' land-use fractions beats its ocean
  share; 920 columns at T255, from the Black Sea, the Caspian and the Great
  Lakes to Tana and Turkana) now integrates its skin on the land cadence
  from the shortwave and longwave it absorbs, its own emission and the
  surface layer's sensible and latent fluxes, over the water heat capacity
  the state carries (a 10 m mixed layer), and is not cooled below 273.15 K.
  The ocean, land and sea ice are unchanged, and a run without lakes is bit
  for bit the previous runtime.  The statics carry the lake share as the new
  surface member `lake_fraction`: a checkpoint written before it reads with
  no lakes and records the absence, so it resumes and runs as it did, and a
  checkpoint written with it is refused by a reader from before it
  ("checkpoint has an unknown array namespace").  Against observations on
  the GDAS 2026-09-25 12Z case (T255, 120 h, one RTX PRO 4500, both arms
  with the two fixes above): against OISST v2.1 on the lake columns it
  covers, the lake surface temperature error at 120 h falls from 1.37 to
  1.11 K RMSE (bias +0.37 to -0.35 K), because the observed lakes cooled 1.0
  K over the five days and a held skin cannot follow;  Lake Superior 1.49 to
  0.66 K, Lake Huron 1.09 to 0.68 K, Lake Michigan 1.15 to 0.68 K, the Black
  Sea 1.22 to 0.99 K, the Caspian 1.64 to 1.39 K, while Lakes Erie and
  Ontario (1.15 to 1.51 K) and Ladoga (0.75 to 0.80 K) move the other way.
  Lake Tana's column cools from 305.3 to 299.6 K by 120 h and evaporates
  13.7 kg/m2 a day over the five days instead of 20.7.  At about 2,700 to
  2,960 surface stations (METAR, 12Z, 24 to 120 h) the 2 m temperature RMSE
  moves by -0.013 to +0.019 K, the 2 m dewpoint by -0.041 to +0.086 K and
  the sea-level pressure by -0.164 to +0.016 hPa; at the 210 to 220 of them
  within 100 km of a lake column the 2 m temperature moves by -0.025 to
  +0.171 K (the largest at 96 h) and the dewpoint by -0.031 to +0.067 K.
  Against 425 to 540 radiosondes (IGRA2, 12 to 72 h, the two arms rerun on
  one RTX 5090) the 500 hPa height RMSE moves by -0.085 to +0.059 m and the
  850 hPa temperature by -0.011 to +0.020 K.  The precipitation total
  against CMORPH (60 S to 60 N), the CPC gauges and MRMS moves by at most
  0.017 of its ratio to the observed total.

## 0.1.2

### Fixed

- A configuration has the same identity on every machine.  The hybrid
  level weights are now computed with the C library's pow and exp one
  element at a time.  With numpy 2.5 on an AVX-512 Linux machine, numpy's
  vector power and exp rounded the last bit differently, so every shipped
  configuration hashed to a different identity there than on Windows or an
  older CPU.  Every recorded identity is unchanged.
- The door bundles carry a licence notice.  Each zip now holds
  `THIRD-PARTY-LICENSES.txt`, generated from the engine workspace's
  `Cargo.lock` for that platform: every crate statically linked into the
  binaries with its licence expression, and every distinct licence and notice
  text those crates ship.  The 0.1.1 zips carried the binaries alone, while
  the MIT, BSD, Apache-2.0, ISC and Unicode licences of the crates inside them
  each ask for their notice to travel with a binary copy.  `fetch-doors`
  stages the notice beside the doors, and `tools/build_door_bundle.py pin`
  refuses a bundle without one.
- The door binaries carry no build-machine paths.  They are built with
  `--remap-path-prefix` for the engine checkout, the Cargo home and the rustup
  home.  The 0.1.1 binaries embedded about 1,100 paths under the Linux build
  host's home directory and about 1,250 under the Windows one.
  `tools/build_door_bundle.py scan` reads every member with its NUL bytes
  stripped, and `pin` refuses a bundle it finds any such path in.
- The door bundles are built from public source.  `bridges.yml` builds them
  from the engine's public `v2.8.0` tag, the first public engine release to
  carry the `rw-atms` crate, the five global observation binaries and the
  `rw_asos` and `rw_goes` subcommands this package calls.  Through 0.1.1 they
  were built from a revision no public repository carried, so nobody could
  rebuild them.  `rw_atms` keeps its source-revision stamp in that build, and
  the release no longer exempts it.
- No shipped file names a private machine or a private case.  Six comment
  lines in the carried physics (`core/gf.py`, `kernels/gf.cu`,
  `kernels/ntiedtke.cu`, `kernels/ysu.cu`), the static covariance receipt and
  its npz twin, and the ABI fast-model table named the machines figures were
  measured on; eight docstring lines in `core/landuse.py`, `core/npref.py`,
  `core/physics.py` and `core/rrtmgp.py` named one private case.  They now
  name the card, the platform or "the reference configuration".  The carried
  files are rewritten by a rule in `tools/resync_from_owner.py`, not by hand,
  and `tests/test_no_provenance.py` asserts both over everything that ships.
  The three kernels change comment text only: their `source_sha256` and the
  kernel-set digest a receipt records move, and no compiled image does.
- `NOTICE` credits Recommendation ITU-R P.676-13, whose oxygen and water
  vapour line tables `woof/globe/microwave/absorption.py` carries, and says
  what of CRTM ships: a regression fitted to CRTM v3.1.1 optical depths, the
  digests of the coefficient files that reference read, and infrared land
  emissivities averaged from CRTM's IGBP table.  CRTM v3.1.1 is dedicated to
  the public domain under CC0 1.0.
- `tools/crtm_reference/`, the CRTM reference driver the ABI operator page,
  the `abi-reference` help and the fast-model table all name, ships.  It is
  this project's Fortran calling an installed CRTM, with no CRTM code in it.
- RRTMGP no longer reads the RFMIP clear-sky input file.  Every radiation
  construction opened `rfmip-clear-sky-inputs.nc` from the engine's
  `recast-woof-data` companion for 136 numbers: sixteen trace-gas global means and
  the median layer pressure and ozone profile.  The engine stops shipping
  that file at 2.8.0, which would have made the first radiation call refuse
  on that companion.  The carried driver now reads the engine's own derived
  table through `woof.core.rrtmgp.load_trace_climatology`, pinned by
  SHA-256, and the RFMIP clear-sky oracle fetches its input through the
  engine's pinned upstream route.  The radiation input is unchanged: every
  value equals what the replaced code computed from the NetCDF, bit for bit.
- The Morrison microphysics kernel takes the four engine fixes of
  2026-09-12: vapor returned by the final condensate cleanup is stored
  instead of dropped, cloud freezing is evaluated in log space so an ordinary
  cloud slope no longer overflows and erases it, exceptional rain freezing
  stays within the joint donor budget, and an in-range number moment is no
  longer rebuilt during slope diagnosis.
- A land cell over the water soil category keeps its land use as its
  vegetation and takes silty clay loam, as real.exe matches it, instead of
  becoming mixed forest in the final reconciliation (engine fix of
  2026-09-27).  Glacier cells still take the ice soil.
- The `rw_asos` door row carries the engine's 2.8 `--abi` line (the v2
  surface record with the observation time and the one-minute route), so one
  binary built from the engine's commit answers both tables.
- Every global statics build on the 2.8 engine raised AttributeError on its
  first sector: the static builder asks the grid for a sampling handle the
  rows grid did not answer.  It does now.
- The statics cache sidecar recorded the build machine's hostname, and two
  probe tools did the same in their reports.  They record the kind of
  machine (system and architecture) instead.
- Six statics tests never ran against a published engine.  The suite's
  probe for the static builder's `rows` grid kind called the bridge with a
  signature and a spec the 2.8 engine does not accept, and read the error as
  "no rows kind".  The probe now sends this package's own rows spec, and the
  six run: the Gaussian-grid build through the crate and the statics door
  that builds the cache a run reads among them.

### Changed

- The engine range is `woof>=2.8.0,<2.9` and `recast-woof-data>=2.8.0,<2.9`.
  The floor names a breakage: the trace-gas loader, the RFMIP fetch route,
  the v2 `rw_asos` line and the static builder's `rows` grid kind are all
  first in 2.8.0.
- The engine seam is re-pinned at the published `woof 2.8.0`: 47 of 47
  files, read off the PyPI wheel and byte-identical in the Windows wheel,
  the pure wheel and the public `v2.8.0` tag.  Eighteen of the 46 files
  pinned at 2.7.3 moved by 2.8.0, and `woof/core/rfmip_upstream.py` joins
  the table because the RFMIP oracle fetches through it.  `tools/measure_boundary.py` reads 234 symbols
  across 67 modules, with no gap among the 194 a CPU host can resolve.
- The door bundles are built from the engine's public `v2.8.0` tag
  (`0164ae0f2d70`), and `door-pins.json` pins them at release `v0.1.2`.  The
  Windows doors are built for `x86_64-pc-windows-gnu`, and
  `tools/build_door_bundle.py pack --triple` resolves that bundle's licence
  notice for the target it was built for and names that target in the
  notice.
- The run manifest carries a `physics` block, as the engine's does from
  2.8.0.
- The carried-physics divergence rows are re-baselined on the published
  `woof 2.8.0`: 428 hunks, every one classified in
  `docs/CARRIED-PHYSICS-DIVERGENCE.md`.

### Measured

- The ten-step acceptance 0.1.1 owed, taken.  The T255 L40 native ten-step
  gate (semi-Lagrangian core, 300 s, the full native physics suite, GDAS
  2026-09-01 00Z, one latitude band) ran once from the model source tree at
  the revision `SOURCE.md` names and once from this package installed against
  published `woof 2.7.3`, on one RTX 4090, from one config file.  Every
  member of every checkpoint is byte-identical: 42 arrays at step 0 and 132
  at steps 5 and 10 (306 arrays, 309 members with the three metadata blobs),
  and the three checkpoints' `self_sha256` agree.  Config hash `e76cca951a57`;
  kernel-set digest in the package receipt `c1fd4a73a864`, the new value the
  comment-only kernel edits above produce (0.1.1: `9e256633abc3`).
  Measured 2026-09-26.
- The trace-gas table swap changes no forecast number.  The same ten-step
  config ran before and after the swap on one RTX 4090, against `woof
  2.7.3`, with RRTMGP in the physics suite.  All three checkpoint files
  are byte-identical, 309 members in all, including the surface and
  top-of-atmosphere radiative fluxes, the heating rates, OLR and 2 m
  temperature.  Measured 2026-09-26.
- On the published engine, from the installed wheel.  `woof global go` on
  the shipped T255 L40 native semi-Lagrangian semi-implicit experiment, cut
  to 6 h (72 steps of 300 s), from the GDAS analysis of 2026-09-01 00Z, on
  one RTX 5090 against `woof 2.8.0`, `recast-woof-data 2.8.0` and
  `cupy-cuda13x` from PyPI: receipt pass, all eight physics modules from
  this package, the engine seam proven 47 of 47, maximum wind 91.536 m/s
  and total water 710.659936 kg/m2 at 6 h, 12 pictures.  Statics built
  fresh by the engine's static builder on the rows grid give the same
  numbers as the cached statics.  The CPU test selection from the installed
  wheel on Python 3.12 against the same engine: 1,906 passed, 79 skipped,
  none failed.  Measured 2026-09-29.

## 0.1.1

### New

- `docs/CARRIED-PHYSICS-DIVERGENCE.md`, which answers "the engine changed this
  file, does this package want the change?" by reading one page. The carried
  physics and the engine's copies of the same files move independently, and
  until now nothing in this repository separated a deliberate adaptation from a
  stale copy: `SOURCE.md` says which revision was cut and
  `woof/globe/data/engine-seam.json` pins the engine files the carried code
  still reaches, and neither says which differences are on purpose. The
  document carries 82 rows over the 19 carried files that differ, one section
  per file, each row giving the carried and engine line ranges, what the common
  ancestor says, the class of the difference, the commit on the side that
  moved, what changes numerically, and a decision: pull, refuse, offer or none.
  Three lists close it: what this package owes the engine, what it has that the
  engine line may want, and what it holds a position on. Measured against the
  published `woof 2.7.3`; the rows on every file but the radiation driver, the
  physics driver, the physics inventory and the kernel loader read the same
  against any `2.7.x`.
- `woof/globe/data/engine-divergence.json`, the same 375 differences keyed by
  a fingerprint that survives a line shift: per carried file, a unified diff of
  the engine's copy against this one at zero context with the carve's rewiring
  applied to the engine side, then a SHA-256 over each hunk's engine-side lines
  and another over its carried-side lines. `tools/fingerprint_engine_divergence.py`
  computes it, documents the normalisation exactly, refuses to measure an
  engine outside the range this package declares, and records the version it
  did measure. All 35 carried files are measured, whatever the suffix, the
  four WRF parameter tables among them; a pair either side of which does not
  decode as text is compared byte for byte instead. The engine-side hash is
  what the engine line verifies a row by: read the row's engine line range out
  of the published engine, rewire, hash, compare.
- `tests/test_engine_divergence.py`, the gate. Against the installed engine it
  recomputes the fingerprints and fails, by file and by line, on a difference no
  row covers; by row name on a row whose difference is gone; on any row left
  unclassified; on an engine fix marked for pulling that the document's PULL
  list does not carry, and on a name that list carries which no row owes; on a
  row whose table in the document and whose entry in the JSON disagree about
  the file, the class, the decision or either line range; and on a carried file
  the measurement does not walk. Its two comparison nodes skip, naming both
  versions, when the installed engine is not the one the rows were measured
  against; the nodes that hold the rows, the document and the file list to each
  other need no engine. Every failure was driven before it shipped: one changed
  line in a carried module fails naming that module, one appended line in a
  parameter table fails naming the table, a deleted row fails naming the hunk
  it abandoned, a fabricated row fails naming itself, a table cell that
  disagrees with its rows fails naming the field, and a suffix filter put back
  into the walker fails naming the four tables it drops.

### Fixed

- The engine version the front page names. The index published `woof 2.7.3`
  on 2026-09-12 and it is what `woof>=2.7.0,<2.8` resolves, so `README.md`
  said the range resolved 2.7.2 while the new divergence document said 2.7.3,
  and a reader had two records of the same thing at two baselines. The page now
  names 2.7.3, carries the boundary reading measured against it (the same 231
  symbols across 65 modules, no gap among the 191 a host without a CUDA runtime
  resolves, Windows desktop 2026-09-12), and states what the engine seam does
  and does not cover. The pins have since moved to 2.7.3 as well, in the entry
  below this one, so the page, the document and the manifest all name the
  engine a fresh install resolves.

- The Noah FRZX fixture's hold on the mirror it imports. It compared the
  imported module's PATH with the path the source assertions read, so a suite
  run from an unpacked sdist against that sdist's own install -- which is how
  the release CPU selection is measured -- errored all four of its numerical
  nodes on two byte-identical copies of one file (a Linux CPU host, Python
  3.14.4, `woof 2.7.3`, 2026-09-12: 1,865 passed, 83 skipped, 96 deselected,
  4 errors). The hold is now a SHA-256 on both sides and the message prints
  both paths and both digests, so the breakage it was written for -- a
  checkout with a different revision installed beside it, source assertions
  reading one tree while the numbers came from the other -- still fires, and
  a correct install stops being refused for its path.

- The engine seam, re-pinned against `woof 2.7.3`. The pins hold one engine
  version at a time and were taken against 2.7.0; the index published 2.7.3 on
  2026-09-12 and it is what a fresh install of this release resolves, so
  `woof global doctor` printed eight files as moved and two nodes of
  `tests/test_arwen_global_engine_seam.py` failed on the only engine anybody
  gets from the index. The eight were read hunk by hunk against 2.7.0 before
  the pins moved: `config.py`, `core/state.py`, `core/preflight.py`,
  `physics_compat.py`, `physics_vertical_contract.py`, `core/microphysics.py`,
  `core/refl.py` and `core/rrtmg_legacy.py`. Of the 41 symbols this package
  imports out of them, 38 are byte-identical between the two engines and the
  three that moved change no number: `RunConfig` and `DomainState` moved in
  comment text with no field, default or validation touched, and
  `radiation_scheme_ids` narrowed a refusal, so a configuration that writes
  `ra_physics=4` beside `ra_lw_physics=4`/`ra_sw_physics=4` now resolves to the
  same `(4, 4)` pair instead of raising, while a contradicting pair is still
  refused. Reached-but-unchanged in behaviour, recorded so a later reader need
  not re-derive it: the scratch arena accepts a same-width dtype it used to
  reject and gained an optional argument nothing here passes; the microphysics
  ring-guard family is now derived from the engine's registry and grew ten
  scheme-native names behind a presence guard, on a path with no lateral
  boundary and so no door of this package enters; `core/refl.py` replaced four
  inline required-field literals with one published table whose rows are the
  same species; and four pinned files now run a registry-agreement check at
  import, which pulls `woof/physics_registry.py` and
  `woof/core/noahmp_kernel_sources.py` into the set of engine modules that
  execute when the carried physics is imported. `tools/measure_boundary.py`
  reads the same 231 symbols across 65 modules with no gap, and
  `tools/measure_engine_signatures.py` the same 28 signature rows, on 2.7.3 as
  on 2.7.2. The two nodes that compare hashes now skip, naming both versions,
  on any engine other than the pinned one, which is the rule the divergence
  gate already followed; the version ceiling stays the refusal. Three seam
  rationales quoted a line count against the engine and were re-measured
  against 2.7.3 (`config.py` 308 to 1,130, `core/state.py` 100 to 141,
  `core/rrtmg_legacy.py` 157 to 177); the first also claimed all of its
  differing lines sat inside `RunConfig`, and the measurement says 105 of
  1,130 do.

- The third-party notices this distribution owes. 0.1.0 shipped sixteen CUDA
  sources under `woof/globe/core/kernels/` and four WRF parameter tables
  verbatim, and carried no licence text for any of them: its NOTICE said
  "Nothing from that engine is copied into this distribution" and "This
  distribution ships none of that code and none of those tables", and
  `license-files` named `LICENSE` and `NOTICE` only. Five conditions went
  unperformed. The RTE+RRTMGP transcriptions (`rrtmgp_gas.cu`,
  `rrtmgp_cloud.cu`, `rrtmgp_rte.cu`, and their driver `core/rrtmgp.py` with
  the float64 mirror in `core/npref.py`) are BSD 3-Clause, whose clause 1 asks
  a source redistribution to retain the notice, the conditions and the
  disclaimer. `rrtmgp_mcica.cu` is a sixth file with the same prefix and a
  different owner: the McICA subcolumn generator the radiation is driven with
  is WRF's RRTMG generator, which is AER's work under its own BSD 3-Clause,
  and the same routine reaches `core/rrtmgp.py` and `core/npref.py` as well.
  `core/kernels/glibc_flt32.cuh` carries Arm's logf, expf, exp2f
  and powf under an MIT that asks for its copyright notice and its permission
  notice in every copy, and FDLIBM's expm1f and lgammaf reduction under a
  notice whose one condition is that it be preserved. UCAR asks that its
  notice travel with any copy of WRF, and the four tables are byte copies.
- 0.1.1 performs all five. `licenses/` holds the texts and ships in the sdist
  and, through `license-files = ["LICENSE", "NOTICE", "licenses/*"]`, in the
  wheel at `gpuwm_global-<version>.dist-info/licenses/licenses/`. The notices
  also sit beside the code: `core/kernels/LICENSE-third-party.txt` covers the
  device sources, which cannot take a comment without moving the
  `source_sha256` a run receipt records, and
  `data/noah_tables/LICENSE-WRF.txt` sits with the tables. `core/rrtmgp.py`
  and `core/npref.py` open with the notice for what they transcribe. NOTICE
  gains a section per work, each naming the carried file and the text that
  covers it. Both beside-the-code notices are carried by
  `tools/resync_from_owner.py`, so a re-cut cannot drop them, and the kernels'
  notice is narrowed by rule to the files this package actually ships rather
  than inheriting the source tree's whole-directory version.
- `rrtmgp_mcica.cu` is filed under the work that wrote it. Its own header has
  always said what it is, a transcription of `mcica_subcol_gen_sw` in WRF
  v4.6.1 `phys/module_ra_rrtmg_sw.F`, and WRF preserves AER's copyright notice
  over that module. The first pass at these notices filed it under RTE+RRTMGP
  on the strength of its filename prefix, so the distribution transcribed an
  AER work while shipping no AER text. NOTICE gains an RRTMG section naming
  the kernel, its host driver and its float64 mirror;
  `licenses/LICENSE-AER-RRTMG-BSD-3-Clause.txt` and
  `licenses/NOTICE-AER-RRTMG-as-distributed-with-WRF.txt` ship with the other
  texts; the beside-the-code notice covers it in its own section; and both
  inline Python headers carry the AER notice beside the RTE+RRTMGP one. Three
  sentences that were false went with it: the notices said all six `rrtmgp_`
  files cite the upstream commit in their headers, when three do
  (`rrtmgp_validation.cu` is this project's own input validation and
  `rrtmgp_planck_common.cuh` its own in-kernel Planck derivation).
  `tests/test_licence_notices_ship.py` now assigns every carried device source
  to the section that covers it and checks the assignment against what the
  file's own header says it transcribes, so a file cannot be filed by its
  name again.
- Said rather than left silent: the gamma routines at the end of
  `core/kernels/glibc_flt32.cuh` (`gfk_gamma_product`, `gfk_gammaf_positive`
  and `gfk_tgamma`) are this project's own work under its Apache-2.0 licence
  and not a transcription of any C library. `NOTICE` and the notice beside
  the kernels say so, and nothing in that block needs a third-party notice.
  The code of that block is unchanged since 0.1.0; the comment text in
  `glibc_flt32.cuh` and in `gf.cu` that described it was rewritten to say
  whose work it is, so the bytes of both files differ from 0.1.0's and the
  assembled-source digests a run receipt records for the `gf` and `ntiedtke`
  modules differ from the ones 0.1.0 wrote, for that reason alone. The
  `noah` and `morrison` digests move because their code moved.
- Noah's frozen-ground infiltration limiter did not limit, so a frozen column
  soaked up everything that fell on it. WRF's REDPRM builds two quantities,
  `FRZFACT = (SMCMAX/SMCREF)*(0.412/0.468)` and `FRZX = FRZK*FRZFACT`, and
  SFLX passes the second one down. The dummy argument that receives it is
  merely spelled `FRZFACT` at every level below SFLX, and SRT names it back to
  `FRZX` before spending it as `ACRT = CVFRZ*FRZX/DICE`. The carried kernel and
  the carried float64 mirror had both followed the name instead of the
  argument and passed `FRZFACT`, so `ACRT` ran `1/FRZK` = 6.67 times large, the
  `CVFRZ` series in `FCR` saturated at one, and the limit was inert. Both pass
  `FRZX` now, and the carried kernel is byte-identical to the published
  engine's copy again, which carries the same repair. Measured against the
  unmodified WRF driver over the four switch fixtures the port is graded on,
  the surface runoff distance falls from 60,641,303 ULP to 2,812, and soil
  liquid water, relative soil moisture and soil moisture go from 6,508 / 4,729
  / 1,627 ULP to exactly bitwise (RTX 3080, 2026-09-12). On the float64 mirror,
  one wet loam column at 266 K under 5 mm of rain reads FCR 0.9653 before and
  0.0837 after, and surface runoff 4.356 mm before and 4.488 mm after.
- What this moves, and where. SRT reaches the limiter only where the column's
  soil ice clears its own `DICE > 1e-2` threshold, so for an early September
  case the change lives in Antarctica, Greenland, the high Arctic and high
  terrain. Soil moisture, soil liquid water, surface runoff and the surface
  fluxes that follow them move there; a column with no soil ice is unchanged
  bit for bit. Frozen-ground columns move under this repair, so a forecast
  that reaches them is not byte-identical to one 0.1.0 wrote. No shipped page
  quotes a ten-step identity hash; the ten-step comparison of this
  distribution against the model source tree at the revision it carries, the
  309-array acceptance 0.1.0 passed, was not taken for 0.1.1 and is owed as
  the next card job.
- `tests/test_arwen_global_noah_frzx.py` holds both ends. It reads the kernel's
  three call sites and the mirror's two out of the shipped source, reads back
  SRT's own declaration and its `ACRT` line, and runs the mirror on a frozen
  column with the whole SRT infiltration path recomputed by hand from the
  arguments SRT was called with, including the arm that forces the old value
  back in and states what it was worth.
- Morrison's sedimentation stage rebuilt two quantities WRF holds still. WRF
  v4.6.1 `module_mp_morr_two_moment.F` builds them once per level inside its
  column loop and then spends them, unchanged, in the sedimentation block that
  runs after the loop closes: the cloud droplet Stokes coefficient
  `ACN(K) = G*RHOW/(18.*MU(K))` at :1438, frozen above the warm branch's small
  snow and graupel melt at :1504 and :1511 and read by the fall speeds at
  :3440-3441; and the particle-size reference density
  `DUM = PRES(K)/(287.15*T3D(K))` at :3405. That second T3D is not a
  sedimentation-time value: the column loop closes at :3332, the sedimentation
  block runs at :3358-3667, and the microphysics tendencies are applied at
  :3710 below all of it, so nothing writes T3D between the melt and the apply
  and the density :3405 reads is the one the process section's own
  reconstruction used at :1558 and :2182. The carried kernel rebuilt both from
  the temperature it held at sedimentation time, which is the post-process one,
  and the carried float64 mirror did the same. The Stokes coefficient is the
  expensive half: `d ln(ACN)/dT` is -0.288 percent per kelvin at 278 K, so both
  cloud droplet fall speeds were wrong by about that much per kelvin of the
  step's own temperature change, on every cloudy level. The reference density
  was the cheap half, stale by the entry cleanup and melt alone, about 8e-4 K
  and 2.4e-6 relative in PGAM.
- The remedy is the one WRF's own structure names: the process stage publishes
  both level quantities and the sedimentation stage consumes them. It is not
  handing the sedimentation stage its current temperature, which would have
  moved PGAM about 2,400 times further from WRF than the error it removes, in
  the wrong direction. `morr_process_level` writes `G*MRHOW/(18*mu)` from the
  pre-melt viscosity it already computes and `pressure/(287.15*temperature)` at
  the moment of its own reconstruction; `morr_terminal_velocity` takes both and
  rebuilds neither, at the Courant pass and at the sedimentation pass alike.
  The mirror follows: `_np_morrison_apply_level` returns the pair and
  `_np_morrison_fall_speeds` requires it instead of falling back to the air
  density it was handed, which is the third value it used to produce.
- `morr_bound` builds PGAM's reference density from the current temperature.
  This is the engine's own repair, taken with the engine's text, so the two
  copies of `morr_bound` agree again. It reaches the process section's
  reconstruction and, at the final call, the cloud droplet effective radius the
  radiation reads.
- Measured against the unmodified WRF driver, over the 28 oracle columns and
  10,948 compared values the port is graded on, desktop, NVIDIA GeForce RTX
  3080, 2026-09-12: values disagreeing with WRF fall from 3,554 to 3,505. Cloud
  water 228 to 195, cloud ice 333 to 327, accumulated rain 10 to 4 of 28,
  per-call rain 22 to 21, frozen fraction 12 to 9. No field gets worse, and
  every field's worst ULP distance is unchanged, so the three pinned per-platform
  signatures the model source tree holds still hold. On the float64 mirror, one
  900 hPa 285 K cloudy column that warms 4.04 K in a 60 s step reads cloud
  droplet mass-weighted fall speed 0.044494 m/s before and 0.044986 after, and
  number-weighted 0.020406 before and 0.020636 after, both 1.1 percent.
- What this moves, and where. Cloud droplet sedimentation touches every column
  that holds cloud water, so unlike the frozen-ground repair this one is not
  confined to a climate band: it moves wherever cloud water exists and the
  microphysics changes the temperature, which is every precipitating column and
  most cloudy ones.
- `tests/test_arwen_global_morrison_sedimentation.py` holds both ends, and
  ships in this distribution's own suite where the CPU selection runs it. It
  reads out of the shipped kernel source that the sedimentation routine takes
  the two published quantities and rebuilds neither, and that the process
  stage publishes the Stokes coefficient above the melt and the reference
  density below it; and it runs the carried float64 mirror on a column whose
  temperature moves during the step, checking the published pair against
  WRF's own two formulae evaluated at WRF's own two temperatures, and the
  cloud fall speeds against WRF :3440-3441, with an arm that states what the
  sedimentation-time rebuild was worth. The same test is maintained in the
  model's own source tree.

## 0.1.0

The first cut of WOOF global as a distribution of its own. Everything below
was previously reachable only from inside the engine's own checkout.

### New

- `woof global`, one console script with 48 commands: the forecast door, the
  statics builder, the assimilation door and its five ensemble legs, the render
  tape export, the regional parent bridge, the radiance operators and every
  inspection and validation leg. The engine's `woof global` reached eleven of
  them; the rest were reachable only as a module invocation.
- The machine seam, so a program drives this model the way it drives the
  engine. `woof global run-plan` reads a `gpuwm.run-plan.v1` plan envelope and
  answers `--catalog`, `--sources`, `--physics-profiles`, `--probe`,
  `--resolve` and `--estimate` as one JSON document each; `woof global
  sources --json` publishes the same source registry on a door of its own. A
  run writes `run-manifest.json` before it allocates anything,
  `run-progress.json` while it works, and an appended `events.jsonl` whose
  sequence is monotonic; a directory a previous run owns is refused by name
  rather than having a second stream interleaved into its file.
  `python -P -m woof.globe.tui_worker --job-dir DIR -- <command>` spawns any
  command through the same handshake the engine's worker uses. Every schema id
  is the engine's, so a client written for one parses the other, and a
  document that cannot answer an engine field names the field and the reason.
  `docs/ARWEN_GLOBAL_CLIENT.md`.
- The spectral core ships inside the package as `woof.globe.spectral`. No
  published engine wheel carried it, so the model could not be installed at
  all before this cut.
- Semi-Lagrangian semi-implicit core at a 300 s step is the default at every
  truncation: six-point quintic gather, order 16 hyperdiffusion at a 720 s
  e-folding time, off-centring 0.55, Lipschitz gate 0.75. The Eulerian core
  `imex_ssp3` stays selectable by name at the step its own refusal admits with
  margin (90 s at T255, 60 s at T383, 40 s at T533).
- A selectable spectral eddy viscosity as the truncation drain, derived from
  the two-point closure theory of turbulence rather than tuned:
  `[diffusion] closure = "spectral_eddy_viscosity"` reads each level's own
  kinetic energy at the cutoff every step and drains at the eddy viscosity
  that energy implies (EDQNM plateau 0.267, cusp 9.21 at decay 3.03, eddy
  Prandtl 0.6 for the scalars, five tail degrees), applied as the exact
  exponential factor per degree and per level the hyperdiffusion is applied
  as. The exponential hyperdiffusion stays the DEFAULT at every truncation,
  and the closure's six fields join a configuration's identity only when the
  closure is selected, so every record config hash is the hash it was before
  this existed. `arwen_global_gdas_t255_native_closure_24h` is the record
  T255 experiment with its `[diffusion]` table changed and nothing else. The
  theory constants are options so a sweep can measure their sensitivity; at
  their defaults nothing in it is tuned. NOT GRADED ON A CARD: the closure
  arm has no card run of its own, so it ships selectable and unmeasured
  against observations, and the drain of record is the one the graded arms
  ran with. The backscatter term of the same closure is not built.
- Native CUDA physics suite: RRTMGP radiation, Morrison microphysics,
  Grell-Freitas and New Tiedtke convection, YSU boundary layer, Noah land
  surface, MM5 surface layer, run a latitude band at a time.
- The card is priced before anything is allocated: the door estimates the
  device peak, weighs it against free VRAM, chooses the latitude band count and
  the host tier, and refuses a plan that will not fit, naming which allocation
  dies first and what the largest truncation that does fit is.
- Ensemble data assimilation: a 32-member T127 LETKF under the T255 control,
  hourly windows, tapered spectral transfer, RTPS inflation, incremental
  analysis update, and a scorecard with four assessments. Eight observation
  doors, all Rust: surface networks, radiosondes, buoys, satellite motion
  vectors, radio occultation, the WMO information system, GOES ABI and ATMS.
  It ships selectable, not default.
- `woof global fetch-analysis` fetches the one whole-globe GDAS object a
  global cold start needs, with the two non-default flags bound, and prints
  the run command it feeds. The transport is the engine's Rust fetch route.
- The six source mappings this model reads ship inside the package, and one
  resolver answers every bare id and every mapping file name: the engine's
  authority table first, the carried copies second. A published engine
  carries none of the six, so before this every shipped GDAS experiment
  refused at its first door. `woof global doctor` prints which table
  answered each row and the SHA-256 of the file that answered it, and a
  mapping both tables carry with different bytes is refused by name rather
  than chosen between. A bare name is a key into those tables, never a file
  in the working directory; a spec that names a path (a directory part, an
  absolute path, or `./name`) opens that file. Every row the resolver answers
  is asked both ways the tables spell it, the file whose name ends at the id
  and the family glob, so one spelling of a row cannot read as a gap on a
  table that carries it.
- The decode scratch is translated rather than refused. This model's decoder
  was called with a `scratch_destination`, which names where several GB of
  float64 frames are staged; a published engine places the same directory by
  `WOOF_COMPOSE_SCRATCH`. `woof.globe.mapped_source_compat` reads the
  environment, places the directory and restores it inside one lock held
  across the decode, so two decodes take turns instead of staging into each
  other, and each decode's receipt names which of the two placed the
  directory it used. A nested decode names this package's enclosing
  placement rather than crediting a caller who set nothing.
- Every product that decodes a source through the engine records how it was
  decoded, beside the mapping digest it already carried: which mechanism
  staged the decoder's multi-GB frame stream and what placed it, whether the
  installed engine's soil-only `preserve_mask` narrowing had to be adapted
  and for which records, and the engine version that did it. The cold start,
  the ATMS columns, the radiation reference cover, the surface-energy state
  product and the upper-air reference all keep the block. The surface-energy
  flux ladder runs no decode at all, since its records are
  product-definition-template 8 read through the engine's GRIB2 bridges, and
  records how its mapping document was read instead.
- A brightness-temperature pack (`gpuwm-obs.goes-bt.v1`, written by this
  package's `rw_goes bt`) reads on a published engine, whose own schema table
  stops one family short. The container parse is the engine's throughout.
- `woof global obs subscribe` opens the WMO information system feed through
  its Rust door, archiving every notification and payload with its integrity
  digest and reporting coverage per centre. `obs fetch` for that stream
  refuses and names it: no decoder writes the neutral table yet.
- The LETKF analysis runs on the card by default: batched eigendecomposition of
  the localised solve, point operators contracting a device-resident member
  stack. `--letkf-solve-path host` keeps the numpy reference for comparison.
- GOES ABI clear-sky infrared operator (bands 13 and 8 over water) and ATMS
  clear-sky over-ocean operator (channels 4 to 14), each a registered operator
  entry with an acceptance contract, with a CRTM reference leg and a trainable
  fast model.
- One-way regional parent bridge: a global run exports a parent series that
  drives a regional WOOF forecast, with a validation leg on every artefact.
- 55 configured experiments ship inside the package, from a four-step T3 numpy
  smoke to the 25 km forecast day.
- `tools/build_cli_reference.py` writes the command reference from the parser
  itself and `--check` fails when the committed page falls behind it.
- `tools/measure_boundary.py` measures which engine symbols this package uses
  and whether the installed engine carries them, so the dependency range is
  re-measured at every engine bump rather than transcribed.
- `tools/check_doc_examples.py` checks every documented command line against
  the parser this package ships, and runs in CI beside the reference check.

- The completed ensemble assimilation system: the balance package, the hybrid
  covariance at beta 0.75 over a packaged static table, localisation and
  observation errors measured on the ensemble each window, and the radiance
  streams. `woof global da fresh` with nothing set runs 32 T127 members under
  the T255 control over every stream the day carries.
- `woof global da static-covariance` estimates the static covariance table,
  and `woof global da localisation` measures the localisation length on an
  ensemble store.
- The static covariance table ships inside the package (779,912 B), so
  `--static-covariance packaged` resolves without a source tree.
- Latitude-banded native physics: the suite runs a band at a time and the whole
  advanced bundle is byte-identical to the resident run at every band count.
  The door's plan prices band count and host tier together against the card's
  free bytes.
- The ABI and ATMS radiance tables ship inside the package: the fast model, the
  operator entries and both satellites' acceptance entries.

- The physics this model was graded with travels inside the package, as
  `woof.globe.core`: eleven engine modules, the float64 mirror the
  scorecards grade against, the NVRTC loader, twelve kernels and three headers
  (629,739 B) and the four Noah and land-use tables. Nine of the eleven
  modules differ from published `woof 2.7.0`, and eight of the fifteen
  kernel files do: seven translation units and one header, measured on the
  Windows desktop 2026-09-10 with line endings normalised. `noah` and
  `morrison` match the published engine apart from the carve's import
  rewrites and are carried anyway, because their kernels are two of the eight
  and the loader binds its own directory. So a bare install now
  integrates the same bytes the grading tree did. The receipt's scheme
  identity and the cumulus refusal name those carried files by path, and
  the carve rule that produces the spelling is derived from the carve
  table rather than written out.
- `woof/globe/data/engine-seam.json` pins the 46 engine files the carried
  physics reaches and does not carry, by path, size and SHA-256 at the engine
  version they were measured against, and the suite fails when an engine
  module a carried file imports is in neither place. `woof global doctor` gains an `engine seam`
  section that hashes the installed engine's copies and reports each as proven
  or moved. Moved is a warning naming the file, and the section's summary
  row reads `note` while anything is unproven: the dependency ceiling is
  the refusal.
- `tools/pin_engine_seam.py` writes that manifest by hashing the installed
  engine, and `--check` compares without writing.
- `tools/resync_from_owner.py` cuts the carried core the same way it cuts the
  model, three ways, and its dry run merges into copies so a re-cut can be
  read before it happens. It gives a merged file the line endings the tree
  stores it with, in both directions.
- `.gitattributes` stops git converting line endings on any platform, and
  `tools/check_line_endings.py` refuses a change that rewrites them: a file
  written back under the other convention diffs whole, so the real change is
  removed and re-added with every other line and nobody reads it. The gate
  prints the CRLF and bare-LF counts on both sides of every flip in a
  revision range and walks the tree for a file that holds both at once.
- A run receipt records which physics module actually integrated, its origin
  and the SHA-256 of the file that was imported, with one digest over the
  kernel directory the loader bound. Two copies of several of these modules
  exist in one process, so a version number cannot answer it.
- A run receipt also carries the seam verdict: the engine version the pins
  were taken against, the engine version that resolved, the proven count and
  any file whose bytes moved. The module hashes cannot see the staying half
  of the physics, and one of those files reaches every carried kernel's
  assembled source.
- `recast-woof-data` is declared as a dependency of this package rather than
  inherited from the engine. Carrying the radiation driver made it an import
  requirement: the driver resolves a member of the companion wheel at module
  scope.

### Fixed

- Four `device_cache_key` imports reached beyond the package's own top level
  and made the installed wheel unimportable.
- The four authority-mapping lookups resolved a path relative to this package
  and named a directory that does not exist once the package left the engine
  tree, so every bare source id in every shipped configuration refused.
- One observation at the exact pole refused the whole analysis. Every surface
  row is evaluated through the lowest level's wind, and a lat-lon vector has no
  direction there; the public surface stream carries a station at latitude
  -90.0000, so the first cycle of a global run stopped. That row is now
  rejected by name, and nothing else is.
- A bare `render` named four products the renderer's catalogue does not carry,
  put every product in the skipped list, exited 0 and left an empty directory.
- The analysis quickstart taught two command lines the parser refuses.
- Thirty-nine written command lines were the console script's name with a
  module name after a dot, which no shell can run. They are `python -m
  arwen_global.<module>` now, and a gate holds the tree to it.
- Three published links named a repository that is not published.
- The stream fetch door reported an empty archive window as the exit code
  reserved for a missing Rust door, so a caller acted on it by restaging
  binaries that were already staged.
- The documented microwave command lines were reported as refused by the
  example checker, which reads one parser and could not see through a door
  that forwards to its own.
- The forecast command wrote no `status.json`. It is the longest-running
  command in the distribution and the only long-running one with no machine
  surface, so a workspace driving a day-long integration had a growing log to
  scrape and no answer to what stage it was in or whether it ended.
- A native forecast against a published engine was accepted, priced the card,
  allocated it, integrated its first steps and then refused its own argument
  vector inside the physics: the published surface layer takes no `vegfra`,
  the published radiation no `column_size_bounding`, and neither published
  cumulus constructor a `column_chunk`. Four refusals stood at the door
  because of it. The carve removes the cause; those four, the YSU
  mixing-length refusal beside them and every skip that cited any of the five
  are retired with it. The last row in that table was the mapped-source
  decoder's `scratch_destination`, which is a placement rather than physics
  and is now translated, so the table is empty: against a published 2.7.0
  there is no call this package makes that the installed engine refuses. The
  measurement stays reachable (`tools/measure_engine_signatures.py`) and a
  row added to the table is still refused at the door by name.
- Selecting `ysu_free_atmosphere_mixing_length = "fixed"` was refused on every
  published engine, because the mode is one more integer argument to the YSU
  kernel. The kernel travels with the package, so the option runs.
- The radiation and boundary-layer scorecards graded against whichever float64
  mirror the installed engine happened to carry, which is not the mirror of
  the kernels this package runs. They grade against the carried mirror.
- A run receipt recorded the configuration, the arithmetic pins, the physics
  identity and the machine, and nothing about the libraries. The spectral
  tables ride on a numpy routine whose bits moved between two numpy releases,
  so a receipt could not answer the first question two differing runs raise.
  It records the interpreter and every installed distribution the arithmetic
  rides on, under the self-hash.
- Two comment lines in the carried physics driver cited files in a
  development-tooling directory on the source tree. Neither path resolves in
  any install, and the source of this distribution ships with it. Both are
  rewritten by the carve rule that reproduces them, and the provenance gate
  now stops that directory name at anything that ships.
- Fourteen test modules set the never-open-the-local-device switch at import
  time. pytest imports every collected module before it runs anything, so one
  CPU-only file decided the device for the whole session: on a card host the
  device selection reported 8 failed and 4 errors, nine of those rows nothing
  but the leak, and under that noise a test that reached an engine module a
  published 2.7 does not carry rode the cut unmarked. The switch is set per
  test through `pytest.mark.cpu_only` and put back, the missing mark is on,
  and two gates read the suite's own source for either shape.
- The engine seam's stated scope read as an import closure. It is the direct
  engine imports of the carried physics plus four files, 46 rows; the closure
  is 109 modules, and the 63 the seam does not pin are reached only through
  runtimes this package's door cannot select. The module and the `doctor` row
  say so.
- A plan naming a render product the renderer does not carry resolved at exit
  0 with no warning, built the statics, integrated the whole forecast and died
  at the render stage naming neither the slug nor anything a client could act
  on. On a T255 day that is a forecast day spent to learn a spelling. Every
  token is checked against the renderer's own product list, its group keywords
  and its skip token, asked through the catalogue door rather than
  transcribed: an unknown slug is a warning in a query mode and a refusal on
  the route that starts work, the parameterized form is reported as unchecked
  rather than unknown, and a machine with no staged renderer says it could not
  ask and names what that costs. The same plan now fails in 1 s with no
  checkpoint written and no picture drawn, and the render stage's own words
  reach the machine channel instead of an exit code.
- A `go` plan with no start date resolved with `render` in its stage list and
  no warning, then finished at exit 0 with the stage silently skipped: a
  reviewer who approved it expecting imagery got a green run and an empty
  directory. Both conditions the stage checks have one spelling now, read by
  the stage and by the resolved document.
- Every `gpuwm.run-plan.resolved.v1` document stated a model top of 0.0 Pa for
  all 55 shipped experiments, because the snapshot read the surface end of the
  half-level ladder instead of its top. It reads the top, and a gate holds it
  against the TOML's own value and against the coordinate's own description on
  every shipped config that loads.
- The checkpoint count was taken from step zero on every plan and always added
  the cold state, while a restart begins at its checkpoint's step and writes no
  cold state. Two T21 restarts, from step 60 and from step 180, both promised
  five checkpoints and wrote three and one, in the resolved document, in the
  estimate's disk block and in the `expected_checkpoints` a client sizes its
  progress bar with. The step is read from the checkpoint's own metadata
  record, the cold state is counted only on a cold start, a restart whose
  checkpoint cannot be read reports no count rather than a wrong one, and the
  restart path joins the declared inputs so a missing one is refused before
  anything starts.
- `run-progress.json` named the second-to-last checkpoint at `status:
  complete` on every run by construction: the final checkpoint lands on disk
  when the writer is joined, after the last heartbeat update. The post-loop
  sweep writes the last committed path, the final model time and the final
  step through the same heartbeat the in-loop callback uses, and the client
  page states that contract.
- The physics-profile menu offered three choices no route can execute: three
  shipped configurations load only through the spectral core's own module
  entry. The document carries one row per shipped experiment naming the door
  it loads through and whether a plan can run it, measured by attempting each
  door's own load, and every profile repeats the subset a plan cannot execute.
- A shipped experiment's bare name resolved on `run` and `go` but not on
  `export`, `assimilate`, `da init`, `da analyze`, `da static-covariance`,
  `da localisation`, `abi-score` or `abi-reference --config`, each of which
  opened it as a file in the working directory. The quickstart's no-card
  rehearsal is three lines and the second is an `export`,
  so a reader with no card met `[Errno 2] No such file or directory:
  'arwen_global_moist_smoke'` on the first thing they could run. Every door
  that takes a config resolves the shipped name now, and a gate reads the
  parser rather than a list of door names, so a door added later is covered by
  existing.

### What a bare install runs

`pip install "recast-woof[gpu-cu13]"` on a CUDA 13 runtime, or
`[gpu-cu12]` on a CUDA 12 one. Measured on the Windows desktop 2026-09-10 in
three virtual environments created from scratch, one for each engine version
the declared range admits and the public index carries: `woof 2.7.0`,
`2.7.1` and `2.7.2`, with `recast-woof-data` matched to each. The reading is the
same on all three: 231 symbols across the boundary with no gap the host could
see, the two absent symbols in the last row below, and 46 of 46 seam files
proven although the pins were taken against 2.7.0. A bare install resolves
2.7.2 today.

| Leg | Where it comes from | What it needs |
|---|---|---|
| the spectral core, the dynamics, the transforms, the T3 numpy smoke, tape export and the render call | inside this package | the wheels: `woof>=2.7.0,<2.8`, `recast-woof-data` beside it, numpy, scipy, netCDF4 |
| the physics the model was graded with: radiation, cumulus, surface layer, boundary layer, land surface, microphysics, the land-use rulebook, twelve kernels and three headers, the float64 mirror, the CUDA loader and the Noah tables | inside this package, `woof.globe.core` | nothing on the engine's side; any `2.7.x` engine integrates the same bytes |
| CUDA forecast at T255 and above | CUDA/CuPy on an NVIDIA card with enough free device memory | The Metal backend covers only implemented Global forecast paths; see [Backends](../../docs/public/BACKENDS.md). |
| GRIB2 source decode, the observation front door, the static-field builder, the `wrfout` writer, the regrid and the renderer | the engine's own bundle, inside the engine's platform wheel | `woof fetch-bridges` where the wheel carries no binaries (the `py3-none-any` install) |
| the eight Rust observation binaries: `rw_asos`, `rw_igra2`, `rw_ndbc`, `rw_amv`, `rw_gnssro`, `rw_wis2`, `rw_goes`, `rw_atms` | this package's companion bundle, `woof global fetch-doors` | a published release, or `--from DIR` for a local build; every byte is verified against the SHA-256 pins inside the wheel before it is used |
| the six source mappings the shipped experiments name | inside this package, `woof/globe/data/authorities` | nothing; the engine is asked first for every row and its copy wins the day it publishes one |
| the LETKF filter core, its eigensolver kernel, the local-GPU switch, `woof/core/constants.py` and the rest of the staying half | the engine, pinned by path, size and SHA-256 in `woof/globe/data/engine-seam.json` | nothing; a file whose bytes moved is a `note` naming the file, and the `woof<2.8` ceiling is the refusal |
| `preflight.measured_free_vram_bytes` and `surface_bias.interpolate_to_tape` | absent from every published 2.7 (2.7.0, 2.7.1 and 2.7.2 read 2026-09-10) | an engine that carries them. They stop the standalone card-pricing check, which no command is wired to, and the surface-energy scorecard's regrid onto the tape. `woof global doctor` prints both and exits 1 |

The observation crates are this package's because no published engine bridge
bundle carries six of them at all, and carries the other two in an older
build without the subcommands this package calls. If the engine takes those
crates onto its own line, `woof/globe/data/door-pins.json` goes empty and
every binary resolves from `woof fetch-bridges`.
