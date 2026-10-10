# Agents: MetalARF

MetalARF is Recast WOOF 1.0.3 with a Metal backend, for Apple Silicon only.
Most of the tree is upstream WOOF; the port is `woof/metal/` plus the Metal
branches in the upstream drivers that call it. If `AGENTS.local.md` exists,
read it too.

## Where things are

- `woof/cli.py`: the `woof` command, also installed as `metalarf`. Models: `woof/core` (regional),
  `woof/hex` (unstructured mesh), `woof/globe` (spectral global).
- `woof/metal/`: the backend. Python drivers (`*_family.py` and friends),
  shaders in `shaders/*.metal`, one ABI fragment per kernel in
  `signature_fragments/`, the merged registry `kernel_signatures.json`, and
  `native/` (the Objective-C++ runtime).
- `woof/libexec/metal/`: the prebuilt bundle (metallibs, dylib,
  `METAL_BUNDLE.json`). Users run it without building shaders.
- Numerical references: NumPy paths in `woof/` and `tests/physics_reference/`.
  Metal results are checked against these, not against CUDA.
- `tools/`: `build_metal_backend.py`, `generate_metal_kernel_signatures.py`,
  `metal_admission_check.py`, `metal_combo_smoke.py`, `build_darwin_native.py`;
  the Rust data bridges are `tools/{grib1_bridge,rustwx,rw_wps,zarr_bridge}`.
- Upstream code we keep but do not port or edit: CUDA kernels (`*.cu`),
  `cuda_*` drivers, multi-GPU `tilestream/`. Read them when porting a kernel.

`.ignore` keeps vendored crates, test data, generated files, CUDA kernels and
legal records out of `rg` results; use `rg -uu` when you need them.

## Rules

- Keep edits to upstream files small, so new WOOF releases still merge.
  Paths MetalARF deletes from upstream are listed in
  `tools/upstream-prune.txt`; reapply that list after each upstream merge.
- Before changing the build, dependencies or repository layout, or merging
  upstream, or cutting a release, read MAINTAINING.md: it records why the
  tree looks the way it does.
- Kernels: add a fragment with
  `tools/generate_metal_kernel_signatures.py --add-fragment <frag>.json`
  (never hand-edit `kernel_signatures.json`), rebuild with
  `python tools/build_metal_backend.py --output woof/libexec/metal`, and
  commit the shader and the bundle together. The build regenerates the
  `.metalh` headers, which are not tracked.
- Kernels need `if (gid >= n) return;` tail guards. Double-precision physics
  uses the soft64 library, never float32 math.
- Metal submits asynchronously, so one green run proves nothing: rerun a
  check 3 times, or once with synchronous submits.
- Checks: `tools/metal_admission_check.py` (CPU), then
  `tools/metal_combo_smoke.py --pairwise`, `--suites` or `--case`. Run few
  GPU processes at once and put each under a timeout; the GPU is shared
  with the desktop.
- Verify through the public CLI (`woof go`, `woof hex`, `woof global run
  --metal`) and read the fields (winds, theta), not only "finite".
- Shipped code has no machine-specific paths and no fixed hardware-memory
  constants; size allocations from `context.recommended_working_set_size`.
- No scaffolding in `woof/`: probes, dumps, audits and harnesses for our
  own runs belong in `tools/` or untracked scratch space.
- Tests: `python -m pytest tests/test_<area>*.py -q` runs on the CPU;
  CUDA tests skip themselves on a Mac.
