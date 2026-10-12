# Metal TKE-budget refusal

[Issue #2](https://github.com/terwgr/MetalARF/issues/2) is resolved in
[MetalARF 1.0.3](https://github.com/terwgr/MetalARF/releases/tag/v1.0.3),
through incorporated [PR #13](https://github.com/terwgr/MetalARF/pull/13)
and the admission-inspector amendment. The budget has no Metal implementation;
its CUDA accumulator is not a fallback for a Metal state.

A nonzero `tke_budget` is refused by physics initialization, direct
`dycore.step` admission and `tools/metal_admission_check.py`. Set
`tke_budget=0` for supported Metal execution. This preserves the existing
TKE-1.5 dynamics route; it does not implement TKE-budget output.

From a MetalARF clone installed as described in the README:

```sh
python tools/metal_tke_admission_regression.py
```

The retained reproducer checks the unsupported `km_opt=2, tke_budget=1`
configuration at initialization and direct stepping, checks inspector
agreement, and executes one supported TKE-1.5 step with synchronous Metal
submissions. It passed during the 1.0.3 integration. Refusal checks do not
allocate a GPU state; the supported step uses the real Metal backend.

The public release is commit
`cacc3962d12b9bfe74541993c74a3d608270dcab`. This follow-up records the shipped
resolution; it does not reapply the already incorporated guard.
