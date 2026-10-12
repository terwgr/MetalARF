# Metal coverage and accuracy claims

[Issue #6](https://github.com/terwgr/MetalARF/issues/6) is resolved in
[MetalARF 1.0.3](https://github.com/terwgr/MetalARF/releases/tag/v1.0.3),
through incorporated [PR #15](https://github.com/terwgr/MetalARF/pull/15).
The README, changelog and backend guide distinguish execution coverage from
numerical comparisons and withdraw unsupported historical cross-model
bitwise and speedup claims.

This follow-up generates the guide's per-option selector table directly
from `METAL_FAMILY_CAPS` in `woof/core/physics.py`. It does not execute or
certify the product of those selectors. Configuration rules, the known
coupling conflicts and dynamics admission still apply.

```sh
python tools/build_metal_coverage_doc.py
python tools/build_metal_coverage_doc.py --check
```

These commands read the source registry on the CPU without importing
physics or allocating a Metal device. `--check` returns nonzero when the
published selector table needs regeneration.

| Evidence | What it establishes |
|---|---|
| Admission/preflight | The selected route passes the checked configuration gates; no physics execution is implied. |
| Finite smoke/selftest | The bounded case executed; numerical agreement is not implied. |
| NumPy/WRF reference comparison | Agreement for the stated case, fields and equality/tolerance criterion. |
| Repeated prepared forecast | Execution, evolving fields, output health and repeatability for those inputs and duration. |
| Checkpoint-resumed equality | Reproduction of the compared continuous final state for that checkpoint and case. |

The 1.0.3 [release ledger](../RELEASE-1.0.3.json), [changelog](../../../CHANGELOG.md)
and [bounded validation notes](../BACKENDS.md#metalarf-103-bounded-validation)
identify the actual comparison scope. Tiny/cold P3 FP32 equality does not
certify arbitrary profiles, all consumers of software FP64, every admitted
coupling or long-duration stability. Synthetic public DA analysis is not a
prepared radar assimilation certification. The Vermont CUDA comparison in
issue #7 remains outstanding.

The selector table is a generated availability record. The software
arithmetic contracts and individual numerical authorities remain the
record of their respective verification limits.
