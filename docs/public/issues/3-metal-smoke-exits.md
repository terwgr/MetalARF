# Metal smoke-command exit status

[Issue #3](https://github.com/terwgr/MetalARF/issues/3) is resolved in
[MetalARF 1.0.3](https://github.com/terwgr/MetalARF/releases/tag/v1.0.3),
through incorporated [PR #14](https://github.com/terwgr/MetalARF/pull/14).
Automation can now use the exit status instead of treating a printed crash
or incomplete sweep as success.

| Outcome | Exit status |
|---|---|
| Finite executed rows or expected unsupported refusals | 0 |
| Crash, NaN row or incomplete inner deadline | 1 |
| Parent process timeout | 124 |

The rule covers focused cases, the default matrix and the shared context
used by pairwise/suite modes. Parent pairwise/suite commands propagate their
child's status. An expected refusal still means that configuration did not
execute; an `ok` smoke row establishes execution, not numerical parity.

From the installed clone, reproduce the command-routing boundaries without
a GPU sweep:

```sh
python tools/metal_smoke_exit_regression.py
```

The retained check injects controlled success/refusal/crash/NaN rows, triggers
a real inner SIGALRM deadline and exercises parent timeout handling. It
passed during the 1.0.3 integration. The public release commit is
`cacc3962d12b9bfe74541993c74a3d608270dcab`; this note records that shipped fix.
