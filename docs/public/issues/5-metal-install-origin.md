# Installing and identifying MetalARF

[Issue #5](https://github.com/terwgr/MetalARF/issues/5) is resolved in
[MetalARF 1.0.3](https://github.com/terwgr/MetalARF/releases/tag/v1.0.3),
through incorporated [PR #15](https://github.com/terwgr/MetalARF/pull/15).
Both the README and backend guide use the clone/build/stage/editable-install
route. Installing `recast-woof` from PyPI installs upstream WOOF, not this
MetalARF tree.

Follow the [README installation](../../../README.md#install) in a fresh
virtual environment. Keep that environment with its checkout; an environment
borrowed from another worktree can report the other worktree's package
metadata while executing different code.

Check both package origin and actual Metal execution:

```sh
python -c 'from pathlib import Path; import woof.metal; print(Path(woof.metal.__file__).resolve())'
git rev-parse HEAD
woof selftest --backend metal --json
```

The module path must resolve to this clone. A version string alone is
insufficient: MetalARF intentionally retains upstream's distribution name
and version under `MAINTAINING.md`. In an editable environment, the installed
`recast_woof-*.dist-info/direct_url.json` identifies the checkout. Public run
manifests additionally record its commit, dirty state, package path and
whether metadata was borrowed; require consistent provenance from the
intended checkout.

The exact 1.0.3 snapshot, commit
`cacc3962d12b9bfe74541993c74a3d608270dcab`, was installed into its own fresh
Python 3.12.14 environment with the verified 42-artifact native build and
shipped Metal bundle. Three forced-Metal selftests passed the Regional, Hex
and Global paths. Its public prepared forecasts recorded consistent,
non-borrowed metadata. This is clone-install evidence; it does not claim a
separate MetalARF PyPI distribution or a certified platform wheel.
