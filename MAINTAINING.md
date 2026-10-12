# Maintaining MetalARF

How the repository is laid out since 1.0.2, how to merge a new WOOF release,
and how to cut a MetalARF release. AGENTS.md has the day-to-day rules.

## Decisions from the 1.0.2 cleanup

1.0.2 cut the tree from about 59,000 files to 17,000 without changing a
forecast. Keep these choices unless there is a reason to revisit them:

- **Rust dependencies come from crates.io.** Each bridge's `Cargo.lock` pins
  every crate by checksum. Do not `cargo vendor` into the tree again. The
  project's own crates stay vendored (`tools/*/vendor/<crate>`), and rustwx's
  pinned git dependencies stay in `tools/rustwx/vendor/crates-io`, which
  `tools/rustwx/.cargo/config.toml` points at (only those; upstream tests
  read them at that path). Some upstream messages still
  print `cargo build ... --offline`; that only works once Cargo's cache is
  warm, so the README build (`tools/build_darwin_native.py`) is the one to use.
- Two upstream tests in `tests/test_licence_notices_ship.py` that required the
  crates.io mirror (`test_simulated_radar_dependency_grants_include_native_library_notices`,
  `test_simulated_radar_writer_crates_resolve_from_the_public_source_tree`)
  were removed; drop them again if an upstream merge brings them back.
- **`recast-woof-data` comes from PyPI**, at the version `pyproject.toml`
  pins. That pin must match the upstream WOOF version in `pyproject.toml`.
- **Metal ABI headers are generated, not tracked.** `tools/build_metal_backend.py`
  regenerates `woof/metal/generated/*.metalh` before compiling. The generated
  `kernel_specs.py` and `registry.json` stay tracked because the runtime reads
  them. If a merge brings tracked `.metalh` files back as modify/delete
  conflicts, resolve them by deleting.
- **The prebuilt Metal bundle stays tracked**, so users run without Xcode.
  Its staleness check hashes every file in `woof/metal/shaders/`,
  `woof/metal/signature_fragments/`, `kernel_signatures.json`, native runtime
  sources and build/generator tools. Every input and shipped artifact is
  bound by content in the v2 manifest; adding, changing or removing an input
  needs a bundle rebuild, committed with the change. A public snapshot can
  verify these bindings without private development history.
- **CUDA stays.** It is woven through about 300 shared modules, and removing
  it would make every upstream merge conflict. The NVIDIA receipts under
  `tests/data/*advect*` also stay: CPU tests that run on a Mac pin them
  against the CUDA kernel sources, so do not edit `*.cu` files either.
- **`metalarf` is a second name for `woof`** (`[project.scripts]` and the
  `ALIASES` table in `tools/build_cli_options_doc.py`). Messages, manuals,
  `~/.woof` and the `WOOF_*` variables keep the upstream name, because a
  rename would conflict with every upstream merge.
- **`.ignore` fences search.** It hides vendored crates, fixtures, generated
  files, CUDA kernels and legal records from `rg`. Add new bulky or
  machine-written paths there rather than letting them into search results.

## Removing dead code

`tools/upstream-prune.txt` lists every upstream path MetalARF deletes. When
you remove more upstream code, add its paths there. Before deleting a module,
check all of these:

- It is unreachable from the entry points in `[project.scripts]`.
- No test, tool or user doc (`docs/public`, README) names it.
- It is not loaded dynamically: `woof.verify.cases` modules are discovered at
  runtime, and short names such as `state` hide relative imports
  (`from .state import ...`).
- For a Metal family, both its shader is absent from
  `tools/build_metal_backend.py` and its kernels are absent from
  `kernel_signatures.json`.

## Merging a new WOOF release

1. Fetch the upstream tag (`origin` is `recastsystems/woof`) and merge it
   into the integration branch.
2. Reapply the prune list:

       grep -v -e '^#' -e '^$' tools/upstream-prune.txt | git rm -r -q --sparse \
           --ignore-unmatch --pathspec-from-file=-

3. If `tools/rustwx/Cargo.lock` changed a `git+https` dependency, refresh
   that crate in `tools/rustwx/vendor/crates-io` (run `cargo vendor` into a
   temporary directory and copy just that crate).
4. Bump the `recast-woof-data` pin to the new upstream version.
5. Rebuild the native bridges and the Metal bundle, run `woof selftest` and
   the CPU tests for the touched areas, and check the CLI reference with
   `python -m tools.build_cli_options_doc --check`.

## Cutting a release

The MetalARF version lives in CHANGELOG.md and the release tag;
`pyproject.toml` keeps the upstream WOOF version. The public `main` branch is
a chain of release snapshots: each release is one commit whose parent is the
previous release. The development history is not published.

Build the snapshot from the git tree, not the working directory, because the
working directory drops tracked-but-ignored files such as the Metal bundle:

    export GIT_INDEX_FILE="$(mktemp)"
    git read-tree HEAD
    git rm -q --cached --sparse --ignore-unmatch \
        AGENTS.local.md PORT_STATE.md METAL_PORT_LEARNINGS.md
    tree=$(git write-tree); unset GIT_INDEX_FILE
    commit=$(git commit-tree "$tree" -p <previous release> -m "MetalARF X.Y.Z")
    git diff --stat <previous release> "$commit"

Review the diff, then push the commit to `main`, tag it `vX.Y.Z`, and publish
a GitHub release with the CHANGELOG section as its notes.
