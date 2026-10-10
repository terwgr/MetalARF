# Contributing

Thanks for considering a contribution to WOOF. Two things shape how
this project accepts changes: every WRF-derived mechanism is gated
against WRF v4.6.1 evidence before it ships, and the pipeline fails
closed rather than approximating. Contributions are expected to keep
both properties.

Before investing in an implementation, open an issue for anything
broad -- a new source, physics scheme, projection, or packaging change
-- so scope and the required evidence can be agreed first.

Reporting a failure rather than proposing a change: run `woof report`
in the run directory and attach the zip it writes. It collects the
receipts, the failure, the logs, this install's identity, the card and
the free space, with machine identity redacted by class and a manifest
of what it contains printed before you send it
([reporting a problem](docs/public/REPORTING-A-PROBLEM.md)).

Keep contributions fail closed:

- never infer that a decodable product is a complete initial state;
- never substitute a projection, physics scheme, level, cadence, or
  missing-data policy silently -- refuse loudly or report the
  substitution explicitly;
- bind input and implementation authorities with sizes and SHA-256
  values;
- add a focused failure test for every new accepted control;
- distinguish structural writer tests from unchanged-stock-WRF
  execution evidence; and
- do not commit credentials, machine-specific absolute paths, private
  data, or source files whose redistribution terms are unclear.

Physics and numerics changes need evidence proportional to the claim:
a transcription change needs its oracle gate updated or extended
(never weakened); a claim of WRF agreement needs the measured
comparison, not a plausibility argument. The physics registry's
maturity labels (docs/public/PHYSICS.md) must stay truthful about what
has and has not been verified.

Practicalities:

- Apple Silicon, Python 3.11+ and a Rust toolchain. Follow the install
  steps in the README: `tools/build_darwin_native.py` builds the native
  bridges (Cargo fetches crates.io dependencies, pinned by each
  `Cargo.lock`), and `pip install -e '.[dev]'` installs the checkout with
  the `recast-woof-data` companion from PyPI. Reach packaged tables
  through `data_assets.data_path()` and never by joining onto `woof/data`.
- After changing a shader, rebuild the Metal library with
  `python tools/build_metal_backend.py --output woof/libexec/metal` and
  commit the shader and the bundle together. See AGENTS.md for how the
  Metal port is laid out.
- Run the focused tests for the changed surface first, then the broad
  CPU suite it touches. Rust changes must pass locked tests and strict
  formatting.
- A Rust change that touches arithmetic a published number is read off
  also runs the mutation gate over its own diff:

      python tools/battery/run_mutation_gate.py --since HEAD~1 --jobs 8

  It asks a question the test suites cannot: not "do the tests still
  pass" but "would they fail if this arithmetic were wrong". It tests
  only mutations inside the lines the diff changed, so a commit that
  touches no Rust costs nothing and a normal one costs a couple of
  minutes. Green means every mutation of the changed code was noticed
  by some test. Red names the mutation that was not, and the answer is
  to write that test -- `tools/battery/mutation_survivors.txt` records
  the holes that predate the gate and is not somewhere to put a new
  one. `tools/battery/mutation_gates.txt` says which packages are on
  it and why the expensive ones only report.
- GPU or real-data evidence should include the exact commit, hardware
  and runtime, argv, manifests, output hashes, timings, and a
  non-finite scan.

Contributions authored with AI assistance are welcome under the same
rules as any other: the evidence gates every change equally, and the
submitter is responsible for the claim their change makes.

Security reports: see [SECURITY.md](SECURITY.md). Do not change the
Apache-2.0 project license or any third-party license notice without
explicit maintainer authorization.
