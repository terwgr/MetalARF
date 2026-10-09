# What an `evidence/...` reference means

The hex model's documents, tests and refusal messages cite measurements by
paths such as `evidence/memory-shape-20260827/` or
`evidence/restart-step16-327/`. This file says what those paths are and why
they are not beside you.

## The short version

An `evidence/<name>/` path is the **identity of a measurement**, not a file
that ships with WOOF.

Each one names a receipt: the commands that were run, the machine and the
card they ran on, the exact engine and mesh bytes they ran against, and the
numbers that came out. The name is stable, so a claim in a document and a
claim in a refusal message can point at the same measurement and mean it.

## Where they live

| surface | carries receipts? |
|---|---|
| the WOOF repository | **no**, except this file |
| the sdist (`recast_woof-*.tar.gz`) | **no** |
| the wheels | **no** |

The receipts were recorded while the hex model was developed as a project of
its own, before it was folded into WOOF, and they stay with that project's
records. WOOF ships the *contracts* they were written against, because those
are things you run: `components/hex/docs/`, `components/hex/tests/`,
`components/hex/verification/` and `components/hex/tools/`. It does not ship
the receipts: they are records of runs already made, on hardware you may not
have, and no test reads them and no command opens them.

Where the code rests on a receipt, it names that receipt byte for byte
without carrying it. Each architecture anchor in
`woof/hex/cuda_backend/arch_admission.py` records the SHA-256 of the evidence
it was admitted on (`evidence_sha256`), and the preflight line and every
forecast receipt say whether the card ran anchored, with that pin, or
unanchored.

So a document that points you at `evidence/something/` is not missing a
file. It is a citation.

## Reading a citation

Two conventions hold for every receipt a document quotes:

- **Numbers carry a tense.** A receipt records what was true on its own
  date, at its own engine pin, on its own card. Later receipts supersede
  earlier ones without editing them, because a record that gets rewritten
  is not a record. When two receipts disagree, the later date and the
  document that quotes it are the current answer.
- **A named limit is part of the result.** A receipt states what it did
  *not* separate: which arm was not run, which card was not available,
  which candidate the evidence does not distinguish. Those sentences carry
  the claim and are not hedging.

## The contracts a receipt is written against

These ship with WOOF:

- `components/hex/verification/manifests/` and
  `components/hex/verification/vertical-specs/`: the schemas and
  vertical-level contracts the init and obs-referee legs are graded on.
- `components/hex/tools/`: the drivers themselves, including the proof
  harness that verifies its own executing modules by SHA-256 before it runs.
- `components/hex/docs/source-matrix.md`: the per-source verdict table,
  which reproduces every verdict verbatim so the table reads without the
  receipts behind it.

## Asking for one

If you need a specific receipt, open an issue on the WOOF repository naming
the exact `evidence/<name>/` path you want and the claim you are checking.
Naming the claim matters: it is usually faster to point you at the
measurement that actually settles your question than at the one the document
happened to cite.
