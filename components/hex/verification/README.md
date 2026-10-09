# Verification assets

- `manifests/obs-referee-283.production.json` is the authoritative production
  scorecard manifest. It pins the repository base, the four selected cases, the
  arm set, metrics, uncertainty, claim rules, and the no-auto-promotion policy.
  The two control cases it once carried as `pending` were selected on
  2026-08-24 against a measured screen of the observation archive; the screen's
  rule and its numbers are in each case's `metadata.selection_basis` and in
  `docs/obs-referee.md`. The manifest carries no result: a manifest that did
  could not be corrected after a run without invalidating the
  `manifest_sha256` every run receipt pins.
- `fixtures/build_synthetic_suite.py` creates a completely offline,
  byte-deterministic four-case test. It writes outside the repository unless
  explicitly pointed into it.
- `schemas/` documents the two receipt contracts for external producers/hooks.
- `engine-verdicts/` holds the records of the retired engine pin.  Until the
  port and the engine moved into one distribution, every published `woof`
  was measured against a sixteen-file seam manifest and one exact engine was
  admitted.  `premeasure-273-20260913.json`, `repin-273-20260913.json`,
  `premeasure-274-20260915.json`, `repin-274-20260915.json`,
  `premeasure-280-20260929.json` and `repin-280-20260929.json` are those
  measurements, kept because the seam changes they record are cited from
  `docs/declared-divergences.md`.  The instrument that wrote them and the
  admitted-engine table retired with the pin; the engine a run executes is
  now measured and recorded (`woof.hex.engine_identity`), and the seam
  contract is held by `tests/test_engine_seam_contract.py`.
  Beside them, `seam-ab-*.json` are the outputs
  of `tools/measure_engine_seam_ab.py`: the engine's column-batch seam driven
  on fixed columns under two engines and compared field by field, which is
  how a re-pin says which physics number moved and by how much (the
  `2.6.5-vs-2.7.3` files are the 2.7.3 re-pin's, taken on an RTX 5070 Ti;
  the `2.7.3-vs-2.7.4` files are the 2.7.4 re-pin's and the
  `2.7.4-vs-2.8.0` files the 2.8.0 re-pin's, both taken on an RTX 5090,
  500 of 500 arrays identical on both profiles).  Their
  ``.npz`` inputs are not carried; the JSON records the SHA-256 of the columns
  that produced them and of the engine sources on each side.

The checked-in evidence under `evidence/obs-referee-283/` was intentionally
`NOT_MEASURED` (no real metric values at all) until 2026-08-25, when it was
replaced wholesale by the output of a complete provenance-pinned run. Its
`RECEIPT.md` names the chain and the SHA-256 of every instrument in it. Replace
it only the same way; never hand-edit values into it.
