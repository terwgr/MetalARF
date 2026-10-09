"""The seam contract, held in the tree instead of at launch.

The port's physics is the engine's ``run_mpas_column_batch`` seam.  Its
published contract is two files: ``woof/core/mpas_column_batch.py`` and
``docs/mpas-seam.md``.  The adapter records their combined digest
(``MPAS_SEAM_CONTRACT_SURFACE_SHA256``) in every receipt, restart identity
and adapter contract digest, and every execution anchor this port carries
was taken on that contract.

THE BREAKAGE THIS TEST PREVENTS: the engine and the port now ship from one
commit and nothing checks the seam at launch, so a change to the column
batch or its contract document would reach every hex run with receipts
still naming the old contract and anchors taken on another one.  This test
fails in the tree that made the change, which is where it can be measured
(the x4 anchor and the regional contract decks) before the constant moves.

It reads the engine this interpreter imports, which inside one distribution
is the same commit.  With no engine importable there is nothing to hold, and
the test skips saying so.

FOLDED INTO RECAST-WOOF the same two files ship renamed: the package
directory and every module path and name in them take the new package's
spelling, and so does the engine's name in their prose.  Their
digest is therefore not the one the adapter records, although the contract
is the same one: :data:`RENAMED_CONTRACT_SURFACE_SHA256` is the digest of
exactly those files as the WOOF rename writes them from the engine whose
unrenamed digest is ``MPAS_SEAM_CONTRACT_SURFACE_SHA256``.  Derived
2026-09-29 from engine ``eb9876971`` (``45de852c...`` as published) and its
renamed tree (``62af3f78...``); 45 lines of the column batch and 13 of the
contract document differ, every one of them a name.  The WOOF text
scrub then replaced 3 em dashes in their prose,
which moved it to ``d478e699...``; no word of the contract changed.  The adapter keeps
recording the unrenamed digest, which is the contract its anchors were
taken on.  When this digest moves, the rename or the contract moved: tell
them apart by digesting the engine's own files before moving either
constant.
"""

from __future__ import annotations

import pytest

from woof.hex import engine_identity

#: The seam contract surface as the WOOF rename writes it (see the module
#: docstring for its derivation).  Only a folded tree compares against it.
RENAMED_CONTRACT_SURFACE_SHA256 = (
    "d478e699aca4c7ae69c5a9dad3b3aae390f79d106492fc02e2868ad8705a51d7"
)
FOLDED = not engine_identity.__name__.startswith("hexcore.")


def test_the_engine_seam_contract_is_the_one_this_port_records():
    root = engine_identity.installed_root()
    if root is None:
        pytest.skip("no woof is importable here, so there is no seam to hold")
    inspection = engine_identity.inspect_seam(root)
    missing = [p for p in engine_identity.CONTRACT_SURFACE_PATHS if p in inspection.absent]
    assert not missing, (
        f"the engine at {root} does not carry the seam contract files {missing}"
    )

    from woof.hex.cuda_arwen_physics_v841 import MPAS_SEAM_CONTRACT_SURFACE_SHA256

    found = engine_identity.contract_surface_sha256(root)
    expected = RENAMED_CONTRACT_SURFACE_SHA256 if FOLDED else MPAS_SEAM_CONTRACT_SURFACE_SHA256
    assert found == expected, (
        "the engine's seam contract moved: "
        f"{found} != {expected}.  "
        "woof/core/mpas_column_batch.py or docs/mpas-seam.md changed.  "
        "Re-run the regional contract decks and the x4 anchor on the new "
        "contract, then move MPAS_SEAM_CONTRACT_SURFACE_SHA256 and re-derive "
        "the self pins with tools/repin_source_tables.py --write."
    )
