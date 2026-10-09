"""The forecast receipt names the engine the run executed.

THE BREAKAGE THIS PREVENTS: through 0.3.1 the receipt's top-level
``arwen_commit`` and glacier digest restated the x4 proof's own sealed
constants, so every forecast receipt named an engine commit older than the
one the run executed.  With the exact engine pin retired, the receipt takes
the commit from the measured git state, the version from the engine tree
and the contract surface from the engine's own seam files
(``woof.hex.engine_identity``), never from the proof's constants.
"""

from __future__ import annotations

from pathlib import Path

from woof.hex.drivers import run_cuda_v841_forecast as forecast


def _source() -> str:
    return Path(forecast.__file__).read_text(encoding="utf-8")


def test_the_receipt_no_longer_restates_the_proof_constants():
    source = _source()
    assert '"arwen_commit": proof.ARWEN_COMMIT' not in source
    assert "proof.ARWEN_GLACIER_COMPOSED_TU_SHA256" not in source
    assert '"arwen_contract_surface_sha256": proof.' not in source


def test_the_receipt_names_the_measured_engine():
    source = _source()
    assert '"arwen_commit": arwen_git_before.get("head")' in source
    assert "engine_identity.declared_version(arwen_checkout)" in source
    assert "engine_identity.contract_surface_sha256(" in source
