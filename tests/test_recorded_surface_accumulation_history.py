"""Compare actual published accumulation fields with their native producer contract."""
import os
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize("field", ["ACHFX", "ACLHF", "ACGRDFLX"])
def test_recorded_surface_accumulation_keeps_native_producer_words(field):
    history = os.environ.get("WOOF_EXACT_HISTORY")
    reference = os.environ.get("WOOF_EXACT_REFERENCE")
    if not (history and reference):
        pytest.skip("set actual emitted history and native reference")
    from woof.verify_exact.recording import load_run_record
    from woof.verify_exact.replay import read_frame
    from woof.verify_exact.compare import field_digest
    stock = load_run_record(Path(reference))
    actual = read_frame(Path(history))
    step = int(os.environ.get("WOOF_EXACT_STEP", "1"))
    metadata = stock.fields[field]
    assert actual[field].shape == metadata.shape
    assert actual[field].dtype == np.dtype(metadata.dtype)
    assert field_digest(actual[field], metadata.dtype, metadata.shape) == stock.digests[step][field]
