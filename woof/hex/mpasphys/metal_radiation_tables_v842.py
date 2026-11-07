"""Authenticated native radiation records for Metal coefficient initialization.

The host reads the pinned MPAS/WRF Fortran records with the existing record
reader and declaration schema. No g-point reduction or table arithmetic runs
here. This intentionally replaces the CUDA route's Rust transport executable;
it does not reproduce or claim its serialized-output certificate.
"""
from __future__ import annotations
import hashlib
import numpy as np
from woof.ingest import rrtmg_coeffs as records
from .table_resources_v842 import certified_path


def load_raw_tables(spectrum, resources=None):
    if spectrum not in ("lw", "sw"):
        raise ValueError("native radiation spectrum must be lw or sw")
    suffix = spectrum.upper()
    filename = "RRTMG_" + suffix + "_DATA"
    path = (records.DATA_DIR / filename if resources is None
            else certified_path(resources["table"]))
    expected = getattr(records, filename + "_SHA256")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("native radiation table differs from the pinned source asset")
    schema = getattr(records, "_" + suffix + "_SPEC")
    raw = {}
    with path.open("rb") as stream:
        for number, (module, (fields, _)) in enumerate(schema.items(), 1):
            payload = records._read_fortran_record(stream, filename, number)
            size = 4 * sum(int(np.prod(shape)) if shape else 1
                           for _, _, shape, _ in fields)
            if len(payload) != size:
                raise ValueError(f"native radiation record {number} has the wrong extent")
            arrays = {}; offset = 0
            for name, kind, shape, _ in fields:
                count = int(np.prod(shape)) if shape else 1
                value = np.frombuffer(payload, dtype=">" + kind,
                                      count=count, offset=offset).astype(kind)
                arrays[name] = value.reshape(shape if shape is not None else (), order="F")
                offset += count * 4
            raw[module] = arrays
        if stream.read(1):
            raise ValueError("native radiation table has unexpected trailing records")
    return raw
