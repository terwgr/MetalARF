"""Atomic self-hashed run receipts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .constants import RECEIPT_SCHEMA
from .pins import ACCEPTED_PINS_HASHES, PINS_HASH


def finalize_receipt(payload: dict) -> dict:
    receipt = dict(payload)
    receipt["schema"] = RECEIPT_SCHEMA
    receipt["pins_hash"] = PINS_HASH
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    receipt["self_sha256"] = hashlib.sha256(encoded).hexdigest()
    return receipt


def write_receipt(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = finalize_receipt(payload)
    encoded = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temp = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temp.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)
    return target


def check_receipt(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    self_hash = payload.pop("self_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != self_hash:
        raise ValueError("receipt self-hash mismatch")
    payload["self_sha256"] = self_hash
    if payload.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"receipt schema mismatch: {payload.get('schema')!r}")
    if payload.get("pins_hash") not in ACCEPTED_PINS_HASHES:
        raise ValueError("receipt arithmetic pin mismatch")
    return payload
