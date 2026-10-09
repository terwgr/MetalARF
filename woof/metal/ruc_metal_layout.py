"""RUC-local packed layout for the fused Metal soil-surface call.

The CUDA fusion uses 847 device pointers. This layout maps those
aliases to the 165 reusable rows recorded by the retained SFCTMP generator,
the named RUC driver rows, and one contiguous parameter-table tail. The Metal
entry points bind the packed word slab instead of a high-arity pointer list.
"""

from __future__ import annotations

from pathlib import Path
import re

from woof.core.ruc_sfctmp_layout import (
    _SFCTMP_ARRAYS, _SFCTMP_OUTPUTS, _SFCTMP_SLOTS,
)


_CORE = Path(__file__).resolve().parents[1] / "core"
_DRIVER_SOURCE = _CORE / "kernels" / "ruc_fused_driver.cuh"
_SFCTMP_SOURCE = _CORE / "ruc_fused.py"
_SFCTMP_NZS = 9


def _driver_rows() -> dict[str, tuple[int, bool]]:
    source = _DRIVER_SOURCE.read_text(encoding="utf-8")
    rows: dict[str, tuple[int, bool]] = {}
    for name, base, level in re.findall(
            r"float\* d_(\w+)\s*=\s*d_storage\+\((\d+)\+(\d+)\*RUC_NZS\)\*n",
            source):
        base_row = int(base)
        rows[name] = (base_row + int(level) * _SFCTMP_NZS,
                      base_row >= 123)
    return rows


def _value_sources() -> dict[str, str]:
    source = _SFCTMP_SOURCE.read_text(encoding="utf-8")
    start = source.index("_SF_VALUES = {")
    end = source.index("\n}\n", start) + 2
    body = source[start:end]
    # The non-literal **{name: name ...} term covers RUC_DRIVER_STALE_LOCALS;
    # every other mapping is a quoted pair in this declaration.
    return dict(re.findall(r'"([a-zA-Z0-9_]+)"\s*:\s*"([a-zA-Z0-9_]+)"', body))


DRIVER_ROWS = _driver_rows()
VALUE_SOURCES = _value_sources()
DRIVER_ROW_UNITS = 123 + 12 * _SFCTMP_NZS
SFCTMP_SCRATCH_UNITS = sum(_SFCTMP_SLOTS)
SFCTMP_BASE_UNITS = DRIVER_ROW_UNITS
SFCTMP_TABLE_NAMES = ("rstbl", "rgltbl", "z0tbl", "lemitbl", "tbq", "zshalf")

# The pointer-table driver copies each public field into its named row before
# doing any arithmetic. Metal does the same layout with buffer blits: the
# rows remain float/int typed views on one `uint32` allocation, so no kernel
# ever needs a resource per public field.
WORKSPACE_UNITS = DRIVER_ROW_UNITS + SFCTMP_SCRATCH_UNITS
INTEGER_ROWS = 5
RUN_ROWS = 1
ALIVE_ROWS = 1
INTEGER_BASE_UNITS = WORKSPACE_UNITS
RUN_BASE_UNITS = INTEGER_BASE_UNITS + INTEGER_ROWS
ALIVE_BASE_UNITS = RUN_BASE_UNITS + RUN_ROWS
TABLE_BASE_UNITS = ALIVE_BASE_UNITS + ALIVE_ROWS


def _check_layout() -> None:
    if len(_SFCTMP_ARRAYS) != 847 or len(_SFCTMP_SLOTS) != 165:
        raise RuntimeError("RUC Metal layout no longer matches the CUDA fusion; "
                           "rerun tools/ruc_fused/generate_metal_*.py")
    missing = sorted(set(VALUE_SOURCES.values()) - DRIVER_ROWS.keys())
    if missing:
        raise RuntimeError(f"RUC Metal SFCTMP rows are missing: {missing}")
    names = {binding[6:] for binding, _, _, _ in _SFCTMP_ARRAYS
             if binding and binding.startswith("value:")}
    missing = sorted(name for name in names
                     if VALUE_SOURCES.get(name, name) not in DRIVER_ROWS)
    if missing:
        raise RuntimeError(f"RUC Metal SFCTMP binding map is incomplete: {missing}")


_check_layout()


def driver_field_row(name: str) -> tuple[int, bool]:
    """Return the zero-based packed row and whether it has nine levels."""
    try:
        return DRIVER_ROWS[name]
    except KeyError as exc:
        raise KeyError(f"RUC driver field has no packed row: {name}") from exc


def driver_field_word_offset(name: str, n: int) -> int:
    """Return the word offset of a field in the per-column driver slab."""
    row, _ = driver_field_row(name)
    return row * int(n)


def sfctmp_output_word_offset(name: str, n: int) -> int:
    """Return the per-column workspace offset of one SFCTMP result array."""
    try:
        pointer_slot = int(_SFCTMP_OUTPUTS[name])
    except KeyError as exc:
        raise KeyError(f"RUC SFCTMP output has no packed slot: {name}") from exc
    location = _SFCTMP_ARRAYS[pointer_slot][3]
    if location is None:
        raise RuntimeError(f"RUC SFCTMP output {name} is not in reusable scratch")
    offset = 0
    for size in _SFCTMP_SLOTS[:int(location)]:
        offset += int(size)
    return (SFCTMP_BASE_UNITS * int(n)) + offset * int(n)


def sfctmp_aliases():
    """Return `(pointer type, offset expression)` for every CUDA alias."""
    slot_offsets = []
    offset = 0
    for size in _SFCTMP_SLOTS:
        slot_offsets.append(offset)
        offset += int(size)
    result = []
    for binding, dtype, profile, location in _SFCTMP_ARRAYS:
        pointer_type = "int" if dtype in ("int32", "bool") else "float"
        if location is not None:
            rows = SFCTMP_BASE_UNITS + slot_offsets[int(location)]
            expr = f"({rows} * p.n)"
        elif binding in SFCTMP_TABLE_NAMES:
            table_index = SFCTMP_TABLE_NAMES.index(binding)
            expr = f"(p.table_base + p.table_offset_{binding})"
        elif binding and binding.startswith("value:"):
            name = binding[6:]
            source = VALUE_SOURCES.get(name, name)
            row, is_profile = DRIVER_ROWS[source]
            expr = f"({row} * p.n)"
        elif binding == "conflx":
            expr = f"({DRIVER_ROWS['conflx'][0]} * p.n)"
        elif binding in ("ivgtyp", "iland", "nroot", "ilnb"):
            int_row = {"ivgtyp": 0, "iland": 2, "nroot": 3, "ilnb": 4}[binding]
            expr = f"(p.int_base + {int_row} * p.n)"
        else:
            raise RuntimeError(f"unmapped RUC Metal SFCTMP pointer: {binding!r}")
        result.append((pointer_type, expr))
    return tuple(result)


__all__ = [
    "ALIVE_BASE_UNITS", "DRIVER_ROWS", "DRIVER_ROW_UNITS",
    "INTEGER_BASE_UNITS", "INTEGER_ROWS", "RUN_BASE_UNITS",
    "SFCTMP_BASE_UNITS", "SFCTMP_SCRATCH_UNITS", "SFCTMP_TABLE_NAMES",
    "TABLE_BASE_UNITS", "WORKSPACE_UNITS", "driver_field_row",
    "driver_field_word_offset", "sfctmp_aliases", "sfctmp_output_word_offset",
]
