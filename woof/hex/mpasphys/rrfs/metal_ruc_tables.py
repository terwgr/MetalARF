"""Decode the pinned native RRFS RUC parameter tables for Metal startup."""

import hashlib
from pathlib import Path

import numpy as np

from ..resources_build import TABLE_FILES
from .ruc_lsm_8c9f1a8 import TABLE_LAYOUT, TABLE_WORDS


# Standalone RUCLSM_SOILVEGPARM, with zero-filled unused slots and the
# documented integer tokens in place of the two Fortran character fields.
_WITNESS = {
    "USGS-RUC": (4, "0b878f14dfe2ead8550acdba990c92ec85da1100f99896dcbd8d239b94a4bc07"),
    "MODI-RUC": (5, "41d01d2b367319bdc60ff465a18e7441b20d16af0fdf3424752dc712de89dfaf"),
}
_VEGETATION = ("albtbl", "z0tbl", "lemitbl", "pctbl", "shdtbl", "ifortbl", "rstbl",
               "rgltbl", "hstbl", "snuptbl", "laitbl", "maxalb")
_VEGETATION_SCALARS = ("topt_data", "cmcmax_data", "cfactr_data", "rsmax_data",
                       "bare", "natural", "crop", "urban")
_SOIL = ("bb", "drysmc", "hc", "maxsmc", "refsmc", "satpsi", "satdk", "satdw", "wltsmc", "qtz")
_GENERAL = ("sbeta_data", "fxexp_data", "csoil_data", "salp_data", "refdk_data", "refkdt_data",
            "frzk_data", "zbot_data", "czil_data", "smlow_data", "smhigh_data")


def _records(directory, filename):
    raw = (directory / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TABLE_FILES[filename][2]:
        raise ValueError(f"RRFS RUC requires the pinned {filename}")
    return raw.decode("ascii").splitlines()


def _tokens(line):
    return line.replace(",", " ").split()


def _section(records, label, fields):
    first = records.index(label) + 1
    count = int(_tokens(records[first])[0])
    rows = [_tokens(line)[1:1 + len(fields)] for line in records[first + 1:first + 1 + count]]
    return count, zip(fields, zip(*rows)), records[first + 1 + count:]


def decode_pinned_ruc_tables(directory, landuse):
    """Return witness-matched little-endian adapter words for STAS-RUC soil."""
    directory = Path(directory)
    if landuse not in _WITNESS:
        raise ValueError(f"RRFS RUC has no pinned table for {landuse!r}")
    token, expected = _WITNESS[landuse]
    words = np.zeros(TABLE_WORDS, dtype="<u4")

    def put(name, values):
        spec = TABLE_LAYOUT[name]
        dtype = "<i4" if spec["type"] == "int" else "<f4"
        offset = spec["offset"]
        values = np.atleast_1d(np.asarray(values, dtype=dtype))
        if values.size > spec["words"]:
            raise ValueError(f"RRFS RUC table {name} exceeds its native extent")
        words[offset:offset + values.size] = values.view("<u4")

    vegetation = _records(directory, "VEGPARM.TBL")
    count, fields, remaining = _section(vegetation, landuse, _VEGETATION)
    put("lucats", count)
    put("lutype", token)
    for name, column in fields:
        put(name, column)
    for name, line in zip(_VEGETATION_SCALARS, remaining[1::2]):
        put(name, _tokens(line)[0])

    soil = _records(directory, "SOILPARM.TBL")
    count, fields, _ = _section(soil, "STAS-RUC", _SOIL)
    put("slcats", count)
    put("sltype", 6)
    for name, column in fields:
        put(name, column)

    general = _records(directory, "GENPARM.TBL")
    count = int(_tokens(general[2])[0])
    put("slpcats", count)
    put("slope_data", [_tokens(line)[0] for line in general[3:3 + count]])
    for name, line in zip(_GENERAL, general[4 + count::2]):
        put(name, _tokens(line)[0])

    if hashlib.sha256(words.tobytes()).hexdigest() != expected:
        raise ValueError("RRFS RUC decoded words differ from the pinned native witness")
    return words
