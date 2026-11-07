#!/usr/bin/env python3
"""Build a mixed-kind table witness from the pinned native RRFS Fortran source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from woof.hex.mpasphys.native_library_build import source_manifest
from woof.hex.mpasphys.rrfs.framework_init_rrfs import FrameworkInitializationRRFS, TEMPO_RADAR_BINS
from woof.hex.mpasphys.rrfs.metal_tempo_v303 import TEMPO_TABLE_WITNESS_SCHEMA, TEMPO_TABLE_WITNESS_SOURCES
from woof.hex.mpasphys.rrfs.tempo_v303 import TABLE_LAYOUT, TEMPO_FORK_PINS


_FLAGS = ("-cpp", "-Dtempo_mpas", "-DSINGLE_PRECISION", "-O2", "-ffp-contract=off",
          "-fno-unsafe-math-optimizations", "-ffree-form", "-ffree-line-length-none")


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pinned(path, row):
    if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
        raise ValueError(f"{path.name} differs from its pinned source or table bytes")
    return path


def _serializer():
    lines = [
        "program dump_tempo_tables",
        " use iso_fortran_env, only: real64",
        " use module_mp_tempo_cfgs, only: ty_tempo_cfgs",
        " use module_mp_tempo_driver, only: tempo_init",
        " use module_mp_tempo_params",
        " implicit none",
        " type(ty_tempo_cfgs) :: cfg",
        " character(len=4096) :: output, arg",
        " integer :: unit, i, flags(4)",
        " if (storage_size(0.0_wp) /= 32 .or. storage_size(0.0_dp) /= 64) error stop 'wrong TEMPO real kinds'",
        " if (command_argument_count() /= 5) error stop 'expected output and four flags'",
        " call get_command_argument(1, output)",
        " do i = 1, 4",
        "   call get_command_argument(i + 1, arg)",
        "   read(arg, *) flags(i)",
        "   if (flags(i) /= 0 .and. flags(i) /= 1) error stop 'flag must be zero or one'",
        " end do",
        " call tempo_init(aerosolaware_flag=flags(1)==1, hailaware_flag=flags(2)==1, &",
        "   ml_for_bl_nc_flag=flags(3)==1, refl10cm_from_melting_flag=flags(4)==1, tempo_cfgs=cfg)",
        " open(newunit=unit, file=trim(output), status='new', form='unformatted', &",
        "   access='stream', convert='little_endian')",
    ]
    radar = {name for name, _, _ in TEMPO_RADAR_BINS}
    for name, shape in TABLE_LAYOUT:
        count = math.prod(shape)
        if name in radar:
            lines.append(f" if (allocated({name})) then")
        if count > 1:
            lines.append(f" if (size({name}) /= {count}) error stop 'wrong {name} extent'")
        lines.append(f" write(unit) real({name}, real64)")
        if name in radar:
            lines += [" else", " if (flags(4) /= 0) error stop 'missing melting table'",
                      f" do i = 1, {count}", "   write(unit) 0.0_real64", " end do", " end if"]
    return "\n".join(lines + [" close(unit)", "end program dump_tempo_tables", ""])


def build(args):
    if len(args.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.expected_sha256):
        raise ValueError("an independent lowercase SHA-256 is required")
    manifest = source_manifest(args.pin_set)
    pins = manifest["pins"]
    if pins["rrfs-model"] not in TEMPO_FORK_PINS:
        raise ValueError("RRFS source pin differs from the retained TEMPO implementation")
    sources = []
    for name in TEMPO_TABLE_WITNESS_SOURCES:
        path = args.source / name
        sources.append(_pinned(path if path.is_file() else args.source / path.name,
                               manifest["files"][name]))
    lookup = {}
    for name, *_ in FrameworkInitializationRRFS.TEMPO_LOOKUP_FILES:
        row = manifest["tables"][name]
        _pinned(args.tables / name, row)
        lookup[name] = row["sha256"]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    words = sum(math.prod(shape) for _, shape in TABLE_LAYOUT)
    layout = json.dumps(TABLE_LAYOUT, separators=(",", ":")).encode()
    with tempfile.TemporaryDirectory(prefix="tempo-native-build-", dir=output.parent) as tmp:
        directory = Path(tmp)
        bridge = directory / "dump_tempo_tables.F90"
        bridge.write_text(_serializer())
        executable = directory / "dump_tempo_tables"
        subprocess.run([args.fortran, *_FLAGS, "-J", tmp, "-I", tmp,
                        *(str(path.resolve()) for path in sources), str(bridge), "-o", str(executable)],
                       cwd=directory, check=True)
        artifact = directory / "artifact"
        artifact.mkdir()
        blob = artifact / "tempo-tables-f64.bin"
        subprocess.run([str(executable), str(blob), *map(str, args.flags)], cwd=args.tables, check=True)
        if blob.stat().st_size != words * 8:
            raise ValueError("native TEMPO table output has a different word count")
        digest = _sha256(blob)
        if digest != args.expected_sha256:
            raise ValueError("native TEMPO table bytes differ from the supplied reference")
        version = subprocess.check_output([args.fortran, "--version"], text=True).splitlines()[0]
        receipt = {"schema": TEMPO_TABLE_WITNESS_SCHEMA, "pins": pins,
                   "flags": args.flags, "rkind": 4, "byte_order": "little",
                   "words": words, "layout_sha256": hashlib.sha256(layout).hexdigest(),
                   "sources": {name: manifest["files"][name]["sha256"] for name in TEMPO_TABLE_WITNESS_SOURCES},
                   "lookup_tables": lookup, "compiler": version, "compiler_flags": _FLAGS,
                   "host_math": "system libm", "blob": {"file": blob.name, "sha256": digest},
                   "reference_sha256": args.expected_sha256}
        (artifact / "tempo-tables.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        artifact.rename(output)
    print(json.dumps({"output": str(output), "words": words, "sha256": digest}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pin-set", choices=("rrfs", "rrfs-44c0849"), default="rrfs-44c0849")
    parser.add_argument("--fortran", default="gfortran")
    parser.add_argument("--flags", required=True, type=int, choices=(0, 1), nargs=4,
                        metavar=("AEROSOL", "HAIL", "SUBGRID", "MELTING"))
    parser.add_argument("--expected-sha256", required=True,
                        help="Require equality to a separately supplied native reference")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
