#!/usr/bin/env python3
"""Build and stage selected WOOF native artifacts for Apple Silicon.

This local helper uses the existing locked Cargo workspaces and stages
artifacts under their existing Unix filenames. ``--artifact NAME`` builds one
artifact; ``--all`` builds the complete declared Darwin bridge set into the
per-artifact layout accepted by ``stage_wheel_bridges.py --from-native-dir
DIR``. The batch mode shares one Cargo target cache across artifacts. It does
not update release pins or package a partial bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from woof import bridge_assets  # noqa: E402

WORKSPACE = ROOT / "tools" / "grib1_bridge"
RUSTWX_WORKSPACE = ROOT / "tools" / "rustwx"
MAPPED_WORKSPACE = ROOT / "tools" / "rw_wps"
TRIPLE = "aarch64-apple-darwin"
ARTIFACT_MARKERS = {
    "gdt101_remap": b"arwen.gdt101-regional-remap.v1",
    "grib2_inventory": b"minimum\tmaximum\t" \
                        b"ensemble_type\tensemble_size\tderived_forecast\tpv",
    "grib2_dump": b"parameter\tcenter\tsubcenter\tmaster_table_version\t" \
                   b"local_table_version\tlevel_type",
    "grib1_bridge": b"usage: grib1_bridge INPUT.grb OUTPUT_DIR",
    "gfs_grib2_bridge": b" must be HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]",
    "hrrr_grib2_bridge": b"usage: hrrr_grib2_bridge WRFNAT_F00 WRFNAT_F01 SOIL_F00 " \
                         b"SOIL_F01 OUTPUT_DIR EXPECTED_CYCLE I_START I_END J_START J_END",
    "gpuwm_preprocess_cpu": b"gpuwm_masked_nearest_f32",
    "rw_fetch": b"usage: rw_fetch <fetch|probe|latest> [OPTIONS]",
    "rw_asos": b"usage: rw_asos",
    "rw_compare": b"rw_compare --store-root DIR --out-dir DIR --reference NAME[,NAME...]",
    "rw_verify": b"rw_verify --request REQUEST.json --json RECEIPT.json",
    "rw_simradar": b"rw_simradar --request REQUEST.json schema=simulated-radar.request/v1",
    "rw_nexrad": b"usage: rw_nexrad <list|fetch|live-list|live-fetch|decode|verify|sites> [OPTIONS]",
    "rw_wrfbatch": b"rw_wrfbatch --help | --abi",
    "rw_netcdf": b"usage: rw_netcdf inventory FILE",
    "rw_stage4": b"usage: rw_stage4 <list|fetch|decode|grid|verify> [OPTIONS]",
    "rw_mrms": b"usage: rw_mrms <list|nearest|fetch|decode|grid|verify> [OPTIONS]",
    "rw_odim": b"gpuwm-obs.odim-pack.v1",
    "rw_opera": b"usage: rw_opera <coverage|nearest|fetch|decode|grid|verify> [OPTIONS]",
    "rw_ndbc": b"usage: rw_ndbc <fetch|table|verify> [OPTIONS]",
    "rw_igra2": b"usage: rw_igra2 <list|fetch|table|verify> [OPTIONS]",
    "rw_amv": b"usage: rw_amv <list|fetch|table|verify> [OPTIONS]",
    "rw_gnssro": b"usage: rw_gnssro <cdaac-fetch|list|fetch|table|verify> [OPTIONS]",
    "rw_atms": b"usage: rw_atms",
    "rw_mpas_mesh": b"rw_mpas_mesh --out GRID.nc [--spec SPEC.json | --background-km KM | --from-centres GRID.nc]",
    "rw_mpas_static": b"rw_mpas_static --grid GRID.nc --out STATIC.nc",
    "rw_mpas_init": b"rw_mpas_init --met MET --static STATIC.nc --capsule CAPSULE.nc",
    "rw_mpas_geometry": b"rw_mpas_geometry --protocol hex-geometry-v1",
    "rw_mpas_hostprep": b"rw_mpas_hostprep --protocol hex-hostprep-v1",
    "rw_mpas_convert": b"gpuwm-rw-mpas-convert-v1\tCONVERTED\tWINDOW\tWEIGHTS\tFINISHED",
    "rw_mpas_lbc": b"rw_mpas_lbc --grid INIT.nc --out-dir DIR " \
                   b"--start-time YYYY-MM-DD_HH:MM:SS --stop-time YYYY-MM-DD_HH:MM:SS",
    "rw_mlexport": b"rw_mlexport --request REQUEST.json schema=ml-export.request/v1 " \
                    b"modes=run,append,finalize progress=jsonl",
    "rw_goes": b"usage: rw_goes <list|fetch|cwp|cloud-top|bt|colocate|quicklook|forward|verify> [OPTIONS]",
    "rw_wis2": b"usage: rw_wis2 <subscribe|probe> [OPTIONS]",
    "gpuwm_mapped_engine": b"gpuwm-mapped-engine-abi frameset=gpuwm-mapped-frameset-v1 " \
                           b"height-interfaces=1 grib2-drt=0,2,3,4,40,41,42,50,51,61,200",
    "netcdf_writer": b"gpuwm_ncwrite_scan_nonfinite",
    "static_fields": b"gpuwm_static_sampling_portable_v1",
    "obs_regrid": b"gpuwm_obsregrid_build_plan",
    "obs_score": b"gpuwm_obsscore_abi_version",
    "rw_isobaric": b"gpuwm_isobaric_abi_version",
    "region_global_dealias": b"bw_dealias_rift_v1",
    "rw_zarr": b"http transfer failed after ",
}
LIBRARY_ARTIFACTS = {
    "gpuwm_preprocess_cpu", "netcdf_writer", "static_fields", "obs_regrid",
    "obs_score", "rw_isobaric", "region_global_dealias",
}
VENDORED_ARTIFACTS = {"region_global_dealias"}
VENDORED_SOURCE_COMMITS = {
    "region_global_dealias": "a7d4baf6b8a11ca5602fe44a533efd8200ef6cea",
}
PACKAGES = {
    "rw_fetch": "rw-fetch",
    "rw_compare": "rw-wrfbatch",
    "rw_verify": "rw-wrfbatch",
    "rw_simradar": "rw-simradar",
    "rw_asos": "rw-obs",
    "rw_nexrad": "rw-nexrad",
    "rw_stage4": "rw-obs",
    "rw_mrms": "rw-obs",
    "rw_odim": "rw-odim",
    "rw_opera": "rw-obs",
    "rw_ndbc": "rw-obs",
    "rw_igra2": "rw-obs",
    "rw_amv": "rw-obs",
    "rw_gnssro": "rw-obs",
    "rw_atms": "rw-atms",
    "rw_mpas_mesh": "rw-mpas",
    "rw_mpas_static": "rw-mpas",
    "rw_mpas_init": "rw-mpas",
    "rw_mpas_geometry": "rw-mpas",
    "rw_mpas_hostprep": "rw-mpas",
    "rw_mpas_convert": "rw-mpas",
    "rw_mpas_lbc": "rw-mpas",
    "rw_mlexport": "rw-mlexport",
    "rw_zarr": "rw-zarr",
    "rw_goes": "rw-goes",
    "rw_wis2": "rw-obs",
    "rw_wrfbatch": "rw-wrfbatch",
    "rw_netcdf": "rw-netcdf",
    "gpuwm_mapped_engine": "mapped-engine",
    "netcdf_writer": "netcdf-writer",
    "static_fields": "static-fields",
    "obs_regrid": "obs-regrid",
    "obs_score": "obs-score",
    "rw_isobaric": "rw-isobaric",
}
ARTIFACT_WORKSPACES = {name: RUSTWX_WORKSPACE for name in PACKAGES
                       if name != "gpuwm_mapped_engine"}
ARTIFACT_WORKSPACES["gpuwm_mapped_engine"] = MAPPED_WORKSPACE
ARTIFACT_WORKSPACES["region_global_dealias"] = ROOT / "tools" / "region_global_dealias"
ARTIFACT_WORKSPACES["rw_zarr"] = ROOT / "tools" / "zarr_bridge"
TOOLCHAINS = {
    "rw_zarr": "1.94.0-aarch64-apple-darwin",
    "rw_simradar": "1.94.0-aarch64-apple-darwin",
}


def run(label: str, command: list[str], *, cwd: Path, env: dict[str, str], log) -> str:
    log.write(f"$ {label}: {' '.join(command)}\n")
    log.flush()
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write(result.stdout)
    log.flush()
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit {result.returncode}; see {log.name}")
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path,
                        help="directory for the artifact, manifest, and build log")
    output.add_argument("--output-root", type=Path,
                        help="root for the stage_wheel_bridges layout; the "
                             "artifact is written under DIR/<artifact>")
    parser.add_argument("--cargo-target-dir", type=Path,
                        help="reuse an existing Cargo target directory")
    build_kind = parser.add_mutually_exclusive_group()
    build_kind.add_argument("--artifact", choices=tuple(ARTIFACT_MARKERS),
                            default="gdt101_remap",
                            help="one declared bridge artifact (default: gdt101_remap)")
    build_kind.add_argument("--all", action="store_true",
                            help="build every declared Darwin bridge into DIR/<artifact>")
    args = parser.parse_args()
    if args.all:
        if args.output_dir is not None:
            parser.error("--all requires --output-root, not --output-dir")
        output_root = args.output_root.resolve()
        shared_target = (args.cargo_target_dir or
                         output_root / "cargo-target").resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        for artifact_name in ARTIFACT_MARKERS:
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--output-root", str(output_root),
                "--cargo-target-dir", str(shared_target),
                "--artifact", artifact_name,
            ]
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                return result.returncode
        return 0
    artifact = args.artifact
    workspace = ARTIFACT_WORKSPACES.get(artifact, WORKSPACE)
    package_args = (["-p", PACKAGES[artifact]] if artifact in PACKAGES else [])
    output = (args.output_dir if args.output_dir is not None
              else args.output_root / artifact).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cargo_filename = (f"lib{artifact}.dylib" if artifact in LIBRARY_ARTIFACTS
                      else artifact)
    staged_filename = cargo_filename
    # Invalidate only this artifact's previous staged result before starting.
    # A failed rebuild must not leave a stale binary/receipt pair that the
    # wheel stager could mistake for the current attempt.
    for stale in (output / staged_filename,
                  output / f".{staged_filename}.pending",
                  output / "artifact.json"):
        stale.unlink(missing_ok=True)
    target_dir = ((args.cargo_target_dir or output / "cargo-target").resolve())
    log_path = output / "build.log"
    env = os.environ.copy()
    for name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS",
                 "MACOSX_DEPLOYMENT_TARGET"):
        env.pop(name, None)
    env.update(
        CARGO_TARGET_DIR=str(target_dir),
        # macOS 27 dyld rejects dylibs stripped by rustc/llvm-objcopy because
        # their LC_SYMTAB string pool may be only 4-byte aligned. Keep this
        # Darwin release build unstripped; we use Apple's strip below after
        # staging. The workspace default remains in force for other builds.
        CARGO_PROFILE_RELEASE_STRIP="none",
        # Keep the deployment floor scoped to Cargo's Apple-Silicon target;
        # global RUSTFLAGS/target env can poison host proc-macro loading.
        CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS=(
            "-C link-arg=-mmacosx-version-min=14.0"),
    )
    if artifact in TOOLCHAINS:
        env["RUSTUP_TOOLCHAIN"] = TOOLCHAINS[artifact]

    with log_path.open("w", encoding="utf-8") as log:
        try:
            if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
                raise RuntimeError("run this helper on an Apple Silicon Mac")
            artifact_record = next(
                item for item in bridge_assets.BUNDLED_ARTIFACTS
                if item.name == artifact)
            input_paths = bridge_assets.NATIVE_BUILD_INPUTS[artifact_record.crate]
            dirty_inputs = run(
                "native source status",
                ["git", "status", "--porcelain", "--untracked-files=all",
                 "--", *input_paths], cwd=ROOT, env=env, log=log)
            if dirty_inputs:
                raise RuntimeError(
                    "native build inputs are modified or untracked; refusing "
                    "to stamp checkout HEAD onto bytes built from a different "
                    "tree:\n" + dirty_inputs)
            revision = run("checkout revision", ["git", "rev-parse", "HEAD"],
                           cwd=ROOT, env=env, log=log)
            rustc = run("rustc version", ["rustc", "-Vv"], cwd=workspace, env=env, log=log)
            target_args = (["--lib"] if artifact in LIBRARY_ARTIFACTS
                           else ["--bin", artifact])
            cargo_command = [
                "cargo", "build", "--locked", "--release",
                "--target", TRIPLE, *package_args, *target_args,
            ]
            run("Cargo release build", cargo_command, cwd=workspace, env=env, log=log)

            built = target_dir / TRIPLE / "release" / cargo_filename
            staged = output / staged_filename
            if not built.is_file():
                raise RuntimeError(f"Cargo succeeded but artifact is missing: {built}")
            architectures = run("Mach-O architecture", ["lipo", "-archs", str(built)],
                                cwd=workspace, env=env, log=log)
            if "arm64" not in architectures.split():
                raise RuntimeError(f"built executable does not contain arm64: {architectures}")
            source_payload = built.read_bytes()
            source_sha256 = hashlib.sha256(source_payload).hexdigest()
            minimum_os = bridge_assets.macho_arm64_minimum_os(
                source_payload, label=f"{artifact}: {built.name}")
            expected_minimum_os = (*bridge_assets.DARWIN_MINIMUM_OS, 0)
            if minimum_os != expected_minimum_os:
                raise RuntimeError(
                    f"built artifact declares minimum macOS {minimum_os!r}; "
                    f"expected {expected_minimum_os!r}")
            marker = ARTIFACT_MARKERS[artifact]
            if marker not in source_payload:
                raise RuntimeError(f"staged executable lacks its declared {artifact} contract marker")
            if run("checkout revision after build", ["git", "rev-parse", "HEAD"],
                   cwd=ROOT, env=env, log=log) != revision:
                raise RuntimeError("checkout revision changed during native build")
            if artifact not in VENDORED_ARTIFACTS:
                revision_marker = b"GPUWM_BRIDGE_SOURCE_REV=" + revision.encode("ascii")
                if revision_marker not in source_payload:
                    raise RuntimeError("staged executable source-revision stamp does not match checkout HEAD")
            pending = output / f".{staged_filename}.pending"
            shutil.copy2(built, pending)
            pending.chmod(0o755)
            install_id = None
            if artifact in LIBRARY_ARTIFACTS:
                install_id = f"@rpath/{staged_filename}"
                run("normalize dylib install ID",
                    ["install_name_tool", "-id", install_id, str(pending)],
                    cwd=workspace, env=env, log=log)
            # Apple's strip keeps the Mach-O string table aligned on macOS 27;
            # rustc's profile strip uses llvm-objcopy and can make dylibs that
            # dyld refuses to load. Apply it to the staged copy only.
            run("strip debug symbols", ["strip", "-S", str(pending)],
                cwd=workspace, env=env, log=log)
            payload = pending.read_bytes()
            minimum_os = bridge_assets.macho_arm64_minimum_os(
                payload, label=f"{artifact}: {staged_filename}")
            expected_minimum_os = (*bridge_assets.DARWIN_MINIMUM_OS, 0)
            if minimum_os != expected_minimum_os:
                raise RuntimeError(
                    f"staged artifact declares minimum macOS {minimum_os!r}; "
                    f"expected {expected_minimum_os!r}")
            if marker not in payload:
                raise RuntimeError(f"staged executable lost its {artifact} marker during staging")
            if artifact in VENDORED_ARTIFACTS:
                pass
            elif revision_marker not in payload:
                raise RuntimeError("staged executable lost its source-revision stamp during staging")
            if install_id is not None:
                install_ids = bridge_assets.macho_dylib_ids(
                    payload, label=f"{artifact}: {staged_filename}")
                if install_ids != (install_id,):
                    raise RuntimeError(
                        f"staged dylib install ID is {install_ids!r}; "
                        f"expected {(install_id,)!r}")
            os.replace(pending, staged)
            manifest = {
                "schema": "woof-darwin-native-artifact-v1",
                "platform": "darwin-arm64",
                "target": TRIPLE,
                "minimum_os": ".".join(
                    map(str, bridge_assets.DARWIN_MINIMUM_OS)),
                "artifact": artifact,
                "filename": staged_filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "cargo_sha256": source_sha256,
                "source_revision": None if artifact in VENDORED_ARTIFACTS else revision,
                "checkout_revision": revision,
                "vendored": artifact in VENDORED_ARTIFACTS,
                "vendored_source_commit": VENDORED_SOURCE_COMMITS.get(artifact),
                "dylib_install_id": install_id,
                "release_strip": "none (Cargo), strip -S (Apple cctools)",
                "cargo": " ".join(cargo_command),
                "target_rustflags": env["CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS"],
                "rustc": rustc.splitlines()[0] if rustc else "",
            }
            (output / "artifact.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(manifest, indent=2))
            print(f"Full build output: {log_path}")
        except Exception as exc:
            print(f"build_darwin_native: {exc}; full output: {log_path}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
