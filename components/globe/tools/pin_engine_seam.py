#!/usr/bin/env python3
"""Pin the engine files this package reaches but does not carry.

THE BREAKAGE THIS PREVENTS.  Most of the physics this model runs is now
CARRIED, under `src/arwen_global/core/`, because the published engine's copy
of it is a different scheme.  The rest is not carried, on purpose: it is
byte-identical to the tree the model was graded in, or it differs only in
code this package never enters.  That decision was measured once, against one
published engine, and then it stops being true the moment somebody publishes
a different one -- silently, because a pip resolution inside `>=2.7.0,<2.8`
prints nothing about a file's contents.

`woof.core.constants` is the sharpest case.  It is byte-identical, it stays,
and it supplies CUDA_DEFINES to the preamble of EVERY carried kernel.  A 2.7.x
that moves it changes every carried kernel's assembled source, its
source_sha256, its PTX and its floating-point contraction, with no other
signal anywhere.

So each staying file is pinned by path, size and SHA-256 at the engine version
the decision was measured against, and `woof global doctor` hashes the
installed engine's copies and reports each as proven or MOVED.  Moved is a
warning naming the file, never a refusal: the version ceiling in the
dependency pin is the refusal, and a package that stopped working because
somebody's comment changed would be worse than one that says which file it no
longer recognises.

THE HASHES ARE NEVER TYPED.  They are read off the installed engine by this
tool, which is why it exists at all.  The RATIONALE beside each row is typed,
once, here: it says what this package reaches in that file, so a reader of a
"moved" row knows whether it matters.

USAGE

    python tools/pin_engine_seam.py                 # write the manifest
    python tools/pin_engine_seam.py --check         # compare, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO.parent.parent / "woof" / "globe" / "data" / "engine-seam.json"

SCHEMA = "gpuwm.arwen-global-engine-seam/v1"

#: Engine-relative path -> what this package reaches in it.
#:
#: THE SCOPE, stated exactly, because the row this table feeds says "N/N
#: files proven" and a scope nobody wrote down is a claim nobody can check:
#: every engine file imported by code under `src/arwen_global/core/` (the
#: carried physics and its float64 mirror) that this package does not carry,
#: plus the three files outside that closure whose CONTENT this package
#: depends on -- the LETKF the assimilation runs on the engine, the local-GPU
#: switch the native suite reads, and the kernel manifest the carried loader
#: files its images in.
#:
#: NOT the door.  The rest of the package reaches the engine's front doors
#: (the run plan, the fetchers, the obs readers, the writers), and those are
#: measured by SYMBOL and by SIGNATURE, in `tools/measure_boundary.py` and
#: `tools/measure_engine_signatures.py`, because what the package depends on
#: there is an API rather than a file's bytes.  Pinning them would print a
#: moved row on every engine release and say nothing about the physics.
#: `tests/test_arwen_global_engine_seam.py` holds both halves: every engine
#: module the tree imports is carried, pinned here, or measured there.
SEAM: tuple[tuple[str, str], ...] = (
    ("woof/core/constants.py",
     "CUDA_DEFINES, the first thing every carried kernel's preamble emits, "
     "and G and CP in the flux-to-heating conversion.  The highest-value pin "
     "in this file: it reaches the assembled source of every carried kernel"),
    ("woof/certify/kernel_manifest.py",
     "record_module, called by the carried loader on every compile; it files "
     "a second, differing image under a deterministic suffix rather than "
     "replacing the first, which is what lets the carried and engine loaders "
     "coexist under one manifest root"),
    ("woof/config.py",
     "DEFAULT_COLUMN_CHUNK, radiation_enabled, radiation_scheme_ids, "
     "RunConfig and SASE_PBL_SCHEME.  RunConfig IS constructed in carried "
     "npref, twice, inside its reference-state builders; no door of this "
     "package and no test calls those.  Against the cut revision the file "
     "differs by 1,130 lines at woof 2.7.3 (308 at 2.7.0), 105 of them "
     "inside RunConfig, and RunConfig's own 2.7.0-to-2.7.3 movement is "
     "comment text with no field, default or validation changed"),
    ("woof/core/state.py",
     "DTYPE, DomainState and the constants wdm6/sase read.  The DomainState "
     "constructor and init_at_rest are reached only from the same two "
     "reference-state builders in carried npref.  Against the cut revision "
     "the file differs by 141 lines at woof 2.7.3 (100 at 2.7.0), 109 of "
     "them inside DomainState; the 2.7.0-to-2.7.3 movement is one comment "
     "in DomainState plus a scratch-arena dtype rule that widens what is "
     "accepted and an optional argument nothing here passes"),
    ("woof/core/grid.py",
     "BaseState, VerticalCoord and rebalance_hydrostatic, bound at module "
     "scope by staying state.py"),
    ("woof/core/preflight.py",
     "EXTERNAL_MARGIN_BYTES, device_memory_probe_reason, "
     "device_memory_probe_subprocess and MEASURED_LOCAL_MEMORY_PROFILE, "
     "which carried ntiedtke reads for its SM count"),
    ("woof/physics_compat.py",
     "WRF_RRTMG_TO_RTE_RRTMGP and WRF_RRTMG_TO_RTE_RRTMGP_V1, two string "
     "tokens carried rrtmgp binds"),
    ("woof/physics_vertical_contract.py",
     "outside_vertical_bounds and refuse_vertical_levels, bound at module "
     "scope by carried ysu.  Nothing in this package binds "
     "MAX_RRTMGP_LAYERS, and carried rrtmgp does not reach this file"),
    ("woof/core/kf.py",
     "_model_clock_dt, imported at module scope by carried gf"),
    ("woof/core/health_ledger.py",
     "the validation-flag ledger carried ysu and carried rrtmgp read and "
     "write.  NOT woof/core/health.py, which differs and is not reached"),
    ("woof/data_assets.py",
     "rrtmgp_data_dir and require_companion_member, called by carried rrtmgp "
     "at MODULE scope: the companion data wheel is an import requirement of "
     "the physics driver, not merely a run requirement"),
    ("woof/local_gpu.py",
     "NO_LOCAL_GPU_ENV and no_local_gpu, read by the native suite"),
    ("woof/core/mynn_radiation.py",
     "merge_mynn_bl_clouds, mynn_bl_cloud_active and wrf_itimestep; the "
     "package runs icloud_bl 0 so the merge is inert, but carried rrtmgp "
     "imports it at module scope"),
    ("woof/core/morrison_constants.py",
     "rimed_ice_constants, which is what morr_rimed_ice = 1 selects; bound "
     "at module scope by carried morrison and carried npref"),
    ("woof/core/microphysics.py",
     "MicrophysicsDiagnostics and its siblings, reached only through "
     "morrison.apply, which this package never calls"),
    ("woof/core/refl.py",
     "the reflectivity constants carried npref's mirrors read"),
    ("woof/core/rfmip_upstream.py",
     "fetch_rfmip, called by carried rrtmgp's RFMIP clear-sky oracle: the "
     "upstream commit, URL and SHA-256 the input file is fetched and verified "
     "against, since the 2.8.0 companion ships no RFMIP file"),
    ("woof/core/terrain.py",
     "bell_hill, reached only from carried npref's reference-case builders"),
    ("woof/core/wdm6_constants.py",
     "WDM6_NUMBER_SPECIES, bound at module scope by staying state.py"),
    ("woof/core/wsm6_constants.py",
     "rimed_ice_constants for the WSM6 mirrors; imports wdm6_constants and "
     "carried npref at module scope"),
    ("woof/core/sase_limits.py",
     "E_MIN, bound at module scope by staying state.py"),
    ("woof/core/jacobi_eigh.py",
     "the LETKF eigensolver.  jacobi_eigh.cu is identical too, so the one "
     "kernel the assimilation touches compiles through the ENGINE's loader "
     "and needs no carry"),
    ("woof/da/letkf.py",
     "gaspari_cohn and the LETKF entries this package's assimilation imports; "
     "the filter runs on the engine unchanged"),

    # ------------------------------------------------------- the rest of it
    #
    # The rows above were written from the engine names the carved SCHEMES
    # bind.  They were not the closure.  Twenty-four more engine files are
    # imported by code under `src/arwen_global/core/` and were in neither
    # place: not carried, not pinned.  Twelve are module-scope imports of the
    # carried physics driver, so they execute on every import of it, and six
    # of the twenty-four differ between the published engine and the tree
    # this model was graded in.  Nothing was measurably wrong, because the
    # six differ in code no door of this package enters, measured symbol by
    # symbol; but the doctor row read "N/N files proven" over a table that
    # covered half of what the carried physics imports.
    #
    # `tests/test_arwen_global_engine_seam.py` now computes the closure from
    # `tools/measure_boundary.py`'s own AST walk and fails when an engine
    # module a carried file imports is neither carried nor pinned, so the
    # next arrival is a red test rather than a silent hole.
    #
    # Reached at MODULE scope by carried physics.py, so they run on import:

    ("woof/__init__.py",
     "the engine's package init, which executes before any engine module the "
     "carried physics imports; carried rrtmgp reaches data_assets through it"),
    ("woof/core/__init__.py",
     "the engine's core package init, executed by every `from woof.core "
     "import ...` in the carried physics.  Empty at the pinned version, and "
     "anything that arrives in it runs on every carried import"),
    ("woof/core/mynn_sfclay.py",
     "MYNN_SURFACE_OUTPUTS, MynnSurfaceResult, launch_mynn_surface_layer and "
     "seed_mynn_surface_first_step.  This model runs sfclay, so they are "
     "bound at import and never called"),
    ("woof/core/noahmp_runtime.py",
     "the Noah-MP state and diagnostic inventories, its cold start and its "
     "step.  This model runs Noah, so they are bound at import and never "
     "called"),
    ("woof/core/ruc_runtime.py",
     "the RUC LSM inventories, cold start and step.  This model runs Noah, "
     "so they are bound at import and never called"),
    ("woof/core/mynn_pbl_runtime.py",
     "the MYNN PBL inventories, mynn_pbl_step and validate_mynn_tendencies.  "
     "This model runs YSU, so they are bound at import and never called"),
    ("woof/core/myjpbl.py",
     "MYJ_PBL_INOUT, myj_pbl_step and validate_myj_pbl_outputs.  This model "
     "runs YSU, so they are bound at import and never called"),
    ("woof/core/myjsfc.py",
     "MYJ_SFCLAY_INOUT, MYJ_SFCLAY_OUTPUTS and launch_myj_sfclay.  This "
     "model runs sfclay, so they are bound at import and never called"),
    ("woof/core/shinhong.py",
     "the ShinHong passenger inventory, its TKE floor and its launchers.  "
     "This model runs YSU, so they are bound at import and never called"),
    ("woof/core/sase.py",
     "the eleven SASE launchers carried physics binds at module scope.  This "
     "model runs YSU, so they are bound at import and never called"),
    ("woof/core/surface_forcing.py",
     "SURFACE_PRECIPITATION_FIELDS and SurfacePrecipitationForcing, the "
     "precipitation carrier inventory the driver binds at import"),
    ("woof/core/radiation_carriers.py",
     "the carrier-source tokens, CarrierContract, CarrierContractError and "
     "consumer_carriers, which every radiation call in this model goes "
     "through.  It differs from the graded tree by one added token "
     "(CARRIER_SOURCE_CAM_OZONE) and the two lists it appears in; every "
     "name the carried code binds is identical"),
    ("woof/ingest/soil.py",
     "NOAH_LAYER_THICKNESS_M, the four Noah layer depths every land-surface "
     "step is handed.  The file differs from the graded tree by 165 lines "
     "and the array is identical in both"),
    ("woof/verify/sase_ref.py",
     "C_K, CP_AIR, E_MIN and prandtl_blend: the SASE closure constants "
     "carried physics single-sources from the NumPy authority instead of "
     "restating.  The file differs from the graded tree by 16 lines, all of "
     "them prose, and the four values are identical"),

    # Reached inside a function, on a path this package's own door refuses or
    # does not offer.  Pinned anyway, because "this import never fires" is a
    # property of today's option validator, not of the file:

    ("woof/core/analytic_radiation.py",
     "AnalyticClearSkyRadiation, imported inside the radiation selector at "
     "ra_lw=ra_sw=90.  The native adapter admits radiation='rrtmgp' only"),
    ("woof/core/dudhia.py",
     "wrf_solar_geometry, imported inside the analytic COSZEN provider a "
     "radiation-free run needs, and DudhiaShortwaveRadiation at ra_sw=1.  "
     "This model runs RRTMGP with radiation active"),
    ("woof/core/rrtm_lw.py",
     "RRTMDudhiaRadiation, imported inside the radiation selector at "
     "ra_lw=ra_sw=1.  The native adapter admits radiation='rrtmgp' only, "
     "which is why its 83 differing lines are not this model's physics"),
    ("woof/core/rrtmg_legacy.py",
     "RRTMGLegacyRadiation, imported inside the radiation selector at "
     "ra_lw=ra_sw=4 with the legacy variant.  The native adapter admits "
     "radiation='rrtmgp' only, which is why its 177 differing lines at "
     "woof 2.7.3 (157 at 2.7.0) are not "
     "this model's physics"),
    ("woof/core/nssl2_contract.py",
     "the NSSL-2 field contract, imported inside the microphysics selector "
     "at mp_physics=18.  The native adapter admits microphysics='morrison'"),
    ("woof/core/nssl2_default_hooks.py",
     "the NSSL-2 default hooks, imported beside the contract at "
     "mp_physics=18.  The native adapter admits microphysics='morrison'"),
    ("woof/core/nssl2_runtime.py",
     "pin_absent_nssl2_fields, imported beside the contract at "
     "mp_physics=18.  The native adapter admits microphysics='morrison'"),
    ("woof/core/p3_tables.py",
     "load_lookup_table_1 and p3_table_root, imported inside carried "
     "rrtmgp's P3 effective-radius path, which only a P3 run reaches"),
    ("woof/core/mpas_column_batch.py",
     "run_mpas_column_batch, served through carried physics' module-level "
     "__getattr__ for the hexagonal line's column batch; no door of this "
     "package asks for that name"),
    ("woof/core/diagnostics.py",
     "update_diagnostics, imported inside carried npref's two "
     "reference-state builders.  Nothing in this package or its suite calls "
     "them, which is where all 26 differing lines live"),
)


def engine_root() -> Path:
    import woof

    return Path(woof.__file__).resolve().parent.parent


def engine_version() -> str:
    from importlib.metadata import version

    return version("recast-woof")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(root: Path) -> list[dict[str, object]]:
    """The installed engine's copy of every seam file, hashed.

    THE HASH IS OF THE INSTALLED BYTES, which is what a doctor row can
    actually check.  Several of these files ship CRLF in the wheel where the
    source tree has LF, so an installed hash and a content verdict are
    computed differently on purpose: the content decision ("this file is safe
    to leave on the engine") was taken on line-ending-normalized bytes, and
    the pin is taken on what is on disk.
    """

    rows: list[dict[str, object]] = []
    for relative, reached in SEAM:
        path = root / relative
        if not path.is_file():
            raise SystemExit(
                f"the installed engine has no {relative}; the seam table "
                "names a file this engine does not ship, which is a decision "
                "rather than a hash to write")
        rows.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "reached": reached,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="compare the manifest with the installed engine "
                             "and write nothing")
    args = parser.parse_args(argv)

    root = engine_root()
    version = engine_version()
    rows = measure(root)
    payload = {
        "schema": SCHEMA,
        "engine": {"distribution": "woof", "version": version},
        "files": rows,
    }

    if args.check:
        if not MANIFEST.is_file():
            print(f"{MANIFEST} does not exist")
            return 1
        stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
        by_path = {row["path"]: row for row in stored.get("files", ())}
        moved = []
        for row in rows:
            was = by_path.get(row["path"])
            if was is None:
                moved.append(f"{row['path']}: not pinned")
            elif was["sha256"] != row["sha256"]:
                moved.append(f"{row['path']}: {was['sha256'][:12]} -> "
                             f"{row['sha256'][:12]}")
        for line in moved:
            print(line)
        print(f"{len(rows) - len(moved)}/{len(rows)} proven against "
              f"woof {version}")
        return 1 if moved else 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8", newline="")
    print(f"{len(rows)} seam files pinned against woof {version} -> "
          f"{MANIFEST.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
