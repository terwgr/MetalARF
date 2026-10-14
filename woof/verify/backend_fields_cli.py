"""CPU-only public door for registered saved-field backend comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _run(args: argparse.Namespace) -> int:
    from woof.verify.backend_fields import verify_fields

    report = verify_fields(
        args.case, args.candidate_dir, args.reference_dir, args.registration,
        args.candidate_config, args.candidate_initializer)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True,
                                         allow_nan=False) + "\n")
    print(f"verdict {report['verdict']}")
    print(f"coverage {report['coverage']}")
    print(f"comparisons {len(report['comparisons'])}")
    for reason in report["blockers"]:
        print(f"blocked: {reason}")
    failed = [row for row in report["comparisons"] if row["verdict"] == "fail"]
    for row in failed[:8]:
        print(f"fail {row['field']}@{row['lead_seconds']}s "
              f"nrms={row['normalized_rms']} rms_delta={row['rms_delta']}")
    print(f"report {args.out_report}")
    return {"pass": 0, "fail": 1, "blocked": 2}[report["verdict"]]


def register_cli(subparsers) -> None:
    command = subparsers.add_parser(
        "verify-fields", help="gate complete saved backend fields against a pinned reference")
    command.add_argument("case", choices=("gabls1", "cbl_dry", "igw", "hill2d"),
                         help="verification case whose saved outputs are compared")
    command.add_argument("--candidate-dir", type=Path, required=True, metavar="DIR",
                         help="directory containing candidate case outputs")
    command.add_argument("--reference-dir", type=Path, required=True, metavar="DIR",
                         help="directory containing pinned reference outputs and provenance")
    command.add_argument("--registration", type=Path,
                         help="independently declared and SHA-256 sealed acceptance JSON; omission blocks")
    command.add_argument("--candidate-config", type=Path,
                         help="saved physical configuration JSON, checked against the reference digest")
    command.add_argument("--candidate-initializer", type=Path,
                         help="saved initializer artifact checked against its independently registered digest")
    command.add_argument("--out-report", type=Path, required=True, metavar="JSON",
                         help="write the saved-field comparison receipt as JSON")
    command.set_defaults(func=_run)
