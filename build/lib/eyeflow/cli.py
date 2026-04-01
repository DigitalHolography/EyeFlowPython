"""Command line entrypoint for the initial EyeFlow Python scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from eyeflow.errors import EyeFlowError, InputDiscoveryError
from eyeflow.io.discovery import discover_h5_sources
from eyeflow.validation import H5ContractValidator, ValidationReport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eyeflow",
        description="EyeFlow Python CLI.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate one H5 file, a folder tree, or a zip archive against the current H5 contract.",
    )
    validate_parser.add_argument(
        "input_path",
        help="Path to an H5 file, a directory to scan recursively, or a zip archive containing H5 files.",
    )
    validate_parser.add_argument(
        "--require-sh",
        action="store_true",
        help="Fail validation when /SH is missing.",
    )
    validate_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first invalid file.",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print machine-readable JSON output instead of text.",
    )
    validate_parser.set_defaults(handler=run_validate_command)

    return parser


def run_validate_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input_path).expanduser()
    sources = discover_h5_sources(input_path)
    validator = H5ContractValidator(require_sh=args.require_sh)

    reports: list[ValidationReport] = []
    for source in sources:
        report = validator.validate_source(source)
        reports.append(report)
        if args.fail_fast and not report.ok:
            break

    if args.json_output:
        print(json.dumps([report.to_dict() for report in reports], indent=2))
    else:
        print(format_reports(reports))

    return 0 if reports and all(report.ok for report in reports) else 1


def format_reports(reports: Sequence[ValidationReport]) -> str:
    lines: list[str] = []

    for report in reports:
        status = "OK" if report.ok else "FAIL"
        lines.append(f"[{status}] {report.source}")

        if report.ok and report.summary:
            summary = report.summary
            lines.append(
                "  "
                + ", ".join(
                    [
                        f"moments={summary['moments_shape']}",
                        f"time={summary['time_source']}",
                        f"spatial={summary['spatial_source']}",
                        f"SH={'yes' if summary['has_sh'] else 'no'}",
                    ]
                )
            )

        for warning in report.warnings:
            lines.append(f"  warning: {warning}")

        for error in report.errors:
            lines.append(f"  error: {error}")

    valid_count = sum(report.ok for report in reports)
    invalid_count = len(reports) - valid_count
    lines.append("")
    lines.append(
        f"Validated {len(reports)} file(s): {valid_count} valid, {invalid_count} invalid."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except InputDiscoveryError as exc:
        parser.exit(status=2, message=f"Input error: {exc}\n")
    except EyeFlowError as exc:
        parser.exit(status=2, message=f"EyeFlow error: {exc}\n")
