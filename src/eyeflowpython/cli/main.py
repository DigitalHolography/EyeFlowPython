from __future__ import annotations

import argparse
from pathlib import Path

from eyeflowpython.processing import default_output_root, process_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process H5 files and write VelocityPerBeat outputs into new H5 files."
    )
    parser.add_argument("input_path", type=Path, help="Input .h5, folder, or .zip.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output folder root. Defaults to <input>_processed.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        help="Seconds per frame. Overrides HDF5 metadata.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_root = args.output or default_output_root(args.input_path)

    results = process_input(
        args.input_path,
        output_root=output_root,
        dt_override=args.dt,
        logger=print,
    )

    print(f"Processed {len(results)} file(s) into {output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
