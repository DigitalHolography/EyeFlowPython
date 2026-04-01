"""Input discovery for files, directory trees, and zip archives."""

from __future__ import annotations

from pathlib import Path

from eyeflow.errors import InputDiscoveryError
from eyeflow.io.h5 import H5Source

H5_SUFFIXES = {".h5", ".hdf5"}


def discover_h5_sources(input_path: Path) -> list[H5Source]:
    if not input_path.exists():
        raise InputDiscoveryError(f"Path does not exist: {input_path}")

    if input_path.is_file():
        suffix = input_path.suffix.lower()

        if suffix in H5_SUFFIXES:
            return [H5Source(path=input_path, display_name=str(input_path.resolve()))]

        if suffix == ".zip":
            sources = H5Source.from_zip_archive(input_path)
            if not sources:
                raise InputDiscoveryError(f"No H5 files found inside archive: {input_path}")
            return sources

        raise InputDiscoveryError(
            f"Unsupported file type: {input_path.suffix or '<no suffix>'}. "
            "Expected .h5, .hdf5, or .zip."
        )

    if input_path.is_dir():
        sources = [
            H5Source(path=path, display_name=str(path.resolve()))
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in H5_SUFFIXES
        ]
        if not sources:
            raise InputDiscoveryError(f"No H5 files found under directory: {input_path}")
        return sources

    raise InputDiscoveryError(f"Unsupported input path: {input_path}")
