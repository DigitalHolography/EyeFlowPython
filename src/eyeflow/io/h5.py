"""H5 source abstraction and open helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Iterator
import zipfile

import h5py


@dataclass(frozen=True, slots=True)
class H5Source:
    """Reference to an H5 file on disk or inside a zip archive."""

    path: Path
    display_name: str
    zip_member: str | None = None

    @property
    def is_archive_member(self) -> bool:
        return self.zip_member is not None

    @classmethod
    def from_zip_archive(cls, archive_path: Path) -> list["H5Source"]:
        sources: list[H5Source] = []
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in sorted(archive.namelist()):
                member_path = Path(member)
                if member.endswith("/") or member_path.suffix.lower() not in {".h5", ".hdf5"}:
                    continue
                display_name = f"{archive_path.resolve()}::{member}"
                sources.append(
                    cls(
                        path=archive_path,
                        display_name=display_name,
                        zip_member=member,
                    )
                )
        return sources


@contextmanager
def open_h5_source(source: H5Source) -> Iterator[h5py.File]:
    """Open a source as an h5py handle.

    Archive members are loaded into memory for this initial validator.
    """

    if source.zip_member is None:
        with h5py.File(source.path, "r") as handle:
            yield handle
        return

    with zipfile.ZipFile(source.path, "r") as archive:
        with archive.open(source.zip_member, "r") as member:
            payload = member.read()

    with h5py.File(io.BytesIO(payload), "r") as handle:
        yield handle
