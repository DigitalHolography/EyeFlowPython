"""Resolve selected `.holo` files into their required HD and DV HDF5 inputs."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..schema import (
    DV_H5_LAYOUT,
    HD_H5_LAYOUT,
    HDF5_SUFFIXES,
    HOLO_COMPANION_H5_LAYOUTS,
    HOLO_DATA_DIR_TEMPLATE,
    HOLO_SUFFIX,
    HoloCompanionH5Layout,
)


@dataclass(frozen=True)
class ResolvedHoloInput:
    holo_path: Path
    relative_holo_path: Path
    data_dir: Path
    hd_dir: Path
    dv_dir: Path
    hd_h5: Path
    dv_h5: Path


@dataclass(frozen=True)
class HoloInputStatus:
    hd: bool
    dv: bool


def _absolute(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else Path.cwd() / resolved


def _h5_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def _choose_h5_file(layout: HoloCompanionH5Layout, folder: Path, stem: str) -> Path:
    candidates = _h5_files(folder)
    if not candidates:
        raise FileNotFoundError(
            f"{layout.label} HDF5 file missing in expected folder:\n{folder}"
        )

    preferred = layout.preferred_filename(stem).lower()
    for candidate in candidates:
        if candidate.name.lower() == preferred:
            return candidate

    if len(candidates) == 1:
        return candidates[0]

    candidate_list = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Multiple {layout.label} HDF5 files found in:\n{folder}\n\n"
        f"Expected one file, preferably named:\n{layout.preferred_filename(stem)}\n\n"
        f"Candidates:\n{candidate_list}"
    )


def _data_dir_for_holo(holo_path: Path) -> Path:
    return holo_path.parent / HOLO_DATA_DIR_TEMPLATE.format(stem=holo_path.stem)


def _resolve_layout_h5(
    layout: HoloCompanionH5Layout,
    *,
    data_dir: Path,
    stem: str,
) -> Path:
    companion_dir = layout.folder_path(data_dir, stem)
    if not companion_dir.is_dir():
        raise FileNotFoundError(f"{layout.label} folder missing:\n{companion_dir}")

    h5_dir = layout.h5_dir(data_dir, stem)
    if not h5_dir.is_dir():
        raise FileNotFoundError(f"{layout.label} HDF5 folder missing:\n{h5_dir}")

    return _choose_h5_file(layout, h5_dir, stem)


def _validate_holo_file(holo_path: Path, *, require_file: bool) -> None:
    if holo_path.suffix.lower() != HOLO_SUFFIX:
        raise ValueError(f"HOLO input must be a {HOLO_SUFFIX} file:\n{holo_path}")
    if not require_file:
        return
    if not holo_path.exists():
        raise FileNotFoundError(f"HOLO input does not exist:\n{holo_path}")
    if not holo_path.is_file():
        raise ValueError(f"HOLO input must be a file:\n{holo_path}")


def resolve_holo_input(
    holo_path: Path,
    *,
    require_holo_file: bool = True,
    relative_holo_path: Path | None = None,
) -> ResolvedHoloInput:
    holo_path = _absolute(holo_path)
    _validate_holo_file(holo_path, require_file=require_holo_file)

    data_dir = _data_dir_for_holo(holo_path)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Could not find the data folder matching the selected {HOLO_SUFFIX} file:\n"
            f"{data_dir}"
        )

    errors: list[str] = []
    resolved: dict[HoloCompanionH5Layout, Path] = {}
    for layout in HOLO_COMPANION_H5_LAYOUTS:
        try:
            resolved[layout] = _resolve_layout_h5(
                layout,
                data_dir=data_dir,
                stem=holo_path.stem,
            )
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if errors:
        raise FileNotFoundError(
            f"Missing required input data for the selected {HOLO_SUFFIX} file:\n\n"
            + "\n\n".join(errors)
        )

    return ResolvedHoloInput(
        holo_path=holo_path,
        relative_holo_path=relative_holo_path or Path(holo_path.name),
        data_dir=data_dir,
        hd_dir=HD_H5_LAYOUT.folder_path(data_dir, holo_path.stem),
        dv_dir=DV_H5_LAYOUT.folder_path(data_dir, holo_path.stem),
        hd_h5=resolved[HD_H5_LAYOUT],
        dv_h5=resolved[DV_H5_LAYOUT],
    )


def _batch_root(holo_paths: Sequence[Path]) -> Path:
    if not holo_paths:
        return Path.cwd()
    if len(holo_paths) == 1:
        return holo_paths[0].parent
    try:
        return Path(os.path.commonpath([str(path.parent) for path in holo_paths]))
    except ValueError:
        return Path.cwd()


def _relative_to_batch(holo_path: Path, batch_root: Path) -> Path:
    try:
        return holo_path.relative_to(batch_root)
    except ValueError:
        anchor = Path(holo_path.anchor)
        drive_token = holo_path.drive.rstrip(":\\/") or "root"
        tail = holo_path.relative_to(anchor) if anchor != holo_path else Path()
        return Path(drive_token) / tail


def resolve_selected_holo_inputs(
    holo_paths: Sequence[Path],
) -> list[ResolvedHoloInput]:
    normalized = [_absolute(path) for path in holo_paths]
    if not normalized:
        raise ValueError(f"Select one or more {HOLO_SUFFIX} files.")

    if len(normalized) == 1:
        return [resolve_holo_input(normalized[0])]

    batch_root = _batch_root(normalized)
    resolved: list[ResolvedHoloInput] = []
    errors: list[str] = []
    for holo_path in normalized:
        try:
            resolved.append(
                resolve_holo_input(
                    holo_path,
                    relative_holo_path=_relative_to_batch(holo_path, batch_root),
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{holo_path}:\n{exc}")

    if errors:
        raise FileNotFoundError(
            f"Missing required input data for one or more selected {HOLO_SUFFIX} files:\n\n"
            + "\n\n".join(errors)
        )

    return resolved


def holo_input_status(
    holo_path: Path,
    *,
    require_holo_file: bool,
) -> HoloInputStatus:
    holo_path = _absolute(holo_path)
    try:
        _validate_holo_file(holo_path, require_file=require_holo_file)
    except (FileNotFoundError, ValueError):
        return HoloInputStatus(hd=False, dv=False)

    data_dir = _data_dir_for_holo(holo_path)
    return HoloInputStatus(
        hd=bool(_h5_files(HD_H5_LAYOUT.h5_dir(data_dir, holo_path.stem))),
        dv=bool(_h5_files(DV_H5_LAYOUT.h5_dir(data_dir, holo_path.stem))),
    )
