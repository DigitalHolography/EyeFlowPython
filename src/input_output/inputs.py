"""Resolve HOLO selections and expose HD/DV/work HDF5 inputs to pipelines."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py

from .schema import (
    DOPPLER_VIEW_SCHEMA,
    HDF5_SUFFIXES,
    HOLODOPPLER_SCHEMA,
    HOLO_COMPANION_H5_LAYOUTS,
    HOLO_DATA_DIR_TEMPLATE,
    HOLO_SUFFIX,
    H5SourceSchema,
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


def resolve_holo_input(
    holo_path: Path,
    *,
    require_holo_file: bool = True,
    relative_holo_path: Path | None = None,
) -> ResolvedHoloInput:
    holo_path = _absolute(holo_path)
    _validate_holo_file(holo_path, require_file=require_holo_file)
    data_dir = _data_dir_for_holo(holo_path)
    _require_dir(data_dir, f"Could not find data folder:\n{data_dir}")

    resolved = _resolve_required_companions(data_dir, holo_path.stem)
    return ResolvedHoloInput(
        holo_path=holo_path,
        relative_holo_path=relative_holo_path or Path(holo_path.name),
        data_dir=data_dir,
        hd_dir=HOLODOPPLER_SCHEMA.layout.companion_folder(data_dir, holo_path.stem),
        dv_dir=DOPPLER_VIEW_SCHEMA.layout.companion_folder(data_dir, holo_path.stem),
        hd_h5=resolved[HOLODOPPLER_SCHEMA.layout],
        dv_h5=resolved[DOPPLER_VIEW_SCHEMA.layout],
    )


def resolve_selected_holo_inputs(
    holo_paths: Sequence[Path],
) -> list[ResolvedHoloInput]:
    normalized = [_absolute(path) for path in holo_paths]
    if not normalized:
        raise ValueError(f"Select one or more {HOLO_SUFFIX} files.")

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
            "Missing required input data for one or more selected "
            f"{HOLO_SUFFIX} files:\n\n"
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
        hd=bool(
            _h5_files(HOLODOPPLER_SCHEMA.layout.h5_folder(data_dir, holo_path.stem))
        ),
        dv=bool(
            _h5_files(DOPPLER_VIEW_SCHEMA.layout.h5_folder(data_dir, holo_path.stem))
        ),
    )


def _lookup_key(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


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
            f"{layout.companion_name} HDF5 file missing in expected folder:\n{folder}"
        )

    preferred = layout.h5_filename(stem).lower()
    matching = [path for path in candidates if path.name.lower() == preferred]
    if matching:
        return matching[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(_multiple_h5_message(layout, folder, stem, candidates))


def _multiple_h5_message(
    layout: HoloCompanionH5Layout,
    folder: Path,
    stem: str,
    candidates: Sequence[Path],
) -> str:
    candidate_list = "\n".join(str(candidate) for candidate in candidates)
    return (
        f"Multiple {layout.companion_name} HDF5 files found in:\n{folder}\n\n"
        f"Expected one file, preferably named:\n{layout.h5_filename(stem)}\n\n"
        f"Candidates:\n{candidate_list}"
    )


def _data_dir_for_holo(holo_path: Path) -> Path:
    return holo_path.parent / HOLO_DATA_DIR_TEMPLATE.format(stem=holo_path.stem)


def _require_dir(path: Path, message: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(message)


def _resolve_layout_h5(
    layout: HoloCompanionH5Layout,
    *,
    data_dir: Path,
    stem: str,
) -> Path:
    companion_dir = layout.companion_folder(data_dir, stem)
    _require_dir(
        companion_dir,
        f"{layout.companion_name} folder missing:\n{companion_dir}",
    )
    h5_dir = layout.h5_folder(data_dir, stem)
    _require_dir(
        h5_dir,
        f"{layout.companion_name} HDF5 folder missing:\n{h5_dir}",
    )
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


def _resolve_required_companions(data_dir: Path, stem: str) -> dict[object, Path]:
    errors: list[str] = []
    resolved: dict[object, Path] = {}
    for layout in HOLO_COMPANION_H5_LAYOUTS:
        try:
            resolved[layout] = _resolve_layout_h5(layout, data_dir=data_dir, stem=stem)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    if errors:
        raise FileNotFoundError(
            f"Missing required input data for the selected {HOLO_SUFFIX} file:\n\n"
            + "\n\n".join(errors)
        )
    return resolved


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


class MergedAttrs(Mapping[str, object]):
    def __init__(self, *sources: h5py.File | Mapping[str, object] | None) -> None:
        self._sources = [
            _attr_source(source) for source in sources if source is not None
        ]

    def __getitem__(self, key: str) -> object:
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        for source in self._sources:
            for key in source.keys():
                if key not in seen:
                    seen.add(key)
                    yield str(key)

    def __len__(self) -> int:
        return sum(1 for _ in self.__iter__())

    def get(self, key: str, default=None):
        for source in self._sources:
            if key in source:
                return source[key]
        return default


class EyeFlowView:
    def __init__(self, work_h5: h5py.File) -> None:
        self.work_h5 = work_h5

    def get(self, key: str, default=None):
        normalized_key = _lookup_key(key)
        if not normalized_key:
            return default
        return self.work_h5.get(normalized_key, default)

    def __getitem__(self, key: str):
        found = self.get(key)
        if found is None:
            raise KeyError(key)
        return found

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None


class PipelineInputView:
    def __init__(
        self,
        *,
        work_h5: h5py.File,
        holodoppler_h5: h5py.File | None = None,
        doppler_vision_h5: h5py.File | None = None,
        preferred_input: str = "both",
    ) -> None:
        self.work_h5 = work_h5
        self.hd_h5 = holodoppler_h5
        self.dv_h5 = doppler_vision_h5
        self.work = work_h5
        self.hd = holodoppler_h5
        self.dv = doppler_vision_h5
        self.ef = EyeFlowView(work_h5)
        self.preferred_input = preferred_input
        self.hd_config = _load_sidecar_config(
            holodoppler_h5,
            source_schema=HOLODOPPLER_SCHEMA,
        )
        self.dv_config = _load_sidecar_config(
            doppler_vision_h5,
            source_schema=DOPPLER_VIEW_SCHEMA,
        )
        self.attrs = MergedAttrs(
            self.work_h5,
            self._preferred_raw_source(),
            self._secondary_raw_source(),
            self.hd_config,
            self.dv_config,
        )

    def _preferred_raw_source(self) -> h5py.File | None:
        if self.preferred_input == "dv":
            return self.dv_h5 or self.hd_h5
        return self.hd_h5 or self.dv_h5

    def _secondary_raw_source(self) -> h5py.File | None:
        preferred = self._preferred_raw_source()
        if preferred is self.hd_h5:
            return self.dv_h5
        if preferred is self.dv_h5:
            return self.hd_h5
        return None

    @property
    def filename(self) -> str:
        primary = self._preferred_raw_source()
        if primary is not None and primary.filename is not None:
            return str(primary.filename)
        if self.work_h5.filename is not None:
            return str(self.work_h5.filename)
        return ""

    def _lookup_in_source(self, source: h5py.File | None, key: str):
        if source is None:
            return None
        return source.get(key)

    def get(self, key: str, default=None):
        normalized_key = _lookup_key(key)
        if not normalized_key:
            return default

        for source in (
            self.work_h5,
            self._preferred_raw_source(),
            self._secondary_raw_source(),
        ):
            found = self._lookup_in_source(source, normalized_key)
            if found is not None:
                return found
        return default

    def __getitem__(self, key: str):
        found = self.get(key)
        if found is None:
            raise KeyError(key)
        return found

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None


def _attr_source(source: h5py.File | Mapping[str, object]) -> Mapping[str, object]:
    return source.attrs if isinstance(source, h5py.File) else source


def _load_sidecar_config(
    h5file: h5py.File | None,
    *,
    source_schema: H5SourceSchema,
) -> dict[str, object]:
    if h5file is None or h5file.filename is None:
        return {}
    if not source_schema.config_dir_name or not source_schema.config_filename:
        return {}
    config_path = _sidecar_config_path(
        Path(h5file.filename),
        folder_name=source_schema.config_dir_name,
        preferred_name=source_schema.config_filename,
    )
    if config_path is None:
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return _normalize_config_keys(payload)


def _sidecar_config_path(
    h5_path: Path,
    *,
    folder_name: str,
    preferred_name: str,
) -> Path | None:
    config_dir = h5_path.parent.parent / folder_name
    if not config_dir.is_dir():
        return None
    preferred = config_dir / preferred_name
    if preferred.is_file():
        return preferred
    json_files = sorted(config_dir.glob("*.json"))
    return json_files[0] if json_files else None


def _normalize_config_keys(value):
    if isinstance(value, dict):
        return {
            str(key).replace(" ", ""): _normalize_config_keys(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_normalize_config_keys(item) for item in value]
    return value
