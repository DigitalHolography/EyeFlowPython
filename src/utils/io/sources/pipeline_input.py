"""Pipeline input wrappers for work, HD, and DV HDF5 files.

Pipelines use these objects to read raw inputs, previous EyeFlow outputs, and
merged file attributes without knowing where each value came from.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import h5py

from ..schema import (
    DV_CONFIG_DIR_NAME,
    DV_CONFIG_FILENAME,
    HD_CONFIG_DIR_NAME,
    HD_CONFIG_FILENAME,
    get_processing_root,
)


def _lookup_key(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


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

        root = get_processing_root(self.work_h5)
        if root is None:
            return default

        explicit = root.get(normalized_key)
        if explicit is not None:
            return explicit

        for pipeline_name in reversed(list(root.keys())):
            candidate = root.get(f"{pipeline_name}/{normalized_key}")
            if candidate is not None:
                return candidate
        return default

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
            folder_name=HD_CONFIG_DIR_NAME,
            preferred_name=HD_CONFIG_FILENAME,
        )
        self.dv_config = _load_sidecar_config(
            doppler_vision_h5,
            folder_name=DV_CONFIG_DIR_NAME,
            preferred_name=DV_CONFIG_FILENAME,
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

        direct = source.get(key)
        if direct is not None:
            return direct

        root = get_processing_root(source)
        if root is None:
            return None

        for pipeline_name in reversed(list(root.keys())):
            candidate = root.get(f"{pipeline_name}/{key}")
            if candidate is not None:
                return candidate
        return None

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
    folder_name: str,
    preferred_name: str,
) -> dict[str, object]:
    if h5file is None or h5file.filename is None:
        return {}
    config_path = _sidecar_config_path(
        Path(h5file.filename),
        folder_name=folder_name,
        preferred_name=preferred_name,
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
