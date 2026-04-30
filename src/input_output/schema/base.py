"""Pydantic models for declared HDF5 source schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HoloCompanionH5Layout(BaseModel):
    """Folder and filename layout for one HOLO companion HDF5 source."""

    model_config = ConfigDict(frozen=True)

    companion_name: str
    h5_folder_name: str
    h5_filename_template: str

    def companion_folder_name(self, stem: str) -> str:
        return f"{stem}_{self.companion_name}"

    def companion_folder(self, root_dir: Path, stem: str) -> Path:
        return root_dir / self.companion_folder_name(stem)

    def h5_folder(self, root_dir: Path, stem: str) -> Path:
        return self.companion_folder(root_dir, stem) / self.h5_folder_name

    def h5_filename(self, stem: str) -> str:
        companion_folder_name = self.companion_folder_name(stem)
        return self.h5_filename_template.format(
            stem=stem,
            folder=companion_folder_name,
            companion=self.companion_name,
        )


class H5DatasetSpec(BaseModel):
    """One named HDF5 dataset in an upstream source file."""

    model_config = ConfigDict(frozen=True)

    key: str
    path: str
    dtype: str | None = None
    dims: tuple[str, ...] = ()
    required: bool = True
    description: str = ""

    @field_validator("path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return value.replace("\\", "/").strip("/")


class JsonConfigValueSpec(BaseModel):
    """One config value that may live in HDF5 or a sidecar JSON file."""

    model_config = ConfigDict(frozen=True)

    key: str
    json_key: str
    h5_path: str | None = None
    section: str | None = None
    default: Any = None
    description: str = ""

    @field_validator("h5_path")
    @classmethod
    def normalize_h5_path(cls, value: str | None) -> str | None:
        return None if value is None else value.replace("\\", "/").strip("/")

    def read_json_config(self, config: dict[str, object]) -> Any:
        source = config.get(self.section, {}) if self.section else config
        if not isinstance(source, dict):
            return self.default
        return source.get(self.json_key, self.default)


class H5SourceSchema(BaseModel):
    """Declared HDF5 source contract for a companion application."""

    model_config = ConfigDict(frozen=True)

    label: str
    layout: HoloCompanionH5Layout
    config_dir_name: str | None = None
    config_filename: str | None = None
    datasets: dict[str, H5DatasetSpec] = Field(default_factory=dict)
    config_values: dict[str, JsonConfigValueSpec] = Field(default_factory=dict)

    def dataset(self, key: str) -> H5DatasetSpec:
        try:
            return self.datasets[key]
        except KeyError as exc:
            raise KeyError(f"{self.label} schema has no dataset '{key}'.") from exc

    def dataset_path(self, key: str) -> str:
        return self.dataset(key).path

    def config_value(self, key: str) -> JsonConfigValueSpec:
        try:
            return self.config_values[key]
        except KeyError as exc:
            message = f"{self.label} schema has no config value '{key}'."
            raise KeyError(message) from exc
