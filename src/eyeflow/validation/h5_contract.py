"""First-pass validator for the EyeFlow H5 input contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from eyeflow.io.h5 import H5Source, open_h5_source

REQUIRED_MOMENT_PATHS = ("/moment0", "/moment1", "/moment2")
REQUIRED_MASK_PATHS = ("/masks/artery", "/masks/vein")
OPTIONAL_SH_PATH = "/SH"

GROUP_PATHS = ("/meta", "/metadata", "/")
TIMESTAMP_KEYS = ("timestamps_us", "time_stamps_us", "record_time_stamps_us")
FRAME_RATE_KEYS = ("frame_rate_hz", "fps")
STRIDE_KEYS = ("stride", "batch_stride")
PIXEL_SIZE_KEYS = ("pixel_size_mm",)


@dataclass(slots=True)
class ValidationReport:
    """Structured validation result for one source."""

    source: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
        }


class H5ContractValidator:
    """Validate the current H5 schema described in the project README."""

    def __init__(self, *, require_sh: bool = False) -> None:
        self.require_sh = require_sh

    def validate_source(self, source: H5Source) -> ValidationReport:
        report = ValidationReport(source=source.display_name)
        with open_h5_source(source) as handle:
            self._validate_handle(handle, report)
        return report

    def _validate_handle(self, handle: h5py.File, report: ValidationReport) -> None:
        moment_datasets = self._load_required_datasets(handle, REQUIRED_MOMENT_PATHS, report)
        mask_datasets = self._load_required_datasets(handle, REQUIRED_MASK_PATHS, report)

        if report.errors:
            return

        moments_shape = self._validate_moments(moment_datasets, report)
        if moments_shape is None:
            return

        self._validate_masks(mask_datasets, moments_shape[:2], report)
        self._validate_optional_sh(handle, moments_shape[:2], report)

        time_source = self._validate_time_metadata(handle, moments_shape[2], report)
        spatial_source = self._validate_spatial_metadata(handle, report)

        if report.ok:
            report.summary = {
                "moments_shape": list(moments_shape),
                "time_source": time_source,
                "spatial_source": spatial_source,
                "has_sh": OPTIONAL_SH_PATH in handle,
            }

    def _load_required_datasets(
        self,
        handle: h5py.File,
        dataset_paths: tuple[str, ...],
        report: ValidationReport,
    ) -> dict[str, h5py.Dataset]:
        datasets: dict[str, h5py.Dataset] = {}
        for dataset_path in dataset_paths:
            if dataset_path not in handle:
                report.add_error(f"Missing required dataset: {dataset_path}")
                continue
            datasets[dataset_path] = handle[dataset_path]
        return datasets

    def _validate_moments(
        self,
        moment_datasets: dict[str, h5py.Dataset],
        report: ValidationReport,
    ) -> tuple[int, int, int] | None:
        expected_shape: tuple[int, int, int] | None = None

        for dataset_path, dataset in moment_datasets.items():
            if dataset.ndim != 3:
                report.add_error(
                    f"{dataset_path} must be a 3D dataset shaped (height, width, frames); "
                    f"found ndim={dataset.ndim}."
                )
                continue

            if not np.issubdtype(dataset.dtype, np.number):
                report.add_error(f"{dataset_path} must be numeric; found dtype={dataset.dtype}.")
                continue

            if min(dataset.shape) <= 0:
                report.add_error(f"{dataset_path} has an invalid shape: {dataset.shape}.")
                continue

            current_shape = tuple(int(dim) for dim in dataset.shape)
            if expected_shape is None:
                expected_shape = current_shape
                continue

            if current_shape != expected_shape:
                report.add_error(
                    "Moment datasets must all share the same shape. "
                    f"Expected {expected_shape}, found {current_shape} at {dataset_path}."
                )

        return expected_shape

    def _validate_masks(
        self,
        mask_datasets: dict[str, h5py.Dataset],
        spatial_shape: tuple[int, int],
        report: ValidationReport,
    ) -> None:
        loaded_masks: dict[str, np.ndarray] = {}

        for dataset_path, dataset in mask_datasets.items():
            if dataset.ndim != 2:
                report.add_error(
                    f"{dataset_path} must be a 2D binary mask shaped (height, width); "
                    f"found ndim={dataset.ndim}."
                )
                continue

            if tuple(int(dim) for dim in dataset.shape) != spatial_shape:
                report.add_error(
                    f"{dataset_path} shape {tuple(dataset.shape)} does not match moment "
                    f"shape {spatial_shape}."
                )
                continue

            if not (np.issubdtype(dataset.dtype, np.number) or np.issubdtype(dataset.dtype, np.bool_)):
                report.add_error(f"{dataset_path} must be numeric or boolean; found dtype={dataset.dtype}.")
                continue

            mask = dataset[()]
            if np.isnan(mask).any():
                report.add_error(f"{dataset_path} contains NaN values.")
                continue

            unique_values = np.unique(mask)
            if unique_values.size > 2 or not np.all(np.isin(unique_values, [0, 1])):
                report.add_error(
                    f"{dataset_path} must contain only binary values 0/1; "
                    f"found values {unique_values.tolist()}."
                )
                continue

            if np.count_nonzero(mask) == 0:
                report.add_error(f"{dataset_path} is empty.")
                continue

            loaded_masks[dataset_path] = mask.astype(bool, copy=False)

        if len(loaded_masks) == len(REQUIRED_MASK_PATHS):
            overlap = np.count_nonzero(
                loaded_masks["/masks/artery"] & loaded_masks["/masks/vein"]
            )
            if overlap > 0:
                report.add_warning(
                    f"Artery and vein masks overlap on {overlap} pixel(s)."
                )

    def _validate_optional_sh(
        self,
        handle: h5py.File,
        spatial_shape: tuple[int, int],
        report: ValidationReport,
    ) -> None:
        if OPTIONAL_SH_PATH not in handle:
            if self.require_sh:
                report.add_error("Missing required dataset: /SH")
            return

        dataset = handle[OPTIONAL_SH_PATH]
        if dataset.ndim < 3:
            report.add_error(
                f"{OPTIONAL_SH_PATH} must be at least 3D; found ndim={dataset.ndim}."
            )
            return

        if tuple(int(dim) for dim in dataset.shape[:2]) != spatial_shape:
            report.add_error(
                f"{OPTIONAL_SH_PATH} spatial shape {tuple(dataset.shape[:2])} does not match "
                f"moment shape {spatial_shape}."
            )
            return

        if not np.issubdtype(dataset.dtype, np.number):
            report.add_error(f"{OPTIONAL_SH_PATH} must be numeric; found dtype={dataset.dtype}.")

    def _validate_time_metadata(
        self,
        handle: h5py.File,
        frame_count: int,
        report: ValidationReport,
    ) -> str | None:
        timestamps_result = _find_array_metadata(handle, TIMESTAMP_KEYS)
        if timestamps_result is not None:
            source, raw_values = timestamps_result
            values = np.asarray(raw_values).reshape(-1)

            if values.size != frame_count:
                report.add_warning(
                    f"{source} was found but has length {values.size}; expected {frame_count}. "
                    "Falling back to frame rate + stride if available."
                )
            elif not np.all(np.isfinite(values)):
                report.add_warning(
                    f"{source} contains non-finite values. Falling back to frame rate + stride if available."
                )
            elif np.any(np.diff(values) <= 0):
                report.add_warning(
                    f"{source} is not strictly increasing. Falling back to frame rate + stride if available."
                )
            else:
                return source

        frame_rate_result = _find_scalar_metadata(handle, FRAME_RATE_KEYS)
        stride_result = _find_scalar_metadata(handle, STRIDE_KEYS)

        if frame_rate_result is None or stride_result is None:
            report.add_error(
                "Missing required time metadata. Provide either timestamps_us or both "
                "frame_rate_hz and stride."
            )
            return None

        frame_rate_source, frame_rate = frame_rate_result
        stride_source, stride = stride_result
        if frame_rate <= 0:
            report.add_error(f"{frame_rate_source} must be > 0; found {frame_rate}.")
            return None
        if stride <= 0:
            report.add_error(f"{stride_source} must be > 0; found {stride}.")
            return None

        return f"{frame_rate_source} + {stride_source}"

    def _validate_spatial_metadata(
        self,
        handle: h5py.File,
        report: ValidationReport,
    ) -> str | None:
        pixel_size_result = _find_scalar_metadata(handle, PIXEL_SIZE_KEYS)
        if pixel_size_result is None:
            report.add_error(
                "Missing required spatial metadata. Provide pixel_size_mm as a scalar dataset or attribute."
            )
            return None

        source, pixel_size = pixel_size_result
        if pixel_size <= 0:
            report.add_error(f"{source} must be > 0; found {pixel_size}.")
            return None

        return source


def _find_scalar_metadata(
    handle: h5py.File,
    keys: tuple[str, ...],
) -> tuple[str, float] | None:
    for key in keys:
        for group_path in GROUP_PATHS:
            dataset_path = _join_group_path(group_path, key)
            if dataset_path in handle:
                value = np.asarray(handle[dataset_path][()])
                if value.size == 1:
                    return dataset_path, float(value.reshape(-1)[0])

            node = _get_node(handle, group_path)
            if node is not None and key in node.attrs:
                value = np.asarray(node.attrs[key])
                if value.size == 1:
                    return f"{group_path}@{key}", float(value.reshape(-1)[0])

    return None


def _find_array_metadata(
    handle: h5py.File,
    keys: tuple[str, ...],
) -> tuple[str, np.ndarray] | None:
    for key in keys:
        for group_path in GROUP_PATHS:
            dataset_path = _join_group_path(group_path, key)
            if dataset_path in handle:
                return dataset_path, np.asarray(handle[dataset_path][()])

            node = _get_node(handle, group_path)
            if node is not None and key in node.attrs:
                return f"{group_path}@{key}", np.asarray(node.attrs[key])

    return None


def _get_node(handle: h5py.File, group_path: str) -> h5py.File | h5py.Group | None:
    if group_path == "/":
        return handle
    if group_path in handle:
        node = handle[group_path]
        if isinstance(node, (h5py.File, h5py.Group)):
            return node
    return None


def _join_group_path(group_path: str, key: str) -> str:
    if group_path == "/":
        return f"/{key}"
    return f"{group_path}/{key}"
