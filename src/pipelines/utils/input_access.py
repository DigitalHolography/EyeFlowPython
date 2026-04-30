"""Fixed-source input readers shared by EyeFlow pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

from utils.io.schema import HD_BATCH_STRIDE_KEY, HD_SAMPLING_FREQ_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from utils.io import PipelineInputView


@dataclass(frozen=True)
class ResolvedArray:
    path: str
    value: np.ndarray


@dataclass(frozen=True)
class HolodopplerTiming:
    sampling_freq: float
    batch_stride: float

    @property
    def dt_seconds(self) -> float:
        return self.batch_stride / self.sampling_freq


def resolve_required_source_array(
    source: h5py.File | None,
    *,
    source_name: str,
    logical_name: str,
    path: str,
) -> ResolvedArray:
    if source is None:
        raise KeyError(f"{source_name} HDF5 input is required for '{logical_name}'.")
    found = source.get(path)
    if not isinstance(found, h5py.Dataset):
        raise KeyError(
            f"Missing {source_name} dataset for '{logical_name}'. Expected: {path}"
        )
    return ResolvedArray(path=path, value=np.asarray(found[()]))


def resolve_holodoppler_timing(
    pipeline_input: "PipelineInputView",
) -> HolodopplerTiming:
    sampling_freq = _read_hd_scalar_or_config(pipeline_input, HD_SAMPLING_FREQ_KEY)
    batch_stride = _read_hd_scalar_or_config(pipeline_input, HD_BATCH_STRIDE_KEY)
    if sampling_freq is None or batch_stride is None:
        raise KeyError(
            "Could not resolve Holodoppler timing. Expected fixed keys "
            f"'{HD_SAMPLING_FREQ_KEY}' and '{HD_BATCH_STRIDE_KEY}' in the HD HDF5 "
            "or its sidecar parameters.json."
        )
    return HolodopplerTiming(float(sampling_freq), float(batch_stride))


def resolve_dt_seconds(pipeline_input: "PipelineInputView") -> float:
    return resolve_holodoppler_timing(pipeline_input).dt_seconds


def read_first_attr(pipeline_input: "PipelineInputView", *keys: str):
    for key in keys:
        value = pipeline_input.attrs.get(key, None)
        scalar = _scalar_from_value(value)
        if scalar is not None:
            return scalar
    return None


def read_int_setting(
    pipeline_input: "PipelineInputView",
    *,
    default: int,
    keys: tuple[str, ...],
) -> int:
    value = read_first_attr(pipeline_input, *keys)
    if value is None:
        return int(default)
    return int(value)


def read_nested_int_setting(
    config: "Mapping[str, object]",
    section: str,
    key: str,
    *,
    default: int,
) -> int:
    section_value = config.get(section, {})
    if not isinstance(section_value, dict):
        return int(default)
    value = _scalar_from_value(section_value.get(key))
    return int(default) if value is None else int(value)


def _read_hd_scalar_or_config(pipeline_input: "PipelineInputView", key: str):
    value = _read_source_scalar(pipeline_input.hd, key)
    if value is not None:
        return value
    return _scalar_from_value(pipeline_input.hd_config.get(key))


def _read_source_scalar(source: h5py.File | None, path: str):
    if source is None:
        return None
    found = source.get(path)
    if not isinstance(found, h5py.Dataset):
        return None
    return _scalar_from_value(found[()])


def _scalar_from_value(value):
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return None
    scalar = array[0]
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return scalar.item() if hasattr(scalar, "item") else scalar
