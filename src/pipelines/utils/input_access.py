from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from eye_flow import _PipelineInputView


@dataclass(frozen=True)
class ResolvedArray:
    path: str
    value: np.ndarray


def dataset_path_candidates(*paths: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for path in paths:
        normalized = str(path).replace("\\", "/").strip("/")
        if not normalized:
            continue
        for candidate in (
            normalized,
            normalized.removesuffix("/value"),
            f"{normalized}/value",
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def resolve_required_array(
    pipeline_input: "_PipelineInputView",
    logical_name: str,
    *candidate_roots: str,
) -> ResolvedArray:
    candidate_paths = dataset_path_candidates(*candidate_roots)
    for candidate_path in candidate_paths:
        found = pipeline_input.get(candidate_path)
        if isinstance(found, h5py.Dataset):
            return ResolvedArray(
                path=candidate_path,
                value=np.asarray(found[()]),
            )

    candidates_text = ", ".join(candidate_paths)
    raise KeyError(
        f"Missing pipeline prerequisite '{logical_name}'. "
        f"Tried dataset paths: {candidates_text}"
    )


def read_first_attr(pipeline_input: "_PipelineInputView", *keys: str):
    for key in keys:
        value = pipeline_input.attrs.get(key, None)
        if value is None:
            continue
        array = np.asarray(value).reshape(-1)
        if array.size == 0:
            continue
        scalar = array[0]
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8")
        return scalar.item() if hasattr(scalar, "item") else scalar
    return None


def resolve_dt_seconds(pipeline_input: "_PipelineInputView") -> float:
    explicit_dt = read_first_attr(
        pipeline_input,
        "dt_seconds",
        "frame_period_seconds",
        "time_step_seconds",
    )
    if explicit_dt is not None:
        return float(explicit_dt)

    stride = read_first_attr(
        pipeline_input,
        "batch_stride",
        "stride",
        "time_transformation_stride",
    )
    fs = read_first_attr(pipeline_input, "fs", "Fs")
    camera_fps = read_first_attr(pipeline_input, "camera_fps")

    if stride is not None and fs is not None:
        return float(stride) / float(fs) / 1000.0
    if stride is not None and camera_fps is not None:
        return float(stride) / float(camera_fps)

    raise KeyError(
        "Could not resolve dt_seconds from input attributes. Expected "
        "dt_seconds directly, or stride + fs/camera_fps."
    )


def read_int_setting(
    pipeline_input: "_PipelineInputView",
    *,
    default: int,
    keys: tuple[str, ...],
) -> int:
    value = read_first_attr(pipeline_input, *keys)
    if value is None:
        return int(default)
    return int(value)
