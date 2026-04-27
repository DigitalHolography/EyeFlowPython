from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from pipelines.core.base import DatasetValue, ProcessResult


@dataclass
class MetricsTree:
    name: str
    metrics: dict[str, object]
    attrs: dict[str, object] | None = None


def safe_h5_key(name: str) -> str:
    """Return a filesystem/HDF5-friendly key derived from a logical name."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return cleaned or "item"


def open_h5(path: Path | str, mode: str = "r") -> h5py.File:
    return h5py.File(Path(path), mode)


def copy_h5_contents(source_file: Path | str | None, dest: h5py.File) -> None:
    """Copy all top-level objects and attributes from an existing HDF5 into dest."""
    if not source_file:
        return
    src_path = Path(source_file)
    if not src_path.exists():
        return
    with open_h5(src_path, "r") as src:
        for key, value in src.attrs.items():
            dest.attrs[key] = value
        for key in src.keys():
            src.copy(src[key], dest, name=key)


def create_h5_file(
    path: Path | str,
    *,
    source_file: Path | str | None = None,
    trim_source: bool = False,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(out_path, "w") as h5file:
        if not trim_source:
            copy_h5_contents(source_file, h5file)
        if source_file:
            h5file.attrs["source_file"] = str(source_file)
    return out_path


def find_first_existing_path(
    group_or_file: h5py.Group | h5py.File,
    candidates: Sequence[str],
) -> str | None:
    for candidate in candidates:
        if candidate in group_or_file:
            return candidate
    return None


def find_child_group_by_attr(
    group: h5py.Group,
    attr_name: str,
    attr_value: object,
) -> h5py.Group | None:
    for child in group.values():
        if isinstance(child, h5py.Group) and child.attrs.get(attr_name) == attr_value:
            return child
    return None


def read_dataset(
    group_or_file: h5py.Group | h5py.File,
    path: str,
    default: object = None,
) -> object:
    try:
        dataset = group_or_file[path]
    except Exception:
        return default
    try:
        return dataset[()]
    except Exception:
        return default


def read_array(
    group_or_file: h5py.Group | h5py.File,
    path: str,
    dtype=None,
) -> np.ndarray | None:
    value = read_dataset(group_or_file, path, default=None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    if arr.shape == ():
        return np.asarray([arr.item()], dtype=dtype)
    return np.ravel(arr)


def create_unique_group(parent: h5py.Group, base_name: str) -> h5py.Group:
    candidate = base_name
    idx = 1
    while candidate in parent:
        candidate = f"{base_name}_{idx}"
        idx += 1
    return parent.create_group(candidate)


def resolve_dataset_target(root_group: h5py.Group, key: str) -> tuple[h5py.Group, str]:
    normalized_key = str(key).replace("\\", "/").strip("/")
    parts = [part for part in normalized_key.split("/") if part]
    if not parts:
        raise ValueError("Dataset key cannot be empty.")

    parent = root_group
    for part in parts[:-1]:
        existing = parent.get(part)
        if existing is None:
            parent = parent.create_group(part)
            continue
        if isinstance(existing, h5py.Group):
            parent = existing
            continue
        raise ValueError(
            f"Cannot create subgroup '{part}' for key '{key}': a dataset already exists at that path."
        )

    return parent, parts[-1]


def set_attr_safe(h5obj: h5py.File | h5py.Group | h5py.Dataset, key: str, value) -> None:
    if isinstance(value, str):
        h5obj.attrs.create(key, value, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    data = value
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, str) for item in value):
            data = np.asarray(value, dtype=h5py.string_dtype(encoding="utf-8"))
        else:
            data = np.asarray(value)
    try:
        h5obj.attrs[key] = data
    except (TypeError, ValueError):
        h5obj.attrs[key] = str(value)


DEFAULT_COMPRESSION_THRESHOLD = 1_000_000


def _normalize_dataset_payload(data, ds_attrs):
    original_class = None
    payload = data

    if isinstance(payload, bool) or (
        isinstance(payload, np.ndarray) and payload.dtype == np.bool_
    ):
        payload = np.asarray(payload, dtype=np.uint8)
        original_class = "bool"
    elif isinstance(payload, (list, tuple)):
        payload = np.asarray(payload)

    if isinstance(payload, np.ndarray):
        if payload.dtype.kind == "f" and payload.dtype.itemsize == 8:
            payload = payload.astype(np.float32)
            original_class = original_class or "float64"

    if original_class is not None:
        ds_attrs = {} if ds_attrs is None else dict(ds_attrs)
        ds_attrs.setdefault("original_class", original_class)

    return payload, ds_attrs


def _get_dataset_creation_kwargs(payload: np.ndarray) -> dict[str, object]:
    if (
        isinstance(payload, np.ndarray)
        and payload.dtype.kind in "bBhHiIlLef"
        and payload.nbytes >= DEFAULT_COMPRESSION_THRESHOLD
        and payload.ndim > 0
    ):
        return {
            "compression": "gzip",
            "compression_opts": 6,
            "chunks": tuple(min(s, 1024) for s in payload.shape),
        }
    return {}


def _get_or_replace_group(parent: h5py.Group, group_name: str) -> h5py.Group:
    existing = parent.get(group_name)
    if existing is not None:
        if isinstance(existing, h5py.Group):
            del parent[group_name]
        else:
            raise ValueError(
                f"Cannot create group '{group_name}': a dataset already exists at that path."
            )
    return parent.create_group(group_name)


def write_value_dataset(group: h5py.Group, key: str, value) -> None:
    from pipelines.core.base import DatasetValue

    ds_attrs = None
    data = value

    if hasattr(value, "data") and hasattr(value, "attrs"):
        data = value.data
        ds_attrs = value.attrs
    elif isinstance(value, DatasetValue):
        data = value.data
        ds_attrs = value.attrs
    elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        data, ds_attrs = value

    target_group, dataset_key = resolve_dataset_target(group, str(key))
    if dataset_key in target_group:
        del target_group[dataset_key]

    payload, ds_attrs = _normalize_dataset_payload(data, ds_attrs)

    if isinstance(payload, str):
        dataset = target_group.create_dataset(
            dataset_key, data=payload, dtype=h5py.string_dtype(encoding="utf-8")
        )
    elif isinstance(payload, (list, tuple)) and all(isinstance(item, str) for item in payload):
        dataset = target_group.create_dataset(
            dataset_key,
            data=np.asarray(payload, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        try:
            create_kwargs = {}
            if isinstance(payload, np.ndarray):
                create_kwargs = _get_dataset_creation_kwargs(payload)
            dataset = target_group.create_dataset(dataset_key, data=payload, **create_kwargs)
        except (TypeError, ValueError):
            if isinstance(payload, np.ndarray) and payload.dtype.kind in {"U", "O"}:
                dataset = target_group.create_dataset(
                    dataset_key,
                    data=np.asarray(payload, dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            else:
                dataset = target_group.create_dataset(
                    dataset_key, data=str(payload), dtype=h5py.string_dtype(encoding="utf-8")
                )

    if ds_attrs:
        for attr_key, attr_val in ds_attrs.items():
            set_attr_safe(dataset, attr_key, attr_val)

    if "nameID" not in (ds_attrs or {}):
        set_attr_safe(dataset, "nameID", str(key))


def write_metrics_tree_group(
    parent: h5py.Group,
    tree: MetricsTree,
    *,
    overwrite: bool = False,
) -> h5py.Group:
    group_name = safe_h5_key(tree.name)
    existing = parent.get(group_name)
    if existing is not None:
        if overwrite:
            del parent[group_name]
        else:
            group = create_unique_group(parent, group_name)
            set_attr_safe(group, "pipeline", tree.name)
            if tree.attrs:
                for key, value in tree.attrs.items():
                    if key == "pipeline":
                        continue
                    set_attr_safe(group, key, value)
            for key, value in tree.metrics.items():
                write_value_dataset(group, key, value)
            return group

    group = parent.create_group(group_name)
    set_attr_safe(group, "pipeline", tree.name)
    if tree.attrs:
        for key, value in tree.attrs.items():
            if key == "pipeline":
                continue
            set_attr_safe(group, key, value)
    for key, value in tree.metrics.items():
        write_value_dataset(group, key, value)
    return group


def write_metrics_trees_to_h5(
    h5_path: Path | str,
    root_path: str,
    trees: Sequence[MetricsTree],
    *,
    overwrite: bool = False,
) -> None:
    with open_h5(h5_path, "r+") as h5file:
        root_group = h5file.require_group(root_path)
        for tree in trees:
            write_metrics_tree_group(
                root_group,
                tree,
                overwrite=overwrite,
            )


def append_metrics_trees_to_h5(
    h5_path: Path | str,
    root_path: str,
    trees: Sequence[MetricsTree],
    *,
    overwrite: bool = True,
) -> None:
    write_metrics_trees_to_h5(
        h5_path,
        root_path,
        trees,
        overwrite=overwrite,
    )


def initialize_output_h5(
    h5file: h5py.File,
    *,
    holodoppler_source_file: str | None = None,
    doppler_vision_source_file: str | None = None,
) -> None:
    if holodoppler_source_file:
        h5file.attrs["holodoppler_source_file"] = holodoppler_source_file
    if doppler_vision_source_file:
        h5file.attrs["doppler_vision_source_file"] = doppler_vision_source_file
    primary_source = holodoppler_source_file or doppler_vision_source_file
    if primary_source:
        h5file.attrs["source_file"] = primary_source


def append_result_group(
    h5file: h5py.File,
    pipeline_name: str,
    result: "ProcessResult",
) -> h5py.Group:
    from pipelines.core.base import ProcessResult

    pipelines_grp = h5file["EyeFlow"] if "EyeFlow" in h5file else h5file.create_group("EyeFlow")
    pipeline_grp = _get_or_replace_group(pipelines_grp, safe_h5_key(pipeline_name))
    pipeline_grp.attrs["pipeline"] = pipeline_name
    if result.attrs:
        for key, value in result.attrs.items():
            if key == "pipeline":
                continue
            set_attr_safe(pipeline_grp, key, value)
    for key, value in result.metrics.items():
        write_value_dataset(pipeline_grp, key, value)
    h5file.flush()
    return pipeline_grp


def write_result_h5(
    result: ProcessResult,
    path: Path | str,
    pipeline_name: str,
    source_file: str | None = None,
) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(out_path, "w") as f:
        copy_h5_contents(source_file, f)
        if "pipeline" not in f.attrs:
            f.attrs["pipeline"] = pipeline_name
        if source_file:
            f.attrs["source_file"] = source_file
        append_result_group(f, pipeline_name, result)
    return str(out_path)


def write_combined_results_h5(
    results: Sequence[tuple[str, ProcessResult]],
    path: Path | str,
    source_file: str | None = None,
    trim_source: bool = False,
) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open_h5(out_path, "w") as f:
        if not trim_source:
            copy_h5_contents(source_file, f)
        if source_file:
            f.attrs["source_file"] = source_file
        for pipeline_name, result in results:
            append_result_group(f, pipeline_name, result)
    return str(out_path)
