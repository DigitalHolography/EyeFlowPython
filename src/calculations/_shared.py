from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def metric_value(
    data,
    *,
    unit: str | None = None,
    dim_desc: Sequence[str] | None = None,
    matlab_function: str | None = None,
    note: str | None = None,
):
    attrs: dict[str, object] = {}
    if unit:
        attrs["unit"] = unit
    if dim_desc:
        attrs["dimDesc"] = list(dim_desc)
    if matlab_function:
        attrs["matlab_function"] = matlab_function
    if note:
        attrs["note"] = note
    return (data, attrs) if attrs else data


def vessel_group_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized.startswith("art"):
        return "Artery"
    if normalized.startswith("vei"):
        return "Vein"
    if not normalized:
        return "Unknown"
    return normalized[:1].upper() + normalized[1:]


def next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("next_power_of_two expects a strictly positive integer.")
    return 1 << (value - 1).bit_length()


def normalize_cycle_boundaries(
    cycle_boundaries,
    signal_length: int,
    *,
    index_base: int | None = None,
) -> np.ndarray:
    boundaries = np.asarray(cycle_boundaries, dtype=np.int64).reshape(-1)
    if boundaries.size < 2:
        raise ValueError("At least two cycle boundaries are required.")
    if signal_length <= 0:
        raise ValueError("signal_length must be positive.")

    inferred_index_base = index_base
    if inferred_index_base is None:
        if np.any(boundaries == 0):
            inferred_index_base = 0
        elif np.any(boundaries == signal_length):
            inferred_index_base = 1
        else:
            # Legacy Matlab outputs are one-based, so prefer that when ambiguous.
            inferred_index_base = 1

    normalized = boundaries - int(inferred_index_base)
    if np.any(np.diff(normalized) <= 0):
        raise ValueError("Cycle boundaries must be strictly increasing.")
    if normalized[0] < 0 or normalized[-1] >= signal_length:
        raise ValueError(
            "Cycle boundaries fall outside the available signal length after "
            f"normalization (index_base={inferred_index_base})."
        )
    return normalized.astype(np.int64, copy=False)
