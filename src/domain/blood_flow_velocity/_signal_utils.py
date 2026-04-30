"""Shared helpers for blood-flow velocity signal analysis."""

from __future__ import annotations

import numpy as np


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

    inferred_index_base = _infer_index_base(boundaries, signal_length, index_base)
    normalized = boundaries - int(inferred_index_base)
    _validate_cycle_boundaries(normalized, signal_length, inferred_index_base)
    return normalized.astype(np.int64, copy=False)


def _infer_index_base(
    boundaries: np.ndarray,
    signal_length: int,
    index_base: int | None,
) -> int:
    if index_base is not None:
        return int(index_base)
    if np.any(boundaries == 0):
        return 0
    if np.any(boundaries == signal_length):
        return 1
    return 1


def _validate_cycle_boundaries(
    normalized: np.ndarray,
    signal_length: int,
    index_base: int,
) -> None:
    if np.any(np.diff(normalized) <= 0):
        raise ValueError("Cycle boundaries must be strictly increasing.")
    if normalized[0] >= 0 and normalized[-1] < signal_length:
        return
    raise ValueError(
        "Cycle boundaries fall outside the available signal length after "
        f"normalization (index_base={index_base})."
    )
