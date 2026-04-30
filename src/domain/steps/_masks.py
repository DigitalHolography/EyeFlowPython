"""Mask helpers used by migrated DopplerView calculation steps."""

from __future__ import annotations

import numpy as np


def elliptical_mask(ny: int, nx: int, radius_frac: float, center=None) -> np.ndarray:
    radius = max(0.0, min(1.0, float(radius_frac)))
    semi_x = max((nx / 2.0) * radius, np.finfo(float).eps)
    semi_y = max((ny / 2.0) * radius, np.finfo(float).eps)
    y_grid, x_grid = np.ogrid[:ny, :nx]
    cy, cx = (ny / 2.0, nx / 2.0) if center is None else center
    normalized = ((x_grid - cx) / semi_x) ** 2 + ((y_grid - cy) / semi_y) ** 2
    return normalized <= 1.0


def elliptical_annulus_mask(
    ny: int,
    nx: int,
    *,
    outer_radius_frac: float = 0.5,
    inner_radius_frac: float = 0.2,
) -> np.ndarray:
    outer = elliptical_mask(ny, nx, outer_radius_frac)
    inner = elliptical_mask(ny, nx, inner_radius_frac)
    return outer & ~inner


def binary_dilation(mask, radius: int) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    radius = int(max(radius, 0))
    if radius == 0:
        return mask_array.copy()

    footprint = _disk_footprint(radius)
    padded = np.pad(mask_array, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask_array, dtype=bool)
    for row, col in np.argwhere(footprint):
        dilated |= padded[
            row : row + mask_array.shape[0],
            col : col + mask_array.shape[1],
        ]
    return dilated


def _disk_footprint(radius: int) -> np.ndarray:
    coords = np.arange(-radius, radius + 1)
    y_grid, x_grid = np.meshgrid(coords, coords, indexing="ij")
    return (x_grid**2 + y_grid**2) <= radius**2
