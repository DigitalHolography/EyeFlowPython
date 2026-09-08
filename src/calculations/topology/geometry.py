"""Geometry for optic-disc-centered retinal regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentRingSettings:
    """Radial regions used to identify and sample vessel segments."""

    inner_radius_frac: float
    outer_radius_frac: float
    ring_width_frac: float
    ring_count: int
    segment_length_frac: float | None = None


def ring_masks(
    image_shape: tuple[int, int],
    optic_disc_center,
    settings: SegmentRingSettings,
) -> np.ndarray:
    """Return the configured non-overlapping annuli."""

    return np.asarray(
        [
            annulus_mask(
                image_shape,
                optic_disc_center,
                *_ring_bounds(settings, ring_index),
            )
            for ring_index in range(settings.ring_count)
        ],
        dtype=bool,
    )


def section_masks(
    image_shape: tuple[int, int],
    optic_disc_center,
    settings: SegmentRingSettings,
) -> np.ndarray:
    """Return the annuli in which branch-centered maps are sampled."""

    length = settings.segment_length_frac
    if length is None:
        length = settings.ring_width_frac
    return np.asarray(
        [
            annulus_mask(
                image_shape,
                optic_disc_center,
                *_ring_bounds(settings, ring_index, length),
            )
            for ring_index in range(settings.ring_count)
        ],
        dtype=bool,
    )


def annulus_mask(
    image_shape: tuple[int, int],
    optic_disc_center,
    inner_radius_frac: float,
    outer_radius_frac: float,
) -> np.ndarray:
    """Return a circular annulus centered on the optic disc.

    Radii are fractions of the image half-diagonal, so moving the optic disc
    translates the annulus without changing its size.
    """

    ny, nx = image_shape
    cy, cx = optic_disc_center_yx(optic_disc_center, ny, nx)
    scale = np.float32(1.0 / max(image_half_diagonal(ny, nx), 1.0))
    y_distance = (
        np.arange(ny, dtype=np.float32)[:, None] - np.float32(cy)
    ) * scale
    x_distance = (
        np.arange(nx, dtype=np.float32)[None, :] - np.float32(cx)
    ) * scale
    radius_sq = x_distance**2 + y_distance**2
    return (radius_sq > inner_radius_frac**2) & (radius_sq <= outer_radius_frac**2)


def optic_disc_center_yx(optic_disc_center, ny: int, nx: int) -> tuple[float, float]:
    """Return a valid ``(y, x)`` center, falling back to the image center."""

    if optic_disc_center is None:
        return ny / 2.0, nx / 2.0
    center = np.asarray(optic_disc_center, dtype=np.float32).reshape(-1)
    if center.size < 2 or not np.all(np.isfinite(center[:2])):
        return ny / 2.0, nx / 2.0
    return float(center[1]), float(center[0])


def _ring_bounds(
    settings: SegmentRingSettings,
    ring_index: int,
    length: float | None = None,
) -> tuple[float, float]:
    inner = settings.inner_radius_frac + ring_index * settings.ring_width_frac
    if length is None:
        length = settings.ring_width_frac
    return inner, min(settings.outer_radius_frac, inner + length)


def image_half_diagonal(ny: int, nx: int) -> float:
    """Return the center-independent image half-diagonal in pixels."""

    return float(np.hypot((ny - 1) / 2.0, (nx - 1) / 2.0))

