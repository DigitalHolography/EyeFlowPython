"""Find vessel segments and extract the corresponding regions from any map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .branch_identity import label_vessel_branches
from .geometry import SegmentRingSettings, section_masks


@dataclass(frozen=True)
class SegmentTopology:
    """Map-independent locations of vessel segments.

    Arrays indexed by segment use ``(annulus, branch, ...)`` ordering.
    ``centerline`` is the full-frame skeleton used during branch identification.
    ``segment_masks`` are fixed-size local masks rather than full-frame masks.
    """

    spatial_shape: tuple[int, int]
    optic_disc_center_xy: tuple[float, float]
    labels: np.ndarray
    centerline: np.ndarray
    branch_ids: np.ndarray
    annulus_masks: np.ndarray
    segment_masks: np.ndarray
    segment_centers_xy: np.ndarray
    window_bounds_xyxy: np.ndarray
    window_side_pixels: int

    @property
    def valid_segments(self) -> np.ndarray:
        return np.all(np.isfinite(self.segment_centers_xy), axis=-1)


def build_segment_topology(
    vessel_mask,
    optic_disc_mask,
    settings: SegmentRingSettings,
    *,
    window_size_percentile_kept: float = 0.95,
    window_side_pixels: int | None = None,
) -> SegmentTopology:
    """Find branch segments around an optic-disc mask.

    Args:
        vessel_mask: Two-dimensional mask containing one vessel class.
        optic_disc_mask: Two-dimensional optic-disc mask. An empty mask uses
            the image center.
        settings: Annulus placement and sampling settings.
        window_size_percentile_kept: Fraction of segment widths and heights
            represented by the automatically selected square window.
        window_side_pixels: Explicit odd window side, overriding the percentile.

    Returns:
        Geometry reusable to extract the same segments from any retinal map.
    """

    vessel = np.asarray(vessel_mask, dtype=bool)
    disc = np.asarray(optic_disc_mask, dtype=bool)
    assert vessel.ndim == 2
    assert disc.shape == vessel.shape
    center_xy = _mask_center_xy(disc, vessel.shape)
    return _build_segment_topology_from_center(
        vessel,
        center_xy,
        settings,
        window_size_percentile_kept=window_size_percentile_kept,
        window_side_pixels=window_side_pixels,
    )


def extract_segments(
    data_map,
    topology: SegmentTopology,
    *,
    spatial_axes: tuple[int, int] = (-2, -1),
) -> np.ndarray:
    """Extract one padded square per annulus and branch.

    Spatial axes are moved to the last two output axes. Other axes retain their
    relative order, so a ``(frame, y, x, component)`` vector map produces
    ``(annulus, branch, frame, component, local_y, local_x)`` when called with
    ``spatial_axes=(1, 2)``.
    """

    shape = tuple(int(size) for size in data_map.shape)
    y_axis, x_axis = _normalized_spatial_axes(len(shape), spatial_axes)
    assert (shape[y_axis], shape[x_axis]) == topology.spatial_shape
    nonspatial_shape = tuple(
        size for axis, size in enumerate(shape) if axis not in (y_axis, x_axis)
    )
    ring_count, branch_count = topology.segment_centers_xy.shape[:2]
    side = topology.window_side_pixels
    extracted = np.full(
        (ring_count, branch_count, *nonspatial_shape, side, side),
        np.nan,
        dtype=np.float32,
    )
    if side == 0:
        return extracted

    target_prefix = (slice(None),) * len(nonspatial_shape)
    for ring_index, branch_index in np.argwhere(topology.valid_segments):
        bounds = topology.window_bounds_xyxy[ring_index, branch_index]
        center = topology.segment_centers_xy[ring_index, branch_index]
        source_slices = [slice(None)] * len(shape)
        source_slices[x_axis] = slice(int(bounds[0]), int(bounds[1]))
        source_slices[y_axis] = slice(int(bounds[2]), int(bounds[3]))
        source = np.asarray(data_map[tuple(source_slices)], dtype=np.float32)
        source = np.moveaxis(source, (y_axis, x_axis), (-2, -1))
        target_y, target_x = _window_target_slices(bounds, center, side)
        extracted[
            int(ring_index),
            int(branch_index),
            *target_prefix,
            target_y,
            target_x,
        ] = source
    return extracted


def _build_segment_topology_from_center(
    vessel_mask: np.ndarray,
    optic_disc_center_xy: tuple[float, float],
    settings: SegmentRingSettings,
    *,
    window_size_percentile_kept: float,
    window_side_pixels: int | None,
) -> SegmentTopology:
    branches = label_vessel_branches(vessel_mask, optic_disc_center_xy, settings)
    centerline = branches.stages.skeleton
    annuli = section_masks(vessel_mask.shape, optic_disc_center_xy, settings)
    side = (
        _segment_window_side(
            branches.labels,
            branches.branch_ids,
            annuli,
            window_size_percentile_kept,
        )
        if window_side_pixels is None
        else int(window_side_pixels)
    )
    if side < 0 or (side > 0 and side % 2 == 0):
        raise ValueError("window_side_pixels must be zero or a positive odd integer.")

    ring_count = int(annuli.shape[0])
    branch_count = int(branches.branch_ids.size)
    centers = np.full((ring_count, branch_count, 2), np.nan, dtype=np.float32)
    bounds = np.full((ring_count, branch_count, 4), -1, dtype=np.int32)
    masks = np.zeros((ring_count, branch_count, side, side), dtype=bool)
    for ring_index, annulus in enumerate(annuli):
        for branch_index, branch_id in enumerate(branches.branch_ids):
            mask = annulus & (branches.labels == int(branch_id))
            segment_centerline = centerline & mask
            center = _segment_center_xy(mask, segment_centerline)
            if center is None:
                continue
            segment_bounds = _centered_window_bounds(vessel_mask.shape, center, side)
            centers[ring_index, branch_index] = center
            bounds[ring_index, branch_index] = segment_bounds
            target_y, target_x = _window_target_slices(segment_bounds, center, side)
            x_start, x_stop, y_start, y_stop = segment_bounds
            masks[ring_index, branch_index, target_y, target_x] = mask[
                y_start:y_stop,
                x_start:x_stop,
            ]

    return SegmentTopology(
        spatial_shape=tuple(vessel_mask.shape),
        optic_disc_center_xy=tuple(float(value) for value in optic_disc_center_xy),
        labels=branches.labels,
        centerline=centerline,
        branch_ids=branches.branch_ids,
        annulus_masks=annuli,
        segment_masks=masks,
        segment_centers_xy=centers,
        window_bounds_xyxy=bounds,
        window_side_pixels=side,
    )


def _mask_center_xy(
    mask: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    if not np.any(mask):
        return image_shape[1] / 2.0, image_shape[0] / 2.0
    center_y, center_x = ndi.center_of_mass(mask)
    return float(center_x), float(center_y)


def _segment_center_xy(
    segment_mask: np.ndarray,
    segment_centerline: np.ndarray,
) -> tuple[int, int] | None:
    source = segment_centerline if np.any(segment_centerline) else segment_mask
    if not np.any(source):
        return None
    point_y, point_x = np.nonzero(source)
    center_x = int(np.floor(np.median(point_x) + 0.5))
    center_y = int(np.floor(np.median(point_y) + 0.5))
    return center_x, center_y


def _segment_window_side(
    labels: np.ndarray,
    branch_ids: np.ndarray,
    annuli: np.ndarray,
    percentile_kept: float,
) -> int:
    percentile = float(percentile_kept)
    if not 0.0 < percentile <= 1.0:
        raise ValueError("window_size_percentile_kept must be in (0, 1].")

    widths: list[int] = []
    heights: list[int] = []
    for annulus in annuli:
        for branch_id in branch_ids:
            segment_y, segment_x = np.nonzero(annulus & (labels == int(branch_id)))
            if segment_x.size == 0:
                continue
            widths.append(int(segment_x.max() - segment_x.min() + 1))
            heights.append(int(segment_y.max() - segment_y.min() + 1))
    if not widths:
        return 0

    width = int(np.quantile(widths, percentile, method="higher"))
    height = int(np.quantile(heights, percentile, method="higher"))
    side = max(width, height)
    return side if side % 2 == 1 else side + 1


def _centered_window_bounds(
    image_shape: tuple[int, int],
    center_xy: tuple[int, int],
    side_pixels: int,
) -> tuple[int, int, int, int]:
    assert side_pixels > 0 and side_pixels % 2 == 1
    half_width = side_pixels // 2
    center_x, center_y = center_xy
    return (
        max(center_x - half_width, 0),
        min(center_x + half_width + 1, int(image_shape[1])),
        max(center_y - half_width, 0),
        min(center_y + half_width + 1, int(image_shape[0])),
    )


def _window_target_slices(
    bounds_xyxy,
    center_xy,
    side_pixels: int,
) -> tuple[slice, slice]:
    x_start, x_stop, y_start, y_stop = (int(value) for value in bounds_xyxy)
    center_x, center_y = (int(value) for value in center_xy)
    conceptual_x_start = center_x - side_pixels // 2
    conceptual_y_start = center_y - side_pixels // 2
    target_x_start = x_start - conceptual_x_start
    target_y_start = y_start - conceptual_y_start
    return (
        slice(target_y_start, target_y_start + y_stop - y_start),
        slice(target_x_start, target_x_start + x_stop - x_start),
    )


def _normalized_spatial_axes(
    dimension_count: int,
    spatial_axes: tuple[int, int],
) -> tuple[int, int]:
    if dimension_count < 2:
        raise ValueError("data_map must contain two spatial axes.")
    y_axis, x_axis = (int(axis) % dimension_count for axis in spatial_axes)
    if y_axis == x_axis:
        raise ValueError("spatial_axes must identify two different axes.")
    return y_axis, x_axis
