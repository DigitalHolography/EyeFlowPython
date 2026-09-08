"""Logical execution order for reusable retinal segment topology."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import SegmentRingSettings
from .segments import SegmentTopology, build_segment_topology, extract_segments
from .transforms import (
    determine_segment_rotations,
    interpolate_segment_masks,
    interpolate_segments,
    rotate_segment_masks,
    rotate_segments,
)


@dataclass(frozen=True)
class PreparedTopology:
    """Map-independent segment geometry ready for map transformations."""

    topology: SegmentTopology
    rotation_degrees: np.ndarray
    interpolated_masks: np.ndarray
    rotated_masks: np.ndarray


@dataclass(frozen=True)
class PreparedSegments:
    """Uniform non-rotated and upright views extracted from one retinal map."""

    interpolated: np.ndarray
    rotated: np.ndarray


def prepare_topology(
    vessel_mask,
    optic_disc_mask,
    settings: SegmentRingSettings,
    *,
    output_side_pixels: int = 128,
    window_size_percentile_kept: float = 0.95,
    window_side_pixels: int | None = None,
) -> PreparedTopology:
    """Build segment geometry, orientations, and uniform masks once."""

    topology = build_segment_topology(
        vessel_mask,
        optic_disc_mask,
        settings,
        window_size_percentile_kept=window_size_percentile_kept,
        window_side_pixels=window_side_pixels,
    )
    rotation_degrees = determine_segment_rotations(topology)
    interpolated_masks = interpolate_segment_masks(
        topology.segment_masks,
        output_side_pixels,
    )
    return PreparedTopology(
        topology=topology,
        rotation_degrees=rotation_degrees,
        interpolated_masks=interpolated_masks,
        rotated_masks=rotate_segment_masks(
            interpolated_masks,
            rotation_degrees,
        ),
    )


def prepare_segments(
    data_map,
    prepared_topology: PreparedTopology,
    *,
    spatial_axes: tuple[int, int] = (-2, -1),
) -> PreparedSegments:
    """Extract, uniformly interpolate, and rotate one map's segment arrays."""

    extracted = extract_segments(
        data_map,
        prepared_topology.topology,
        spatial_axes=spatial_axes,
    )
    interpolated = interpolate_segments(
        extracted,
        prepared_topology.interpolated_masks.shape[-1],
    )
    return PreparedSegments(
        interpolated=interpolated,
        rotated=rotate_segments(
            interpolated,
            prepared_topology.rotation_degrees,
        ),
    )
