"""Export branch identity stages as PNG debug artifacts."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from calculations.topology import (
    BranchIdentityStages,
    SegmentRingSettings,
    annulus_mask,
)

def export_branch_identity_stage_pngs(
    output,
    stages: BranchIdentityStages,
    prefix: str,
    optic_disc_center,
    ring_settings: SegmentRingSettings,
    *,
    segment_center_xy: np.ndarray | None = None,
    profile_window_bounds_xyxy: np.ndarray | None = None,
) -> list[str]:
    paths = []
    stage_images = list(_stage_images(stages, optic_disc_center, ring_settings))
    if segment_center_xy is not None and profile_window_bounds_xyxy is not None:
        stage_images.append(
            (
                "13_substack_boxes_on_labels_with_rings",
                _labels_with_substack_boxes(
                    stages.per_circle_cleaned_labels,
                    optic_disc_center,
                    ring_settings,
                    segment_center_xy,
                    profile_window_bounds_xyxy,
                ),
            )
        )
    for name, image in stage_images:
        path = output.write_png(
            image,
            f"branch_identity/{prefix}_{name}.png",
        )
        paths.append(str(path))
    return paths


def _stage_images(
    stages: BranchIdentityStages,
    optic_disc_center,
    ring_settings: SegmentRingSettings,
):
    return (
        ("01_input", _mask_image(stages.vessel)),
        ("02_mask_section", _mask_image(stages.section)),
        ("03_skeleton", _mask_image(stages.skeleton)),
        ("04_branch_points", _overlay_points(stages.skeleton, stages.branch_points)),
        ("05_cleaned_skeleton", _mask_image(stages.cleaned_skeleton)),
        ("06_marker_labels", _label_image(stages.marker_labels)),
        ("07_distance_topography", _topography_image(stages.distance_topography)),
        ("08_imposed_minima_topography", _topography_image(stages.imposed_minima_topography)),
        ("09_watershed_labels", _label_image(stages.watershed_labels)),
        ("10_annulus_refined_labels", _label_image(stages.annulus_refined_labels)),
        ("11_per_circle_cleaned_labels", _label_image(stages.per_circle_cleaned_labels)),
        (
            "12_per_circle_cleaned_labels_with_rings",
            _labels_with_ring_overlay(
                stages.per_circle_cleaned_labels,
                optic_disc_center,
                ring_settings,
            ),
        ),
    )


def _mask_image(mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[np.asarray(mask, dtype=bool)] = (255, 255, 255)
    return image


def _overlay_points(base: np.ndarray, points: np.ndarray) -> np.ndarray:
    image = _mask_image(base)
    image[np.asarray(points, dtype=bool)] = (255, 0, 0)
    return image


def _label_image(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    image = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label_id in np.unique(labels[labels > 0]):
        image[labels == label_id] = _label_color(int(label_id))
    return image


def _labels_with_ring_overlay(
    labels: np.ndarray,
    optic_disc_center,
    settings: SegmentRingSettings,
) -> np.ndarray:
    image = _label_image(labels)
    image[_ring_boundaries(labels.shape, optic_disc_center, settings)] = (255, 255, 255)
    return image


def _labels_with_substack_boxes(
    labels: np.ndarray,
    optic_disc_center,
    ring_settings: SegmentRingSettings,
    segment_center_xy: np.ndarray,
    profile_window_bounds_xyxy: np.ndarray,
) -> np.ndarray:
    """Overlay the actual rotating substacks on the stage-12 label image.

    Box colors identify the ring. Pixels shared by two or more box outlines
    are magenta, making exact box overlap visible instead of hiding it under
    the later box drawn.
    """
    image = _labels_with_ring_overlay(labels, optic_disc_center, ring_settings)
    centers = np.asarray(segment_center_xy, dtype=np.float32)
    bounds = np.asarray(profile_window_bounds_xyxy, dtype=np.int32)
    if (
        centers.ndim != 3
        or centers.shape[2] != 2
        or bounds.ndim != 3
        or bounds.shape[2] != 4
    ):
        return image

    box_counts = np.zeros(labels.shape, dtype=np.uint8)
    box_colors = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for branch_index in range(centers.shape[0]):
        for ring_index in range(
            min(centers.shape[1], int(ring_settings.ring_count))
        ):
            center = centers[branch_index, ring_index]
            if (
                not np.all(np.isfinite(center))
                or ring_index >= bounds.shape[0]
                or branch_index >= bounds.shape[1]
            ):
                continue
            x_start, x_stop, y_start, y_stop = (
                int(value) for value in bounds[ring_index, branch_index]
            )
            if x_start >= x_stop or y_start >= y_stop:
                continue
            boundary = np.zeros(labels.shape, dtype=bool)
            boundary[y_start, x_start:x_stop] = True
            boundary[y_stop - 1, x_start:x_stop] = True
            boundary[y_start:y_stop, x_start] = True
            boundary[y_start:y_stop, x_stop - 1] = True
            box_counts[boundary] = np.minimum(box_counts[boundary] + 1, 255)
            box_colors[boundary] = _ring_box_color(ring_index)

    single_box = box_counts == 1
    overlapping_boxes = box_counts > 1
    image[single_box] = box_colors[single_box]
    image[overlapping_boxes] = (255, 0, 255)
    return image


def _ring_box_color(ring_index: int) -> tuple[int, int, int]:
    colors = (
        (255, 255, 0),
        (0, 255, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 64, 192),
        (64, 192, 255),
        (255, 64, 64),
        (128, 255, 64),
        (192, 64, 255),
        (64, 255, 192),
    )
    return colors[ring_index % len(colors)]


def _ring_boundaries(
    image_shape: tuple[int, int],
    optic_disc_center,
    settings: SegmentRingSettings,
) -> np.ndarray:
    boundaries = np.zeros(image_shape, dtype=bool)
    for ring_index in range(settings.ring_count):
        ring_inner = settings.inner_radius_frac + ring_index * settings.ring_width_frac
        ring = annulus_mask(
            image_shape,
            optic_disc_center,
            ring_inner,
            ring_inner + settings.ring_width_frac,
        )
        boundaries |= ring & ~ndi.binary_erosion(ring)
    return boundaries


def _topography_image(topography: np.ndarray) -> np.ndarray:
    finite = np.isfinite(topography)
    image = np.zeros((*topography.shape, 3), dtype=np.uint8)
    image[np.isposinf(topography)] = (0, 0, 48)
    image[np.isneginf(topography)] = (255, 0, 0)
    if np.any(finite):
        values = topography[finite]
        span = np.max(values) - np.min(values)
        if span <= 0:
            gray = np.full(values.shape, 255, dtype=np.uint8)
        else:
            gray = np.rint((values - np.min(values)) / span * 255).astype(np.uint8)
        image[finite] = np.column_stack((gray, gray, gray))
    return image


def _label_color(label_id: int) -> tuple[int, int, int]:
    hue = (label_id * 0.61803398875) % 1.0
    return _hsv_to_rgb(hue, 0.75, 1.0)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    chroma = value * saturation
    x = chroma * (1 - abs((hue * 6) % 2 - 1))
    match int(hue * 6):
        case 0:
            rgb = (chroma, x, 0)
        case 1:
            rgb = (x, chroma, 0)
        case 2:
            rgb = (0, chroma, x)
        case 3:
            rgb = (0, x, chroma)
        case 4:
            rgb = (x, 0, chroma)
        case _:
            rgb = (chroma, 0, x)
    m = value - chroma
    return tuple(int(round((channel + m) * 255)) for channel in rgb)
