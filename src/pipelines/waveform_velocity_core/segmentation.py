"""Pack segmentation products created by the waveform velocity core."""

from __future__ import annotations

import numpy as np

from calculations.topology import (
    optic_disc_center_yx,
)
from input_output.schema import EyeFlowOutputPaths

from .dopplerview.outputs import metric_data

OPTIC_DISC_LABEL = -1
REGION_AXIS_LABEL = -2


def pack_segmentation_outputs(
    source_data,
    artery_segments,
    vein_segments,
    output_paths: EyeFlowOutputPaths | str | None = None,
) -> dict[str, object]:
    """Pack masks and enriched branch-label maps below ``Segmentation``.

    The source arrays use EyeFlow's image frame, whose Y direction is inverted
    relative to the lower-left image frame used by the published maps.  All
    maps are therefore flipped vertically and transposed to ``(x, y)`` before
    writing. Quadrant calculations continue to use the original in-memory
    arrays and do not consume these visualization overlays.
    """
    schema = _resolve_output_paths(output_paths)
    image_shape = tuple(int(size) for size in source_data.retinal_artery_mask.shape)
    optic_disc_mask, mask_source = _optic_disc_mask(source_data, image_shape)
    center_xy = _optic_disc_center_xy(source_data.optic_disc_center, image_shape)

    segmentation = schema.segmentation
    metrics = {
        segmentation.optic_disc.mask: _segmentation_value(
            _serialize_spatial_image(optic_disc_mask),
            _mask_attrs(mask_source),
        ),
    }
    metrics.update(
        _pack_vessel_segmentation(
            segmentation.artery,
            artery_segments,
            source_data.retinal_artery_mask,
            optic_disc_mask,
            center_xy,
        )
    )
    metrics.update(
        _pack_vessel_segmentation(
            segmentation.vein,
            vein_segments,
            source_data.retinal_vein_mask,
            optic_disc_mask,
            center_xy,
        )
    )
    return metrics


def _pack_vessel_segmentation(
    paths,
    segments,
    vessel_mask,
    optic_disc_mask: np.ndarray,
    center_xy: np.ndarray,
) -> dict[str, object]:
    expected_shape = tuple(int(size) for size in vessel_mask.shape)
    labels = (
        np.zeros(expected_shape, dtype=np.int32)
        if segments is None
        else np.asarray(segments.labels, dtype=np.int32)
    )
    if labels.shape != expected_shape:
        raise ValueError(
            f"segment labels must have shape {expected_shape}, got {labels.shape}."
        )

    return {
        paths.mask: _segmentation_value(
            _serialize_spatial_image(np.asarray(vessel_mask, dtype=bool)),
            _mask_attrs("dopplerview_segmentation"),
        ),
        paths.branch_label_map: _segmentation_value(
            _serialize_branch_label_map(labels, optic_disc_mask, center_xy),
            _branch_label_attrs(_axis_thickness(labels.shape)),
        ),
    }


def _serialize_branch_label_map(
    labels: np.ndarray,
    optic_disc_mask: np.ndarray,
    center_xy: np.ndarray,
) -> np.ndarray:
    normalized_labels = np.flip(labels, axis=0).copy()
    normalized_optic_disc_mask = np.flip(optic_disc_mask, axis=0)
    normalized_center = center_xy.copy()
    normalized_center[1] = labels.shape[0] - 1 - normalized_center[1]

    image = normalized_labels
    image[normalized_optic_disc_mask] = OPTIC_DISC_LABEL
    axis_thickness = _axis_thickness(labels.shape)
    center_x = int(np.floor(normalized_center[0]))
    center_y = int(np.floor(normalized_center[1]))
    x_start = max(0, center_x - axis_thickness // 2)
    x_stop = min(labels.shape[1], x_start + axis_thickness)
    y_start = max(0, center_y - axis_thickness // 2)
    y_stop = min(labels.shape[0], y_start + axis_thickness)
    image[:, x_start:x_stop] = REGION_AXIS_LABEL
    image[y_start:y_stop, :] = REGION_AXIS_LABEL
    return image.T.copy()


def _serialize_spatial_image(image: np.ndarray) -> np.ndarray:
    return np.flip(np.asarray(image), axis=0).T.copy()


def _axis_thickness(image_shape: tuple[int, int]) -> int:
    return max(3, int(round(min(image_shape) / 128.0)))


def _mask_attrs(source: str) -> dict[str, object]:
    return {
        "dimDesc": ["x", "y"],
        "coordinate_system": "image_pixel",
        "image_origin": "lower_left",
        "source": source,
        "y_axis_direction": "increasing_toward_north",
    }


def _branch_label_attrs(axis_thickness: int) -> dict[str, object]:
    return {
        "axis_label": REGION_AXIS_LABEL,
        "axis_thickness_pixels": axis_thickness,
        "background_label": 0,
        "branch_labels": "original in-memory branch labels",
        "coordinate_system": "image_pixel",
        "description": (
            "Two-dimensional branch label map with optic-disc and "
            "quadrant-axis overlays"
        ),
        "dimDesc": ["x", "y"],
        "image_origin": "lower_left",
        "optic_disc_label": OPTIC_DISC_LABEL,
        "overlay_priority": "quadrant axes, optic disc, vessel branches",
        "y_axis_direction": "increasing_toward_north",
    }


def _segmentation_value(data, attrs: dict[str, object]):
    return metric_data(data), attrs


def _optic_disc_mask(source_data, image_shape: tuple[int, int]):
    if source_data.optic_disc_mask is not None:
        mask = np.asarray(source_data.optic_disc_mask, dtype=bool)
        if mask.shape != image_shape:
            raise ValueError(
                f"optic_disc_mask must have shape {image_shape}, got {mask.shape}."
            )
        return mask, "dopplerview_segmentation"

    mask = _ellipse_mask(
        image_shape,
        source_data.optic_disc_center,
        source_data.optic_disc_width,
        source_data.optic_disc_height,
    )
    if np.any(mask):
        return mask, "reconstructed_from_dopplerview_center_width_height"
    return mask, "unavailable"


def _ellipse_mask(
    image_shape: tuple[int, int],
    optic_disc_center,
    optic_disc_width,
    optic_disc_height,
) -> np.ndarray:
    width = _positive_scalar(optic_disc_width)
    height = _positive_scalar(optic_disc_height)
    if width is None or height is None:
        return np.zeros(image_shape, dtype=bool)

    center_x, center_y = _optic_disc_center_xy(optic_disc_center, image_shape)
    y, x = np.indices(image_shape, dtype=np.float32)
    x_radius = np.float32(width / 2.0)
    y_radius = np.float32(height / 2.0)
    return (
        ((x - center_x) / x_radius) ** 2
        + ((y - center_y) / y_radius) ** 2
        <= 1.0
    )


def _optic_disc_center_xy(optic_disc_center, image_shape: tuple[int, int]) -> np.ndarray:
    center_y, center_x = optic_disc_center_yx(
        optic_disc_center,
        image_shape[0],
        image_shape[1],
    )
    return np.asarray([center_x, center_y], dtype=np.float32)


def _positive_scalar(value) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array[0]) or array[0] <= 0:
        return None
    return float(array[0])


def _resolve_output_paths(
    output_paths: EyeFlowOutputPaths | str | None,
) -> EyeFlowOutputPaths:
    if isinstance(output_paths, EyeFlowOutputPaths):
        return output_paths
    return EyeFlowOutputPaths.active(output_paths)
