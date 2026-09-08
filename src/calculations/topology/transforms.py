"""Map-independent interpolation and orientation of retinal segments."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from calculations.compute_backend import optional_cupy_backend

from .geometry import image_half_diagonal, optic_disc_center_yx
from .segments import SegmentTopology


INTERPOLATED_SEGMENT_SIDE = 128


def determine_segment_rotations(topology: SegmentTopology) -> np.ndarray:
    """Return the rotation that makes every valid vessel segment upright.

    The returned array has ``(annulus, branch)`` shape. Directions are measured
    along the existing centerline and point away from the optic disc. This
    preserves a consistent sign for later vector-basis correction.
    """

    rotations = np.full(topology.valid_segments.shape, np.nan, dtype=np.float32)
    for annulus_index, branch_index in np.argwhere(topology.valid_segments):
        segment = topology.annulus_masks[annulus_index] & (
            topology.labels == int(topology.branch_ids[branch_index])
        )
        segment_centerline = topology.centerline & segment
        tilt = _segment_tilt(
            segment_centerline,
            topology.optic_disc_center_xy,
        )
        if np.isfinite(tilt):
            rotations[annulus_index, branch_index] = np.float32(tilt + 90.0)
    return rotations


def interpolate_segments(
    segment_maps: np.ndarray,
    output_side_pixels: int = INTERPOLATED_SEGMENT_SIDE,
) -> np.ndarray:
    """Interpolate segment values to one square size while preserving NaNs.

    Only the final two axes are resized; every preceding axis is retained.
    """

    values = np.asarray(segment_maps, dtype=np.float32)
    _assert_spatial_array(values, output_side_pixels)
    if values.shape[-2:] == (output_side_pixels, output_side_pixels):
        return values.copy()
    if values.shape[-2] == 0 or values.shape[-1] == 0:
        return np.full(
            (*values.shape[:-2], output_side_pixels, output_side_pixels),
            np.nan,
            dtype=np.float32,
        )
    return _interpolate_values(values, output_side_pixels)


def interpolate_segment_masks(
    segment_masks: np.ndarray,
    output_side_pixels: int = INTERPOLATED_SEGMENT_SIDE,
) -> np.ndarray:
    """Interpolate Boolean segment masks with nearest-neighbor sampling."""

    masks = np.asarray(segment_masks, dtype=bool)
    _assert_spatial_array(masks, output_side_pixels)
    if masks.shape[-2:] == (output_side_pixels, output_side_pixels):
        return masks.copy()
    output_shape = (*masks.shape[:-2], output_side_pixels, output_side_pixels)
    if masks.shape[-2] == 0 or masks.shape[-1] == 0:
        return np.zeros(output_shape, dtype=bool)

    zoom = _spatial_zoom(masks, output_side_pixels)
    backend = optional_cupy_backend()
    if backend is not None:
        try:
            resized = backend.ndimage.zoom(
                backend.cupy.asarray(masks, dtype=backend.cupy.float32),
                zoom,
                order=0,
                mode="grid-constant",
                cval=0.0,
                prefilter=False,
                grid_mode=True,
            )
            return backend.cupy.asnumpy(resized) >= np.float32(0.5)
        except Exception:
            pass
    return ndi.zoom(
        masks.astype(np.float32),
        zoom,
        order=0,
        mode="grid-constant",
        cval=0.0,
        prefilter=False,
        grid_mode=True,
    ) >= np.float32(0.5)


def rotate_segments(
    interpolated_maps: np.ndarray,
    rotation_degrees: np.ndarray,
) -> np.ndarray:
    """Rotate uniformly sized segment maps on a diagonal-sized canvas.

    ``interpolated_maps`` must use ``(annulus, branch, ..., y, x)`` ordering.
    The rotation array must have matching ``(annulus, branch)`` dimensions.
    """

    values = np.asarray(interpolated_maps, dtype=np.float32)
    _assert_segment_array(values, rotation_degrees)
    canvas_side = _rotation_canvas_side(values.shape[-1])
    rotated = np.full(
        (*values.shape[:-2], canvas_side, canvas_side),
        np.nan,
        dtype=np.float32,
    )
    for annulus_index, branch_index in np.argwhere(np.isfinite(rotation_degrees)):
        segment = _pad_for_rotation(values[annulus_index, branch_index], np.nan)
        rotated[annulus_index, branch_index] = _rotate_values(
            segment,
            float(rotation_degrees[annulus_index, branch_index]),
        )
    return rotated


def rotate_segment_masks(
    interpolated_masks: np.ndarray,
    rotation_degrees: np.ndarray,
) -> np.ndarray:
    """Rotate uniformly sized Boolean masks on a diagonal-sized canvas."""

    masks = np.asarray(interpolated_masks, dtype=bool)
    _assert_segment_array(masks, rotation_degrees)
    if masks.ndim != 4:
        raise ValueError("interpolated_masks must have (annulus, branch, y, x) shape.")
    canvas_side = _rotation_canvas_side(masks.shape[-1])
    rotated = np.zeros((*masks.shape[:-2], canvas_side, canvas_side), dtype=bool)
    for annulus_index, branch_index in np.argwhere(np.isfinite(rotation_degrees)):
        segment = _pad_for_rotation(masks[annulus_index, branch_index], False)
        rotated[annulus_index, branch_index] = ndi.rotate(
            segment.astype(np.float32),
            float(rotation_degrees[annulus_index, branch_index]),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ) >= np.float32(0.5)
    return rotated


def _segment_tilt(
    segment_centerline: np.ndarray,
    optic_disc_center_xy: tuple[float, float],
) -> float:
    point_y, point_x = np.nonzero(segment_centerline)
    if point_x.size < 2:
        return float("nan")

    points = np.column_stack((point_x, point_y)).astype(np.float64)
    center = np.median(points, axis=0)
    centered = points - center
    moment = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(moment)
    if eigenvalues[-1] <= eigenvalues[0]:
        return float("nan")

    axis = eigenvectors[:, -1]
    projections = centered @ axis
    if np.ptp(projections) < 1.0:
        return float("nan")

    radii = _radius_grid(segment_centerline.shape, optic_disc_center_xy)[
        point_y,
        point_x,
    ]
    lower_projection, upper_projection = np.quantile(
        projections,
        (0.25, 0.75),
    )
    lower_radius = float(
        np.median(radii[projections <= lower_projection])
    )
    upper_radius = float(
        np.median(radii[projections >= upper_projection])
    )
    if upper_radius < lower_radius:
        axis = -axis
    elif np.isclose(upper_radius, lower_radius):
        optic_center_y, optic_center_x = optic_disc_center_yx(
            optic_disc_center_xy,
            *segment_centerline.shape,
        )
        radial_direction = np.asarray(
            (center[0] - optic_center_x, center[1] - optic_center_y),
            dtype=np.float64,
        )
        radial_alignment = float(np.dot(axis, radial_direction))
        if np.isclose(radial_alignment, 0.0):
            return float("nan")
        if radial_alignment < 0.0:
            axis = -axis
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


def _radius_grid(
    image_shape: tuple[int, int],
    optic_disc_center_xy: tuple[float, float],
) -> np.ndarray:
    ny, nx = image_shape
    center_y, center_x = optic_disc_center_yx(optic_disc_center_xy, ny, nx)
    scale = np.float32(1.0 / max(image_half_diagonal(ny, nx), 1.0))
    y = (np.arange(ny, dtype=np.float32)[:, None] - np.float32(center_y)) * scale
    x = (np.arange(nx, dtype=np.float32)[None, :] - np.float32(center_x)) * scale
    return np.sqrt(x**2 + y**2)


def _interpolate_values(values: np.ndarray, output_side_pixels: int) -> np.ndarray:
    zoom = _spatial_zoom(values, output_side_pixels)
    backend = optional_cupy_backend()
    if backend is not None:
        try:
            gpu_values = backend.cupy.asarray(values)
            valid = backend.cupy.isfinite(gpu_values)
            resized_values = backend.ndimage.zoom(
                backend.cupy.where(valid, gpu_values, backend.cupy.float32(0.0)),
                zoom,
                order=1,
                mode="grid-constant",
                cval=0.0,
                prefilter=False,
                grid_mode=True,
            )
            resized_weights = backend.ndimage.zoom(
                valid.astype(backend.cupy.float32),
                zoom,
                order=1,
                mode="grid-constant",
                cval=0.0,
                prefilter=False,
                grid_mode=True,
            )
            resized = backend.cupy.full(
                resized_values.shape,
                backend.cupy.nan,
                dtype=backend.cupy.float32,
            )
            backend.cupy.divide(
                resized_values,
                resized_weights,
                out=resized,
                where=resized_weights > backend.cupy.float32(1e-6),
            )
            return backend.cupy.asnumpy(resized)
        except Exception:
            pass

    valid = np.isfinite(values)
    resized_values = ndi.zoom(
        np.where(valid, values, np.float32(0.0)),
        zoom,
        order=1,
        mode="grid-constant",
        cval=0.0,
        prefilter=False,
        grid_mode=True,
    ).astype(np.float32, copy=False)
    resized_weights = ndi.zoom(
        valid.astype(np.float32),
        zoom,
        order=1,
        mode="grid-constant",
        cval=0.0,
        prefilter=False,
        grid_mode=True,
    ).astype(np.float32, copy=False)
    resized = np.full(resized_values.shape, np.nan, dtype=np.float32)
    np.divide(
        resized_values,
        resized_weights,
        out=resized,
        where=resized_weights > np.float32(1e-6),
    )
    return resized


def _rotate_values(values: np.ndarray, angle_degrees: float) -> np.ndarray:
    backend = optional_cupy_backend()
    if backend is not None:
        try:
            gpu_values = backend.cupy.asarray(values, dtype=backend.cupy.float32)
            valid = backend.cupy.isfinite(gpu_values)
            rotated_values = backend.ndimage.rotate(
                backend.cupy.where(valid, gpu_values, backend.cupy.float32(0.0)),
                angle_degrees,
                axes=(-2, -1),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            rotated_weights = backend.ndimage.rotate(
                valid.astype(backend.cupy.float32),
                angle_degrees,
                axes=(-2, -1),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            rotated = backend.cupy.full(
                rotated_values.shape,
                backend.cupy.nan,
                dtype=backend.cupy.float32,
            )
            backend.cupy.divide(
                rotated_values,
                rotated_weights,
                out=rotated,
                where=rotated_weights >= backend.cupy.float32(0.5),
            )
            return backend.cupy.asnumpy(rotated)
        except Exception:
            pass

    valid = np.isfinite(values)
    rotated_values = ndi.rotate(
        np.where(valid, values, np.float32(0.0)),
        angle_degrees,
        axes=(-2, -1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    rotated_weights = ndi.rotate(
        valid.astype(np.float32),
        angle_degrees,
        axes=(-2, -1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    rotated = np.full(rotated_values.shape, np.nan, dtype=np.float32)
    np.divide(
        rotated_values,
        rotated_weights,
        out=rotated,
        where=rotated_weights >= np.float32(0.5),
    )
    return rotated.astype(np.float32, copy=False)


def _spatial_zoom(values: np.ndarray, output_side_pixels: int) -> tuple[float, ...]:
    return (1.0,) * (values.ndim - 2) + (
        output_side_pixels / values.shape[-2],
        output_side_pixels / values.shape[-1],
    )


def _rotation_canvas_side(interpolated_side_pixels: int) -> int:
    return int(np.ceil(np.sqrt(2.0) * (interpolated_side_pixels - 1))) + 1


def _pad_for_rotation(
    values: np.ndarray,
    fill_value: float | bool,
) -> np.ndarray:
    canvas_side = _rotation_canvas_side(values.shape[-1])
    total_padding = canvas_side - values.shape[-1]
    padding_before = total_padding // 2
    padding_after = total_padding - padding_before
    padding = [(0, 0)] * values.ndim
    padding[-2] = (padding_before, padding_after)
    padding[-1] = (padding_before, padding_after)
    return np.pad(values, padding, mode="constant", constant_values=fill_value)


def _assert_spatial_array(values: np.ndarray, output_side_pixels: int) -> None:
    if values.ndim < 2:
        raise ValueError("segment arrays must end with spatial (y, x) axes.")
    if output_side_pixels <= 0:
        raise ValueError("output_side_pixels must be positive.")


def _assert_segment_array(
    values: np.ndarray,
    rotation_degrees: np.ndarray,
) -> None:
    if values.ndim < 4:
        raise ValueError(
            "segment arrays must have (annulus, branch, ..., y, x) shape."
        )
    if values.shape[-2] != values.shape[-1]:
        raise ValueError("interpolated segment arrays must be square.")
    if tuple(rotation_degrees.shape) != tuple(values.shape[:2]):
        raise ValueError(
            "rotation_degrees must match the segment annulus and branch axes."
        )
