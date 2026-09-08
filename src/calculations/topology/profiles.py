"""Generic profile reductions for uniformly oriented segment maps."""

from __future__ import annotations

import numpy as np

from calculations.math import nanmean_float32


def transverse_profiles(
    segments,
    segment_masks: np.ndarray | None = None,
) -> np.ndarray:
    """Average segment values along Y, retaining the transverse X axis."""

    return nanmean_float32(_masked_segments(segments, segment_masks), axis=-2)


def longitudinal_profiles(
    segments,
    segment_masks: np.ndarray | None = None,
) -> np.ndarray:
    """Average segment values along X, retaining the longitudinal Y axis."""

    return nanmean_float32(_masked_segments(segments, segment_masks), axis=-1)


def fit_inverse_parabola_profiles(
    profiles,
    x_values: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a downward-opening quadratic to every profile along its last axis.

    Every combination of leading indexes is fitted independently, so the
    function can be used with any segment-array layout. Only finite samples
    participate in a fit. Profiles with fewer than three usable samples, a
    rank-deficient fit, or a non-negative quadratic coefficient remain NaN.

    The fitted quadratic is evaluated at every supplied X value and the
    returned array has the same shape as ``profiles``.
    """

    fitted, _ = fit_inverse_parabola_profiles_with_roots(
        profiles,
        x_values=x_values,
    )
    return fitted


def fit_inverse_parabola_profiles_with_roots(
    profiles,
    x_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit inverse parabolas and return their ordered X-axis roots.

    The fitted profiles have the same shape as ``profiles``. The roots have
    shape ``(*profiles.shape[:-1], 2)`` and are ordered from lowest to highest
    X value. Fits without real roots retain NaN root values.
    """

    values = np.asarray(profiles, dtype=np.float32)
    if values.ndim == 0:
        raise ValueError("profiles must have a spatial sample axis.")

    sample_count = values.shape[-1]
    if x_values is None:
        x = np.linspace(-1.0, 1.0, sample_count, dtype=np.float64)
    else:
        x = np.asarray(x_values, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != sample_count:
            raise ValueError(
                "x_values must be one-dimensional and match the profile "
                "spatial sample count."
            )

    fitted = np.full(values.shape, np.nan, dtype=np.float32)
    roots = np.full((*values.shape[:-1], 2), np.nan, dtype=np.float32)
    if sample_count < 3 or values.size == 0:
        return fitted, roots

    finite_x = np.isfinite(x)
    flat_values = values.reshape(-1, sample_count)
    flat_fitted = fitted.reshape(-1, sample_count)
    flat_roots = roots.reshape(-1, 2)
    for profile_index, profile in enumerate(flat_values):
        fit_samples = finite_x & np.isfinite(profile)
        if np.count_nonzero(fit_samples) < 3:
            continue

        fit_x = x[fit_samples]
        design = np.column_stack((fit_x * fit_x, fit_x, np.ones_like(fit_x)))
        coefficients, _, rank, _ = np.linalg.lstsq(
            design,
            profile[fit_samples].astype(np.float64),
            rcond=None,
        )
        if rank < 3 or not np.all(np.isfinite(coefficients)):
            continue
        if coefficients[0] >= 0.0:
            continue

        evaluation_x = x[finite_x]
        flat_fitted[profile_index, finite_x] = (
            coefficients[0] * evaluation_x * evaluation_x
            + coefficients[1] * evaluation_x
            + coefficients[2]
        ).astype(np.float32)

        discriminant = (
            coefficients[1] * coefficients[1]
            - 4.0 * coefficients[0] * coefficients[2]
        )
        if not np.isfinite(discriminant) or discriminant < 0.0:
            continue
        root_delta = np.sqrt(discriminant)
        profile_roots = np.sort(
            np.asarray(
                [
                    (-coefficients[1] + root_delta) / (2.0 * coefficients[0]),
                    (-coefficients[1] - root_delta) / (2.0 * coefficients[0]),
                ],
                dtype=np.float64,
            )
        )
        flat_roots[profile_index] = profile_roots.astype(np.float32)

    return fitted, roots


def mean_profiles(profiles, *, axis: int) -> np.ndarray:
    """Return the NaN-aware mean of every profile along one named axis."""

    return nanmean_float32(np.asarray(profiles, dtype=np.float32), axis=axis)


def profile_deviation_power(
    profiles,
    mean: np.ndarray | None = None,
    *,
    axis: int,
) -> np.ndarray:
    """Return squared deviations from a supplied or calculated profile mean."""

    values = np.asarray(profiles, dtype=np.float32)
    normalized_axis = int(axis) % values.ndim
    if mean is None:
        profile_mean = mean_profiles(values, axis=normalized_axis)
    else:
        profile_mean = np.asarray(mean, dtype=np.float32)
    expected_shape = (*values.shape[:normalized_axis], *values.shape[normalized_axis + 1 :])
    if profile_mean.shape != expected_shape:
        raise ValueError(
            f"mean must have shape {expected_shape}, got {profile_mean.shape}."
        )
    centered = values - np.expand_dims(profile_mean, axis=normalized_axis)
    return np.square(centered).astype(np.float32, copy=False)


def _masked_segments(
    segments,
    segment_masks: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(segments, dtype=np.float32)
    if segment_masks is None:
        return values

    masks = np.asarray(segment_masks, dtype=bool)
    expected_shape = (*values.shape[:2], *values.shape[-2:])
    if masks.shape != expected_shape:
        raise ValueError(
            "segment_masks must match the segment, annulus, branch, and spatial axes."
        )
    expanded_shape = (
        *masks.shape[:2],
        *((1,) * (values.ndim - 4)),
        *masks.shape[-2:],
    )
    return np.where(masks.reshape(expanded_shape), values, np.float32(np.nan))
