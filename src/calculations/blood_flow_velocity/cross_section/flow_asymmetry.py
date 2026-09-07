"""Paired lumen-profile asymmetry and its persistent/dynamic components."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .profile_processing import interpolate_velocity_profiles_per_beat


@dataclass(frozen=True)
class FlowAsymmetryResult:
    """Time series use (time, beat, branch, radius); summaries omit time."""

    asymmetry: np.ndarray
    magnitude: np.ndarray
    temporal_mean: np.ndarray
    rms: np.ndarray
    dynamic_power: np.ndarray
    dynamic_rms: np.ndarray
    early_dynamic_rms: np.ndarray
    late_dynamic_rms: np.ndarray
    redistribution_ratio: np.ndarray
    temporal_sample_count: np.ndarray
    early_sample_count: np.ndarray
    late_sample_count: np.ndarray
    ffa: float
    ffar: float
    pfa: float


def calculate_flow_asymmetry(
    centered_profiles: np.ndarray,
    centered_x_micrometers: np.ndarray,
    cycle_boundary_indexes,
    *,
    index_base: int = 0,
) -> FlowAsymmetryResult:
    """Calculate article metrics from profiles spanning the fitted [-R, R].

    Input profiles have shape (radius, branch, frame, x), with coordinates
    (radius, branch, x). The existing centering step defines R=(x2-x1)/2
    using the positive time-mean profile's fitted zero roots.

    For 0 < x <= R, sum va=(v(x)-v(-x))/2 and vs=(v(x)+v(-x))/2.
    Resample these two sums to the standard per-beat time grid, then take
    their ratio A. Linearity makes this equivalent to resampling every
    paired profile sample before summation, without allocating that cube.
    Signed samples are preserved and x=0 is excluded.
    """
    antisymmetric, symmetric = paired_profile_sums(
        centered_profiles, centered_x_micrometers
    )
    components = interpolate_velocity_profiles_per_beat(
        np.stack((antisymmetric, symmetric), axis=-1),
        cycle_boundary_indexes,
        index_base=index_base,
    )
    return summarize_flow_asymmetry(_finite_ratio(components[0], components[1]))


def paired_profile_sums(
    centered_profiles: np.ndarray,
    centered_x_micrometers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed antisymmetric/symmetric sums in (radius, branch, frame).

    Every sample pair must be finite; missing pairs invalidate the frame
    instead of changing its spatial support. All-NaN coordinates represent
    an unavailable lumen fit. Finite grids must increase uniformly from
    -R to R so index reversal pairs equal distances from the lumen center.
    """
    values = np.asarray(centered_profiles, dtype=np.float64)
    x = np.asarray(centered_x_micrometers, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] < 2:
        raise ValueError("centered_profiles must have shape (radius, branch, frame, x>=2).")
    if x.shape != (*values.shape[:2], values.shape[-1]):
        raise ValueError("centered_x_micrometers must have shape (radius, branch, x).")

    antisymmetric = np.full(values.shape[:-1], np.nan, dtype=np.float64)
    symmetric = np.full_like(antisymmetric, np.nan)
    pair_count = values.shape[-1] // 2
    for segment in np.ndindex(values.shape[:2]):
        grid = x[segment]
        if np.all(np.isnan(grid)):
            continue
        spacing = np.diff(grid)
        if (
            not np.all(np.isfinite(grid))
            or not np.all(spacing > 0)
            or not np.allclose(grid, -grid[::-1], rtol=1e-5, atol=1e-6)
            or not np.allclose(spacing, spacing[0], rtol=1e-4, atol=1e-6)
        ):
            raise ValueError("Each centered x grid must be uniform and symmetric about zero.")
        negative = values[segment][..., :pair_count][..., ::-1]
        positive = values[segment][..., -pair_count:]
        valid = np.all(np.isfinite(negative) & np.isfinite(positive), axis=-1)
        antisymmetric[segment] = np.where(
            valid, np.sum(0.5 * (positive - negative), axis=-1), np.nan
        )
        symmetric[segment] = np.where(
            valid, np.sum(0.5 * (positive + negative), axis=-1), np.nan
        )
    return antisymmetric, symmetric


def summarize_flow_asymmetry(asymmetry: np.ndarray) -> FlowAsymmetryResult:
    """Reduce A(time, beat, branch, radius) using population time averages.

    Early/late windows contain the first/last floor(Nt/3) samples. Both use
    the full-beat mean in p_A=(A-mean(A))**2, not separate window means.
    Nonfinite A samples are excluded consistently from all time averages;
    returned counts record their actual denominators. Empty windows/segments
    and zero-denominator ratios return NaN, without epsilon regularization.
    """
    values = np.asarray(asymmetry, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("asymmetry must have shape (time>0, beat, branch, radius).")
    values = np.where(np.isfinite(values), values, np.nan)
    mean, count = _temporal_mean(values)
    power = (values - mean[None, ...]) ** 2
    mean_power, _ = _temporal_mean(power)
    mean_square, _ = _temporal_mean(values**2)
    dynamic_rms = np.sqrt(mean_power)
    window_count = values.shape[0] // 3
    early_power, early_count = _temporal_mean(power[:window_count])
    late_power, late_count = _temporal_mean(power[values.shape[0] - window_count:])
    early_rms = np.sqrt(early_power)
    late_rms = np.sqrt(late_power)
    ratio = _finite_ratio(early_rms, late_rms)
    return FlowAsymmetryResult(
        asymmetry=values,
        magnitude=np.abs(values),
        temporal_mean=mean,
        rms=np.sqrt(mean_square),
        dynamic_power=power,
        dynamic_rms=dynamic_rms,
        early_dynamic_rms=early_rms,
        late_dynamic_rms=late_rms,
        redistribution_ratio=ratio,
        temporal_sample_count=count,
        early_sample_count=early_count,
        late_sample_count=late_count,
        ffa=hierarchical_asymmetry_median(dynamic_rms),
        ffar=hierarchical_asymmetry_median(ratio),
        pfa=hierarchical_asymmetry_median(np.abs(mean)),
    )


def hierarchical_asymmetry_median(values: np.ndarray) -> float:
    """Apply median_r(median_b(median_k(values[b, k, r])))."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("values must have shape (beat, branch, radius).")
    if not values.size:
        return float("nan")
    values = np.where(np.isfinite(values), values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_beat_radius = np.nanmedian(values, axis=1)
        per_radius = np.nanmedian(per_beat_radius, axis=0)
        return float(np.nanmedian(per_radius))


def _temporal_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    count = np.sum(valid, axis=0, dtype=np.int32)
    total = np.sum(np.where(valid, values, 0.0), axis=0, dtype=np.float64)
    return _finite_ratio(total, count), count


def _finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(numerator, denominator, out=result, where=valid)
    return np.where(np.isfinite(result), result, np.nan)
