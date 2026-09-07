"""Export paired velocity-profile flow-asymmetry metrics to HDF5."""

from __future__ import annotations

import numpy as np

from calculations.blood_flow_velocity.cross_section.flow_asymmetry import (
    calculate_flow_asymmetry,
)
from pipeline_engine.base import DatasetValue


_METRICS = (
    ("A", "asymmetry", "sum_x(v(x)-v(-x))/sum_x(v(x)+v(-x))"),
    ("A_abs", "magnitude", "abs(A)"),
    ("A_mean", "temporal_mean", "mean_t(A)"),
    ("A_RMS", "rms", "sqrt(mean_t(A**2))"),
    ("p_A", "dynamic_power", "(A-A_mean)**2"),
    ("a", "dynamic_rms", "sqrt(mean_t(p_A))"),
    ("a_early", "early_dynamic_rms", "sqrt(mean_early(p_A))"),
    ("a_late", "late_dynamic_rms", "sqrt(mean_late(p_A))"),
    ("R_a", "redistribution_ratio", "a_early/a_late"),
    ("N_t", "temporal_sample_count", "count_finite_t(A)"),
    ("N_early", "early_sample_count", "count_finite_early(A)"),
    ("N_late", "late_sample_count", "count_finite_late(A)"),
    ("FFA", "ffa", "median_r(median_b(median_k(a)))"),
    ("FFAR", "ffar", "median_r(median_b(median_k(R_a)))"),
    ("PFA", "pfa", "median_r(median_b(median_k(abs(A_mean))))"),
)


def pack_flow_asymmetry_outputs(
    root: str,
    segments,
    cycle_boundary_indexes,
    *,
    index_base: int = 0,
) -> dict[str, DatasetValue]:
    """Publish per-vessel time series, beat summaries, and global indices."""
    result = calculate_flow_asymmetry(
        segments.centered_velocity_profiles,
        segments.centered_profile_x_micrometers,
        cycle_boundary_indexes,
        index_base=index_base,
    )
    time_count = result.asymmetry.shape[0]
    window_count = time_count // 3
    common_attrs = {
        "unit": "1",
        "profile_source": "centered_transverse_velocity_profile",
        "centerline_source": "midpoint_of_time_mean_positive_profile_zero_roots",
        "spatial_region": "paired_lumen_positions_0<x<=R",
        "lumen_radius": "R=(x2-x1)/2",
        "side_convention": "positive_x_minus_negative_x",
        "temporal_interpolation": "interpft_spatial_component_sums_before_ratio",
        "temporal_centering": "full_beat_mean",
        "missing_data_policy": "finite_time_samples_and_finite_hierarchical_medians",
        "zero_denominator_policy": "NaN_without_epsilon",
        "window_definition": "first_and_last_floor(Nt/3)_samples",
        "early_window_start_index": 0,
        "early_window_stop_index_exclusive": window_count,
        "late_window_start_index": time_count - window_count,
        "late_window_stop_index_exclusive": time_count,
        "aggregation_order": "branch_then_beat_then_radius",
    }
    outputs = {}
    for name, field, formula in _METRICS:
        data = np.asarray(getattr(result, field))
        dimensions = (
            ["time", "beat", "branch", "radius"]
            if data.ndim == 4
            else ["beat", "branch", "radius"] if data.ndim == 3 else []
        )
        outputs[f"{root}/{name}/value"] = DatasetValue(
            data=data,
            attrs={**common_attrs, "dimDesc": dimensions, "formula": formula},
        )
    return outputs
