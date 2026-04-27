from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calculations._shared import metric_value, vessel_group_name
from calculations.bloodflow_velocity.per_beat_signal_analysis import (
    per_beat_signal_analysis,
)


@dataclass(frozen=True)
class CrossSectionWaveformInput:
    vessel_name: str
    velocity_per_segment: np.ndarray
    safe_velocity_per_segment: np.ndarray | None
    velocity_profiles_per_segment: np.ndarray | None
    systolic_acceleration_peak_indexes: np.ndarray
    band_limited_signal_harmonic_count: int
    modal_decomposition_n_modes: int
    index_base: int | None = None


@dataclass(frozen=True)
class PerBeatModeDecomposition:
    u: np.ndarray
    a: np.ndarray
    wc: np.ndarray
    mu: np.ndarray
    w: np.ndarray
    valid_mask: np.ndarray
    singular_values: np.ndarray


def _coerce_segment_time_array(values, *, field_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(
            f"{field_name} must be a 3D array shaped as "
            "(circle, branch, time). Got shape {array.shape!r}."
        )
    return array


def _coerce_profile_array(values, *, field_name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    if array.ndim != 4:
        raise ValueError(
            f"{field_name} must be a 4D array shaped as "
            "(circle, branch, profile_sample, time). Got shape {array.shape!r}."
        )
    return array


def build_segment_per_beat_signals(
    velocity_per_segment: np.ndarray,
    systolic_acceleration_peak_indexes,
    band_limited_signal_harmonic_count: int,
    *,
    index_base: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity_array = _coerce_segment_time_array(
        velocity_per_segment,
        field_name="velocity_per_segment",
    )
    circle_count, branch_count, _ = velocity_array.shape

    template = per_beat_signal_analysis(
        velocity_array[0, 0, :],
        systolic_acceleration_peak_indexes,
        band_limited_signal_harmonic_count,
        index_base=index_base,
    )
    beat_shape = template.velocity_signal_per_beat.shape

    velocity_signal_per_beat = np.full(
        (circle_count, branch_count, *beat_shape),
        np.nan,
        dtype=np.float64,
    )
    velocity_signal_per_beat_fft = np.full(
        (circle_count, branch_count, *beat_shape),
        np.nan + 0j,
        dtype=np.complex128,
    )
    velocity_signal_per_beat_band_limited = np.full(
        (circle_count, branch_count, *beat_shape),
        np.nan,
        dtype=np.float64,
    )

    for circle_index in range(circle_count):
        for branch_index in range(branch_count):
            signal = velocity_array[circle_index, branch_index, :]
            if np.all(np.isnan(signal)):
                continue
            result = per_beat_signal_analysis(
                signal,
                systolic_acceleration_peak_indexes,
                band_limited_signal_harmonic_count,
                index_base=index_base,
            )
            velocity_signal_per_beat[circle_index, branch_index, :, :] = (
                result.velocity_signal_per_beat
            )
            velocity_signal_per_beat_fft[circle_index, branch_index, :, :] = (
                result.velocity_signal_per_beat_fft
            )
            velocity_signal_per_beat_band_limited[circle_index, branch_index, :, :] = (
                result.velocity_signal_per_beat_band_limited
            )

    return (
        velocity_signal_per_beat,
        velocity_signal_per_beat_fft,
        velocity_signal_per_beat_band_limited,
    )


def decompose_per_beat_modes(
    velocity_signal_per_beat_per_segment: np.ndarray,
    n_modes: int,
) -> PerBeatModeDecomposition:
    waveforms = np.asarray(velocity_signal_per_beat_per_segment, dtype=np.float64)
    if waveforms.ndim != 4:
        raise ValueError(
            "decompose_per_beat_modes expects a 4D array shaped as "
            "(time, beat, branch, circle)."
        )
    if n_modes < 1:
        raise ValueError("n_modes must be a strictly positive integer.")

    time_count, beat_count, branch_count, circle_count = waveforms.shape
    flattened_count = beat_count * branch_count * circle_count

    with np.errstate(invalid="ignore"):
        mu4 = np.nanmean(waveforms, axis=0, keepdims=True)
    centered = waveforms - mu4
    centered_matrix = centered.reshape(time_count, flattened_count)

    valid_columns = np.all(np.isfinite(centered_matrix), axis=0)
    valid_matrix = centered_matrix[:, valid_columns]

    actual_mode_count = min(int(n_modes), int(valid_matrix.shape[1]))
    temporal_modes = np.empty((0, time_count), dtype=np.float64)
    mode_scores = np.empty(
        (0, beat_count, branch_count, circle_count),
        dtype=np.float64,
    )
    reconstructed_components = np.empty(
        (0, time_count, beat_count, branch_count, circle_count),
        dtype=np.float64,
    )
    cumulative_components = np.zeros_like(centered_matrix, dtype=np.float64)
    singular_values = np.empty((0,), dtype=np.float64)

    if actual_mode_count > 0:
        left_vectors, singular_matrix, right_vectors_t = np.linalg.svd(
            valid_matrix,
            full_matrices=False,
        )
        singular_values = singular_matrix[:actual_mode_count]

        temporal_modes = np.full(
            (actual_mode_count, time_count),
            np.nan,
            dtype=np.float64,
        )
        mode_scores = np.full(
            (actual_mode_count, beat_count, branch_count, circle_count),
            np.nan,
            dtype=np.float64,
        )
        reconstructed_components = np.full(
            (actual_mode_count, time_count, beat_count, branch_count, circle_count),
            np.nan,
            dtype=np.float64,
        )

        for mode_index in range(actual_mode_count):
            temporal_mode = left_vectors[:, mode_index]
            score_valid = singular_values[mode_index] * right_vectors_t[mode_index, :]

            score_all = np.full((flattened_count,), np.nan, dtype=np.float64)
            score_all[valid_columns] = score_valid

            reconstructed_matrix = np.outer(temporal_mode, score_all)
            cumulative_components += np.nan_to_num(reconstructed_matrix, nan=0.0)

            temporal_modes[mode_index, :] = temporal_mode
            mode_scores[mode_index, :, :, :] = score_all.reshape(
                beat_count,
                branch_count,
                circle_count,
            )
            reconstructed_components[mode_index, :, :, :, :] = reconstructed_matrix.reshape(
                time_count,
                beat_count,
                branch_count,
                circle_count,
            )

    mean_matrix = np.broadcast_to(
        mu4.reshape(1, flattened_count),
        (time_count, flattened_count),
    )
    reconstructed_waveforms = cumulative_components + mean_matrix
    reconstructed_waveforms[:, ~valid_columns] = np.nan

    return PerBeatModeDecomposition(
        u=temporal_modes,
        a=mode_scores,
        wc=reconstructed_components,
        mu=np.squeeze(mu4, axis=0),
        w=reconstructed_waveforms.reshape(
            time_count,
            beat_count,
            branch_count,
            circle_count,
        ),
        valid_mask=valid_columns.reshape(beat_count, branch_count, circle_count),
        singular_values=singular_values,
    )


def _build_mode_signal_metrics(
    decomposition: PerBeatModeDecomposition,
    vessel_name: str,
) -> dict[str, object]:
    vessel_group = vessel_group_name(vessel_name)
    base_path = (
        f"{vessel_group}/VelocityPerBeat/Segments/"
        "VelocityModeSignalPerBeatPerSegment"
    )
    return {
        f"{base_path}/mu/value": metric_value(
            decomposition.mu,
            dim_desc=("beat", "branch", "circle"),
            matlab_function="exportMode1StructPerBeat",
        ),
        f"{base_path}/a/value": metric_value(
            decomposition.a,
            dim_desc=("mode", "beat", "branch", "circle"),
            matlab_function="exportMode1StructPerBeat",
        ),
        f"{base_path}/u/value": metric_value(
            decomposition.u,
            dim_desc=("mode", "time"),
            matlab_function="exportMode1StructPerBeat",
        ),
        f"{base_path}/wc/value": metric_value(
            decomposition.wc,
            dim_desc=("mode", "time", "beat", "branch", "circle"),
            matlab_function="exportMode1StructPerBeat",
        ),
        f"{base_path}/w/value": metric_value(
            decomposition.w,
            dim_desc=("time", "beat", "branch", "circle"),
            matlab_function="exportMode1StructPerBeat",
        ),
        f"{base_path}/validMask/value": metric_value(
            decomposition.valid_mask,
            dim_desc=("beat", "branch", "circle"),
            matlab_function="exportMode1StructPerBeat",
        ),
        f"{base_path}/s/value": metric_value(
            decomposition.singular_values,
            dim_desc=("mode",),
            matlab_function="exportMode1StructPerBeat",
        ),
    }


def build_cross_section_waveform_metrics(
    inputs: CrossSectionWaveformInput,
) -> dict[str, object]:
    velocity_per_segment = _coerce_segment_time_array(
        inputs.velocity_per_segment,
        field_name="velocity_per_segment",
    )
    safe_velocity_per_segment = (
        velocity_per_segment
        if inputs.safe_velocity_per_segment is None
        else _coerce_segment_time_array(
            inputs.safe_velocity_per_segment,
            field_name="safe_velocity_per_segment",
        )
    )
    velocity_profiles_per_segment = _coerce_profile_array(
        inputs.velocity_profiles_per_segment,
        field_name="velocity_profiles_per_segment",
    )

    (
        velocity_signal_per_beat_per_segment,
        velocity_signal_per_beat_per_segment_fft,
        velocity_signal_per_beat_per_segment_band_limited,
    ) = build_segment_per_beat_signals(
        velocity_per_segment,
        inputs.systolic_acceleration_peak_indexes,
        inputs.band_limited_signal_harmonic_count,
        index_base=inputs.index_base,
    )
    per_beat_layout = np.transpose(
        velocity_signal_per_beat_per_segment,
        (3, 2, 1, 0),
    )
    decomposition = decompose_per_beat_modes(
        per_beat_layout,
        inputs.modal_decomposition_n_modes,
    )

    vessel_group = vessel_group_name(inputs.vessel_name)
    base_path = f"{vessel_group}/VelocityPerBeat/Segments"
    cross_section_path = f"{vessel_group}/CrossSections"
    metrics: dict[str, object] = {
        f"{base_path}/VelocitySignalPerBeatPerSegment/value": metric_value(
            velocity_signal_per_beat_per_segment,
            unit="mm/s",
            dim_desc=("circle", "branch", "beat", "sample"),
            matlab_function="exportProfilesToH5",
        ),
        f"{base_path}/VelocitySignalPerBeatPerSegmentFFT_abs/value": metric_value(
            np.abs(velocity_signal_per_beat_per_segment_fft),
            unit="a.u.",
            dim_desc=("circle", "branch", "beat", "frequency_bin"),
            matlab_function="exportProfilesToH5",
        ),
        f"{base_path}/VelocitySignalPerBeatPerSegmentFFT_arg/value": metric_value(
            np.angle(velocity_signal_per_beat_per_segment_fft),
            unit="rad",
            dim_desc=("circle", "branch", "beat", "frequency_bin"),
            matlab_function="exportProfilesToH5",
        ),
        f"{base_path}/VelocitySignalPerBeatPerSegmentBandLimited/value": metric_value(
            velocity_signal_per_beat_per_segment_band_limited,
            unit="mm/s",
            dim_desc=("circle", "branch", "beat", "sample"),
            matlab_function="exportProfilesToH5",
        ),
        f"{cross_section_path}/VelocityPerSegmentTruncated/value": metric_value(
            velocity_per_segment,
            unit="mm/s",
            dim_desc=("circle", "branch", "time"),
            matlab_function="exportProfilesToH5",
            note=(
                "Canonical Python path for the truncated per-segment velocity used "
                "to reproduce the Matlab exportProfilesToH5 dependency chain."
            ),
        ),
        f"{cross_section_path}/VelocityPerSegment/value": metric_value(
            safe_velocity_per_segment,
            unit="mm/s",
            dim_desc=("circle", "branch", "time"),
            matlab_function="exportProfilesToH5",
        ),
    }
    metrics.update(_build_mode_signal_metrics(decomposition, inputs.vessel_name))

    if velocity_profiles_per_segment is not None:
        metrics[f"{cross_section_path}/VelocityProfileSeg/value"] = metric_value(
            velocity_profiles_per_segment,
            unit="mm/s",
            dim_desc=("circle", "branch", "profile_sample", "time"),
            matlab_function="exportProfilesToH5",
        )

    return metrics
