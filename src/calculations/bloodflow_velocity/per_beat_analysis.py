from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calculations._shared import metric_value, vessel_group_name
from calculations.bloodflow_velocity.per_beat_signal_analysis import (
    per_beat_signal_analysis,
)


@dataclass(frozen=True)
class PerBeatAnalysisInput:
    arterial_velocity_signal: np.ndarray
    venous_velocity_signal: np.ndarray
    systolic_acceleration_peak_indexes: np.ndarray
    band_limited_signal_harmonic_count: int
    dt_seconds: float
    index_base: int | None = None


def _per_beat_analysis_handle(
    velocity_signal,
    vessel_name: str,
    sys_idx_list,
    band_limited_signal_harmonic_count: int,
    dt_seconds: float,
    *,
    index_base: int | None = None,
) -> dict[str, object]:
    result = per_beat_signal_analysis(
        velocity_signal,
        sys_idx_list,
        band_limited_signal_harmonic_count,
        index_base=index_base,
    )
    vessel_group = vessel_group_name(vessel_name)
    base_path = f"{vessel_group}/VelocityPerBeat"

    return {
        f"{base_path}/VelocitySignalPerBeat/value": metric_value(
            result.velocity_signal_per_beat,
            unit="mm/s",
            dim_desc=("beat", "sample"),
            matlab_function="perBeatAnalysis",
        ),
        f"{base_path}/VelocitySignalPerBeatFFT_abs/value": metric_value(
            np.abs(result.velocity_signal_per_beat_fft),
            unit="a.u.",
            dim_desc=("beat", "frequency_bin"),
            matlab_function="perBeatAnalysis",
        ),
        f"{base_path}/VelocitySignalPerBeatFFT_arg/value": metric_value(
            np.angle(result.velocity_signal_per_beat_fft),
            unit="rad",
            dim_desc=("beat", "frequency_bin"),
            matlab_function="perBeatAnalysis",
        ),
        f"{base_path}/VelocitySignalPerBeatBandLimited/value": metric_value(
            result.velocity_signal_per_beat_band_limited,
            unit="mm/s",
            dim_desc=("beat", "sample"),
            matlab_function="perBeatAnalysis",
        ),
        f"{base_path}/VmaxPerBeatBandLimited/value": metric_value(
            np.max(result.velocity_signal_per_beat_band_limited, axis=1),
            unit="mm/s",
            dim_desc=("beat",),
            matlab_function="perBeatAnalysis",
        ),
        f"{base_path}/VminPerBeatBandLimited/value": metric_value(
            np.min(result.velocity_signal_per_beat_band_limited, axis=1),
            unit="mm/s",
            dim_desc=("beat",),
            matlab_function="perBeatAnalysis",
        ),
        f"{base_path}/VTIPerBeat/value": metric_value(
            np.sum(result.velocity_signal_per_beat, axis=1) * float(dt_seconds),
            unit="mm",
            dim_desc=("beat",),
            matlab_function="perBeatAnalysis",
        ),
    }


def run_per_beat_analysis(inputs: PerBeatAnalysisInput) -> dict[str, object]:
    metrics: dict[str, object] = {}
    sys_idx_list = np.asarray(
        inputs.systolic_acceleration_peak_indexes,
        dtype=np.int64,
    ).reshape(-1)

    metrics["Artery/VelocityPerBeat/beatPeriodIdx/value"] = metric_value(
        np.diff(sys_idx_list).astype(np.int32, copy=False),
        unit="frame",
        dim_desc=("beat",),
        matlab_function="perBeatAnalysis",
    )
    metrics["Artery/VelocityPerBeat/beatPeriodSeconds/value"] = metric_value(
        np.diff(sys_idx_list).astype(np.float64, copy=False) * float(inputs.dt_seconds),
        unit="s",
        dim_desc=("beat",),
        matlab_function="perBeatAnalysis",
    )
    metrics.update(
        _per_beat_analysis_handle(
            inputs.venous_velocity_signal,
            "vein",
            sys_idx_list,
            inputs.band_limited_signal_harmonic_count,
            inputs.dt_seconds,
            index_base=inputs.index_base,
        )
    )
    metrics.update(
        _per_beat_analysis_handle(
            inputs.arterial_velocity_signal,
            "artery",
            sys_idx_list,
            inputs.band_limited_signal_harmonic_count,
            inputs.dt_seconds,
            index_base=inputs.index_base,
        )
    )
    return metrics
