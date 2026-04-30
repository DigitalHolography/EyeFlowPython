"""Port of BloodFlowVelocity/perBeatAnalysis.m."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .per_beat_signal import PerBeatSignalAnalysisResult, per_beat_signal_analysis


@dataclass(frozen=True)
class VesselPerBeatAnalysisResult:
    signal: PerBeatSignalAnalysisResult
    vmax_band_limited: np.ndarray
    vmin_band_limited: np.ndarray
    vti_per_beat: np.ndarray


@dataclass(frozen=True)
class PerBeatAnalysisInput:
    arterial_velocity_signal: np.ndarray
    venous_velocity_signal: np.ndarray
    systolic_acceleration_peak_indexes: np.ndarray
    band_limited_signal_harmonic_count: int
    dt_seconds: float
    beat_period_seconds: np.ndarray | None = None
    index_base: int | None = None


@dataclass(frozen=True)
class PerBeatAnalysisResult:
    beat_period_idx: np.ndarray
    beat_period_seconds: np.ndarray
    artery: VesselPerBeatAnalysisResult
    vein: VesselPerBeatAnalysisResult


def run_per_beat_analysis(inputs: PerBeatAnalysisInput) -> PerBeatAnalysisResult:
    sys_idx_list = np.asarray(
        inputs.systolic_acceleration_peak_indexes,
        dtype=np.int64,
    ).reshape(-1)
    beat_period_idx = np.diff(sys_idx_list).astype(np.int32, copy=False)
    return PerBeatAnalysisResult(
        beat_period_idx=beat_period_idx,
        beat_period_seconds=_beat_period_seconds(inputs, beat_period_idx),
        vein=_run_vessel(inputs.venous_velocity_signal, sys_idx_list, inputs),
        artery=_run_vessel(inputs.arterial_velocity_signal, sys_idx_list, inputs),
    )


def _beat_period_seconds(
    inputs: PerBeatAnalysisInput,
    beat_period_idx: np.ndarray,
) -> np.ndarray:
    if inputs.beat_period_seconds is not None:
        periods = np.asarray(inputs.beat_period_seconds, dtype=np.float64).reshape(-1)
        if periods.size == beat_period_idx.size:
            return periods
    return beat_period_idx.astype(np.float64, copy=False) * float(inputs.dt_seconds)


def _run_vessel(
    velocity_signal,
    sys_idx_list: np.ndarray,
    inputs: PerBeatAnalysisInput,
) -> VesselPerBeatAnalysisResult:
    signal = per_beat_signal_analysis(
        velocity_signal,
        sys_idx_list,
        inputs.band_limited_signal_harmonic_count,
        index_base=inputs.index_base,
    )
    return VesselPerBeatAnalysisResult(
        signal=signal,
        vmax_band_limited=np.max(signal.velocity_signal_per_beat_band_limited, axis=1),
        vmin_band_limited=np.min(signal.velocity_signal_per_beat_band_limited, axis=1),
        vti_per_beat=(
            np.sum(signal.velocity_signal_per_beat, axis=1) * float(inputs.dt_seconds)
        ),
    )

