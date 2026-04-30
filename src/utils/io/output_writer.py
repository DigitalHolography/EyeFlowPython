"""HDF5 path constants and writers for EyeFlow calculation outputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from domain.blood_flow_velocity import PerBeatAnalysisResult
from .schema import (
    DOPPLERVIEW_ANALYSIS_ROOT,
    DOPPLERVIEW_ARTERIAL_VELOCITY_SIGNAL_PATH,
    DOPPLERVIEW_BEAT_INDICES_PATH,
    DOPPLERVIEW_FILTERED_PER_BEAT_PATH,
    DOPPLERVIEW_FILTERED_SIGNAL_PATH,
    DOPPLERVIEW_FRMS_AVG_PATH,
    DOPPLERVIEW_FRMS_BKG_AVG_PATH,
    DOPPLERVIEW_RETINAL_VELOCITY_ARRAY_PATH,
    DOPPLERVIEW_TIME_PER_BEAT_PATH,
    DOPPLERVIEW_VELOCITY_MAP_AVG_PATH,
    DOPPLERVIEW_VENOUS_VELOCITY_SIGNAL_PATH,
)


ANGIOEYE_OUTPUT_ROOTS = ("Artery", "Vein", "Meta")
DOPPLERVIEW_ANALYSIS_OUTPUT_ROOTS = (DOPPLERVIEW_ANALYSIS_ROOT,)
ROOT_MIRRORED_OUTPUT_ROOTS = ANGIOEYE_OUTPUT_ROOTS + DOPPLERVIEW_ANALYSIS_OUTPUT_ROOTS

ZERO_BASED_INDEX_PATHS = frozenset(
    {
        DOPPLERVIEW_BEAT_INDICES_PATH,
    }
)


@dataclass(frozen=True)
class VelocityPerBeatOutputPaths:
    velocity_signal: str
    velocity_signal_fft_abs: str
    velocity_signal_fft_arg: str
    velocity_signal_band_limited: str
    vmax_band_limited: str
    vmin_band_limited: str
    vti: str


ARTERY_PER_BEAT_PATHS = VelocityPerBeatOutputPaths(
    velocity_signal="Artery/VelocityPerBeat/VelocitySignalPerBeat/value",
    velocity_signal_fft_abs="Artery/VelocityPerBeat/VelocitySignalPerBeatFFT_abs/value",
    velocity_signal_fft_arg="Artery/VelocityPerBeat/VelocitySignalPerBeatFFT_arg/value",
    velocity_signal_band_limited=(
        "Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    ),
    vmax_band_limited="Artery/VelocityPerBeat/VmaxPerBeatBandLimited/value",
    vmin_band_limited="Artery/VelocityPerBeat/VminPerBeatBandLimited/value",
    vti="Artery/VelocityPerBeat/VTIPerBeat/value",
)
VEIN_PER_BEAT_PATHS = VelocityPerBeatOutputPaths(
    velocity_signal="Vein/VelocityPerBeat/VelocitySignalPerBeat/value",
    velocity_signal_fft_abs="Vein/VelocityPerBeat/VelocitySignalPerBeatFFT_abs/value",
    velocity_signal_fft_arg="Vein/VelocityPerBeat/VelocitySignalPerBeatFFT_arg/value",
    velocity_signal_band_limited=(
        "Vein/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
    ),
    vmax_band_limited="Vein/VelocityPerBeat/VmaxPerBeatBandLimited/value",
    vmin_band_limited="Vein/VelocityPerBeat/VminPerBeatBandLimited/value",
    vti="Vein/VelocityPerBeat/VTIPerBeat/value",
)


def is_angioeye_output_key(key: str) -> bool:
    return _has_h5_root(key, ANGIOEYE_OUTPUT_ROOTS)


def is_root_mirrored_output_key(key: str) -> bool:
    return _has_h5_root(key, ROOT_MIRRORED_OUTPUT_ROOTS)


def _has_h5_root(key: str, roots: tuple[str, ...]) -> bool:
    normalized = _normalize_h5_key(key)
    return any(
        normalized == root or normalized.startswith(f"{root}/")
        for root in roots
    )


def systolic_index_base_for_path(path: str) -> int | None:
    normalized = _normalize_h5_key(path)
    return 0 if normalized in ZERO_BASED_INDEX_PATHS else None


def pack_velocity_per_beat_outputs(
    result: PerBeatAnalysisResult,
) -> dict[str, object]:
    metrics = {
        "Artery/VelocityPerBeat/beatPeriodIdx/value": _metric_value(
            result.beat_period_idx,
            unit="frame",
            dim_desc=("beat",),
            matlab_function="perBeatAnalysis",
        ),
        "Artery/VelocityPerBeat/beatPeriodSeconds/value": _metric_value(
            result.beat_period_seconds,
            unit="s",
            dim_desc=("beat",),
            matlab_function="perBeatAnalysis",
        ),
    }
    metrics.update(_pack_vessel_outputs(ARTERY_PER_BEAT_PATHS, result.artery))
    metrics.update(_pack_vessel_outputs(VEIN_PER_BEAT_PATHS, result.vein))
    return metrics


def pack_dopplerview_analysis_outputs(
    analysis: Mapping[str, object],
) -> dict[str, object]:
    return {
        DOPPLERVIEW_RETINAL_VELOCITY_ARRAY_PATH: _metric_value(
            analysis["retinal_vessel_velocity"],
            unit="mm/s",
            matlab_function="DopplerView vessel_velocity_estimator",
        ),
        DOPPLERVIEW_ARTERIAL_VELOCITY_SIGNAL_PATH: _metric_value(
            analysis["retinal_artery_velocity_signal"],
            unit="mm/s",
        ),
        DOPPLERVIEW_VENOUS_VELOCITY_SIGNAL_PATH: _metric_value(
            analysis["retinal_vein_velocity_signal"],
            unit="mm/s",
        ),
        DOPPLERVIEW_VELOCITY_MAP_AVG_PATH: _metric_value(
            analysis["velocity_map_avg"],
        ),
        DOPPLERVIEW_FRMS_AVG_PATH: _metric_value(analysis["fRMS_avg"]),
        DOPPLERVIEW_FRMS_BKG_AVG_PATH: _metric_value(analysis["fRMS_bkg_avg"]),
        DOPPLERVIEW_FILTERED_PER_BEAT_PATH: _metric_value(
            analysis["retinal_artery_velocity_signal_filtered_perbeat"],
            unit="mm/s",
        ),
        DOPPLERVIEW_FILTERED_SIGNAL_PATH: _metric_value(
            analysis["retinal_artery_velocity_signal_filtered"],
            unit="mm/s",
        ),
        DOPPLERVIEW_BEAT_INDICES_PATH: _metric_value(analysis["beat_indices"]),
        DOPPLERVIEW_TIME_PER_BEAT_PATH: _metric_value(
            analysis["time_per_beat"],
            unit="s",
        ),
    }


def _pack_vessel_outputs(
    paths: VelocityPerBeatOutputPaths,
    vessel,
) -> dict[str, object]:
    signal = vessel.signal
    return {
        paths.velocity_signal: _metric_value(
            signal.velocity_signal_per_beat,
            unit="mm/s",
            dim_desc=("beat", "sample"),
            matlab_function="perBeatAnalysis",
        ),
        paths.velocity_signal_fft_abs: _metric_value(
            np.abs(signal.velocity_signal_per_beat_fft),
            unit="a.u.",
            dim_desc=("beat", "frequency_bin"),
            matlab_function="perBeatAnalysis",
        ),
        paths.velocity_signal_fft_arg: _metric_value(
            np.angle(signal.velocity_signal_per_beat_fft),
            unit="rad",
            dim_desc=("beat", "frequency_bin"),
            matlab_function="perBeatAnalysis",
        ),
        paths.velocity_signal_band_limited: _metric_value(
            signal.velocity_signal_per_beat_band_limited,
            unit="mm/s",
            dim_desc=("beat", "sample"),
            matlab_function="perBeatAnalysis",
        ),
        paths.vmax_band_limited: _metric_value(vessel.vmax_band_limited, unit="mm/s"),
        paths.vmin_band_limited: _metric_value(vessel.vmin_band_limited, unit="mm/s"),
        paths.vti: _metric_value(vessel.vti_per_beat, unit="mm"),
    }


def _metric_value(
    data,
    *,
    unit: str | None = None,
    dim_desc: Iterable[str] | None = None,
    matlab_function: str | None = None,
):
    attrs: dict[str, object] = {}
    if unit:
        attrs["unit"] = unit
    if dim_desc:
        attrs["dimDesc"] = list(dim_desc)
    if matlab_function:
        attrs["matlab_function"] = matlab_function
    return (data, attrs) if attrs else data


def _normalize_h5_key(key: str) -> str:
    return str(key).replace("\\", "/").strip("/")
