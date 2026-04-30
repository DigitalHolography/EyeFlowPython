"""Output schema and manager for EyeFlow HDF5 calculation results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from domain.blood_flow_velocity import PerBeatAnalysisResult
from .schema import DOPPLER_VIEW_ANALYSIS_SCHEMA

if TYPE_CHECKING:
    import h5py

    from pipelines.core.base import ProcessResult


ZERO_BASED_INDEX_PATHS = frozenset(
    {
        DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path("beat_indices"),
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


class EyeFlowOutputManager:
    """Owns EyeFlow HDF5 output writes for one open file."""

    def __init__(self, h5file: "h5py.File") -> None:
        self.h5file = h5file

    def initialize(
        self,
        *,
        holodoppler_source_file: str | None = None,
        doppler_vision_source_file: str | None = None,
    ) -> None:
        from .hdf5 import initialize_output_h5

        initialize_output_h5(
            self.h5file,
            holodoppler_source_file=holodoppler_source_file,
            doppler_vision_source_file=doppler_vision_source_file,
        )

    def append_pipeline_result(
        self,
        pipeline_name: str,
        result: "ProcessResult",
    ):
        from .hdf5 import append_result_group

        return append_result_group(self.h5file, pipeline_name, result)

    def write_metric(self, path: str, value) -> None:
        from .hdf5 import write_value_dataset

        write_value_dataset(self.h5file, path, value)


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
        _analysis_path("retinal_velocity_array"): _metric_value(
            analysis["retinal_vessel_velocity"],
            unit="mm/s",
            matlab_function="DopplerView vessel_velocity_estimator",
        ),
        _analysis_path("retinal_artery_velocity_signal"): _metric_value(
            analysis["retinal_artery_velocity_signal"],
            unit="mm/s",
        ),
        _analysis_path("retinal_vein_velocity_signal"): _metric_value(
            analysis["retinal_vein_velocity_signal"],
            unit="mm/s",
        ),
        _analysis_path("velocity_map_avg"): _metric_value(
            analysis["velocity_map_avg"],
        ),
        _analysis_path("fRMS_avg"): _metric_value(analysis["fRMS_avg"]),
        _analysis_path("fRMS_bkg_avg"): _metric_value(analysis["fRMS_bkg_avg"]),
        _analysis_path("velocitysignal_per_beat"): _metric_value(
            analysis["retinal_artery_velocity_signal_filtered_perbeat"],
            unit="mm/s",
        ),
        _analysis_path("velocitysignal_filtered"): _metric_value(
            analysis["retinal_artery_velocity_signal_filtered"],
            unit="mm/s",
        ),
        _analysis_path("beat_indices"): _metric_value(analysis["beat_indices"]),
        _analysis_path("time_per_beat"): _metric_value(
            analysis["time_per_beat"],
            unit="s",
        ),
    }


def _analysis_path(key: str) -> str:
    return DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(key)


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
