"""Pipeline 1 MVP: compute DopplerView analysis, then AE waveform metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from domain.blood_flow_velocity import PerBeatAnalysisInput, run_per_beat_analysis
from domain.steps import (
    ArterialWaveformAnalysisStep,
    VesselVelocityEstimatorStep,
)
from pipelines.core.base import ProcessPipeline, ProcessResult, registerPipeline
from pipelines.utils.input_access import (
    HolodopplerTiming,
    read_int_setting,
    read_nested_int_setting,
    resolve_holodoppler_timing,
    resolve_required_source_array,
)
from input_output import (
    DOPPLER_VIEW_ANALYSIS_SCHEMA,
    DOPPLER_VIEW_SCHEMA,
    HOLODOPPLER_SCHEMA,
    pack_dopplerview_analysis_outputs,
    pack_velocity_per_beat_outputs,
    systolic_index_base_for_path,
)

LEGACY_BAND_LIMITED_SIGNAL_HARMONIC_COUNT = 13
DOPPLERVIEW_DEFAULT_LOCAL_BACKGROUND_DIST = 2


@dataclass(frozen=True)
class WaveformShapeMetricsContext:
    per_beat_analysis: PerBeatAnalysisInput
    dopplerview_analysis: dict[str, object]
    attrs: dict[str, object]


@dataclass
class DopplerViewStepContext:
    cache: dict[str, object]
    holodoppler_config: dict[str, object]
    dopplerview_config: dict[str, object]

    def require(self, key: str):
        if key not in self.cache:
            raise RuntimeError(f"Missing required context key: '{key}'")
        return self.cache[key]

    def set(self, key: str, value) -> None:
        self.cache[key] = value


def _build_waveform_shape_metrics_context(
    pipeline_input,
) -> WaveformShapeMetricsContext:
    timing = resolve_holodoppler_timing(pipeline_input)
    dopplerview_analysis = _run_dopplerview_analysis(pipeline_input, timing)
    harmonic_count = _band_limited_harmonic_count(pipeline_input)

    return WaveformShapeMetricsContext(
        per_beat_analysis=_per_beat_input_from_analysis(
            dopplerview_analysis,
            timing,
            harmonic_count,
        ),
        dopplerview_analysis=dopplerview_analysis,
        attrs=_context_attrs(timing, harmonic_count),
    )


def _band_limited_harmonic_count(pipeline_input) -> int:
    return read_int_setting(
        pipeline_input,
        default=LEGACY_BAND_LIMITED_SIGNAL_HARMONIC_COUNT,
        keys=("BandLimitedSignalHarmonicCount", "band_limited_signal_harmonic_count"),
    )


def _per_beat_input_from_analysis(
    analysis: Mapping[str, object],
    timing: HolodopplerTiming,
    harmonic_count: int,
) -> PerBeatAnalysisInput:
    return PerBeatAnalysisInput(
        arterial_velocity_signal=np.asarray(
            analysis["retinal_artery_velocity_signal"],
            dtype=np.float64,
        ),
        venous_velocity_signal=np.asarray(
            analysis["retinal_vein_velocity_signal"],
            dtype=np.float64,
        ),
        systolic_acceleration_peak_indexes=np.asarray(
            analysis["beat_indices"],
            dtype=np.int64,
        ),
        band_limited_signal_harmonic_count=harmonic_count,
        dt_seconds=timing.dt_seconds,
        beat_period_seconds=np.asarray(analysis["time_per_beat"], dtype=np.float64),
        index_base=systolic_index_base_for_path(
            DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path("beat_indices")
        ),
    )


def _run_dopplerview_analysis(
    pipeline_input,
    timing: HolodopplerTiming,
) -> dict[str, object]:
    ctx = DopplerViewStepContext(
        cache=_dopplerview_cache_from_h5(pipeline_input),
        holodoppler_config={
            "sampling_freq": timing.sampling_freq,
            "batch_stride": timing.batch_stride,
        },
        dopplerview_config={
            DOPPLER_VIEW_SCHEMA.config_value("local_background_dist").section: {
                DOPPLER_VIEW_SCHEMA.config_value(
                    "local_background_dist"
                ).json_key: _local_background_dist(pipeline_input),
            }
        },
    )
    VesselVelocityEstimatorStep().run(ctx)
    ArterialWaveformAnalysisStep().run(ctx)
    return ctx.cache


def _dopplerview_cache_from_h5(pipeline_input) -> dict[str, object]:
    return {
        "moment0": _read_required_float_array(
            pipeline_input.hd,
            "HD",
            "moment0",
            HOLODOPPLER_SCHEMA.dataset_path("moment0"),
            dopplerview_moment=True,
        ),
        "moment2": _read_required_float_array(
            pipeline_input.hd,
            "HD",
            "moment2",
            HOLODOPPLER_SCHEMA.dataset_path("moment2"),
            dopplerview_moment=True,
        ),
        "retinal_artery_mask": _read_required_bool_array(
            pipeline_input.dv,
            "DV",
            "retinal artery mask",
            DOPPLER_VIEW_SCHEMA.dataset_path("retinal_artery_mask"),
        ),
        "retinal_vein_mask": _read_required_bool_array(
            pipeline_input.dv,
            "DV",
            "retinal vein mask",
            DOPPLER_VIEW_SCHEMA.dataset_path("retinal_vein_mask"),
        ),
    }


def _local_background_dist(pipeline_input) -> int:
    spec = DOPPLER_VIEW_SCHEMA.config_value("local_background_dist")
    value = read_nested_int_setting(
        pipeline_input.dv_config,
        spec.section or "",
        spec.json_key,
        default=int(spec.default or DOPPLERVIEW_DEFAULT_LOCAL_BACKGROUND_DIST),
    )
    return int(value)


def _read_required_float_array(
    source,
    source_name: str,
    logical_name: str,
    path: str,
    *,
    dopplerview_moment: bool = False,
):
    resolved = resolve_required_source_array(
        source,
        source_name=source_name,
        logical_name=logical_name,
        path=path,
    )
    value = (
        _coerce_dopplerview_moment(resolved.value)
        if dopplerview_moment
        else resolved.value
    )
    return np.asarray(value, dtype=np.float64)


def _coerce_dopplerview_moment(value) -> np.ndarray:
    squeezed = np.squeeze(np.asarray(value))
    if squeezed.ndim != 3:
        raise ValueError(
            "Holodoppler moment datasets must become 3-D after squeeze, "
            f"got shape {squeezed.shape}."
        )
    return np.transpose(squeezed, (0, 2, 1))


def _read_required_bool_array(source, source_name: str, logical_name: str, path: str):
    resolved = resolve_required_source_array(
        source,
        source_name=source_name,
        logical_name=logical_name,
        path=path,
    )
    return np.asarray(resolved.value, dtype=bool)


def _context_attrs(
    timing: HolodopplerTiming,
    harmonic_count: int,
) -> dict[str, object]:
    return {
        "dependency_chain": [
            "DopplerView vessel_velocity_estimator",
            "DopplerView arterial_waveform_analysis",
            "perBeatSignalAnalysis",
            "perBeatAnalysis",
        ],
        "analysis_source": "computed_dopplerview_steps",
        "arterial_velocity_signal_path": DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
            "retinal_artery_velocity_signal"
        ),
        "venous_velocity_signal_path": DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
            "retinal_vein_velocity_signal"
        ),
        "systolic_peak_indexes_path": DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
            "beat_indices"
        ),
        "beat_period_seconds_path": DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
            "time_per_beat"
        ),
        "sampling_freq": float(timing.sampling_freq),
        "batch_stride": float(timing.batch_stride),
        "dt_seconds": float(timing.dt_seconds),
        "band_limited_signal_harmonic_count": int(harmonic_count),
    }


def run_waveform_shape_metrics(
    pipeline_input,
) -> tuple[dict[str, object], dict[str, object]]:
    if pipeline_input.hd is None or pipeline_input.dv is None:
        raise ValueError("waveform_shape_metrics requires both HD and DV inputs.")

    context = _build_waveform_shape_metrics_context(pipeline_input)
    per_beat_result = run_per_beat_analysis(context.per_beat_analysis)
    metrics = pack_dopplerview_analysis_outputs(context.dopplerview_analysis)
    metrics.update(pack_velocity_per_beat_outputs(per_beat_result))
    return metrics, context.attrs


@registerPipeline(
    name="waveform_shape_metrics",
    description="Pipeline 1 MVP: global per-beat velocity outputs for AngioEye.",
    required_deps=["numpy", "h5py", "scipy", "skimage"],
)
class WaveformShapeMetrics(ProcessPipeline):
    input_slot = "both"

    def run(self, h5file) -> ProcessResult:
        metrics, attrs = run_waveform_shape_metrics(h5file)
        return ProcessResult(metrics=metrics, attrs={"pipeline": self.name, **attrs})
