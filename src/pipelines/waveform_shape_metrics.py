from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from calculations._shared import metric_value
from calculations.bloodflow_velocity import PerBeatAnalysisInput, run_per_beat_analysis
from calculations.cross_section import (
    CrossSectionWaveformInput,
    build_cross_section_waveform_metrics,
)
from pipelines.utils.input_access import (
    read_int_setting,
    resolve_dt_seconds,
    resolve_required_array,
)
from pipelines.core.base import ProcessPipeline, ProcessResult, registerPipeline

if TYPE_CHECKING:
    from eye_flow import _PipelineInputView


LEGACY_BAND_LIMITED_SIGNAL_HARMONIC_COUNT = 13
LEGACY_MODAL_DECOMPOSITION_N_MODES = 5


@dataclass(frozen=True)
class WaveformShapeMetricsContext:
    per_beat_analysis: PerBeatAnalysisInput
    artery_cross_section: CrossSectionWaveformInput
    vein_cross_section: CrossSectionWaveformInput
    attrs: dict[str, object]


def _cross_section_input(
    pipeline_input: "_PipelineInputView",
    *,
    vessel_group: str,
    sys_idx_list: np.ndarray,
    band_limited_signal_harmonic_count: int,
    modal_decomposition_n_modes: int,
) -> tuple[CrossSectionWaveformInput, bool, dict[str, str]]:
    safe_velocity = resolve_required_array(
        pipeline_input,
        f"{vessel_group} safe velocity per segment",
        f"{vessel_group}/CrossSections/VelocityPerSegment",
    )
    try:
        truncated_velocity = resolve_required_array(
            pipeline_input,
            f"{vessel_group} truncated velocity per segment",
            f"{vessel_group}/CrossSections/VelocityPerSegmentTruncated",
            f"{vessel_group}/CrossSections/velocity_trunc_seg_mean",
        )
        used_safe_fallback = False
    except KeyError:
        truncated_velocity = safe_velocity
        used_safe_fallback = True

    try:
        profiles = resolve_required_array(
            pipeline_input,
            f"{vessel_group} velocity profiles per segment",
            f"{vessel_group}/CrossSections/VelocityProfileSeg",
        )
        profile_value = np.asarray(profiles.value, dtype=np.float64)
        profile_path = profiles.path
    except KeyError:
        profile_value = None
        profile_path = ""

    return (
        CrossSectionWaveformInput(
            vessel_name=vessel_group.lower(),
            velocity_per_segment=np.asarray(truncated_velocity.value, dtype=np.float64),
            safe_velocity_per_segment=np.asarray(safe_velocity.value, dtype=np.float64),
            velocity_profiles_per_segment=profile_value,
            systolic_acceleration_peak_indexes=sys_idx_list,
            band_limited_signal_harmonic_count=band_limited_signal_harmonic_count,
            modal_decomposition_n_modes=modal_decomposition_n_modes,
        ),
        used_safe_fallback,
        {
            "truncated_velocity_path": truncated_velocity.path,
            "safe_velocity_path": safe_velocity.path,
            "profile_path": profile_path,
        },
    )


def _build_waveform_shape_metrics_context(
    pipeline_input: "_PipelineInputView",
) -> WaveformShapeMetricsContext:
    artery_velocity_signal = resolve_required_array(
        pipeline_input,
        "arterial velocity signal",
        "Artery/Velocity/VelocitySignal",
    )
    vein_velocity_signal = resolve_required_array(
        pipeline_input,
        "venous velocity signal",
        "Vein/Velocity/VelocitySignal",
    )
    systolic_peak_indexes = resolve_required_array(
        pipeline_input,
        "systolic acceleration peak indexes",
        "Artery/Velocity/SystolicAccelerationPeakIndexes",
    )

    band_limited_signal_harmonic_count = read_int_setting(
        pipeline_input,
        default=LEGACY_BAND_LIMITED_SIGNAL_HARMONIC_COUNT,
        keys=(
            "BandLimitedSignalHarmonicCount",
            "band_limited_signal_harmonic_count",
        ),
    )
    modal_decomposition_n_modes = read_int_setting(
        pipeline_input,
        default=LEGACY_MODAL_DECOMPOSITION_N_MODES,
        keys=(
            "ModalDecompositionNModes",
            "modal_decomposition_n_modes",
        ),
    )
    dt_seconds = resolve_dt_seconds(pipeline_input)
    sys_idx_list = np.asarray(systolic_peak_indexes.value, dtype=np.int64).reshape(-1)

    artery_cross_section, artery_safe_fallback, artery_paths = _cross_section_input(
        pipeline_input,
        vessel_group="Artery",
        sys_idx_list=sys_idx_list,
        band_limited_signal_harmonic_count=band_limited_signal_harmonic_count,
        modal_decomposition_n_modes=modal_decomposition_n_modes,
    )
    vein_cross_section, vein_safe_fallback, vein_paths = _cross_section_input(
        pipeline_input,
        vessel_group="Vein",
        sys_idx_list=sys_idx_list,
        band_limited_signal_harmonic_count=band_limited_signal_harmonic_count,
        modal_decomposition_n_modes=modal_decomposition_n_modes,
    )

    attrs: dict[str, object] = {
        "dependency_chain": [
            "perBeatSignalAnalysis",
            "perBeatAnalysis",
            "exportProfilesToH5",
            "exportCrossSectionResults",
        ],
        "dt_seconds": float(dt_seconds),
        "band_limited_signal_harmonic_count": int(
            band_limited_signal_harmonic_count
        ),
        "modal_decomposition_n_modes": int(modal_decomposition_n_modes),
        "arterial_velocity_signal_path": artery_velocity_signal.path,
        "venous_velocity_signal_path": vein_velocity_signal.path,
        "systolic_peak_indexes_path": systolic_peak_indexes.path,
        "artery_truncated_velocity_path": artery_paths["truncated_velocity_path"],
        "artery_safe_velocity_path": artery_paths["safe_velocity_path"],
        "artery_velocity_profiles_path": artery_paths["profile_path"],
        "vein_truncated_velocity_path": vein_paths["truncated_velocity_path"],
        "vein_safe_velocity_path": vein_paths["safe_velocity_path"],
        "vein_velocity_profiles_path": vein_paths["profile_path"],
        "artery_used_safe_segment_velocity_fallback": int(artery_safe_fallback),
        "vein_used_safe_segment_velocity_fallback": int(vein_safe_fallback),
    }

    return WaveformShapeMetricsContext(
        per_beat_analysis=PerBeatAnalysisInput(
            arterial_velocity_signal=np.asarray(
                artery_velocity_signal.value,
                dtype=np.float64,
            ),
            venous_velocity_signal=np.asarray(
                vein_velocity_signal.value,
                dtype=np.float64,
            ),
            systolic_acceleration_peak_indexes=sys_idx_list,
            band_limited_signal_harmonic_count=band_limited_signal_harmonic_count,
            dt_seconds=dt_seconds,
            index_base=None,
        ),
        artery_cross_section=artery_cross_section,
        vein_cross_section=vein_cross_section,
        attrs=attrs,
    )


def run_waveform_shape_metrics(
    pipeline_input: "_PipelineInputView",
) -> tuple[dict[str, object], dict[str, object]]:
    if pipeline_input.hd is None or pipeline_input.dv is None:
        raise ValueError("waveform_shape_metrics requires both HD and DV inputs.")

    context = _build_waveform_shape_metrics_context(pipeline_input)
    metrics: dict[str, object] = {}
    metrics.update(run_per_beat_analysis(context.per_beat_analysis))
    metrics.update(build_cross_section_waveform_metrics(context.artery_cross_section))
    metrics.update(build_cross_section_waveform_metrics(context.vein_cross_section))
    metrics["waveform_shape_metrics/complete/value"] = metric_value(
        np.uint8(1),
        unit="bool",
        matlab_function="waveform_shape_metrics",
    )
    metrics["waveform_shape_metrics/step_count/value"] = metric_value(
        np.int32(3),
        unit="count",
        matlab_function="waveform_shape_metrics",
    )
    return metrics, context.attrs


@registerPipeline(
    name="waveform_shape_metrics",
    description="Waveform-shape metrics and per-beat calculations.",
    required_deps=["numpy", "h5py"],
)
class WaveformShapeMetrics(ProcessPipeline):
    input_slot = "both"

    def run(self, h5file) -> ProcessResult:
        metrics, attrs = run_waveform_shape_metrics(h5file)
        return ProcessResult(metrics=metrics, attrs={"pipeline": self.name, **attrs})
