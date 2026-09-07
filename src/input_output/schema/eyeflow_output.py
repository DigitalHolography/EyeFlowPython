"""Versioned EyeFlow output HDF5 paths."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import h5py

ANGIOEYE_FULL_OUTPUT_SCHEMA = "angioeye_full"
SLIM_TEMP_OUTPUT_SCHEMA = "slim_temp"
EYEFLOW_V2_OUTPUT_SCHEMA = "eyeflow_v2"
ACTIVE_OUTPUT_SCHEMA_VARIANT = EYEFLOW_V2_OUTPUT_SCHEMA


@dataclass(frozen=True)
class DopplerViewAnalysisOutputPaths:
    retinal_velocity_array: str | None
    retinal_artery_velocity_signal: str
    retinal_vein_velocity_signal: str
    retinal_artery_velocity_signal_band_limited: str
    retinal_vein_velocity_signal_band_limited: str
    velocity_map_avg: str | None
    fRMS_avg: str
    fRMS_bkg_avg: str
    velocitysignal_per_beat: str | None
    velocitysignal_filtered: str | None
    beat_indices: str
    time_per_beat: str

    @property
    def retinal_artery_velocity_signal_filtered(self) -> str:
        """Compatibility alias for the previous H5 path field name."""
        return self.retinal_artery_velocity_signal_band_limited

    @property
    def retinal_vein_velocity_signal_filtered(self) -> str:
        """Compatibility alias for the previous H5 path field name."""
        return self.retinal_vein_velocity_signal_band_limited


@dataclass(frozen=True)
class SegmentVelocityOutputPaths:
    velocity_signal: str | None
    velocity_signal_band_limited: str | None = None
    velocity_map_per_segment: str | None = None
    segments: str | None = None


@dataclass(frozen=True)
class VelocityPerBeatOutputPaths:
    velocity_signal: str
    velocity_signal_fft_abs: str
    velocity_signal_fft_arg: str
    velocity_signal_band_limited: str
    segment_velocity_signal: str | None = None
    segment_velocity_signal_band_limited: str | None = None


@dataclass(frozen=True)
class OpticDiscSegmentationOutputPaths:
    mask: str


@dataclass(frozen=True)
class VesselSegmentationOutputPaths:
    mask: str
    branch_label_map: str


@dataclass(frozen=True)
class SegmentationOutputPaths:
    optic_disc: OpticDiscSegmentationOutputPaths
    artery: VesselSegmentationOutputPaths
    vein: VesselSegmentationOutputPaths


@dataclass(frozen=True)
class VelocityProfileOutputPaths:
    transverse_velocity_profile_unmasked: str
    transverse_velocity_profile_masked: str
    longitudinal_velocity_profile_unmasked: str
    longitudinal_velocity_profile_masked: str
    flow_asymmetry_root: str


@dataclass(frozen=True)
class HeartbeatOutputPaths:
    systolic_peak_frame_indices: str
    systolic_cycle_duration_seconds: str
    spectral_fundamental_frequency_hz: str
    spectral_heart_rate_bpm: str
    spectral_heart_rate_standard_error_bpm: str
    spectral_period_seconds: str


@dataclass(frozen=True)
class EyeFlowOutputPaths:
    name: str
    analysis: DopplerViewAnalysisOutputPaths
    artery_segments: SegmentVelocityOutputPaths
    vein_segments: SegmentVelocityOutputPaths
    artery_per_beat: VelocityPerBeatOutputPaths
    vein_per_beat: VelocityPerBeatOutputPaths
    artery_per_beat_safe: SegmentVelocityOutputPaths
    vein_per_beat_safe: SegmentVelocityOutputPaths
    segmentation: SegmentationOutputPaths
    artery_velocity_profiles: VelocityProfileOutputPaths
    vein_velocity_profiles: VelocityProfileOutputPaths
    heartbeat: HeartbeatOutputPaths
    displacement_map: str
    beat_period_seconds: str
    waveform_shape_metrics_root: str
    absolute_waveform_metrics_root: str
    lowrank_waveform_decomposition_root: str
    meta_root: str

    @classmethod
    def active(cls, name: str | None = None) -> "EyeFlowOutputPaths":
        name = ACTIVE_OUTPUT_SCHEMA_VARIANT if name is None else name
        try:
            return OUTPUT_PATH_VARIANTS[name]
        except KeyError as exc:
            known = ", ".join(sorted(OUTPUT_PATH_VARIANTS))
            raise ValueError(
                f"Unknown EyeFlow output schema '{name}'. Known: {known}."
            ) from exc


def _segmentation_paths(root: str) -> SegmentationOutputPaths:
    return SegmentationOutputPaths(
        optic_disc=OpticDiscSegmentationOutputPaths(
            mask=f"{root}/OpticDisc/Mask/value",
        ),
        artery=VesselSegmentationOutputPaths(
            mask=f"{root}/Artery/Mask/value",
            branch_label_map=f"{root}/Artery/BranchLabelMap/value",
        ),
        vein=VesselSegmentationOutputPaths(
            mask=f"{root}/Vein/Mask/value",
            branch_label_map=f"{root}/Vein/BranchLabelMap/value",
        ),
    )


def _velocity_profile_paths(
    root: str,
    *,
    velocity_profile_name: str = "VelocityProfile",
) -> VelocityProfileOutputPaths:
    return VelocityProfileOutputPaths(
        transverse_velocity_profile_unmasked=(
            f"{root}/Transverse{velocity_profile_name}Unmasked/value"
        ),
        transverse_velocity_profile_masked=(
            f"{root}/Transverse{velocity_profile_name}Masked/value"
        ),
        longitudinal_velocity_profile_unmasked=(
            f"{root}/Longitudinal{velocity_profile_name}Unmasked/value"
        ),
        longitudinal_velocity_profile_masked=(
            f"{root}/Longitudinal{velocity_profile_name}Masked/value"
        ),
        flow_asymmetry_root=f"{root}/FlowAsymmetry",
    )


LEGACY_HEARTBEAT_OUTPUT = HeartbeatOutputPaths(
    systolic_peak_frame_indices="analysis/heartbeat/systolic_peak_frame_indices",
    systolic_cycle_duration_seconds="analysis/heartbeat/systolic_cycle_duration_seconds",
    spectral_fundamental_frequency_hz="analysis/heartbeat/spectral_fundamental_frequency_hz",
    spectral_heart_rate_bpm="analysis/heartbeat/spectral_heart_rate_bpm",
    spectral_heart_rate_standard_error_bpm=(
        "analysis/heartbeat/spectral_heart_rate_standard_error_bpm"
    ),
    spectral_period_seconds="analysis/heartbeat/spectral_period_seconds",
)

HEARTBEAT_OUTPUT = HeartbeatOutputPaths(
    systolic_peak_frame_indices="Processing/Heartbeat/Systole/PeakFrameIndices/value",
    systolic_cycle_duration_seconds=(
        "Processing/Heartbeat/Systole/CycleDurationSeconds/value"
    ),
    spectral_fundamental_frequency_hz=(
        "Processing/Heartbeat/Spectral/FundamentalFrequencyHz/value"
    ),
    spectral_heart_rate_bpm="Processing/Heartbeat/Spectral/HeartRateBpm/value",
    spectral_heart_rate_standard_error_bpm=(
        "Processing/Heartbeat/Spectral/HeartRateStandardErrorBpm/value"
    ),
    spectral_period_seconds="Processing/Heartbeat/Spectral/PeriodSeconds/value",
)


ANGIOEYE_FULL_OUTPUT = EyeFlowOutputPaths(
    name=ANGIOEYE_FULL_OUTPUT_SCHEMA,
    analysis=DopplerViewAnalysisOutputPaths(
        retinal_velocity_array="analysis/retinal_velocity_array",
        retinal_artery_velocity_signal="analysis/retinal_artery_velocity_signal",
        retinal_vein_velocity_signal="analysis/retinal_vein_velocity_signal",
        retinal_artery_velocity_signal_band_limited="analysis/velocitysignal_filtered",
        retinal_vein_velocity_signal_band_limited="analysis/vein_velocitysignal_filtered",
        velocity_map_avg="analysis/velocity_map_avg",
        fRMS_avg="analysis/fRMS_avg",
        fRMS_bkg_avg="analysis/fRMS_bkg_avg",
        velocitysignal_per_beat="analysis/velocitysignal_per_beat",
        velocitysignal_filtered="analysis/velocitysignal_filtered",
        beat_indices="analysis/beat_indices",
        time_per_beat="analysis/time_per_beat",
    ),
    artery_segments=SegmentVelocityOutputPaths(velocity_signal=None),
    vein_segments=SegmentVelocityOutputPaths(velocity_signal=None),
    artery_per_beat=VelocityPerBeatOutputPaths(
        velocity_signal="Artery/VelocityPerBeat/VelocitySignalPerBeat/value",
        velocity_signal_fft_abs="Artery/VelocityPerBeat/VelocitySignalPerBeatFFT_abs/value",
        velocity_signal_fft_arg="Artery/VelocityPerBeat/VelocitySignalPerBeatFFT_arg/value",
        velocity_signal_band_limited=(
            "Artery/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
        ),
        segment_velocity_signal=(
            "Artery/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
        ),
        segment_velocity_signal_band_limited=(
            "Artery/VelocityPerBeat/Segments/"
            "VelocitySignalPerBeatPerSegmentBandLimited/value"
        ),
    ),
    vein_per_beat=VelocityPerBeatOutputPaths(
        velocity_signal="Vein/VelocityPerBeat/VelocitySignalPerBeat/value",
        velocity_signal_fft_abs="Vein/VelocityPerBeat/VelocitySignalPerBeatFFT_abs/value",
        velocity_signal_fft_arg="Vein/VelocityPerBeat/VelocitySignalPerBeatFFT_arg/value",
        velocity_signal_band_limited=(
            "Vein/VelocityPerBeat/VelocitySignalPerBeatBandLimited/value"
        ),
        segment_velocity_signal=(
            "Vein/VelocityPerBeat/Segments/VelocitySignalPerBeatPerSegment/value"
        ),
        segment_velocity_signal_band_limited=(
            "Vein/VelocityPerBeat/Segments/"
            "VelocitySignalPerBeatPerSegmentBandLimited/value"
        ),
    ),
    artery_per_beat_safe=SegmentVelocityOutputPaths(velocity_signal=None),
    vein_per_beat_safe=SegmentVelocityOutputPaths(velocity_signal=None),
    segmentation=_segmentation_paths("Segmentation"),
    artery_velocity_profiles=_velocity_profile_paths(
        "Artery/CrossSections/RawProfile",
        velocity_profile_name="VelocityProfileSeg",
    ),
    vein_velocity_profiles=_velocity_profile_paths(
        "Vein/CrossSections/RawProfile",
        velocity_profile_name="VelocityProfileSeg",
    ),
    heartbeat=LEGACY_HEARTBEAT_OUTPUT,
    displacement_map="Processing/DisplacementMap",
    beat_period_seconds="Artery/VelocityPerBeat/beatPeriodSeconds/value",
    waveform_shape_metrics_root="Metrics/waveform_shape_metrics",
    absolute_waveform_metrics_root="Metrics/absolute_waveform_metrics",
    lowrank_waveform_decomposition_root=(
        "Metrics/lowrank_waveform_decomposition"
    ),
    meta_root="Meta",
)


SLIM_TEMP_OUTPUT = EyeFlowOutputPaths(
    name=SLIM_TEMP_OUTPUT_SCHEMA,
    analysis=DopplerViewAnalysisOutputPaths(
        retinal_velocity_array=None,
        retinal_artery_velocity_signal="artery/velocity/signal/value",
        retinal_vein_velocity_signal="vein/velocity/signal/value",
        retinal_artery_velocity_signal_band_limited="artery/velocity/filtered_signal/value",
        retinal_vein_velocity_signal_band_limited="vein/velocity/filtered_signal/value",
        velocity_map_avg="topo/velocity_map_avg/value",
        fRMS_avg="topo/fRMS_avg/value",
        fRMS_bkg_avg="topo/fRMS_bkg_avg/value",
        velocitysignal_per_beat="artery/velocity/perbeat/filtered_signal/value",
        velocitysignal_filtered="artery/velocity/filtered_signal/value",
        beat_indices="perbeat/beat_indices/value",
        time_per_beat="perbeat/time_per_beat/value",
    ),
    artery_segments=SegmentVelocityOutputPaths(velocity_signal=None),
    vein_segments=SegmentVelocityOutputPaths(velocity_signal=None),
    artery_per_beat=VelocityPerBeatOutputPaths(
        velocity_signal="artery/velocity/perbeat/signal/value",
        velocity_signal_fft_abs="artery/velocity/perbeat/fft_abs/value",
        velocity_signal_fft_arg="artery/velocity/perbeat/fft_arg/value",
        velocity_signal_band_limited="artery/velocity/perbeat/band_limited/value",
        segment_velocity_signal="artery/velocity/perbeat/segments/signal/value",
        segment_velocity_signal_band_limited=(
            "artery/velocity/perbeat/segments/band_limited/value"
        ),
    ),
    vein_per_beat=VelocityPerBeatOutputPaths(
        velocity_signal="vein/velocity/perbeat/signal/value",
        velocity_signal_fft_abs="vein/velocity/perbeat/fft_abs/value",
        velocity_signal_fft_arg="vein/velocity/perbeat/fft_arg/value",
        velocity_signal_band_limited="vein/velocity/perbeat/band_limited/value",
        segment_velocity_signal="vein/velocity/perbeat/segments/signal/value",
        segment_velocity_signal_band_limited=(
            "vein/velocity/perbeat/segments/band_limited/value"
        ),
    ),
    artery_per_beat_safe=SegmentVelocityOutputPaths(velocity_signal=None),
    vein_per_beat_safe=SegmentVelocityOutputPaths(velocity_signal=None),
    segmentation=_segmentation_paths("Segmentation"),
    artery_velocity_profiles=_velocity_profile_paths(
        "artery/cross_sections/RawProfile"
    ),
    vein_velocity_profiles=_velocity_profile_paths(
        "vein/cross_sections/RawProfile"
    ),
    heartbeat=LEGACY_HEARTBEAT_OUTPUT,
    displacement_map="Processing/DisplacementMap",
    beat_period_seconds="perbeat/beat_period_seconds/value",
    waveform_shape_metrics_root="Metrics/waveform_shape_metrics",
    absolute_waveform_metrics_root="Metrics/absolute_waveform_metrics",
    lowrank_waveform_decomposition_root=(
        "Metrics/lowrank_waveform_decomposition"
    ),
    meta_root="Meta",
)


EYEFLOW_V2_OUTPUT = EyeFlowOutputPaths(
    name=EYEFLOW_V2_OUTPUT_SCHEMA,
    analysis=DopplerViewAnalysisOutputPaths(
        retinal_velocity_array=None,
        retinal_artery_velocity_signal="Processing/Velocity/global/Artery/Raw/value",
        retinal_vein_velocity_signal="Processing/Velocity/global/Vein/Raw/value",
        retinal_artery_velocity_signal_band_limited=(
            "Processing/Velocity/global/Artery/BandLimited/value"
        ),
        retinal_vein_velocity_signal_band_limited=(
            "Processing/Velocity/global/Vein/BandLimited/value"
        ),
        velocity_map_avg=None,
        fRMS_avg="Processing/FrequencyMaps/fRMS_avg/value",
        fRMS_bkg_avg="Processing/FrequencyMaps/fRMS_bkg_avg/value",
        velocitysignal_per_beat=None,
        velocitysignal_filtered=None,
        beat_indices="Processing/Heartbeat/Systole/PeakFrameIndices/value",
        time_per_beat="Processing/Heartbeat/Systole/CycleDurationSeconds/value",
    ),
    artery_segments=SegmentVelocityOutputPaths(
        velocity_signal="Processing/Velocity/segments/Artery/Raw/value",
        velocity_signal_band_limited=(
            "Processing/Velocity/segments/Artery/BandLimited/value"
        ),
        velocity_map_per_segment="Processing/VelocityMapPerSegment/Artery",
        segments="Segmentation/Artery/Segments",
    ),
    vein_segments=SegmentVelocityOutputPaths(
        velocity_signal="Processing/Velocity/segments/Vein/Raw/value",
        velocity_signal_band_limited=(
            "Processing/Velocity/segments/Vein/BandLimited/value"
        ),
        velocity_map_per_segment="Processing/VelocityMapPerSegment/Vein",
        segments="Segmentation/Vein/Segments",
    ),
    artery_per_beat=VelocityPerBeatOutputPaths(
        velocity_signal="Processing/VelocityPerBeat/Artery/Raw/value",
        velocity_signal_fft_abs="Processing/VelocityPerBeat/Artery/FFTAbs/value",
        velocity_signal_fft_arg="Processing/VelocityPerBeat/Artery/FFTPhase/value",
        velocity_signal_band_limited=(
            "Processing/VelocityPerBeat/Artery/BandLimited/value"
        ),
        segment_velocity_signal=(
            "Processing/VelocityPerBeat/Artery/Segments/Raw/value"
        ),
        segment_velocity_signal_band_limited=(
            "Processing/VelocityPerBeat/Artery/Segments/BandLimited/value"
        ),
    ),
    vein_per_beat=VelocityPerBeatOutputPaths(
        velocity_signal="Processing/VelocityPerBeat/Vein/Raw/value",
        velocity_signal_fft_abs="Processing/VelocityPerBeat/Vein/FFTAbs/value",
        velocity_signal_fft_arg="Processing/VelocityPerBeat/Vein/FFTPhase/value",
        velocity_signal_band_limited=(
            "Processing/VelocityPerBeat/Vein/BandLimited/value"
        ),
        segment_velocity_signal=(
            "Processing/VelocityPerBeat/Vein/Segments/Raw/value"
        ),
        segment_velocity_signal_band_limited=(
            "Processing/VelocityPerBeat/Vein/Segments/BandLimited/value"
        ),
    ),
    artery_per_beat_safe=SegmentVelocityOutputPaths(
        velocity_signal=(
            "Processing/VelocityPerBeatSafe/Artery/Segments/Raw/value"
        ),
        velocity_signal_band_limited=(
            "Processing/VelocityPerBeatSafe/Artery/Segments/BandLimited/value"
        ),
    ),
    vein_per_beat_safe=SegmentVelocityOutputPaths(
        velocity_signal=(
            "Processing/VelocityPerBeatSafe/Vein/Segments/Raw/value"
        ),
        velocity_signal_band_limited=(
            "Processing/VelocityPerBeatSafe/Vein/Segments/BandLimited/value"
        ),
    ),
    segmentation=_segmentation_paths("Segmentation"),
    artery_velocity_profiles=_velocity_profile_paths(
        "Processing/VelocityProfiles/Artery"
    ),
    vein_velocity_profiles=_velocity_profile_paths(
        "Processing/VelocityProfiles/Vein"
    ),
    heartbeat=HEARTBEAT_OUTPUT,
    displacement_map="Processing/DisplacementMap",
    beat_period_seconds="Processing/VelocityPerBeat/BeatPeriodSeconds/value",
    waveform_shape_metrics_root="Processing/Metrics/waveform_shape_metrics",
    absolute_waveform_metrics_root="Processing/Metrics/absolute_waveform_metrics",
    lowrank_waveform_decomposition_root=(
        "Processing/Metrics/lowrank_waveform_decomposition"
    ),
    meta_root="Meta",
)


OUTPUT_PATH_VARIANTS = {
    ANGIOEYE_FULL_OUTPUT_SCHEMA: ANGIOEYE_FULL_OUTPUT,
    SLIM_TEMP_OUTPUT_SCHEMA: SLIM_TEMP_OUTPUT,
    EYEFLOW_V2_OUTPUT_SCHEMA: EYEFLOW_V2_OUTPUT,
}

ZERO_BASED_INDEX_PATHS = frozenset(
    paths.analysis.beat_indices for paths in OUTPUT_PATH_VARIANTS.values()
)


def systolic_index_base_for_path(path: str) -> int | None:
    from input_output.writers.h5 import normalize_h5_path

    normalized = normalize_h5_path(path)
    return 0 if normalized in ZERO_BASED_INDEX_PATHS else None


def iter_metric_datasets(group: h5py.Group) -> Iterator[tuple[str, h5py.Dataset]]:
    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    datasets: list[tuple[str, h5py.Dataset]] = []
    group.visititems(visitor)
    yield from datasets
