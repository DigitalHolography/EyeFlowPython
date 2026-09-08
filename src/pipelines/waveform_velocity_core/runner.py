"""Build shared DopplerView, spatial, and segment-analysis state."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

from calculations.blood_flow_velocity import (
    CrossSectionSignalResult,
    HeartbeatAnalysisResult,
    PerBeatAnalysisInput,
    segment_velocity_results,
    spectral_heartbeat_analysis,
)
from calculations.topology import (
    SegmentRingSettings,
    image_half_diagonal,
)
from input_output import EyeFlowOutputPaths
from pipelines.displacement_map.constants import DEFAULT_REGISTRATION_METHOD
from pipelines.displacement_map.runner import (
    DISPLACEMENT_MAP_STATE,
    DisplacementMapArtifacts,
)
from pipeline_engine.imports import (
    HolodopplerTiming,
    np,
    read_int_setting,
)
from utils.logger import Logger

from .dopplerview.constants import (
    LEGACY_FILTER_VELOCITY_SIGNALS,
    LEGACY_VELOCITY_SIGNAL_LOWPASS_HZ,
)
from .constants import (
    LEGACY_BAND_LIMITED_SIGNAL_HARMONIC_COUNT,
    NUMBER_OF_RADII_IN_FOV,
    SEGMENT_INNER_RADIUS_FRAC,
    SEGMENT_OUTER_RADIUS_FRAC,
)
from .cross_section_images import export_rotated_mean_pngs
from .dopplerview.outputs import (
    pack_dopplerview_shared_outputs,
)
from .dopplerview.runner import run_dopplerview_analysis
from .scratch import waveform_scratch_h5
from .sources import WaveformVelocitySourceData, WaveformVelocitySources
from .branch_identity_debug import export_branch_identity_stage_pngs
from .figures import export_pulse_pngs
from .per_beat import run_velocity_per_beat_metrics
from .segmentation import pack_segmentation_outputs


WAVEFORM_CONTEXT_STATE = "waveform_velocity_context"
VELOCITY_PER_BEAT_RESULT_STATE = "velocity_per_beat_result"
VELOCITY_PER_BEAT_OUTPUTS_STATE = "velocity_per_beat_outputs"


@dataclass(frozen=True)
class WaveformVelocityCoreContext:
    source_data: WaveformVelocitySourceData
    per_beat_analysis: PerBeatAnalysisInput
    artery_segment_result: CrossSectionSignalResult | None
    vein_segment_result: CrossSectionSignalResult | None
    velocity_analysis: dict[str, object]
    attrs: dict[str, object]


def run_waveform_velocity_core(
    ctx,
) -> tuple[dict[str, object], dict[str, object]]:
    """Run the shared DopplerView and spatial velocity foundation once."""
    ctx.require_inputs("hd", "dv")

    core_started = perf_counter()
    with waveform_scratch_h5(ctx) as scratch_h5:
        Logger.log("Starting waveform velocity core context build...")
        segments_required = _segments_required(ctx)
        context = _build_waveform_velocity_core_context(
            ctx,
            scratch_h5,
            segments_required=segments_required,
        )
        metrics = pack_dopplerview_shared_outputs(context.velocity_analysis)
        metrics.update(_pack_meta_outputs(context))
        metrics.update(
            pack_segmentation_outputs(
                context.source_data,
                context.artery_segment_result,
                context.vein_segment_result,
            )
        )
        ctx.state.set(WAVEFORM_CONTEXT_STATE, context)

        if _per_beat_required(ctx):
            with _logged_stage("shared per-beat velocity analysis"):
                per_beat_result, velocity_outputs = run_velocity_per_beat_metrics(context)
            ctx.state.set(VELOCITY_PER_BEAT_RESULT_STATE, per_beat_result)
            ctx.state.set(VELOCITY_PER_BEAT_OUTPUTS_STATE, velocity_outputs)
            if _pulse_pngs_required(ctx):
                _export_pulse_pngs(ctx, context, per_beat_result)

    Logger.log(f"Completed waveform velocity core in {perf_counter() - core_started:.1f}s.")
    return metrics, context.attrs


def _per_beat_required(ctx) -> bool:
    if ctx.pipeline_scheduled("lowrank_waveform_decomposition"):
        return True
    velocity_options = ctx.options_for("waveform_velocity")
    metric_options = ctx.options_for("waveform_shape_metrics")
    absolute_options = (
        ctx.options_for("absolute_waveform_metrics")
        if ctx.pipeline_scheduled("absolute_waveform_metrics")
        else frozenset()
    )
    if ctx.pipeline_scheduled("pdf_report"):
        return True
    if ctx.pipeline_scheduled("waveform_velocity"):
        return bool(
            {"per_beat", "quadrants"} & velocity_options
            or "quadrants" in metric_options
            or absolute_options
        )
    return bool(
        (
            metric_options and ctx.pipeline_scheduled("waveform_shape_metrics")
        )
        or (
            absolute_options
            and ctx.pipeline_scheduled("absolute_waveform_metrics")
        )
    )


def _segments_required(ctx) -> bool:
    """Return whether any selected product needs spatial vessel segments."""
    if ctx.pipeline_scheduled("lowrank_waveform_decomposition"):
        return True
    velocity_options = ctx.options_for("waveform_velocity")
    metric_options = ctx.options_for("waveform_shape_metrics")
    absolute_options = (
        ctx.options_for("absolute_waveform_metrics")
        if ctx.pipeline_scheduled("absolute_waveform_metrics")
        else frozenset()
    )
    if ctx.pipeline_scheduled("waveform_velocity"):
        return bool(
            {
                "segments",
                "segment_velocity_maps",
                "velocity_profiles",
                "quadrants",
            }
            & velocity_options
            or "quadrants" in metric_options
            or "segments" in absolute_options
            or "quadrants" in absolute_options
        )
    return bool(
        {"segments", "quadrants"} & metric_options
        or "segments" in absolute_options
    )


def _pulse_pngs_required(ctx) -> bool:
    return bool(
        ctx.pipeline_scheduled("pdf_report")
        or (
            ctx.pipeline_scheduled("waveform_velocity")
            and ctx.option_enabled("per_beat", pipeline="waveform_velocity")
        )
    )


def _displacement_segment_maps_required(ctx) -> bool:
    """Return whether full rotated displacement maps must remain in memory."""
    return bool(
        ctx.pipeline_scheduled("waveform_velocity")
        and ctx.option_enabled(
            "segment_velocity_maps",
            pipeline="waveform_velocity",
        )
    )

def _build_waveform_velocity_core_context(
    ctx,
    scratch_h5,
    *,
    segments_required: bool,
) -> WaveformVelocityCoreContext:
    with _logged_stage("waveform source loading"):
        source_data = WaveformVelocitySources.from_context(ctx).load()
    timing = source_data.timing
    velocity_analysis, analysis_source = _resolve_velocity_analysis(
        source_data,
        ctx,
        scratch_h5,
        retain_velocity_video=(segments_required or _pulse_pngs_required(ctx)),
    )
    with _loaded_displacement_maps(
        ctx,
        enabled=segments_required,
    ) as displacement_maps:
        velocity_map = velocity_analysis["velocity_map"] if segments_required else None
        harmonic_count = _band_limited_harmonic_count(ctx)
        number_of_radii_in_fov = _number_of_radii_in_fov(ctx)
        per_beat_analysis, artery_segments, vein_segments = (
            _per_beat_input_from_analysis(
                velocity_analysis,
                source_data,
                timing,
                harmonic_count,
                ctx,
                velocity_map=velocity_map,
                displacement_maps=displacement_maps,
                number_of_radii_in_fov=number_of_radii_in_fov,
                segments_required=segments_required,
            )
        )

    return WaveformVelocityCoreContext(
        source_data=source_data,
        per_beat_analysis=per_beat_analysis,
        artery_segment_result=artery_segments,
        vein_segment_result=vein_segments,
        velocity_analysis=velocity_analysis,
        attrs=_context_attrs(
            source_data,
            timing,
            harmonic_count,
            analysis_source,
            per_beat_analysis.heartbeat,
            number_of_radii_in_fov,
        ),
    )


def _resolve_velocity_analysis(
    source_data: WaveformVelocitySourceData,
    ctx,
    scratch_h5,
    *,
    retain_velocity_video: bool,
) -> tuple[dict[str, object], str]:
    with _logged_stage("EyeFlow velocity analysis from HD moments"):
        velocity_analysis = run_dopplerview_analysis(
            source_data,
            scratch_h5,
            retain_velocity_video=retain_velocity_video,
        )
    return velocity_analysis, "eyeflow_recomputed_dopplerview_analysis"


@contextmanager
def _loaded_displacement_maps(ctx, *, enabled: bool):
    if not enabled:
        yield {}
        return
    with _logged_stage("displacement map loading"):
        displacement_maps = _load_displacement_maps(ctx)
    try:
        yield displacement_maps
    finally:
        _release_displacement_maps(ctx, displacement_maps)


def _load_displacement_maps(ctx) -> dict[str, dict[str, object]]:
    if not ctx.pipeline_scheduled("displacement_map"):
        return {}

    artifacts = ctx.state.get(DISPLACEMENT_MAP_STATE)
    if not isinstance(artifacts, DisplacementMapArtifacts):
        raise RuntimeError(
            "The scheduled displacement_map pipeline did not prepare its "
            "in-run displacement artifacts."
        )
    method = _displacement_method_name(
        artifacts.registration_method or DEFAULT_REGISTRATION_METHOD
    )
    loaded_by_path: dict[str, object] = {}
    displacement_maps: dict[str, dict[str, object]] = {}
    for vessel, field_path in artifacts.field_paths_by_vessel.items():
        normalized_path = str(field_path.resolve())
        displacement_map = loaded_by_path.get(normalized_path)
        if displacement_map is None:
            displacement_map = np.load(field_path, mmap_mode="r")
            loaded_by_path[normalized_path] = displacement_map
        displacement_maps[vessel] = {method: displacement_map}
    if not displacement_maps:
        raise RuntimeError("No vessel displacement-map artifacts were prepared.")
    return displacement_maps


def _release_displacement_maps(
    ctx,
    displacement_maps: Mapping[str, Mapping[str, object]],
) -> None:
    closed: set[int] = set()
    for maps_for_vessel in displacement_maps.values():
        for displacement_map in maps_for_vessel.values():
            identity = id(displacement_map)
            if identity in closed:
                continue
            closed.add(identity)
            mmap = getattr(displacement_map, "_mmap", None)
            if mmap is not None:
                mmap.close()
    artifacts = ctx.state.get(DISPLACEMENT_MAP_STATE)
    if isinstance(artifacts, DisplacementMapArtifacts):
        artifacts.cleanup()


def _displacement_method_name(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    method = str(value).strip()
    if not method or "/" in method:
        raise ValueError(
            "Displacement registration method names must be non-empty HDF5 path segments."
        )
    return method


def _band_limited_harmonic_count(ctx) -> int:
    return read_int_setting(
        ctx,
        default=LEGACY_BAND_LIMITED_SIGNAL_HARMONIC_COUNT,
        keys=("BandLimitedSignalHarmonicCount", "band_limited_signal_harmonic_count"),
    )


def _number_of_radii_in_fov(ctx) -> int:
    value = read_int_setting(
        ctx,
        default=NUMBER_OF_RADII_IN_FOV,
        keys=(
            "number_of_radii_in_FOV",
            "number_of_radii_in_fov",
            "NumberOfRadiiInFOV",
            # Accept the earlier spelling for compatibility.
            "number_of_radii_over_FOV",
            "number_of_radii_over_fov",
            "NumberOfRadiiOverFOV",
        ),
    )
    if value < 1:
        raise ValueError("number_of_radii_in_FOV must be positive.")
    return value


def _per_beat_input_from_analysis(
    velocity_analysis: Mapping[str, object],
    source_data: WaveformVelocitySourceData,
    timing: HolodopplerTiming,
    harmonic_count: int,
    ctx,
    *,
    velocity_map=None,
    displacement_maps: Mapping[str, Mapping[str, object]] | None = None,
    number_of_radii_in_fov: int = NUMBER_OF_RADII_IN_FOV,
    segments_required: bool,
) -> tuple[
    PerBeatAnalysisInput,
    CrossSectionSignalResult | None,
    CrossSectionSignalResult | None,
]:
    if segments_required:
        if velocity_map is None:
            raise ValueError("velocity_map is required for segment extraction.")
        ring_settings = _segment_ring_settings(
            source_data.optic_disc_width,
            source_data.optic_disc_height,
            image_shape=velocity_map.shape[-2:],
            optic_disc_center=source_data.optic_disc_center,
            number_of_radii_in_FOV=number_of_radii_in_fov,
        )
        artery_segments, vein_segments = _segment_velocity_inputs(
            velocity_map,
            displacement_maps or {},
            source_data,
            ring_settings,
            ctx,
        )
    else:
        Logger.log("Skipping segment velocity extraction; no selected output requires it.")
        artery_segments, vein_segments = None, None
    arterial_velocity_signal, venous_velocity_signal = (
        _raw_velocity_signals_for_per_beat(velocity_analysis)
    )
    beat_indexes = np.asarray(
        velocity_analysis["beat_indices"],
        dtype=np.int32,
    )
    cached_heartbeat = velocity_analysis.get("_heartbeat_analysis_result")
    heartbeat = (
        cached_heartbeat.spectral
        if isinstance(cached_heartbeat, HeartbeatAnalysisResult)
        else spectral_heartbeat_analysis(
            arterial_velocity_signal,
            timing.dt_seconds,
            beat_indexes.size,
        )
    )
    inputs = PerBeatAnalysisInput(
        arterial_velocity_signal=arterial_velocity_signal,
        venous_velocity_signal=venous_velocity_signal,
        cycle_boundary_indexes=beat_indexes,
        band_limited_signal_harmonic_count=harmonic_count,
        heartbeat=heartbeat,
        dt_seconds=timing.dt_seconds,
        arterial_velocity_segments=_waveform_segment_input(
            artery_segments,
            include_segments=segments_required,
        ),
        venous_velocity_segments=_waveform_segment_input(
            vein_segments,
            include_segments=segments_required,
        ),
        arterial_safe_velocity_segments=_safe_waveform_segment_input(
            artery_segments,
            include_segments=segments_required,
        ),
        venous_safe_velocity_segments=_safe_waveform_segment_input(
            vein_segments,
            include_segments=segments_required,
        ),
        index_base=source_data.provenance["beat_index_base"],
    )
    return inputs, artery_segments, vein_segments


def _raw_velocity_signals_for_per_beat(
    velocity_analysis: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            velocity_analysis["retinal_artery_velocity_signal"],
            dtype=np.float32,
        ),
        np.asarray(
            velocity_analysis["retinal_vein_velocity_signal"],
            dtype=np.float32,
        ),
    )


def _segment_velocity_inputs(
    velocity_map,
    displacement_maps: Mapping[str, Mapping[str, object]],
    source_data: WaveformVelocitySourceData,
    ring_settings: SegmentRingSettings,
    ctx,
) -> tuple[CrossSectionSignalResult, CrossSectionSignalResult]:
    with _logged_stage("segment velocity extraction"):
        artery, vein = segment_velocity_results(
            velocity_map,
            source_data.retinal_artery_mask,
            source_data.retinal_vein_mask,
            source_data.optic_disc_center,
            ring_settings,
            source_data.cross_section_settings,
            artery_displacement_maps=displacement_maps.get("artery", {}),
            vein_displacement_maps=displacement_maps.get("vein", {}),
            retain_displacement_maps=_displacement_segment_maps_required(ctx),
        )
    if ctx.output.available:
        with _logged_stage("rotated mean PNG export"):
            export_rotated_mean_pngs(ctx.output, artery, "arteries")
            export_rotated_mean_pngs(ctx.output, vein, "veins")
    _export_branch_identity_debug(
        ctx,
        artery,
        source_data.optic_disc_center,
        ring_settings,
        "artery",
    )
    _export_branch_identity_debug(
        ctx,
        vein,
        source_data.optic_disc_center,
        ring_settings,
        "vein",
    )
    return artery, vein


def _waveform_segment_input(
    result: CrossSectionSignalResult | None,
    *,
    include_segments: bool,
) -> np.ndarray | None:
    if not include_segments or result is None or result.branch_ids.size == 0:
        return None
    return result.velocity


def _safe_waveform_segment_input(
    result: CrossSectionSignalResult | None,
    *,
    include_segments: bool,
) -> np.ndarray | None:
    if not include_segments or result is None or result.branch_ids.size == 0:
        return None
    return result.safe_velocity


def _export_branch_identity_debug(
    ctx,
    result: CrossSectionSignalResult,
    optic_disc_center,
    ring_settings: SegmentRingSettings,
    prefix: str,
) -> None:
    if not ctx.output.available:
        return
    export_branch_identity_stage_pngs(
        ctx.output,
        result.branch_identity.stages,
        prefix,
        optic_disc_center,
        ring_settings,
        segment_center_xy=result.segment_center_xy,
        profile_window_bounds_xyxy=result.profile_window_bounds_xyxy,
    )


def _export_pulse_pngs(ctx, context: WaveformVelocityCoreContext, per_beat_result) -> None:
    if not ctx.output.available:
        return
    with _logged_stage("pulse-analysis PNG export"):
        export_pulse_pngs(ctx.output, context, per_beat_result)


@contextmanager
def _logged_stage(label: str):
    started = perf_counter()
    Logger.log(f"Starting {label}...")
    yield
    Logger.log(f"Completed {label} in {perf_counter() - started:.1f}s.")


def _segment_ring_settings(
    optic_disc_width=None,
    optic_disc_height=None,
    *,
    image_shape=None,
    optic_disc_center=None,
    number_of_radii_in_FOV: int = NUMBER_OF_RADII_IN_FOV,
) -> SegmentRingSettings:
    if number_of_radii_in_FOV < 1:
        raise ValueError("number_of_radii_in_FOV must be positive.")
    width_px = _positive_geometry_scalar(optic_disc_width)
    height_px = _positive_geometry_scalar(optic_disc_height)
    if width_px is not None and height_px is not None and image_shape is not None:
        ny, nx = (int(size) for size in image_shape)
        radius_scale = image_half_diagonal(ny, nx)
        ring_width_px = max(nx, ny) / float(number_of_radii_in_FOV)
        radial_step = ring_width_px / max(radius_scale, 1.0)
        inner = min((max(width_px, height_px) / 2.0) / radius_scale, 1.0)
        outer = 1.0
        count = max(1, int(np.ceil((outer - inner) / radial_step)))
        return SegmentRingSettings(
            inner,
            outer,
            radial_step,
            count,
            radial_step,
        )

    # Use the historical radial range when image/disc geometry is unavailable.
    inner = SEGMENT_INNER_RADIUS_FRAC
    outer = SEGMENT_OUTER_RADIUS_FRAC
    radial_step = 1.0 / float(number_of_radii_in_FOV)
    count = max(1, int(np.ceil((outer - inner) / radial_step)))
    return SegmentRingSettings(
        inner,
        outer,
        radial_step,
        count,
        radial_step,
    )


def _positive_geometry_scalar(value) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array[0]) or array[0] <= 0:
        return None
    return float(array[0])


def _context_attrs(
    source_data: WaveformVelocitySourceData,
    timing: HolodopplerTiming,
    harmonic_count: int,
    analysis_source: str,
    heartbeat,
    number_of_radii_in_fov: int,
) -> dict[str, object]:
    output_paths = EyeFlowOutputPaths.active()
    analysis_paths = output_paths.analysis
    dependency_chain = (
        ["dopplerview.h5.analysis"]
        if analysis_source == "dopplerview_h5_analysis"
        else [
            "holodoppler.h5.moment0_moment2",
            "dopplerview.h5.segmentation",
            "eyeflow.dopplerview_analysis.recomputed",
        ]
    )
    width = _positive_geometry_scalar(source_data.optic_disc_width)
    height = _positive_geometry_scalar(source_data.optic_disc_height)
    return {
        "dependency_chain": dependency_chain + [
            "blood_flow_velocity.signal_analysis.heartbeat.spectral",
            "blood_flow_velocity.signal_analysis.per_beat.signal",
            "blood_flow_velocity.signal_analysis.per_beat.runner",
        ],
        "analysis_source": analysis_source,
        "output_schema": output_paths.name,
        "velocity_section_geometry": (
            "optic_disc_relative"
            if width is not None and height is not None
            else "frame_relative_fallback"
        ),
        "velocity_section_inner_radius_fraction": float(
            SEGMENT_INNER_RADIUS_FRAC
        ),
        "velocity_section_outer_radius_fraction": float(
            SEGMENT_OUTER_RADIUS_FRAC
        ),
        "velocity_section_outer_to_disc_radius": float(
            SEGMENT_OUTER_RADIUS_FRAC / SEGMENT_INNER_RADIUS_FRAC
        ),
        "number_of_radii_in_FOV": int(number_of_radii_in_fov),
        "arterial_velocity_signal_path": (
            analysis_paths.retinal_artery_velocity_signal
        ),
        "venous_velocity_signal_path": (
            analysis_paths.retinal_vein_velocity_signal
        ),
        "systolic_peak_indexes_path": analysis_paths.beat_indices,
        "beat_period_seconds_path": output_paths.beat_period_seconds,
        "heart_rate_hz": float(heartbeat.heart_rate_hz),
        "heart_rate_bpm": float(heartbeat.heart_rate_bpm),
        "heart_rate_ste_hz": float(heartbeat.heart_rate_ste_hz),
        "heart_rate_ste_bpm": float(heartbeat.heart_rate_ste_bpm),
        "sampling_freq": float(timing.sampling_freq),
        "batch_stride": float(timing.batch_stride),
        "dt_seconds": float(timing.dt_seconds),
        "band_limited_signal_harmonic_count": int(harmonic_count),
        "filter_velocity_signals": bool(LEGACY_FILTER_VELOCITY_SIGNALS),
        "velocity_signal_lowpass_hz": float(LEGACY_VELOCITY_SIGNAL_LOWPASS_HZ),
    }


def _pack_meta_outputs(context: WaveformVelocityCoreContext) -> dict[str, object]:
    schema = EyeFlowOutputPaths.active()
    timing = context.source_data.timing
    return {
        f"{schema.meta_root}/SamplingFrequencyHz/value": (
            np.float32(timing.sampling_freq),
            {"unit": "Hz"},
        ),
        f"{schema.meta_root}/BatchStride/value": np.float32(timing.batch_stride),
        f"{schema.meta_root}/FrameIntervalSeconds/value": (
            np.float32(timing.dt_seconds),
            {"unit": "s"},
        ),
    }
