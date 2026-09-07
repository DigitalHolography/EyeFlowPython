"""Orchestrate selectable velocity output products."""

from time import perf_counter

from input_output import EyeFlowOutputPaths
from pipelines.waveform_velocity_core.runner import (
    VELOCITY_PER_BEAT_OUTPUTS_STATE,
    VELOCITY_PER_BEAT_RESULT_STATE,
    WAVEFORM_CONTEXT_STATE,
)
from pipelines.waveform_velocity_core.per_beat import run_velocity_per_beat_metrics
from utils.logger import Logger

from .continuous import (
    pack_continuous_velocity_outputs,
    pack_segment_velocity_outputs,
)
from .quadrants import pack_quadrant_velocity_outputs
from .profiles import (
    pack_cross_section_displacement_profile_outputs,
    pack_cross_section_profile_outputs,
    pack_displacement_magnitude_outputs,
    pack_displacement_profile_outputs,
)
from .segment_maps import (
    pack_displacement_segment_map_outputs,
    pack_segment_map_outputs,
)
from .segment_velocity_map_avi import export_segment_velocity_map_avis


def run_waveform_velocity(ctx) -> dict[str, object]:
    """Publish base velocity plus the selected derived velocity products."""
    context = _required_state(ctx, WAVEFORM_CONTEXT_STATE)
    selected = ctx.options_for("waveform_velocity")
    metrics = pack_continuous_velocity_outputs(context.velocity_analysis)
    segments_selected = "segments" in selected
    maps_selected = "segment_velocity_maps" in selected
    if segments_selected:
        metrics.update(
            pack_segment_velocity_outputs(
                context.artery_segment_result,
                context.vein_segment_result,
                source_data=context.source_data,
            )
        )
    if maps_selected:
        map_started = perf_counter()
        Logger.log("Starting per-beat segment velocity-map interpolation...")
        segment_map_outputs = pack_segment_map_outputs(
            context.artery_segment_result,
            context.vein_segment_result,
            context.per_beat_analysis.cycle_boundary_indexes,
            index_base=int(context.source_data.provenance["beat_index_base"]),
        )
        Logger.log(
            "Completed per-beat segment velocity-map interpolation in "
            f"{perf_counter() - map_started:.1f}s."
        )
        metrics.update(segment_map_outputs)
        metrics.update(
            pack_displacement_segment_map_outputs(
                context.artery_segment_result,
                context.vein_segment_result,
                context.per_beat_analysis.cycle_boundary_indexes,
                index_base=int(
                    context.source_data.provenance["beat_index_base"]
                ),
            )
        )
        output = getattr(ctx, "output", None)
        if getattr(output, "available", False):
            avi_started = perf_counter()
            Logger.log("Starting segment velocity-map AVI export...")
            export_segment_velocity_map_avis(
                output,
                context.artery_segment_result,
                context.vein_segment_result,
                segment_map_outputs,
            )
            Logger.log(
                "Completed segment velocity-map AVI export in "
                f"{perf_counter() - avi_started:.1f}s."
            )

    per_beat_result = ctx.state.get(VELOCITY_PER_BEAT_RESULT_STATE)
    velocity_outputs = ctx.state.get(VELOCITY_PER_BEAT_OUTPUTS_STATE, {})
    if "per_beat" in selected or ctx.pipeline_scheduled("pdf_report"):
        if per_beat_result is None:
            per_beat_result, velocity_outputs = run_velocity_per_beat_metrics(context)
            ctx.state.set(VELOCITY_PER_BEAT_RESULT_STATE, per_beat_result)
            ctx.state.set(VELOCITY_PER_BEAT_OUTPUTS_STATE, velocity_outputs)
        if segments_selected:
            metrics.update(velocity_outputs)
        else:
            schema = EyeFlowOutputPaths.active()
            segment_paths = {
                schema.artery_per_beat.segment_velocity_signal,
                schema.artery_per_beat.segment_velocity_signal_band_limited,
                schema.vein_per_beat.segment_velocity_signal,
                schema.vein_per_beat.segment_velocity_signal_band_limited,
                schema.artery_per_beat_safe.velocity_signal,
                schema.artery_per_beat_safe.velocity_signal_band_limited,
                schema.vein_per_beat_safe.velocity_signal,
                schema.vein_per_beat_safe.velocity_signal_band_limited,
            }
            metrics.update(
                {
                    key: value
                    for key, value in velocity_outputs.items()
                    if key not in segment_paths
                }
            )

    if "velocity_profiles" in selected:
        cycle_boundaries = (
            per_beat_result.cycle_boundary_indexes
            if per_beat_result is not None
            else context.per_beat_analysis.cycle_boundary_indexes
        )
        index_base = (
            0
            if per_beat_result is not None
            else int(context.source_data.provenance["beat_index_base"])
        )
        metrics.update(
            pack_cross_section_profile_outputs(
                context.artery_segment_result,
                context.vein_segment_result,
                cycle_boundaries,
                index_base=index_base,
            )
        )
        # Displacement profile metrics are temporarily disabled.
        # metrics.update(
        #     pack_displacement_profile_outputs(
        #         context.artery_segment_result,
        #         context.vein_segment_result,
        #         cycle_boundaries,
        #         index_base=index_base,
        #     )
        # )
        metrics.update(
            pack_displacement_magnitude_outputs(
                context.artery_segment_result,
                context.vein_segment_result,
                cycle_boundaries,
                index_base=index_base,
            )
        )
        metrics.update(
            pack_cross_section_displacement_profile_outputs(
                context.artery_segment_result,
                context.vein_segment_result,
                cycle_boundaries,
                index_base=index_base,
            )
        )

    if "quadrants" in selected:
        metrics.update(
            pack_quadrant_velocity_outputs(
                velocity_outputs,
                context.source_data,
                context.artery_segment_result,
                context.vein_segment_result,
            )
        )

    return metrics


def _required_state(ctx, key: str):
    value = ctx.state.get(key)
    if value is None:
        raise RuntimeError(
            f"Required pipeline state '{key}' is unavailable; "
            "check the pipeline DAG dependencies."
        )
    return value
