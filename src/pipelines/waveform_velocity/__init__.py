"""Selectable waveform velocity products."""

from pipeline_engine.imports import PipelineOption, pipeline

from .runner import run_waveform_velocity


@pipeline(
    name="waveform_velocity",
    description=(
        "Compute raw and band-limited waveform velocity with optional derived products."
    ),
    requires=["numpy", "h5py", "scipy", "skimage"],
    dag_requires=["waveform_velocity_core"],
    dag_produces=["waveform_velocity"],
    options=[
        PipelineOption(
            "segments",
            "Segments",
            "Spatial vessel segments used by regional and profile products.",
        ),
        PipelineOption(
            "segment_velocity_maps",
            "Segment velocity maps",
            (
                "Per-beat segment velocity-map datasets and first-beat "
                "artery/vein mosaic movies."
            ),
            default_enabled=False,
        ),
        PipelineOption(
            "velocity_profiles",
            "Velocity profiles",
            "Per-beat cross-section velocity profiles.",
            requires=("per_beat", "segments"),
        ),
        PipelineOption(
            "per_beat",
            "Per beat",
            "Raw and band-limited vessel velocity for each beat.",
        ),
        PipelineOption(
            "quadrants",
            "Quadrants",
            "Four-quadrant velocity and per-beat velocity aggregates.",
            requires=("per_beat",),
        ),
    ],
    input_slot="both",
)
def run(ctx):
    return run_waveform_velocity(ctx)


__all__ = ["run"]
