"""DopplerView HDF5 input schema and known analysis dataset paths."""

from __future__ import annotations

from .base import (
    H5DatasetSpec,
    H5SourceSchema,
    HoloCompanionH5Layout,
    JsonConfigValueSpec,
)

DV_CONFIG_DIR_NAME = "config"
DV_CONFIG_FILENAME = "DV_params.json"
DOPPLERVIEW_VELOCITY_ESTIMATION_SECTION = "VelocityEstimation"
DOPPLERVIEW_LOCAL_BACKGROUND_DIST_KEY = "LocalBackgroundDist"
DOPPLERVIEW_ANALYSIS_ROOT = "analysis"

DOPPLER_VIEW_SCHEMA = H5SourceSchema(
    label="DV",
    layout=HoloCompanionH5Layout(
        companion_name="DV",
        h5_folder_name="h5",
        h5_filename_template="{folder}.h5",
    ),
    config_dir_name=DV_CONFIG_DIR_NAME,
    config_filename=DV_CONFIG_FILENAME,
    datasets={
        "retinal_artery_mask": H5DatasetSpec(
            key="retinal_artery_mask",
            path="segmentation/Retina/artery_mask",
            dtype="bool",
            dims=("y", "x"),
            description="DopplerView retinal artery segmentation mask.",
        ),
        "retinal_vein_mask": H5DatasetSpec(
            key="retinal_vein_mask",
            path="segmentation/Retina/vein_mask",
            dtype="bool",
            dims=("y", "x"),
            description="DopplerView retinal vein segmentation mask.",
        ),
    },
    config_values={
        "local_background_dist": JsonConfigValueSpec(
            key="local_background_dist",
            json_key=DOPPLERVIEW_LOCAL_BACKGROUND_DIST_KEY,
            section=DOPPLERVIEW_VELOCITY_ESTIMATION_SECTION,
            default=2,
            description="Pixel radius used by DopplerView local background correction.",
        ),
    },
)

DOPPLER_VIEW_ANALYSIS_SCHEMA = H5SourceSchema(
    label="DV analysis",
    layout=DOPPLER_VIEW_SCHEMA.layout,
    datasets={
        "retinal_velocity_array": H5DatasetSpec(
            key="retinal_velocity_array",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/retinal_velocity_array",
            dtype="float",
            dims=("frame", "y", "x"),
        ),
        "retinal_artery_velocity_signal": H5DatasetSpec(
            key="retinal_artery_velocity_signal",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/retinal_artery_velocity_signal",
            dtype="float",
            dims=("frame",),
        ),
        "retinal_vein_velocity_signal": H5DatasetSpec(
            key="retinal_vein_velocity_signal",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/retinal_vein_velocity_signal",
            dtype="float",
            dims=("frame",),
        ),
        "velocity_map_avg": H5DatasetSpec(
            key="velocity_map_avg",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/velocity_map_avg",
            dtype="float",
            dims=("y", "x"),
        ),
        "fRMS_avg": H5DatasetSpec(
            key="fRMS_avg",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/fRMS_avg",
            dtype="float",
            dims=("y", "x"),
        ),
        "fRMS_bkg_avg": H5DatasetSpec(
            key="fRMS_bkg_avg",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/fRMS_bkg_avg",
            dtype="float",
            dims=("y", "x"),
        ),
        "velocitysignal_per_beat": H5DatasetSpec(
            key="velocitysignal_per_beat",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/velocitysignal_per_beat",
            dtype="float",
            dims=("beat", "sample"),
        ),
        "velocitysignal_filtered": H5DatasetSpec(
            key="velocitysignal_filtered",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/velocitysignal_filtered",
            dtype="float",
            dims=("frame",),
        ),
        "beat_indices": H5DatasetSpec(
            key="beat_indices",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/beat_indices",
            dtype="int",
            dims=("beat",),
        ),
        "time_per_beat": H5DatasetSpec(
            key="time_per_beat",
            path=f"{DOPPLERVIEW_ANALYSIS_ROOT}/time_per_beat",
            dtype="float",
            dims=("beat",),
        ),
    },
)

DV_H5_LAYOUT = DOPPLER_VIEW_SCHEMA.layout
DV_RETINAL_ARTERY_MASK_PATH = DOPPLER_VIEW_SCHEMA.dataset_path("retinal_artery_mask")
DV_RETINAL_VEIN_MASK_PATH = DOPPLER_VIEW_SCHEMA.dataset_path("retinal_vein_mask")
DOPPLERVIEW_RETINAL_VELOCITY_ARRAY_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "retinal_velocity_array"
)
DOPPLERVIEW_ARTERIAL_VELOCITY_SIGNAL_PATH = (
    DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path("retinal_artery_velocity_signal")
)
DOPPLERVIEW_VENOUS_VELOCITY_SIGNAL_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "retinal_vein_velocity_signal"
)
DOPPLERVIEW_VELOCITY_MAP_AVG_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "velocity_map_avg"
)
DOPPLERVIEW_FRMS_AVG_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path("fRMS_avg")
DOPPLERVIEW_FRMS_BKG_AVG_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "fRMS_bkg_avg"
)
DOPPLERVIEW_FILTERED_PER_BEAT_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "velocitysignal_per_beat"
)
DOPPLERVIEW_FILTERED_SIGNAL_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "velocitysignal_filtered"
)
DOPPLERVIEW_BEAT_INDICES_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "beat_indices"
)
DOPPLERVIEW_TIME_PER_BEAT_PATH = DOPPLER_VIEW_ANALYSIS_SCHEMA.dataset_path(
    "time_per_beat"
)
