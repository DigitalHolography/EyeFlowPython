"""Holodoppler HDF5 and sidecar JSON input schema."""

from __future__ import annotations

from .base import (
    H5DatasetSpec,
    H5SourceSchema,
    HoloCompanionH5Layout,
    JsonConfigValueSpec,
)

HD_CONFIG_DIR_NAME = "json"
HD_CONFIG_FILENAME = "parameters.json"

HOLODOPPLER_SCHEMA = H5SourceSchema(
    label="HD",
    layout=HoloCompanionH5Layout(
        companion_name="HD",
        h5_folder_name="h5",
        h5_filename_template="{folder}_output.h5",
    ),
    config_dir_name=HD_CONFIG_DIR_NAME,
    config_filename=HD_CONFIG_FILENAME,
    datasets={
        "moment0": H5DatasetSpec(
            key="moment0",
            path="moment0",
            dtype="float",
            dims=("frame", "y", "x"),
            description="Holodoppler moment 0 stack.",
        ),
        "moment2": H5DatasetSpec(
            key="moment2",
            path="moment2",
            dtype="float",
            dims=("frame", "y", "x"),
            description="Holodoppler moment 2 stack.",
        ),
    },
    config_values={
        "sampling_freq": JsonConfigValueSpec(
            key="sampling_freq",
            json_key="sampling_freq",
            h5_path="sampling_freq",
            description="Camera sampling frequency in hertz.",
        ),
        "batch_stride": JsonConfigValueSpec(
            key="batch_stride",
            json_key="batch_stride",
            h5_path="batch_stride",
            description="Frame stride between exported Holodoppler batches.",
        ),
    },
)

HD_H5_LAYOUT = HOLODOPPLER_SCHEMA.layout
HD_MOMENT0_PATH = HOLODOPPLER_SCHEMA.dataset_path("moment0")
HD_MOMENT2_PATH = HOLODOPPLER_SCHEMA.dataset_path("moment2")
HD_SAMPLING_FREQ_KEY = HOLODOPPLER_SCHEMA.config_value("sampling_freq").json_key
HD_BATCH_STRIDE_KEY = HOLODOPPLER_SCHEMA.config_value("batch_stride").json_key
