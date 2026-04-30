"""Schema facts for `.holo` selections and their companion HD/DV HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HOLO_SUFFIX = ".holo"
HDF5_SUFFIXES = (".h5", ".hdf5")
HOLO_DATA_DIR_TEMPLATE = "{stem}"
HOLO_H5_SUBDIR = "h5"
HD_MOMENT0_PATH = "moment0"
HD_MOMENT2_PATH = "moment2"
HD_CONFIG_DIR_NAME = "json"
HD_CONFIG_FILENAME = "parameters.json"
HD_SAMPLING_FREQ_KEY = "sampling_freq"
HD_BATCH_STRIDE_KEY = "batch_stride"
DV_RETINAL_ARTERY_MASK_PATH = "segmentation/Retina/artery_mask"
DV_RETINAL_VEIN_MASK_PATH = "segmentation/Retina/vein_mask"
DV_CONFIG_DIR_NAME = "config"
DV_CONFIG_FILENAME = "DV_params.json"
DOPPLERVIEW_VELOCITY_ESTIMATION_SECTION = "VelocityEstimation"
DOPPLERVIEW_LOCAL_BACKGROUND_DIST_KEY = "LocalBackgroundDist"
DOPPLERVIEW_ANALYSIS_ROOT = "analysis"
DOPPLERVIEW_RETINAL_VELOCITY_ARRAY_PATH = (
    f"{DOPPLERVIEW_ANALYSIS_ROOT}/retinal_velocity_array"
)
DOPPLERVIEW_ARTERIAL_VELOCITY_SIGNAL_PATH = (
    f"{DOPPLERVIEW_ANALYSIS_ROOT}/retinal_artery_velocity_signal"
)
DOPPLERVIEW_VENOUS_VELOCITY_SIGNAL_PATH = (
    f"{DOPPLERVIEW_ANALYSIS_ROOT}/retinal_vein_velocity_signal"
)
DOPPLERVIEW_VELOCITY_MAP_AVG_PATH = f"{DOPPLERVIEW_ANALYSIS_ROOT}/velocity_map_avg"
DOPPLERVIEW_FRMS_AVG_PATH = f"{DOPPLERVIEW_ANALYSIS_ROOT}/fRMS_avg"
DOPPLERVIEW_FRMS_BKG_AVG_PATH = f"{DOPPLERVIEW_ANALYSIS_ROOT}/fRMS_bkg_avg"
DOPPLERVIEW_FILTERED_PER_BEAT_PATH = (
    f"{DOPPLERVIEW_ANALYSIS_ROOT}/velocitysignal_per_beat"
)
DOPPLERVIEW_FILTERED_SIGNAL_PATH = (
    f"{DOPPLERVIEW_ANALYSIS_ROOT}/velocitysignal_filtered"
)
DOPPLERVIEW_BEAT_INDICES_PATH = f"{DOPPLERVIEW_ANALYSIS_ROOT}/beat_indices"
DOPPLERVIEW_TIME_PER_BEAT_PATH = f"{DOPPLERVIEW_ANALYSIS_ROOT}/time_per_beat"


@dataclass(frozen=True)
class HoloCompanionH5Layout:
    label: str
    folder_suffix: str
    h5_subdir: str
    preferred_filename_template: str

    def folder_name(self, holo_stem: str) -> str:
        return f"{holo_stem}{self.folder_suffix}"

    def folder_path(self, data_dir: Path, holo_stem: str) -> Path:
        return data_dir / self.folder_name(holo_stem)

    def h5_dir(self, data_dir: Path, holo_stem: str) -> Path:
        return self.folder_path(data_dir, holo_stem) / self.h5_subdir

    def preferred_filename(self, holo_stem: str) -> str:
        folder_name = self.folder_name(holo_stem)
        return self.preferred_filename_template.format(
            stem=holo_stem,
            folder=folder_name,
        )


HD_H5_LAYOUT = HoloCompanionH5Layout(
    label="HD",
    folder_suffix="_HD",
    h5_subdir=HOLO_H5_SUBDIR,
    preferred_filename_template="{folder}_output.h5",
)
DV_H5_LAYOUT = HoloCompanionH5Layout(
    label="DV",
    folder_suffix="_DV",
    h5_subdir=HOLO_H5_SUBDIR,
    preferred_filename_template="{folder}.h5",
)
HOLO_COMPANION_H5_LAYOUTS = (HD_H5_LAYOUT, DV_H5_LAYOUT)
