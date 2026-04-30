"""Shared `.holo` selection schema and companion HD/DV layouts."""

from __future__ import annotations

from .base import HoloCompanionH5Layout
from .doppler_view import DV_H5_LAYOUT
from .holodoppler import HD_H5_LAYOUT

HOLO_SUFFIX = ".holo"
HDF5_SUFFIXES = (".h5", ".hdf5")
HOLO_DATA_DIR_TEMPLATE = "{stem}"
HOLO_H5_SUBDIR = "h5"
HOLO_COMPANION_H5_LAYOUTS = (HD_H5_LAYOUT, DV_H5_LAYOUT)
