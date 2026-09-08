"""Segment velocity arrays from CrossSection/generateCrossSectionSignals.m."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from calculations.compute_backend import optional_cupy_backend
from calculations.topology import SegmentRingSettings
from utils.logger import Logger

from .generate_cross_section_signals import (
    CrossSectionSignalResult,
    CrossSectionSignalSettings,
    _fixed_substack_side_pixels,
    _generate_cross_section_signals_from_geometry,
    _prepare_cross_section_geometry,
    _validate_velocity_map,
)


def segment_velocity_inputs(
    velocity_map,
    artery_mask,
    vein_mask,
    optic_disc_center,
    ring_settings: SegmentRingSettings,
    cross_section_settings: CrossSectionSignalSettings | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    artery, vein = segment_velocity_results(
        velocity_map,
        artery_mask,
        vein_mask,
        optic_disc_center,
        ring_settings,
        cross_section_settings,
    )
    return artery.velocity, vein.velocity


def segment_velocity_results(
    velocity_map,
    artery_mask,
    vein_mask,
    optic_disc_center,
    ring_settings: SegmentRingSettings,
    cross_section_settings: CrossSectionSignalSettings | None = None,
    *,
    displacement_maps: Mapping[str, object] | None = None,
    artery_displacement_maps: Mapping[str, object] | None = None,
    vein_displacement_maps: Mapping[str, object] | None = None,
    retain_displacement_maps: bool = True,
) -> tuple[CrossSectionSignalResult, CrossSectionSignalResult]:
    settings = _cross_section_settings(cross_section_settings)
    artery_vessel = np.asarray(artery_mask, dtype=bool)
    vein_vessel = np.asarray(vein_mask, dtype=bool)
    _validate_velocity_map(velocity_map, artery_vessel)
    _validate_velocity_map(velocity_map, vein_vessel)
    backend = optional_cupy_backend()
    Logger.log(
        "Cross-section compute backend: "
        + ("CuPy/CUDA" if backend is not None else "CPU with parallel segments")
        + "."
    )
    artery_geometry = _prepare_cross_section_geometry(
        artery_vessel,
        optic_disc_center,
        ring_settings,
    )
    vein_geometry = _prepare_cross_section_geometry(
        vein_vessel,
        optic_disc_center,
        ring_settings,
    )
    substack_side_pixels = _fixed_substack_side_pixels(
        (artery_geometry, vein_geometry),
        settings.submask_size_percentile_kept,
    )
    artery_maps = (
        displacement_maps
        if artery_displacement_maps is None
        else artery_displacement_maps
    )
    vein_maps = (
        displacement_maps
        if vein_displacement_maps is None
        else vein_displacement_maps
    )
    return (
        _generate_cross_section_signals_from_geometry(
            velocity_map,
            artery_geometry,
            optic_disc_center,
            ring_settings,
            settings,
            substack_side_pixels,
            displacement_maps=artery_maps,
            retain_displacement_maps=retain_displacement_maps,
        ),
        _generate_cross_section_signals_from_geometry(
            velocity_map,
            vein_geometry,
            optic_disc_center,
            ring_settings,
            settings,
            substack_side_pixels,
            displacement_maps=vein_maps,
            retain_displacement_maps=retain_displacement_maps,
        ),
    )


def _cross_section_settings(value: CrossSectionSignalSettings | None):
    if value is not None:
        return value
    return CrossSectionSignalSettings(
        hydrodynamic_diameters=True,
        velocity_profile_threshold=0.5,
        rotate_from_mask=False,
        pixel_size_mm=0.0191,
    )
