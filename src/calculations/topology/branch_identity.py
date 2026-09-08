"""Identify individual vessel branches inside radial retinal regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as label_components
from skimage.morphology import disk, skeletonize
from skimage.segmentation import find_boundaries, watershed

from .geometry import SegmentRingSettings, annulus_mask


LOW_RES_SMALL_BRANCH_PIXELS = 10
BRANCH_POINT_CENTER_WEIGHT = 10
BRANCH_POINT_MIN_NEIGHBORS = 3
BRANCH_POINT_DILATION_RADIUS = 2
STREL_SIZE = BRANCH_POINT_DILATION_RADIUS
EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
BRANCH_POINT_KERNEL = np.array(
    [[1, 1, 1], [1, BRANCH_POINT_CENTER_WEIGHT, 1], [1, 1, 1]],
    dtype=np.int16,
)
BRANCH_POINT_THRESHOLD = BRANCH_POINT_CENTER_WEIGHT + BRANCH_POINT_MIN_NEIGHBORS


@dataclass(frozen=True)
class BranchIdentityStages:
    """Intermediate arrays retained for branch-label diagnostics."""

    vessel: np.ndarray
    section: np.ndarray
    skeleton: np.ndarray
    branch_points: np.ndarray
    cleaned_skeleton: np.ndarray
    marker_labels: np.ndarray
    distance_topography: np.ndarray
    imposed_minima_topography: np.ndarray
    watershed_labels: np.ndarray
    annulus_refined_labels: np.ndarray
    per_circle_cleaned_labels: np.ndarray


@dataclass(frozen=True)
class BranchIdentityResult:
    """Branch labels and diagnostic stages produced by the same run."""

    labels: np.ndarray
    branch_ids: np.ndarray
    stages: BranchIdentityStages


def label_vessel_branches(
    vessel_mask,
    optic_disc_center,
    settings: SegmentRingSettings,
    *,
    small_branch_pixels: int = LOW_RES_SMALL_BRANCH_PIXELS,
    strel_size: int = STREL_SIZE,
) -> BranchIdentityResult:
    """Label continuous vessel branches across the configured radial region."""

    vessel = np.asarray(vessel_mask, dtype=bool)
    if vessel.ndim != 2:
        raise ValueError("vessel_mask must be a 2-D array.")

    stages = _branch_identity_stages(
        vessel,
        optic_disc_center,
        settings,
        small_branch_pixels=small_branch_pixels,
        strel_size=strel_size,
    )
    labels = stages.per_circle_cleaned_labels
    return BranchIdentityResult(
        labels.astype(np.int32, copy=False),
        np.arange(1, int(labels.max()) + 1, dtype=np.int32),
        stages,
    )


def _branch_identity_stages(
    vessel: np.ndarray,
    optic_disc_center,
    settings: SegmentRingSettings,
    *,
    small_branch_pixels: int = LOW_RES_SMALL_BRANCH_PIXELS,
    strel_size: int = STREL_SIZE,
) -> BranchIdentityStages:
    section = annulus_mask(
        vessel.shape,
        optic_disc_center,
        settings.inner_radius_frac,
        settings.outer_radius_frac,
    )
    skeleton = skeletonize(vessel)

    branch_points = _branch_points(skeleton, min_arm_pixels=small_branch_pixels)
    branch_footprint = disk(max(0, int(strel_size))).astype(bool, copy=False)
    cleaned_skeleton = skeleton & ~ndi.binary_dilation(
        branch_points,
        structure=branch_footprint,
    )
    cleaned_skeleton = _remove_small(cleaned_skeleton, max(0, int(small_branch_pixels)))
    marker_labels = label_components(cleaned_skeleton, connectivity=2).astype(
        np.int32,
        copy=False,
    )

    distance_topography = -ndi.distance_transform_edt(vessel).astype(np.float32, copy=False)
    distance_topography[~(vessel & section)] = -np.inf
    imposed_topography = _impose_marker_minima(distance_topography, marker_labels > 0)

    watershed_mask = vessel & np.isfinite(imposed_topography)
    markers = (marker_labels * watershed_mask).astype(np.int32, copy=False)
    if int(markers.max()) == 0:
        watershed_labels = np.zeros(vessel.shape, dtype=np.int32)
    else:
        watershed_labels = watershed(
            imposed_topography,
            markers=markers,
            mask=watershed_mask,
            watershed_line=True,
        ).astype(np.int32, copy=False)
        watershed_labels[find_boundaries(watershed_labels, mode="inner")] = 0

    annulus_refined = label_components((watershed_labels > 0) & section, connectivity=2)
    annulus_refined = annulus_refined.astype(np.int32, copy=False)
    cleaned_labels = _per_circle_cleaned_labels(
        annulus_refined,
        optic_disc_center,
        settings,
    )
    return BranchIdentityStages(
        vessel,
        section,
        skeleton,
        branch_points,
        cleaned_skeleton,
        marker_labels,
        distance_topography,
        imposed_topography,
        watershed_labels,
        annulus_refined,
        cleaned_labels,
    )


def _branch_points(skeleton: np.ndarray, min_arm_pixels: int = 1) -> np.ndarray:
    skel = np.asarray(skeleton, dtype=bool)
    scores = ndi.convolve(skel.astype(np.int16), BRANCH_POINT_KERNEL, mode="constant")
    candidates = skel & (scores >= BRANCH_POINT_THRESHOLD)
    arm_min = max(1, int(min_arm_pixels))
    if arm_min <= 1 or not np.any(candidates):
        return candidates

    labeled, count = ndi.label(candidates, structure=EIGHT_CONNECTED)
    branch_points = np.zeros(candidates.shape, dtype=bool)
    for cluster_id in range(1, count + 1):
        cluster = labeled == cluster_id
        if _substantial_arm_count(skel, cluster, arm_min) >= BRANCH_POINT_MIN_NEIGHBORS:
            branch_points |= cluster
    return branch_points


def _substantial_arm_count(
    skeleton: np.ndarray,
    branch_point_cluster: np.ndarray,
    min_arm_pixels: int,
) -> int:
    cut_skeleton = skeleton & ~branch_point_cluster
    labeled, count = ndi.label(cut_skeleton, structure=EIGHT_CONNECTED)
    if count == 0:
        return 0
    neighborhood = ndi.binary_dilation(branch_point_cluster, structure=EIGHT_CONNECTED)
    touching_ids = np.unique(labeled[neighborhood & cut_skeleton])
    touching_ids = touching_ids[touching_ids > 0]
    if touching_ids.size == 0:
        return 0
    sizes = np.bincount(labeled.reshape(-1))
    return int(np.count_nonzero(sizes[touching_ids] >= int(min_arm_pixels)))


def _impose_marker_minima(
    topography: np.ndarray,
    marker_mask: np.ndarray,
) -> np.ndarray:
    imposed = np.asarray(topography, dtype=np.float32).copy()
    finite = np.isfinite(imposed)
    valid_markers = np.asarray(marker_mask, dtype=bool) & finite
    if np.any(valid_markers):
        floor = np.nanmin(imposed[finite])
        imposed[valid_markers] = np.nextafter(np.float32(floor), np.float32(-np.inf))
    return imposed


def _per_circle_cleaned_labels(
    labels: np.ndarray,
    optic_disc_center,
    settings: SegmentRingSettings,
) -> np.ndarray:
    min_area = max(1, labels.shape[0] // 10)
    section = annulus_mask(
        labels.shape,
        optic_disc_center,
        settings.inner_radius_frac,
        settings.outer_radius_frac,
    )
    cleaned = _remove_small((labels > 0) & section, min_area)
    return label_components(cleaned, connectivity=2).astype(np.int32, copy=False)


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    labeled, _ = ndi.label(mask, structure=EIGHT_CONNECTED)
    sizes = np.bincount(labeled.reshape(-1))
    keep = sizes >= int(min_area)
    keep[0] = False
    return keep[labeled]
