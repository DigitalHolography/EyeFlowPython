"""General-purpose topology calculations for retinal maps."""

from .branch_identity import (
    BranchIdentityResult,
    BranchIdentityStages,
    label_vessel_branches,
)
from .geometry import (
    SegmentRingSettings,
    annulus_mask,
    image_half_diagonal,
    optic_disc_center_yx,
    ring_masks,
    section_masks,
)
from .segments import SegmentTopology, build_segment_topology, extract_segments
from .transforms import (
    determine_segment_rotations,
    interpolate_segment_masks,
    interpolate_segments,
    rotate_segment_masks,
    rotate_segments,
)

__all__ = [
    "BranchIdentityResult",
    "BranchIdentityStages",
    "SegmentRingSettings",
    "SegmentTopology",
    "annulus_mask",
    "build_segment_topology",
    "extract_segments",
    "determine_segment_rotations",
    "interpolate_segment_masks",
    "interpolate_segments",
    "image_half_diagonal",
    "label_vessel_branches",
    "optic_disc_center_yx",
    "ring_masks",
    "section_masks",
    "rotate_segment_masks",
    "rotate_segments",
]
