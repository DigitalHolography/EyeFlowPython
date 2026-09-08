from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calculations.topology import (  # noqa: E402
    SegmentRingSettings,
    SegmentTopology,
    build_segment_topology,
    extract_segments,
)


class SegmentTopologyTests(unittest.TestCase):
    def test_builds_map_independent_segment_windows(self) -> None:
        vessel = np.zeros((41, 41), dtype=bool)
        vessel[18:23, 5:36] = True
        optic_disc = np.zeros_like(vessel)
        optic_disc[18:23, 18:23] = True

        topology = build_segment_topology(
            vessel,
            optic_disc,
            SegmentRingSettings(0.1, 0.6, 0.25, 2),
            window_size_percentile_kept=1.0,
        )

        self.assertEqual((20.0, 20.0), topology.optic_disc_center_xy)
        self.assertEqual((2, 2, 2), topology.segment_centers_xy.shape)
        self.assertEqual((2, 2, 7, 7), topology.segment_masks.shape)
        self.assertEqual((41, 41), topology.centerline.shape)
        self.assertEqual(np.bool_, topology.centerline.dtype)
        self.assertEqual(7, topology.window_side_pixels)
        self.assertTrue(np.all(topology.valid_segments))

    def test_extracts_vector_maps_with_explicit_spatial_axes(self) -> None:
        topology = _edge_topology()
        vector_map = np.arange(2 * 4 * 5 * 2, dtype=np.float32).reshape(2, 4, 5, 2)

        segments = extract_segments(vector_map, topology, spatial_axes=(1, 2))

        self.assertEqual((1, 1, 2, 2, 3, 3), segments.shape)
        self.assertTrue(np.all(np.isnan(segments[0, 0, :, :, 0, :])))
        self.assertTrue(np.all(np.isnan(segments[0, 0, :, :, :, 0])))
        np.testing.assert_array_equal(
            segments[0, 0, :, :, 1:, 1:],
            np.moveaxis(vector_map[:, :2, :2, :], (1, 2), (-2, -1)),
        )

    def test_extracts_scalar_stacks_without_topology_recalculation(self) -> None:
        topology = _edge_topology()
        scalar_stack = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)

        first = extract_segments(scalar_stack, topology)
        second = extract_segments(scalar_stack + 100.0, topology)

        self.assertEqual((1, 1, 2, 3, 3), first.shape)
        np.testing.assert_array_equal(
            second[..., 1:, 1:],
            first[..., 1:, 1:] + 100.0,
        )


def _edge_topology() -> SegmentTopology:
    return SegmentTopology(
        spatial_shape=(4, 5),
        optic_disc_center_xy=(2.0, 2.0),
        labels=np.ones((4, 5), dtype=np.int32),
        centerline=np.ones((4, 5), dtype=bool),
        branch_ids=np.asarray([1], dtype=np.int32),
        annulus_masks=np.ones((1, 4, 5), dtype=bool),
        segment_masks=np.ones((1, 1, 3, 3), dtype=bool),
        segment_centers_xy=np.asarray([[[0.0, 0.0]]], dtype=np.float32),
        window_bounds_xyxy=np.asarray([[[0, 2, 0, 2]]], dtype=np.int32),
        window_side_pixels=3,
    )


if __name__ == "__main__":
    unittest.main()
