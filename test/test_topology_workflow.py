"""Tests for the ordered topology workflow."""

from __future__ import annotations

import unittest

import numpy as np

from calculations.topology.geometry import SegmentRingSettings
from calculations.topology.workflow import prepare_segments, prepare_topology


class TopologyWorkflowTests(unittest.TestCase):
    def test_reuses_one_topology_for_multiple_maps(self) -> None:
        vessel = np.zeros((41, 41), dtype=bool)
        vessel[18:23, 5:36] = True
        optic_disc = np.zeros_like(vessel)
        optic_disc[18:23, 18:23] = True
        prepared = prepare_topology(
            vessel,
            optic_disc,
            SegmentRingSettings(0.1, 0.6, 0.25, 2),
            window_size_percentile_kept=1.0,
        )

        first = prepare_segments(np.ones((2, 41, 41), dtype=np.float32), prepared)
        second = prepare_segments(
            np.full((2, 41, 41), 2.0, dtype=np.float32),
            prepared,
        )

        ring_count, branch_count = prepared.topology.valid_segments.shape
        self.assertEqual(
            (ring_count, branch_count, 2, 128, 128),
            first.interpolated.shape,
        )
        self.assertEqual(
            (ring_count, branch_count, 2, 181, 181),
            first.rotated.shape,
        )
        self.assertEqual(
            (ring_count, branch_count, 128, 128),
            prepared.interpolated_masks.shape,
        )
        self.assertEqual(
            (ring_count, branch_count, 181, 181),
            prepared.rotated_masks.shape,
        )
        np.testing.assert_allclose(
            second.interpolated[np.isfinite(second.interpolated)],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
