"""Tests for map-independent topology interpolation and rotation."""

from __future__ import annotations

import unittest

import numpy as np

from calculations.topology.geometry import annulus_mask
from calculations.topology.segments import SegmentTopology
from calculations.topology.transforms import (
    determine_segment_rotations,
    interpolate_segment_masks,
    interpolate_segments,
    rotate_segment_masks,
    rotate_segments,
)


class TestTopologyTransforms(unittest.TestCase):
    def test_rotation_uses_a_truncated_multiline_centerline(self) -> None:
        shape = (21, 21)
        center_xy = (3.0, 10.0)
        labels = np.zeros(shape, dtype=np.int32)
        labels[8:13, 8:16] = 1
        centerline = np.zeros(shape, dtype=bool)
        centerline[10, 8:11] = True
        centerline[10, 13:16] = True
        centerline[9, 10:14] = True
        centerline[11, 10:14] = True
        centerline[9:12, 10] = True
        centerline[9:12, 13] = True
        annuli = annulus_mask(shape, center_xy, 0.1, 0.95)[None, ...]
        topology = SegmentTopology(
            spatial_shape=shape,
            optic_disc_center_xy=center_xy,
            labels=labels,
            centerline=centerline,
            branch_ids=np.asarray([1], dtype=np.int32),
            annulus_masks=annuli,
            segment_masks=np.ones((1, 1, 9, 9), dtype=bool),
            segment_centers_xy=np.asarray([[[12.0, 10.0]]], dtype=np.float32),
            window_bounds_xyxy=np.asarray([[[8, 17, 6, 15]]], dtype=np.int32),
            window_side_pixels=9,
        )

        rotations = determine_segment_rotations(topology)

        np.testing.assert_allclose(rotations, [[90.0]])

    def test_values_and_masks_use_separate_interpolation(self) -> None:
        values = np.full((1, 1, 2, 2), np.float32(7.0))
        values[..., 0, 0] = np.nan
        masks = np.asarray([[[[False, True], [True, True]]]], dtype=bool)

        interpolated_values = interpolate_segments(values, output_side_pixels=5)
        interpolated_masks = interpolate_segment_masks(
            masks,
            output_side_pixels=5,
        )

        self.assertEqual((1, 1, 5, 5), interpolated_values.shape)
        self.assertEqual(np.float32, interpolated_values.dtype)
        self.assertEqual((1, 1, 5, 5), interpolated_masks.shape)
        self.assertEqual(np.bool_, interpolated_masks.dtype)
        np.testing.assert_allclose(
            interpolated_values[np.isfinite(interpolated_values)],
            7.0,
        )

    def test_rotation_occurs_after_interpolation_on_a_larger_canvas(self) -> None:
        values = np.arange(18, dtype=np.float32).reshape(1, 1, 2, 3, 3)
        masks = np.ones((1, 1, 3, 3), dtype=bool)
        rotations = np.asarray([[0.0]], dtype=np.float32)

        interpolated_values = interpolate_segments(values)
        interpolated_masks = interpolate_segment_masks(masks)
        rotated_values = rotate_segments(interpolated_values, rotations)
        rotated_masks = rotate_segment_masks(interpolated_masks, rotations)

        self.assertEqual((1, 1, 2, 128, 128), interpolated_values.shape)
        self.assertEqual((1, 1, 128, 128), interpolated_masks.shape)
        self.assertEqual((1, 1, 2, 181, 181), rotated_values.shape)
        self.assertEqual((1, 1, 181, 181), rotated_masks.shape)
        np.testing.assert_allclose(
            rotated_values[..., 26:154, 26:154],
            interpolated_values,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            rotated_masks[..., 26:154, 26:154],
            interpolated_masks,
        )

    def test_invalid_rotation_keeps_segment_outputs_empty(self) -> None:
        values = np.ones((1, 1, 4, 4), dtype=np.float32)
        masks = np.ones((1, 1, 4, 4), dtype=bool)
        rotations = np.asarray([[np.nan]], dtype=np.float32)

        rotated_values = rotate_segments(values, rotations)
        rotated_masks = rotate_segment_masks(masks, rotations)

        self.assertTrue(np.all(np.isnan(rotated_values)))
        self.assertFalse(np.any(rotated_masks))


if __name__ == "__main__":
    unittest.main()
