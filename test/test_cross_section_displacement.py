from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calculations.blood_flow_velocity.cross_section import (  # noqa: E402
    CrossSectionSignalSettings,
    CrossSectionTopology,
    generate_cross_section_signals,
)
from calculations.topology import SegmentRingSettings  # noqa: E402
from calculations.blood_flow_velocity.cross_section.generate_cross_section_signals import (  # noqa: E402
    _correct_displacement_basis,
    _cross_sectional_radial_metrics,
    _project_displacement_map,
)


class CrossSectionDisplacementTests(unittest.TestCase):
    def test_radial_metrics_use_mask_geometry_but_sample_outside_it(self) -> None:
        vessel_mask = np.zeros((5, 7), dtype=bool)
        vessel_mask[1:4, 2:5] = True
        vectors = np.zeros((1, 5, 7, 2), dtype=np.float32)
        vectors[:, 1:4, 0:2, 0] = -2.0
        vectors[:, 1:4, 5:7, 0] = 4.0
        vectors[..., 1] = 100.0

        amplitude, asymmetry = _cross_sectional_radial_metrics(
            vectors,
            vessel_mask,
        )

        np.testing.assert_allclose(amplitude, [2.0], atol=1e-6)
        np.testing.assert_allclose(asymmetry, [-1.0 / 3.0], atol=1e-6)

    def test_basis_correction_keeps_signed_local_y(self) -> None:
        dx = np.full((2, 3, 4), 2.0, dtype=np.float32)
        dy = np.full_like(dx, 3.0)

        corrected = _correct_displacement_basis(dx, dy, 90.0)

        np.testing.assert_allclose(corrected[..., 0], 3.0, atol=1e-6)
        np.testing.assert_allclose(corrected[..., 1], -2.0, atol=1e-6)

    def test_projection_keeps_and_sums_the_full_unmasked_local_xy_map(
        self,
    ) -> None:
        topology = _single_segment_topology(frame_count=2, angle=90.0)
        displacement = np.empty((2, 5, 5, 2), dtype=np.float32)
        displacement[..., 0] = 1.0
        displacement[..., 1] = 2.0

        with patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._resize_subimage_stack',
            side_effect=_constant_resize,
        ), patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._rotate_stack_with_nan',
            side_effect=lambda values, _angle: values,
        ):
            result = _project_displacement_map(displacement, topology)

        self.assertEqual((1, 1, 2), result.displacement.shape)
        self.assertEqual(
            (1, 1, 2, 181, 181, 2),
            result.displacement_maps_per_segment.shape,
        )
        np.testing.assert_allclose(result.displacement, -1.0, atol=1e-6)
        np.testing.assert_allclose(result.safe_displacement, -1.0, atol=1e-6)
        vectors = result.displacement_maps_per_segment[0, 0]
        resized_window = np.zeros(vectors.shape[1:3], dtype=bool)
        resized_window[26:154, 26:154] = True
        outside_segment = resized_window & ~topology.segment_masks[0, 0]
        self.assertTrue(np.all(np.isfinite(vectors[:, outside_segment, :])))
        self.assertTrue(np.all(np.isnan(vectors[:, ~resized_window, :])))
        np.testing.assert_allclose(
            vectors[:, resized_window, 0],
            2.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            vectors[:, resized_window, 1],
            -1.0,
            atol=1e-6,
        )
        expected_magnitude = np.hypot(2.0, -1.0)
        transverse_unmasked = (
            result.transverse_displacement_profiles_unmasked[0, 0]
        )
        transverse_masked = result.transverse_displacement_profiles_masked[0, 0]
        longitudinal_unmasked = (
            result.longitudinal_displacement_profiles_unmasked[0, 0]
        )
        longitudinal_masked = result.longitudinal_displacement_profiles_masked[0, 0]
        self.assertEqual((2, 181), transverse_unmasked.shape)
        self.assertEqual((2, 181), transverse_masked.shape)
        self.assertEqual((2, 181), longitudinal_unmasked.shape)
        self.assertEqual((2, 181), longitudinal_masked.shape)
        np.testing.assert_array_equal(
            np.flatnonzero(np.isfinite(transverse_unmasked[0])),
            np.arange(26, 154),
        )
        np.testing.assert_array_equal(
            np.flatnonzero(np.isfinite(longitudinal_unmasked[0])),
            np.arange(26, 154),
        )
        np.testing.assert_array_equal(
            np.flatnonzero(np.isfinite(transverse_masked[0])),
            np.arange(40, 140),
        )
        np.testing.assert_array_equal(
            np.flatnonzero(np.isfinite(longitudinal_masked[0])),
            np.arange(30, 150),
        )
        np.testing.assert_allclose(
            transverse_masked[:, 40:140],
            expected_magnitude,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            longitudinal_masked[:, 30:150],
            expected_magnitude,
            atol=1e-6,
        )
        pixel_count = np.count_nonzero(resized_window)
        self.assertEqual(
            (1, 1, 2),
            result.x_sum_displacement_profile.shape,
        )
        self.assertEqual(
            (1, 1, 2),
            result.y_sum_displacement_profile.shape,
        )
        np.testing.assert_allclose(
            result.x_sum_displacement_profile,
            2.0 * pixel_count,
        )
        np.testing.assert_allclose(
            result.y_sum_displacement_profile,
            -1.0 * pixel_count,
        )
        np.testing.assert_allclose(
            result.cross_sectional_radial_movement_amplitude,
            2.0,
        )
        np.testing.assert_allclose(
            result.cross_sectional_radial_asymmetry_index,
            0.0,
            atol=1e-6,
        )

    def test_sum_profiles_preserve_signed_components_per_frame(self) -> None:
        topology = _single_segment_topology(frame_count=2, angle=0.0)
        displacement = np.empty((2, 5, 5, 2), dtype=np.float32)
        displacement[..., 0] = np.asarray([3.0, 4.0])[:, None, None]
        displacement[..., 1] = np.asarray([-5.0, -6.0])[:, None, None]

        with patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._resize_subimage_stack',
            side_effect=_constant_resize,
        ), patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._rotate_stack_with_nan',
            side_effect=lambda values, _angle: values,
        ):
            result = _project_displacement_map(displacement, topology)

        pixel_count = 128 * 128
        np.testing.assert_allclose(
            result.x_sum_displacement_profile[0, 0],
            np.asarray([3.0, 4.0]) * pixel_count,
        )
        np.testing.assert_allclose(
            result.y_sum_displacement_profile[0, 0],
            np.asarray([-5.0, -6.0]) * pixel_count,
        )

    def test_lazy_displacement_reads_one_bounded_vector_slice(self) -> None:
        topology = _single_segment_topology(frame_count=2, angle=90.0)
        displacement = np.empty((2, 5, 5, 2), dtype=np.float32)
        displacement[..., 0] = 1.0
        displacement[..., 1] = 2.0
        lazy_displacement = _RecordingLazyArray(displacement)

        with patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._resize_subimage_stack',
            side_effect=_constant_resize,
        ), patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._rotate_stack_with_nan',
            side_effect=lambda values, _angle: values,
        ):
            result = _project_displacement_map(lazy_displacement, topology)

        self.assertEqual(
            [
                (
                    slice(None),
                    slice(1, 4),
                    slice(1, 4),
                    slice(None),
                ),
            ],
            lazy_displacement.reads,
        )
        np.testing.assert_allclose(result.displacement, -1.0, atol=1e-6)

    def test_zero_branches_support_optional_multiple_methods(self) -> None:
        velocity_map = np.zeros((2, 9, 9), dtype=np.float32)
        vessel_mask = np.zeros((9, 9), dtype=bool)
        displacement = np.zeros((2, 9, 9, 2), dtype=np.float32)

        result = generate_cross_section_signals(
            velocity_map,
            vessel_mask,
            (4, 4),
            SegmentRingSettings(0.0, 0.5, 0.5, 1),
            CrossSectionSignalSettings(False, 0.5, False, 0.01),
            displacement_maps={'method_a': displacement, 'method_b': displacement},
        )

        self.assertEqual((1, 0, 2), result.velocity.shape)
        self.assertEqual((1, 0), result.topology.valid_segments.shape)
        self.assertEqual({'method_a', 'method_b'}, set(result.displacements))
        for displacement_result in result.displacements.values():
            self.assertEqual((1, 0, 2), displacement_result.displacement.shape)
            self.assertEqual(
                (1, 0, 2, 181, 181, 2),
                displacement_result.displacement_maps_per_segment.shape,
            )
            self.assertEqual(
                (1, 0, 2),
                displacement_result.x_sum_displacement_profile.shape,
            )
            self.assertEqual(
                (1, 0, 2),
                displacement_result.y_sum_displacement_profile.shape,
            )
            self.assertEqual(
                (1, 0, 2, 181),
                displacement_result.transverse_displacement_profiles_unmasked.shape,
            )
            self.assertEqual(
                (1, 0, 2, 181),
                displacement_result.transverse_displacement_profiles_masked.shape,
            )
            self.assertEqual(
                (1, 0, 2, 181),
                displacement_result.longitudinal_displacement_profiles_unmasked.shape,
            )
            self.assertEqual(
                (1, 0, 2, 181),
                displacement_result.longitudinal_displacement_profiles_masked.shape,
            )
            self.assertEqual(
                (1, 0, 2),
                displacement_result.cross_sectional_radial_movement_amplitude.shape,
            )
            self.assertEqual(
                (1, 0, 2),
                displacement_result.cross_sectional_radial_asymmetry_index.shape,
            )

    def test_multiple_methods_reuse_one_velocity_fitted_topology(self) -> None:
        vessel_mask = np.zeros((9, 9), dtype=bool)
        vessel_mask[2:7, 3:6] = True
        branches = SimpleNamespace(
            labels=vessel_mask.astype(np.int32),
            branch_ids=np.asarray([1], dtype=np.int32),
            stages=SimpleNamespace(),
        )
        velocity_map = np.ones((2, 9, 9), dtype=np.float32)
        first = np.empty((2, 9, 9, 2), dtype=np.float32)
        first[..., 0] = 1.0
        first[..., 1] = 2.0
        second = np.empty_like(first)
        second[..., 0] = 3.0
        second[..., 1] = 4.0

        with patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals.label_vessel_branches',
            return_value=branches,
        ), patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals.section_masks',
            return_value=vessel_mask[None, ...],
        ), patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._estimate_orientation',
            return_value=90.0,
        ) as estimate_orientation, patch(
            'calculations.blood_flow_velocity.cross_section.'
            'generate_cross_section_signals._cross_section_limits',
            return_value=(0, 180),
        ) as cross_section_limits:
            result = generate_cross_section_signals(
                velocity_map,
                vessel_mask,
                (4, 4),
                SegmentRingSettings(0.0, 0.5, 0.5, 1),
                CrossSectionSignalSettings(False, 0.5, False, 0.01),
                displacement_maps={'first': first, 'second': second},
            )

        self.assertEqual(1, estimate_orientation.call_count)
        self.assertEqual(1, cross_section_limits.call_count)
        self.assertEqual({'first', 'second'}, set(result.displacements))
        np.testing.assert_array_equal(result.topology.profile_rotation_degrees, [[90.0]])
        np.testing.assert_allclose(result.displacements['first'].displacement, -1.0)
        np.testing.assert_allclose(result.displacements['second'].displacement, -3.0)
        rotated_mask = result.segment_masks[0, 0]
        for displacement_result in result.displacements.values():
            self.assertTrue(
                np.any(
                    np.isfinite(
                        displacement_result.displacement_maps_per_segment[
                            0, 0, :, ~rotated_mask, :
                        ]
                    )
                )
            )

    def test_velocity_only_and_displacement_shape_validation(self) -> None:
        velocity_map = np.zeros((2, 9, 9), dtype=np.float32)
        vessel_mask = np.zeros((9, 9), dtype=bool)
        settings = SegmentRingSettings(0.0, 0.5, 0.5, 1)
        cross_section_settings = CrossSectionSignalSettings(
            False,
            0.5,
            False,
            0.01,
        )

        result = generate_cross_section_signals(
            velocity_map,
            vessel_mask,
            (4, 4),
            settings,
            cross_section_settings,
        )
        self.assertEqual({}, result.displacements)

        with self.assertRaisesRegex(ValueError, 'must have shape'):
            generate_cross_section_signals(
                velocity_map,
                vessel_mask,
                (4, 4),
                settings,
                cross_section_settings,
                displacement_maps={
                    'bad': np.zeros((2, 9, 9, 3), dtype=np.float32)
                },
            )


def _single_segment_topology(
    *,
    frame_count: int,
    angle: float,
) -> CrossSectionTopology:
    rotated_mask = np.zeros((1, 1, 181, 181), dtype=bool)
    rotated_mask[:, :, 50:130, 60:120] = True
    return CrossSectionTopology(
        spatial_shape=(5, 5),
        frame_count=frame_count,
        labels=np.ones((5, 5), dtype=np.int32),
        branch_ids=np.asarray([1], dtype=np.int32),
        section_masks=np.ones((1, 5, 5), dtype=bool),
        segment_masks=rotated_mask,
        segment_center_xy=np.asarray([[[2.0, 2.0]]], dtype=np.float32),
        profile_window_bounds_xyxy=np.asarray([[[1, 4, 1, 4]]], dtype=np.int32),
        profile_window_side_pixels=3,
        profile_pixel_size_mm=0.01,
        profile_rotation_degrees=np.asarray([[angle]], dtype=np.float32),
        profile_integration_limits_pixels=np.asarray([[[0, 180]]], dtype=np.int32),
        valid_segments=np.asarray([[True]], dtype=bool),
        branch_identity=SimpleNamespace(),
    )


def _constant_resize(values: np.ndarray) -> np.ndarray:
    frame_values = np.nanmean(values, axis=(1, 2), dtype=np.float32)
    return np.broadcast_to(frame_values[:, None, None], (values.shape[0], 128, 128))


class _RecordingLazyArray:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.shape = values.shape
        self.reads = []

    def __getitem__(self, key):
        if (
            isinstance(key, tuple)
            and len(key) == 2
            and key[0] is Ellipsis
            and isinstance(key[1], int)
        ):
            raise AssertionError('whole displacement components must not be read')
        self.reads.append(key)
        return self._values[key]


if __name__ == '__main__':
    unittest.main()
