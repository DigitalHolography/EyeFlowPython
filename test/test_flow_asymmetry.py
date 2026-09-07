"""Numerical checks for the article's paired-profile asymmetry metrics."""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calculations.blood_flow_velocity.cross_section.flow_asymmetry import (  # noqa: E402
    calculate_flow_asymmetry,
    hierarchical_asymmetry_median,
    paired_profile_sums,
    summarize_flow_asymmetry,
)
from calculations.blood_flow_velocity.cross_section.profile_processing import (  # noqa: E402
    interpolate_velocity_profiles_per_beat,
    process_velocity_profiles,
)


class PairedProfileAsymmetryTests(unittest.TestCase):
    def test_ratio_of_sums_excludes_center_and_preserves_sign(self) -> None:
        # Pair sums: va=(3-1)/2+(9-1)/2=5; vs=(3+1)/2+(9+1)/2=7.
        # The mean of pointwise ratios would instead be (0.5+0.8+NaN)/3.
        profiles = np.array([[[[0, 1, 1, np.nan, 3, 9, 0]]]], dtype=float)
        x = np.arange(-3, 4, dtype=float)[None, None, :]
        antisymmetric, symmetric = paired_profile_sums(profiles, x)
        np.testing.assert_allclose(antisymmetric, 5)
        np.testing.assert_allclose(symmetric, 7)
        flipped, unchanged = paired_profile_sums(profiles[..., ::-1], x)
        np.testing.assert_allclose(flipped, -5)
        np.testing.assert_allclose(unchanged, 7)

        # Negative velocities must not be rectified or the ratio clipped.
        signed = np.array([[[[3.0, -1.0]]]])
        antisymmetric, symmetric = paired_profile_sums(signed, x[..., [0, -1]])
        np.testing.assert_allclose(antisymmetric / symmetric, -2)

    def test_even_grid_and_missing_pairs(self) -> None:
        x = np.linspace(-2, 2, 4)[None, None, :]
        profiles = np.array([[[[0, 1, 3, 0], [0, np.nan, 3, 0]]]])
        antisymmetric, symmetric = paired_profile_sums(profiles, x)
        np.testing.assert_allclose(antisymmetric[0, 0, 0], 1)
        np.testing.assert_allclose(symmetric[0, 0, 0], 2)
        self.assertTrue(np.isnan(antisymmetric[0, 0, 1]))
        self.assertTrue(np.isnan(symmetric[0, 0, 1]))
        invalid_fit = np.full_like(x, np.nan)
        self.assertTrue(np.isnan(paired_profile_sums(profiles, invalid_fit)[0]).all())

    def test_rejects_uncentered_or_nonuniform_grids(self) -> None:
        profiles = np.ones((1, 1, 3, 4))
        for grid in ([0, 1, 2, 3], [-3, -2, 2, 3], [-3, -1, np.nan, 3]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                paired_profile_sums(profiles, np.asarray(grid)[None, None, :])

    def test_beat_result_matches_ratio_after_full_profile_interpolation(self) -> None:
        rng = np.random.default_rng(731)
        profiles = rng.uniform(0.1, 10, (3, 2, 7, 6)).astype(np.float32)
        x = np.broadcast_to(np.linspace(-10, 10, 6), (3, 2, 6))
        boundaries = [0, 2, 6]
        result = calculate_flow_asymmetry(profiles, x, boundaries)
        full = interpolate_velocity_profiles_per_beat(profiles, boundaries)
        positive = full[3:]
        negative = full[:3][::-1]
        expected = np.sum(positive - negative, axis=0) / np.sum(
            positive + negative, axis=0
        )
        self.assertEqual((4, 2, 2, 3), result.asymmetry.shape)
        np.testing.assert_allclose(result.asymmetry, expected, rtol=2e-6, atol=2e-7)
        one_based = calculate_flow_asymmetry(profiles, x, [1, 3, 7], index_base=1)
        np.testing.assert_array_equal(result.asymmetry, one_based.asymmetry)

        reflected = calculate_flow_asymmetry(profiles[..., ::-1], x, boundaries)
        np.testing.assert_allclose(reflected.asymmetry, -result.asymmetry, atol=1e-7)
        self.assertAlmostEqual(result.ffa, reflected.ffa)
        self.assertAlmostEqual(result.pfa, reflected.pfa)

    def test_off_center_parabola_becomes_symmetric_inside_fitted_lumen(self) -> None:
        x = np.arange(31, dtype=float) - 15
        profile = 9 - (x - 3) ** 2 / 4
        profiles = np.broadcast_to(profile, (1, 1, 7, 31)).copy()
        centered = process_velocity_profiles(
            profiles, pixel_size_mm=0.001, velocity_profile_threshold=0.5
        )
        np.testing.assert_allclose(centered.lumen_edges_micrometers, [[[-3, 9]]])
        result = calculate_flow_asymmetry(
            centered.centered_velocity, centered.centered_x_micrometers, [0, 6]
        )
        np.testing.assert_allclose(result.asymmetry, 0, atol=1e-7)

    def test_zero_symmetric_component_is_undefined(self) -> None:
        profiles = np.broadcast_to([-1, 1], (1, 1, 7, 2))
        result = calculate_flow_asymmetry(profiles, np.array([[[-1, 1]]]), [0, 6])
        self.assertTrue(np.isnan(result.asymmetry).all())
        self.assertTrue(np.isnan(result.ffa))
        np.testing.assert_array_equal(result.temporal_sample_count, 0)


class TemporalAsymmetryTests(unittest.TestCase):
    def test_constant_asymmetry_is_persistent_with_no_dynamic_power(self) -> None:
        result = summarize_flow_asymmetry(np.full((9, 2, 3, 4), -0.375))
        np.testing.assert_array_equal(result.temporal_mean, -0.375)
        np.testing.assert_array_equal(result.magnitude, 0.375)
        np.testing.assert_array_equal(result.rms, 0.375)
        np.testing.assert_array_equal(result.dynamic_power, 0)
        np.testing.assert_array_equal(result.dynamic_rms, 0)
        self.assertEqual(0.375, result.pfa)
        self.assertEqual(0, result.ffa)
        self.assertTrue(np.isnan(result.ffar))

    def test_windows_use_full_beat_mean_and_population_rms_identity(self) -> None:
        values = np.array([0, 0, 0.3, 0.3, 0.1, 0.1])[:, None, None, None]
        result = summarize_flow_asymmetry(values)
        np.testing.assert_allclose(result.temporal_mean, 2 / 15)
        np.testing.assert_allclose(result.early_dynamic_rms, 2 / 15)
        np.testing.assert_allclose(result.late_dynamic_rms, 1 / 30)
        np.testing.assert_allclose(result.redistribution_ratio, 4)
        np.testing.assert_allclose(result.ffar, 4)
        np.testing.assert_allclose(
            result.rms**2, result.temporal_mean**2 + result.dynamic_rms**2
        )
        np.testing.assert_array_equal(result.early_sample_count, 2)
        np.testing.assert_array_equal(result.late_sample_count, 2)

    def test_nondivisible_and_short_beats_have_explicit_window_counts(self) -> None:
        result = summarize_flow_asymmetry(np.arange(8)[:, None, None, None])
        np.testing.assert_array_equal(result.early_sample_count, 2)
        np.testing.assert_array_equal(result.late_sample_count, 2)
        np.testing.assert_allclose(result.redistribution_ratio, 1)
        short = summarize_flow_asymmetry(np.ones((2, 1, 1, 1)))
        np.testing.assert_array_equal(short.early_sample_count, 0)
        np.testing.assert_array_equal(short.late_sample_count, 0)
        self.assertTrue(np.isnan(short.ffar))

    def test_invalid_samples_are_counted_consistently_and_identity_holds(self) -> None:
        values = np.array([np.nan, 0, 0.3, np.inf, 0.1, 0.1])[:, None, None, None]
        result = summarize_flow_asymmetry(values)
        np.testing.assert_array_equal(result.temporal_sample_count, 4)
        np.testing.assert_array_equal(result.early_sample_count, 1)
        np.testing.assert_array_equal(result.late_sample_count, 2)
        np.testing.assert_allclose(result.temporal_mean, 0.125)
        np.testing.assert_allclose(
            result.rms**2, result.temporal_mean**2 + result.dynamic_rms**2
        )

    def test_all_invalid_and_empty_branches_remain_nan_without_warnings(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for values in (np.full((6, 2, 3, 4), np.nan), np.empty((6, 2, 0, 4))):
                result = summarize_flow_asymmetry(values)
                self.assertTrue(np.isnan(result.ffa))
                self.assertTrue(np.isnan(result.ffar))
                self.assertTrue(np.isnan(result.pfa))

    def test_hierarchical_medians_reduce_branch_before_beat_before_radius(self) -> None:
        # Per-beat branch medians are [2, 4, 6], giving 4. Flattening gives 5;
        # reducing beats first gives 5. Repeat the construction across radii.
        first_radius = np.array([[1, 2, 100], [3, 100, 4], [100, 5, 6]])
        values = np.stack((first_radius, first_radius + 10), axis=-1)
        self.assertEqual(9, hierarchical_asymmetry_median(values))
        self.assertNotEqual(9, float(np.median(values)))


if __name__ == "__main__":
    unittest.main()
