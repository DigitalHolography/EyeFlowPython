from __future__ import annotations

import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calculations.blood_flow_velocity.cross_section.profile_processing import (  # noqa: E402
    _matlab_poiseuille_fit,
    interpolate_velocity_profiles_per_beat,
    process_velocity_profiles,
)
from calculations.blood_flow_velocity.signal_analysis.per_beat.signal import (  # noqa: E402
    per_beat_signal_analysis,
)
from calculations.math import rotate_image_with_nan  # noqa: E402
from input_output.output_manager import OutputType  # noqa: E402
from input_output.schema import EyeFlowOutputPaths  # noqa: E402
from input_output.writers.h5 import write_value_dataset  # noqa: E402
from input_output.writers.png import FigureArtifactWriter, write_png_file  # noqa: E402
from pipelines.waveform_velocity_core.figures.profiles import (  # noqa: E402
    _finite_median,
    _hierarchical_profile_median,
    _nanmedian,
    _positive_focused_limits,
    export_cross_section_profile_artifacts,
)
from pipelines.waveform_velocity.flow_asymmetry import pack_flow_asymmetry_outputs  # noqa: E402
from pipelines.waveform_velocity.profiles import (  # noqa: E402
    pack_cross_section_profile_outputs,
)


class CrossSectionProfilePackingTests(unittest.TestCase):
    def test_h5_export_contains_transverse_and_longitudinal_profiles(self) -> None:
        artery = _segments(radius_count=2, branch_count=1)
        vein = _segments(radius_count=2, branch_count=0)
        cycle_boundaries = np.asarray([0, 2, 5], dtype=np.int32)
        metrics = pack_cross_section_profile_outputs(
            artery,
            vein,
            cycle_boundaries,
        )
        schema = EyeFlowOutputPaths.active()
        self.assertEqual(8, len(metrics))
        artery_paths = schema.artery_velocity_profiles
        vein_paths = schema.vein_velocity_profiles
        self.assertEqual(
            "Processing/VelocityProfiles/Artery/"
            "TransverseVelocityProfileUnmasked/value",
            artery_paths.transverse_velocity_profile_unmasked,
        )
        self.assertEqual(
            "Processing/VelocityProfiles/Artery/"
            "LongitudinalVelocityProfileMasked/value",
            artery_paths.longitudinal_velocity_profile_masked,
        )
        self.assertEqual(
            "Processing/VelocityProfiles/Artery/"
            "LongitudinalVelocityProfileUnmasked/value",
            artery_paths.longitudinal_velocity_profile_unmasked,
        )
        self.assertEqual(
            "Processing/VelocityProfiles/Vein/"
            "LongitudinalVelocityProfileMasked/value",
            vein_paths.longitudinal_velocity_profile_masked,
        )
        self.assertEqual(
            "Processing/VelocityProfiles/Vein/"
            "LongitudinalVelocityProfileUnmasked/value",
            vein_paths.longitudinal_velocity_profile_unmasked,
        )
        self.assertEqual(
            "Processing/VelocityProfiles/Vein/"
            "TransverseVelocityProfileUnmasked/value",
            vein_paths.transverse_velocity_profile_unmasked,
        )

        with h5py.File("profiles.h5", "w", driver="core", backing_store=False) as h5:
            for path, value in metrics.items():
                write_value_dataset(h5, path, value)
            raw_dataset = h5[artery_paths.transverse_velocity_profile_unmasked]
            transverse_dataset = h5[artery_paths.transverse_velocity_profile_masked]
            longitudinal_unmasked_dataset = h5[
                artery_paths.longitudinal_velocity_profile_unmasked
            ]
            longitudinal_dataset = h5[artery_paths.longitudinal_velocity_profile_masked]
            self.assertEqual((181, 4, 2, 1, 2), raw_dataset.shape)
            self.assertEqual(raw_dataset.shape, transverse_dataset.shape)
            self.assertEqual(raw_dataset.shape, longitudinal_unmasked_dataset.shape)
            self.assertEqual(raw_dataset.shape, longitudinal_dataset.shape)
            self.assertEqual(
                list(raw_dataset.attrs["dimDesc"]),
                ["x", "time", "beat", "branch", "radius"],
            )
            self.assertEqual(
                list(transverse_dataset.attrs["dimDesc"]),
                ["x", "time", "beat", "branch", "radius"],
            )
            self.assertEqual(
                list(longitudinal_unmasked_dataset.attrs["dimDesc"]),
                ["y", "time", "beat", "branch", "radius"],
            )
            self.assertEqual(
                list(longitudinal_dataset.attrs["dimDesc"]),
                ["y", "time", "beat", "branch", "radius"],
            )
            self.assertEqual("mm/s", raw_dataset.attrs["unit"])
            self.assertEqual("gzip", raw_dataset.compression)
            self.assertEqual(4, raw_dataset.compression_opts)
            self.assertTrue(raw_dataset.shuffle)
            self.assertEqual((181, 4, 1, 1, 1), raw_dataset.chunks)
            self.assertFalse(
                np.array_equal(raw_dataset[...], transverse_dataset[...], equal_nan=True)
            )
            self.assertFalse(
                np.array_equal(
                    longitudinal_unmasked_dataset[...],
                    longitudinal_dataset[...],
                    equal_nan=True,
                )
            )
            self.assertFalse(
                np.array_equal(
                    transverse_dataset[...],
                    longitudinal_dataset[...],
                    equal_nan=True,
                )
            )
            empty_longitudinal = h5[
                vein_paths.longitudinal_velocity_profile_masked
            ]
            self.assertEqual((181, 4, 2, 0, 2), empty_longitudinal.shape)
            self.assertNotIn("Processing/CrossSections", h5)
            for vessel in ("Artery", "Vein"):
                root = f"Processing/VelocityProfiles/{vessel}"
                self.assertNotIn(f"{root}/RawProfile", h5)
                self.assertNotIn(f"{root}/FlowAsymmetry", h5)

    def test_flow_asymmetry_hdf5_values_dimensions_and_windows(self) -> None:
        segments = _segments(radius_count=2, branch_count=1)
        empty = _segments(radius_count=2, branch_count=0)
        schema = EyeFlowOutputPaths.active()
        root = schema.artery_velocity_profiles.flow_asymmetry_root
        outputs = pack_flow_asymmetry_outputs(root, segments, [0, 2, 5], index_base=0)
        outputs.update(
            pack_flow_asymmetry_outputs(
                schema.vein_velocity_profiles.flow_asymmetry_root,
                empty,
                [0, 2, 5],
                index_base=0,
            )
        )
        self.assertEqual("Processing/VelocityProfiles/Artery/FlowAsymmetry", root)
        with h5py.File("asymmetry.h5", "w", driver="core", backing_store=False) as h5:
            for path, dataset in outputs.items():
                write_value_dataset(h5, path, dataset)
            series = h5[f"{root}/A/value"]
            self.assertEqual((4, 2, 1, 2), series.shape)
            self.assertEqual(["time", "beat", "branch", "radius"], list(series.attrs["dimDesc"]))
            self.assertEqual("1", series.attrs["unit"])
            self.assertEqual("positive_x_minus_negative_x", series.attrs["side_convention"])
            self.assertEqual(1, series.attrs["early_window_stop_index_exclusive"])
            self.assertEqual(3, series.attrs["late_window_start_index"])
            self.assertEqual("full_beat_mean", series.attrs["temporal_centering"])
            mean = h5[f"{root}/A_mean/value"][...]
            power = h5[f"{root}/p_A/value"][...]
            np.testing.assert_allclose(mean, np.mean(series[...], axis=0), atol=1e-8)
            np.testing.assert_allclose(power, (series[...] - mean[None]) ** 2, atol=1e-8)
            np.testing.assert_allclose(
                h5[f"{root}/A_RMS/value"][...] ** 2,
                mean**2 + h5[f"{root}/a/value"][...] ** 2,
                atol=1e-8,
            )
            for name in ("A_mean", "A_RMS", "a", "a_early", "a_late", "R_a"):
                dataset = h5[f"{root}/{name}/value"]
                self.assertEqual((2, 1, 2), dataset.shape)
                self.assertEqual(["beat", "branch", "radius"], list(dataset.attrs["dimDesc"]))
            np.testing.assert_array_equal(h5[f"{root}/N_t/value"][...], 4)
            for name in ("FFA", "FFAR", "PFA"):
                dataset = h5[f"{root}/{name}/value"]
                self.assertEqual((), dataset.shape)
                self.assertEqual("branch_then_beat_then_radius", dataset.attrs["aggregation_order"])
                empty_root = schema.vein_velocity_profiles.flow_asymmetry_root
                self.assertTrue(np.isnan(h5[f"{empty_root}/{name}/value"][()]))

    def test_profile_time_axis_matches_standard_per_beat_interpolation(self) -> None:
        segments = _segments(radius_count=1, branch_count=1)
        cycle_boundaries = np.asarray([0, 2, 5], dtype=np.int32)

        result = interpolate_velocity_profiles_per_beat(
            segments.centered_velocity_profiles,
            cycle_boundaries,
        )
        expected = per_beat_signal_analysis(
            segments.centered_velocity_profiles[0, 0, :, 100],
            cycle_boundaries,
            1,
            index_base=0,
        ).velocity_signal_per_beat.T

        self.assertEqual((256, 4, 2, 1, 1), result.shape)
        np.testing.assert_allclose(
            result[100, :, :, 0, 0],
            expected,
            equal_nan=True,
        )

    def test_centering_anchors_edges_without_clipping_negative_velocity(self) -> None:
        raw = np.asarray(
            [[[
                [-8.0, 2.0, 8.0, -2.0, 7.0, 1.0, -9.0],
                [-8.0, 2.0, 8.0, 22.0, 7.0, 1.0, -9.0],
            ]]],
            dtype=np.float32,
        )

        result = process_velocity_profiles(
            raw,
            pixel_size_mm=0.01,
            velocity_profile_threshold=0.5,
            interpolation_points=33,
        )

        centered = result.centered_velocity[0, 0, 0]
        np.testing.assert_array_equal(result.raw_profile.velocity, raw)
        np.testing.assert_array_equal(
            result.raw_profile.x_micrometers,
            result.raw_x_micrometers,
        )
        np.testing.assert_array_equal(
            result.interpolated_profile.velocity,
            result.centered_velocity,
        )
        np.testing.assert_array_equal(
            result.interpolated_profile.x_micrometers,
            result.centered_x_micrometers,
        )
        self.assertAlmostEqual(0.0, float(centered[0]), places=5)
        self.assertAlmostEqual(0.0, float(centered[-1]), places=5)
        self.assertTrue(np.allclose(
            result.centered_x_micrometers[0, 0],
            -result.centered_x_micrometers[0, 0, ::-1],
        ))
        self.assertGreater(np.count_nonzero(centered < 0), 0)

    def test_poiseuille_fit_matches_matlab_custom_poly_functions(self) -> None:
        x_um = np.arange(-30, 31, 10, dtype=np.float32)
        profile = np.asarray([-8, 2, 8, 10, 7, 1, -9], dtype=np.float32)

        result = _matlab_poiseuille_fit(x_um, profile, 0.5)

        self.assertIsNotNone(result)
        coefficients, origin_um, roots_um, r_squared = result
        np.testing.assert_allclose(
            coefficients,
            [-0.025, -0.05, 10.0],
            rtol=1e-6,
        )
        self.assertEqual(0.0, origin_um)
        np.testing.assert_allclose(
            roots_um,
            [-15.1774468788, 13.1774468788],
            rtol=1e-6,
        )
        self.assertAlmostEqual(1.0, r_squared)

    def test_nan_rotation_interpolates_only_finite_values(self) -> None:
        image = np.full((5, 5), np.nan, dtype=np.float32)
        image[1:4, 1:4] = np.arange(1, 10, dtype=np.float32).reshape(
            3,
            3,
            order="F",
        )
        expected = np.asarray(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, 2.901924, 5.633975, 7.633975, np.nan],
                [np.nan, 1.901924, 5.0, 8.098076, np.nan],
                [np.nan, 2.366025, 4.366025, 7.098076, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            rotate_image_with_nan(image, 30.0),
            expected,
            rtol=1e-6,
            equal_nan=True,
        )

        low_velocity = np.full((7, 7), np.nan, dtype=np.float32)
        low_velocity[2:5, 2:5] = np.float32(0.2)
        finite_rotated = rotate_image_with_nan(low_velocity, 30.0)
        np.testing.assert_allclose(
            finite_rotated[np.isfinite(finite_rotated)],
            np.float32(0.2),
            rtol=1e-6,
        )


class ProfileArtifactTests(unittest.TestCase):
    def test_profile_median_handles_twelve_sparse_branches_by_radius(self) -> None:
        values = np.arange(2 * 12 * 3 * 4, dtype=np.float32).reshape(2, 12, 3, 4)
        values[0, :10, 1, 2] = np.nan
        values[1, :, 2, 3] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            expected = np.nanmedian(np.nanmedian(values, axis=1), axis=0)

        actual = _hierarchical_profile_median(values)

        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_profile_median_falls_back_when_masked_sort_indexing_fails(self) -> None:
        values = np.arange(2 * 12 * 3, dtype=np.float32).reshape(2, 12, 3)
        values[:, :9, 1] = np.nan
        values[0, :, 2] = np.inf
        expected = _finite_median(values, axis=1)

        with patch(
            "pipelines.waveform_velocity_core.figures.profiles.np.nanmedian",
            side_effect=IndexError(
                "index -9223372036854775799 is out of bounds for axis 1 with size 12"
            ),
        ):
            actual = _nanmedian(values, axis=1)

        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_profile_plot_limits_focus_positive_flow(self) -> None:
        y_min, y_max = _positive_focused_limits(
            np.asarray([-40.0, -12.0, 0.0, 4.0, 3.0], dtype=np.float32)
        )

        self.assertAlmostEqual(-1.12, y_min, places=5)
        self.assertAlmostEqual(4.48, y_max, places=5)

    def test_pngs_and_gifs_are_written_for_available_profiles(self) -> None:
        try:
            import matplotlib  # noqa: F401
            import PIL  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("profile artifact dependencies are not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            output = _FakeOutput(Path(temp_dir))
            writer = FigureArtifactWriter(output, "sample")
            context = SimpleNamespace(
                source_data=SimpleNamespace(timing=SimpleNamespace(dt_seconds=0.05)),
                artery_segment_result=_segments(radius_count=1, branch_count=1),
                vein_segment_result=_segments(radius_count=1, branch_count=1),
            )

            paths = export_cross_section_profile_artifacts(
                writer,
                context,
                max_gif_frames=4,
            )

            self.assertEqual(8, len(paths))
            for path in paths:
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)
            self.assertEqual(2, sum(path.suffix == ".gif" for path in paths))
            self.assertTrue(
                all("velocityProfiles" in path.parts for path in paths)
            )
            self.assertEqual(
                2,
                sum("poiseuille_profile" in path.name for path in paths),
            )


def _segments(*, radius_count: int, branch_count: int):
    frames = 6
    width = 181
    x_pixels = np.linspace(-3.0, 3.0, width, dtype=np.float32)
    profiles = np.empty(
        (radius_count, branch_count, frames, width),
        dtype=np.float32,
    )
    for radius in range(radius_count):
        for branch in range(branch_count):
            for frame in range(frames):
                center = 10.0 + frame + radius + branch
                profiles[radius, branch, frame] = (
                    center - 0.8 * x_pixels**2 + 0.3 * x_pixels
                )
    processed = process_velocity_profiles(
        profiles,
        pixel_size_mm=0.01,
        velocity_profile_threshold=0.5,
    )
    profiles_masked = profiles.copy()
    if branch_count:
        profiles_masked[..., 0] = np.nan
    return SimpleNamespace(
        branch_ids=np.arange(1, branch_count + 1, dtype=np.int32),
        velocity_profiles=profiles,
        transverse_velocity_profiles_masked=profiles_masked,
        longitudinal_velocity_profiles_unmasked=(profiles + np.float32(50.0)),
        longitudinal_velocity_profiles_masked=(profiles_masked + np.float32(100.0)),
        profile_x_micrometers=processed.raw_x_micrometers,
        profile_sample_count=np.full(
            (radius_count, branch_count),
            width,
            dtype=np.int32,
        ),
        profile_rotation_degrees=np.zeros(
            (radius_count, branch_count),
            dtype=np.float32,
        ),
        centered_velocity_profiles=processed.centered_velocity,
        centered_profile_x_micrometers=processed.centered_x_micrometers,
        profile_center_micrometers=processed.center_micrometers,
        profile_lumen_edges_micrometers=processed.lumen_edges_micrometers,
        profile_centering_fit_r_squared=processed.centering_fit_r_squared,
        poiseuille_coefficients=processed.poiseuille_coefficients,
        poiseuille_origin_micrometers=(
            processed.poiseuille_origin_micrometers
        ),
        poiseuille_roots_micrometers=(
            processed.poiseuille_roots_micrometers
        ),
        poiseuille_r_squared=processed.poiseuille_r_squared,
        poiseuille_profile_spatial_std=np.zeros(
            (radius_count, branch_count, width),
            dtype=np.float32,
        ),
    )


class _FakeOutput:
    available = True

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manager = SimpleNamespace(layout=SimpleNamespace(stem="sample"))

    def path_for(self, output_type: OutputType, filename: str | None = None) -> Path:
        if output_type not in (OutputType.PNG, OutputType.GIF):
            raise AssertionError(output_type)
        return self.root / output_type.value / (filename or "sample")

    def write_png(self, output, filename: str | None = None) -> Path:
        return write_png_file(self.path_for(OutputType.PNG, filename), output)


if __name__ == "__main__":
    unittest.main()
