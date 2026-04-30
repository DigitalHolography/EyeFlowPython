from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipelines.dual_input_tutorial import DualInputTutorial  # noqa: E402
from input_output import PipelineInputView  # noqa: E402


class DualInputTutorialTests(unittest.TestCase):
    def test_pipeline_reads_hd_and_dv_inputs_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hd_path = tmp_path / "sample_holodoppler.h5"
            dv_path = tmp_path / "sample_doppler_vision.h5"
            work_path = tmp_path / "work.h5"

            with h5py.File(hd_path, "w") as hd_h5:
                hd_h5.create_dataset("moment0", data=np.array([[1.0, 3.0], [5.0, 7.0]]))
                hd_h5.create_group("segmentation")

            with h5py.File(dv_path, "w") as dv_h5:
                artery = dv_h5.create_group("Artery")
                velocity = artery.create_group("VelocityPerBeat")
                vti = velocity.create_group("VTIPerBeat")
                vti.create_dataset("value", data=np.array([2.0, 4.0, 6.0], dtype=float))
                dv_h5.create_group("Meta")

            with (
                h5py.File(work_path, "w") as work_h5,
                h5py.File(hd_path, "r") as hd_h5,
                h5py.File(dv_path, "r") as dv_h5,
            ):
                input_view = PipelineInputView(
                    work_h5=work_h5,
                    holodoppler_h5=hd_h5,
                    doppler_vision_h5=dv_h5,
                    preferred_input="both",
                )
                self.assertIs(input_view.hd, hd_h5)
                self.assertIs(input_view.dv, dv_h5)
                self.assertIs(input_view.work, work_h5)
                result = DualInputTutorial().run(input_view)

            self.assertEqual(
                2,
                int(result.metrics["summary/hd_root_group_count"].data),
            )
            self.assertEqual(
                2,
                int(result.metrics["summary/dv_root_group_count"].data),
            )
            self.assertEqual(
                0,
                int(result.metrics["summary/shared_root_group_count"].data),
            )
            self.assertEqual(
                4,
                int(result.metrics["summary/hd_example_size"].data),
            )
            self.assertEqual(
                3,
                int(result.metrics["summary/dv_example_size"].data),
            )
            self.assertAlmostEqual(
                4.0,
                float(result.metrics["summary/hd_example_mean"].data),
            )
            self.assertAlmostEqual(
                4.0,
                float(result.metrics["summary/dv_example_mean"].data),
            )
            self.assertAlmostEqual(
                0.0,
                float(result.metrics["summary/hd_minus_dv_example_mean"].data),
            )
            self.assertEqual(
                "moment0",
                result.attrs["hd_example_path"],
            )
            self.assertEqual(
                "Artery/VelocityPerBeat/VTIPerBeat/value",
                result.attrs["dv_example_path"],
            )

    def test_eye_flow_proxy_resolves_root_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            hd_path = tmp_path / "sample_holodoppler.h5"
            dv_path = tmp_path / "sample_doppler_vision.h5"
            work_path = tmp_path / "work.h5"

            with h5py.File(hd_path, "w") as hd_h5:
                hd_h5.create_dataset("moment0", data=np.array([1.0], dtype=float))

            with h5py.File(dv_path, "w") as dv_h5:
                dv_h5.create_dataset("Meta/example", data=np.array([2.0], dtype=float))

            with h5py.File(work_path, "w") as work_h5:
                work_h5.create_dataset(
                    "summary/value",
                    data=np.array([22.0], dtype=float),
                )
                work_h5.create_dataset(
                    "other/path",
                    data=np.array([33.0], dtype=float),
                )

            with (
                h5py.File(work_path, "r") as work_h5,
                h5py.File(hd_path, "r") as hd_h5,
                h5py.File(dv_path, "r") as dv_h5,
            ):
                input_view = PipelineInputView(
                    work_h5=work_h5,
                    holodoppler_h5=hd_h5,
                    doppler_vision_h5=dv_h5,
                )

                latest_value = np.asarray(input_view.ef["summary/value"])
                explicit_value = np.asarray(input_view.ef["other/path"])

            np.testing.assert_allclose(latest_value, np.array([22.0], dtype=float))
            np.testing.assert_allclose(explicit_value, np.array([33.0], dtype=float))

if __name__ == "__main__":
    unittest.main()
