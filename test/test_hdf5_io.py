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

from pipelines.core.base import DatasetValue  # noqa: E402
from input_output.hdf5 import write_value_dataset  # noqa: E402


class HDF5IoTests(unittest.TestCase):
    def test_write_value_dataset_writes_nameid_and_user_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "out.h5"

            with h5py.File(h5_path, "w") as h5file:
                write_value_dataset(
                    h5file,
                    "summary/mean",
                    DatasetValue(np.array([1.0, 2.0], dtype=np.float64), {"unit": "a.u."}),
                )

            with h5py.File(h5_path, "r") as h5file:
                dataset = h5file["summary"]["mean"]
                self.assertEqual(dataset.attrs["unit"], "a.u.")
                self.assertEqual(dataset.attrs["nameID"], "summary/mean")
                self.assertEqual(dataset.dtype, np.dtype("float64"))

    def test_write_value_dataset_converts_booleans_to_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "out.h5"

            with h5py.File(h5_path, "w") as h5file:
                write_value_dataset(
                    h5file,
                    "mask",
                    DatasetValue(np.array([[True, False]], dtype=bool), {"unit": "bool"}),
                )

            with h5py.File(h5_path, "r") as h5file:
                dataset = h5file["mask"]
                self.assertEqual(dataset.dtype, np.dtype("uint8"))
                self.assertEqual(dataset.attrs["original_class"], "bool")
                self.assertEqual(dataset.attrs["nameID"], "mask")

    def test_write_value_dataset_does_not_compress_large_numeric_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            h5_path = tmp_path / "out.h5"
            large_array = np.ones((256, 256, 20), dtype=np.float32)

            with h5py.File(h5_path, "w") as h5file:
                write_value_dataset(h5file, "big/values", large_array)

            with h5py.File(h5_path, "r") as h5file:
                dataset = h5file["big"]["values"]
                self.assertIsNone(dataset.compression)
                self.assertIsNone(dataset.chunks)


if __name__ == "__main__":
    unittest.main()
