from __future__ import annotations

import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="dual_input_tutorial")
class DualInputTutorial(ProcessPipeline):
    """
    Tutorial pipeline showing how to consume HD and DV inputs at the same time.

    When launched from the EyeFlow UI runtime, the incoming `h5file` object exposes:
    - `h5file.hd`: the holodoppler input handle
    - `h5file.dv`: the doppler vision input handle
    - `h5file.work`: the current EyeFlow output/work file
    """

    description = "Tutorial: read HD and DV inputs simultaneously in one pipeline."
    input_slot = "both"

    HD_EXAMPLE_PATHS = (
        "moment0",
        "segmentation/Retina/binary/branch_signals",
    )
    DV_EXAMPLE_PATHS = (
        "Artery/VelocityPerBeat/VTIPerBeat/value",
        "Artery/VelocityPerBeat/beatPeriodSeconds/value",
        "Vein/VelocityPerBeat/VTIPerBeat/value",
    )

    @staticmethod
    def _find_first_dataset(
        source: h5py.File,
        candidate_paths: tuple[str, ...],
    ) -> tuple[str | None, h5py.Dataset | None]:
        for path in candidate_paths:
            dataset = source.get(path)
            if isinstance(dataset, h5py.Dataset):
                return path, dataset
        return None, None

    @staticmethod
    def _numeric_dataset_summary(
        dataset: h5py.Dataset | None,
    ) -> tuple[np.float32, np.int32]:
        if dataset is None:
            return np.float32(np.nan), np.int32(0)

        values = np.asarray(dataset[()])
        if values.size == 0:
            return np.float32(np.nan), np.int32(0)

        try:
            numeric = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return np.float32(np.nan), np.int32(values.size)

        return np.float32(np.nanmean(numeric)), np.int32(numeric.size)

    def run(self, h5file) -> ProcessResult:
        hd_h5 = h5file.hd
        dv_h5 = h5file.dv
        if hd_h5 is None or dv_h5 is None:
            raise ValueError(
                "dual_input_tutorial requires both HD and DV inputs from the UI runtime."
            )

        hd_path, hd_dataset = self._find_first_dataset(hd_h5, self.HD_EXAMPLE_PATHS)
        dv_path, dv_dataset = self._find_first_dataset(dv_h5, self.DV_EXAMPLE_PATHS)

        hd_mean, hd_size = self._numeric_dataset_summary(hd_dataset)
        dv_mean, dv_size = self._numeric_dataset_summary(dv_dataset)
        mean_delta = (
            np.float32(hd_mean - dv_mean)
            if np.isfinite(hd_mean) and np.isfinite(dv_mean)
            else np.float32(np.nan)
        )

        hd_root_keys = sorted(str(key) for key in hd_h5.keys())
        dv_root_keys = sorted(str(key) for key in dv_h5.keys())
        shared_root_keys = sorted(set(hd_root_keys) & set(dv_root_keys))

        metrics = {
            "summary/hd_root_group_count": with_attrs(
                np.int32(len(hd_root_keys)),
                {"unit": ["count"]},
            ),
            "summary/dv_root_group_count": with_attrs(
                np.int32(len(dv_root_keys)),
                {"unit": ["count"]},
            ),
            "summary/shared_root_group_count": with_attrs(
                np.int32(len(shared_root_keys)),
                {"unit": ["count"]},
            ),
            "summary/hd_example_size": with_attrs(hd_size, {"unit": ["samples"]}),
            "summary/dv_example_size": with_attrs(dv_size, {"unit": ["samples"]}),
            "summary/hd_example_mean": with_attrs(hd_mean, {"unit": ["a.u."]}),
            "summary/dv_example_mean": with_attrs(dv_mean, {"unit": ["a.u."]}),
            "summary/hd_minus_dv_example_mean": with_attrs(
                mean_delta,
                {"unit": ["a.u."]},
            ),
            "summary/both_inputs_available": with_attrs(
                np.uint8(1),
                {"unit": ["bool"]},
            ),
        }
        attrs = {
            "hd_source_file": str(hd_h5.filename or ""),
            "dv_source_file": str(dv_h5.filename or ""),
            "hd_example_path": hd_path or "",
            "dv_example_path": dv_path or "",
            "shared_root_groups": shared_root_keys,
        }
        return ProcessResult(metrics=metrics, attrs=attrs)
