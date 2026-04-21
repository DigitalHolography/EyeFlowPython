from __future__ import annotations

import json

import h5py
import numpy as np

from .core.base import ProcessPipeline, ProcessResult, registerPipeline, with_attrs


@registerPipeline(name="script_eyeflow")
class ScriptEyeFlow(ProcessPipeline):
    description = (
        "Generate artery and vein per-beat velocity datasets from EyeFlow "
        "segmentation masks and branch signals."
    )

    HD_PARAMETERS = "HD_parameters"
    BRANCH_SIGNALS = "segmentation/Retina/binary/branch_signals"
    LABELED_VESSELS = "segmentation/Retina/binary/labeled_vessels"
    ARTERY_MASK = "segmentation/Retina/av/artery_mask"
    VEIN_MASK = "segmentation/Retina/av/vein_mask"
    HARMONIC_COUNT = 13
    ARTERY_VPB = "Artery/VelocityPerBeat"
    VEIN_VPB = "Vein/VelocityPerBeat"

    @staticmethod
    def _pick_labels(
        h5file: h5py.File, vessel_kind: str, branch_count: int
    ) -> list[int]:
        labels = np.asarray(h5file[ScriptEyeFlow.LABELED_VESSELS])
        artery = np.asarray(h5file[ScriptEyeFlow.ARTERY_MASK]) > 0
        vein = np.asarray(h5file[ScriptEyeFlow.VEIN_MASK]) > 0
        primary = artery if vessel_kind == "artery" else vein
        secondary = vein if vessel_kind == "artery" else artery
        picked: list[int] = []

        for label in (int(value) for value in np.unique(labels) if value > 0):
            if label > branch_count:
                continue
            label_mask = labels == label
            if np.count_nonzero(label_mask & primary) > np.count_nonzero(
                label_mask & secondary
            ):
                picked.append(label)

        return picked

    @staticmethod
    def _moving_average(values: np.ndarray, dt: float) -> np.ndarray:
        width = max(3, int(round(0.05 / dt)))
        if width % 2 == 0:
            width += 1
        kernel = np.ones(width, dtype=float) / width
        return np.convolve(values, kernel, mode="same")

    @staticmethod
    def _find_peaks(values: np.ndarray, min_distance: int) -> np.ndarray:
        threshold = np.percentile(values, 95)
        peaks = (
            np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))
            + 1
        )
        peaks = peaks[values[peaks] >= threshold]

        kept: list[int] = []
        for peak in peaks:
            if not kept or peak - kept[-1] >= min_distance:
                kept.append(int(peak))

        return np.asarray(kept, dtype=int)

    @staticmethod
    def _load_hd_parameters(h5file: h5py.File) -> dict[str, object]:
        dataset = h5file.get(ScriptEyeFlow.HD_PARAMETERS)
        if dataset is None:
            return {}

        payload = dataset[()]
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if not isinstance(payload, str):
            return {}

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @classmethod
    def _infer_dt(cls, h5file: h5py.File) -> float:
        params = cls._load_hd_parameters(h5file)
        sampling_freq = float(
            h5file.attrs.get(
                "sampling_freq",
                h5file.attrs.get("fs", params.get("sampling_freq", params.get("fs", 0))),
            )
        )
        batch_step = float(
            h5file.attrs.get(
                "batch_size",
                h5file.attrs.get(
                    "batch_stride",
                    params.get("batch_size", params.get("batch_stride", 0)),
                ),
            )
        )
        if sampling_freq <= 0 or batch_step <= 0:
            raise ValueError(
                "Could not infer dt. Expected sampling_freq and batch_size or "
                "batch_stride in root attrs or HD_parameters."
            )
        return batch_step / sampling_freq

    @classmethod
    def _select_signal(cls, h5file: h5py.File, vessel: str) -> np.ndarray:
        branch_signals = np.asarray(h5file[cls.BRANCH_SIGNALS], dtype=float)
        labels = cls._pick_labels(h5file, vessel, branch_signals.shape[0])

        if labels:
            signal = np.nanmean(branch_signals[np.asarray(labels) - 1], axis=0)
        else:
            signal = np.nanmean(branch_signals, axis=0)

        return np.nan_to_num(signal - np.nanmean(signal))

    @classmethod
    def _detect_systolic_peaks(cls, signal: np.ndarray, dt: float) -> np.ndarray:
        derivative = np.gradient(cls._moving_average(signal, dt))
        peaks = cls._find_peaks(derivative, max(1, int(0.5 / dt)))

        if peaks.size < 2:
            raise ValueError("Could not detect at least two systolic peaks.")

        return peaks

    @staticmethod
    def _interp_cycle(beat: np.ndarray, n_fft: int) -> np.ndarray:
        x = np.arange(beat.size, dtype=float)
        xp = np.linspace(0.0, float(beat.size), n_fft + 1, endpoint=True)[:-1]
        return np.interp(xp, x, beat, period=float(beat.size))

    @classmethod
    def _per_beat_signal_analysis(
        cls, signal: np.ndarray, sys_idx_list: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_beats = sys_idx_list.size - 1
        n_fft = 1 << int(np.ceil(np.log2(np.max(np.diff(sys_idx_list)))))
        raw = np.full((n_beats, n_fft), np.nan, dtype=np.float32)
        fft_full = np.full((n_beats, n_fft), np.nan + 0j, dtype=np.complex64)
        band = np.full((n_beats, n_fft), np.nan, dtype=np.float32)

        for beat_idx in range(n_beats):
            start = int(sys_idx_list[beat_idx])
            end = int(sys_idx_list[beat_idx + 1])
            beat = np.asarray(signal[start : end + 1], dtype=float)
            beat_interp = cls._interp_cycle(beat, n_fft)
            beat_fft = np.fft.fft(beat_interp, n_fft)

            keep = min(max(1, cls.HARMONIC_COUNT), beat_fft.size)
            band_spectrum = np.zeros(n_fft, dtype=np.complex64)
            band_spectrum[:keep] = (2.0 * beat_fft[:keep]).astype(np.complex64)
            band_spectrum[0] = np.complex64(beat_fft[0])

            raw[beat_idx] = beat_interp.astype(np.float32)
            fft_full[beat_idx] = beat_fft.astype(np.complex64)
            band[beat_idx] = np.abs(np.fft.ifft(band_spectrum, n_fft)).astype(
                np.float32
            )

        return raw, fft_full, band

    @staticmethod
    def _dataset_key(prefix: str, name: str) -> str:
        return f"{prefix}/{name}/value"

    @classmethod
    def _velocity_per_beat_metrics(
        cls,
        prefix: str,
        signal: np.ndarray,
        sys_idx_list: np.ndarray,
        dt: float,
    ) -> dict[str, object]:
        raw, fft_full, band = cls._per_beat_signal_analysis(signal, sys_idx_list)
        return {
            cls._dataset_key(prefix, "VelocitySignalPerBeat"): with_attrs(
                raw, {"unit": ["a.u."]}
            ),
            cls._dataset_key(prefix, "VelocitySignalPerBeatFFT_abs"): with_attrs(
                np.abs(fft_full).astype(np.float32), {"unit": ["a.u."]}
            ),
            cls._dataset_key(prefix, "VelocitySignalPerBeatFFT_arg"): with_attrs(
                np.angle(fft_full).astype(np.float32), {"unit": ["rad"]}
            ),
            cls._dataset_key(prefix, "VelocitySignalPerBeatBandLimited"): with_attrs(
                band, {"unit": ["a.u."]}
            ),
            cls._dataset_key(prefix, "VmaxPerBeatBandLimited"): with_attrs(
                np.max(band, axis=1).astype(np.float32), {"unit": ["a.u."]}
            ),
            cls._dataset_key(prefix, "VminPerBeatBandLimited"): with_attrs(
                np.min(band, axis=1).astype(np.float32), {"unit": ["a.u."]}
            ),
            cls._dataset_key(prefix, "VTIPerBeat"): with_attrs(
                (np.sum(raw, axis=1) * dt).astype(np.float32), {"unit": ["a.u.*s"]}
            ),
        }

    def run(self, h5file: h5py.File) -> ProcessResult:
        dt = self._infer_dt(h5file)
        if dt <= 0:
            raise ValueError("dt must be > 0.")

        artery_signal = self._select_signal(h5file, "artery")
        vein_signal = self._select_signal(h5file, "vein")
        sys_idx_list = self._detect_systolic_peaks(artery_signal, dt)
        beat_count = int(sys_idx_list.size - 1)
        beat_period_idx = np.diff(sys_idx_list).astype(np.int32)[np.newaxis, :]
        beat_period_seconds = beat_period_idx.astype(np.float32) * np.float32(dt)

        metrics: dict[str, object] = {
            self._dataset_key(self.ARTERY_VPB, "beatPeriodIdx"): with_attrs(
                beat_period_idx, {"unit": ["frames"]}
            ),
            self._dataset_key(self.ARTERY_VPB, "beatPeriodSeconds"): with_attrs(
                beat_period_seconds, {"unit": ["s"]}
            ),
        }
        metrics.update(
            self._velocity_per_beat_metrics(
                self.ARTERY_VPB,
                artery_signal,
                sys_idx_list,
                dt,
            )
        )
        metrics.update(
            self._velocity_per_beat_metrics(
                self.VEIN_VPB,
                vein_signal,
                sys_idx_list,
                dt,
            )
        )

        attrs = {
            "dt_seconds": float(dt),
            "beat_count": beat_count,
            "harmonic_count": int(self.HARMONIC_COUNT),
        }
        return ProcessResult(metrics=metrics, attrs=attrs)
