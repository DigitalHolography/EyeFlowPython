from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

BRANCH_SIGNALS = "segmentation/Retina/binary/branch_signals"
LABELED_VESSELS = "segmentation/Retina/binary/labeled_vessels"
ARTERY_MASK = "segmentation/Retina/av/artery_mask"
VEIN_MASK = "segmentation/Retina/av/vein_mask"
HARMONIC_COUNT = 13
ARTERY_VPB = "Artery/VelocityPerBeat"
VEIN_VPB = "Vein/VelocityPerBeat"


def pick_labels(h5: h5py.File, vessel_kind: str, branch_count: int) -> list[int]:
    labels = np.asarray(h5[LABELED_VESSELS])
    artery = np.asarray(h5[ARTERY_MASK]) > 0
    vein = np.asarray(h5[VEIN_MASK]) > 0
    primary = artery if vessel_kind == "artery" else vein
    secondary = vein if vessel_kind == "artery" else artery
    picked = []

    for label in (int(x) for x in np.unique(labels) if x > 0):
        if label > branch_count:
            continue
        label_mask = labels == label
        if np.count_nonzero(label_mask & primary) > np.count_nonzero(label_mask & secondary):
            picked.append(label)

    return picked


def moving_average(x: np.ndarray, dt: float) -> np.ndarray:
    width = max(3, int(round(0.05 / dt)))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(x, kernel, mode="same")


def find_peaks(x: np.ndarray, min_distance: int) -> np.ndarray:
    threshold = np.percentile(x, 95)
    peaks = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    peaks = peaks[x[peaks] >= threshold]

    kept: list[int] = []
    for peak in peaks:
        if not kept or peak - kept[-1] >= min_distance:
            kept.append(int(peak))

    return np.asarray(kept, dtype=int)


def infer_dt(h5: h5py.File) -> float:
    sampling_freq = float(h5.attrs.get("sampling_freq", h5.attrs.get("fs", 0.0)))
    batch_size = float(h5.attrs.get("batch_size", h5.attrs.get("batch_stride", 0.0)))
    if sampling_freq <= 0 or batch_size <= 0:
        raise ValueError(
            "Could not infer dt from HDF5 root attrs. Expected sampling_freq and batch_size."
        )
    return batch_size / sampling_freq


def select_signal(h5: h5py.File, vessel: str) -> np.ndarray:
    branch_signals = np.asarray(h5[BRANCH_SIGNALS], dtype=float)
    labels = pick_labels(h5, vessel, branch_signals.shape[0])

    if labels:
        signal = np.nanmean(branch_signals[np.asarray(labels) - 1], axis=0)
    else:
        signal = np.nanmean(branch_signals, axis=0)

    return np.nan_to_num(signal - np.nanmean(signal))


def detect_systolic_peaks(signal: np.ndarray, dt: float) -> np.ndarray:
    derivative = np.gradient(moving_average(signal, dt))
    peaks = find_peaks(derivative, max(1, int(0.5 / dt)))

    if peaks.size < 2:
        raise ValueError("Could not detect at least two systolic peaks.")

    return peaks


def interp_cycle(beat: np.ndarray, n_fft: int) -> np.ndarray:
    x = np.arange(beat.size, dtype=float)
    xp = np.linspace(0.0, float(beat.size), n_fft + 1, endpoint=True)[:-1]
    return np.interp(xp, x, beat, period=float(beat.size))


def per_beat_signal_analysis(
    signal: np.ndarray, sys_idx_list: np.ndarray, harmonic_count: int
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
        beat_interp = interp_cycle(beat, n_fft)
        beat_fft = np.fft.fft(beat_interp, n_fft)

        keep = min(max(1, harmonic_count), beat_fft.size)
        band_spectrum = np.zeros(n_fft, dtype=np.complex64)
        band_spectrum[:keep] = (2.0 * beat_fft[:keep]).astype(np.complex64)
        band_spectrum[0] = np.complex64(beat_fft[0])

        raw[beat_idx] = beat_interp.astype(np.float32)
        fft_full[beat_idx] = beat_fft.astype(np.complex64)
        band[beat_idx] = np.abs(np.fft.ifft(band_spectrum, n_fft)).astype(np.float32)

    return raw, fft_full, band


def write_value_dataset(
    h5: h5py.File, group_path: str, data: np.ndarray, unit: str | None = None
) -> None:
    group = h5.require_group(group_path)
    if "value" in group:
        del group["value"]
    dataset = group.create_dataset("value", data=data)
    if unit is not None:
        dataset.attrs["unit"] = unit


def write_velocity_per_beat_fields(h5: h5py.File, dt: float) -> tuple[np.ndarray, int]:
    artery_signal = select_signal(h5, "artery")
    vein_signal = select_signal(h5, "vein")
    sys_idx_list = detect_systolic_peaks(artery_signal, dt)
    beat_period_idx = np.diff(sys_idx_list).astype(np.int32)[np.newaxis, :]
    beat_period_seconds = beat_period_idx.astype(np.float32) * np.float32(dt)

    write_value_dataset(h5, f"{ARTERY_VPB}/beatPeriodIdx", beat_period_idx)
    write_value_dataset(h5, f"{ARTERY_VPB}/beatPeriodSeconds", beat_period_seconds, "s")

    for prefix, signal in ((ARTERY_VPB, artery_signal), (VEIN_VPB, vein_signal)):
        raw, fft_full, band = per_beat_signal_analysis(signal, sys_idx_list, HARMONIC_COUNT)
        write_value_dataset(h5, f"{prefix}/VelocitySignalPerBeat", raw)
        write_value_dataset(h5, f"{prefix}/VelocitySignalPerBeatFFT_abs", np.abs(fft_full).astype(np.float32))
        write_value_dataset(h5, f"{prefix}/VelocitySignalPerBeatFFT_arg", np.angle(fft_full).astype(np.float32))
        write_value_dataset(h5, f"{prefix}/VelocitySignalPerBeatBandLimited", band)
        write_value_dataset(h5, f"{prefix}/VmaxPerBeatBandLimited", np.max(band, axis=1).astype(np.float32))
        write_value_dataset(h5, f"{prefix}/VminPerBeatBandLimited", np.min(band, axis=1).astype(np.float32))
        write_value_dataset(h5, f"{prefix}/VTIPerBeat", (np.sum(raw, axis=1) * dt).astype(np.float32))

    return beat_period_seconds, sys_idx_list.size - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5", type=Path)
    parser.add_argument("--dt", type=float, help="Seconds per frame. Overrides HDF5 metadata.")
    parser.add_argument("--output", type=Path, help="Optional output HDF5 path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target = args.input_h5 if args.output is None else args.output
    if args.output is not None:
        shutil.copy2(args.input_h5, target)

    with h5py.File(target, "r+") as h5:
        dt = infer_dt(h5) if args.dt is None else args.dt
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        beat_periods, n_beats = write_velocity_per_beat_fields(h5, dt)

    print(
        f"wrote VelocityPerBeat fields for {n_beats} beats; beatPeriodSeconds shape {tuple(beat_periods.shape)} "
        f"using dt={dt:.9f}s to {target}"
    )


if __name__ == "__main__":
    main()
