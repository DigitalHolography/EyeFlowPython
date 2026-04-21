from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import zipfile

import h5py
import numpy as np

BRANCH_SIGNALS = "segmentation/Retina/binary/branch_signals"
LABELED_VESSELS = "segmentation/Retina/binary/labeled_vessels"
ARTERY_MASK = "segmentation/Retina/av/artery_mask"
VEIN_MASK = "segmentation/Retina/av/vein_mask"
HARMONIC_COUNT = 13
ARTERY_VPB = "Artery/VelocityPerBeat"
VEIN_VPB = "Vein/VelocityPerBeat"


@dataclass
class ProcessedFile:
    source: Path
    output: Path
    beat_count: int
    dt: float


def _noop_logger(_: str) -> None:
    pass


def _is_h5(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".h5"


def _is_zip(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def default_output_root(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_processed"
    return input_path.parent / f"{input_path.stem}_processed"


def _safe_output_file(output_root: Path, relative_path: Path, source_path: Path) -> Path:
    candidate = output_root / relative_path
    if candidate.resolve() == source_path.resolve():
        candidate = candidate.with_name(f"{candidate.stem}_processed{candidate.suffix}")
    return candidate


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
        if np.count_nonzero(label_mask & primary) > np.count_nonzero(
            label_mask & secondary
        ):
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
        raw, fft_full, band = per_beat_signal_analysis(
            signal, sys_idx_list, HARMONIC_COUNT
        )
        write_value_dataset(h5, f"{prefix}/VelocitySignalPerBeat", raw)
        write_value_dataset(
            h5,
            f"{prefix}/VelocitySignalPerBeatFFT_abs",
            np.abs(fft_full).astype(np.float32),
        )
        write_value_dataset(
            h5,
            f"{prefix}/VelocitySignalPerBeatFFT_arg",
            np.angle(fft_full).astype(np.float32),
        )
        write_value_dataset(h5, f"{prefix}/VelocitySignalPerBeatBandLimited", band)
        write_value_dataset(
            h5,
            f"{prefix}/VmaxPerBeatBandLimited",
            np.max(band, axis=1).astype(np.float32),
        )
        write_value_dataset(
            h5,
            f"{prefix}/VminPerBeatBandLimited",
            np.min(band, axis=1).astype(np.float32),
        )
        write_value_dataset(
            h5, f"{prefix}/VTIPerBeat", (np.sum(raw, axis=1) * dt).astype(np.float32)
        )

    return beat_period_seconds, sys_idx_list.size - 1


def process_h5_file(
    input_path: Path, output_path: Path, dt_override: float | None = None
) -> ProcessedFile:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "r+") as h5:
        dt = infer_dt(h5) if dt_override is None else dt_override
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        _, beat_count = write_velocity_per_beat_fields(h5, dt)

    return ProcessedFile(
        source=input_path,
        output=output_path,
        beat_count=beat_count,
        dt=dt,
    )


def _collect_folder_h5_files(source_dir: Path, output_root: Path) -> list[Path]:
    files: list[Path] = []
    output_root_resolved = output_root.resolve()

    for path in sorted(source_dir.rglob("*")):
        if not _is_h5(path):
            continue
        if path.exists() and _is_relative_to(path.resolve(), output_root_resolved):
            continue
        files.append(path)

    return files


def process_input(
    input_path: Path,
    output_root: Path | None = None,
    dt_override: float | None = None,
    logger=None,
) -> list[ProcessedFile]:
    log = logger or _noop_logger
    source = input_path.resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    output_root = (output_root or default_output_root(source)).resolve()
    results: list[ProcessedFile] = []

    if _is_h5(source):
        output_path = _safe_output_file(output_root, Path(source.name), source)
        log(f"Processing file: {source}")
        results.append(process_h5_file(source, output_path, dt_override))
        log(f"Created: {output_path}")
        return results

    if source.is_dir():
        files = _collect_folder_h5_files(source, output_root)
        if not files:
            raise ValueError(f"No .h5 files found in {source}")
        log(f"Found {len(files)} h5 file(s) in {source}")
        for file_path in files:
            relative_path = file_path.relative_to(source)
            output_path = _safe_output_file(output_root, relative_path, file_path)
            log(f"Processing file: {file_path}")
            results.append(process_h5_file(file_path, output_path, dt_override))
            log(f"Created: {output_path}")
        return results

    if _is_zip(source):
        with tempfile.TemporaryDirectory(prefix="eyeflow_zip_") as tmpdir:
            extracted_root = Path(tmpdir)
            with zipfile.ZipFile(source) as archive:
                archive.extractall(extracted_root)
            files = sorted(path for path in extracted_root.rglob("*") if _is_h5(path))
            if not files:
                raise ValueError(f"No .h5 files found in {source}")
            log(f"Found {len(files)} h5 file(s) in {source}")
            for file_path in files:
                relative_path = file_path.relative_to(extracted_root)
                output_path = _safe_output_file(output_root, relative_path, file_path)
                log(f"Processing file: {relative_path}")
                results.append(process_h5_file(file_path, output_path, dt_override))
                log(f"Created: {output_path}")
        return results

    raise ValueError(f"Unsupported input: {source}")
