"""Port of BloodFlowVelocity/perBeatSignalAnalysis.m."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._signal_utils import next_power_of_two, normalize_cycle_boundaries


@dataclass(frozen=True)
class PerBeatSignalAnalysisResult:
    velocity_signal_per_beat: np.ndarray
    velocity_signal_per_beat_fft: np.ndarray
    velocity_signal_per_beat_band_limited: np.ndarray


def _interpft_real(signal: np.ndarray, target_length: int) -> np.ndarray:
    source = np.asarray(signal, dtype=np.float64).reshape(-1)
    source_length = int(source.size)
    if source_length == 0:
        raise ValueError("interpft requires a non-empty signal.")
    if target_length <= 0:
        raise ValueError("interpft target_length must be positive.")
    if target_length == source_length:
        return source.copy()

    spectrum = np.fft.fft(source)
    resized = np.zeros(int(target_length), dtype=np.complex128)
    _copy_resized_spectrum(spectrum, resized, source_length)
    interpolated = np.fft.ifft(resized) * (float(target_length) / float(source_length))
    return interpolated.real


def _copy_resized_spectrum(
    spectrum: np.ndarray,
    resized: np.ndarray,
    source_length: int,
) -> None:
    if source_length % 2 == 0:
        half = source_length // 2
        resized[:half] = spectrum[:half]
        resized[-(source_length - half - 1) :] = spectrum[half + 1 :]
        resized[half] = spectrum[half] / 2.0
        resized[resized.size - half] = spectrum[half] / 2.0
        return

    pivot = source_length // 2 + 1
    resized[:pivot] = spectrum[:pivot]
    resized[-(source_length // 2) :] = spectrum[pivot:]


def per_beat_signal_analysis(
    signal,
    sys_idx_list,
    band_limited_signal_harmonic_count: int,
    *,
    index_base: int | None = None,
) -> PerBeatSignalAnalysisResult:
    signal_array = np.asarray(signal, dtype=np.float64).reshape(-1)
    if signal_array.size == 0:
        raise ValueError("signal must contain at least one sample.")
    if band_limited_signal_harmonic_count < 1:
        raise ValueError("band_limited_signal_harmonic_count must be positive.")

    cycle_boundaries = normalize_cycle_boundaries(
        sys_idx_list,
        signal_array.size,
        index_base=index_base,
    )
    return _analyze_cycles(
        signal_array,
        cycle_boundaries,
        int(band_limited_signal_harmonic_count),
    )


def _analyze_cycles(
    signal_array: np.ndarray,
    cycle_boundaries: np.ndarray,
    harmonic_count: int,
) -> PerBeatSignalAnalysisResult:
    number_of_beats = int(cycle_boundaries.size - 1)
    n_fft = next_power_of_two(int(np.max(np.diff(cycle_boundaries))))
    per_beat, per_beat_fft, band_limited = _empty_outputs(number_of_beats, n_fft)

    for beat_index in range(number_of_beats):
        start = int(cycle_boundaries[beat_index])
        stop = int(cycle_boundaries[beat_index + 1]) + 1
        beat_interp = _interpft_real(signal_array[start:stop], n_fft + 1)[:-1]
        beat_fft = np.fft.fft(beat_interp, n=n_fft)
        per_beat[beat_index, :] = beat_interp
        per_beat_fft[beat_index, :] = beat_fft
        band_limited[beat_index, :] = _band_limited_signal(
            beat_fft,
            n_fft,
            harmonic_count,
        )

    return PerBeatSignalAnalysisResult(per_beat, per_beat_fft, band_limited)


def _empty_outputs(
    number_of_beats: int,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    per_beat = np.full((number_of_beats, n_fft), np.nan, dtype=np.float64)
    per_beat_fft = np.full((number_of_beats, n_fft), np.nan + 0j, dtype=np.complex128)
    band_limited = np.full((number_of_beats, n_fft), np.nan, dtype=np.float64)
    return per_beat, per_beat_fft, band_limited


def _band_limited_signal(
    beat_fft: np.ndarray,
    n_fft: int,
    harmonic_count: int,
) -> np.ndarray:
    count = min(int(harmonic_count), n_fft)
    band_limited_spectrum = beat_fft[:count].copy() * 2.0
    band_limited_spectrum[0] = beat_fft[0]
    return np.abs(np.fft.ifft(band_limited_spectrum, n=n_fft))
