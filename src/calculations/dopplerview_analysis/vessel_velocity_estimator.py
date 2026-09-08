"""Chunked DopplerView-compatible vessel velocity estimation."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy import ndimage as ndi

from calculations.topology import annulus_mask
from utils.logger import Logger

SCRATCH_FRAME_CHUNK_SIZE = 32
SECTION_INNER_RADIUS_FRAC = 0.10
SECTION_OUTER_RADIUS_FRAC = 0.35
DEFAULT_LASER_WAVELENGTH_METERS = 8.52e-7
DEFAULT_NUMERICAL_APERTURE = 0.124


def _velocity_from_delta_frequency(
    delta_frequency,
    laser_wavelength: float = DEFAULT_LASER_WAVELENGTH_METERS,
    numerical_aperture: float = DEFAULT_NUMERICAL_APERTURE,
) -> np.ndarray:
    """Convert a Doppler-frequency shift in Hz to velocity in mm/s."""
    delta_frequency = np.asarray(delta_frequency, dtype=np.float32)
    return (
        np.float32(1e3) * laser_wavelength * delta_frequency / numerical_aperture
    ).astype(np.float32, copy=False)


def run_chunked_velocity_estimator(
    *,
    moment0,
    moment2,
    artery_mask,
    vein_mask,
    optic_disc_center=None,
    optic_disc_width=None,
    optic_disc_height=None,
    section_inner_radius_frac: float = SECTION_INNER_RADIUS_FRAC,
    section_outer_radius_frac: float = SECTION_OUTER_RADIUS_FRAC,
    local_background_dist: int,
    scratch_h5,
    laser_wavelength: float = DEFAULT_LASER_WAVELENGTH_METERS,
    numerical_aperture: float = DEFAULT_NUMERICAL_APERTURE,
    retain_velocity_video: bool = True,
) -> dict[str, object]:
    """Estimate velocity into scratch datasets without materializing full videos."""

    if tuple(moment0.shape) != tuple(moment2.shape) or len(moment0.shape) != 3:
        raise ValueError(
            "moment0 and moment2 must be matching 3-D datasets, got "
            f"{moment0.shape} and {moment2.shape}."
        )
    frame_count, height, width = (int(size) for size in moment0.shape)
    artery = np.asarray(artery_mask, dtype=bool)
    vein = np.asarray(vein_mask, dtype=bool)
    if artery.shape != (height, width) or vein.shape != (height, width):
        raise ValueError("Velocity masks must match the HD moment spatial shape.")

    Logger.log("Velocity estimator uses raw HD moments.")
    Logger.log(
        f"Velocity estimator uses {SCRATCH_FRAME_CHUNK_SIZE}-frame batched "
        "inpainting and summary-only frequency intermediates."
    )

    group = scratch_h5.require_group("waveform")
    velocity_dataset = (
        group.create_dataset(
            "velocity",
            shape=(frame_count, height, width),
            dtype=np.float32,
            chunks=(
                min(64, frame_count),
                min(32, height),
                min(32, width),
            ),
            compression=None,
        )
        if retain_velocity_video
        else None
    )
    vessel_mask = artery | vein
    disk, inpaint = _skimage_dependencies()
    inpaint_mask = _dilated_mask(vessel_mask, disk(int(local_background_dist)))
    section_mask = annulus_mask(
        (height, width),
        optic_disc_center,
        section_inner_radius_frac,
        section_outer_radius_frac,
    )
    artery_section = section_mask & artery
    vein_section = section_mask & vein

    averages = {
        name: np.zeros((height, width), dtype=np.float64)
        for name in ("moment0", "velocity", "fRMS", "fRMS_bkg", "deltafRMS")
    }
    signals = {
        name: np.full(frame_count, np.nan, dtype=np.float32)
        for name in (
            "artery_velocity",
            "vein_velocity",
            "artery_fRMS",
            "vein_fRMS",
            "artery_fRMS_bkg",
            "vein_fRMS_bkg",
            "vessel_fRMS_bkg",
            "artery_deltafRMS",
            "vein_deltafRMS",
        )
    }

    estimation_started = perf_counter()
    chunk_count = max(1, (frame_count + SCRATCH_FRAME_CHUNK_SIZE - 1) // SCRATCH_FRAME_CHUNK_SIZE)
    for chunk_index, start in enumerate(range(0, frame_count, SCRATCH_FRAME_CHUNK_SIZE), start=1):
        stop = min(start + SCRATCH_FRAME_CHUNK_SIZE, frame_count)
        frame_slice = slice(start, stop)
        m0 = _read_moment_chunk(moment0, frame_slice)
        m2 = _read_moment_chunk(moment2, frame_slice)
        mean_m0 = np.mean(m0, axis=(-1, -2), keepdims=True, dtype=np.float32)
        f_rms = np.sqrt(
            np.divide(
                m2,
                mean_m0,
                out=np.zeros_like(m2, dtype=np.float32),
                where=mean_m0 != 0,
            )
        ).astype(np.float32, copy=False)
        f_rms_background = _inpaint_frame_batch(
            f_rms,
            inpaint_mask,
            inpaint,
        )
        delta = _signed_rms_difference(f_rms, f_rms_background)
        velocity = _velocity_from_delta_frequency(
            delta,
            laser_wavelength=laser_wavelength,
            numerical_aperture=numerical_aperture,
        )

        if velocity_dataset is not None:
            velocity_dataset[frame_slice] = velocity
        averages["moment0"] += np.sum(m0, axis=0, dtype=np.float64)
        averages["velocity"] += np.sum(velocity, axis=0, dtype=np.float64)
        averages["fRMS"] += np.sum(f_rms, axis=0, dtype=np.float64)
        averages["fRMS_bkg"] += np.sum(f_rms_background, axis=0, dtype=np.float64)
        averages["deltafRMS"] += np.sum(delta, axis=0, dtype=np.float64)
        signals["artery_velocity"][frame_slice] = _masked_signal(
            velocity,
            artery_section,
        )
        signals["vein_velocity"][frame_slice] = _masked_signal(
            velocity,
            vein_section,
        )
        signals["artery_fRMS"][frame_slice] = _masked_signal(f_rms, artery_section)
        signals["vein_fRMS"][frame_slice] = _masked_signal(f_rms, vein_section)
        signals["artery_fRMS_bkg"][frame_slice] = _masked_signal(
            f_rms_background,
            artery_section,
        )
        signals["vein_fRMS_bkg"][frame_slice] = _masked_signal(
            f_rms_background,
            vein_section,
        )
        signals["vessel_fRMS_bkg"][frame_slice] = _masked_signal(
            f_rms_background,
            artery_section | vein_section,
        )
        signals["artery_deltafRMS"][frame_slice] = _masked_signal(
            delta,
            artery_section,
        )
        signals["vein_deltafRMS"][frame_slice] = _masked_signal(
            delta,
            vein_section,
        )
        if chunk_index == chunk_count or chunk_index % 10 == 0:
            Logger.log(
                f"Velocity estimation completed chunk {chunk_index}/{chunk_count} "
                f"({stop}/{frame_count} frames)."
            )

    Logger.log(
        f"Completed chunked velocity estimation in {perf_counter() - estimation_started:.1f}s."
    )

    divisor = np.float64(max(frame_count, 1))
    return {
        "fRMS": None,
        "fRMS_bkg": None,
        "deltafRMS": None,
        "velocity_map": velocity_dataset,
        "retinal_vessel_velocity": velocity_dataset,
        "moment0_avg": (averages["moment0"] / divisor).astype(np.float32),
        "velocity_map_avg": (averages["velocity"] / divisor).astype(np.float32),
        "fRMS_avg": (averages["fRMS"] / divisor).astype(np.float32),
        "fRMS_bkg_avg": (averages["fRMS_bkg"] / divisor).astype(np.float32),
        "deltafRMS_avg": (averages["deltafRMS"] / divisor).astype(np.float32),
        "velocity_section_mask": section_mask,
        "velocity_section_geometry": (
            "optic_disc_relative"
            if _has_optic_disc_geometry(optic_disc_width, optic_disc_height)
            else "frame_relative_fallback"
        ),
        "retinal_artery_velocity_signal": signals["artery_velocity"],
        "retinal_vein_velocity_signal": signals["vein_velocity"],
        "retinal_artery_fRMS_signal": signals["artery_fRMS"],
        "retinal_vein_fRMS_signal": signals["vein_fRMS"],
        "retinal_artery_fRMS_bkg_signal": signals["artery_fRMS_bkg"],
        "retinal_vein_fRMS_bkg_signal": signals["vein_fRMS_bkg"],
        "retinal_vessel_fRMS_bkg_signal": signals["vessel_fRMS_bkg"],
        "retinal_artery_deltafRMS_signal": signals["artery_deltafRMS"],
        "retinal_vein_deltafRMS_signal": signals["vein_deltafRMS"],
    }


def _has_optic_disc_geometry(width, height) -> bool:
    for value in (width, height):
        if value is None:
            return False
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.isfinite(array[0]) or array[0] <= 0:
            return False
    return True


def _read_moment_chunk(volume, frame_slice: slice) -> np.ndarray:
    return np.asarray(volume[frame_slice], dtype=np.float32)


def _inpaint_frame_batch(
    frames: np.ndarray,
    mask: np.ndarray,
    inpaint,
) -> np.ndarray:
    """Inpaint all frames as channels so the sparse system is built once."""
    source = np.asarray(frames, dtype=np.float32)
    channels_last = np.moveaxis(source, 0, -1)
    result = inpaint.inpaint_biharmonic(
        channels_last,
        np.asarray(mask, dtype=bool),
        channel_axis=-1,
    )
    inpainted = np.moveaxis(result, -1, 0).astype(np.float32, copy=False)
    square_safe_limit = np.float32(np.sqrt(np.finfo(np.float32).max))
    if np.all(np.isfinite(inpainted)) and np.all(
        np.abs(inpainted) <= square_safe_limit
    ):
        return inpainted
    return _bounded_inpaint_result(inpainted, source, mask)


def _bounded_inpaint_result(
    inpainted: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Keep each inpainted frame within its finite background range."""
    background = np.asarray(source, dtype=np.float32)[:, ~np.asarray(mask, dtype=bool)]
    finite = np.isfinite(background)
    if background.shape[1] == 0 or np.any(~np.any(finite, axis=1)):
        raise ValueError("Background inpainting requires a finite unmasked pixel per frame.")

    finite_background = np.where(finite, background, np.nan)
    lower = np.nanmin(finite_background, axis=1)
    upper = np.nanmax(finite_background, axis=1)
    fallback = np.nanmean(finite_background, axis=1, dtype=np.float32)
    bounded = np.where(
        np.isfinite(inpainted),
        inpainted,
        fallback[:, None, None],
    )
    return np.clip(
        bounded,
        lower[:, None, None],
        upper[:, None, None],
    ).astype(np.float32, copy=False)


def _signed_rms_difference(
    f_rms: np.ndarray,
    f_rms_background: np.ndarray,
) -> np.ndarray:
    """Return signed root-square difference without float32 square overflow.
    Reason: Encountered overflow.
    """
    foreground64 = np.asarray(f_rms, dtype=np.float64)
    background64 = np.asarray(f_rms_background, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        difference64 = np.square(foreground64) - np.square(background64)
        delta64 = np.sign(difference64) * np.sqrt(np.abs(difference64))
    return delta64.astype(np.float32, copy=False)


def _skimage_dependencies():
    try:
        from skimage.morphology import disk
        from skimage.restoration import inpaint
    except ModuleNotFoundError as exc:
        raise ImportError(
            "DopplerView velocity estimation requires scikit-image."
        ) from exc
    return disk, inpaint


def _dilated_mask(vessel_mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    mask = np.asarray(vessel_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"vessel_mask must be 2-D for dilation, got {mask.shape}.")
    return ndi.binary_dilation(mask, structure=np.asarray(footprint, dtype=bool))


def _masked_signal(velocity_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = velocity_map[:, np.asarray(mask, dtype=bool)]
    if not np.any(np.isfinite(selected)):
        return np.full((velocity_map.shape[0],), np.nan, dtype=np.float32)
    return np.nanmean(selected, axis=1, dtype=np.float32).astype(
        np.float32,
        copy=False,
    )
