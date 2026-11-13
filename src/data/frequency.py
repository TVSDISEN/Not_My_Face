"""Frequency-domain transforms used by the dual-stream model."""

from __future__ import annotations

import numpy as np


def compute_fft_spectrogram(image: np.ndarray) -> np.ndarray:
    """Return a normalized 2D log-magnitude FFT map as float32.

    Args:
        image: A non-empty grayscale (H, W) or RGB (H, W, 3/4) image.
            RGB channel ordering is expected for color inputs.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        grayscale = array.astype(np.float32, copy=False)
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3].astype(np.float32, copy=False)
        grayscale = (
            0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        )
    else:
        raise ValueError("Expected a grayscale or RGB image with shape (H, W[, 3/4]).")

    if grayscale.size == 0:
        raise ValueError("Cannot transform an empty image.")
    if not np.isfinite(grayscale).all():
        raise ValueError("Image values must be finite.")

    spectrum = np.fft.fftshift(np.fft.fft2(grayscale))
    log_magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    nonzero = log_magnitude[log_magnitude > 0]
    if nonzero.size == 0:
        return np.zeros(log_magnitude.shape, dtype=np.float32)

    scale = float(np.percentile(nonzero, 99.5))
    if scale <= 0:
        return np.zeros(log_magnitude.shape, dtype=np.float32)
    return np.clip(log_magnitude / scale, 0.0, 1.0).astype(np.float32)


def frequency_map_to_rgb(frequency_map: np.ndarray) -> np.ndarray:
    """Quantize a normalized 2D frequency map into the model's 3-channel input."""
    array = np.asarray(frequency_map, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Expected a non-empty 2D frequency map.")
    if not np.isfinite(array).all():
        raise ValueError("Frequency map values must be finite.")
    quantized = np.rint(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(quantized[..., np.newaxis], repeats=3, axis=2)
