from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import zoom


def ensure_volume_channel_first(fmri: np.ndarray) -> np.ndarray:
	"""Normalize fMRI volumes to [C, H, W, D, T]."""
	if fmri.ndim == 4:
		return np.expand_dims(fmri, axis=0)
	if fmri.ndim == 5:
		return fmri
	raise ValueError(f"fMRI volume must be [H,W,D,T] or [C,H,W,D,T], got {fmri.shape}")


def _center_pad_or_crop_axis(array: np.ndarray, axis: int, target_size: int, pad_value: float) -> np.ndarray:
	current_size = array.shape[axis]
	if current_size == target_size:
		return array

	if current_size > target_size:
		start = (current_size - target_size) // 2
		end = start + target_size
		slices = [slice(None)] * array.ndim
		slices[axis] = slice(start, end)
		return array[tuple(slices)]

	pad_before = (target_size - current_size) // 2
	pad_after = target_size - current_size - pad_before
	pad_width = [(0, 0)] * array.ndim
	pad_width[axis] = (pad_before, pad_after)
	return np.pad(array, pad_width=pad_width, mode="constant", constant_values=pad_value)


def center_pad_or_crop_volume(fmri: np.ndarray, target_shape: Sequence[int], pad_value: float = 0.0) -> np.ndarray:
	"""Center crop or zero-pad a [C,H,W,D,T] volume to the target shape."""
	target_shape = tuple(int(item) for item in target_shape)
	if fmri.shape[1:] == target_shape:
		return fmri.astype(np.float32, copy=False)

	output = fmri
	for axis, target_size in enumerate(target_shape, start=1):
		output = _center_pad_or_crop_axis(output, axis=axis, target_size=int(target_size), pad_value=pad_value)
	return output.astype(np.float32)


def interpolate_volume(fmri: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
	"""Resize a [C,H,W,D,T] volume with linear interpolation on mismatched axes."""
	target_shape = tuple(int(item) for item in target_shape)
	channels, height, width, depth, timepoints = fmri.shape
	zoom_factors = (
		1.0,
		target_shape[0] / max(height, 1),
		target_shape[1] / max(width, 1),
		target_shape[2] / max(depth, 1),
		target_shape[3] / max(timepoints, 1),
	)
	return zoom(fmri, zoom=zoom_factors, order=1).astype(np.float32)


def resize_volume_by_strategy(
	fmri: np.ndarray,
	target_shape: Sequence[int] | None,
	spatial_strategy: str = "none",
	temporal_strategy: str = "none",
	pad_value: float = 0.0,
) -> np.ndarray:
	"""Resize a [C,H,W,D,T] volume according to configured spatial and temporal strategies."""
	if target_shape is None:
		return fmri.astype(np.float32)

	target_shape = tuple(int(item) for item in target_shape)
	if len(target_shape) != 4:
		raise ValueError(f"fmri_target_shape must contain 4 integers, got {target_shape}")

	current_shape = fmri.shape[1:]
	if current_shape == target_shape:
		return fmri.astype(np.float32, copy=False)

	spatial_target = target_shape[:3]
	time_target = target_shape[3]
	output = fmri

	if spatial_strategy not in {"none", "pad_or_crop", "interpolate"}:
		raise ValueError(f"Unsupported fmri_spatial_strategy: {spatial_strategy}")
	if temporal_strategy not in {"none", "pad_or_crop", "interpolate"}:
		raise ValueError(f"Unsupported fmri_temporal_strategy: {temporal_strategy}")

	spatial_mismatch = current_shape[:3] != spatial_target
	temporal_mismatch = current_shape[3] != time_target

	if spatial_mismatch and spatial_strategy == "interpolate":
		output = interpolate_volume(
			output,
			target_shape=(spatial_target[0], spatial_target[1], spatial_target[2], output.shape[-1]),
		)
	if temporal_mismatch and temporal_strategy == "interpolate":
		output = interpolate_volume(
			output,
			target_shape=(output.shape[1], output.shape[2], output.shape[3], time_target),
		)

	if spatial_mismatch and spatial_strategy == "pad_or_crop":
		output = center_pad_or_crop_volume(
			output,
			target_shape=(spatial_target[0], spatial_target[1], spatial_target[2], output.shape[-1]),
			pad_value=pad_value,
		)
	if temporal_mismatch and temporal_strategy == "pad_or_crop":
		output = center_pad_or_crop_volume(
			output,
			target_shape=(output.shape[1], output.shape[2], output.shape[3], time_target),
			pad_value=pad_value,
		)

	return output.astype(np.float32)


def zscore_volume(fmri: np.ndarray, nonzero_only: bool = True) -> np.ndarray:
	"""Apply z-score normalization to a [C,H,W,D,T] volume."""
	if nonzero_only:
		mask = np.abs(fmri) > 1e-8
		if not np.any(mask):
			return fmri.astype(np.float32)
		mean = float(fmri[mask].mean())
		std = float(fmri[mask].std()) + 1e-6
		output = np.array(fmri, copy=True)
		output[mask] = (output[mask] - mean) / std
		output[~mask] = 0.0
		return output.astype(np.float32)

	mean = float(fmri.mean())
	std = float(fmri.std()) + 1e-6
	return ((fmri - mean) / std).astype(np.float32)
