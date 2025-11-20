"""
Utility functions for the CGH / model training pipeline.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Article: 
S. Choi, C. Jang, D. Lanman, G. Wetzstein, 
"Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue",
Nature Photonics, 2025

"""

import math
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import omegaconf


def im2float(im, dtype=np.float32, im_max=None):
    """Convert uint16 or uint8 image to float32, with range scaled to 0-1.

    :param im: Image array.
    :param dtype: Output float dtype, default np.float32.
    :param im_max: Optional maximum value for normalization.
    :return: Float image with range [0,1].
    """
    if issubclass(im.dtype.type, np.floating):
        return im.astype(dtype)
    elif issubclass(im.dtype.type, np.integer):
        if im_max is not None:
            return im / im_max
        else:
            return im / dtype(np.iinfo(im.dtype).max)
    else:
        raise ValueError(f"Unsupported data type {im.dtype}")


def srgb_gamma2lin(im_in):
    """Convert from sRGB to linear color space."""
    thresh = 0.04045
    if torch.is_tensor(im_in):
        low_val = im_in <= thresh
        im_out = torch.zeros_like(im_in)
        im_out[low_val] = 25 / 323 * im_in[low_val]
        im_out[~low_val] = ((200 * im_in[~low_val] + 11) / 211) ** (12 / 5)
    else:
        im_out = np.where(
            im_in <= thresh,
            im_in / 12.92,
            ((im_in + 0.055) / 1.055) ** (12 / 5),
        )
    return im_out


def srgb_lin2gamma(im_in):
    """Convert from linear to sRGB color space."""
    thresh = 0.0031308
    im_out = np.where(
        im_in <= thresh,
        12.92 * im_in,
        1.055 * (im_in ** (1 / 2.4)) - 0.055,
    )
    return im_out


def pad_image(
    field,
    target_shape,
    pytorch=True,
    stacked_complex=False,
    padval=0,
    mode="constant",
    lf=False
):
    """
    Pads a 2D field up to target_shape in size.

    Padding is done such that when used with crop_image(), odd/even dims are
    handled correctly to properly undo the padding.

    Args:
        field: The field to be padded. May have leading dimensions.
        target_shape: The 2D target output dimensions. No padding if shape is smaller.
        pytorch: Use torch functions if True, otherwise numpy.
        stacked_complex: For pytorch=True, indicates real+imag last dimension.
        padval: Value to pad with.
        mode: Padding mode for numpy or torch.
        lf: Whether the data is light field (pad over [-4:-2] dims).
    """
    if lf:
        size_diff = np.array(target_shape) - np.array(field.shape[-4:-2])
        odd_dim = np.array(field.shape[-4:-2]) % 2
    else:
        if pytorch:
            if stacked_complex:
                size_diff = np.array(target_shape) - np.array(field.shape[-3:-1])
                odd_dim = np.array(field.shape[-3:-1]) % 2
            else:
                size_diff = np.array(target_shape) - np.array(field.shape[-2:])
                odd_dim = np.array(field.shape[-2:]) % 2
        else:
            size_diff = np.array(target_shape) - np.array(field.shape[-2:])
            odd_dim = np.array(field.shape[-2:]) % 2

    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2

        if pytorch:
            pad_axes = [
                int(p)
                for tple in zip(pad_front[::-1], pad_end[::-1])
                for p in tple
            ]
            if lf:
                original_shape = field.shape
                padded = nn.functional.pad(
                    field.permute(0, 1, 4, 5, 2, 3).view(-1, 1, *original_shape[-4:-2]),
                    pad_axes,
                    mode=mode,
                    value=padval,
                ).reshape(*original_shape[:2], *original_shape[-2:], *target_shape)
                return padded.permute(0, 1, 4, 5, 2, 3)
            else:
                if stacked_complex:
                    # pad_stacked_complex must be defined elsewhere
                    return pad_stacked_complex(
                        field, pad_axes, mode=mode, padval=padval
                    )
                else:
                    return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
        else:
            leading_dims = field.ndim - 2  # pad only the last two dims
            pad_front_full = pad_front
            pad_end_full = pad_end
            if leading_dims > 0:
                pad_front_full = np.concatenate(([0] * leading_dims, pad_front))
                pad_end_full = np.concatenate(([0] * leading_dims, pad_end))
            return np.pad(
                field,
                tuple(zip(pad_front_full, pad_end_full)),
                mode,
                constant_values=padval,
            )
    else:
        return field


def crop_image(field, target_shape, pytorch=True, stacked_complex=False, lf=False):
    """
    Crops a 2D field; see pad_image() for details. No cropping if target_shape is already smaller than field.
    """
    if target_shape is None:
        return field

    if lf:
        size_diff = np.array(field.shape[-4:-2]) - np.array(target_shape)
        odd_dim = np.array(field.shape[-4:-2]) % 2
    else:
        if pytorch:
            if stacked_complex:
                size_diff = np.array(field.shape[-3:-1]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-3:-1]) % 2
            else:
                size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
                odd_dim = np.array(field.shape[-2:]) % 2
        else:
            size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
            odd_dim = np.array(field.shape[-2:]) % 2

    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2

        crop_slices = [slice(int(f), int(-e) if e else None) for f, e in zip(crop_front, crop_end)]
        if lf:
            return field[(..., *crop_slices, slice(None), slice(None))]
        else:
            if pytorch and stacked_complex:
                return field[(..., *crop_slices, slice(None))]
            else:
                return field[(..., *crop_slices)]
    else:
        return field


def steered_angle_to_pupil_position(f, steered_angle):
    """Convert steered_angle to pupil-plane position using eyepiece focal length."""
    if steered_angle is None:
        return (0.0, 0.0)
    return tuple(np.tan(a) * f for a in steered_angle)


def get_exit_pupil_size(cfg_cgh, cfg_model):
    """Compute exit pupil size in meters."""
    f = cfg_cgh.focal_length_eyepiece
    wvl = cfg_model.wavelengths[cfg_model.channel]
    p = cfg_model.pixel_pitch
    return f * wvl / p


def normalize_steered_angle_input(steered_angle):
    """
    Ensure that steered_angle is always a list of angles.
    If it is a single (2,) vector/tuple/list, wraps it in a list.
    """
    if steered_angle is None:
        steered_angle = [(0.0, 0.0)]
    elif (
        isinstance(
            steered_angle,
            (list, tuple, omegaconf.listconfig.ListConfig),
        )
        and len(steered_angle) == 2
        and all(isinstance(x, (int, float)) for x in steered_angle)
    ):
        steered_angle = [tuple(steered_angle)]
    return steered_angle


def get_start_and_end_position_idx(steered_angle,
                                   cfg_cgh, cfg_model,
                                   return_round=True,
                                   return_support_indices=False):
    """
    Returns the start and end indices of the light field subaperture grid,
    based on the SLM etendue, steered angle, and light field rendering configuration.
    This defines the angular bounds over which the synthetic aperture is constructed for rendering.
    """
    # Initialize start/end indices of the LF subap grid
    start_lf_idx = (float("inf"), float("inf"))
    end_lf_idx = (float("-inf"), float("-inf"))
    interval_lf = cfg_cgh.lf_data.interval_lf  # (meters between LF views)
    support_indices = [] if return_support_indices else None

    for angle in steered_angle:
        # Center pupil pos in meters
        center_pupil_position = steered_angle_to_pupil_position(
            cfg_cgh.focal_length_eyepiece, angle
        )
        # Exit pupil size (meters)
        exit_pupil_size = get_exit_pupil_size(cfg_cgh, cfg_model)

        # Compute start/end spatial positions (meters)
        start_position = tuple(
            c - exit_pupil_size / 2 - ilf / 2
            for c, ilf in zip(center_pupil_position, interval_lf)
        )
        end_position = tuple(
            c + exit_pupil_size / 2 + ilf / 2
            for c, ilf in zip(center_pupil_position, interval_lf)
        )

        # Central LF index
        central_lf = tuple((c - 1) // 2 for c in cfg_cgh.lf_data.total_num_lf_views)

        # Indices of top-left, bottom-right LF views
        start_position_idx = tuple(
            s / ilf + c for s, ilf, c in zip(start_position, interval_lf, central_lf)
        )
        end_position_idx = tuple(
            s / ilf + c for s, ilf, c in zip(end_position, interval_lf, central_lf)
        )

        if return_round:
            start_position_idx = tuple(np.ceil(s).astype(int) for s in start_position_idx)
            end_position_idx = tuple(np.floor(e).astype(int) for e in end_position_idx)

        # Elementwise min/max to grow bounds across all angles
        start_lf_idx = tuple(
            min(start_position_idx[i], start_lf_idx[i]) for i in range(len(start_position_idx))
        )
        end_lf_idx = tuple(
            max(end_position_idx[i], end_lf_idx[i]) for i in range(len(end_position_idx))
        )

        if return_support_indices and support_indices is not None:
            # Add all indices within the box from start_position_idx (top left) to end_position_idx (bottom right)
            for i in range(start_position_idx[0], end_position_idx[0] + 1):
                for j in range(start_position_idx[1], end_position_idx[1] + 1):
                    idx = (i, j)
                    if idx not in support_indices:
                        support_indices.append(idx)

    if return_support_indices:
        return start_lf_idx, end_lf_idx, support_indices
    else:
        return start_lf_idx, end_lf_idx


def stacked_to_complex(input_field):
    n = input_field.shape[1] // 2
    return input_field[:, :n, ...] + 1j * input_field[:, n:, ...]


def normalize_angle(angle):
    return ((angle + np.pi) % (2 * np.pi)) / (2 * np.pi)


def intensity_blur(intensity, blur):
    """
    Blurs intensity of specified amplitude.

    Parameters
    ----------
    intensity : torch.Tensor
        The intensity tensor, usually the squared amplitude.
    blur : tuple or list of int
        The blur size as a 2-tuple indicating desired loss in resolution.

    Returns
    -------
    torch.Tensor
        Blurred intensity tensor.
    """
    # intensity = amplitude**2

    # Construct Box Blur kernel (Other kernels can be explored)
    channel_count = intensity.shape[1]
    kernel = torch.ones(1, 1, blur[0], blur[1], device=intensity.device)
    kernel = kernel / kernel.sum()
    kernel = kernel.repeat(channel_count, 1, 1, 1)
    pad_count = tuple([b // 2 for b in blur])
    conv_filter = nn.Conv2d(
        in_channels=channel_count,
        out_channels=channel_count,
        kernel_size=(blur[0], blur[1]),
        padding=pad_count,
        padding_mode="reflect",
        groups=channel_count,
        bias=False
    ).to(intensity.device)
    conv_filter.weight.data = kernel
    conv_filter.weight.requires_grad = False
    intensity_out = conv_filter(intensity)

    # if blur % 2 == 0:
    #     Drop last row/col to get original image size
    #     intensity_out = intensity_out[...,:-1,:-1]

    return intensity_out

def interpolate_complex(input_field, size, mode='bilinear', align_corners=False, coord='polar'):
    """
    Interpolates a complex-valued field.
    """
    size = list(size)
    if not torch.is_complex(input_field):
        return torch.nn.functional.interpolate(input_field, size=size, mode=mode, align_corners=align_corners)
    else:
        if coord == 'polar':
            a = torch.nn.functional.interpolate(input_field.abs(), size=size, mode=mode, align_corners=align_corners)
            angle = torch.nn.functional.interpolate(input_field.angle(), size=size, mode=mode, align_corners=align_corners)
            return a * torch.exp(1j * angle)
        elif coord == 'cartesian':
            return torch.nn.functional.interpolate(input_field.real, size=size, mode=mode, align_corners=align_corners) + 1j * torch.nn.functional.interpolate(input_field.imag, size=size, mode=mode, align_corners=align_corners)
        else:
            raise ValueError(f"Invalid coordinate system: {coord}")