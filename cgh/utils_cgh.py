"""
Utility functions for the CGH algorithms.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu).

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC). In a nutshell:
    - The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    - The material is provided as-is, with no warranties whatsoever.
    - If you publish any code, data, or scientific work based on this, please cite our work.

Article:
S. Choi, C. Jang, D. Lanman, G. Wetzstein,
"Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue",
Nature Photonics, 2025
"""

import math
import os
import random
import logging

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import omegaconf

import utils.utils as utils

CHANNEL_STR = {
    0: "red",
    1: "green",
    2: "blue",
}


def check_cgh_config(cfg):
    """
    Check the configuration parameters for the CGH algorithm.
    """
    # If the steered_angle is set as a string (e.g., '[[0.0,0.0],[0.0,0.1]]'), parse it to a list of lists.
    if hasattr(cfg.cgh, "steered_angle") and cfg.cgh.steered_angle is not None:
        if isinstance(cfg.cgh.steered_angle, str):
            import ast
            cfg.cgh.steered_angle = ast.literal_eval(cfg.cgh.steered_angle)

    assert (
        cfg.cgh.steered_angle is None
        or isinstance(cfg.cgh.steered_angle, (list, omegaconf.listconfig.ListConfig))
    )
    cfg.cgh.steered_angle = utils.normalize_steered_angle_input(cfg.cgh.steered_angle)

    # Match the length of the steered angle to the rank.
    if cfg.cgh.steered_angle is not None:
        if len(cfg.cgh.steered_angle) == 1:
            cfg.cgh.steered_angle = [cfg.cgh.steered_angle[0]] * cfg.cgh.rank
        assert (
            len(cfg.cgh.steered_angle) == cfg.cgh.rank
        ), "steered_angle must be a list of length rank"


def phase_encoding(phase, encoding_type="naive_8bits"):
    if encoding_type == "naive_8bits":
        phase = ((phase + math.pi) % (2 * math.pi)) / (2 * math.pi)
        phase = 1 - phase
        phase = (phase * 255).round()
        return phase
    else:
        raise ValueError(f"Encoding type {encoding_type} not supported")


def get_minimal_target_amp_per_steered_angle(cfg_cgh, cfg_model, target_amp, steered_angle):
    """
    Here we assume that
    """
    pass  # Function intentionally left as a stub.


def get_gt_amp_and_sa(cfg_cgh, cfg_model, target_amp, t_mb, steered_angle=None):
    """
    Get the ground truth amplitude and synthetic aperture for the given target amplitude and steered angle.
    """
    device = target_amp.device
    synthetic_aperture = None

    if cfg_cgh.data_type == "2d":
        assert len(target_amp.shape) == 4
        gt_amp = target_amp

    elif cfg_cgh.data_type in ["4d"]:
        assert len(target_amp.shape) == 6
        def sample_offset_dist(cfg_supervision, t):
            if cfg_supervision.name in ["3d"]:
                return cfg_supervision.offset_dists[t]
            elif cfg_supervision.name in ["sah"]:
                return random.choice(cfg_supervision.offset_dists)
            elif cfg_supervision.name in ["4d"]:
                return 0.0
            else:
                raise NotImplementedError(f"Supervision {cfg_supervision.name} not implemented")
        offset_dist = sample_offset_dist(cfg_cgh.supervision, t_mb)

        def sample_subapertures_idx(
            cfg_supervision,
            random_sampling=True,
            t_mb=None,
            full_lf_indices=None,
            prob_num_subapertures="reciprocal",
        ):
            # Pick a number of subapertures.
            if cfg_supervision.name == "4d":
                min_num_subapertures_to_sample = 1
                max_num_subapertures_to_sample = 1
            elif cfg_supervision.name == "3d":
                min_num_subapertures_to_sample = len(full_lf_indices)
                max_num_subapertures_to_sample = len(full_lf_indices)
            elif cfg_supervision.name == "sah":
                min_num_subapertures_to_sample = cfg_supervision.min_num_subapertures_to_sample
                max_num_subapertures_to_sample = cfg_supervision.max_num_subapertures_to_sample
            else:
                raise NotImplementedError(f"Supervision {cfg_supervision.name} not implemented")

            # Populate a list of tuples of (x, y), where 0 <= x, y < total_num_subapertures
            total_subapertures_idx = full_lf_indices

            # Sample the subapertures randomly or sequentially.
            if prob_num_subapertures == "reciprocal":
                # Create a probability distribution where P(X) ∝ 1/X.
                possible_values = list(
                    range(min_num_subapertures_to_sample, max_num_subapertures_to_sample + 1)
                )
                probs = np.array([1.0 / x for x in possible_values], dtype=np.float64)
                probs /= probs.sum()  # Normalize so sum == 1.
                num_subapertures = int(np.random.choice(possible_values, p=probs))
            elif prob_num_subapertures == "uniform":
                num_subapertures = random.randint(
                    min_num_subapertures_to_sample,
                    max_num_subapertures_to_sample,
                )
            else:
                raise NotImplementedError(
                    f"Probability of number of subapertures {prob_num_subapertures} not implemented"
                )

            if random_sampling:
                sampled_subapertures_idx = random.sample(total_subapertures_idx, num_subapertures)
            else:
                sampled_subapertures_idx = [total_subapertures_idx[t_mb]]

            return sampled_subapertures_idx

        if cfg_cgh.supervision.name == "4d":
            random_sampling = False  # Light field supervision. Go through all the light field views.
        else:
            random_sampling = True

        # Get indices of light field that can affect the steered wavefront.
        full_lf_indices = get_full_lf_indices(cfg_cgh, cfg_model, steered_angle)

        # Using those index pool, sample the subapertures.
        sampled_subapertures_idx = sample_subapertures_idx(
            cfg_cgh.supervision,
            random_sampling=random_sampling,
            t_mb=t_mb,
            full_lf_indices=full_lf_indices,
        )

        # The indicator function in Eq. (3) of the manuscript.
        synthetic_aperture = generate_synthetic_aperture(
            sampled_subapertures_idx,
            steered_angle,
            cfg_cgh,
            cfg_model,
            device=device,
        )
        synthetic_aperture = synthetic_aperture.reshape(
            len(synthetic_aperture), 1, *synthetic_aperture.shape[-2:]
        )

        # Shift LF based on offset dist,
        # and get views corresponding to the synthetic aperture,
        # and then add up incoherently.
        gt_amp = integral_over_synthetic_aperture(
            target_amp,
            offset_dist,
            steered_angle,
            sampled_subapertures_idx,
            cfg_cgh,
            cfg_model,
        )
    else:
        raise NotImplementedError(f"Data type {cfg_cgh.data_type} not implemented")

    return gt_amp, synthetic_aperture, offset_dist


def generate_synthetic_aperture(
    sampled_subapertures_idx,
    steered_angle,
    cfg_cgh,
    cfg_model,
    device=torch.device("cuda"),
):
    fourier_resolution = [2 * p for p in cfg_model.slm_resolution]

    # Create an empty synthetic aperture mask (Fourier domain)
    mask = torch.zeros(
        (len(steered_angle), 1, *fourier_resolution),
        dtype=torch.float32,
        device=device,
    )

    # Determine the size of each subaperture segment
    eyebox_size = utils.get_exit_pupil_size(cfg_cgh, cfg_model)
    size_of_subaperture_in_pixels = [
        ilf / eyebox_size * fr
        for ilf, fr in zip(cfg_cgh.lf_data.interval_lf, fourier_resolution)
    ]
    start_position_idx, _ = utils.get_start_and_end_position_idx(steered_angle, cfg_cgh, cfg_model)
    offset_y = start_position_idx[0] - cfg_cgh.lf_data.total_num_lf_views[0] // 2
    offset_x = start_position_idx[1] - cfg_cgh.lf_data.total_num_lf_views[1] // 2

    # TODO: consider accelerating this loop.
    for i_m, angle in enumerate(steered_angle):
        if angle is not None:
            f = cfg_cgh.focal_length_eyepiece
            steered_center_in_pixels = [
                f * math.tan(theta) / eyebox_size * fr
                for theta, fr in zip(angle, fourier_resolution)
            ]
            center_u = -steered_center_in_pixels[0]
            center_v = -steered_center_in_pixels[1]
        else:
            center_u = 0.0
            center_v = 0.0

        # For every sampled (i, j) index, fill in the corresponding mask region.
        if len(sampled_subapertures_idx) > 0:
            idx_y = torch.tensor([idx[0] for idx in sampled_subapertures_idx], device=device)
            idx_x = torch.tensor([idx[1] for idx in sampled_subapertures_idx], device=device)

            uu = center_u + (idx_y + offset_y) * size_of_subaperture_in_pixels[0] + fourier_resolution[0] / 2
            vv = center_v + (idx_x + offset_x) * size_of_subaperture_in_pixels[1] + fourier_resolution[1] / 2

            y0 = uu - size_of_subaperture_in_pixels[0] / 2
            x0 = vv - size_of_subaperture_in_pixels[1] / 2
            y1 = uu + size_of_subaperture_in_pixels[0] / 2
            x1 = vv + size_of_subaperture_in_pixels[1] / 2

            # y0, x0, y1, x1 are tensors (may be 1D), so convert to int tensors and clamp to valid limits.
            y0 = torch.clamp(torch.floor(y0).long(), min=0, max=mask.shape[-2])
            x0 = torch.clamp(torch.floor(x0).long(), min=0, max=mask.shape[-1])
            y1 = torch.clamp(torch.ceil(y1).long(), min=0, max=mask.shape[-2])
            x1 = torch.clamp(torch.ceil(x1).long(), min=0, max=mask.shape[-1])

            # For each segment, fill with 1.0.
            for i in range(len(idx_y)):
                mask[i_m, :, y0[i]:y1[i], x0[i]:x1[i]] = 1.0

    return mask


def integral_over_synthetic_aperture(
    target_amp,
    offset_dist,
    steered_angle,
    sampled_subapertures_idx,
    cfg_cgh,
    cfg_model,
):
    # Shift and add light field.
    target_amp_shifted = shift_and_add_lf(
        target_amp,
        offset_dist,
        steered_angle=steered_angle,
        cfg_cgh=cfg_cgh,
        cfg_model=cfg_model,
    )  # (1, 1, H, W, M, N)

    # Calculate a tensor of weights for each view (in intensity domain).
    view_weights = calculate_view_weights(
        target_amp.shape,
        sampled_subapertures_idx,
        steered_angle=steered_angle,
        cfg_cgh=cfg_cgh,
        cfg_model=cfg_model,
        wigner_shearing=False,
        device=target_amp.device,
    )  # shape of (1, 1, 1, 1, M, N)

    target_focal_slice = ((target_amp_shifted ** 2) * view_weights).mean(dim=[-2, -1]).sqrt()  # (1, 1, H, W)
    return target_focal_slice


def get_scaling_factor(cfg, recon_amp, gt_amp):
    if cfg.scaling_factor == "min_mse":
        s = (recon_amp * gt_amp).mean() / (recon_amp ** 2).mean()
    elif isinstance(cfg.scaling_factor, float):
        s = cfg.scaling_factor
    else:
        raise NotImplementedError(f"Scaling factor {cfg.scaling_factor} not implemented")
    return s
    # If other scaling_factor, 's' will be undefined; could raise or handle differently.


def shift_and_add_lf(
    light_field,
    z,
    steered_angle,
    cfg_cgh,
    cfg_model,
    return_fs=False,
):
    """
    Implement the shift-and-add algorithm.
    """
    if abs(z) < 1e-6:
        # If z is 0, return the light field as is.
        return light_field

    original_shape = light_field.shape
    f = cfg_cgh.focal_length_eyepiece
    delta_u, delta_v = tuple(interval_lf / f for interval_lf in cfg_cgh.lf_data.interval_lf)
    total_num_lf_views = cfg_cgh.lf_data.total_num_lf_views

    start_position_idx, _ = utils.get_start_and_end_position_idx(steered_angle, cfg_cgh, cfg_model)
    u_idx_offset = start_position_idx[0] - total_num_lf_views[0] // 2  # y offset
    v_idx_offset = start_position_idx[1] - total_num_lf_views[1] // 2  # x offset

    pixel_pitch = cfg_model.pixel_pitch
    thetas = []
    focal_stack = torch.zeros_like(light_field[..., 0, 0])
    for u in range(light_field.shape[-2]):
        for v in range(light_field.shape[-1]):
            uu = u + u_idx_offset  # y
            vv = v + v_idx_offset  # x

            # Shift in meter space.
            x_shift = z * math.tan(vv * delta_v) / (pixel_pitch * focal_stack.shape[-1] / 2)
            y_shift = z * math.tan(uu * delta_u) / (pixel_pitch * focal_stack.shape[-2] / 2)

            theta = torch.tensor(
                [[1.0, 0.0, -x_shift], [0.0, 1.0, -y_shift]]
            ).to(light_field.device).unsqueeze(0)
            thetas.append(theta)

    thetas = torch.cat(thetas, 0)
    light_field = (
        light_field.view(*light_field.shape[:4], -1)
        .permute(4, 0, 1, 2, 3)
        .squeeze(1)
    )
    aff_grid = F.affine_grid(thetas, light_field.shape)
    focal_stack = F.grid_sample(light_field, aff_grid)

    if return_fs:
        # Returning intensity of focal stack.
        return focal_stack.mean(0, keepdims=True)
    else:
        shift_add_lf = (
            focal_stack.unsqueeze(-1)
            .permute(4, 1, 2, 3, 0)
            .reshape(*original_shape)
        )
        # Return intensity as well.
        return shift_add_lf


def calculate_view_weights(
    weights_shape,
    sampled_subapertures_idx,
    steered_angle,
    cfg_cgh,
    cfg_model,
    wigner_shearing=False,
    device=torch.device("cuda"),
):
    """
    Calculate the weights for each sampled subaperture.

    The weights are in intensity domain: if sampled, the weights are 1.0, otherwise, 0.0.
    If the view is clipped in the pupil plane, it is attenuated by the ratio of area.

    If Wigner shearing is used, it is additionally attenuated by the envelope from wave propagation.
    """
    # shape: (1, 1, 1, 1, M, N)
    weights = torch.zeros(1, 1, 1, 1, *weights_shape[-2:], device=device)

    total_num_lf_views = cfg_cgh.lf_data.total_num_lf_views
    start_position_idx, _ = utils.get_start_and_end_position_idx(steered_angle, cfg_cgh, cfg_model)
    u_idx_offset = start_position_idx[0] - total_num_lf_views[0] // 2
    v_idx_offset = start_position_idx[1] - total_num_lf_views[1] // 2
    exit_pupil_size = utils.get_exit_pupil_size(cfg_cgh, cfg_model)

    system_min = (float("inf"), float("inf"))
    system_max = (float("-inf"), float("-inf"))
    for angle in steered_angle:
        center_pupil_position = utils.steered_angle_to_pupil_position(
            cfg_cgh.focal_length_eyepiece, angle
        )
        system_min = tuple(
            min(system_min[i], center_pupil_position[i] - exit_pupil_size / 2)
            for i in range(2)
        )
        system_max = tuple(
            max(system_max[i], center_pupil_position[i] + exit_pupil_size / 2)
            for i in range(2)
        )

    sublf_area = cfg_cgh.lf_data.interval_lf[0] * cfg_cgh.lf_data.interval_lf[1]

    for sampled_subaperture_idx in sampled_subapertures_idx:
        u = sampled_subaperture_idx[0] + u_idx_offset
        v = sampled_subaperture_idx[1] + v_idx_offset

        lf_min = (
            (u - 0.5) * cfg_cgh.lf_data.interval_lf[0],
            (v - 0.5) * cfg_cgh.lf_data.interval_lf[1],
        )
        lf_max = (
            (u + 0.5) * cfg_cgh.lf_data.interval_lf[0],
            (v + 0.5) * cfg_cgh.lf_data.interval_lf[1],
        )

        overlapped_area = get_overlapped_area(lf_min, lf_max, system_min, system_max)
        weight = overlapped_area / sublf_area
        weights[..., sampled_subaperture_idx[0], sampled_subaperture_idx[1]] = weight

    return weights


def get_overlapped_area(lf_min, lf_max, system_min, system_max):
    """
    Compute the overlapping area between two axis-aligned rectangles.
    Coordinates are in (y, x) order.

    Parameters:
        lf_min (tuple): (y_min, x_min) of the first rectangle
        lf_max (tuple): (y_max, x_max) of the first rectangle
        system_min (tuple): (y_min, x_min) of the second rectangle
        system_max (tuple): (y_max, x_max) of the second rectangle

    Returns:
        float: The overlapping area (0 if there is no overlap)
    """
    # Compute overlap boundaries (y first, x second)
    overlap_y_min = max(lf_min[0], system_min[0])
    overlap_x_min = max(lf_min[1], system_min[1])
    overlap_y_max = min(lf_max[0], system_max[0])
    overlap_x_max = min(lf_max[1], system_max[1])

    # Compute overlap dimensions
    overlap_height = max(0.0, overlap_y_max - overlap_y_min)
    overlap_width = max(0.0, overlap_x_max - overlap_x_min)

    # Area = height * width
    return overlap_height * overlap_width


def get_full_lf_indices(cfg_cgh, cfg_model, steered_angle, offset=True):
    """
    Get the indices of the light field that can affect the steered wavefront.
    """
    # start_position_idx, end_position_idx = utils.get_start_and_end_position_idx(steered_angle, cfg_cgh, cfg_model)
    start_position_idx, _, support_indices = utils.get_start_and_end_position_idx(steered_angle, cfg_cgh, cfg_model, return_support_indices=True)

    if offset:
        # Offset support_indices by start_position_idx
        support_indices = [(y - start_position_idx[0], x - start_position_idx[1]) for (y, x) in support_indices]
    return support_indices
