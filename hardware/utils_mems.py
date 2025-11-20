"""MEMS utility functions. (using calibrated angle map)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import math


class AngleConversion(nn.Module):
    """
    Convert angle units:
      - MEMS step to angles in radians,
      - angles in radians to normalized angle [-1, 1],
      - normalized angle to MEMS step.

    Args:
        angle_map_path (str): Path to the angle map file.
    Returns:
        angle_rad: Angles in radians.
        angle_normalized: Normalized angles in [-1, 1].
        angle_step: MEMS step.
    """

    def __init__(self, angle_map_path=None):
        super().__init__()
        if angle_map_path is not None:
            data_dict = sio.loadmat(angle_map_path)
            self.y_max = torch.tensor(data_dict["y_max"]).float()
            self.y_min = torch.tensor(data_dict["y_min"]).float()
            self.x_max = torch.tensor(data_dict["x_max"]).float()
            self.x_min = torch.tensor(data_dict["x_min"]).float()
            self.angle_x = torch.tensor(data_dict["angle_x"]).float().repeat(1, 1, 1, 1)
            self.angle_y = torch.tensor(data_dict["angle_y"]).float().repeat(1, 1, 1, 1)
            self.a_x = torch.tensor(data_dict["a_x"]).float()
            self.a_y = torch.tensor(data_dict["a_y"]).float()
            self.b_x = torch.tensor(data_dict["b_x"]).float()
            self.b_y = torch.tensor(data_dict["b_y"]).float()

        else:
            self.y_max = torch.tensor([0.05]).float()
            self.y_min = torch.tensor([-0.05]).float()
            self.x_max = torch.tensor([0.05]).float()
            self.x_min = torch.tensor([-0.05]).float()
            self.angle_x = torch.tensor([[0.00, 0.00]]).float().repeat(1, 1, 1, 1)
            self.angle_y = torch.tensor([[0.00, 0.00]]).float().repeat(1, 1, 1, 1)
            self.a_x = torch.tensor([0.00]).float()
            self.a_y = torch.tensor([0.00]).float()

    def forward(self, angle_step, dev=None):
        if angle_step is None:
            return None, None

        if not torch.is_tensor(angle_step):
            angle_step = torch.tensor(angle_step).to(self.angle_x.device)

        # Normalize angle_step to [-1, 1]
        angle_normalized = self.normalized_angle(angle_step)
        angle_normalized = angle_normalized.reshape(-1, 1, 1, 2)

        ang_x_rad = F.grid_sample(
            self.angle_x, angle_normalized, align_corners=True, padding_mode="border"
        ).squeeze()
        ang_y_rad = F.grid_sample(
            self.angle_y, angle_normalized, align_corners=True, padding_mode="border"
        ).squeeze()

        ang_y_rad = ang_y_rad * math.pi / 180
        ang_x_rad = ang_x_rad * math.pi / 180

        if dev is not None:
            ang_y_rad = ang_y_rad.to(dev)
            ang_x_rad = ang_x_rad.to(dev)
            angle_normalized = angle_normalized.to(dev)

        return torch.stack((ang_y_rad, ang_x_rad), dim=-1)

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        self.y_max = self.y_max.to(*args, **kwargs)
        self.y_min = self.y_min.to(*args, **kwargs)
        self.x_max = self.x_max.to(*args, **kwargs)
        self.x_min = self.x_min.to(*args, **kwargs)
        self.angle_x = self.angle_x.to(*args, **kwargs)
        self.angle_y = self.angle_y.to(*args, **kwargs)
        self.a_x = self.a_x.to(*args, **kwargs)
        self.a_y = self.a_y.to(*args, **kwargs)
        self.b_x = self.b_x.to(*args, **kwargs)
        self.b_y = self.b_y.to(*args, **kwargs)
        return slf

    def normalized_angle(self, angle_rad):
        angle_normalized = torch.zeros_like(angle_rad)
        angle_normalized[..., 0] = (
            ((angle_rad[..., 0] - self.y_min) / (self.y_max - self.y_min)) * 2 - 1
        )
        angle_normalized[..., 1] = (
            ((angle_rad[..., 1] - self.x_min) / (self.x_max - self.x_min)) * 2 - 1
        )
        return angle_normalized

