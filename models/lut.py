import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from models.monotonenorm import direct_norm, GroupSort, SigmaNet


def lipschitz_norm(module, power=3):
    """Apply Lipschitz (one-norm) constraint to a layer."""
    return direct_norm(
        module,
        "one",
        max_norm=1.2 ** (1 / power),
    )


class MonotonicMLPLUT(nn.Module):
    """
    Monotonic, Lipschitz-constrained neural network for LUT approximation.
    """

    def __init__(
        self,
        num_layers: int = 3,
        num_feats: int = 16,
        num_codes=(0, 0),
        lip_const: float = 1.2,
    ):
        super().__init__()
        num_input_ch = 1 + num_codes[0]
        self.num_codes = num_codes

        model1 = nn.Sequential(
            lipschitz_norm(nn.Linear(num_input_ch, num_feats), power=num_layers),
            GroupSort(2),
        )
        model2 = nn.Sequential(
            lipschitz_norm(
                nn.Linear(num_feats + num_codes[1], num_feats), power=num_layers
            ),
            GroupSort(2),
            lipschitz_norm(nn.Linear(num_feats, 1), power=num_layers),
        )
        self.model = SigmaNet(
            model1,
            model2,
            sigma=lip_const,
            monotone_constraints=[1] * (1 + num_codes[0]),
        )

    def forward(self, x, code_slm=None):
        x_orig_shape = x.shape
        code_slm_2 = None

        if code_slm is not None:
            if code_slm.shape[0] != x.shape[0]:
                code_slm = code_slm.repeat(
                    x.shape[0] // code_slm.shape[0], 1, 1, 1
                )
            code_slm_1 = code_slm[:, : self.num_codes[0], ...]
            code_slm_2 = (
                code_slm[:, -self.num_codes[1] :, ...]
                if self.num_codes[1] > 0
                else None
            )
            x = torch.cat((x, code_slm_1), 1)

        x = x.permute(1, 0, 2, 3)  # (C, B, M, N)
        x = (
            x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            .permute(1, 0)
        )  # (BMN, C)
        y = self.model(x, code_slm_2)
        y = y.reshape(-1, *x_orig_shape[2:]).unsqueeze(1)  # (B, 1, M, N)
        return y


class LUT(nn.Module):
    """
    Wrapper for phase nonlinearity look-up-table.
    """

    def __init__(
        self, lut_type: str = "monotone", num_layers=3, num_codes=0, lut_num_ch=16, **kwargs
    ):
        super().__init__()
        if lut_type == "monotone":
            self.lut_mlp = MonotonicMLPLUT(
                num_layers, lut_num_ch, num_codes=num_codes
            )
        elif lut_type.lower() == "none":
            self.lut_mlp = None
        else:
            self.lut_mlp = None  # fallback for unsupported types
        self.num_codes = num_codes

    def forward(self, x, code_slm=None):
        # Modularize to ensure input in [-pi, pi]
        if self.lut_mlp is not None:
            modular_x = (x + math.pi) % (2 * math.pi) - math.pi
            return self.lut_mlp(modular_x, code_slm)
        return x

    def visualize_tensorboard(
        self, logger, global_step, lut_code=None, *args, **kwargs
    ):
        add_lut_mlp_mean_var(
            logger,
            self,
            lut_code=lut_code,
            global_step=global_step,
            show_identity=True,
            num_x=256,
        )


def add_lut_mlp_mean_var(
    logger, lut, lut_code=None, global_step=0, show_identity=True, num_x=256
):
    """
    Log phase nonlinearity LUT to Tensorboard.
    Args:
        logger: tensorboard logger
        lut: LUT module
        lut_code: LUT code input
        global_step: global step for logging
        show_identity: bool, also plot y = x
        num_x: number of input samples
    """
    with torch.no_grad():
        if lut is not None:
            input_phase, output_mean, output_std = lut_mlp_mean_var(
                lut, lut_code, num_x=num_x
            )

            fig = plt.figure()
            if show_identity:
                plt.plot(
                    input_phase,
                    output_mean,
                    "b",
                    input_phase,
                    input_phase - input_phase[num_x // 2]
                    + output_mean[num_x // 2],
                    "k--",
                )
                plt.fill_between(
                    input_phase,
                    output_mean - output_std,
                    output_mean + output_std,
                    alpha=0.5,
                )
            logger.add_figure("lut_mlp/voltage-to-phase", fig, global_step)
            plt.close()


def lut_mlp_mean_var(
    lut, lut_code=None, slm_res=(1080, 1920), num_x=256
):
    """
    Get mean and std LUT output for uniformly spaced phase input.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_phase = torch.linspace(
        -math.pi, math.pi, num_x, dtype=torch.float32
    )
    input_phase = test_phase.cpu().numpy()
    test_phase = test_phase.view(num_x, 1, 1, 1)
    output_mean = np.empty(num_x)
    output_std = np.empty(num_x)

    for v in range(num_x):
        x = test_phase[v, ...].repeat(1, 1, *slm_res).to(device)
        output_phase = lut(x, lut_code).detach().cpu().numpy().squeeze()
        output_mean[v] = np.mean(output_phase)
        output_std[v] = np.std(output_phase)
    return input_phase, output_mean, output_std

