"""
Code adapted from https://github.com/niklasnolte/MonotonicNetworks
"""

import torch
import typing as t


class SigmaNet(torch.nn.Module):
    def __init__(
        self,
        nn: torch.nn.Module,  # Must already be sigma lipschitz
        nn_2: torch.nn.Module,  # Must already be sigma lipschitz
        sigma: float,
        monotone_constraints: t.Optional[t.Iterable] = None,
    ):
        """
        Implementation of a monotone network with a sigma Lipschitz constraint.

        Args:
            nn (torch.nn.Module): Lipschitz-constrained network with Lipschitz constant sigma.
            nn_2 (torch.nn.Module): Another Lipschitz-constrained network with Lipschitz constant sigma.
            sigma (float): Lipschitz constant of the network given in nn.
            monotone_constraints (t.Optional[t.Iterable], optional): Iterable of the
                monotonic features. For example, if a network which takes a vector of size 3
                is meant to be monotonic in the last feature only, then monotone_constraints
                should be [0, 0, 1]. Defaults to all features (i.e. a vector of ones everywhere).
                Monotonically decreasing features should have value -1 instead of 1.
        """
        super().__init__()
        self.nn = nn
        self.nn_2 = nn_2
        self.register_buffer("sigma", torch.tensor([sigma]))
        if monotone_constraints is None:
            monotone_constraints = [1]
        self.register_buffer(
            "monotone_constraints", torch.tensor(monotone_constraints).float()
        )

    def forward(self, x: torch.Tensor, code_slm_2: torch.Tensor = None) -> torch.Tensor:
        self.nn = self.nn.to(x.device)
        self.nn_2 = self.nn_2.to(x.device)
        y = self.nn(x)
        if code_slm_2 is not None:
            y = torch.cat((y, code_slm_2), dim=1)
        y2 = self.nn_2(y)
        return y2 + self.sigma * (x * self.monotone_constraints).sum(
            dim=-1, keepdim=True
        )
