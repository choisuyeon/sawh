"""
Implementations of

Implicit Neural Waveguide Model described in the paper
"Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue", by S. Choi et al.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC). In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Article:
S. Choi, C. Jang, D. Lanman, G. Wetzstein,
"Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue",
Nature Photonics, 2025

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import utils.utils as utils

class ExplicitWavefront(nn.Module):
    """
    Explicit parameterization of the wavefronts as learnable tensors.
    """
    def __init__(
        self,
        num_ch: int = 2,
        rank: int = 1,
        output_resolution: list = [1080, 1920],
        angle_range_min: tuple = (-1e-9, -1e-9),
        angle_range_max: tuple = (1e-9, 1e-9),
        num_wavefronts: list = [1, 1],
        **kwargs,
    ):
        """
        Args:
            num_ch: Number of channels. 2 for complex-valued field, 1 for phase.
            rank: Number of stacked wavefronts.
            output_resolution: Resolution of the wavefront, [height, width].
            num_wavefronts: Number of wavefronts in each dimension (e.g., [1, 1] or [5, 5]).
            angle_range_min: Minimum incident angle (y, x) (tuple of floats).
            angle_range_max: Maximum incident angle (y, x) (tuple of floats).
        """
        super().__init__()
        self.num_ch = num_ch
        self.rank = rank
        self.output_resolution = output_resolution
        self.num_wavefronts = num_wavefronts

        # Register range of angles as buffers
        self.register_buffer(
            'angle_range_min', torch.tensor(angle_range_min, dtype=torch.float32).reshape(1, 2)
        )
        self.register_buffer(
            'angle_range_max', torch.tensor(angle_range_max, dtype=torch.float32).reshape(1, 2)
        )

        # Initialize the wavefront as a complex tensor: amplitude 1, phase 0
        amp = torch.ones(1, self.rank, *self.output_resolution, *self.num_wavefronts)
        # For rank > 1, you may want to inject random noise to the phase to avoid symmetry
        phase = torch.zeros(1, self.rank, *self.output_resolution, *self.num_wavefronts)
        field = amp * torch.exp(1j * phase)

        # Store real and imaginary parts as learnable parameters
        self.wavefronts_real = nn.Parameter(field.real)
        if self.num_ch == 2:
            self.wavefronts_imag = nn.Parameter(field.imag)
        else:
            self.wavefronts_imag = None

    def forward(self, input_angle=None):
        """
        Returns:
            Complex wavefront tensor, optionally interpolated by angle.
        """
        if input_angle is None:
            assert self.num_wavefronts == (1, 1), "Only one explicit wavefront is supported when num_wavefronts == (1, 1)."
            # Return the single wavefront at index [0, 0]
            output = self.wavefronts_real[..., 0, 0]
            if self.wavefronts_imag is not None:
                output = output + 1j * self.wavefronts_imag[..., 0, 0]
            return output
        else:
            angle_normalized = (input_angle - self.angle_range_min) / (self.angle_range_max - self.angle_range_min)
            return self.interpolate_wavefronts(angle_normalized)

    def interpolate_wavefronts(self, angle_normalized):

        wavefronts = self.wavefronts_real
        if self.wavefronts_imag is not None:
            wavefronts = wavefronts + 1j * self.wavefronts_imag
            
        idx = [(a + 1) / 2 * (s - 1) for a, s in zip(angle_normalized.squeeze(), wavefronts.shape[-2:])]
        idx_y = idx[0]
        idx_x = idx[1]

        # Get integer indices for the four corners
        y0 = int(math.floor(idx_y.item()))
        x0 = int(math.floor(idx_x.item()))
        
        # Ensure we don't go out of bounds
        y1 = min(y0 + 1, wavefronts.shape[-2] - 1)
        x1 = min(x0 + 1, wavefronts.shape[-1] - 1)

        # Get weights for bilinear interpolation
        wy1 = idx_y - y0
        wx1 = idx_x - x0
        wy0 = 1 - wy1
        wx0 = 1 - wx1

        # Perform bilinear interpolation
        wavefront = (wavefronts[..., y0, x0] * (wy0 * wx0) +
                    wavefronts[..., y1, x0] * (wy1 * wx0) +
                    wavefronts[..., y0, x1] * (wy0 * wx1) +
                    wavefronts[..., y1, x1] * (wy1 * wx1))
        
        return wavefront


class ImplicitWavefront(nn.Module):
    """
    Implicit neural wavefront representation using a combination of
    spatial grid encoding, hash encoding, and MLP-based decoding.
    Reference: Instant-ngp: https://github.com/NVlabs/instant-ngp
               torch-ngp: https://github.com/ashawkey/torch-ngp
    """
    def __init__(
        self,
        num_ch: int = 2,
        rank: int = 1,
        output_resolution: list = [1080, 1920],
        angle_range_min: tuple = (-1e-9, -1e-9),
        angle_range_max: tuple = (1e-9, 1e-9),
        backbone: str = 'tcnn',
        log2_hashmap_size=18,
        desired_resolution=None,
        angle_aux_code=True,
        tn_num_layers=3,
        tn_num_ch=32,
        dec_num_layers=3,
        dec_num_ch=32,
        **kwargs,
    ):
        """
        Args:
            num_ch: Number of channels (2 for complex, 1 for phase).
            rank: Number of stacked wavefronts.
            output_resolution: Output spatial resolution as [height, width].
            angle_range_min: Minimum incident angle (tuple of floats, shape [2]).
            angle_range_max: Maximum incident angle (tuple of floats, shape [2]).
            backbone: Encoding backend ('tcnn', 'torch-ngp', etc.).
            log2_hashmap_size: log2 of hash grid table size.
            desired_resolution: Target high-resolution mapping for encoding.
            angle_aux_code: Append normalized angle as aux code to decoder MLP input.
            tn_num_layers: Number of transnet layers.
            tn_num_ch: Number of hidden channels per transnet layer.
            dec_num_layers: Number of decoder MLP layers.
            dec_num_ch: Number of channels per decoder MLP layer.
        """
        super().__init__()

        # Register buffers for min/max angle ranges
        self.register_buffer(
            'angle_range_min', torch.tensor(angle_range_min, dtype=torch.float32).reshape(1, 2)
        )
        self.register_buffer(
            'angle_range_max', torch.tensor(angle_range_max, dtype=torch.float32).reshape(1, 2)
        )

        self.num_ch = num_ch

        self.output_resolution = output_resolution
        self.log2_hashmap_size = log2_hashmap_size

        # Set desired hash encoding resolution (default to max dim)
        if desired_resolution is None:
            desired_resolution = max(output_resolution)
        self.desired_resolution = desired_resolution

        # Internal parameters
        self.backend = backbone
        self.output_res = self.output_resolution
        self.rank = rank
        self.input_dim = 2  # Input dimension to hash encoding (y, x)
        self.angle_aux_code = angle_aux_code
        self.output_ch = num_ch  # Number of final output channels

        # Build spatial input grid
        self.x_in = self.load_spatial_grid(self.output_res)

        # Transnet: small MLP for angle-dependent warping
        self.transnet = self.load_transnet(
            num_ch=tn_num_ch, num_layers=tn_num_layers
        )

        # Multi-resolution hash-based encoding
        self.hash_encoding = self.load_hash_encoding(
            desired_resolution=self.desired_resolution,
            log2_hashmap_size=self.log2_hashmap_size
        )

        # Final decoder MLP
        self.decoder_mlp = self.load_decoder_mlp(
            num_layers=dec_num_layers, hidden_dim=dec_num_ch
        )

    def forward(self, angle_in):
        """
        Args:
            angle_in: Incident angle tensor, shape [2] or [N, 2].
        Returns:
            Output complex (or real) wavefront, shape [1, ..., H, W].
        """

        # Normalize angle to [0,1] range for encoding
        angle_normalized = (angle_in - self.angle_range_min) / (self.angle_range_max - self.angle_range_min)

        # Broadcast the angle to match spatial coords, shape [MN, 2]
        angle_expanded = angle_normalized.squeeze().expand_as(self.x_in)

        # ---- Transnet: concatenate input coords and angles ----
        if self.transnet is not None:
            encoder_input = torch.cat(
                (self.x_in.to(angle_normalized.device), angle_expanded), dim=-1
            )
            # Pass through transnet (small MLP)
            for l in range(self.transnet_num_layers):
                encoder_input = self.transnet[l](encoder_input)
                if l != self.transnet_num_layers - 1:
                    encoder_input = F.relu(encoder_input, inplace=True)

            # warp spatial coordinates nonlinearly
            # For 2D, apply tanh-warp to coordinates 
            # to ensure it is in the range of [-1, 1]
            warped_coords = self.x_in.unsqueeze(2).expand(
                -1, -1, encoder_input.shape[-1] // self.x_in.shape[-1]
            )
            encoder_input = torch.tanh(
                warped_coords.reshape(*encoder_input.shape) + encoder_input
            )
        else:
            # Use only input spatial grid (no warping)
            encoder_input = self.x_in.to(angle_normalized.device)

        # ---- Hash grid encoding (multi-scale spatial encoding) ----
        encoder = self.hash_encoding
        x = encoder(encoder_input)

        # Optionally append normalized angle as aux code to decoder input
        if self.angle_aux_code:
            h = torch.cat(
                (x, angle_normalized.reshape(1, 2).repeat(x.shape[0], 1)), dim=1
            )
        else:
            h = x

        # ---- Pass through decoder MLP (with optional skip connections) ----
        for l in range(len(self.decoder_mlp)):
            if hasattr(self, "skips") and l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.decoder_mlp[l](h)
            if l != len(self.decoder_mlp) - 1:
                h = F.relu(h, inplace=True)

        # ---- Reshape and convert output ----
        wavefront = h.permute(1, 0).view(1, -1, *self.output_res)  # [1, C, H, W] or similar
        if self.output_ch == 2:
            # Convert channel-major real/imag to complex tensor
            wavefront = utils.stacked_to_complex(wavefront)
        return wavefront

    def load_spatial_grid(self, output_res):
        """
        Build a normalized grid of coordinates for the output resolution.
        Args:
            output_res: [height, width]
        Returns:
            spatial_grid: shape [num_pixels, 2] with per-pixel (y,x) coordinates
        """
        if self.backend == 'tcnn':
            # For tcnn, use [0, 1] coordinates, pixel-centered
            half_dy = 0.5 / output_res[0]
            half_dx = 0.5 / output_res[1]
            yy = torch.linspace(half_dy, 1 - half_dy, output_res[0])
            xx = torch.linspace(half_dx, 1 - half_dx, output_res[1])
        else:
            # Fallback: [-aspect, aspect] for y, [-1, 1] for x
            yy = torch.linspace(-self.output_resolution[0] / self.output_resolution[1], 
                                 self.output_resolution[0] / self.output_resolution[1], output_res[0])
            xx = torch.linspace(-1.0, 1.0, output_res[1])

        # meshgrid produces grid of pixel centers
        if hasattr(torch.meshgrid, 'indexing'):
            grid = torch.meshgrid(yy, xx, indexing='ij')
        else:
            grid = torch.meshgrid(yy, xx)
        stacked_coord = torch.stack(grid, -1)
        spatial_grid = stacked_coord.unsqueeze(0)
        spatial_grid = spatial_grid.view(-1, 2)  # Flatten to [num_pixels, 2]
        return spatial_grid

    def load_transnet(self, num_ch, num_layers):
        """
        Construct a small MLP ("transnet") for coordinate warping, conditioned on angle.
        """
        if num_layers == 0:
            return None
        transnet = []
        hidden_dim = num_ch
        self.transnet_num_layers = num_layers
        for l in range(self.transnet_num_layers):
            if l == 0:
                # Input: [x, y, angle_y, angle_x] (4 features)
                in_dim = 4
            else:
                in_dim = hidden_dim

            if l == self.transnet_num_layers - 1:
                # Output: for hash grid(s), matches input to hash encoding
                out_dim = self.input_dim
            else:
                out_dim = hidden_dim
            transnet.append(nn.Linear(in_dim, out_dim, bias=True))
        return nn.ModuleList(transnet)

    def load_hash_encoding(self, level_dim=2, desired_resolution=None, log2_hashmap_size=18):
        """
        Create multi-level hash-based encoder for spatial input.
        """
        self.in_dim = level_dim * 16  # 16 levels, each 'level_dim'-dim vector
        if desired_resolution is not None:
            per_level_scale = np.exp2(
                np.log2(desired_resolution / 16) / (16 - 1)
            )
        else:
            per_level_scale = 2.0

        encoder = None
        if self.backend == 'torch-ngp':
            try:
                from ingp import get_encoder
                encoder, self.in_dim = get_encoder(
                    "hashgrid",
                    input_dim=self.input_dim,
                    level_dim=level_dim,
                    desired_resolution=desired_resolution,
                    log2_hashmap_size=log2_hashmap_size,
                )
            except ImportError:
                raise RuntimeError("ingp.get_encoder not found in environment.")
        elif self.backend == 'tcnn':
            try:
                import tinycudann as tcnn
            except ImportError:
                raise RuntimeError("tinycudann (tcnn) is required for 'tcnn' backend.")
            hash_config = {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "per_level_scale": per_level_scale,
                "base_resolution": 16,
            }
            encoder = tcnn.Encoding(
                n_input_dims=self.input_dim, encoding_config=hash_config
            )
        return encoder

    def load_decoder_mlp(self, num_layers=3, hidden_dim=32):
        """
        Build a decoder MLP to convert encoded input to wavefront fields.
        Supports skip connections if self.skips is populated.
        """
        backbone = []
        self.skips = []  # Optionally fill with skip-layer indices
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
                if self.angle_aux_code:
                    # Optionally append 2 channels for normalized angle
                    in_dim += 2
            elif l in self.skips:
                in_dim = hidden_dim + self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = self.output_ch * getattr(self, 'num_modes_out', 1)
            else:
                out_dim = hidden_dim

            backbone.append(nn.Linear(in_dim, out_dim, bias=False))
        return nn.ModuleList(backbone)

    def to(self, *args, **kwargs):
        """
        Move model (and x_in spatial grid) to target device or dtype.
        """
        self = super().to(*args, **kwargs)
        self.x_in = self.x_in.to(*args, **kwargs)
        return self
