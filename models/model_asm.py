"""
Implementations of:

1) The off-axis angular spectrum method (ASM)
   described in the paper "Shifted angular spectrum method for off-axis numerical propagation", by K. Matsushima
2) The coherent wave propagation model with bandwidth migration
   described in the Methods section of the paper
   "Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue", by S. Choi et al.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

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
import logging
import warnings

import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F

import utils.utils as utils

warnings.filterwarnings("ignore", category=UserWarning)

class ASM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.H = None
        self._prop_dist = None
        self._angle = None

        self.prop_dist = cfg.prop_dist
        self.prop_dist0 = self.prop_dist
        if torch.is_tensor(self.prop_dist0):
            # static propagation distance (that of WRP plane)
            self.prop_dist0 = self.prop_dist0.detach()

    def forward(
        self,
        input_field,
        input_angle=None,
        prop_dist=None,
        synthetic_aperture=None,
        use_cached_params=True,
        crop_to_original_resolution=True,
    ):
        """
        Propagating the input field using the Angular Spectrum Method.

        Args:
            input_field (torch.Tensor): The input field, can be phase or complex.
            input_angle (optional): Input propagation angle, if applicable.
            prop_dist (optional): Propagation distance for the wave field.
            synthetic_aperture (optional): Optional mask in the frequency domain.
            use_cached_params (bool): If True, use cached propagation parameters if available.
            crop_to_original_resolution (bool): If True, output has same resolution as input.

        Returns:
            torch.Tensor: The propagated complex field.
        """
        # Convert input from phase to complex field if necessary
        if input_field.dtype != torch.complex64:
            input_field = torch.exp(1j * input_field)

        # Update propagation distance property if a new one is provided
        if prop_dist is not None:
            self.prop_dist = prop_dist

        # Compute the propagation kernel H if it is not already computed or cached
        if self.H is None or not use_cached_params:
            self.H = self.compute_H(
                input_field,
                angle_in=input_angle,
                angle_sample=input_angle,
                high_order=self.cfg.high_order
            )

        # Apply synthetic aperture (frequency domain mask) if provided
        if synthetic_aperture is not None:
            # If resolutions differ, interpolate synthetic_aperture to match self.H's spatial size
            synthetic_aperture = utils.interpolate_complex(synthetic_aperture, size=self.H.shape[-2:], mode='bilinear', align_corners=False)
            H = self.H * synthetic_aperture
        else:
            H = self.H

        # Perform propagation in the frequency domain
        output_field = self.prop(
            input_field, H,
            high_order=self.cfg.high_order,
            crop_to_original_resolution=crop_to_original_resolution
        )

        return output_field

    def generate_grid(self, resolution, high_order=False, device=torch.device("cuda")):
        """
        Generates frequency grids for the Fourier domain, optionally supporting higher-order propagation.

        Args:
            resolution (list or tuple): [height, width] of the grid.
            high_order (bool): If True, generates higher-order frequency grid (e.g., for 3rd order diffraction).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (FY, FX), each of shape [1, 1, H, W].
        """
        # Determine upsampling for high-order propagation
        ho_y, ho_x = (3, 3) if high_order else (1, 1)

        # Generate 1D frequency bins, scaled for the correct range
        fy = (
            torch.linspace(-1, 1, resolution[-2], dtype=torch.float32, device=device)
            * (ho_y / (2 * self.cfg.pixel_pitch))
            * (1 - 1 / resolution[-2])
        )
        fx = (
            torch.linspace(-1, 1, resolution[-1], dtype=torch.float32, device=device)
            * (ho_x / (2 * self.cfg.pixel_pitch))
            * (1 - 1 / resolution[-1])
        )

        # Create 2D frequency grid with meshgrid
        fx_grid, fy_grid = torch.meshgrid(fx, fy, indexing="ij")

        # Arrange shape and dimensions: [1, 1, H, W]
        FX = fx_grid.permute(1, 0).unsqueeze(0).unsqueeze(0)
        FY = fy_grid.permute(1, 0).unsqueeze(0).unsqueeze(0)

        # Return as (FY, FX), consistent with ASM convention (y, x)
        return FY, FX

    def compute_H(
        self,
        input_field,
        angle_in=None,
        angle_sample=None,
        apply_sinc=True,
        high_order=False,
    ):
        """
        Compute the propagation kernel H for the Angular Spectrum Method (ASM).

        Args:
            input_field (torch.Tensor): The input field tensor.
            angle_in (optional): Input propagation angle, if applicable.
            angle_sample (optional): Sampled propagation angle, if applicable.
            apply_sinc (bool): Whether to apply the sinc envelope to compensate for finite pixel size.
            high_order (bool): Whether to use higher-order propagation (for modeling higher diffraction orders).

        Returns:
            torch.Tensor: The computed propagation kernel H on the same device as input_field.
        """
        device = input_field.device
        # Determine upsampling factor for higher-order propagation
        ho = 3 if high_order else 1

        # Calculate the resolution for the Fourier domain kernel
        H_resolution = [2 * ho * p for p in input_field.shape[-2:]]

        # Generate the frequency grids (FY, FX) in the Fourier domain
        FY, FX = self.generate_grid(H_resolution, high_order=high_order, device=device)

        # Compute magnitude: sinc envelope compensates for the finite pixel size
        if apply_sinc:
            sinc_env = self.generate_sinc_envelope((FY, FX))
        else:
            sinc_env = 1.0
        H_filter = sinc_env

        # Compute phase: angular spectrum phase term
        wavelength = self.cfg.wavelengths[self.cfg.channel]
        # Angular spectrum method: phase advance corresponding to propagation in free space
        G = 2 * math.pi * ((1 / wavelength) ** 2 - (FX ** 2 + FY ** 2)).sqrt()
        H_exp = G.reshape(1, 1, *G.shape[-2:])

        # Construct the complex-valued propagation kernel
        H = H_filter * torch.exp(1j * (H_exp * self.prop_dist))
        return H.to(input_field.device)

    def generate_sinc_envelope(self, frequency_grid):
        """
        Generates a two-dimensional sinc envelope for the frequency grid to compensate
        for the finite pixel size in the spatial domain.

        Args:
            frequency_grid (tuple): Tuple (FY, FX) of torch tensors representing the
                                    frequency coordinates in the y and x directions.

        Returns:
            sinc_env (torch.Tensor): The computed 2D sinc envelope as a tensor reshaped to match
                                     the input grid with appropriate broadcasting dimensions.
        """
        FY, FX = frequency_grid

        # Compute the argument to the sinc function for x and y, scaled by pixel pitch
        cFX = math.pi * FX * self.cfg.pixel_pitch

        # Compute the normalized sinc in x (safeguard denominator from zero)
        sincFX = torch.sin(cFX) / (cFX + 1e-12)
        # Replace zeros to avoid numerical issues (sinc(0) = 1)
        sincFX[FX == 0.0] = 1
        # Start sinc envelope as |sinc(x)|
        sinc_env = sincFX.abs()

        # Repeat for y dimension
        cFY = math.pi * FY * self.cfg.pixel_pitch
        sincFY = torch.sin(cFY) / (cFY + 1e-12)
        sincFY[FY == 0.0] = 1
        # Element-wise multiply |sinc(x)| * |sinc(y)|
        sinc_env *= sincFY.abs()

        # Reshape for correct broadcasting (add singleton channel dimensions)
        sinc_env = sinc_env.reshape(1, 1, *sinc_env.shape[-2:])

        return sinc_env

    def prop(
        self,
        input_field,
        H,
        high_order=False,
        crop_to_original_resolution=True,
    ):
        """
        Propagate the input complex field using the propagation kernel H.

        Args:
            input_field (torch.Tensor): Input field tensor [B, C, H, W].
            H (torch.Tensor): Frequency domain propagation kernel.
            high_order (bool): If True, simulate higher-order diffraction by upsampling,
                               blurring and then downsampling the output.
            crop_to_original_resolution (bool): If False, double the output resolution;
                                                otherwise, outputs are the same
                                                spatial size as input.

        Returns:
            torch.Tensor: Propagated field tensor.
        """
        # Get input and output resolutions
        input_resolution = input_field.shape[-2:]
        output_resolution = input_resolution

        # Update output resolution if high order or full-res output is used
        if high_order:
            output_resolution = tuple(dim * 3 for dim in input_resolution)

        # Pad the input field for linear convolution (avoid circular artifacts)
        conv_size = [dim * 2 for dim in input_resolution]
        u_in = utils.pad_image(input_field, conv_size, stacked_complex=False)

        # TODO: Handle resolution mismatches or kernel size inconsistencies

        # FFT to frequency domain, centered (shifted dc to center)
        U1 = tfft.fftshift(
            tfft.fftn(u_in, dim=(-2, -1), norm="ortho"),
            dim=(-2, -1),
        )

        # Repeat Fourier spectrum for higher-order modeling
        # (e.g., simulating multiple diffraction orders)
        if high_order:
            U1 = U1.repeat(1, 1, 3, 3)

        # Apply propagation kernel in frequency domain     
        U2 = U1 * H

        # Transform back to spatial domain
        u_out = tfft.ifftn(
            tfft.ifftshift(U2, dim=(-2, -1)),
            dim=(-2, -1),
            norm="ortho",
        )

        # Crop to the desired output resolution if required
        if crop_to_original_resolution:
            u_out = utils.crop_image(
                u_out, output_resolution, stacked_complex=False
            )

        # Apply box blur and downsampling if simulating higher-order propagation
        if high_order:
            # Blur intensity and recover amplitude
            blurred_recon = utils.intensity_blur((u_out.abs()) ** 2, (3, 3)).sqrt()
            # Downsample by a factor of 3 in spatial dimensions
            blurred_recon = blurred_recon[..., 1::3, 1::3]
            blurred_angle = u_out.angle()[..., 1::3, 1::3]
            # Reconstruct output field with blurred amplitude and original phase
            u_out = blurred_recon * torch.exp(1j * blurred_angle)

        return u_out

    @property
    def prop_dist(self):
        return self._prop_dist

    @property
    def angle(self):
        return self._angle

    @prop_dist.setter
    def prop_dist(self, new_prop_dist):
        if self._prop_dist is None:
            self._prop_dist = new_prop_dist
        else:
            if self._prop_dist != new_prop_dist:
                self._prop_dist = new_prop_dist
                self.H = None  # reset H

    @angle.setter
    def angle(self, new_angle):
        self._angle = new_angle


class OffAxisWrapper(nn.Module):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model

    def forward(
        self,
        input_field,
        input_angle,
        prop_dist=None,
        synthetic_aperture=None,
    ):
        """
        Output is amplitude of the input field shifted by the geometric shift.
        """
        if prop_dist is None:
            prop_dist = self.model.prop_dist0
        output_field = self.model(
            input_field,
            input_angle,
            prop_dist,
            synthetic_aperture,
            crop_to_original_resolution=False,
        )
        shifted_abs = self.off_axis_shift(
            output_field,
            input_angle,
            prop_dist,
        )
        return shifted_abs

    def off_axis_shift(self, input_field, input_angle, prop_dist):
        """
        Shift the output field by the geometric shift.
        """
        # Perform bilinear interpolation with the intensity
        input_device = input_field.device
        if input_angle is None:
            return input_field

        output_intensity = input_field.abs() ** 2
        pixel_pitch = self.cfg.pixel_pitch
        canvas_resolution = self.cfg.canvas_resolution
        canvas = torch.zeros(
            len(input_angle), 1, *canvas_resolution, device=input_field.device
        )

        # Convert input_angle to tensor of shape (N, 2) if not already
        input_angle = torch.tensor(input_angle, dtype=torch.float32, device=input_device)
        if input_angle.ndim == 1:
            input_angle = input_angle.unsqueeze(0)  # shape (1, 2)
        N = input_angle.shape[0]

        # Ensure prop_dist is broadcastable (scalar or (N,))
        def check_prop_dist(prop_dist, N, device):
            if prop_dist is None:
                return None
            if isinstance(prop_dist, (float, int)):
                prop_dist = torch.full((N,), float(prop_dist), dtype=torch.float32, device=device)
            elif torch.is_tensor(prop_dist):
                prop_dist = prop_dist.to(dtype=torch.float32, device=device)
                if prop_dist.ndim == 0 or prop_dist.shape[0] != N:
                    prop_dist = prop_dist.expand(N)
            else:
                raise TypeError("prop_dist must be float, int, or torch.Tensor")
            return prop_dist

        prop_dist = check_prop_dist(prop_dist, N, input_device)

        # Scales for spatial dimensions (y, x)
        scale = [
            canvas_resolution[-2] / output_intensity.shape[-2],  # scale for y
            canvas_resolution[-1] / output_intensity.shape[-1],  # scale for x
        ]
        scale = torch.tensor(scale, dtype=torch.float32, device=input_device)  # (2,)

        # Calculate translation for each angle in (N, 2)
        trans = prop_dist.unsqueeze(-1) * torch.tan(input_angle) / (
            pixel_pitch * torch.tensor(input_field.shape[-2:], dtype=torch.float32, device=input_device)
        )  # (N, 2), order: (y, x)

        # Construct affine matrix theta for each entry in batch (N, 2, 3)
        theta = torch.zeros((N, 2, 3), dtype=torch.float32, device=input_device)
        theta[:, 0, 0] = scale[1]  # x scaling
        theta[:, 1, 1] = scale[0]  # y scaling
        theta[:, 0, 2] = -trans[:, 1]  # x translation
        theta[:, 1, 2] = -trans[:, 0]  # y translation

        affine_grid = F.affine_grid(theta, canvas.shape)
        output_intensity = F.grid_sample(output_intensity, affine_grid)

        # Returning amplitude
        return output_intensity.sqrt()

    @property
    def prop_dist(self):
        return self.model.prop_dist

    @prop_dist.setter
    def prop_dist(self, new_prop_dist):
        self.model.prop_dist = new_prop_dist

    @property
    def prop_dist0(self):
        return self.model.prop_dist0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        return self