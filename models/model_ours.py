"""
The official implementation of the partially coherent model described in the paper "Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue".

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
import warnings
import math

import torch
import torch.nn as nn
from models.model_asm import ASM
from models.lut import LUT
from models.model_wavefront import ExplicitWavefront, ImplicitWavefront

import utils.utils as utils


class FactorizedMI(nn.Module):
    def __init__(
        self,
        rank,
        rank_per_model,
        cfg,
        num_ch=2,
        native_resolution=(1080, 1920),
        output_resolution=(1080, 1920),
        amp_max=None,
    ):
        super(FactorizedMI, self).__init__()
        wgs = []
        for _ in range(rank // rank_per_model):
            wgs.append(
                eval(cfg.wf_rep.name)(
                    rank=cfg.prop.rank_per_model,
                    num_ch=num_ch,
                    output_resolution=native_resolution,
                    angle_range_min=cfg.angle_range_min,
                    angle_range_max=cfg.angle_range_max,
                    **cfg.wf_rep
                )
            )
        self.wgs = nn.ModuleList(wgs)
        self.amp_max = amp_max
        self.output_resolution = output_resolution
        self.native_resolution = native_resolution

    def forward(self, angle):
        inc_fields = []
        for wg in self.wgs:
            inc_field = wg(angle)
            inc_fields.append(inc_field)
        inc_fields = torch.cat(inc_fields, 1)
        if self.amp_max is not None:
            inc_fields = (
                self.amp_max
                * torch.sigmoid(inc_fields.abs())
                * torch.exp(1j * inc_fields.angle())
            )

        if self.output_resolution != self.native_resolution:
            inc_fields = utils.interpolate_complex(
                inc_fields,
                size=self.output_resolution,
                coord="cartesian",
            )
        return inc_fields

    def to(self, *args, **kwargs):
        slf = super().to(*args, **kwargs)
        if slf.wgs is not None:
            new_wgs = []
            for wg in slf.wgs:
                new_wgs.append(wg.to(*args, **kwargs))
            slf.wgs = nn.ModuleList(new_wgs)
        try:
            slf.dev = next(slf.parameters()).device
        except StopIteration:  # no parameters
            device_arg = torch._C._nn._parse_to(*args, **kwargs)[0]
            if device_arg is not None:
                slf.dev = device_arg
        return slf


class ParameterizedWavePropagation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lut_pi_offset = math.pi

        # Lookup table for the SLM phase
        self.lut = LUT(
            cfg.prop.lut_type,
            cfg.prop.lut_num_layers,
            cfg.prop.lut_num_codes,
            cfg.prop.lut_num_ch,
        )

        # Latent code for spatially varying lookup table
        if sum(cfg.prop.lut_num_codes) > 0:
            self.lut_code_model = FactorizedMI(
                rank=sum(cfg.prop.lut_num_codes),
                rank_per_model=1,
                cfg=cfg,
                num_ch=1,
                output_resolution=cfg.slm_resolution,
                native_resolution=[int(r // 4) for r in cfg.slm_resolution],
            )
        else:
            self.lut_code_model = None

        # Incident wavefront on SLM
        self.inc_field = FactorizedMI(
            rank=cfg.prop.rank,
            rank_per_model=cfg.prop.rank_per_model,
            cfg=cfg,
            num_ch=2,
            output_resolution=cfg.slm_resolution,
            native_resolution=cfg.prop.inc_field_resolution,
        )

        # DC wavefront at the SLM
        self.dc_field = FactorizedMI(
            rank=cfg.prop.rank,
            rank_per_model=cfg.prop.rank_per_model,
            cfg=cfg,
            num_ch=2,
            output_resolution=cfg.slm_resolution,
            native_resolution=[int(r // 3) for r in cfg.slm_resolution],
            amp_max=cfg.prop.dc_max,
        )

        # Learned phase in Fourier domain.
        # If you have a Fourier filter in your setup,
        # you can declare a separate module for Fourier amplitude.
        self.fourier_phase = FactorizedMI(
            rank=1,
            rank_per_model=1,
            cfg=cfg,
            num_ch=1,
            output_resolution=[cfg.high_order * 2 * r for r in cfg.slm_resolution],
            native_resolution=cfg.slm_resolution,
        )

        # Angular spectrum method
        self.asm = ASM(cfg)

    def forward(
        self,
        input_phase,
        input_angle=None,
        prop_dist=None,
        synthetic_aperture=None,
        crop_to_original_resolution=False,
    ):
        """ input should be the phase of the SLM field.
            if complex-valued field is provided, it skips the LUT
        """
        if input_phase.dtype != torch.complex64:
            input_phase = (input_phase + math.pi) % (2 * math.pi) - self.lut_pi_offset
            if self.lut is not None:
                if self.lut_code_model is not None:
                    lut_code = self.lut_code_model(input_angle)
                else:
                    lut_code = None
                input_phase = self.lut(input_phase, lut_code)
            input_field = torch.exp(1j * input_phase)
        else:
            warnings.warn("Complex-valued field provided, skipping LUT")
            input_field = input_phase

        # Incident field on SLM
        if self.inc_field is not None:
            slm_incident_field = self.inc_field(input_angle)
            slm_field = slm_incident_field * input_field
        else:
            slm_field = input_field

        if self.dc_field is not None:
            dc_field = self.dc_field(input_angle)
            slm_field = slm_field + dc_field  # broadcasting

        # Learned phase in Fourier domain.
        if self.fourier_phase is not None:
            fourier_field = torch.exp(1j * self.fourier_phase(input_angle))
            if synthetic_aperture is not None:
                synthetic_aperture = fourier_field * synthetic_aperture
            else:
                synthetic_aperture = fourier_field
        else:
            fourier_field = None

        # Propagate the field using the angular spectrum method (with synthetic aperture)
        output_field = self.asm(
            slm_field,
            input_angle=input_angle,
            prop_dist=prop_dist,
            synthetic_aperture=synthetic_aperture,
            crop_to_original_resolution=crop_to_original_resolution,
        )

        # Partially coherent sum of all modes
        output_field = (output_field.abs() ** 2).mean(1, keepdim=True).sqrt()

        return output_field

    @property
    def prop_dist0(self):
        return self.asm.prop_dist0

    @prop_dist0.setter
    def prop_dist0(self, new_prop_dist0):
        self.asm.prop_dist0 = new_prop_dist0

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if self.lut is not None:
            self.lut = self.lut.to(*args, **kwargs)
        if self.lut_code_model is not None:
            self.lut_code_model = self.lut_code_model.to(*args, **kwargs)
        if self.inc_field is not None:
            self.inc_field = self.inc_field.to(*args, **kwargs)
        if self.dc_field is not None:
            self.dc_field = self.dc_field.to(*args, **kwargs)
        if self.fourier_phase is not None:
            self.fourier_phase = self.fourier_phase.to(*args, **kwargs)
        if self.asm is not None:
            self.asm = self.asm.to(*args, **kwargs)
        return self