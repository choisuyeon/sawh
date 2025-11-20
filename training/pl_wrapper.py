"""
PyTorch Lightning wrapper for the training loop of the partially coherent model.

Any questions about the code can be addressed to Suyeon Choi (suyeon@stanford.edu)

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC)
    - Non-commercial use only (commercial licenses from Stanford)
    - Provided as-is, with no warranties
    - Please cite our work if you publish based on this

Article:
S. Choi, C. Jang, D. Lanman, G. Wetzstein,
"Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue",
Nature Photonics, 2025
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import utils.utils as utils
import torch.nn.functional as F
import logging


class TrainingWrapper(pl.LightningModule):
    """ PyTorch Lightning wrapper for the training loop.
    Args:
        model (nn.Module): The model to train.
        loss_fn (nn.Module): The loss function to use.
        lr_train (float): The learning rate to use.
        **kwargs: Additional arguments.
    """
    def __init__(self, 
                 forward_model, 
                 loss_fn=nn.L1Loss(), 
                 lr_train=0.0003, 
                 use_mask=False,
                 mask_threshold=1/256,
                 roi_res=[1080, 1920],
                 *args,
                 **kwargs):
        """
            Args:
            forward_model (nn.Module): The forward model to train.
            loss_fn (nn.Module): The loss function to use.
            lr_train (float): The learning rate to use.
            use_mask (bool): Whether to use a mask for the training.
            mask_threshold (float): The threshold for the mask.
            roi_res (list): The region of interest to use.
        """
        super().__init__()
        self.forward_model = forward_model
        self.loss_fn = loss_fn
        self.lr_train = lr_train
        self.roi_res = roi_res
        self.use_mask = use_mask
        self.mask_threshold = mask_threshold

    def forward(self, x, *args, **kwargs):
        return self.forward_model(x, *args, **kwargs)

    def perform_step(self, batch, batch_idx, prefix):
        slm_phase, target_amp, angle_steered = batch

        # forward pass
        recon_amp = self.forward_model(slm_phase, angle_steered)
        with torch.no_grad():
            self.slm_phase_temp = slm_phase.squeeze(0).cpu().detach()

        if self.use_mask:
            with torch.no_grad():
                train_mask = self.forward_model(torch.zeros_like(slm_phase), angle_steered)
                train_mask = train_mask.abs() > self.mask_threshold
                self.train_mask = train_mask
        else:
            train_mask = None
            self.train_mask = train_mask

        # compute loss
        loss = self.loss_func(
            recon_amp.clamp(0.0, 1.0),
            target_amp.clamp(0.0, 1.0),
            self.loss_fn,
            train_mask=train_mask,
            prefix=prefix,
        )

        with torch.no_grad():
            l2_value = self.loss_func(
                recon_amp, target_amp, F.mse_loss, train_mask=train_mask, prefix=prefix
            )
            if train_mask is not None:
                s = (train_mask.shape[-2] * train_mask.shape[-1]) / train_mask.sum()  # reciprocal
            else:
                s = 1.0
            l2_value = l2_value * s
            psnr_value = 10 * math.log10(target_amp.max() / l2_value)
            self.log(
                f"PSNR_{prefix}", psnr_value, on_step=True, on_epoch=True, sync_dist=True
            )
            self.log(
                f"Loss_{prefix}", loss, on_step=True, on_epoch=True, sync_dist=True
            )

        return loss

    def loss_func(self, recon_amp, target_amp, loss_fn, train_mask, prefix, scale=False):
        recon_amp = utils.crop_image(recon_amp, self.roi_res)
        target_amp = utils.crop_image(target_amp, self.roi_res)

        with torch.no_grad():
            self.recon_amp = recon_amp.squeeze().reshape(1, *recon_amp.shape[-2:])
            self.target_amp = target_amp.squeeze().reshape(1, *target_amp.shape[-2:])

        if train_mask is not None:
            train_mask = utils.crop_image(train_mask, self.roi_res)
            recon_amp = recon_amp * train_mask
            target_amp = target_amp * train_mask

        if scale:  # for evaluation (do not use this for training)
            s = (recon_amp * target_amp).mean() / (recon_amp ** 2).mean()
        else:
            s = 1.0

        return loss_fn(s * recon_amp, target_amp)

    def training_step(self, batch, batch_idx):
        return self.perform_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.perform_step(batch, batch_idx, "validation")
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.evaluate_step(batch, batch_idx, "test")

    # PL lightning 2.x
    if int(pl.__version__.split(".")[0]) >= 2:
        def on_test_epoch_end(self) -> None:
            self.epoch_end_images("train")

        def on_train_epoch_end(self) -> None:
            self.epoch_end_images("train")

        def on_validation_epoch_end(self) -> None:
            pass
    else:
        def test_epoch_end(self, outputs) -> None:
            self.epoch_end_images("test")

        def training_epoch_end(self, outputs) -> None:
            self.epoch_end_images("train")

        def validation_epoch_end(self, outputs) -> None:
            self.epoch_end_images("validation")
            pass

    def epoch_end_images(self, prefix):
        with torch.no_grad():
            if self.local_rank == 0:
                logger = self.logger.experiment
                logger.add_image(f'amp_recon/{prefix}', self.recon_amp.abs().clip(0, 1), self.global_step)
                logger.add_image(f'amp_target/{prefix}', self.target_amp.abs().clip(0, 1), self.global_step)
                logger.add_image(f'slm_phase/{prefix}', utils.normalize_angle(self.slm_phase_temp).clip(0, 1), self.global_step)
                # logger.add_image(f'waveguide_amp/avg_{prefix}', (self.waveguide_temp.abs()**2).mean(0, keepdim=True).sqrt().clip(0, 1), self.global_step)
                angle_in = torch.zeros(1, 2, device=self.recon_amp.device)
                inc_field = self.forward_model.model.inc_field(angle_in).squeeze(0)
                logger.add_image(f'waveguide_amp/avg_{prefix}', (inc_field.abs()**2).mean(0, keepdim=True).sqrt().clip(0, 1).cpu().detach(), self.global_step)
                for ic in range(len(inc_field)):
                    logger.add_image(f'waveguide_amp/{ic}_{prefix}', inc_field[ic:ic+1,...].abs().clip(0, 1).cpu().detach(), self.global_step)
                    logger.add_image(f'waveguide_angle/{ic}_{prefix}', utils.normalize_angle(inc_field[ic:ic+1,...].angle()).clip(0, 1).cpu().detach(), self.global_step)
                
                if self.forward_model.model.dc_field is not None:
                    dc_field = self.forward_model.model.dc_field(angle_in).squeeze(0)
                    logger.add_image(f'waveguide_amp/avg_{prefix}', (inc_field.abs()**2).mean(0, keepdim=True).sqrt().clip(0, 1).cpu().detach(), self.global_step)
                    for ic in range(len(dc_field)):
                        logger.add_image(f'dc_field/{ic}_{prefix}', dc_field[ic:ic+1,...].abs().clip(0, 1).cpu().detach(), self.global_step)
                        logger.add_image(f'dc_field_angle/{ic}_{prefix}', utils.normalize_angle(dc_field[ic:ic+1,...].angle()).clip(0, 1).cpu().detach(), self.global_step)
                
                fourier_phase = self.forward_model.model.fourier_phase(angle_in).squeeze(0)
                logger.add_image(f'fourier_phase/{prefix}', utils.normalize_angle(fourier_phase).clip(0, 1).cpu().detach(), self.global_step)
                if self.forward_model.model.lut is not None:
                    if self.forward_model.model.lut_code_model is not None:
                        lut_code = self.forward_model.model.lut_code_model(angle_in)
                        lut_code = (lut_code - lut_code.min()) / (lut_code.max() - lut_code.min())
                        logger.add_image(f'lut_code/{prefix}', lut_code.squeeze(0).cpu().detach(), self.global_step)
                    else:
                        lut_code = None
                    self.forward_model.model.lut.visualize_tensorboard(logger, self.global_step, lut_code=lut_code)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.forward_model.parameters(), lr=self.lr_train)
        return optimizer

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.forward_model is not None:
            self.forward_model = self.forward_model.to(*args, **kwargs)

class PSNRProgressBar(pl.callbacks.TQDMProgressBar):
    def get_metrics(self, trainer, model):
        # include all standard metrics plus PSNR_train_step if available
        standard_metrics = super().get_metrics(trainer, model)
        metrics = dict(standard_metrics)
        # Add 'PSNR_train_step' if it exists in the callback metrics
        try:
            # Lightning puts logge
            # 0d metrics into trainer.progress_bar_callback.trainer.logged_metrics
            pb_metrics = trainer.callback_metrics
            if "PSNR_train_step" in pb_metrics:
                psnr = pb_metrics["PSNR_train_step"]
                # Convert to float for better display
                try:
                    psnr = float(psnr)
                except Exception:
                    pass
                metrics["PSNR_train_step"] = psnr
        except Exception:
            pass
        return metrics

        
class ForwardModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        # 1. Let Lightning save the normal .ckpt
        super()._save_checkpoint(trainer, filepath)

        # 2. Now save ONLY the forward_model in a separate file
        # Save state_dict, but use state_dict(cpu()) so the actual model stays on the device.
        forward_model = trainer.lightning_module.forward_model
        state_dict = {k: v.detach().cpu() for k, v in forward_model.state_dict().items()}
        fm_path = filepath + "_forward.pt"
        torch.save(state_dict, fm_path)