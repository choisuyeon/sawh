"""
Implmentation of the CGH algorithm in the paper "Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue".

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

import torch
import tqdm
import logging
import cgh.utils_cgh as utils_cgh
import utils.utils as utils
from models import load_forward_model


def load_cgh_method(cfg):
    return eval(cfg.method)

def load_phase_init(cfg):
    phase_init = eval(cfg.cgh.init_type)
    return phase_init

def rand_tmnh(cfg):
    """ initialization of phase """
    return (-0.5 + torch.rand(cfg.cgh.rank, 1, *cfg.model.slm_resolution)) * cfg.cgh.init_uniform_phase_range

def gd(target_amp, initial_phase, forward_model, cfg):
    dev = target_amp.device
    assert dev == initial_phase.device, 'target_amp.device and initial_phase.device must be the same'

    slm_phase = initial_phase.requires_grad_(True)
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam([initial_phase], lr=cfg.lr)
    loss_fn = torch.nn.MSELoss().to(dev)

    # Set up forward model
    steered_angle = cfg.steered_angle  # this could be a set of angles (matches len(slm_phase)) or a single float number in radians
    logging.info(f"CGH: steered_angle: {steered_angle} ...")

    # Run optimization loop
    best_loss_val = float('inf')
    logging.info(f'CGH: Running gradient descent using {cfg.supervision.name} supervision for {cfg.num_iters} iterations ...')
    tqdm_bar = tqdm.tqdm(range(cfg.num_iters))
    for t_iter in tqdm_bar:

        # zero out gradients
        optimizer.zero_grad()   

        loss_val_total = 0.0
        # sub-loop for minibatch
        for t_mb in range(cfg.supervision.minibatch_size):

            # get synthetic aperture and corresponding gt amplitude
            gt_amp, sa_amp, offset_dist = utils_cgh.get_gt_amp_and_sa(cfg, forward_model.cfg, target_amp, 
                                                                      steered_angle=steered_angle, t_mb=t_mb)
            
            # get recon amplitude through our simulation model
            recon_amp = forward_model(slm_phase, 
                                      input_angle=steered_angle, 
                                      prop_dist=forward_model.prop_dist0 + offset_dist,
                                      synthetic_aperture=sa_amp).abs()
            # time multiplexing
            recon_amp = (recon_amp ** 2).mean(dim=0, keepdim=True).sqrt()

            # crop image
            gt_amp = utils.crop_image(gt_amp, cfg.roi_res)
            recon_amp = utils.crop_image(recon_amp, cfg.roi_res)

            # get scaling factor
            s = utils_cgh.get_scaling_factor(cfg, recon_amp, gt_amp)

            # calculate loss
            loss = loss_fn(s * recon_amp, gt_amp) / cfg.supervision.minibatch_size

            # accumulate gradients
            loss.backward()

            # for logging
            with torch.no_grad():
                loss_val_total += loss.item()

        optimizer.step()

        with torch.no_grad():
            tqdm_bar.set_description(f'loss_val: {loss_val_total:.6f}, s:{s:.6f}')
            if loss_val_total < best_loss_val:
                best_loss_val = loss_val_total
                best_slm_phase = slm_phase.clone()
                best_iter = t_iter  
                best_recon_amp = s * recon_amp.clone()
                best_s = s

    return {'loss_final': best_loss_val,
            'slm_phase': best_slm_phase,
            'iter': best_iter,
            'recon_amp': best_recon_amp,
            'scaling_factor': best_s
            }