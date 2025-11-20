import os
import imageio
import numpy as np
from omegaconf import DictConfig, OmegaConf
from cgh.utils_cgh import phase_encoding

CHANNEL_STR = {
    0: 'R',
    1: 'G',
    2: 'B',
}

def save_results(results, data, cfg):
    """Save reconstruction, phase, and ground truth amplitude images to disk."""
    out_dir = cfg.cgh.out_path
    os.makedirs(out_dir, exist_ok=True)

    # Save reconstructed amplitude
    out_amp_fname = (
        f'out_amp_{data["target_id"]}_{CHANNEL_STR[cfg.model.channel]}.png'
    )
    out_amp_img = (
        255 * results['recon_amp']
    ).clamp(0, 255).squeeze().cpu().detach().numpy().astype(np.uint8)
    imageio.imwrite(os.path.join(out_dir, out_amp_fname), out_amp_img)

    # Encode and save SLM phase
    slm_phase = results['slm_phase']
    slm_phase_encoded = phase_encoding(slm_phase, 'naive_8bits')

    for i in range(slm_phase_encoded.shape[0]):
        # Save phase image for each item in batch (using target_id as identifier)
        phase_fname = f'phase_{data["target_id"]}.png'
        phase_img = slm_phase_encoded[i].squeeze().cpu().detach().numpy().astype(np.uint8)
        # logging.info(f'Saving to {os.path.join(out_dir, phase_fname)} ...')
        imageio.imwrite(os.path.join(out_dir, phase_fname), phase_img)

    # Save ground truth amplitude
    target_fname = (
        f'gt_amp_{data["target_id"]}_{CHANNEL_STR[cfg.model.channel]}.png'
    )
    # logging.info(f"  -- target_fname: {target_fname}")
    # logging.info(f"  -- data['target_amp'].shape: {data['target_amp'].shape}")
    if len(data['target_amp'].shape) == 6:
        target_img = (data['target_amp']**2).mean((-2, -1), keepdims=True).sqrt()
    else:
        target_img = data['target_amp']
    target_img = (255 * target_img.squeeze().cpu().detach().numpy()).astype(np.uint8)
    # logging.info(f"  -- target_img: {target_img.shape}")
    imageio.imwrite(os.path.join(out_dir, target_fname), target_img)

def save_focal_stack(results, forward_model, data, cfg):
    """Save focal stack images to disk."""
    out_dir = cfg.cgh.out_path
    os.makedirs(out_dir, exist_ok=True)

    # Save focal stack
    for offset_dist in cfg.cgh.offset_dists_eval:
        slm_phase = results['slm_phase']
        recon_amp = results['scaling_factor'] * forward_model(slm_phase,
                                                    cfg.cgh.steered_angle,
                                                    prop_dist=forward_model.prop_dist0 + offset_dist).abs()
        if recon_amp.shape[0] > 1:
            recon_amp = (recon_amp**2).mean(0, keepdims=True).sqrt()  # time multiplexing
            
        focal_stack_fname = (
            f'focal_stack_{data["target_id"]}_{CHANNEL_STR[cfg.model.channel]}_{offset_dist:.3f}.png'
        )
        focal_stack_img = (
            255 * recon_amp
        ).clamp(0, 255).squeeze().cpu().detach().numpy().astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, focal_stack_fname), focal_stack_img)

    
