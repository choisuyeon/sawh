"""
Run Coherence Retrieval to retrieve low-rank approximated Mutual Intensity (MI) out of the waveguide from training data.

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

import os
import logging

import torch
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from utils.data_loader import PairsLoader
from models import load_forward_model
from training.pl_wrapper import TrainingWrapper, PSNRProgressBar, ForwardModelCheckpoint

import os
import logging

import torch
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from utils.data_loader import PairsLoader
from models import load_forward_model
from training.pl_wrapper import TrainingWrapper, PSNRProgressBar, SaveForwardModelCallback

@hydra.main(version_base=None, config_path="configs", config_name="config_training.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main training script entrypoint.

    Args:
        cfg (DictConfig): Hydra configuration loaded from YAML files.
    """

    # Set device (prefer GPU if available)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('medium')

    # Access configuration sections
    cfg_model = cfg.model

    # overrides the supported angle range from the dataset
    cfg_model.angle_range_min = cfg.dataset.angle_range_min
    cfg_model.angle_range_max = cfg.dataset.angle_range_max
    # TODO: align model parameters with the dataset parameters

    # Multi-GPU distributed setup
    num_device_count = torch.cuda.device_count()
    num_workers = cfg.num_workers * num_device_count
    cfg.lr_train = cfg.lr_train * num_device_count

    # Build training DataLoader
    train_loader = DataLoader(
        PairsLoader(cfg, dataset="train", num_phases=cfg.dataset_size),
        num_workers=num_workers,
        batch_size=cfg.batch_size,
    )

    # Build validation DataLoader (typically 10x fewer samples than training)
    val_loader = DataLoader(
        PairsLoader(cfg, dataset="val", num_phases=cfg.dataset_size // 10),
        num_workers=num_workers,
        batch_size=cfg.batch_size,
    )

    # Instantiate forward model and wrap for training
    forward_model = load_forward_model(cfg_model).to(dev)
    training_wrapper = TrainingWrapper(forward_model=forward_model, **cfg)

    # Optionally resume from checkpoint if path provided in config
    if getattr(cfg, "resume", None) is not None:
        checkpoint = torch.load(cfg.resume, map_location=dev)
        training_wrapper.load_state_dict(checkpoint["state_dict"])
        logging.info(f"Main: Using the model loaded from {cfg.resume} ...")

    # Callback to save models according to validation PSNR
    checkpoint_callback = ForwardModelCheckpoint(
    monitor="PSNR_validation_epoch",
    dirpath=cfg.checkpoint_path,
    filename="model-{epoch:02d}-{PSNR_validation_epoch:.2f}",
    every_n_epochs=1,
    save_top_k=2,
    mode="max",
    )   
    ddp = DDPStrategy(process_group_backend="gloo") if num_device_count > 1 else "auto"

    # Set up PyTorch Lightning Trainer
    # Custom progress bar callback to monitor PSNR_train_step in tqdm
    trainer = pl.Trainer(
        default_root_dir=cfg.checkpoint_path,
        accelerator="gpu",
        devices=num_device_count,
        enable_progress_bar=True,
        strategy=ddp,
        log_every_n_steps=cfg.log_every_n_steps,
        max_epochs=cfg.num_epochs,
        callbacks=[checkpoint_callback,
        PSNRProgressBar()],
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    # Train the model
    trainer.fit(training_wrapper, train_loader, val_loader)
    logging.info("  - Training completed ...")


if __name__ == "__main__":
    # Entry point for script execution
    main()
