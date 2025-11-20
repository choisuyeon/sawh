from .model_ours import ParameterizedWavePropagation
from .model_asm import ASM, OffAxisWrapper
from training.pl_wrapper import TrainingWrapper
import torch
import logging

PROPAGATIONS = {
    'asm': ASM,
    'param': ParameterizedWavePropagation,
}

def load_forward_model(cfg):
    wrapper = OffAxisWrapper(cfg, PROPAGATIONS[cfg.prop.name](cfg))

    if cfg.model_path is not None:
        # you can also specify the path for model config here and instantiate the model from there ...
        wrapper.load_state_dict(torch.load(cfg.model_path))
        logging.info(f"Loaded model from {cfg.model_path}")
    
    return wrapper