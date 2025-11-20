"""
Run CGH algorithms with the given configuration.

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

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cgh import load_cgh_method, load_phase_init
import cgh.save_and_load as save_and_load
import cgh.utils_cgh as utils_cgh

from models import load_forward_model
from utils.data_loader import load_data_loader
import utils.utils as utils

def index_integer(lst, i):
    return lst[int(i)] if isinstance(i, int) else i
OmegaConf.register_new_resolver("index_integer", index_integer)

@hydra.main(version_base=None, config_path="configs", config_name="config_cgh.yaml")
def main(cfg: DictConfig) -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('medium')
    
    utils_cgh.check_cgh_config(cfg)
    data_loader = load_data_loader(cfg, dev)
    algorithm = load_cgh_method(cfg.cgh)
    run_phase_init = load_phase_init(cfg)
    forward_model = load_forward_model(cfg.model).to(dev)

    for i, data in enumerate(data_loader):
        logging.info(f'Main: Running CGH for data {i} ...')
        initial_phase = run_phase_init(cfg).to(dev)
        results = algorithm(data['target_amp'], 
                            initial_phase, 
                            forward_model, 
                            cfg.cgh)
        save_and_load.save_results(results, data, cfg)
        save_and_load.save_focal_stack(results, forward_model, data, cfg)

if __name__ == "__main__":
    main()