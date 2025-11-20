# Synthetic Aperture Waveguide Holography | Nature Photonics

### [Project Page](https://www.computationalimaging.org/publications/synthetic-aperture-waveguide-holography/) | [Paper](https://www.nature.com/articles/s41566-025-01718-w)

[Suyeon Choi*](http://choisuyeon.github.io/), [Changwon Jang*](https://www.linkedin.com/in/changwonjang), [Douglas Lanman](https://www.linkedin.com/in/dlanman), [Gordon Wetzstein](http://stanford.edu/~gordonwz/)

\*Denotes equal contribution.

This repository contains the scripts associated with the Nature Photonics article, "Synthetic aperture waveguide holography for compact mixed-reality displays with large étendue."

## Getting Started

You can set up a conda environment with all dependencies as follows:

```bash
conda env create -f env.yml
conda activate sah
```

## Overview

This repository is organized into two main components:
1. **Computer-Generated Holography (CGH) Framework for Synthetic Aperture Waveguide Holography**
2. **Partially Coherent Implicit Neural Waveguide Model**

Each part resides in its own folder ([`cgh`](https://github.com/choisuyeon/sawh/blob/master/cgh), [`models`](https://github.com/choisuyeon/sawh/blob/master/models)). Configuration and script files are in the [`configs`](https://github.com/choisuyeon/sawh/blob/master/configs) and [`scripts`](https://github.com/choisuyeon/sawh/blob/master/scripts) folders, respectively.

## Running the Code

To run CGH, place your LF data (e.g., [OLAS light field](https://drive.google.com/file/d/1qmWAVQQRNAbus2koYrFIGzaphnvhye_t/view)) in the `data_lf` folder and set the LF parameters (see [`configs/cgh/lf_data/olas.yaml`](https://github.com/choisuyeon/sawh/blob/master/configs/cgh/lf_data/olas.yaml)). Then, execute the following command:

```bash
# Run coherence retrieval of rank 2
sh scripts/run_cgh.sh 0 2
```

To train the partially coherent model, capture pairs of phase and intensity images to `dataset/phase` and `dataset/captured`, then run:

```bash
# Run coherence retrieval of rank 2
sh scripts/train_model.sh 0 2
```

To run CGH with the partially coherent model, execute:

```bash
sh scripts/run_cgh_model.sh 0 $model_path 2
```

## Citation

If you find our work useful in your research, please cite:

```
@article{choi2025synthetic,
  title={Synthetic aperture waveguide holography for compact mixed-reality displays with large {\'e}tendue},
  author={Choi, Suyeon and Jang, Changwon and Lanman, Douglas and Wetzstein, Gordon},
  journal={Nature Photonics},
  pages={1--10},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## Contributions

I am actively maintaining this repository and welcome contributions from the community! If you have suggestions or improvements, please feel free to open a pull request. For any questions, please contact me at [suyeon@stanford.edu](mailto:suyeon@stanford.edu).