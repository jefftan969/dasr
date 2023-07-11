# Distilling Neural Fields for Real-Time Articulated Shape Reconstruction
Jeff Tan, Gengshan Yang, and Deva Ramanan (CVPR 2023)

### Goal: Train feed-forward shape and motion predictors by distilling differentiable rendering optimizers (e.g. category-level dynamic NeRFs)

### Paper
- Please see the latest version: [link](https://jefftan969.github.io/dasr/paper.pdf)

### Key Features
- Representation
  - Category-level rest shape (mesh)
  - Deformation (skeleton + linear blend skinning)
  - Global appearance embeddings
- Architecture
  - Category-level priors (derived from dynamic NeRF teacher)
  - Temporal encoder (smooth latent codes over time, avoid jitter)
  - Camera multiplex (improve optimization when many poses are likely)

### Get Started
- Requirements
  - Linux machine with at least 1 GPU (we tested on 3090s)
  - Conda: Follow [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to install Miniconda
- Set up the environment
  - Clone the repository. Then, create a conda environment with the required packages and download the data/checkpoints (about 20GB):
    ```
    git clone git@github.com:jefftan969/dasr.git --recursive
    cd dasr
    conda env create -f environment.yml
    conda activate dasr
    bash download.sh
    ```
- Running the evaluation code
  - Reproduce the numbers reported in the paper (about 12min on a 3090 GPU):
    ```
    python metrics_all_human.py
    ```
  - We will make available the training code, demos, and more extensive visualizations in July 2023.

### Timeline
- [x] Evaluation code
- [ ] Full release (training code, demos, visualizations, developer docs): July 2023

### References
- Our category-level priors are derived from [BANMo](https://github.com/facebookresearch/banmo)
- Our pre-processing pipeline is built upon the following open-sourced repos:
  - Segmentation: [MinVIS](https://github.com/NVlabs/MinVIS)
  - Features: [DensePose-CSE](https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/projects/DensePose/doc/DENSEPOSE_CSE.md)
  - Optical Flow: [VCNPlus](https://github.com/gengshan-y/rigidmask)
- If you use this project for your research, please consider citing our paper:
  ```
  @inproceedings{tan2023distilling,
    title={Distilling Neural Fields for Real-Time Articulated Shape Reconstruction},
    author={Tan, Jeff and Yang, Gengshan and Ramanan, Deva},
    booktitle={CVPR},
    year={2023}
  }
  ```
