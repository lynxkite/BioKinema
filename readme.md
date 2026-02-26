# BioKinema: Physically Grounded Generative Modeling of All-Atom Biomolecular Dynamics

[![Paper](https://img.shields.io/badge/Paper-bioRxiv-green)](https://www.biorxiv.org/content/10.64898/2026.02.15.705956v1)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green?style=flat-square)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red?style=flat-square)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![GitHub Link](https://img.shields.io/badge/GitHub-blue?style=flat-square&logo=github)](https://github.com/IDEA-XL/BioKinema)

## Introduction

**BioKinema** is a physically grounded generative model that predicts continuous-time, all-atom biomolecular trajectories at a fraction of the cost of traditional molecular dynamics (MD) simulations.

Predicting the kinetic pathways of biomolecular systems at all-atom resolution is crucial for understanding protein function and drug efficacy, yet this task is hindered by the immense computational cost of conventional MD simulations. While deep learning has revolutionized static structure prediction and equilibrium ensemble sampling, simulating the kinetics of conformational transitions remains a critical challenge.

BioKinema addresses these challenges through two key innovations. First, it utilizes a **physically grounded diffusion architecture** with temporal attention mechanisms derived from Langevin dynamics. Second, it employs a **hierarchical forecasting-and-interpolation strategy** to overcome the error accumulation that often plagues long-horizon generation.

Through extensive validation, we demonstrate that BioKinema generates physically stable and dynamically accurate trajectories suitable for rigorous downstream analysis. The model captures key conformational transitions related to protein function. For protein-ligand complex systems, it successfully elucidates mechanisms such as ligand-driven conformational changes and allosteric interactions. Furthermore, BioKinema leverages enhanced sampling data to predict rare kinetic events, emerging as a powerful tool for estimating ligand unbinding pathways.

## Installation

```bash
# Clone the repository
git clone https://github.com/IDEA-XL/BioKinema.git
cd BioKinema

# Install dependencies
conda env create -f environment.yml
conda activate biokinema
```

### Optional Dependencies

For optimal performance, you may also configure the following environment variables in **inference.sh**:

```bash
export CUTLASS_PATH=/path/to/cutlass
export CUDA_HOME=/path/to/cuda
```

### Download checkpoint

Trained checkpoint is avaliable in https://huggingface.co/fengb/BioKinema

## Usage

### Basic Inference

Run trajectory generation using the inference script:

```bash
bash inference.sh \
    --checkpoint_path ./checkpoints/biokinema.pt \
    --dump_dir ./output \
    --input_file ./examples/example.pdb
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--checkpoint_path` | Path to the trained model checkpoint |
| `--dump_dir` | Directory for saving output trajectories |
| `--input_file` | Path to input **.pdb** or **.cif** file containing initial structure |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--N_sample` | 1 | Number of trajectory samples to generate |
| `--coarse_frame_num` | 50 | Number of coarse frames including the initial structure |
| `--coarse_interval` | 2 | Temporal spacing between coarse frames (in nanosecond) |
| `--fine_frame_num` | 1 | Number of sub-intervals per coarse interval (1 means no interpolation) |
| `--W_H` | 1 | History window size for coarse forecasting |
| `--W_G` | 50 | Generation window size for coarse forecasting |
| `--N_step` | 20 | Number of diffusion steps |
| `--N_cycle` | 10 | Number of model cycles |
| `--seed` | 101 | Random seed |
| `--lambda` | 1.75 | Noise scale parameter |
| `--eta` | 1.5 | Step scale parameter |

### Example Commands

**Standard trajectory generation:**

```bash
bash inference.sh \
    --checkpoint_path ./checkpoints/biokinema.pt \
    --dump_dir ./results/trajectory \
    --input_file ./examples/example.pdb \
    --coarse_frame_num 50 \
    --coarse_interval 1 \
    --fine_frame_num 1
```

**Generation with interpolation:**

```bash
bash inference.sh \
    --checkpoint_path ./checkpoints/biokinema.pt \
    --dump_dir ./results/highres \
    --input_file ./examples/example.pdb \
    --coarse_frame_num 50 \
    --coarse_interval 10 \
    --fine_frame_num 10
```

### Hierarchical Generation Pipeline

BioKinema employs a two-stage hierarchical generation approach.

**Stage 1: Coarse-grained Forecasting** generates the coarse-grained trajectory at coarse temporal resolution. The model autoregressively predicts future frames using a sliding window approach, where `W_H` controls the history context and `W_G` determines the batch size for each generation step.

**Stage 2: Fine-grained Interpolation** refines the trajectory by filling in intermediate frames between coarse keyframes. This stage is activated when `fine_frame_num > 1`, producing smoother and more temporally detailed trajectories.

## Citation

```bibtex
@article{feng2026physically,
  title={Physically Grounded Generative Modeling of All-Atom Biomolecular Dynamics},
  author={Feng, Bin and Zhang, Jiying and Zhang, Xinni and Zhang, Ming and Barth, Patrick and Liu, Zijing and Li, Yu},
  journal={bioRxiv},
  pages={2026--02},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Acknowledgements

This project was built based on [Protenix](https://github.com/bytedance/Protenix), an open-source biomolecular structure prediction framework developed by ByteDance. 

## Contact

For questions or collaborations, please open an issue or contact us at [fengbin@idea.edu.cn](mailto:fengbin@idea.edu.cn).