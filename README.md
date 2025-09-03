<table>
  <tr>
    <td align="center" width="50%">
      <img src="images/tsne_classes.svg" alt="t-SNE Visualization by Class" width="100%">
      <br>
      <strong>By Class</strong>
    </td>
    <td align="center" width="50%">
      <img src="images/tsne_experts.svg" alt="t-SNE Visualization by Expert" width="100%">
      <br>
      <strong>By Expert</strong>
    </td>
  </tr>
</table>

## SMoE‑VAE: Mixture‑of‑Experts Variational Autoencoder

This repository contains the code for the accompanying paper on a Variational Autoencoder with a mixture of decoder experts (SMoE‑VAE). A shared encoder maps inputs to a latent space, while a gating network routes each sample to specialized decoder experts. Training combines standard VAE losses with load‑balancing and entropy terms to encourage both expert specialization and diverse expert usage. This repo provides training code, experiment scripts, and utilities to reproduce the paper’s figures.


## What’s inside

- `vae/vae_mixture_of_experts.py` — core model (encoder, decoders, gating) and the training/evaluation loop.
- `vae/train_vae.py` — experiment runner that reproduces most results by sweeping dataset fractions and number of experts.
- `vae/config.yaml` — configuration presets (`train`, `debug`) for datasets, hyperparameters, and model sizes.
- `vae/visualization_utils.py` — utilities for t‑SNE, reconstructions, expert analysis, and latent interpolations.
- `vae/visualizing.ipynb` — notebook to reproduce all plots in the paper using saved results.
- `vae/results/` — precomputed metrics and artifacts used to generate the figures.
- `snapshots/` or `vae/snapshots/` — outputs from runs (models, plots, embeddings), depending on where you launch scripts.

## Setup

Prerequisites:

- Python 3.9+ (CPU works; CUDA accelerates training if available)
- Windows PowerShell or any POSIX shell

Install dependencies in a virtual environment:

```powershell
# From the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer, you can install packages manually (torch, torchvision, numpy, matplotlib, scikit‑learn, tqdm, Pillow, requests, PyYAML).

## Datasets

Supported datasets: `mnist`, `fashion` (Fashion‑MNIST), `cifar` (converted to grayscale 28×28), `quickdraw`, and `combined` (MNIST + Fashion‑MNIST + QuickDraw). Data is downloaded automatically on first run into `vae/data/` (or `data/` relative to where you run).

QuickDraw sketches are fetched as `.npz` files and rasterized to images on the fly; ensure an active internet connection if using `quickdraw` or `combined`.

## How to run

You can train the model and reproduce nearly all experimental results via `train_vae.py`. For single runs, call `vae_mixture_of_experts.py` with a selected config.

### Option A — Reproduce experiment sweeps

`train_vae.py` loops over dataset percentages and numbers of experts (as used in the paper), saving models and plots for each run.

```powershell
# From repository root
.\.venv\Scripts\Activate.ps1
cd vae
python train_vae.py
```

Notes:

- The script reads `vae/config.yaml` (relative to the `vae/` folder) and uses the `train` block as a template.
- Outputs are saved under `vae/snapshots/<dataset>/moe_ld<...>_ne<...>_<...>/`.
- These sweeps can be time‑consuming; start with the `debug` preset if you just want a quick sanity check.

### Option B — Single training run

Edit `vae/config.yaml` to set hyperparameters in either `debug` (fast) or `train` (full) blocks, then run the main script:

```powershell
# From repository root (important: this path expectation matches the script)
.\.venv\Scripts\Activate.ps1
python vae\vae_mixture_of_experts.py
```

By default, the script’s `__main__` picks the `debug` config; switch it to `train` inside `vae_mixture_of_experts.py` if desired. Outputs are saved under `snapshots/<dataset>/...` at the repository root when launched this way.

## Producing the figures

- All visualization helpers live in `vae/visualization_utils.py` (latent space t‑SNE, reconstructions, expert usage/correlation, specialization, latent interpolations). These are called automatically at the end of training and save plots next to the model checkpoints.
- The notebook `vae/visualizing.ipynb` reproduces every plot from the paper using artifacts saved in `snapshots/` and precomputed data in `vae/results/`.
- The directory `vae/results/` contains all the data needed to regenerate the figures without re‑training.

## Outputs and checkpoints

Each run saves to a timestamped/config‑encoded subfolder, including:

- `vae_moe_model_epoch_*.pth`, `vae_moe_model_final.pth` — model weights
- `losses_detailed.png`, `train_recon_losses.npy`, `test_recon_losses.npy` — training curves
- `tsne_latent_space_*.png`, `reconstructions_moe.png`, `expert_*` plots — analysis and figure assets
- `embeddings/` — cached latent vectors and 2D projections for faster plotting

## Citation

If you use this code, please cite the accompanying paper describing SMoE‑VAE.
