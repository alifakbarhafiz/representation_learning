# From Linear to Self-Supervised Representations: A Comparative Study on RetinaMNIST

Compare PCA, convolutional autoencoders, and pretrained ViT-based encoders on RetinaMNIST using a shared MLP classifier.

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
![PyTorch](https://img.shields.io/badge/PyTorch-experiment-EE4C2C?logo=pytorch&logoColor=white)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](./tests)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[USERNAME]/[REPO]/blob/main/train.ipynb)

---

## Table of Contents

- [Overview](#overview)
- [Methods](#methods)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Google Colab](#google-colab)
- [Usage](#usage)
  - [Quick Start (Dry Run)](#quick-start-dry-run)
  - [Full Training](#full-training)
  - [MAE Weight Options](#mae-weight-options)
- [Configuration](#configuration)
  - [Optional: Official MAE checkpoints (research fork)](#optional-official-mae-checkpoints-research-fork)
- [Experiment Notebook](#experiment-notebook)
- [Results](#results)
- [Testing](#testing)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

Medical imaging often operates under **limited labeled data**. Representation learning transforms raw pixels into compact features that are easier for simple classifiers to use, potentially improving generalization without increasing model complexity on the head. Understanding how far **linear** methods, **learned supervised or unsupervised neural encoders**, and **large-scale self-supervised** encodings differ on a modest clinical-looking benchmark is practically useful when choosing pipelines for student-scale compute.

This repository implements a **controlled comparison** on **[RetinaMNIST](https://medmnist.com/)**: the same downstream **multi-layer perceptron (MLP)** reads features from representations produced by **PCA**, a trained **convolutional autoencoder (AE)**, or a **pretrained ViT encoder loaded via [`timm`](https://github.com/huggingface/pytorch-image-models)** (used here as an MAE-style self-supervised vision backbone surrogate). Holding the classifier architecture fixed isolates differences in representation quality rather than classifier capacity.

This project includes a **`DRY_RUN` mode** that subsamples RetinaMNIST to **200 / 50 / 50** train, validation, and test images so you can validate the pipeline end-to-end quickly before committing GPU time to full training.

**Key findings:** [TO BE FILLED AFTER RESULTS]

---

## Methods

Representations differ in linearity, source of supervision, and whether pretrained knowledge is leveraged. Dimensions are aligned with defaults in [`configs/config.py`](configs/config.py): PCA and AE use **128**-dimensional features; ViT-Base CLS tokens typically have **768** dimensions.

| Method | Type | Pretrained | Feature Dim | Training Required |
|--------|------|------------|-------------|-------------------|
| PCA | Linear (unsupervised / no labels for fit beyond pixels) | No | `PCA_N_COMPONENTS` (default 128) | Fit only on train split pixels |
| AE | Convolutional encoder–decoder reconstruction | No | `AE_LATENT_DIM` (default 128) | Yes (train AE on train loader) |
| MAE / ViT | Self-supervised / ImageNet pretrained encoder (ViT backbone via timm; common MAE-style starting point for research narratives) | Yes (ViT pretrained weights loaded by timm) | Typically **768** (ViT-B CLS) | Encoder frozen; classifier trained on features |

**Shared MLP classifier:** For each representation, the same blueprint applies: configurable hidden widths (`MLP_HIDDEN_DIMS`, default `[256, 128]`), each block as **Linear → BatchNorm1d → ReLU → Dropout(0.3)**, capped by a final **Linear → 5 logits** for RetinaMNIST. Training minimizes cross-entropy on train features with validation monitoring on held-out features (`models/classifier.py`).

---

## Project Structure

```plaintext
project/
├── data/
│   └── dataset.py              # Loads RetinaMNIST (medmnist), normalization to [0,1], loaders + arrays, optional dry-run subsampling
├── models/
│   ├── autoencoder.py          # Conv AE: encoder-decoder pair, latent vector, reconstruction (MSE-friendly)
│   ├── mae_encoder.py          # timm pretrained ViT feature extractor factory (CLS features, frozen)
│   └── classifier.py           # MLP classifier, train_epoch / evaluate / train_mlp utilities
├── representations/
│   ├── pca_repr.py             # Fit PCA on train images only; transform train/val/test
│   ├── ae_repr.py              # Train AE end-to-end; extract encoder embeddings for splits
│   └── mae_repr.py             # Run MAE/ViT feature extraction timings for splits
├── utils/
│   ├── timer.py                # Wall-clock Timer + log_time helper for experiments
│   ├── metrics.py              # Accuracy, macro F1, per-class accuracy helpers + ASCII-style table printer
│   └── visualization.py        # Saves training curves, timing bars, reconstructions, t-SNE, final comparison charts
├── configs/
│   └── config.py               # Frozen Config dataclass; global seed setter
├── tests/
│   ├── conftest.py             # Synthetic dataset + session fixtures for fast offline tests
│   ├── test_dataset.py         # Sizes, normalization, loader tensor shapes for dry-run semantics
│   ├── test_autoencoder.py     # Tensor shapes, Sigmoid bounds, gradients / loss sanity
│   ├── test_classifier.py      # logits shape, epoch APIs, train_mlp histories
│   ├── test_pca_repr.py        # PCA dict keys, leakage checks, timings, NaN checks
│   ├── test_ae_repr.py         # Feature shapes versus dry-run splits, histories
│   ├── test_mae_repr.py        # MAE paths with mocked pretrained loading (offline)
│   ├── test_timer.py           # Timing utilities and printable log format
│   ├── test_metrics.py         # Matches sklearn aggregates on controlled labels
│   └── test_visualization.py   # Headless plotting (Agg) saves to tmp_path
├── requirements.txt            # Pin-compatible runtime deps (torch stack, sklearn, plotting, etc.)
├── requirements-dev.txt      # pytest (optional dev installs)
├── pytest.ini                 # Discovery + pythonpath for package imports during tests
└── train.ipynb                 # Orchestrating Colab-first notebook wiring all sections together
```

---

## Installation

### Local Setup

```bash
git clone https://github.com/[USERNAME]/[REPO].git
cd [REPO]/project
python -m venv .venv
# Windows PowerShell example:
# .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional: run tests locally
python -m pip install -r requirements-dev.txt
```

- **Python 3.8+** is recommended (matching modern PyTorch and scientific stack wheels).
- The first dataset download (~RetinaMNIST cache under `project/data_cache/`) requires network access.

### Google Colab

1. Open [`train.ipynb`](train.ipynb) in Colab via the badge above (update `[USERNAME]` / `[REPO]` to your fork when publishing).
2. **Upload vs. repo:** Upload the whole `project/` folder to Drive, or **`git clone`** your repo in a notebook cell into `/content/` and **add the project folder to `sys.path`** exactly as Cell 1 in the notebook already illustrates.
3. Run the installs in **Section 0**:

```python
# Install dependencies (Colab)
!pip -q install medmnist timm umap-learn ipywidgets seaborn tqdm scikit-learn

# Optionally pin torch stacks to Colab defaults or pin versions aligned with requirements.txt:
# !pip install -q torch torchvision
```

Pick **one upload strategy**:

- **`git clone`:** clone into `/content/`, then `sys.path.insert(0, "/content/[REPO]/project")`.
- **Google Drive:** mount Drive, browse to where you stored `project/`, prepend that path similarly.

RetinaMNIST will download automatically the first time `medmnist` is invoked.

---

## Usage

### Quick Start (Dry Run)

In [`train.ipynb`](train.ipynb), **Section 1 — Config & Dry Run Toggle**, set:

```python
DRY_RUN = True   # ← CHANGE THIS TO FALSE FOR FULL TRAINING
```

This overrides configuration so only **200 train / 50 val / 50 test** images are loaded. Typical wall time on dry run is modest (often ~5 minutes depending on GPU and cached downloads), which is sufficient to verify imports, PCA, short AE epochs, feature extraction, MLP training, figures, and saved outputs under `./outputs/`.

### Full Training

Set:

```python
DRY_RUN = False
```

Expect longer runs (more AE and MLP epochs as in `Config`). On a **Colab T4 GPU**, a full pass is often on the order of **tens of minutes to a couple of hours**, depending on batch size, cache state, and whether ViT weights are already cached. CPU-only full runs are possible but significantly slower.

### MAE Weight Options

**As implemented in this repository**, the MAE branch uses **`timm.create_model(MAE_MODEL_NAME, pretrained=True, num_classes=0)`** and reads **CLS token features** from `forward_features` (`models/mae_encoder.py`). This is the default path in the notebook and requires no manual checkpoint management beyond the first timm weight download.

**Option A — timm (default)**  
- Set `MAE_MODEL_NAME` to a supported ViT identifier (default: `vit_base_patch16_224`).  
- timm handles weight resolution; treat the first run as a one-time download.

**Option B — Official Facebook MAE weights (extension)**  
This repository’s `Config` **does not yet** expose `USE_OFFICIAL_MAE` or `MAE_CHECKPOINT_PATH`; wiring official weights is a small, explicit fork task. If you extend the code, the intended workflow is:

1. Clone or read the official implementation: [facebookresearch/mae](https://github.com/facebookresearch/mae).
2. Download **`mae_pretrain_vit_base.pth`** (or the matching checkpoint named in the MAE README for ViT-Base).
3. Store it under a stable path (e.g. `project/checkpoints/mae_pretrain_vit_base.pth`).
4. Add fields such as `USE_OFFICIAL_MAE: bool` and `MAE_CHECKPOINT_PATH: str` to your fork’s `Config`, and load the state dict in `models/mae_encoder.py` instead of (or in addition to) timm’s `pretrained=True`.

**Rigor note:** timm’s “pretrained ViT” is excellent for reproducible course-scale experiments; **strict Facebook MAE checkpoint loading** better matches the original MAE training recipe at the cost of more engineering and exact architecture alignment.

---

## Configuration

All hyperparameters live in the frozen dataclass `Config` in [`configs/config.py`](configs/config.py).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DRY_RUN` | `False` | If `True`, subsample to 200/50/50 train/val/test after loading RetinaMNIST (can be overridden from the notebook). |
| `SEED` | `42` | Global random seed (Python, NumPy, PyTorch; CUDA handled when available). |
| `BATCH_SIZE` | `64` | DataLoader batch size for AE image batches and MLP feature batches. |
| `AE_EPOCHS` | `30` | Integer number of full passes over the AE training loader. |
| `AE_LR` | `1e-3` | Float learning rate for Adam when training the autoencoder. |
| `AE_LATENT_DIM` | `128` | Integer latent dimensionality of the convolutional encoder output. |
| `PCA_N_COMPONENTS` | `128` | Integer number of retained principal components. |
| `MLP_HIDDEN_DIMS` | `[256, 128]` | List of integers: hidden layer widths for the shared MLP head. |
| `MLP_EPOCHS` | `50` | Integer training epochs for the downstream classifier per representation. |
| `MLP_LR` | `1e-3` | Float learning rate for Adam when training the MLP. |
| `MAE_MODEL_NAME` | `"vit_base_patch16_224"` | String timm model name for the frozen ViT encoder. |
| `DEVICE` | `"cuda"` if available else `"cpu"` | String device for PyTorch modules; auto-detected at `Config` construction time. |

### Optional: Official MAE checkpoints (research fork)

If you extend this project to load **Facebook MAE** weights directly, add (for example):

| Parameter (suggested) | Example | Description |
|----------------------|---------|-------------|
| `USE_OFFICIAL_MAE` | `False` | Boolean switch between timm pretrained weights and a local MAE checkpoint. |
| `MAE_CHECKPOINT_PATH` | `""` or path string | Filesystem path to `mae_pretrain_vit_base.pth` (or compatible). |

These fields are **not present** in the stock `Config`; they are documented here for forks and thesis/code-release clarity.

---

## Experiment Notebook

The central entry point is [`train.ipynb`](train.ipynb). Each section is designed to run top-to-bottom on Colab or locally.

| Section | Purpose | Expected output |
|---------|---------|-----------------|
| **0 — Setup** | Installs dependencies, wires `sys.path`, imports project modules | Ready kernel; `project` on path |
| **1 — Config & Dry Run Toggle** | Sets `DRY_RUN`, prints mode banner | Clear console banner for dry vs full run |
| **2 — Data Loading** | Loads RetinaMNIST with chosen dry-run setting; shows stats and sample grid | Counts, class histograms, sample figure under `outputs/` |
| **3 — PCA Representations** | Fits PCA on train only; logs fit/transform times; explained variance plot | Timings, cumulative variance curve |
| **4 — Autoencoder Training** | Trains AE with progress; loss curve, epoch times, reconstructions | AE checkpoints in memory; PNG exports |
| **5 — MAE Feature Extraction** | Builds timm ViT extractor; logs per-split extraction time | Feature arrays + timing table |
| **6 — MLP Classifier Training** | Trains three MLPs (PCA / AE / MAE features) with histories | Per-method loss/acc curves and timing charts |
| **7 — Evaluation on Test Set** | Computes accuracy, macro F1, per-class accuracy; unified table | Printed comparison + metrics dict |
| **8 — Visualizations** | t-SNE embeddings and bar comparison of methods | PNGs in `./outputs/` |
| **9 — Summary** | Human-readable best-method summary | One-line takeaway in the console |

---

## Results

Results will be updated after full training run.

| Method | Test Acc | Macro F1 | Feature Time | Train Time |
|--------|----------|----------|--------------|------------|
| PCA    | -        | -        | -            | -          |
| AE     | -        | -        | -            | -          |
| MAE    | -        | -        | -            | -          |

After running the notebook, reference real figures under **`./outputs/`**, for example:

- `./outputs/samples_grid.png`
- `./outputs/pca_explained_variance.png`
- `./outputs/mlp_curves_{pca,ae,mae}.png`
- `./outputs/tsne_test_all.png`
- `./outputs/final_comparison.png`

To embed previews in GitHub READMEs, uncomment or add standard markdown images after your run, for example:

```markdown
![](./outputs/tsne_test_all.png)
```

---

## Testing

Install the test runner:

```bash
pip install -r requirements-dev.txt
# or explicitly:
pip install pytest
```

Run the full suite (from the `project/` directory):

```bash
pytest tests/ -v
```

Run a single file:

```bash
pytest tests/test_autoencoder.py -v
```

Collect tests without executing:

```bash
pytest tests/ --co -q
```

| Test module | Covers |
|-------------|--------|
| `test_dataset.py` | Dry-run sizes, flattened shapes, loaders, normalization, NaNs |
| `test_autoencoder.py` | Encoder/decoder shapes, Sigmoid bounds, gradient flow sanity |
| `test_classifier.py` | MLP logits, `train_epoch` / `evaluate`, `train_mlp` history contracts |
| `test_pca_repr.py` | PCA dict API, leakage-free mean alignment, timings |
| `test_ae_repr.py` | AE feature shapes and history keys after short training fixtures |
| `test_mae_repr.py` | MAE wrappers with mocks (no pretrained weight downloads) |
| `test_timer.py` | Timer behavior and printable `log_time` format |
| `test_metrics.py` | Metrics agreement with sklearn on controlled splits |
| `test_visualization.py` | Figures save correctly under matplotlib `Agg` (tmp dirs) |

- **Offline / CPU:** Automated tests intentionally **mock pretrained loading** where needed and use **CPU-only** synthetic or small-batch paths; they do **not** download ViT checkpoints during pytest.
- **Note:** pytest configuration lives in [`pytest.ini`](pytest.ini); session fixtures reuse short trained models for efficiency.

---

## Visualizations

All experiment figures are saved to **`./outputs/`** relative to where you execute the notebook (usually the `project/` working directory):

| Artifact | Description |
|----------|-------------|
| Sample image grid | First look at normalized RetinaMNIST patches and labels |
| PCA explained variance curve | Shows retained variance versus number of components |
| AE loss curve | Reconstruction objective over AE epochs |
| AE epoch timing bars | Wall-clock variability per AE epoch |
| AE reconstructions | Side-by-side originals and decoded reconstructions |
| Per-method MLP training curves | Train/val loss and accuracy traces with best-epoch markers |
| MLP epoch timing bars | Per-epoch durations for classifier training |
| t-SNE feature panels | Separate low-dimensional embeddings per representation, colored by class |
| Final bar chart | Grouped bars for accuracy and macro-F1 across methods |

---

## Requirements

Pinned or lower-bounded runtime dependencies appear in [`requirements.txt`](requirements.txt). Development testing adds [`requirements-dev.txt`](requirements-dev.txt).

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Autoencoder training, classifier, tensor IO |
| `numpy` | Array pipeline for PCA/features/metrics glue |
| `scikit-learn` | PCA decomposition, metrics cross-checks, t-SNE |
| `matplotlib`, `seaborn` | Static plots and aesthetics |
| `tqdm`, `ipywidgets` | Notebook progress UX |
| `medmnist` | RetinaMNIST dataset loaders |
| `timm` | Pretrained ViT encoder construction |
| `umap-learn` | Optional exploratory embeddings (referenced in installs; PCA/AE workflows focus on sklearn t-SNE in shipped plots) |

---

## Citation

If you use this code or framing, please cite this project (fill in author details and canonical URL):

```bibtex
@misc{yourname2025retinamnist_repr,
  title        = {From Linear to Self-Supervised Representations: A Comparative Study on RetinaMNIST},
  author       = {[Author Name]},
  year         = {2025},
  howpublished = {GitHub repository},
  url          = {[GitHub URL]}
}
```

**MedMNIST / RetinaMNIST benchmark:**

```bibtex
@article{yang2023medmnist,
  title   = {{MedMNIST} v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author  = {Yang, Jiancheng and Shi, Ruiyi and Wei, Donglai and others},
  journal = {Scientific Data},
  year    = {2023},
  volume  = {10},
  number  = {1},
  doi     = {10.1038/s41597-023-02511-9}
}
```

**Masked Autoencoders:**

```bibtex
@inproceedings{he2022masked,
  title        = {Masked Autoencoders Are Scalable Vision Learners},
  author       = {He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yang and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle    = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages        = {16000--16009},
  year         = {2022}
}
```

**timm (PyTorch Image Models):**

```bibtex
@misc{rw2019timm,
  author = {Wightman, Ross},
  title  = {{PyTorch Image Models}},
  year   = {2019},
  url    = {https://github.com/huggingface/pytorch-image-models}
}
```

---

## License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this software with attribution. Full standard text:

[MIT License](https://opensource.org/licenses/MIT)

---

## Acknowledgements

- **[MedMNIST](https://medmnist.com/)** — RetinaMNIST and standardized biomedical MNIST-class benchmarks simplify rigorous small-data comparisons.
- **[Facebook Research MAE](https://github.com/facebookresearch/mae)** — reference implementation and pretrained checkpoints motivating the masked-autoencoder lineage in vision representation learning.
- **[timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)** — pretrained ViT backbones (`MAE_MODEL_NAME`) with a pragmatic API for reproducible experimentation.
