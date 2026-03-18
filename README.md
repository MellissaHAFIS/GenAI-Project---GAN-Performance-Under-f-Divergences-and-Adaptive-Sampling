# f-Divergences and Adaptive Sampling Strategies for GANs

## Project Overview

This repository presents an empirical study on the impact of **f-divergences** and **sampling strategies** on **Generative Adversarial Networks (GANs)** trained on MNIST.

We compare:

* A **Vanilla GAN (baseline)**
* **f-GAN variants** using:

  * Jensen–Shannon divergence
  * Kullback–Leibler divergence
  * Pearson χ² divergence

We also evaluate advanced **sampling and post-training refinement strategies**, including:

* Normal Gaussian sampling
* Hard truncation
* Soft truncation
* **Discriminator Gradient Flow (DGFlow)**

The project is designed for **full reproducibility**: models can be trained, sampled, refined, and evaluated using provided scripts and pre-trained checkpoints.

---

## Repository Structure

```
root/
│
├── checkpoints/                  # Saved and pre-trained models (.pth)
│   ├── G_js.pth, D_js.pth
│   ├── G_kl.pth, D_kl.pth
│   ├── G_pearson.pth, D_pearson.pth
│   ├── G_Baseline.pth, D_Baseline.pth
│   └── cnn_mnist_features_extractor.pkl
│
├── samples/                      # Generated images (default output)
├── images_for_report/            # Example images (10 per model × sampling)
│
├── data/                         # MNIST dataset (auto-downloaded)
│
├── report.pdf            # Full project report (methodology, experiments, results, analysis)
├── slides.pdf            # Presentation slides (high-level overview and key findings)
│
├── model.py                      # Generator & Discriminator architectures
├── utils.py                      # Helper functions (I/O, loading, etc.)
│
├── train.py                      # Training script for Vanilla GAN
├── train_fgan.py                 # Training script for f-GAN variants
├── generate.py                   # Image generation & sampling strategies
│
├── fgan_utils.py                 # f-divergences, conjugate functions, losses
├── sampling_utils.py             # Truncation & DGFlow utilities
│
├── metrics.py                    # FID, Precision, Recall
├── train_feature_extractor.py    # CNN for MNIST feature extraction (FID)
│
├── evaluate_all.py               # Main evaluation script
├── select_10img.py               # Select 10 images (0–9) from generated samples
│
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

* Python ≥ 3.8
* PyTorch
* NumPy
* Matplotlib
* tqdm
* scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

The MNIST dataset is downloaded automatically when training or generating images.

---

## Training Models

### 1. Vanilla GAN (Baseline)

Train the classical GAN:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --epochs 100 --lr 0.0002 --batch_size 64 --gpus -1
```

Arguments:

* `--epochs`: number of training epochs
* `--lr`: learning rate
* `--batch_size`: batch size
* `--gpus`: number of GPUs (`-1` = all available)

---

### 2. f-GAN Variants

Train a GAN with a specific f-divergence:

```bash
python train_fgan.py --divergence js
```

Available divergences:

* `js` → Jensen–Shannon
* `kl` → Kullback–Leibler
* `pearson` → Pearson χ²

Optional arguments:

```bash
python train_fgan.py --divergence kl --epochs 100 --batch_size 64 --gpus -1
```

---

## Generating Images

Generate images from a trained model using different sampling strategies:

```bash
python generate.py
```

Key arguments:

```bash
python generate.py \
  --divergence js \
  --sampling dgflow \
  --batch_size 2048 \
  --dgflow_steps 25 \
  --dgflow_eta 0.1
```

### Model selection (`--divergence`)

* `Baseline` → Vanilla GAN
* `js` → f-GAN (Jensen–Shannon)
* `kl` → f-GAN (KL)
* `pearson` → f-GAN (Pearson χ²)

### Sampling strategies (`--sampling`)

* `normal` → Standard Gaussian sampling
* `hard` → Hard truncation
* `soft` → Soft truncation
* `dgflow` → Discriminator Gradient Flow refinement

### Sampling parameters

* `--threshold`: threshold for hard truncation
* `--psi`: psi value for soft truncation
* `--dgflow_steps`: number of DGFlow steps
* `--dgflow_eta`: DGFlow step size

Generated images are saved by default in:

```
samples/
```

---

## Evaluation

Evaluate generated samples using **FID, Precision, and Recall**:

```bash
python evaluate_all.py
```

Optional argument:

```bash
python evaluate_all.py --samples_dir samples
```

Notes:

* FID is computed using a **custom CNN trained on MNIST**
* The feature extractor is already trained and stored as:

  ```
  cnn_mnist_features_extractor.pkl
  ```

  No retraining is required.

---

## Reproducibility

* Pre-trained models are provided in `checkpoints/`
* All experiments are script-driven
* Sampling and refinement do **not** modify trained generators
* Results can be reproduced by rerunning training, generation, and evaluation scripts

---
Below is the **updated section** you can directly insert into your existing `README.md`.
It cleanly documents `report.pdf` and `slides.pdf` in a way that is clear for **recruiters, students, and evaluators**.

---

## Additional Documentation

### `report.pdf`

The report provides:

* Theoretical background on **GANs** and **f-divergences**
* Detailed explanation of **Vanilla GAN** and **f-GAN** objectives
* Description of **sampling strategies** and **DGFlow refinement**
* Experimental setup and evaluation protocol
* Quantitative results (FID, Precision, Recall)
* Qualitative analysis of generated samples

This document is suitable for academic review and technical evaluation.

### `slides.pdf`

The slides offer:

* A concise, visual overview of the project
* Key motivations and design choices
* Summary of experimental results
* Main conclusions and insights

---

## Intended Audience

* **Students** learning GANs and divergence-based objectives
* **Researchers** studying sampling and post-training refinement
* **Recruiters** evaluating applied deep learning and experimental rigor

This repository emphasizes **clarity**, **modularity**, and **end-to-end reproducibility**.
