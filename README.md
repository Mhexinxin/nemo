
# NEMO: Noise-robust Excess Multi-task Optimization Strategy for Human-centric Acute Toxicity Prediction

This repository contains the official implementation of **NEMO** (**N**oise-robust **E**xcess **M**ulti-task **O**ptimization), a deep learning framework designed for human-centric acute toxicity prediction. NEMO introduces a noise-robust multi-task optimization strategy that improves robustness against molecular graph noise while leveraging cross-task knowledge sharing for more reliable toxicity estimation.

## 📋 Table of Contents

- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
  - [Training](#1-training)
  - [Evaluation](#2-evaluation)
- [License](#-license)

---

## 📂 Project Structure

```
NEMO/
├── architecture/           # Model definitions (GNN encoder, task heads, etc.)
├── data/                   # Raw CSV data storage
├── weighting/              # Multi-task weighting strategy implementations
├── config.py               # Experiment configuration and hyperparameters
├── dataset.py              # Dataset loading and preprocessing
├── main.py                 # Main entry point for training and evaluation
├── metric.py               # Evaluation metrics (RMSE, etc.)
├── record.py               # Logging and result recording utilities
├── setup.py                # Build/installation script
├── trainer.py              # Training loop and optimization logic
├── utils.py                # Shared utility functions
└── README.md               # This file
```

---

## 📦 Requirements

- Python 3.8+
- PyTorch >= 1.10
- PyTorch Geometric
- RDKit
- Pandas, NumPy, Tqdm

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Installation

Clone the repository and install:

```bash
git clone https://github.com/<your-username>/NEMO.git
cd NEMO
python setup.py install
```

---

## 🧪 Data Preparation

1. Place your raw data (SMILES format in CSV, e.g., `toxacute.csv`) in the `data/` directory.

2. Data loading and preprocessing are handled automatically by `dataset.py` at runtime. No separate preprocessing step is required.

---

## 🚀 Usage

The `main.py` script handles training and evaluation.

### 1. Training

To train NEMO with the noise-robust excess multi-task optimization strategy:

```bash
python main.py \
  --mode train \
  --dataset toxacute \
  --data_dir ./data \
  --weighting NEMO \
  --epochs 50 \
  --bs 64 \
  --gpu_id 0 \
```

### 2. Evaluation

To evaluate a trained NEMO model on the test set:

```bash
python main.py \
  --mode eval \
  --dataset toxacute \
  --data_dir ./data \
  --load_path ./ckpt \
  --ckpt_name NEMO_ckpt \
  --gpu_id 0
```

## 🪪 License

This project is released under the MIT License. See the `LICENSE` file for details.

---

