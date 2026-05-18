# DCML: Double Causal Multimodal Learning for E-Commerce Recommendation Analysis

Official implementation of the paper:

> DCML: A Double Causal Multimodal Learning Framework for Hierarchical Causal Decomposition of E-Commerce Recommendation Signals

---

# Overview

DCML is a multimodal causal inference framework designed to disentangle:

- System-1 conformity-driven marketing signals
- System-2 interest-driven informational signals

from multimodal e-commerce content.

The framework integrates:

- Large Multimodal Models (LMMs)
- Hierarchical atomic feature extraction
- Double Machine Learning (DML)
- Orthogonalized causal estimation
- Mediation analysis
- Placebo testing
- Robustness value analysis

across multi-stage recommendation funnels:

- Click
- Cart
- Purchase

---

# Framework Pipeline

## Stage 1: Multimodal Atomic Extraction

Using Qwen2-VL to extract:

### Marketing Atoms


and aggregate them into macro-level latent treatments.

---

## Stage 2: DML Residualization

Train nuisance models:

- Treatment models: `E[T|X]`
- Outcome models: `E[Y|X]`

using LightGBM.

Generate orthogonal residual representations for causal estimation.

---

## Stage 3: Orthogonalized Causal Estimation

Perform:

- Gram-Schmidt orthogonalization
- ATE estimation
- CATE estimation

for:

- Macro-level treatments
- Atomic-level treatments

---

## Stage 4: Mediation Analysis

Bootstrap mediation analysis with:

- indirect effects
- confidence intervals
- Bonferroni correction

---

## Stage 5: Robustness & Placebo Tests

Including:

- placebo treatment shuffling
- robustness value (RV) analysis
- baseline estimator comparisons

---

# Repository Structure

```text
DCML/
│
├── model/
│   ├── MCRE/
│   ├── DML/
│   └── ...
│
├── data_preprocessing/
│
├── configs/
│   └── config.yaml
│
├── requirements.txt
├── LICENSE
└── README.md



# Environment
	•	Python 3.10+
	•	CUDA 11.8+
	•	PyTorch 2.x


# Installation
git clone https://github.com/Angelaviviwong-3/DCML.git

cd DCML

pip install -r requirements.txt







# Datasets

This repository does NOT include the datasets used in the paper.

## Suning Dataset

The Suning dataset is a proprietary industrial dataset provided through enterprise collaboration and is not publicly distributable.

Researchers interested in accessing the dataset for academic purposes may contact the authors for potential research collaboration and data access discussion.

## Public Components

The framework is compatible with standard e-commerce recommendation datasets containing:

- product metadata
- product images
- review text
- user interaction logs

stored in parquet or tabular formats.






# License
## requirements.txt

```text
pandas
numpy
scipy
scikit-learn
statsmodels
lightgbm
joblib
matplotlib
seaborn
tqdm
transformers
accelerate
torch
torchvision
qwen-vl-utils
pyarrow
fastparquet
