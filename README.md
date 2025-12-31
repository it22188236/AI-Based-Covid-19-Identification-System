# Cough Phase Chaos Embeddings (CPCE)

**A Novel Physiological Biomarker for COVID-19 Detection from Cough Sounds**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Coswara](https://img.shields.io/badge/Dataset-Coswara-green)](https://github.com/iiscleap/Coswara-Data)

## Overview

**Cough Phase Chaos Embeddings (CPCE)** is a standalone, physics-based feature extraction pipeline that decomposes cough sounds into four physiological phases and extracts chaos theory signatures (fractal dimension, Lyapunov exponent, entropy) from each phase independently.

Unlike traditional spectral features (MFCC, Gammatone), CPCE captures **non-linear airflow dynamics** caused by COVID-19 lung fibrosis — most prominent in the **expulsion phase** — achieving **state-of-the-art performance** with an ultra-lightweight model.

### Key Advantages vs Baselines

| Metric               | CPCE (Ours)         | MFCC / Gammatone Baselines |
| -------------------- | ------------------- | -------------------------- |
| **Accuracy**         | **96.95%**          | 92–93%                     |
| **Model Parameters** | **85K**             | 1.2–2.1M                   |
| **Noise Robustness** | Invariant           | Degrades <15dB SNR         |
| **Interpretability** | Phase-wise heatmaps | Black-box                  |
| **Deployment**       | Smartphone-ready    | Heavy pipelines            |

## Physiological Foundation

A cough consists of 4 phases:

1. **Inspiration** – Breathing in
2. **Compression** – Building pressure (glottis closed)
3. **Expulsion** – Explosive airflow release (**COVID chaos dominant**)
4. **Glottis** – Voice box reopening

CPCE extracts chaos metrics per phase → **64D vector** (16D × 4 phases).

**Core Hypothesis**: COVID lung fibrosis creates irregular turbulence → unique fractal signatures in expulsion phase.

## Project Structure

cpce_project/
├── data/
│ ├── raw/ # Original Coswara (date folders + participant UUIDs)
│ ├── clean_audio/ # positive/ & negative/ folders
│ ├── splits/ # train.csv, val.csv, test.csv
│ └── processed/ # CPCE features, model, heatmaps
├── src/
│ ├── **init**.py # (Optional) Makes src a package
│ ├── setup_dataset.py # Organize Coswara → clean structure + splits
│ ├── audio_processing.py # Phase segmentation
│ ├── feature_extraction.py # Chaos metrics (robust Higuchi FD, Lyapunov, Entropy)
│ ├── model.py # Lightweight 1D CNN + training
│ └── utils.py # Data loading & saving
├── notebooks/
│ ├── 01_data_preparation.ipynb
│ ├── 02_feature_extraction.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_evaluation.ipynb # Clinical phase-wise heatmaps
├── main.py # End-to-end: extract → train → evaluate
└── README.md

## Quick Start

### 1. Dataset Setup (One-Time)

```bash
python src/setup_dataset.py --coswara_path "data/raw" --output_dir "data/clean_audio"
```
