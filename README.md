# EEG Laser-Evoked Nociceptive Pain Classification (`eeg_pain_research`)

An end-to-end Python pipeline for spectral feature extraction (Absolute Power, Relative Power, and ERD/ERS%) and machine learning classification of laser-evoked pain intensity from 62-channel electroencephalography (EEG) recordings ($N = 51$ subjects).

---

## Overview & Key Features

* **MATLAB-Free `.mat` v7.3 MCOS Parser:** Directly extracts `Eventos1` UTF-16LE event strings from binary MATLAB v7.3 subsystems without requiring MATLAB Engine or `h5py` workarounds.
* **Leakage-Free Spectral Windowing:** Strictly isolates pre-stimulus baseline (`[-1.0 s, 0.0 s]`) and post-stimulus response (`[0.0 s, +1.0 s]`) windows sampled at $500\text{ Hz}$ using Welch's Power Spectral Density (PSD) estimation.
* **Spatial Dimensionality Analysis:** Compares full-head montage (**62 channels**) against a targeted **20-channel somatosensory/motor subset**.
* **Subjective Pain Stratification:** Labels trials into Low, Medium, and High perceived pain levels using subject-specific tolerance duration terciles (33rd and 66th percentiles).
* **Dual Validation Framework:** Evaluates both personalized **Intra-Subject** models and cross-subject generalization (**70/30 Group Hold-Out** and **Leave-One-Subject-Out [LOSO]**), fitting `StandardScaler` strictly within training folds to prevent data leakage.

---

## Methodology

### 1. Frequency Bands & Spectral Features
PSD is integrated via trapezoidal numerical integration across six canonical frequency bands ($1.5\text{–}60.0\text{ Hz}$):

| Band | Frequency Range (Hz) | Extracted Metrics |
| :--- | :---: | :--- |
| **Delta** | $1.5 - 4.0$ | Absolute Power ($\mu\text{V}^2$), Relative Power (%), ERD/ERS (%) |
| **Theta** | $4.1 - 8.0$ | Absolute Power ($\mu\text{V}^2$), Relative Power (%), ERD/ERS (%) |
| **Alpha** | $8.1 - 12.0$ | Absolute Power ($\mu\text{V}^2$), Relative Power (%), ERD/ERS (%) |
| **Beta 1** | $12.1 - 20.0$ | Absolute Power ($\mu\text{V}^2$), Relative Power (%), ERD/ERS (%) |
| **Beta 2** | $20.1 - 31.0$ | Absolute Power ($\mu\text{V}^2$), Relative Power (%), ERD/ERS (%) |
| **Gamma** | $31.1 - 60.0$ | Absolute Power ($\mu\text{V}^2$), Relative Power (%), ERD/ERS (%) |

* **Relative Power (%):** $\frac{P_{\text{band}}}{P_{1.5\text{–}60\text{ Hz}}} \times 100$
* **ERD/ERS (%):** $\frac{P_{\text{post}} - P_{\text{pre}}}{P_{\text{pre}}} \times 100$

### 2. Classification Scenarios & Models
Four classifier families are benchmarked across binary (*Pain vs. No-Pain*, *Low vs. High*) and multi-class (*3 Pain Levels*) tasks:
* **K-Nearest Neighbors (KNN):** $k = 5$, cosine distance metric.
* **Support Vector Machine (SVM):** Linear kernel.
* **Random Forest (RF):** $100$ estimators, `min_samples_leaf=10`.
* **Multi-Layer Perceptron (MLP):** Hidden layers `(20, 10)`, ReLU activation, `max_iter=500`.

---

## Repository Structure

```text
eeg_pain_research/
├── eeg_pain_research.py   # Main processing, feature extraction, and validation pipeline
├── requirements.txt       # Python package dependencies
├── .gitignore             # Excludes raw .mat EEG recordings and generated .csv outputs
└── README.md              # Project documentation
