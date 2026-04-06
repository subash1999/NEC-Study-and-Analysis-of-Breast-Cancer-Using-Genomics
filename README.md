<p align="center">
  <h1 align="center">Breast Cancer Prediction Using Genomics</h1>
  <p align="center">
    <strong>ML-based breast cancer classification using microarray gene expression data</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.7-3776AB?style=flat-square&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
    <img src="https://img.shields.io/badge/Django-Web_App-092E20?style=flat-square&logo=django&logoColor=white" />
    <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square" />
  </p>
</p>

---

## Overview

Bachelor's thesis project at **Nepal Engineering College (Pokhara University)** — a machine learning pipeline that predicts breast cancer from microarray gene expression data containing **22,283 genes** across multiple datasets.

The system applies **feature selection** to reduce dimensionality, trains **7 classification models**, and identifies the best-performing combination through systematic evaluation.

---

## Pipeline

```
                    ┌─────────────────────┐
                    │   Microarray Data    │
                    │   (22,283 genes)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Feature Selection   │
                    │  ┌────────────────┐  │
                    │  │ Chi-Square (X²)│  │
                    │  │ ReliefF        │  │
                    │  └────────────────┘  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │    SVM       │ │    KNN       │ │ Naive Bayes  │
     │  Decision    │ │  Gaussian NB │ │ (Bernoulli,  │
     │   Tree       │ │              │ │  Complement, │
     │              │ │              │ │  Multinomial) │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            └────────────────┼────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Best Model Selection│
                  │  (by Test Accuracy)  │
                  └─────────────────────┘
```

---

## Results

Best performing models by feature selection method:

| Model | Feature Selection | Genes Used | Train Acc | Test Acc |
|-------|-------------------|-----------|-----------|----------|
| **Decision Tree** | **Chi-Square (X²)** | **800** | **0.75** | **0.90** |
| Gaussian NB | Chi-Square (X²) | 1,600 | 0.88 | 0.83 |
| KNN | Chi-Square (X²) | 1,000 | 0.80 | 0.83 |
| Gaussian NB | No Selection | 22,283 | 0.86 | 0.82 |
| SVM | Chi-Square (X²) | 1,000 | 0.73 | 0.80 |
| Bernoulli NB | ReliefF | 2,000 | 0.72 | 0.76 |

> **Key finding:** Chi-Square feature selection with 800 genes + Decision Tree achieved **90% test accuracy** — outperforming models using all 22,283 genes.

---

## Feature Selection Methods

| Method | Approach | Best Result |
|--------|----------|-------------|
| **Chi-Square (X²)** | Statistical test measuring independence between features and target | 90% accuracy with 800 genes |
| **ReliefF** | Instance-based algorithm that estimates feature quality by sampling | 76% accuracy with 2,000 genes |
| No Selection | All 22,283 genes | 82% accuracy |

---

## Classification Models

| Model | Description |
|-------|-------------|
| Support Vector Machine (SVM) | Finds optimal hyperplane for binary classification |
| K-Nearest Neighbours (KNN) | Instance-based learning using distance metrics |
| Decision Tree | Rule-based splits for interpretable classification |
| Gaussian Naive Bayes | Probabilistic classifier assuming normal distribution |
| Bernoulli Naive Bayes | Binary feature-based Naive Bayes variant |
| Complement Naive Bayes | Handles imbalanced datasets better than standard NB |
| Multinomial Naive Bayes | Suitable for discrete feature counts |

---

## Project Structure

```
├── Classification_Models/     # 7 ML classifier implementations (OOP with inheritance)
│   ├── support_vector_machine.py
│   ├── k_nearest_neighbour.py
│   ├── decision_tree.py
│   ├── naives_bayes.py        # Gaussian, Bernoulli, Complement, Multinomial
│   ├── k_means.py
│   └── linear_regression.py
├── Top_Ranking_Models/        # Feature selection methods
│   ├── chi_square.py
│   └── relieff.py
├── Data/                      # Microarray datasets
├── best_trained_models/       # Serialized best models (.pkl) + performance CSV
├── performance_report/        # Detailed performance analysis
├── train_and_predict_model/   # Training and prediction pipeline
├── django/                    # Web interface for predictions
├── __train__.py               # Main training script
├── __predict__.py             # Prediction script
└── requirements.txt           # Python dependencies
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/subash1999/NEC-Study-and-Analysis-of-Breast-Cancer-Using-Genomics.git
cd NEC-Study-and-Analysis-of-Breast-Cancer-Using-Genomics

# Install dependencies
pip install -r requirements.txt

# Train models
python __train__.py

# Run prediction
python __predict__.py
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.7 |
| ML Framework | scikit-learn |
| Feature Selection | Chi-Square, ReliefF |
| Data Processing | pandas, NumPy |
| Web Interface | Django |
| Model Persistence | pickle (.pkl) |

---

## Academic Context

- **Institution:** Nepal Engineering College, Pokhara University
- **Degree:** Bachelor of Computer Engineering (GPA: 3.7/4.0)
- **Type:** Bachelor's Thesis Project
- **Domain:** Bioinformatics / Machine Learning

---

<p align="center">
  <sub>Built as part of a bachelor's thesis on applying machine learning to genomic breast cancer prediction.</sub>
</p>
