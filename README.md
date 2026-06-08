# 🏭 Labor Risk Profiling
### Income Vulnerability Detection via ML — Random Forest · XGBoost · Clustering · MLP

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M-Barreca/01_Labor_Risk_Profiling/blob/main/notebooks/labor_risk_profiling.ipynb)

---

## 📌 Objective

Identify individuals at **high risk of income vulnerability** using a multi-model ML pipeline on the UCI Adult Census dataset. The project covers the full data science lifecycle: exploratory analysis → preprocessing → model training → interpretability (SHAP, LIME, PDP) → clustering → model comparison.

---

## 📊 Dataset

**UCI Adult Census Income** (Kaggle snapshot, ~48K records, 14 features)  
Source: [kaggle.com/datasets/sagnikpatra/uci-adult-census-data-dataset](https://www.kaggle.com/datasets/sagnikpatra/uci-adult-census-data-dataset)

Key features: `age`, `education.num`, `occupation`, `hours.per.week`, `capital.gain/loss`, `marital.status`, `sex`, `native.country`  
Target: `income > $50K/year` (binary, 3:1 class imbalance)

---

## 🔬 Methodology

### EDA Highlights
- **Gender gap:** Men reach 43–45% high-income probability in peak years (40–60); women never exceed 18% in the same brackets
- **Strongest categorical predictors:** `relationship` and `marital.status` (Cramér's V ≈ 0.45)
- **Strongest numerical predictors:** `education.num` and `capital.gain` (Spearman correlation)
- **Outlier handling:** Heavy skew in `capital.gain`/`fnlwgt` → RobustScaler chosen over StandardScaler

### Preprocessing Pipeline
- `RobustScaler` for numerical features (outlier-resistant)
- `OneHotEncoder(drop='first')` for categorical features
- Rare `native.country` values (<100 occurrences) collapsed to `'Other'`
- All transformations inside a `Pipeline` to prevent data leakage

### Models Trained

| Model | Approach | Tuning |
|---|---|---|
| Random Forest | Ensemble of 100 independent trees, `class_weight='balanced'` | RandomizedSearchCV (F1 scoring) |
| XGBoost | Sequential gradient boosting | GridSearchCV (AUC scoring) |
| Deep Learning (MLP) | 2-layer neural net (100→50), Adam, early stopping | — |
| K-Means Clustering | Unsupervised segmentation, K=3 | Elbow + Silhouette |

### Interpretability
- **Permutation Importance** — global feature ranking
- **Partial Dependence Plots (PDP)** — marginal effect of top features
- **SHAP (TreeExplainer)** — global bar + beeswarm + waterfall for XGBoost
- **LIME** — local explanation for individual predictions (Random Forest)

---

## 📈 Results

### Classification

| Model | AUC | F1 (macro) | Precision | Recall |
|---|---|---|---|---|
| **XGBoost** | **0.929** | — | — | — |
| Deep Learning (MLP) | 0.915 | — | — | — |
| Random Forest | 0.910 | — | — | — |

> Fill in F1/Precision/Recall from the summary table cell in the notebook after running.

### Clustering (K-Means, K=3)

| Cluster | High-Income Rate | Interpretation |
|---|---|---|
| 0 | ~21% | Low-income majority |
| 1 | ~100% | Pure high-income segment |
| 2 | ~95% | Near-homogeneous high-income group |

### SHAP — Top Predictors (XGBoost)

```
cat__marital.status_Married-civ-spouse   ████████████████  (bidirectional)
num__capital.gain                        ██████████████    (extreme values → >50K)
num__capital.loss                        ████████████      (high variance)
num__education.num                       ████████          (consistent positive)
num__age                                 ██████
```

---

## 🗂️ Repository Structure

```
01_Labor_Risk_Profiling/
├── README.md
├── requirements.txt
├── Dockerfile
├── data/
│   └── adult.csv              ← place dataset here (not tracked by git)
├── notebooks/
│   └── labor_risk_profiling.ipynb   ← main notebook
├── src/
│   └── inference.py           ← standalone inference script
└── images/
    ├── roc_comparison.png
    ├── shap_beeswarm.png
    ├── confusion_matrix.png
    └── elbow_silhouette.png
```

---

## 🚀 Quickstart

### Option A — Local

```bash
git clone https://github.com/M-Barreca/01_Labor_Risk_Profiling.git
cd 01_Labor_Risk_Profiling
pip install -r requirements.txt
# place adult.csv in data/
jupyter notebook notebooks/labor_risk_profiling.ipynb
```

### Option B — Google Colab

Click the badge at the top of the notebook. Mount your Drive and set `DATA_PATH` accordingly.

### Option C — Docker

```bash
docker build -t labor-risk .
docker run --rm labor-risk
```

---

## 🧰 Tech Stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap` · `lime` · `matplotlib` · `seaborn` · `missingno` · `scipy` · `Docker`

---

## ⚠️ Limitations

The dataset reflects **1994 US census** demographics. Income distributions, labour markets, and gender gaps have shifted significantly. This model should not be used for real-world employment decisions without retraining on current data.

---

## 📄 License

MIT
