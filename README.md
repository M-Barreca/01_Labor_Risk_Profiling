# Labor Risk Profiling: Predicting Income Inequality

A machine learning pipeline that predicts whether an individual's income exceeds **$50K/year** based on U.S. census data. The project combines interpretability analysis, clustering, and deep learning to build a comprehensive income classification system.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M-Barreca/01_Labor_Risk_Profiling/blob/colab-work/labor_risk_profiling.ipynb)

---

## Dataset

- **Source:** [UCI Adult Census Data – Kaggle](https://www.kaggle.com/datasets/sagnikpatra/uci-adult-census-data-dataset)
- **Target variable:** `income` (>50K / <=50K)
- **Features:** age, education, occupation, sex, race, hours per week, capital gain/loss, and more.

---

## Project Structure

```
01_Labor_Risk_Profiling/
│
└── labor_risk_profiling.ipynb   # Main notebook (EDA → Preprocessing → Modeling)
```

---

## Notebook Overview

### 1. Introduction & Data Loading
Import of libraries and loading of the dataset from Google Drive.

### 2. Exploratory Data Analysis (EDA)

- **2.1 Missing Analysis** — Structural missingness matrix (workclass & occupation co-missing)
- **2.2 Bivariate Analysis** — Working hours vs. education, sex, and age (boxplot, violin, scatter, strip plot)
- **2.3 Income by Age & Gender** — Heatmap showing the gender income gap across age brackets
- **2.4 Income by Education & Sex** — Faceted catplot across age cohorts
- **2.5 Feature Distribution & Correlation** — Outlier analysis, histograms, pairplot
- **2.6 Spearman Correlation Matrix** — Key predictors: `education.num`, `capital.gain`
- **2.7 Contingency Table & Chi-Square Test** — Income distribution by ethnic group, Cramér's V

### 3. Data Preprocessing & Pipeline Construction

- **3.1** Feature/target separation
- **3.2** ColumnTransformer definition (numerical scaling + categorical encoding)
- **3.3** Preprocessor assembly
- **3.4** Full Scikit-learn Pipeline (preprocessing + model)
- **3.5** Random Forest — classification results + SHAP / PDP / Permutation Importance
- **3.6** XGBoost — comparison with Random Forest
- **3.7** Clustering — unsupervised risk profile segmentation
- **3.8** Deep Learning — neural network classifier

### 4. ROC Curve Final Comparison
Side-by-side ROC/AUC comparison of all models (Random Forest, XGBoost, Clustering, Deep Learning).

---

## Key Findings

- **Gender gap:** Men in peak earning years (40–60) reach ~43–45% probability of high income; women in the same bracket never exceed ~18%.
- **Top predictors:** `education.num` and `capital.gain` show the strongest correlation with income.
- **Ethnic disparity:** Asian-Pacific Islander and White groups show the highest relative frequency of high income; the association is statistically significant (Chi-Square) but moderate (Cramér's V ≈ 0.10).

---

## Requirements

```
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
missingno
shap
scipy
tensorflow / torch
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Results

| Model | AUC |
|-------|-----|
| Random Forest | TBD |
| XGBoost | TBD |
| Deep Learning | TBD |

---

## License

MIT
