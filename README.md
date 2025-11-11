# MachineLearning — Regression & Classification on Tabular Data

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-FF9F1C)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A compact, reproducible project that demonstrates **tabular ML** for two real-world problems:

- **Regression (Used-Car Price Prediction)** — predict `selling_price` from specs & usage.
- **Classification (Telecom Churn)** — predict whether a customer will churn.

Both notebooks use **sklearn pipelines** (impute → scale/encode → model), clear **metrics**, and tidy **plots**.  
Classification additionally shows **class-imbalance handling** with **SMOTE** and **model tuning**.

---

## 📦 Repository Structure
MachineLearning/
├─ regression.ipynb # Used-car price regression (RF vs MLP, tuning, learning curves)
├─ classification.ipynb # Telco churn classification (RF vs MLP, SMOTE, tuning)
├─ cars.csv # Regression dataset
├─ churn.csv # Classification dataset
└─ README.md


---

## ⚙️ Environment

```bash
# (optional) create & activate a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# launch notebooks
jupyter notebook


pip install -U pip
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn jupyter
```
## 🚀 Quickstart

Open regression.ipynb or classification.ipynb.

Run all cells from top to bottom (datasets are included).

Inspect outputs: metrics, confusion matrices/ROC (classification), learning curves, and feature importance.

## 📘 What’s Inside the Notebooks
1) regression.ipynb — Used-Car Price

Data cleaning: parse units (e.g., "23.4 kmpl" → 23.4), create mileage_num, engine_cc, max_power_bhp.

Outlier control: drop top 1% prices to stabilize fit.

Pipelines:

Numeric → median impute + standardize

Categorical → most-frequent impute + one-hot

Models: RandomForestRegressor vs MLPRegressor

Evaluation: MAE, MSE, RMSE, R²

Extras: learning curves, MLP loss/val curves, feature importances, correlation heatmap.

Headline results (example run)

Tuned Random Forest: RMSE ≈ 132k, R² ≈ 0.962

MLP: higher error on this tabular mix → RF is the preferred baseline.

2) classification.ipynb — Telco Churn

Preprocessing: stratified split (Train/Val/Test), robust OHE, scaling, imputation.

Imbalance handling:

RF with class_weight="balanced"

MLP trained on SMOTE-oversampled Train split

Tuning:

RF: randomized search over trees/depth/splits/leaf/max_features (select by Val ROC-AUC)

MLP: grid of hidden sizes, alpha, learning_rate_init, batch_size (select by Val ROC-AUC)

Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC curve

Explainability: Random-Forest feature importances (top-k bar chart)

Headline results (example run)

RF (tuned): ROC-AUC ≈ 0.84, balanced overall with good recall

MLP (tuned + SMOTE): ROC-AUC ≈ 0.84 with higher minority-class recall/F1

Choose MLP when missing churners is costly; choose RF for slightly more balanced precision and simpler serving.

## 🧩 Key Ideas & Why They Matter

Pipelines prevent leakage; preprocessing is identical at train/test time.

SMOTE is applied only on Train to avoid inflating validation/test.

Validation-based tuning yields robust configs (not just train performance).

Feature importance (RF) provides quick explainability for stakeholders.

## 📊 Example Figures (auto-generated)

Regression: Actual vs Predicted, Residuals, Learning Curves, MLP loss/val curves, Feature Importances.

Classification: Confusion Matrices (Val/Test), ROC Curves (Val/Test), Top-k Feature Importances.

All images are generated when you run the notebooks; no external files required.

## ✅ Reproducibility Tips

Set random_state=42 (already used).

Run on Python 3.10+ and scikit-learn 1.x.

On Windows, a harmless warning about physical cores may appear; we cap n_jobs=1 where needed.

## 📝 License

Released under the MIT License.

## 🙌 Acknowledgements

Datasets: educational examples for cars pricing and Telco churn.
Libraries: pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, Jupyter.
