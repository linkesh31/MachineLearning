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
