MachineLearning — Regression & Classification on Tabular Data

A compact, reproducible project that demonstrates tabular ML for two real-world problems:

Regression (Used-Car Price Prediction) — predict selling_price from specs & usage.

Classification (Telecom Churn) — predict whether a customer will churn.

Both notebooks use sklearn pipelines (impute → scale/encode → model), clear metrics, and tidy plots.
Classification additionally shows class-imbalance handling with SMOTE and model tuning.

📦 Repository Structure
MachineLearning/
├─ regression.ipynb         # Used-car price regression (RF vs MLP, tuning, learning curves)
├─ classification.ipynb     # Telco churn classification (RF vs MLP, SMOTE, tuning)
├─ cars.csv                 # Regression dataset
├─ churn.csv                # Classification dataset
└─ README.md

⚙️ Environment
# create env (optional)
python -m venv .venv
# activate: Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn jupyter


Open either notebook:

jupyter notebook

🚀 Quickstart

Open regression.ipynb or classification.ipynb.

Run all cells from top to bottom (datasets are included).

Inspect outputs: metrics, confusion matrices/ROC (classification), learning curves, and feature importance.

📘 What’s Inside the Notebooks
1) regression.ipynb — Used-Car Price

Data cleaning: parse units (e.g., "23.4 kmpl" → 23.4), create mileage_num, engine_cc, max_power_bhp.

Outlier control: drop top 1% prices to stabilize fit.

Pipelines:

Numeric → median impute + standardize

Categorical → most-frequent impute + one-hot

Models: RandomForestRegressor vs MLPRegressor

Evaluation: MAE, MSE, RMSE, R²

Extras: learning curves, MLP loss/val curves, feature importances, correlation heatmap.

Headline results (example run):

Tuned Random Forest: RMSE ≈ 132k, R² ≈ 0.962

MLP: higher error on this tabular mix → RF is preferred baseline.

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

Headline results (example run):

RF (tuned): ROC-AUC ≈ 0.84, balanced overall with good recall

MLP (tuned + SMOTE): ROC-AUC ≈ 0.84 with higher minority-class recall/F1

Pick MLP when missing churners is very costly; pick RF for slightly more balanced precision/serving simplicity.

🧩 Key Ideas & Why They Matter

Pipelines prevent leakage and keep preprocessing consistent at train/test time.

SMOTE is applied only on Train to avoid inflating validation/test.

Validation-based tuning selects robust configs (not just train performance).

Feature importance (RF) gives quick model explainability for stakeholders.

📊 Example Figures (produced by notebooks)

Regression: Actual vs Predicted, Residuals, Learning Curves, MLP loss/val curves, Feature Importances

Classification: Confusion Matrices (Val/Test), ROC Curves (Val/Test), Top-k Feature Importances

All images are generated when you run the notebooks; no external files required.

✅ Reproducibility Tips

Set random_state=42 (already used).

Run on Python 3.10+ and sklearn 1.x.

If you see UserWarning: Could not find the number of physical cores on Windows, it’s safe to ignore (we restrict n_jobs=1 in places to keep output tidy).

📝 License

This project is released under the MIT License. Feel free to use, adapt, and cite.

🙌 Acknowledgements

Datasets: public educational examples for cars pricing and Telco churn.
Libraries: pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn, Jupyter.
