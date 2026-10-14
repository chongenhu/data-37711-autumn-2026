from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

CSV_NAME = "mcs_ds_edited_iter_shuffled.csv"
feature_columns = ["anchor_ratio", "trans_range", "node_density", "iterations"]
target_column = "ale"
alpha = 0.05

csv_path = Path(__file__).resolve().parent / "data" / CSV_NAME
if not csv_path.is_file():
    raise FileNotFoundError(f"Real UCI ALE data missing: {csv_path}. No synthetic data is used.")
df = pd.read_csv(csv_path)

print("ALE source: Singh et al. (2020), UCI 844, https://doi.org/10.24432/C5ZP6R, CC BY 4.0")
print("ALE shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:\n" + df.isna().sum().to_string())
if df.shape != (107, 6) or df.columns.tolist() != feature_columns + [target_column, "sd_ale"]:
    raise ValueError("The downloaded file differs from the published schema; no data were modified.")
if not all(pd.api.types.is_numeric_dtype(df[c]) for c in feature_columns + [target_column]):
    raise TypeError("The predictors and target must be numeric.")
X = df[feature_columns]
y = df[target_column].to_numpy(dtype=float)
if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y).all():
    raise ValueError("Non-finite or missing model inputs; no imputation or row removal is performed.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training/test samples: {len(y_train)}/{len(y_test)}")
models = {
    "LinearRegression": LinearRegression(),
    "SVR (RBF)": make_pipeline(
        StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
    ),
    "DecisionTreeRegressor": DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=5, random_state=42
    ),
}
filenames = ["linear_regression", "svr_rbf", "decision_tree"]
figure_dir = Path(__file__).resolve().parent / "figures"
figure_dir.mkdir(exist_ok=True)
rows = []

for (name, model), filename in zip(models.items(), filenames):
    model.fit(X_train, y_train)
    prediction = np.asarray(model.predict(X_test), dtype=float)
    if y_test.ndim != 1 or prediction.ndim != 1 or prediction.shape != y_test.shape:
        raise ValueError(f"{name}: predictions and targets must be aligned one-dimensional arrays.")
    residuals = y_test - prediction
    if not np.isfinite(residuals).all() or residuals.size < 8 or np.var(residuals) == 0:
        raise ValueError(f"{name}: normaltest needs finite, nonconstant residuals and at least 8 values.")
    try:
        test = stats.normaltest(residuals)
        statistic, p_value = float(test.statistic), float(test.pvalue)
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            raise ValueError("normaltest returned a non-finite result")
        decision = ("Reject normality at the 5% level." if p_value < alpha
                    else "Fail to reject normality at the 5% level.")
    except Exception as error:
        statistic, p_value = np.nan, np.nan
        decision = f"Normality test unavailable: {error}"
        print(name, decision)
    rows.append({"model": name, "n_test": len(residuals),
                 "MAE": mean_absolute_error(y_test, prediction),
                 "normaltest_statistic": statistic, "p_value": p_value,
                 "decision_at_0.05": decision})

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    stats.probplot(residuals, dist="norm", fit=True, plot=ax)
    ax.set_title(f"{name} — Test Residuals")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Ordered signed residuals (ale, meters)")
    ax.get_lines()[0].set_label("Ordered test residuals")
    ax.get_lines()[1].set_label("Fitted reference line")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / f"Q4_{filename}.png", dpi=180)
    plt.close(fig)

summary = pd.DataFrame(rows)
result = "Held-out test residuals; alpha=0.05\n\n" + summary.to_string(index=False, formatters={
    "MAE": "{:.9f}".format, "normaltest_statistic": "{:.9f}".format,
    "p_value": "{:.9e}".format,
})
print("\n" + result)
result_dir = Path(__file__).resolve().parent / "results"
result_dir.mkdir(exist_ok=True)
(result_dir / "Q4.txt").write_text(result + "\n", encoding="utf-8")
