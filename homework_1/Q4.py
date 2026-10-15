from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
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
df = pd.read_csv(csv_path)

print("ALE source: Singh et al. (2020), UCI 844, https://doi.org/10.24432/C5ZP6R, CC BY 4.0")
X = df[feature_columns]
y = df[target_column].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    residuals = y_test - prediction
    test = stats.normaltest(residuals)
    statistic, p_value = float(test.statistic), float(test.pvalue)
    decision = ("Reject normality at the 5% level." if p_value < alpha
                else "Fail to reject normality at the 5% level.")
    rows.append({"model": name, "n_test": len(residuals),
                 "normaltest_statistic": statistic, "p_value": p_value,
                 "decision_at_0.05": decision})

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    stats.probplot(residuals, dist="norm", fit=True, plot=ax)
    ax.set_title(f"{name} — Test Residuals")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Ordered signed residuals (ale, meters)")
    fig.tight_layout()
    fig.savefig(figure_dir / f"Q4_{filename}.png", dpi=180)
    plt.close(fig)

summary = pd.DataFrame(rows)
result = "Held-out test residuals; alpha=0.05\n\n" + summary.to_string(index=False, formatters={
    "normaltest_statistic": "{:.9f}".format,
    "p_value": "{:.9e}".format,
})
print("\n" + result)
result_dir = Path(__file__).resolve().parent / "results"
result_dir.mkdir(exist_ok=True)
(result_dir / "Q4.txt").write_text(result + "\n", encoding="utf-8")
