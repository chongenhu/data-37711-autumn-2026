"""HW1 Q5: sklearn baselines and four simple PyTorch models trained with SGD.

Requires NumPy, pandas, scikit-learn, and PyTorch; run with `python Q5.py`.
ALE data are read from the included real UCI CSV; nothing is exported.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Implementation choices, fixed before test evaluation; not optimality claims.
CONFIG = dict(seed=42, dtype=torch.float64, learning_rate=0.01, batch_size=32,
              epochs=1000, momentum=0, weight_decay=0, probability_clip_eps=1e-12)
FEATURE_COLUMNS = ["anchor_ratio", "trans_range", "node_density", "iterations"]
TARGET_COLUMN = "ale"
CSV_PATH = Path(__file__).resolve().parent / "data" / "mcs_ds_edited_iter_shuffled.csv"
# UCI dataset 844: Singh et al. (2020), DOI 10.24432/C5ZP6R, CC BY 4.0.
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.set_num_threads(1)  # Small models run on CPU, using one thread.
print("Versions:", {"NumPy": np.__version__, "pandas": pd.__version__,
                    "sklearn": sklearn.__version__, "PyTorch": torch.__version__})
print("SGD configuration:", CONFIG, "; device=cpu; CPU threads=1")
print("Both splits: test_size=0.2, random_state=42; stratify only classification.")

cancer = load_breast_cancer()
Xc = np.asarray(cancer.data, dtype=np.float64)
yc = np.asarray(cancer.target, dtype=np.int64).reshape(-1)
print("\nOriginal classification target_names:", cancer.target_names)
print("Classification X/y shapes:", Xc.shape, yc.shape, "; dtypes:", Xc.dtype, yc.dtype)
print("Classification missing values:", int(np.isnan(Xc).sum()), int(pd.isna(yc).sum()))
assert Xc.shape == (569, 30) and np.array_equal(np.unique(yc), [0, 1])
assert np.isfinite(Xc).all()
ct, cv = train_test_split(np.arange(len(yc)), test_size=0.2, random_state=42, stratify=yc)

if not CSV_PATH.is_file():
    raise FileNotFoundError("Included real ALE CSV is missing: " + str(CSV_PATH)
                            + "; no synthetic data are substituted.")
csv_bytes = CSV_PATH.read_bytes()
ale = pd.read_csv(CSV_PATH)
print("\nALE actual shape:", ale.shape, "; UCI reports 107 samples")
print("ALE columns:", list(ale.columns))
print("ALE dtypes:\n", ale.dtypes.to_string(), sep="")
print("ALE missing values:\n", ale.isna().sum().to_string(), sep="")
if ale.shape != (107, 6) or list(ale.columns) != FEATURE_COLUMNS + ["ale", "sd_ale"]:
    raise ValueError("Unexpected ALE shape or schema; inspect the printed values. No data modified.")
if not all(pd.api.types.is_numeric_dtype(ale[c]) for c in ale.columns):
    raise TypeError("ALE columns must be numeric; no silent conversion is performed.")
if ale.isna().any().any() or not np.isfinite(ale.to_numpy(dtype=np.float64)).all():
    raise ValueError("ALE has missing or non-finite values; no rows are deleted or imputed.")
if hashlib.sha256(csv_bytes).hexdigest() != "5c8d3e608bcfef50d21c2f16e5313d23f749d8b3bced76b8238acca8c05c2be1":
    raise ValueError("Official ALE CSV differs from the previously validated homework data.")
Xr = ale.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
yr = ale.loc[:, TARGET_COLUMN].to_numpy(dtype=np.float64).reshape(-1)
print("Regression features:", FEATURE_COLUMNS, "; target:", TARGET_COLUMN, "; sd_ale ignored")
print("Regression X/y shapes:", Xr.shape, yr.shape, "; original ale units: meters")
rt, rv = train_test_split(np.arange(len(yr)), test_size=0.2, random_state=42)
print("Classification train/test:", len(ct), len(cv), "; regression train/test:", len(rt), len(rv))

# Each dataset has one scaler fitted only on its training features; y is not scaled.
c_scaler, r_scaler = StandardScaler().fit(Xc[ct]), StandardScaler().fit(Xr[rt])
Xct, Xcv = c_scaler.transform(Xc[ct]), c_scaler.transform(Xc[cv])
Xrt, Xrv = r_scaler.transform(Xr[rt]), r_scaler.transform(Xr[rv])
yct, ycv, yrt, yrv = yc[ct], yc[cv], yr[rt], yr[rv]
C0, R0 = LogisticRegression(max_iter=5000), LinearRegression()
print("\nC0 actual parameters:", C0.get_params())
print("C0 retains default regularization; custom BCE has no regularization term.")
C0.fit(Xct, yct)
R0.fit(Xrt, yrt)

normal = torch.distributions.Normal(torch.tensor(0.0, dtype=CONFIG["dtype"]),
                                    torch.tensor(1.0, dtype=CONFIG["dtype"]))
student_t = torch.distributions.StudentT(df=torch.tensor(5.0, dtype=CONFIG["dtype"]),
                                        loc=torch.tensor(0.0, dtype=CONFIG["dtype"]),
                                        scale=torch.tensor(1.0, dtype=CONFIG["dtype"]))

def bce(p, y):
    assert p.ndim == y.ndim == 1 and p.shape == y.shape
    p_safe = p.clamp(CONFIG["probability_clip_eps"], 1 - CONFIG["probability_clip_eps"])
    return -(y * p_safe.log() + (1 - y) * (1 - p_safe).log()).mean()

def objective(name, prediction, y):
    assert prediction.ndim == y.ndim == 1 and prediction.shape == y.shape
    if name == "C1":
        return bce(torch.sigmoid(prediction), y)
    if name == "C2":
        return bce(normal.cdf(prediction), y)  # Standard normal CDF, not PDF or sigmoid.
    if name == "R1":
        return ((y - prediction) ** 2).mean()
    return -student_t.log_prob(y - prediction).mean()  # Fixed df=5 and scale=1.

def finite(value, context):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError("Non-finite " + context)

def train(name, X, y):
    X = torch.as_tensor(X, dtype=CONFIG["dtype"], device="cpu")
    y = torch.as_tensor(y, dtype=CONFIG["dtype"], device="cpu")
    finite(X, name + " inputs")
    finite(y, name + " targets")
    model = nn.Linear(X.shape[1], 1, bias=True, dtype=CONFIG["dtype"], device="cpu")
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG["learning_rate"],
                                momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(CONFIG["seed"])
    losses = []  # Current full-training mean loss is recorded for every epoch.
    for epoch in range(CONFIG["epochs"]):
        model.train()
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), CONFIG["batch_size"]):
            indices = order[start:start + CONFIG["batch_size"]]
            optimizer.zero_grad()
            prediction = model(X[indices]).squeeze(-1)
            finite(prediction, name + " batch predictions")
            loss = objective(name, prediction, y[indices])
            finite(loss, name + " batch loss")
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise RuntimeError(name + " has a missing gradient")
                finite(parameter.grad, name + " gradients")
            optimizer.step()
            for parameter in model.parameters():
                finite(parameter, name + " updated parameters")
        model.eval()
        with torch.no_grad():
            prediction = model(X).squeeze(-1)
            finite(prediction, name + " full-training predictions")
            loss = objective(name, prediction, y)
            finite(loss, name + " full-training loss")
            losses.append(loss.item())
    return model, losses

# Independent zero-initialized models, sharing the same per-dataset batch-order sequence.
C1, C1_losses = train("C1", Xct, yct)
C2, C2_losses = train("C2", Xct, yct)
R1, R1_losses = train("R1", Xrt, yrt)
R2, R2_losses = train("R2", Xrt, yrt)

# Evaluate test data only after both baselines and all four SGD models have fitted.
for model in [C1, C2, R1, R2]:
    model.eval()
with torch.no_grad():
    Xcv_t = torch.as_tensor(Xcv, dtype=CONFIG["dtype"], device="cpu")
    Xrv_t = torch.as_tensor(Xrv, dtype=CONFIG["dtype"], device="cpu")
    p1, p2 = torch.sigmoid(C1(Xcv_t).squeeze(-1)), normal.cdf(C2(Xcv_t).squeeze(-1))
    r1, r2 = R1(Xrv_t).squeeze(-1), R2(Xrv_t).squeeze(-1)
    for prediction in [p1, p2, r1, r2]:
        finite(prediction, "test predictions/probabilities")
    c_predictions = [C0.predict(Xcv), (p1 >= 0.5).numpy(), (p2 >= 0.5).numpy()]
    r_predictions = [R0.predict(Xrv), r1.numpy(), r2.numpy()]
for prediction in c_predictions:
    assert prediction.ndim == 1 and prediction.shape == ycv.shape
for prediction in r_predictions:
    assert prediction.ndim == 1 and prediction.shape == yrv.shape and np.isfinite(prediction).all()
accuracies = [accuracy_score(ycv, prediction) for prediction in c_predictions]
maes = [mean_absolute_error(yrv, prediction) for prediction in r_predictions]
print("\nClassification:")
print(pd.DataFrame({"Model": ["C0 sklearn LogisticRegression", "C1 Logistic + BCE + SGD", "C2 Probit + BCE + SGD"],
                    "Test Accuracy": accuracies,
                    "Correct / Test Samples": [f"{np.sum(p == ycv)} / {len(ycv)}" for p in c_predictions]
                   }).to_string(index=False, float_format=lambda x: f"{x:.8f}"))
print("\nRegression (MAE in original ale units, meters):")
print(pd.DataFrame({"Model": ["R0 sklearn LinearRegression", "R1 Linear + SGD", "R2 Linear + SGD"],
                    "Training Objective": ["Ordinary least squares", "MSE", "Student-t NLL (df=5, scale=1)"],
                    "Test MAE": maes}).to_string(index=False, float_format=lambda x: f"{x:.10f}"))
print(f"\nClassification (a): Logistic-SGD accuracy is {accuracies[1]:.8f}, versus sklearn's "
      f"{accuracies[0]:.8f}. Regularization and optimization differ, so exact agreement is not expected.")
print(f"Classification (b): Probit-SGD accuracy is {accuracies[2]:.8f}, versus sklearn's "
      f"{accuracies[0]:.8f}; this split does not establish a generally superior probability link.")
print(f"Regression (a): MSE-SGD test MAE is {maes[1]:.10f}, versus sklearn's {maes[0]:.10f}. "
      "MSE is the training loss; MAE is the common test metric.")
print(f"Regression (b): Student-t-SGD test MAE is {maes[2]:.10f}, versus sklearn's {maes[0]:.10f} "
      f"and MSE-SGD's {maes[1]:.10f}. Different training-loss magnitudes do not rank model quality.")
print("All finite checks passed. Final full-training losses:",
      {"C1": C1_losses[-1], "C2": C2_losses[-1], "R1": R1_losses[-1], "R2": R2_losses[-1]})
print("Classifier losses still declined slowly near epoch 1000; regression losses approached a plateau. "
      "Exact convergence is not claimed. No test metric selected training settings. Comparisons use "
      "one fixed split (114 classification and 22 regression test samples) and may not generalize.")
