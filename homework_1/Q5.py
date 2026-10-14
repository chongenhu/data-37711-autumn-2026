from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import sklearn

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import jaxlib
import optax
import jax.scipy.stats as jstats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONFIG = dict(seed=42, dtype=jnp.float64, learning_rate=0.01, batch_size=32,
              epochs=1000, momentum=0, weight_decay=0, probability_clip_eps=1e-12)

def logistic_probability(z):
    return jax.nn.sigmoid(z)

def binary_cross_entropy(p, y):
    assert p.ndim == y.ndim == 1 and p.shape == y.shape
    p_safe = jnp.clip(p, CONFIG['probability_clip_eps'], 1 - CONFIG['probability_clip_eps'])
    return -jnp.mean(y * jnp.log(p_safe) + (1 - y) * jnp.log(1 - p_safe))

def logistic_loss(z, y):
    return binary_cross_entropy(logistic_probability(z), y)

def probit_probability(z):
    return jstats.norm.cdf(z, loc=0.0, scale=1.0)

def probit_loss(z, y):
    return binary_cross_entropy(probit_probability(z), y)

def mse_loss(prediction, y):
    assert prediction.ndim == y.ndim == 1 and prediction.shape == y.shape
    return jnp.mean((y - prediction) ** 2)

def student_t_loss(prediction, y):
    assert prediction.ndim == y.ndim == 1 and prediction.shape == y.shape
    return -jnp.mean(jstats.t.logpdf(y - prediction, df=5.0, loc=0.0, scale=1.0))

def predict(params, X):
    return X @ params['w'] + params['b']

def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(tree)]))

def train(name, X, y, objective):
    X, y = (jnp.asarray(X, dtype=CONFIG['dtype']), jnp.asarray(y, dtype=CONFIG['dtype']))
    assert X.ndim == 2 and y.ndim == 1 and (X.shape[0] == y.shape[0])
    if not bool(all_finite((X, y))):
        raise FloatingPointError(name + ' has non-finite inputs or targets')
    params = {'w': jnp.zeros(X.shape[1], dtype=CONFIG['dtype']), 'b': jnp.array(0.0, dtype=CONFIG['dtype'])}
    optimizer = optax.sgd(learning_rate=CONFIG['learning_rate'], momentum=CONFIG['momentum'])
    state = optimizer.init(params)
    key = jax.random.PRNGKey(CONFIG['seed'])
    losses = []

    def batch_loss(params, X_batch, y_batch):
        return objective(predict(params, X_batch), y_batch)

    @jax.jit
    def step(params, state, X_batch, y_batch):
        loss, gradients = jax.value_and_grad(batch_loss)(params, X_batch, y_batch)
        updates, next_state = optimizer.update(gradients, state, params)
        next_params = optax.apply_updates(params, updates)
        valid = all_finite((predict(params, X_batch), loss, gradients, next_params))
        return (next_params, next_state, valid)

    @jax.jit
    def full_training_loss(params):
        prediction = predict(params, X)
        loss = objective(prediction, y)
        return (loss, all_finite((prediction, loss)))
    for epoch in range(CONFIG['epochs']):
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, len(y))
        for start in range(0, len(y), CONFIG['batch_size']):
            indices = order[start:start + CONFIG['batch_size']]
            params, state, valid = step(params, state, X[indices], y[indices])
            if not bool(valid):
                raise FloatingPointError(f'{name}, epoch {epoch + 1}: non-finite batch prediction, loss, gradient, or updated parameter')
        loss, valid = full_training_loss(params)
        if not bool(valid):
            raise FloatingPointError(name + ' has non-finite full-training predictions or loss')
        losses.append(float(loss))
    return (params, losses)

FEATURE_COLUMNS = ["anchor_ratio", "trans_range", "node_density", "iterations"]
TARGET_COLUMN = "ale"
CSV_PATH = Path(__file__).resolve().parent / "data" / "mcs_ds_edited_iter_shuffled.csv"

np.random.seed(CONFIG["seed"])
print("Versions:", {"NumPy": np.__version__, "pandas": pd.__version__,
                    "sklearn": sklearn.__version__, "JAX": jax.__version__,
                    "jaxlib": jaxlib.__version__, "Optax": optax.__version__})
print("SGD configuration:", CONFIG, "; backend:", jax.default_backend(), "; x64:", jax.config.x64_enabled)
print("test_size:", 0.2, "random_state:", 42, "classification stratify:", True)

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
print("\nALE source: Singh et al. (2020), UCI 844, https://doi.org/10.24432/C5ZP6R, CC BY 4.0")
print("ALE shape:", ale.shape)
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

c_scaler, r_scaler = StandardScaler().fit(Xc[ct]), StandardScaler().fit(Xr[rt])
Xct, Xcv = c_scaler.transform(Xc[ct]), c_scaler.transform(Xc[cv])
Xrt, Xrv = r_scaler.transform(Xr[rt]), r_scaler.transform(Xr[rv])
yct, ycv, yrt, yrv = yc[ct], yc[cv], yr[rt], yr[rv]
C0, R0 = LogisticRegression(max_iter=5000), LinearRegression()
print("\nC0 actual parameters:", C0.get_params())
C0.fit(Xct, yct)
R0.fit(Xrt, yrt)

C1, C1_losses = train("C1", Xct, yct, logistic_loss)
C2, C2_losses = train("C2", Xct, yct, probit_loss)
R1, R1_losses = train("R1", Xrt, yrt, mse_loss)
R2, R2_losses = train("R2", Xrt, yrt, student_t_loss)

Xcv_j, Xrv_j = jnp.asarray(Xcv, dtype=CONFIG["dtype"]), jnp.asarray(Xrv, dtype=CONFIG["dtype"])
p1 = logistic_probability(predict(C1, Xcv_j))
p2 = probit_probability(predict(C2, Xcv_j))
r1, r2 = predict(R1, Xrv_j), predict(R2, Xrv_j)
if not bool(all_finite((p1, p2, r1, r2))):
    raise FloatingPointError("Non-finite test predictions/probabilities")
c_predictions = [C0.predict(Xcv), np.asarray(p1 >= 0.5), np.asarray(p2 >= 0.5)]
r_predictions = [R0.predict(Xrv), np.asarray(r1), np.asarray(r2)]
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
print("finite_checks: passed")
print("Training loss:")
for name, losses in [("C1", C1_losses), ("C2", C2_losses), ("R1", R1_losses), ("R2", R2_losses)]:
    previous, recent = np.mean(losses[-200:-100]), np.mean(losses[-100:])
    print(f"{name}: final loss={losses[-1]:.10f}; mean epochs 801-900={previous:.10f}; "
          f"mean epochs 901-1000={recent:.10f}; change={recent - previous:+.10f}")
