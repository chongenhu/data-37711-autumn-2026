from pathlib import Path

import numpy as np
import pandas as pd

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import optax
import jax.scipy.stats as jstats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONFIG = dict(seed=42, dtype=jnp.float64, learning_rate=0.01, batch_size=32,
              epochs=1000, momentum=0, probability_clip_eps=1e-12)

def logistic_probability(z):
    return jax.nn.sigmoid(z)

def binary_cross_entropy(p, y):
    p_safe = jnp.clip(p, CONFIG['probability_clip_eps'], 1 - CONFIG['probability_clip_eps'])
    return -jnp.mean(y * jnp.log(p_safe) + (1 - y) * jnp.log(1 - p_safe))

def logistic_loss(z, y):
    return binary_cross_entropy(logistic_probability(z), y)

def probit_probability(z):
    return jstats.norm.cdf(z, loc=0.0, scale=1.0)

def probit_loss(z, y):
    return binary_cross_entropy(probit_probability(z), y)

def mse_loss(prediction, y):
    return jnp.mean((y - prediction) ** 2)

def student_t_loss(prediction, y):
    return -jnp.mean(jstats.t.logpdf(y - prediction, df=5.0, loc=0.0, scale=1.0))

def predict(params, X):
    return X @ params['w'] + params['b']

def train(X, y, objective):
    X, y = (jnp.asarray(X, dtype=CONFIG['dtype']), jnp.asarray(y, dtype=CONFIG['dtype']))
    params = {'w': jnp.zeros(X.shape[1], dtype=CONFIG['dtype']), 'b': jnp.array(0.0, dtype=CONFIG['dtype'])}
    optimizer = optax.sgd(learning_rate=CONFIG['learning_rate'], momentum=CONFIG['momentum'])
    state = optimizer.init(params)
    key = jax.random.PRNGKey(CONFIG['seed'])

    def batch_loss(params, X_batch, y_batch):
        return objective(predict(params, X_batch), y_batch)

    @jax.jit
    def step(params, state, X_batch, y_batch):
        _, gradients = jax.value_and_grad(batch_loss)(params, X_batch, y_batch)
        updates, next_state = optimizer.update(gradients, state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_state

    for _ in range(CONFIG['epochs']):
        key, shuffle_key = jax.random.split(key)
        order = jax.random.permutation(shuffle_key, len(y))
        for start in range(0, len(y), CONFIG['batch_size']):
            indices = order[start:start + CONFIG['batch_size']]
            params, state = step(params, state, X[indices], y[indices])
    return params

FEATURE_COLUMNS = ["anchor_ratio", "trans_range", "node_density", "iterations"]
TARGET_COLUMN = "ale"
CSV_PATH = Path(__file__).resolve().parent / "data" / "mcs_ds_edited_iter_shuffled.csv"

np.random.seed(CONFIG["seed"])

cancer = load_breast_cancer()
Xc = np.asarray(cancer.data, dtype=np.float64)
yc = np.asarray(cancer.target, dtype=np.int64).reshape(-1)
ct, cv = train_test_split(np.arange(len(yc)), test_size=0.2, random_state=42, stratify=yc)

ale = pd.read_csv(CSV_PATH)
print("ALE source: Singh et al. (2020), UCI 844, https://doi.org/10.24432/C5ZP6R, CC BY 4.0")
Xr = ale.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
yr = ale.loc[:, TARGET_COLUMN].to_numpy(dtype=np.float64).reshape(-1)
rt, rv = train_test_split(np.arange(len(yr)), test_size=0.2, random_state=42)

c_scaler, r_scaler = StandardScaler().fit(Xc[ct]), StandardScaler().fit(Xr[rt])
Xct, Xcv = c_scaler.transform(Xc[ct]), c_scaler.transform(Xc[cv])
Xrt, Xrv = r_scaler.transform(Xr[rt]), r_scaler.transform(Xr[rv])
yct, ycv, yrt, yrv = yc[ct], yc[cv], yr[rt], yr[rv]
C0, R0 = LogisticRegression(max_iter=5000), LinearRegression()
C0.fit(Xct, yct)
R0.fit(Xrt, yrt)

C1 = train(Xct, yct, logistic_loss)
C2 = train(Xct, yct, probit_loss)
R1 = train(Xrt, yrt, mse_loss)
R2 = train(Xrt, yrt, student_t_loss)

Xcv_j, Xrv_j = jnp.asarray(Xcv, dtype=CONFIG["dtype"]), jnp.asarray(Xrv, dtype=CONFIG["dtype"])
p1 = logistic_probability(predict(C1, Xcv_j))
p2 = probit_probability(predict(C2, Xcv_j))
r1, r2 = predict(R1, Xrv_j), predict(R2, Xrv_j)
c_predictions = [C0.predict(Xcv), np.asarray(p1 >= 0.5), np.asarray(p2 >= 0.5)]
r_predictions = [R0.predict(Xrv), np.asarray(r1), np.asarray(r2)]
accuracies = [accuracy_score(ycv, prediction) for prediction in c_predictions]
maes = [mean_absolute_error(yrv, prediction) for prediction in r_predictions]
classification = pd.DataFrame({"Model": ["C0 sklearn LogisticRegression", "C1 Logistic + BCE + SGD", "C2 Probit + BCE + SGD"],
                               "Test Accuracy": accuracies,
                               "Correct / Test Samples": [f"{np.sum(p == ycv)} / {len(ycv)}" for p in c_predictions]
                              }).to_string(index=False, float_format=lambda x: f"{x:.8f}")
regression = pd.DataFrame({"Model": ["R0 sklearn LinearRegression", "R1 Linear + SGD", "R2 Linear + SGD"],
                           "Training Objective": ["Ordinary least squares", "MSE", "Student-t NLL (df=5, scale=1)"],
                           "Test MAE": maes}).to_string(index=False, float_format=lambda x: f"{x:.10f}")
result = "Classification\n" + classification + "\n\nRegression (MAE in original ale units, meters)\n" + regression
print("\n" + result)
result_dir = Path(__file__).resolve().parent / "results"
result_dir.mkdir(exist_ok=True)
(result_dir / "Q5.txt").write_text(result + "\n", encoding="utf-8")
