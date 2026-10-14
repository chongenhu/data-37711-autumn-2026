# DATA37711Autumn2026 Foundations of Machine Learning and AI-PartI

This course repository currently contains **Homework 1, Questions 4 and 5 only**.
The notebooks include the real input data, executed results, and short English conclusions.
Both questions can be rerun from a fresh clone without downloading the ALE dataset.
The recorded experiments were tested with Python **3.13.5** and the exact package versions in
[`requirements.txt`](requirements.txt).

## Setup

Clone the course repository:

```sh
git clone https://github.com/chongenhu/data-37711-autumn-2026.git
cd data-37711-autumn-2026
```

From the root of your cloned repository, create and activate a Python 3.13 environment:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the completed homework

Execute both questions from top to bottom:

```sh
python run_homework.py homework_1
```

Run a single question:

```sh
python run_homework.py homework_1 --question q4
python run_homework.py homework_1 --question q5
```

`--question all` is equivalent to the default. The runner uses the active Python interpreter
as the Jupyter kernel and sets each kernel's working directory to its notebook directory.
Temporary writable directories hold Matplotlib, IPython, and Jupyter runtime caches.

Freshly executed notebook copies are saved under `outputs/homework_1/q4/` and
`outputs/homework_1/q5/`, which are excluded from Git. The committed notebook files are not
rewritten by the runner. Each notebook also saves its numeric results, and Q4 saves its three
figures, in its own `results/` or `figures/` directory. A failed execution exits with a nonzero
status and preserves partial notebook output when available.

For interactive use, open either `.ipynb` file in a Jupyter-capable editor, select the Python
interpreter from `.venv`, and run all cells in order. Set the kernel working directory to the
folder containing that notebook so its `data/` paths resolve correctly. The command-line runner
already handles this working-directory setting.

## Current directory layout

```text
homework_1/
  q4/
    ML1_HW1_Q4.ipynb
    data/
    figures/
    results/
    requirements.txt
  q5/
    ML1_HW1_Q5.ipynb
    data/
    results/
    requirements.txt
requirements.txt
run_homework.py
README.md
```

Future assignments can be added as `homework_2/`, `homework_3/`, and so on when their actual
work is available. No unassigned homework or placeholder experiments are included.

## Homework 1 experiment scope

- **Question 4:** predict `ale` from `anchor_ratio`, `trans_range`, `node_density`, and
  `iterations` using LinearRegression, scaled RBF SVR, and a fixed depth-limited decision tree.
  Diagnose signed held-out test residuals with three separate normal probability plots and
  `scipy.stats.normaltest`; report test MAE as an additional prediction-error measure.
- **Question 5:** compare the sklearn logistic baseline with PyTorch logistic-BCE-SGD and
  probit-BCE-SGD on the breast cancer dataset's original 0/1 labels and all 30 features.
  Compare the sklearn linear baseline with PyTorch linear-MSE-SGD and linear-Student-t-NLL-SGD
  on ALE. Student-t degrees of freedom are fixed at 5 and scale at 1. Report test Accuracy
  for classification and test MAE for regression. No Q4 normality tests are repeated in Q5.

Q4 and Q5 both use the same ALE split: `test_size=0.2`, `random_state=42`, without
stratification. Q5 includes a local copy of Q4's saved split and verifies its settings, row
indices, and the real CSV's SHA-256 checksum before reuse. Input scaling is fitted only on
training data. The ALE target remains in its original units (meters), and `sd_ale` is ignored
as an input and target. All published numeric values come from actual execution; test metrics
were not used to select model settings.

## Data source and credit

ALE comes from [UCI dataset 844: Average Localization Error (ALE) in sensor node localization
process in WSNs](https://archive.ics.uci.edu/dataset/844/average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns).
The official file is `mcs_ds_edited_iter_shuffled.csv`, containing 107 rows and six columns,
with no missing values in the included copy. The official archive was downloaded on
October 14, 2026. Included copies are unchanged.

Singh, A., Kotiyal, V., Sharma, S., Nagar, J., & Lee, C. (2020).
*Average Localization Error (ALE) in sensor node localization process in WSNs* [Dataset].
UCI Machine Learning Repository. [DOI: 10.24432/C5ZP6R](https://doi.org/10.24432/C5ZP6R).
The ALE dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The classification data are loaded using
[`sklearn.datasets.load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).
The original labels are retained: `0 = malignant`, `1 = benign`.
