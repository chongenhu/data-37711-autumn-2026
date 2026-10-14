# Homework 1

Use Python 3.13. Setup on macOS or Linux:

```sh
git clone https://github.com/chongenhu/data-37711-autumn-2026.git
cd data-37711-autumn-2026
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.3 pandas==2.3.3 scipy==1.16.2 scikit-learn==1.8.0 matplotlib==3.10.7 "jax[cpu]==0.11.2" optax==0.2.8
```

Run from the repository root:

```sh
python homework_1/Q4.py
python homework_1/Q5.py
```

Q4 prints the test results and saves three plots in `homework_1/figures/`.
Q5 runs all four subquestions and prints Accuracy, MAE, and training losses.
The ALE data are included. Recorded outputs are in `homework_1/results/Q4.txt` and `homework_1/results/Q5.txt`.
