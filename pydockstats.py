import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from calcs import optimal_threshold, calculate_hits, calculate_EF, calculate_TG, calculate_pTG, bedroc
from curves import PC, ROC
from putils import scale, num_derivative, find_nearest, read_result_file

# Constants
FORMATS = {"csv": pd.read_csv, "excel": pd.read_excel}
MODEL = LogisticRegression(solver="lbfgs", penalty="l2")

# Functions
def read(file: str):
    format_func = FORMATS["excel"] if file.endswith((".xlsx", ".ods")) else FORMATS["csv"]
    return format_func(file, sep=None, engine='python')

def fit_predict(x, y):
    x = x.reshape(-1, 1)
    clf = MODEL.fit(x, y)
    predictions = clf.predict_proba(x)[:, 1]
    return predictions

def generate_percentiles(predictions):
    percentiles = np.arange(1, len(predictions) + 1) / len(predictions)
    return percentiles

def preprocess_data(df):
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.dropna(axis=1, how='all')

    cols = df.columns
    scores = df[cols[0]].values
    actives = df[cols[1]].values

    return np.array(scores), np.array(actives)

def generate_plots(program_name, scores, activity):

    x = scores
    predictions = fit_predict(x, activity)
    sorted_indices = np.argsort(predictions)

    fpr, tpr, _ = roc_curve(activity[sorted_indices], predictions[sorted_indices], pos_label=1)

    selected_x = calculate_selected_x(predictions)
    hits_x, hits_t = calculate_hits(activity[sorted_indices], selected_x, activity)
    enrichment_factor = calculate_EF(hits_x, hits_t, 1 - selected_x)

    print(f"[*] {program_name}")
    print(f"Top {(1 - selected_x) * 100:.2f}% of the dataset:")
    print(f"-> EF: {enrichment_factor:.3f}")

    pc_x, pc_y = generate_percentiles(predictions[sorted_indices]), predictions[sorted_indices]
    
    roc_data = dict(x=fpr, y=tpr, auc=auc(fpr, tpr))
    pc_data= dict(x=pc_x, y=pc_y, auc=bedroc(pc_x, pc_y, 0.1))

    return pc_data, roc_data

def calculate_selected_x(predictions):
    x_prime, y_hat_prime = num_derivative(generate_percentiles(predictions), predictions)
    x_prime, y_hat_prime = scale(x_prime), scale(y_hat_prime)

    threshold_idx = np.argmax(y_hat_prime > 0.34)
    return x_prime[threshold_idx]
