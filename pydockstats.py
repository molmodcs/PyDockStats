# -*- coding: utf-8 -*-
"""
Created on 06/03/2022

@authors: Matheus C. Mattos, Luciano T. Costa

This project is licensed under the GNU License - see the LICENSE.md file for details

PyDockStats version 1.0 (746241f) compiled by 'matheuscamposmattos@id.uff.br' on 2022-07-25

PyDockStats is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDockStats is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from calcs import optimal_threshold, calculate_hits, calculate_EF, calculate_TG, calculate_pTG, bedroc
from curves import PC, ROC
from utils import scale, num_derivative, find_nearest, read_result_file


NAME = "PyDockStats"
FORMULA = "activity ~ scores"
plt.style.use("fast")
formats = {"csv": pd.read_csv, "excel": pd.read_excel}
properties = dict()
model = LogisticRegression(solver="lbfgs", penalty="l2")



def read(file: str):
    if file.endswith((".xlsx", ".ods")):
        return formats["excel"](file)
    else:
        return formats["csv"](file, sep=None, engine='python')


def fit_predict(x, y):        
    x = x.reshape(-1, 1)
    print(y)
    clf = model.fit(x, y)
    predictions = clf.predict_proba(x)[:, 1]

    return predictions

def generateX(predictions):
    x = []
    # create the percentile axis (x)
    for idx, y_hat in enumerate(predictions):
        pct = (idx + 1) / len(predictions)

        x.append(pct)

    return np.array(x)

def generate_plots(names, scores, actives):
    pc = plt.figure(FigureClass=PC)
    roc = plt.figure(FigureClass=ROC)
    # iterate through the data for each name
    for i in range(0, len(names)):
        name = names[i]

        x = scores[i]
        activity = actives[i]
        predictions = fit_predict(x, activity)
        df = (x, predictions)

        # sorting
        y_true_sorted = [x for _, x in sorted(zip(predictions, activity))]
        predictions = np.array(sorted(predictions))

        # details
        x = np.concatenate(([0], generateX(predictions)))
        y_hat = np.concatenate(([0], predictions))

        # generating the roc curve
        fpr, tpr, thresholds = roc_curve([0] + y_true_sorted, y_hat, pos_label=1)

        # calculating derivatives

        x_prime, y_hat_prime = num_derivative(x, y_hat)
        x_prime, y_hat_prime = scale(x_prime), scale(y_hat_prime)

        # selecting which derivative is bigger than 0.4
        idx = np.where(y_hat_prime > 0.34)[0]
        if idx[0] != 0:
            idx = idx[0]
        elif len(idx) > 1:
            idx = idx[1]
        else:
            idx = idx[0]

        selected_t = y_hat_prime[idx]
        selected_x = x_prime[idx]

        # calculating the hits
        hits_x, hits_t = calculate_hits(y_true_sorted, idx, activity)

        # adding the metrics to the properties dict
        enrichment_factor = calculate_EF(hits_x, hits_t, 1 - selected_x)
        #self.properties["TG"] = self.calculate_TG(y_hat, p)
        #self.properties["pTG"] = self.calculate_pTG(y_hat, idx, p)

        # printing
        print(f"[*] {name}")
        print(f"Top {(1 - selected_x) * 100:.2f}% of the dataset:")
        print(f"-> EF: {enrichment_factor:.3f}")
        #print(f"-> pTG: {self.properties['pTG']:.3f}\n")

        # plotting

        plot = pc.plot(x, y_hat, label=names[i] + f" | {selected_x:.2f}")
        pc.ax.axvline(x=selected_x, linestyle="dashed", color=plot[0].get_c())

        AUC = auc(fpr, tpr)
        roc.plot(fpr, tpr, label=f"{name} | AUC = {AUC:.2f}", linewidth=2)

        pc.ax.legend()
        roc.ax.legend()

    plt.show()

def generate_data(names, scores, actives):
    pc = dict(x=[], y=[])
    roc = dict(x=[], y=[])
    # iterate through the data for each name
    for i in range(0, len(names)):
        name = names[i]

        x = scores[i]
        activity = actives[i]
        predictions = fit_predict(x, activity)
        df = (x, predictions)

        # sorting
        y_true_sorted = [x for _, x in sorted(zip(predictions, activity))]
        predictions = np.array(sorted(predictions))

        # details
        x = np.concatenate(([0], generateX(predictions)))
        y_hat = np.concatenate(([0], predictions))

        # generating the roc curve
        fpr, tpr, thresholds = roc_curve([0] + y_true_sorted, y_hat, pos_label=1)

        # calculating derivatives

        x_prime, y_hat_prime = num_derivative(x, y_hat)
        x_prime, y_hat_prime = scale(x_prime), scale(y_hat_prime)

        # selecting which derivative is bigger than 0.4
        idx = np.where(y_hat_prime > 0.34)[0]
        if idx[0] != 0:
            idx = idx[0]
        elif len(idx) > 1:
            idx = idx[1]
        else:
            idx = idx[0]

        selected_t = y_hat_prime[idx]
        selected_x = x_prime[idx]

        # calculating the hits
        hits_x, hits_t = calculate_hits(y_true_sorted, idx, activity)

        # adding the metrics to the properties dict
        enrichment_factor = calculate_EF(hits_x, hits_t, 1 - selected_x)
        #self.properties["TG"] = self.calculate_TG(y_hat, p)
        #self.properties["pTG"] = self.calculate_pTG(y_hat, idx, p)

        # printing
        print(f"[*] {name}")
        print(f"Top {(1 - selected_x) * 100:.2f}% of the dataset:")
        print(f"-> EF: {enrichment_factor:.3f}")
        #print(f"-> pTG: {self.properties['pTG']:.3f}\n")

        pc["x"].append(x)
        pc["y"].append(y_hat)

        roc["x"].append(fpr)
        roc["y"].append(tpr)

    return (pc, roc)
def get_names(number: int):
    names = []

    for i in range(number):
        names.append(f"program{i}")

    return names

def preprocess(df: pd.DataFrame):
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.dropna(axis=1, how='all')

        cols = df.columns

        names = get_names(int(len(cols) / 3))

        nprograms = len(names)

        scores = []
        actives = []
        for n in range(nprograms):
            idx = 3 * n + 1
            score_actives = df[[cols[idx], cols[idx+1]]]
            score_actives = score_actives.dropna(axis=0, how='all')

            coluna_score = score_actives.iloc[:, 0]
            coluna_actives = score_actives.iloc[:, 1]

            scores.append(coluna_score.to_numpy())
            actives.append(coluna_actives.to_numpy())

        return names, np.array(scores), np.array(actives)


def main():
    filename = "mpro2.csv"  # Replace with your data file name
    names = ["program1", "program2"]  # Replace with your program names
    ofname = "out.png"  # Replace with your output image name
    model_type = "logistic_regression"  # Replace with your desired model type

    # Preprocess the data
    names, scores, actives = preprocess(filename)

    # Calculate the PC and the ROC
    generate_plots(names, scores, actives)