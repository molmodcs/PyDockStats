# -*- coding: utf-8 -*-
"""
Created on 06/03/2022

@authors: Matheus C. Mattos, Luciano T. Costa

This project is licensed under the GNU License - see the LICENSE.md file for details

PyDockStats version 1.0 (746241f) compiled by 'matheuscamposmattos@id.uff.br' on 2022-07-25

PyDockStats is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDockStats is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
"""
import streamlit as st
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
from curves import ROC, PC
from calcs import *
import utils

NAME = "PyDockStats"
FORMULA = "activity ~ scores"
plt.style.use("fast")


def parseArguments():
    parser = argparse.ArgumentParser(
        prog=f"{NAME}",
        description=f"""{NAME} is a Python tool that builds a ROC (Receiver operating characteristic) curve a
        and a Predictiveness Curve for Virtual Screening programs.""",
    )

    parser.add_argument("-f", "--file", dest="file", type=str, help="Data file")
    parser.add_argument(
        "-p",
        "--programs",
        dest="names",
        type=str,
        default="",
        help="Programs names separated by comma",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        default="out.png",
        help="Output image name",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        default="logistic_regression",
        help="Model type (logistic regression only)",
    )

    args = parser.parse_args()
    if not args.file:
        print("Sorry but I need a data file (pc_roc.py -f 'filename.csv')")
        exit(0)

    return args


curves = {
    "pc": plt.figure(
        FigureClass=PC, figsize=(12, 7), num="Predictiveness Curve"
    ),
    "roc": plt.figure(
        FigureClass=ROC,
        figsize=(10, 7),
        num="Receiver Operating Characteristic",
    ),
}
properties = {}
model = LogisticRegression(solver="lbfgs", penalty="l2")
df = None

def save_plots():
    for name, curve in curves.items():
        curve.savefig(f"{name}_{ofname}", dpi=300)

def get_names(number: int):
    names = []

    for i in range(number):
        names.append(f"program{i}")

    return names

def preprocess(filename: str):
    df = utils.read_result_file(filename)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.dropna(axis=1, how='all')

    cols = df.columns

    if not names:
        names = get_names(int(len(cols) / 3))

    else:
        names = names.split(",")

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

    return np.array(scores), np.array(actives)

def generateX(predictions):
    x = []
    # create the percentile axis (x)
    for idx, y_hat in enumerate(predictions):
        pct = (idx + 1) / len(predictions)

        x.append(pct)
    return np.array(x)



def fit_predict(x, y):        
    x = x.reshape(-1, 1)
    clf = model.fit(x, y)
    predictions = clf.predict_proba(x)[:, 1]

    return predictions

# Define a function to generate the plot
def generate_plots(scores, actives):
    names = names
    pc = curves["pc"]
    roc = curves["roc"]
    # iterate through the data for each name
    for i in range(0, nprograms):
        name = names[i]

        x = scores[i]
        activity = actives[i]
        predictions = fit_predict(x, activity)

        # sorting
        y_true_sorted = [x for _, x in sorted(zip(predictions, activity))]
        predictions = np.array(sorted(predictions))

        # details
        x = np.concatenate(([0], generateX(predictions)))
        y_hat = np.concatenate(([0], predictions))

        # generating the roc curve
        fpr, tpr, _ = roc_curve([0] + y_true_sorted, y_hat, pos_label=1)

        # calculating derivatives

        x_prime, y_hat_prime = utils.num_derivative(x, y_hat)
        x_prime, y_hat_prime = utils.scale(x_prime), utils.scale(y_hat_prime)

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
        properties["EF"] = calculate_EF(hits_x, hits_t, 1 - selected_x)
        #properties["TG"] = calculate_TG(y_hat, p)
        #properties["pTG"]...

    # Plot the ROC curve using Streamlit's `line_chart` function
    st.line_chart(pd.DataFrame({"fpr": fpr, "tpr": tpr}))

# Create a Streamlit app
def app():
    st.title("Interactive ROC Curve")

    # Add a file uploader to allow users to upload their data file
    data_file = st.file_uploader("Upload Data File", type=["csv"])


    # Add a button to generate the plot
    if st.button("Generate Plot"):

        if not data_file:
            st.warning("No data file.")
        else:
            # Read the data file using Pandas
            data = pd.read_csv(data_file)

            # Extract the scores and actives from the data file
            scores, actives = preprocess(data)

            # Call the generate_plots function to generate the ROC curve
            generate_plots(scores, actives)

# Run the app
if __name__ == "__main__":
    app()
