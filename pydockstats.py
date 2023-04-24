# -*- coding: utf-8 -*-
"""
Created on 06/03/2022

@authors: Matheus C. Mattos, Luciano T. Costa

This project is licensed under the GNU License - see the LICENSE.md file for details

PyDockStats version 1.0 (746241f) compiled by 'matheuscamposmattos@id.uff.br' on 2022-07-25

PyDockStats is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDockStats is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
"""
import argparse
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression


NAME = "PyDockStats"
FORMULA = "activity ~ scores"
plt.style.use("fast")
formats = {"csv": pd.read_csv, "excel": pd.read_excel}


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


def scale(x: np.array) -> np.array:
    _max = x.max()
    new = x / _max

    return new


def num_derivative(x: np.array, y: np.array) -> np.array:
    yprime = np.diff(y) / np.diff(x)
    xprime = []

    for i in range(len(yprime)):
        xtemp = (x[i + 1] + x[i]) / 2
        xprime = np.append(xprime, xtemp)

    return xprime, yprime


# aux functions
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def read(file: str):
    if file.endswith((".xlsx", ".ods")):
        return formats["excel"](file)
    else:
        return formats["csv"](file, sep=None, engine='python')


# inheriting from matplotlib Figure
class Curve(Figure):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ax = self.add_axes([0.08, 0.1, 0.86, 0.84])
        self.ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
        self.ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        self.ax.margins(x=0, y=0)

    def plot(self, x, y, **kwargs):
        return self.ax.plot(x, y, **kwargs)

    def get_color(self, idx: int) -> str:
        lines = self.ax.get_lines()
        return lines[idx].get_c()


class PC(Curve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Predictiveness Curve"
        self.ax.set_title(self.name)
        self.ax.set_ylabel("Activity probability")
        self.ax.set_xlabel("Quantile")


class ROC(Curve):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Receiver Operating Characteristic"
        self.ax.set_title("ROC curve")
        self.ax.set_ylabel("Sensitivity")
        self.ax.set_xlabel("1 - specificity")
        self.ax.plot(
            np.linspace(0, 1),
            np.linspace(0, 1),
            linestyle="dashed",
            label="Random",
            color="red",
        )


class PyDockStats:
    def __init__(self, names: List, model: str, ofname: str):
        self.names = names
        self.ofname = ofname
        self.model_type = model
        self.nprograms = 0
        self.curves = {
            "pc": plt.figure(
                FigureClass=PC, figsize=(12, 7), num="Predictiveness Curve"
            ),
            "roc": plt.figure(
                FigureClass=ROC,
                figsize=(10, 7),
                num="Receiver Operating Characteristic",
            ),
        }
        self.properties = {}
        self.model = LogisticRegression(solver="lbfgs", penalty="l2")
        self.df = None

    def save_plots(self):
        for name, curve in self.curves.items():
            curve.savefig(f"{name}_{self.ofname}", dpi=300)

    def get_names(self, number: int):
        names = []

        for i in range(number):
            names.append(f"program{i}")

        return names

    def preprocess(self, filename: str):
        df = read(filename)
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.dropna(axis=1, how='all')

        cols = df.columns

        if not self.names:
            self.names = self.get_names(int(len(cols) / 3))

        else:
            self.names = self.names.split(",")

        self.nprograms = len(self.names)

        scores = []
        actives = []
        for n in range(self.nprograms):
            idx = 3 * n + 1
            score_actives = df[[cols[idx], cols[idx+1]]]
            score_actives = score_actives.dropna(axis=0, how='all')

            coluna_score = score_actives.iloc[:, 0]
            coluna_actives = score_actives.iloc[:, 1]

            scores.append(coluna_score.to_numpy())
            actives.append(coluna_actives.to_numpy())

        return np.array(scores), np.array(actives)

    def generateX(self, predictions):
        x = []
        # create the percentile axis (x)
        for idx, y_hat in enumerate(predictions):
            pct = (idx + 1) / len(predictions)

            x.append(pct)

        return np.array(x)

    def optimal_threshold(self, fpr, tpr, thresholds):
        # selecting the optimal threshold based on ROC
        selected_t = thresholds[np.argmin(np.abs(fpr + tpr - 1))]

        return selected_t

    def calculate_hits(self, y_true_sorted, idx_selected_t, activity):

        activity_topx = np.array(y_true_sorted[idx_selected_t:])

        hits_x = np.squeeze(np.where(activity_topx == 1)).size
        hits_t = np.squeeze(np.where(np.array(activity) == 1)).size

        return hits_x, hits_t

    # Enrichment Factor
    def calculate_EF(self, hits_x, hits_t, topx_percent):
        return (hits_x) / (hits_t * topx_percent)

    # Total Gain
    def calculate_TG(self, y_hat, p):
        return sum(abs(y_hat - p) / len(y_hat)) / (2 * p * (1 - p))

    # Partial total gain from the selected threshold
    def calculate_pTG(self, y_hat, idx_selected_t, p):
        return sum(abs(y_hat[idx_selected_t:] - p) / len(y_hat[idx_selected_t:])) / (
            2 * p * (1 - p)
        )

    def fit_predict(self, x, y):        
        x = x.reshape(-1, 1)
        clf = self.model.fit(x, y)
        predictions = clf.predict_proba(x)[:, 1]

        return predictions

    def generate_plots(self, scores, actives):
        names = self.names
        pc = self.curves["pc"]
        roc = self.curves["roc"]
        # iterate through the data for each name
        for i in range(0, self.nprograms):
            name = names[i]

            x = scores[i]
            activity = actives[i]
            predictions = self.fit_predict(x, activity)
            self.df = (x, predictions)

            # sorting
            y_true_sorted = [x for _, x in sorted(zip(predictions, activity))]
            predictions = np.array(sorted(predictions))

            # details
            x = np.concatenate(([0], self.generateX(predictions)))
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
            hits_x, hits_t = self.calculate_hits(y_true_sorted, idx, activity)

            # adding the metrics to the properties dict
            self.properties["EF"] = self.calculate_EF(hits_x, hits_t, 1 - selected_x)
            #self.properties["TG"] = self.calculate_TG(y_hat, p)
            #self.properties["pTG"] = self.calculate_pTG(y_hat, idx, p)

            # printing
            print(f"[*] {name}")
            print(f"Top {(1 - selected_x) * 100:.2f}% of the dataset:")
            print(f"-> EF: {self.properties['EF']:.3f}")
            #print(f"-> pTG: {self.properties['pTG']:.3f}\n")

            # plotting

            plot = pc.plot(x, y_hat, label=names[i] + f" | {selected_x:.2f}")
            pc.ax.axvline(x=selected_x, linestyle="dashed", color=plot[0].get_c())

            AUC = auc(fpr, tpr)
            roc.plot(fpr, tpr, label=f"{name} | AUC = {AUC:.2f}", linewidth=2)

            pc.ax.legend()
            roc.ax.legend()
        self.save_plots()

        plt.show()


import cProfile

if __name__ == "__main__":
    args = vars(parseArguments())

    filename = args["file"]
    names = args["names"]
    nprograms = len(names.split(","))
    ofname = args["output"]
    model_type = args["model"]

    pydock = PyDockStats(names, model_type, ofname)

    # preprocess the data
    scores, actives = pydock.preprocess(filename)

    # calculate the PC and the ROC
    pydock.generate_plots(scores, actives)
