#/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:33:08 2022

@authors: Matheus Campos de Mattos (matheuscamposmattos@id.uff.br)
Prof. Luciano T. Costa (ltcosta@id.uff.br)
MolMod-CS research group (www.molmodcs.uff.br)

This project is licensed under the GNU License - see the LICENSE.md file for details

PyDockStats version 1.0 (746241f) compiled by 'matheuscamposmattos@id.uff.br' on 2022-07-25

PyDockStats is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

PyDockStats is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import sys, os
import argparse
from sklearn.linear_model import LogisticRegression

NAME = "PyDockStats"

glm_names = ['glm','GLM', 'binomial family', 'glms', 'generalized linear models']

def parseArguments():
    parser = argparse.ArgumentParser(prog=f"{NAME}",
                                     description=f"""{NAME} is a Python tool that builds ROC (Receiver operating characteristic) and Predictiveness Curves from Virtual Screening data using different Docking programs.""")
    
    
    parser.add_argument("-f", "--file", dest="file", type=str, help="CSV or TXT input file with the Docking scores and activities")
    parser.add_argument("-p", "--progs", dest="progs", type=str, default="", help="Docking programs used separated by comma; example: -p vina,gold,dockthor. Look at that this order must be the same in your CSV file")
    parser.add_argument("-o", "--output", dest="output", type=str, default="output.png", help="ROC and PC Output files in PNG format ")
    parser.add_argument("-m", "--model", dest="model", type=str, default='logistic_regression', help="Model type (logistic regression or GLM)")

    args = parser.parse_args()
    if (not args.file):
        print("Sorry but I need a CSV or TXT input file (usage: pydockstats.py -f 'filename.csv')")
        sys.exit(0)

    return args


FORMULA = 'activity ~ scores'
plt.style.use('fast')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class PyDockStats:
    def __init__(self, arguments):

        self.filename = arguments['file']
        self.names = arguments['progs']
        self.nprograms = len(self.names.split(','))
        self.ofname = arguments['output']
        self.model_type = arguments['model']

        self.fpr = []
        self.tpr = []
        self.fig, self.ax = None, None
        self.fig_roc, self.ax_roc = None, None
        self.properties = {}
        self.model = LogisticRegression("none")

    def main(self):
        f, ext = os.path.splitext(self.ofname)

        # prepare dataframe
        df, scores, actives = self.prepare_df()
        # setup pc for matplotlib
        self.setup_pc()
        # setup roc for matplotlib
        self.setup_roc()

        # calculate the PC and the ROC
        self.calculate_plot(df, scores, actives, self.names)

        return 0

    # setup the PC
    def setup_pc(self):
        # dataset PC

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(8, 5)
        plt.tight_layout(pad=2.6)

        self.ax.set_title('Predictiveness Curve')
        self.ax.set_ylabel('Activity probability')
        self.ax.set_xlabel('Quantile')

        self.ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
        self.ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        self.ax.margins(x=0, y=0)

    # setup the ROC curve
    def setup_roc(self):
        self.fig_roc, self.ax_roc = plt.subplots()
        self.fig_roc.set_size_inches(7, 5)
        plt.tight_layout(pad=2.6)

        self.ax_roc.set_title('ROC curve')
        self.ax_roc.set_ylabel('Sensitivity')
        self.ax_roc.set_xlabel('1 - specificity')

        x_linear = np.linspace(0, 1)
        y_linear = np.linspace(0, 1)

        self.ax_roc.plot(x_linear, y_linear, linestyle='dashed', label='Random', color='red')
        self.ax_roc.margins(x=0, y=0)

    def get_names(self, number):
        names = []

        for i in range(number):
            names.append(f"program{i}")

        return names

    def prepare_df(self):

        df = pd.read_csv(self.filename, sep=',')
        df = df.sample(frac=1).reset_index(drop=True)
        cols = df.columns

        if not self.names:
            self.names = self.get_names(int(len(cols) / 3))
            self.nprograms = len(self.names)

        else:
            self.names = self.names.split(',')

        self.nprograms

        scores = []
        actives = []
        for n in range(self.nprograms):
            scores.append(df[cols[3 * n + 1]].to_numpy())
            actives.append(df[cols[3 * n + 2]].to_numpy())

        df = pd.DataFrame()
        return df, np.array(scores), np.array(actives)

    def plot_roc(self, fpr, tpr, name):
        AUC = auc(fpr, tpr)
        self.ax_roc.plot(fpr, tpr, label=f"{name} | AUC = {AUC:.2f}", linewidth=2)

    def init_model(self):
        if self.model_type in glm_names:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf



    def calculate_plot(self, df, scores, actives, names):

        # iterate through the data for each name
        for i in range(0, self.nprograms):
            name = names[i];
            df['scores'] = scores[i]
            df['activity'] = actives[i]

            X = scores[i]
            activity = actives[i]

            # calculate prevalence
            p = round(df.activity.mean(), 6)  # hitrate

            X = X.reshape(-1, 1)

            clf = LogisticRegression("none").fit(X, activity)
            predictions = clf.predict_proba(X)[:, 1]

            if self.model_type in glm_names:
                print("GLMs!")
                # define the model
                model = smf.glm(formula=FORMULA, data=df, family=sm.families.Binomial())

                # fit the model
                predictions = model.fit().predict()

            y_true_sorted = [x for _, x in sorted(zip(predictions, activity))]
            predictions = sorted(predictions)

            X = [0]
            Y = np.array([0])

            Y = np.concatenate((Y, predictions))

            # create the percentile axis (X)
            for idx, y in enumerate(predictions):
                pct = (idx + 1) / len(predictions)

                X.append(pct)

            X = np.array(X)

            # calculates the roc curve
            self.fpr, self.tpr, thresholds = roc_curve(y_true_sorted, predictions, pos_label=1)

            # selecting the optimal threshold based on ROC
            selected_t = thresholds[np.argmin(np.abs(self.fpr + self.tpr - 1))]

            # translate the selected threshold to the X axis
            interp = np.interp(selected_t, Y, X)

            # calculate all metrics
            topx_percent = 1 - interp
            idx_selected_t = np.where(Y == selected_t)[0][0]

            activity_topx = np.array(y_true_sorted[idx_selected_t:])

            hits_x = np.squeeze(np.where(activity_topx == 1)).size
            hits_t = np.squeeze(np.where(np.array(activity) == 1)).size

            # Nx = np.squeeze(np.where(Y > selected_t)).size
            # Nt = len(activity)

            # Enrichment Factor
            EF = (hits_x) / (hits_t * topx_percent)
            # Total Gain
            TG = sum(abs(Y - p) / len(Y)) / (2 * p * (1 - p))
            # Partial total gain from the selected threshold
            pTG = sum(abs(Y[idx_selected_t:] - p) / len(Y[idx_selected_t:])) / (2 * p * (1 - p))

            # adding the metrics to the properties dict
            self.properties['EF'] = EF
            self.properties['TG'] = TG
            self.properties['pTG'] = pTG

            print(f"[*] {name}")
            print(f"Top {(topx_percent) * 100:.2f}% of the dataset:")
            print(f"-> EF: {EF:.3f}")
            print(f"-> pTG: {pTG:.3f}\n")

            # self.ax.axhline(y=selected_t, color=COLORS[i], linestyle='dashed')
            
            plot = self.ax.plot(X, Y, label=names[i] + f" | {interp:.2f}")
            self.ax.axvline(x=interp, linestyle='dashed', color = plot[0].get_color())
            self.plot_roc(self.fpr, self.tpr, names[i])

            self.ax.legend()
            self.ax_roc.legend()
        plt.show()

        # self.ax.axhline(y=hitrate, color='grey', linestyle='dashed', alpha=0.5)

        # saving the curves
        self.fig.savefig("pc_" + self.ofname, dpi=300)
        self.fig_roc.savefig("roc_" + self.ofname, dpi=300)


if __name__ == "__main__":
    args = parseArguments()
    PyDockStats = PyDockStats(vars(args))
    sys.exit(PyDockStats.main())
