# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 14:33:08 2022

@author: mathe
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from sklearn.metrics import auc, roc_curve
from matplotlib.widgets import Cursor
import sys, os
from random import sample


ID, SCORE, ACTIVITY = 0, 1, 2
COLORS = list(TABLEAU_COLORS)

FORMULA = 'activity ~ scores'
plt.style.use('fast')

class Docking:
    def __init__(self, argv=[__name__]):
        if len(sys.argv) != 5:
            print("Usage: <file> <number of programs> <names> <image>")
            return

        self.filename = sys.argv[1]
        self.nprograms = int(sys.argv[2])
        self.names = (sys.argv[3]).split(',')
        self.ofname = sys.argv[4]
        
        self.fpr=[]
        self.tpr=[]
        self.fig, self.ax = None, None
        self.fig_roc, self.ax_roc = None, None

    def main(self, argv=[__name__]):
        if len(sys.argv) != 5:
            return 1
        f, ext = os.path.splitext(self.ofname)
        
        #prepare dataframe
        df, scores, actives = self.prepare_df(self.filename, self.nprograms, self.names)
        #setup pc for matplotlib
        self.setup_pc()
        #setup roc for matplotlib
        self.setup_roc()
        
        #calculate the PC and the ROC
        self.calculate_plot(df, scores, actives, self.names)
    
        return 0
    
    def setup_pc(self):
        #dataset PC

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(12, 7)
        plt.tight_layout(pad=2.4)
        
    
        self.ax.set_title('Predictiveness Curve')
        self.ax.set_ylabel('Activity probability')
        self.ax.set_xlabel('Quantile')
        
        self.ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
        self.ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        
    
    
    def setup_roc(self):
        self.fig_roc, self.ax_roc = plt.subplots()
        self.fig_roc.set_size_inches(10,6)
        
        self.ax_roc.set_title('ROC curve')
        self.ax_roc.set_ylabel('Sensitivity')
        self.ax_roc.set_xlabel('1 - specificity')
        
        x_linear = np.linspace(0, 1)
        y_linear = np.linspace(0, 1)
        
        self.ax_roc.plot(x_linear, y_linear, linestyle='dashed', label='Random', color='red')
    
    def prepare_df(self, filenames, nprograms, names):
            
        df = pd.read_csv(filenames, sep=',')
        df = df.sample(frac=1).reset_index(drop=True)
        cols = df.columns
        scores = []
        actives = []
        for n in range(nprograms):
            scores.append(df[cols[3*n + 1]].to_numpy())
            actives.append(df[cols[3*n + 2]].to_numpy())
        
        df = pd.DataFrame()
        return df, scores, actives
    
    
    
    def plot_roc(self, fpr, tpr, name):
        AUC = auc(fpr, tpr)
        self.ax_roc.plot(fpr,tpr, label=f"{name} | AUC = {AUC:.2f}", linewidth=2)
        
    
    def calculate_plot(self, df, scores, actives, names):
    
        for i in range(0, len(names)):
            df['scores'] = scores[i]
            df['activity'] = actives[i]
            
            activity = actives[i]
            # calculate prevalence
            p = round(df.activity.mean(), 6)
            hitrate=p
            
            # define the model
            model = smf.glm(formula = FORMULA, data=df, family=sm.families.Binomial())
            #fit the model
            result = model.fit()
            
            predictions = list(sorted(result.predict()))
            X = [0]
            Y = np.array([0])

            Y = np.concatenate((Y, predictions))
            
            y_true_sorted = [x for _, x in sorted(zip(result.predict(), activity))]

            # create the
            for idx, y in enumerate(predictions):
                pct = (idx + 1) / len(predictions)
                
                X.append(pct)
            
            X = np.array(X)
            
            thresholds = np.arange(0, 1, 0.01)
            y_true = actives[i]
            
            self.fpr, self.tpr, thresholds = roc_curve(y_true, result.predict(), pos_label=1)
            selected_t = thresholds[np.argmin(np.abs(self.fpr+self.tpr-1))]
            
            interp = np.interp(selected_t, Y, X)
            
            
            #X = X[round(interp * len(X)):]
            #Y = Y[round(interp * len(Y)):]
            
            #calculate all metrics
            
            topx_percent = 1 - selected_t
            idx_selected_t = np.where(Y == selected_t)[0][0]
            
            activity_topx = np.array(y_true_sorted[idx_selected_t:])
            
            hits_x = np.squeeze(np.where(activity_topx==1)).size
            hits_t = np.squeeze(np.where(np.array(activity)==1)).size
            
            Nx = np.squeeze(np.where(Y > selected_t)).size
            Nt = len(activity)
            
            EF = (hits_x / Nx) / (hits_t / Nt)
            TG = sum( abs(Y - p)/len(Y)  ) / (2*p * (1 - p))
        
            print(f"EF {(1-interp)*100:.2f}% : {EF:.3f}")
            print(f"TG: {TG:.3f}")
            
            print(hits_x, Nx)
        
            #self.ax.axhline(y=selected_t, color=COLORS[i], linestyle='dashed')
            self.ax.axvline(x=interp, color=COLORS[i], linestyle='dashed', label = f'Cut = {interp:.2f}')
            
            self.ax.plot(X,Y, label = names[i], color=COLORS[i])
            self.plot_roc(self.fpr, self.tpr, names[i])
    

            self.ax.legend()
            self.ax_roc.legend()
        plt.show()
        
        self.ax.axhline(y=hitrate, color='grey', linestyle='dashed', alpha=0.5)
    
if __name__ == "__main__":
    docking = Docking(sys.argv)
    sys.exit(docking.main())       
    
    
    


    


