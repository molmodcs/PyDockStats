import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

formats = {"csv": pd.read_csv, "excel": pd.read_excel}
def read_result_file(file: str):
    if file.endswith((".xlsx", ".ods")):
        return formats["excel"](file)
    else:
        return formats["csv"](file, sep=None, engine='python')
    
def save_plots(pc_data, roc_data, names, save_path="."):
    # create the pc fig and the roc fig each with the programs plotted
    # save the figs to the save_path

    # pc fig
    fig_pc = plt.figure(figsize=(10, 10))
    ax_pc = fig_pc.add_subplot(111)
    ax_pc.set_title("Predictiveness Curve")
    ax_pc.set_xlabel("Quantile")
    ax_pc.set_ylabel("Activity probability")
    ax_pc.set_ylim([0, 1])
    ax_pc.set_xlim([0, 1])
    ax_pc.grid(True)

    # roc fig
    fig_roc = plt.figure(figsize=(10, 10))
    ax_roc = fig_roc.add_subplot(111)
    ax_roc.set_title("ROC (Receiver Operating Characteristic)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_ylim([0, 1])
    ax_roc.set_xlim([0, 1])
    ax_roc.grid(True)

    # plot the data
    for i, name in enumerate(names):
        # pc
        x_pc, y_pc = pc_data[i]
        ax_pc.plot(x_pc, y_pc, label=name)

        # roc
        x_roc, y_roc = roc_data[i]
        ax_roc.plot(x_roc, y_roc, label=name)

    # save the figs
    fig_path = f"{save_path}/figs/pc.png"
    fig_pc.savefig(fig_path)

    roc_path = f"{save_path}/figs/roc.png"
    fig_roc.savefig(roc_path)

    return [fig_path, roc_path]

# Function to generate artificial scores for ligands and decoys
def generate_artificial_scores(num_ligands, num_decoys):
    # Generate random scores for ligands and decoys (between 0 and 1)

    # ligand scores between 0.5 and 1
    ligand_scores = np.random.uniform(0.5, 1, num_ligands)

    # decoy scores between 0 and 0.5
    decoy_scores = np.random.uniform(0, 0.5, num_decoys)

    ligand_labels = np.ones(num_ligands)
    decoy_labels = np.zeros(num_decoys)

    combined_ligand_data = np.column_stack((ligand_scores, ligand_labels))
    combined_decoy_data = np.column_stack((decoy_scores, decoy_labels))

    # save to csv with pandas
    df_ligands = pd.DataFrame(combined_ligand_data, columns=['scores', 'activity'])
    df_decoys = pd.DataFrame(combined_decoy_data, columns=['scores', 'activity'])

    df_ligands.to_csv('ligands.csv', index=False, header=True)
    df_decoys.to_csv('decoys.csv', index=False, header=True)

    return True
