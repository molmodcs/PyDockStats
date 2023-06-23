import numpy as np
import pandas as pd

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


