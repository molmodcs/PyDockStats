import numpy as np

def optimal_threshold(fpr, tpr, thresholds):
    # selecting the optimal threshold based on ROC
    selected_t = thresholds[np.argmin(np.abs(fpr + tpr - 1))]

    return selected_t

def calculate_hits(y_true_sorted, idx_selected_t, activity):
    activity_topx = np.array(y_true_sorted[int(idx_selected_t):])


    hits_x = np.squeeze(np.where(activity_topx == 1)).size
    hits_t = np.squeeze(np.where(np.array(activity) == 1)).size

    return hits_x, hits_t

# Enrichment Factor
def calculate_EF(hits_x, hits_t, topx_percent):
    return (hits_x) / (hits_t * topx_percent)

# Total Gain
def calculate_TG(y_hat, p):
    return sum(abs(y_hat - p) / len(y_hat)) / (2 * p * (1 - p))

# Partial total gain from the selected threshold
def calculate_pTG(y_hat, idx_selected_t, p):
    return sum(abs(y_hat[idx_selected_t:] - p) / len(y_hat[idx_selected_t:])) / (
        2 * p * (1 - p)
    )

import numpy as np

def bedroc(x, y, decreasing=True, alpha=20.0):
    if len(x) != len(y):
        raise ValueError("The number of scores must be equal to the number of labels.")
    N = len(y)
    n = len(np.where(y == 1)[0])
    ord = np.argsort(x)[::-1] if decreasing else np.argsort(x)
    m_rank = np.where(y[ord] == 1)[0]
    s = np.sum(np.exp(-alpha * m_rank / N))
    ra = n / N
    ri = (N - n) / N
    random_sum = ra * np.exp(-alpha / N) * (1.0 - np.exp(-alpha)) / (1.0 - np.exp(-alpha / N))
    fac = ra * np.sinh(alpha / 2.0) / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra))
    cte = 1.0 / (1 - np.exp(alpha * ri))
    return s / random_sum * fac + cte
