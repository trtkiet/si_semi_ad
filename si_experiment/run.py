import numpy as np
import torch

from .detection import anomaly_detection, get_ad_intervals_fast
from .dnn import get_model_intervals
from .util import gen_data, load_models, truncated_cdf, compute_etajTsigmaetaj_a_b

import time


def run(
    seed: int,
    delta: float = 0,
    n: int = 150,
    mu: float = 0,
    d: int = 8,
    anomaly_rate: float = 0.0,
    known_label_rate: float = 0.2,
    top_k_percent: float = 0.05,
    deepsad_encoder=None,
    deepsad_c=None,
    device=None,    
):
    start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    if deepsad_encoder is None:
        deepsad_encoder, deepsad_c, device = load_models()

    X, true_y, known_y = gen_data(mu, delta, n, d, anomaly_rate, known_label_rate)

    O = anomaly_detection(
        X,
        top_k_percent=top_k_percent,
        deepsad_encoder=deepsad_encoder,
        deepsad_c=deepsad_c,
    )
    O = [i for i in O if known_y[i] == -1 or (known_y[i] == 1 and true_y[i] == 0)]
    true_O = [i for i in range(n) if true_y[i] == 1]
    # if len(true_O) == 0:
    #     print(f"No anomalies in seed {seed}, skipping...")
    #     return None
    # j = np.random.choice(true_O)
    Oc = [i for i in range(n) if i not in O]
    
    j = np.random.choice(O)

    mu_vec = np.full((n * d, 1), mu)
    # sigma = np.identity(n * d)

    etj = np.zeros((n * d, 1))
    etOc = np.zeros((n * d, 1))

    diff = X[j, :] - np.mean(X[Oc, :], axis=0)
    flip_mask = diff < 0

    etj[j * d : (j + 1) * d] = 1

    flip_indices = j * d + np.where(flip_mask)[0]
    etj[flip_indices] = -1

    for i in Oc:
        etOc[i * d : (i + 1) * d] = 1 / len(Oc)
    for i in Oc:
        etOc[i * d + np.where(flip_mask)[0]] = -1 / len(Oc)
    etaj = etj - etOc
    etajTx = etaj.T @ X.reshape(-1, 1)
    # print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")

    mu_vec = np.full((n * d, 1), mu)
    etajTmu = etaj.T.dot(mu_vec)
    etajTsigmaetaj, a, b = compute_etajTsigmaetaj_a_b(etaj, etajTx, X, n, d, S=None)
    # print(f"etaj^T sigma etaj for seed {seed}: {etajTsigmaetaj[0][0]}")
    # print(f"Shapes of a and b for seed {seed}: {a.shape}, {b.shape}")
    
    a = a.reshape(n, d)
    b = b.reshape(n, d)

    avg_x_oc = np.mean(X[Oc, :], axis=0)
    mean_a_oc = np.mean(a[Oc, :], axis=0)
    mean_b_oc = np.mean(b[Oc, :], axis=0)

    postivie_sign = np.sign(X[j, :] - avg_x_oc)

    itv = [-20 * np.sqrt(etajTsigmaetaj[0][0]), 20 * np.sqrt(etajTsigmaetaj[0][0])]
    for i in range(d):
        new_a = (a[j, i] - mean_a_oc[i]) * postivie_sign[i]
        new_b = (b[j, i] - mean_b_oc[i]) * postivie_sign[i]

        if abs(new_b) < 1e-12:
            continue
        z = -new_a / new_b
        if new_b > 0:
            itv = [max(itv[0], z), itv[1]]
        else:
            itv = [itv[0], min(itv[1], z)]

    itv[0] = itv[0].item() if isinstance(itv[0], np.ndarray) else itv[0]
    itv[1] = itv[1].item() if isinstance(itv[1], np.ndarray) else itv[1]
    # print(f"Initial interval for seed {seed}: {itv}")
    if etajTx[0][0] > itv[1]:
        return 0.0

    intervals = [(itv[0], itv[1], a, b)]
    intervals = get_model_intervals(deepsad_encoder, intervals)
    print(f"Length of intervals after DNN processing for seed {seed}: {len(intervals)}")
    # print(f"Time after DNN processing for seed {seed}: {time.time() - start} seconds")
    intervals = get_ad_intervals_fast(
        intervals, top_k_percent=top_k_percent, deepsad_c=deepsad_c
    )
    print(f"Length of intervals after AD processing for seed {seed}: {len(intervals)}")
    # print(f"Time after AD processing for seed {seed}: {time.time() - start} seconds")
    final_intervals = []
    for left, right, Oz in intervals:
        Oz = [i for i in Oz if known_y[i] == -1 or (known_y[i] == 1 and true_y[i] == 0)]
        Oz = sorted(Oz)
        final_intervals.append((left / np.sqrt(etajTsigmaetaj[0][0]), right / np.sqrt(etajTsigmaetaj[0][0]), Oz))

    cdf = truncated_cdf(
        0, 1, final_intervals, O, etajTx[0][0]/np.sqrt(etajTsigmaetaj[0][0])
    )
    p_value = 2 * min(cdf, 1 - cdf)
    print(f"p-value for seed {seed}: {p_value}")
    return p_value
