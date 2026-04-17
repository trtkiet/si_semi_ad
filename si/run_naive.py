import numpy as np
import torch
import scipy.stats
import mpmath as mp

mp.dps=500

from .detection import anomaly_detection, get_ad_intervals_fast, top_k_normal_indices, get_top_k_normal_intervals
from .dnn.dnn import get_model_intervals as get_model_intervals_cpu
from .dnn_gpu.dnn import get_model_intervals as get_model_intervals_gpu
from .dnn_para.dnn import get_model_intervals as get_model_intervals_para
from .util import (
    gen_data,
    load_models,
    truncated_cdf,
    compute_etajTsigmaetaj_a_b,
    load_odds_data_for_si,
    load_known_normal_data
)

import time


def run(
    seed: int,
    delta: float = 0,
    n: int = 150,
    mu: float = 0,
    d: int = 8,
    anomaly_rate: float = 0.0,
    top_k_percent: float = 0.05,
    top_k_normal_percent: float = 0.3,
    deepsad_encoder=None,
    deepsad_c=None,
    device: str = "auto",
    dataset_name: str = None,
    data_root: str = "data",
    test_index_class: str = "normal",
    Sigma: np.ndarray = None,
):
    start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    requested_device = "auto" if device is None else str(device).lower()
    if requested_device not in {"auto", "cpu", "cuda", "dnn_para"}:
        raise ValueError(
            f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, dnn_para."
        )

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    if deepsad_encoder is None:
        load_device = None if requested_device == "auto" else requested_device
        deepsad_encoder, deepsad_c, _ = load_models(device=load_device)

    model_device = next(deepsad_encoder.parameters()).device
    target_model_device = requested_device
    if requested_device in {"auto", "dnn_para"}:
        target_model_device = model_device.type

    if (
        target_model_device in {"cpu", "cuda"}
        and model_device.type != target_model_device
    ):
        deepsad_encoder = deepsad_encoder.to(target_model_device)
        model_device = next(deepsad_encoder.parameters()).device
    if dataset_name is None:
        X, true_y, _ = gen_data(mu, delta, n, d, anomaly_rate, 0.0)
    else:
        X, true_y, _ = load_odds_data_for_si(
            dataset_name=dataset_name,
            root=data_root,
            train=False,
            random_state=seed,
            known_label_rate=0.0,
            percent_test_sample_size=0.5,
        )
        n, d = X.shape
    # print(f"Number of samples: {n}, Number of features: {d}")

    O = anomaly_detection(
        X,
        top_k_percent=top_k_percent,
        deepsad_encoder=deepsad_encoder,
        deepsad_c=deepsad_c,
    )
    if test_index_class not in {"normal", "anomaly"}:
        raise ValueError("test_index_class must be either 'normal' or 'anomaly'.")
    if test_index_class == "anomaly":
        candidates = [i for i in O if true_y[i] == 1]
    else:
        candidates = [i for i in O if true_y[i] == 0]
    if len(candidates) == 0:
        print(f"No '{test_index_class}' points for seed {seed}, skipping...")
        return None
    j = np.random.choice(candidates)
    if dataset_name is not None:
        X_normal = load_known_normal_data(
            dataset_name=dataset_name,
            root=data_root,
            random_state=seed
        )
    else:
        X_normal = gen_data(mu, 0, 1000, d, 0.0, 1.0)[0]
    X_mean = np.mean(X_normal, axis=0)
    # print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
    
    c = 0
    etaj = np.zeros(n * d)
    test_statistic = 0
    
    for i in range(d):
        sign = 1 if X[j, i] - X_mean[i] >= 0 else -1
        # print(f"Feature {i}, sign: {sign}, X[j, i]: {X[j, i]}, X_mean[i]: {X_mean[i]}")
        etaj[j * d + i] = sign
        c += -sign * X_mean[i]
        test_statistic += sign * (X[j, i] - X_mean[i])
    etajTx = etaj.T @ X.reshape(-1, 1)
    etajTx = etajTx.reshape(1, 1)
    # print(f"Test statistic for seed {seed}: {test_statistic}")
    # print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
    # print(f"c for seed {seed}: {c}")

    mu_vec = np.full((n * d, 1), mu)
    etajTmu = etaj.T.dot(mu_vec)
    etajTsigmaetaj, a, b = compute_etajTsigmaetaj_a_b(etaj, etajTx, X, n, d, S=Sigma)
    # print(f"etaj^T sigma etaj for seed {seed}: {etajTsigmaetaj[0][0]}")
    # print(f"Shapes of a and b for seed {seed}: {a.shape}, {b.shape}")

    a = a.reshape(n, d)
    b = b.reshape(n, d)

    postivie_sign = np.sign(X[j, :] - X_mean)

    itv = [-20 * np.sqrt(etajTsigmaetaj[0][0]), 20 * np.sqrt(etajTsigmaetaj[0][0])]
    # for i in range(d):
    #     new_a = (a[j, i] - mean_a_oc[i]) * postivie_sign[i]
    #     new_b = (b[j, i] - mean_b_oc[i]) * postivie_sign[i]

    #     if abs(new_b) < 1e-12:
    #         continue
    #     z = -new_a / new_b
    #     if new_b > 0:
    #         itv = [max(itv[0], z), itv[1]]
    #     else:
    #         itv = [itv[0], min(itv[1], z)]

    # itv[0] = itv[0].item() if isinstance(itv[0], np.ndarray) else itv[0]
    # itv[1] = itv[1].item() if isinstance(itv[1], np.ndarray) else itv[1]
    # # print(f"Initial interval for seed {seed}: {itv}")
    # if etajTx[0][0] > itv[1]:
    #     return 0.0

    # intervals = [(itv[0], itv[1], a, b)]

    # if requested_device == "dnn_para":
    #     para_device = "cuda" if torch.cuda.is_available() else "cpu"
    #     intervals = get_model_intervals_para(deepsad_encoder, intervals, para_device)
    # else:
    #     use_cuda_dnn = model_device.type == "cuda"
    #     if requested_device == "cpu":
    #         use_cuda_dnn = False
    #     elif requested_device == "cuda":
    #         use_cuda_dnn = True

    #     if use_cuda_dnn:
    #         model_dtype = next(deepsad_encoder.parameters()).dtype
    #         intervals_gpu = [
    #             (
    #                 left,
    #                 right,
    #                 torch.as_tensor(a_i, dtype=model_dtype, device=model_device),
    #                 torch.as_tensor(b_i, dtype=model_dtype, device=model_device),
    #             )
    #             for left, right, a_i, b_i in intervals
    #         ]
    #         intervals_gpu = get_model_intervals_gpu(deepsad_encoder, intervals_gpu)
    #         intervals = [
    #             (left, right, a_i.detach().cpu().numpy(), b_i.detach().cpu().numpy())
    #             for left, right, a_i, b_i in intervals_gpu
    #         ]
    #     else:
    #         intervals = get_model_intervals_cpu(deepsad_encoder, intervals)

    # print(f"Length of intervals after DNN processing for seed {seed}: {len(intervals)}")
    # print(f"Time after DNN processing for seed {seed}: {time.time() - start} seconds")
    # intervals = get_ad_intervals_fast(
    #     intervals, top_k_percent=top_k_percent, deepsad_c=deepsad_c
    # )
    # print(f"Length of intervals after AD processing for seed {seed}: {len(intervals)}")
    # print(f"Time after AD processing for seed {seed}: {time.time() - start} seconds")
    # final_intervals = []
    # for left, right, Oz in intervals:
    #     Oz = [i for i in Oz if known_y[i] == -1 or (known_y[i] == 1 and true_y[i] == 0)]
    #     Oz = sorted(Oz)
    #     final_intervals.append(
    #         (
    #             left / np.sqrt(etajTsigmaetaj[0][0]),
    #             right / np.sqrt(etajTsigmaetaj[0][0]),
    #             Oz,
    #         )
    #     )

    z = mp.mpf(etajTx[0][0] + c) / mp.sqrt(mp.mpf(etajTsigmaetaj[0][0]))

    cdf = mp.ncdf(z, mu=0, sigma=1)
    if cdf is None:
        print(f"Warning: CDF computation failed for seed {seed}. Skipping this run.")
        return None

    p_value = 2 * min(cdf, 1 - cdf)
    p_value = max(p_value, mp.mpf('1e-50'))  # cap at 1e-300 to avoid underflow to 0.0

    # Bonferroni in mpmath (no float conversion yet)
    # p_value = min(mp.mpf(1), p_value * mp.power(2, n))

    print(f"p-value for seed {seed}: {mp.nstr(p_value, 10)}")

    # Only convert to float if it won't underflow
    # if p_value < mp.mpf('1e-300'):
    #     print(f"Warning: p_value too small for float64, returning mpmath value")
    #     return p_value   # return as mpmath.mpf — don't force float()

    return float(p_value)
