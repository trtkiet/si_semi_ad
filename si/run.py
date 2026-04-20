import numpy as np
import torch

from .detection import anomaly_detection, get_j_in_topk_intervals, get_j_in_topk_intervals_v2
from .dnn.dnn import get_model_intervals as get_model_intervals_cpu
from .dnn_gpu.dnn import get_model_intervals as get_model_intervals_gpu
from .dnn_para.dnn import get_model_intervals as get_model_intervals_para
from .util import (
    gen_data,
    load_models,
    truncated_cdf,
    compute_etajTsigmaetaj_a_b,
    load_odds_data_for_si,
)

import time

def run_all(
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
        X, true_y = gen_data(mu, delta, n, d, anomaly_rate)
    else:
        X, true_y = load_odds_data_for_si(
            dataset_name=dataset_name,
            split='test',
            root=data_root,
            random_state=seed,
            percent_sample_size=0.5,
        )
    n, d = X.shape
    if not np.all(np.isin(true_y, (-1, 1))):
        raise ValueError("true_y contains invalid labels; expected values in {-1,0,1}.")
    # print(f"Number of samples: {n}, Number of features: {d}")

    O = anomaly_detection(
        X,
        top_k_percent=top_k_percent,
        deepsad_encoder=deepsad_encoder,
        deepsad_c=deepsad_c,
    )
    # print(f"Percent of real anomalies in top {top_k_percent*100}% for seed {seed}: {np.mean(true_y[O] == -1) * 100:.2f}%")
    if test_index_class not in {"normal", "anomaly"}:
        raise ValueError("test_index_class must be either 'normal' or 'anomaly'.")
    if test_index_class == "anomaly":
        candidates = [i for i in O if true_y[i] == -1]
    else:
        candidates = [i for i in O if true_y[i] == 1]
    if len(candidates) == 0:
        print(f"No '{test_index_class}' points for seed {seed}, skipping...")
        return []
    p_values = []
    for j in candidates:
        if dataset_name is not None:
            X_normal, _ = load_odds_data_for_si(
                dataset_name=dataset_name,
                root=data_root,
                split="reference",
                random_state=seed,
                percent_sample_size=0.8
            )
        else:
            X_normal = gen_data(mu, 0, 100, d, 0.0)[0]
        X_mean = np.mean(X_normal, axis=0)
        # print(f"Max absolute value in normal mean: {np.max(np.abs(X_mean))}")
        # print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
        X_concat = np.vstack([X, X_normal])
        
        c = 0
        etaj = np.zeros(n * d + X_normal.shape[0] * X_normal.shape[1])
        test_statistic = 0
        
        for i in range(d):
            sign = 1 if X[j, i] - X_mean[i] >= 0 else -1
            # print(f"Feature {i}, sign: {sign}, X[j, i]: {X[j, i]}, X_mean[i]: {X_mean[i]}")
            etaj[j * d + i] = sign
            for u in range(X_normal.shape[0]):
                etaj[n * d + u * d + i] = -sign / X_normal.shape[0]
            test_statistic += sign * (X[j, i] - X_mean[i])

        etajTx = etaj.T @ X_concat.reshape(-1, 1)
        etajTx = etajTx.reshape(1, 1)
        # print(f"Test statistic for seed {seed}: {test_statistic}")
        # print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
        # print(f"c for seed {seed}: {c}")

        # mu_vec = np.full((n * d, 1), mu)
        # etajTmu = etaj.T.dot(mu_vec)
        etajTsigmaetaj, a, b = compute_etajTsigmaetaj_a_b(etaj, etajTx, X_concat, n + X_normal.shape[0], d, S=Sigma)
        # print(f"etaj^T sigma etaj for seed {seed}: {np.sqrt(etajTsigmaetaj[0][0])}")
        # print(f"Shapes of a and b for seed {seed}: {a.shape}, {b.shape}")

        a = a.reshape(n + X_normal.shape[0], d)
        b = b.reshape(n + X_normal.shape[0], d)

        postivie_sign = np.sign(X[j, :] - X_mean)

        itv = [-20 * np.sqrt(etajTsigmaetaj[0][0]), 20 * np.sqrt(etajTsigmaetaj[0][0])]
        a_mean = a[n :, :].mean(axis=0)
        b_mean = b[n :, :].mean(axis=0)
        for i in range(d):
            new_a = (a[j, i] - a_mean[i]) * postivie_sign[i]
            new_b = (b[j, i] - b_mean[i]) * postivie_sign[i]

            if abs(new_b) < 1e-16:
                continue
            z = -new_a / new_b
            # print(f"Feature {i}, new_a: {new_a}, new_b: {new_b}, z: {z}")
            if new_b > 0:
                itv = [max(itv[0], z), itv[1]]
            else:
                itv = [itv[0], min(itv[1], z)]

        itv[0] = itv[0].item() if isinstance(itv[0], np.ndarray) else itv[0]
        itv[1] = itv[1].item() if isinstance(itv[1], np.ndarray) else itv[1]
        # print(f"Initial interval for seed {seed}: {itv}")
        if etajTx[0][0] > itv[1]:
            p_values.append(0.0)
            continue

        intervals = [(itv[0], itv[1], a[:n], b[:n])]

        if requested_device == "dnn_para":
            para_device = "cuda" if torch.cuda.is_available() else "cpu"
            intervals = get_model_intervals_para(deepsad_encoder, intervals, para_device)
        else:
            use_cuda_dnn = model_device.type == "cuda"
            if requested_device == "cpu":
                use_cuda_dnn = False
            elif requested_device == "cuda":
                use_cuda_dnn = True

            if use_cuda_dnn:
                si_dtype = torch.float64
                intervals_gpu = [
                    (
                        left,
                        right,
                        torch.as_tensor(a_i, dtype=si_dtype, device=model_device),
                        torch.as_tensor(b_i, dtype=si_dtype, device=model_device),
                    )
                    for left, right, a_i, b_i in intervals
                ]
                intervals_gpu = get_model_intervals_gpu(deepsad_encoder, intervals_gpu)
                intervals = [
                    (left, right, a_i.detach().cpu().numpy(), b_i.detach().cpu().numpy())
                    for left, right, a_i, b_i in intervals_gpu
                ]
            else:
                intervals = get_model_intervals_cpu(deepsad_encoder, intervals)

        # print(f"Length of intervals after DNN processing for seed {seed}: {len(intervals)}")
        # print(f"Time after DNN processing for seed {seed}: {time.time() - start} seconds")
        intervals = get_ad_intervals(
            intervals, top_k_percent=top_k_percent, deepsad_c=deepsad_c
        )
        # print(f"Length of intervals after AD processing for seed {seed}: {len(intervals)}")
        # print(f"Time after AD processing for seed {seed}: {time.time() - start} seconds")
        final_intervals = []
        for left, right, Oz in intervals:
            Oz = sorted(Oz)
            final_intervals.append(
                (
                    (left),
                    (right),
                    j in Oz,
                )
            )

        cdf = truncated_cdf(
            0, np.sqrt(etajTsigmaetaj[0][0]), final_intervals, j in O, (etajTx[0][0])
        )
        # print full precision cdf value for debugging
        # print(f"Truncated CDF for seed {seed}: {cdf:.10f}")
        if cdf is None:
            print(f"Warning: CDF computation failed for seed {seed}. Skipping this run.")
            print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
            print(f"Test statistic for seed {seed}: {test_statistic}")
            print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
            print(f"Initial interval for seed {seed}: {itv}")
            continue
        p_value = 2 * min(cdf, 1 - cdf)
        if p_value == 0:
            print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
            print(f"Test statistic for seed {seed}: {test_statistic}")
            print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
            print(f"Initial interval for seed {seed}: {itv}")
        if p_value < 0.05:
            print(f"Average Distance per dimension of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean)) / d}")
            print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
            print(f"Test statistic for seed {seed}: {test_statistic}")
            print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
            print(f"Initial interval for seed {seed}: {itv}")
            print(f"Etaj^T sigma etaj for seed {seed}: {np.sqrt(etajTsigmaetaj[0][0])}")
        print(f"p-value for seed {seed}: {p_value}")
        p_values.append(p_value)
    return p_values

def run_one(
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
        X, true_y = gen_data(mu, delta, n, d, anomaly_rate)
    else:
        X, true_y = load_odds_data_for_si(
            dataset_name=dataset_name,
            split='test',
            root=data_root,
            random_state=seed,
            percent_sample_size=0.9,
        )
    n, d = X.shape
    if not np.all(np.isin(true_y, (-1, 1))):
        raise ValueError("true_y contains invalid labels; expected values in {-1,0,1}.")
    # print(f"Number of samples: {n}, Number of features: {d}")

    O = anomaly_detection(
        X,
        top_k_percent=top_k_percent,
        deepsad_encoder=deepsad_encoder,
        deepsad_c=deepsad_c,
    )
    # print(f"Percent of real anomalies in top {top_k_percent*100}% for seed {seed}: {np.mean(true_y[O] == -1) * 100:.2f}%")
    if test_index_class not in {"normal", "anomaly"}:
        raise ValueError("test_index_class must be either 'normal' or 'anomaly'.")
    if test_index_class == "anomaly":
        candidates = [i for i in O if true_y[i] == -1]
    else:
        candidates = [i for i in O if true_y[i] == 1]
    if len(candidates) == 0:
        print(f"No '{test_index_class}' points for seed {seed}, skipping...")
        return []
    j = np.random.choice(candidates)
    if dataset_name is not None:
        X_normal, _ = load_odds_data_for_si(
            dataset_name=dataset_name,
            root=data_root,
            split="reference",
            random_state=seed,
            percent_sample_size=1.0
        )
    else:
        X_normal = gen_data(mu, 0, 100, d, 0.0)[0]
    X_mean = np.mean(X_normal, axis=0)
    # print(f"Max absolute value in normal mean: {np.max(np.abs(X_mean))}")
    # print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
    X_concat = np.vstack([X, X_normal])
    
    c = 0
    etaj = np.zeros(n * d + X_normal.shape[0] * X_normal.shape[1])
    test_statistic = 0
    
    for i in range(d):
        sign = 1 if X[j, i] - X_mean[i] >= 0 else -1
        # print(f"Feature {i}, sign: {sign}, X[j, i]: {X[j, i]}, X_mean[i]: {X_mean[i]}")
        etaj[j * d + i] = sign
        for u in range(X_normal.shape[0]):
            etaj[n * d + u * d + i] = -sign / X_normal.shape[0]
        test_statistic += sign * (X[j, i] - X_mean[i])

    etajTx = etaj.T @ X_concat.reshape(-1, 1)
    etajTx = etajTx.reshape(1, 1)
    # print(f"Test statistic for seed {seed}: {test_statistic}")
    # print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
    # print(f"c for seed {seed}: {c}")

    # mu_vec = np.full((n * d, 1), mu)
    # etajTmu = etaj.T.dot(mu_vec)
    etajTsigmaetaj, a, b = compute_etajTsigmaetaj_a_b(etaj, etajTx, X_concat, n + X_normal.shape[0], d, S=Sigma)
    # print(f"etaj^T sigma etaj for seed {seed}: {np.sqrt(etajTsigmaetaj[0][0])}")
    # print(f"Shapes of a and b for seed {seed}: {a.shape}, {b.shape}")

    a = a.reshape(n + X_normal.shape[0], d)
    b = b.reshape(n + X_normal.shape[0], d)

    postivie_sign = np.sign(X[j, :] - X_mean)

    itv = [-20 * np.sqrt(etajTsigmaetaj[0][0]), 20 * np.sqrt(etajTsigmaetaj[0][0])]
    a_mean = a[n :, :].mean(axis=0)
    b_mean = b[n :, :].mean(axis=0)
    for i in range(d):
        new_a = (a[j, i] - a_mean[i]) * postivie_sign[i]
        new_b = (b[j, i] - b_mean[i]) * postivie_sign[i]

        if abs(new_b) < 1e-16:
            continue
        z = -new_a / new_b
        # print(f"Feature {i}, new_a: {new_a}, new_b: {new_b}, z: {z}")
        if new_b > 0:
            itv = [max(itv[0], z), itv[1]]
        else:
            itv = [itv[0], min(itv[1], z)]

    itv[0] = itv[0].item() if isinstance(itv[0], np.ndarray) else itv[0]
    itv[1] = itv[1].item() if isinstance(itv[1], np.ndarray) else itv[1]
    # print(f"Initial interval for seed {seed}: {itv}")
    if etajTx[0][0] > itv[1]:
        return [0.0]

    intervals = [(itv[0], itv[1], a[:n], b[:n])]

    if requested_device == "dnn_para":
        para_device = "cuda" if torch.cuda.is_available() else "cpu"
        intervals = get_model_intervals_para(deepsad_encoder, intervals, para_device)
    else:
        use_cuda_dnn = model_device.type == "cuda"
        if requested_device == "cpu":
            use_cuda_dnn = False
        elif requested_device == "cuda":
            use_cuda_dnn = True

        if use_cuda_dnn:
            si_dtype = torch.float64
            intervals_gpu = [
                (
                    left,
                    right,
                    torch.as_tensor(a_i, dtype=si_dtype, device=model_device),
                    torch.as_tensor(b_i, dtype=si_dtype, device=model_device),
                )
                for left, right, a_i, b_i in intervals
            ]
            intervals_gpu = get_model_intervals_gpu(deepsad_encoder, intervals_gpu)
            intervals = [
                (left, right, a_i.detach().cpu().numpy(), b_i.detach().cpu().numpy())
                for left, right, a_i, b_i in intervals_gpu
            ]
        else:
            intervals = get_model_intervals_cpu(deepsad_encoder, intervals)

    # print(f"Length of intervals after DNN processing for seed {seed}: {len(intervals)}")
    # print(f"Time after DNN processing for seed {seed}: {time.time() - start} seconds")
    intervals = get_j_in_topk_intervals_v2(
        intervals, top_k_percent=top_k_percent, deepsad_c=deepsad_c, j=j
    )
    # print(f"Length of intervals after AD processing for seed {seed}: {len(intervals)}")
    # print(f"Time after AD processing for seed {seed}: {time.time() - start} seconds")
    final_intervals = []
    # print(f"Observed O: {O}")
    for left, right, Oz in intervals:
        # print(f"Interval: ({left}, {right}), Oz: {Oz}, j in Oz: {j in Oz}")
        final_intervals.append(
            (
                (left),
                (right),
                Oz,
                # Oz
            )
        )

    cdf = truncated_cdf(
        0, np.sqrt(etajTsigmaetaj[0][0]), final_intervals, j in O, (etajTx[0][0])
    )
    # print full precision cdf value for debugging
    # print(f"Truncated CDF for seed {seed}: {cdf:.10f}")
    if cdf is None:
        print(f"Warning: CDF computation failed for seed {seed}. Skipping this run.")
        print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
        print(f"Test statistic for seed {seed}: {test_statistic}")
        print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
        print(f"Initial interval for seed {seed}: {itv}")
        return [None]
    p_value = 2 * min(cdf, 1 - cdf)
    if p_value == 0:
        print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
        print(f"Test statistic for seed {seed}: {test_statistic}")
        print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
        print(f"Initial interval for seed {seed}: {itv}")
    if p_value < 0.05:
        print(f"Average Distance per dimension of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean)) / d}")
        print(f"Distance of test point to normal mean for seed {seed}: {np.sum(np.abs(X[j] - X_mean))}")
        print(f"Test statistic for seed {seed}: {test_statistic}")
        print(f"etaj^T x for seed {seed}: {etajTx[0][0]}")
        print(f"Initial interval for seed {seed}: {itv}")
        print(f"Etaj^T sigma etaj for seed {seed}: {np.sqrt(etajTsigmaetaj[0][0])}")
    print(f"p-value for seed {seed}: {p_value}")
    return [p_value]

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
    pass
