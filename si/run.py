import numpy as np
import torch

from .detection import anomaly_detection, get_ad_intervals_fast, top_k_normal_indices, get_top_k_normal_intervals
from .dnn.dnn import get_model_intervals as get_model_intervals_cpu
from .dnn_gpu.dnn import get_model_intervals as get_model_intervals_gpu
from .dnn_para.dnn import get_model_intervals as get_model_intervals_para
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
    top_k_normal_percent: float = 0.3,
    deepsad_encoder=None,
    deepsad_c=None,
    device: str = "auto",
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
    Oc = top_k_normal_indices(X, top_k_percent=top_k_normal_percent, deepsad_encoder=deepsad_encoder, deepsad_c=deepsad_c)
    for i in range(n):
        if known_y[i] == 1 and true_y[i] == 0 and i not in Oc:
            Oc.append(i)

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
            model_dtype = next(deepsad_encoder.parameters()).dtype
            intervals_gpu = [
                (
                    left,
                    right,
                    torch.as_tensor(a_i, dtype=model_dtype, device=model_device),
                    torch.as_tensor(b_i, dtype=model_dtype, device=model_device),
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

    print(f"Length of intervals after DNN processing for seed {seed}: {len(intervals)}")
    print(f"Time after DNN processing for seed {seed}: {time.time() - start} seconds")
    intervals = get_top_k_normal_intervals(
        intervals, top_k_percent=top_k_normal_percent, deepsad_c=deepsad_c
    )
    print(f"Length of intervals after AD processing for seed {seed}: {len(intervals)}")
    print(f"Time after AD processing for seed {seed}: {time.time() - start} seconds")
    final_intervals = []
    known_normals = set(i for i in range(n) if known_y[i] == 1 and true_y[i] == 0)
    for left, right, Ocz in intervals:
        Ocz = set(Ocz)
        Ocz = Ocz.union(known_normals)
        Ocz = [i for i in Ocz]
        Ocz = sorted(Ocz)
        final_intervals.append(
            (
                left / np.sqrt(etajTsigmaetaj[0][0]),
                right / np.sqrt(etajTsigmaetaj[0][0]),
                Ocz,
            )
        )

    cdf = truncated_cdf(
        0, 1, final_intervals, Oc, etajTx[0][0] / np.sqrt(etajTsigmaetaj[0][0])
    )
    if cdf is None:
        print(f"Warning: CDF computation failed for seed {seed}. Skipping this run.")
        return None
    p_value = 2 * min(cdf, 1 - cdf)
    print(f"p-value for seed {seed}: {p_value}")
    return p_value
