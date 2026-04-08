import os
from typing import Tuple

import mpmath as mp
import numpy as np
import torch
from torch import nn

from deep_sad import MLP

mp.dps = 500

def gen_data(
    mu: float,
    delta: float,
    n: int,
    d: int,
    anomaly_rate: float = 0.05,
    known_label_rate: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = mu + np.random.normal(0, 1, (n, d))
    true_labels = np.zeros(n, dtype=int)

    n_anomalies = int(n * anomaly_rate)
    anomaly_idx = np.random.choice(n, n_anomalies, replace=False)
    normal_idx = np.setdiff1d(np.arange(n), anomaly_idx)

    if n_anomalies > 0:
        directions = np.random.choice([-1, 1], size=(n_anomalies, d))
        X[anomaly_idx] += delta * directions
        true_labels[anomaly_idx] = 1

    known_labels = np.full(n, -1)

    n_known_anom = int(n_anomalies * known_label_rate)
    if n_known_anom > 0:
        kn_anom_idx = np.random.choice(anomaly_idx, n_known_anom, replace=False)
        known_labels[kn_anom_idx] = 1

    n_known_norm = int(len(normal_idx) * known_label_rate)
    if n_known_norm > 0:
        kn_norm_idx = np.random.choice(normal_idx, n_known_norm, replace=False)
        known_labels[kn_norm_idx] = 0

    return X, true_labels, known_labels


def load_models(
    device: str = None,
    model_dir: str = "models",
    model_name: str = "deepsad",
    d: int = 8,
    h_dims: list = None,
    rep_dim: int = 1,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if h_dims is None:
        h_dims = [128, 64, 32]

    deepsad_model = MLP(x_dim=d, h_dims=h_dims, rep_dim=rep_dim, bias=True).to(device)

    deepsad_state = torch.load(
        os.path.join(model_dir, f"{model_name}_model.pth"), map_location=device
    )
    deepsad_c = torch.load(
        os.path.join(model_dir, f"{model_name}_c.pth"), map_location=device
    )

    deepsad_model.load_state_dict(deepsad_state)
    deepsad_model.eval()

    deepsad_c = deepsad_c.detach().cpu().numpy()

    return deepsad_model, deepsad_c, device

def normal_interval_prob(left, right, mu, sigma):
    z_left = (left - mu) / sigma
    z_right = (right - mu) / sigma
    
    tail_left = 0.5 * mp.erfc(z_left / mp.sqrt(2))
    tail_right = 0.5 * mp.erfc(z_right / mp.sqrt(2))
    
    return tail_left - tail_right

def truncated_cdf(mu, sigma, intervals, O, etajTX):
    numerator = 0
    denominator = 0
    for left, right, Oz in intervals:
        # print(f"Processing interval [{left}, {right}] with Oz: {Oz}")
        # print(f"Matching O: {np.array_equal(O, Oz)}")
        if np.array_equal(O, Oz) is False:
            if (etajTX >= left) and (etajTX < right):
                print(f"Different found Oz: {Oz}, O: {O}")
            continue

        denominator = (
            denominator + normal_interval_prob(left, right, mu, sigma)
        )
        if etajTX >= right:
            numerator = (
                numerator + normal_interval_prob(left, right, mu, sigma)
            )
        if (etajTX >= left) and (etajTX < right):
            numerator = (
                numerator
                + normal_interval_prob(left, etajTX, mu, sigma)
            )
        # print(f"Interval [{left}, {right}]: numerator={numerator}, denominator={denominator}")
        # print(f"Value of truncated CDF for this interval: {normal_interval_prob(left, right, mu, sigma)}")
    if denominator != 0:
        return float(numerator / denominator)
    return None

def compute_etajTsigmaetaj_a_b(etaj, etajTx, X, n, d, S=None):
    if S is None:
        S = np.eye(d)                          # special case: sigma = I_{n*d}
    etaj_blocks = etaj.reshape(n, d)           # (n, d)
    S_etaj = etaj_blocks @ S.T                 # (n, d)
    etajTsigmaetaj = np.einsum('ij,ij->', etaj_blocks, S_etaj).reshape(1, 1)
    
    
    S_etaj_flat = S_etaj.reshape(-1, 1)                                     # (n*d, 1)
    b = S_etaj_flat / etajTsigmaetaj                                        # (n*d, 1)

    X_flat = X.reshape(-1, 1)
    a = X_flat - b * etajTx    
    return etajTsigmaetaj, a, b