import os
from typing import Optional, Tuple

import mpmath as mp
import numpy as np
import torch
from torch import nn
from datasets.odds_dataset import ODDSDataset

from deep_sad import MLP

mp.dps = 500

ODDS_DATASET_NAMES = (
    "arrhythmia",
    "cardio",
    "satellite",
    "satimage-2",
    "shuttle",
    "thyroid",
)


def _sample_known_labels(
    true_labels: np.ndarray,
    known_label_rate: float,
    random_state: int,
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    known_labels = np.full(len(true_labels), -1, dtype=int)

    anomaly_idx = np.where(true_labels == 1)[0]
    normal_idx = np.where(true_labels == 0)[0]

    n_known_anom = int(len(anomaly_idx) * known_label_rate)
    if len(anomaly_idx) > 0:
        n_known_anom = max(1, n_known_anom)
        n_known_anom = min(n_known_anom, len(anomaly_idx))
    if n_known_anom > 0:
        chosen = rng.choice(anomaly_idx, n_known_anom, replace=False)
        known_labels[chosen] = 1

    n_known_norm = int(len(normal_idx) * known_label_rate)
    if len(normal_idx) > 0:
        n_known_norm = max(1, n_known_norm)
        n_known_norm = min(n_known_norm, len(normal_idx))
    if n_known_norm > 0:
        chosen = rng.choice(normal_idx, n_known_norm, replace=False)
        known_labels[chosen] = 0

    return known_labels

def load_known_normal_data(
    dataset_name: str,
    root: str,
    random_state: int,
):
    dataset = ODDSDataset(
        root=root,
        dataset_name=dataset_name,
        train=True,
        random_state=random_state,
        download=True,
    )

    X = dataset.data.detach().cpu().numpy()
    labels = dataset.targets.detach().cpu().numpy().astype(int)

    idx_normal = np.where(labels == 0)[0]
    perm_normal = np.random.permutation(len(idx_normal))

    idx_known_normal = idx_normal[perm_normal].tolist()

    return X[idx_known_normal]

def load_odds_data_for_si(
    dataset_name: str,
    root: str,
    random_state: int,
    known_label_rate: float = 0.1,
    train: bool = True,
    test_sample_size: Optional[int] = None,
    percent_test_sample_size: Optional[float] = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset_name not in ODDS_DATASET_NAMES:
        raise ValueError(
            f"Unsupported ODDS dataset '{dataset_name}'. "
            f"Choose from: {', '.join(ODDS_DATASET_NAMES)}"
        )

    dataset = ODDSDataset(
        root=root,
        dataset_name=dataset_name,
        train=train,
        random_state=random_state,
        download=True,
    )

    X = dataset.data.detach().cpu().numpy()
    true_y = dataset.targets.detach().cpu().numpy().astype(int)
    if test_sample_size is None and percent_test_sample_size is not None:
        test_sample_size = int(len(true_y) * percent_test_sample_size)
        test_sample_size = max(200, test_sample_size)  # Ensure at least 150 samples
    if (not train) and (test_sample_size is not None):
        if test_sample_size <= 0:
            raise ValueError("test_sample_size must be > 0")
        if test_sample_size < len(true_y):
            rng = np.random.RandomState(random_state)
            selected_idx = rng.choice(len(true_y), size=test_sample_size, replace=False)
            X = X[selected_idx]
            true_y = true_y[selected_idx]

    if train:
        known_y = _sample_known_labels(true_y, known_label_rate, random_state)
    else:
        # Test split is treated as fully unlabeled.
        known_y = np.full(len(true_y), -1, dtype=int)

    return X, true_y, known_y

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
    # print(f"Computing truncated CDF with etajTX: {etajTX}, mu: {mu}, sigma: {sigma}")
    for left, right, Oz in intervals:
        # print(f"Processing interval [{left}, {right}] with Oz: {Oz}")
        # print(f"Matching O: {np.array_equal(O, Oz)}")
        if O != Oz:
            if (etajTX >= left) and (etajTX < right):
                print(f"Different found Oz: {Oz}, O: {O}")
                return None
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