import os
from typing import Tuple

import mpmath as mp
import numpy as np
import torch
from torch import nn

from deep_sad import MLP

mp.dps = 500


def parse_torch_model(model: nn.Module):
    results = []

    for module in model.modules():
        if isinstance(module, nn.Sequential) or module is model:
            continue

        if isinstance(module, nn.LeakyReLU):
            alpha = module.negative_slope
            results.append(("LeakyReLU", alpha))
            continue

        if isinstance(module, nn.BatchNorm1d):
            gamma = (
                module.weight.detach().cpu().numpy()
                if module.affine and module.weight is not None
                else None
            )
            beta = (
                module.bias.detach().cpu().numpy()
                if module.affine and module.bias is not None
                else None
            )
            running_mean = (
                module.running_mean.detach().cpu().numpy()
                if module.track_running_stats and module.running_mean is not None
                else None
            )
            running_var = (
                module.running_var.detach().cpu().numpy()
                if module.track_running_stats and module.running_var is not None
                else None
            )
            eps = module.eps
            results.append(
                ("BatchNorm1d", (gamma, beta, running_mean, running_var, eps))
            )
            continue

        if hasattr(module, "weight") or hasattr(module, "bias"):
            w = (
                module.weight.detach()
                if hasattr(module, "weight") and module.weight is not None
                else None
            )
            b = (
                module.bias.detach()
                if hasattr(module, "bias") and module.bias is not None
                else None
            )

            w = w.cpu().numpy().T if w is not None else None
            b = b.cpu().numpy() if b is not None else None

            results.append((module.__class__.__name__, (w, b)))
            continue

        if not any(p.requires_grad for p in module.parameters()):
            results.append((module.__class__.__name__, None))

    return results


def is_torch_model(model):
    return isinstance(model, nn.Module)


def parse_model(model):
    if is_torch_model(model):
        return parse_torch_model(model)
    raise TypeError(f"Unsupported model type: {type(model)}")


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


def truncated_cdf(mu, sigma, intervals, O, etajTX):
    numerator = 0
    denominator = 0
    for left, right, Oz in intervals:
        if np.array_equal(O, Oz) is False:
            if (etajTX >= left) and (etajTX < right):
                print(f"Different found Oz: {Oz}, O: {O}")
            continue

        denominator = (
            denominator + mp.ncdf((right - mu) / sigma) - mp.ncdf((left - mu) / sigma)
        )
        if etajTX >= right:
            numerator = (
                numerator + mp.ncdf((right - mu) / sigma) - mp.ncdf((left - mu) / sigma)
            )
        if (etajTX >= left) and (etajTX < right):
            numerator = (
                numerator
                + mp.ncdf((etajTX - mu) / sigma)
                - mp.ncdf((left - mu) / sigma)
            )
    if denominator != 0:
        return float(numerator / denominator)
    return None
