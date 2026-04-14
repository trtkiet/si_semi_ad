import argparse
import os
import logging
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_sad import MLP_Autoencoder, AETrainer, DeepSADTrainer, MLP
from si.util import load_odds_data_for_si, ODDS_DATASET_NAMES


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


def create_dataloader(
    X,
    true_labels,
    semi_labels,
    batch_size=32,
    shuffle=True,
    drop_last=False,
):
    semi_labels_processed = semi_labels.copy()
    semi_labels_processed[semi_labels_processed == -1] = 0

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(true_labels, dtype=torch.long),
        torch.tensor(semi_labels_processed, dtype=torch.long),
        torch.arange(len(X)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(level=logging.INFO)

    if args.dataset_name is None:
        Xt, true_yt, known_yt = gen_data(
            args.mu, args.delta, args.n, args.d, args.anomaly_rate, args.known_label_rate
        )
    else:
        Xt, true_yt, known_yt = load_odds_data_for_si(
            dataset_name=args.dataset_name,
            root=args.data_root,
            random_state=args.seed,
            known_label_rate=args.known_label_rate,
            train=True,
        )
        args.d = Xt.shape[1]

    h_dims = [int(x) for x in args.h_dims.split(",")]

    train_batch_size = min(args.batch_size, len(Xt))
    if train_batch_size < 2:
        raise ValueError(
            "Training requires at least 2 samples to support BatchNorm. "
            f"Got {len(Xt)} samples."
        )

    train_loader = create_dataloader(
        Xt,
        true_yt,
        known_yt,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
    )

    print("=" * 60)
    print("Step 1: Autoencoder Pretraining")
    print("=" * 60)
    ae_net = MLP_Autoencoder(
        x_dim=args.d, h_dims=h_dims, rep_dim=args.rep_dim, bias=True
    )
    ae_trainer = AETrainer(
        lr=args.lr, n_epochs=args.ae_epochs, batch_size=args.batch_size, device=device
    )
    ae_net = ae_trainer.train(train_loader, ae_net)

    print("\n" + "=" * 60)
    print("Step 2: Initialize Deep-SAD Encoder")
    print("=" * 60)
    net = MLP(x_dim=args.d, h_dims=h_dims, rep_dim=args.rep_dim, bias=True)
    net_dict = net.state_dict()
    ae_net_dict = ae_net.state_dict()
    ae_net_dict = {
        k.replace("encoder.", ""): v
        for k, v in ae_net_dict.items()
        if k.startswith("encoder.")
    }
    net_dict.update(ae_net_dict)
    net.load_state_dict(net_dict)
    print("Encoder weights initialized from pretrained autoencoder.")

    print("\n" + "=" * 60)
    print("Step 3: Deep-SAD Training")
    print("=" * 60)
    sad_trainer = DeepSADTrainer(
        c=None,
        eta=args.eta,
        lr=args.lr,
        n_epochs=args.sad_epochs,
        batch_size=args.batch_size,
        device=device,
    )
    net = sad_trainer.train(train_loader, net)

    return net, sad_trainer, device, Xt


def evaluate(args, net, sad_trainer, device):
    if args.dataset_name is None:
        Xt_test, yt_true_test, yt_known_test = gen_data(
            args.mu,
            args.delta,
            args.n_test,
            args.d,
            args.anomaly_rate,
            args.known_label_rate,
        )
    else:
        Xt_test, yt_true_test, yt_known_test = load_odds_data_for_si(
            dataset_name=args.dataset_name,
            root=args.data_root,
            random_state=args.seed + 1,
            known_label_rate=args.known_label_rate,
            train=False,
        )

    test_loader = create_dataloader(
        Xt_test, yt_true_test, yt_known_test, batch_size=args.batch_size, shuffle=False
    )
    test_labels, test_scores = sad_trainer.test(test_loader, net)

    auc = roc_auc_score(test_labels, test_scores)
    print(f"Test AUC: {auc:.4f}")
    return auc


def save_model(args, net, sad_trainer, Xt):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.covariance_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, f"{args.name}_model.pth")
    center_path = os.path.join(args.model_dir, f"{args.name}_c.pth")
    covariance_path = os.path.join(args.covariance_dir, f"{args.name}_cov.npy")

    torch.save(net.state_dict(), model_path)
    torch.save(sad_trainer.c, center_path)

    covariance = np.cov(Xt, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.array([[float(covariance)]], dtype=np.float64)
    np.save(covariance_path, covariance)

    print(f"Model saved to {model_path}")
    print(f"Center saved to {center_path}")
    print(f"Covariance saved to {covariance_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Deep-SAD without domain adaptation"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the saved model (e.g. 'baseline_mu0_d8')",
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--anomaly-rate", type=float, default=0.05)
    parser.add_argument("--known-label-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--h-dims", type=str, default="128, 64, 32", help="Hidden dims, comma-separated"
    )
    parser.add_argument("--rep-dim", type=int, default=8)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--ae-epochs", type=int, default=5)
    parser.add_argument("--sad-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument(
        "--covariance-dir",
        type=str,
        default="covariances",
        help="Directory to save covariance matrices.",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--no-save", action="store_true", help="Skip saving model")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        choices=ODDS_DATASET_NAMES,
        help="ODDS dataset name. If omitted, synthetic data generation is used.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Directory where ODDS .mat files are stored/downloaded.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for ODDS train/test splitting.",
    )
    args = parser.parse_args()

    if args.dataset_name is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    net, sad_trainer, device, Xt = train(args)

    if not args.no_eval:
        evaluate(args, net, sad_trainer, device)

    if not args.no_save:
        save_model(args, net, sad_trainer, Xt)


if __name__ == "__main__":
    main()
