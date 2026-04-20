import argparse
import os
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_sad import MLP_Autoencoder, AETrainer, DeepSADTrainer, MLP
from si.util import ODDS_DATASET_NAMES, load_odds_data_for_si
from datasets.odds import ODDSADDataset


def gen_data(
    mu: float,
    delta: float,
    n: int,
    d: int,
    anomaly_rate: float = 0.05,
    known_label_rate: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = mu + np.random.normal(0, 1, (n, d))
    true_labels = np.zeros(n, dtype=int)

    n_anomalies = int(n * anomaly_rate)
    anomaly_idx = np.random.choice(n, n_anomalies, replace=False)
    normal_idx = np.setdiff1d(np.arange(n), anomaly_idx)

    if n_anomalies > 0:
        X[anomaly_idx] += delta
        true_labels[anomaly_idx] = -1

    known_labels = np.full(n, -1)

    n_known_anom = int(n_anomalies * known_label_rate)
    if n_known_anom > 0:
        kn_anom_idx = np.random.choice(anomaly_idx, n_known_anom, replace=False)
        known_labels[kn_anom_idx] = -1

    n_known_norm = int(len(normal_idx) * known_label_rate)
    if n_known_norm > 0:
        kn_norm_idx = np.random.choice(normal_idx, n_known_norm, replace=False)
        known_labels[kn_norm_idx] = 1

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
    odds_dataset: Optional[ODDSADDataset] = None

    if args.dataset_name is None:
        Xt, true_yt, known_yt = gen_data(
            args.mu, args.delta, args.n, args.d, args.anomaly_rate, args.known_label_rate
        )
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
    else:
        odds_dataset = ODDSADDataset(
            root=args.data_root,
            dataset_name=args.dataset_name,
            n_known_outlier_classes=1,
            random_state=args.seed,
        )

        train_set = odds_dataset.train_set
        train_size = len(train_set)
        train_batch_size = min(args.batch_size, train_size)
        if train_batch_size < 2:
            raise ValueError(
                "Training requires at least 2 samples to support BatchNorm. "
                f"Got {train_size} samples."
            )

        train_loader = odds_dataset.train_loader(
            batch_size=train_batch_size,
            shuffle_train=True,
            num_workers=0,
        )

        full_train_set = train_set.dataset
        train_indices = np.asarray(train_set.indices, dtype=np.int64)
        Xt = full_train_set.data[train_indices].detach().cpu().numpy()
        true_yt = full_train_set.targets[train_indices].detach().cpu().numpy().astype(int)
        semi_targets = (
            full_train_set.semi_targets[train_indices].detach().cpu().numpy().astype(int)
        )
        known_yt = np.full(len(semi_targets), -1, dtype=int)
        known_yt[semi_targets == 1] = 0
        known_yt[semi_targets == -1] = 1

        args.d = Xt.shape[1]

    h_dims = [int(x) for x in args.h_dims.split(",")]

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

    return net, sad_trainer, device, Xt, true_yt, known_yt, odds_dataset


def evaluate(args, net, sad_trainer, device, odds_dataset: Optional[ODDSADDataset] = None):
    if args.dataset_name is None or odds_dataset is None:
        Xt_test, yt_true_test, yt_known_test = gen_data(
            args.mu,
            args.delta,
            args.n_test,
            args.d,
            args.anomaly_rate,
            args.known_label_rate,
        )
        test_loader = create_dataloader(
            Xt_test,
            yt_true_test,
            yt_known_test,
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        test_loader = odds_dataset.test_loader(
            batch_size=args.batch_size,
            shuffle_test=False,
            num_workers=0,
        )

    test_labels, test_scores = sad_trainer.test(test_loader, net)

    auc = roc_auc_score(test_labels, test_scores)
    print(f"Test AUC: {auc:.4f}")
    return auc


def save_model(args, net, sad_trainer, Xt, known_yt=None):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.covariance_dir, exist_ok=True)
    
    if args.dataset_name is not None:
        X_reference, y_reference = load_odds_data_for_si(dataset_name=args.dataset_name, 
                                            root=args.data_root, 
                                            random_state=args.seed,
                                            split="reference",
                                            percent_sample_size=1.0)
        
        Xt = np.concatenate([Xt, X_reference], axis=0)
        known_yt = np.concatenate([known_yt, np.full(len(X_reference), 1, dtype=int)], axis=0)

    model_path = os.path.join(args.model_dir, f"{args.name}_model.pth")
    center_path = os.path.join(args.model_dir, f"{args.name}_c.pth")
    covariance_path = os.path.join(args.covariance_dir, f"{args.name}_cov.npy")

    torch.save(net.state_dict(), model_path)
    torch.save(sad_trainer.c, center_path)

    covariance = np.cov(Xt[known_yt != -1], rowvar=False)
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
    parser.add_argument("--known-label-rate", type=float, default=0.2)
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
        choices=ODDS_DATASET_NAMES + (None,),
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

    net, sad_trainer, device, Xt, true_yt, known_yt, odds_dataset = train(args)

    if not args.no_eval:
        evaluate(args, net, sad_trainer, device, odds_dataset)

    if not args.no_save:
        save_model(args, net, sad_trainer, Xt, known_yt)


if __name__ == "__main__":
    main()
