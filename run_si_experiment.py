import argparse
import time
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kstest

from si_experiment import run, load_models


def main():
    parser = argparse.ArgumentParser(
        description="Run SI experiment with configurable parameters"
    )
    parser.add_argument(
        "--delta", type=float, default=0.0, help="Delta parameter for data generation"
    )
    parser.add_argument("--n", type=int, default=150, help="Number of instances")
    parser.add_argument("--d", type=int, default=10, help="Feature dimension")
    parser.add_argument(
        "--mu", type=float, default=0.0, help="Mu parameter for data generation"
    )
    parser.add_argument("--anomaly-rate", type=float, default=0.0, help="Anomaly rate")
    parser.add_argument(
        "--known-label-rate", type=float, default=0.2, help="Known label rate"
    )
    parser.add_argument(
        "--top-k-percent",
        type=float,
        default=0.05,
        help="Top k percent for anomaly detection",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=120, help="Number of seeds to run"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepsad",
        help="Model name prefix (e.g. 'deepsad' loads deepsad_model.pth and deepsad_c.pth)",
    )
    parser.add_argument(
        "--h-dims",
        type=str,
        default="128,64,32",
        help="Hidden dims for the model, comma-separated",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=1,
        help="Representation dimension for the model",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save results"
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_dims = [int(x) for x in args.h_dims.split(",")]
    deepsad_encoder, deepsad_c, device = load_models(
        device=device,
        model_dir=args.model_dir,
        model_name=args.model_name,
        d=args.d,
        h_dims=h_dims,
        rep_dim=args.rep_dim,
    )

    p_values = []
    start = time.time()

    for seed in range(args.n_seeds):
        p_value = run(
            seed=seed,
            delta=args.delta,
            n=args.n,
            mu=args.mu,
            d=args.d,
            anomaly_rate=args.anomaly_rate,
            known_label_rate=args.known_label_rate,
            top_k_percent=args.top_k_percent,
            deepsad_encoder=deepsad_encoder,
            deepsad_c=deepsad_c,
            device=device,
        )
        if p_value is None:
            continue
        p_values.append(p_value)

    end = time.time()
    print(f"Total time taken: {end - start} seconds")
    print(f"Avg time per run: {(end - start) / len(p_values)} seconds")

    # Save results
    result_prefix = f"delta_{args.delta}_n_{args.n}"

    with open(
        os.path.join(args.results_dir, f"{result_prefix}_p_values.txt"), "w"
    ) as f:
        for p in p_values:
            f.write(f"{p}\n")

    np.save(
        os.path.join(args.results_dir, f"{result_prefix}_p_values.npy"),
        np.array(p_values),
    )

    # Plot histogram
    plt.hist(p_values)
    plt.title(f"Histogram of p-values from SI (delta={args.delta}, n={args.n})")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.results_dir, f"{result_prefix}_histogram.png"))
    plt.close()

    # KS test
    ks_result = kstest(p_values, "uniform")
    print(f"KS test against uniform distribution: {ks_result}")

    with open(os.path.join(args.results_dir, f"{result_prefix}_ks_test.txt"), "w") as f:
        f.write(f"KS test against uniform distribution: {ks_result}\n")


if __name__ == "__main__":
    import torch

    main()
