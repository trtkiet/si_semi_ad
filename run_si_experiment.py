import argparse
import time
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kstest

from si import load_models
from si import run as run_normal


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
        "--seed",
        type=int,
        default=None,
        help="Single seed to run; if omitted, runs seed range [0, n-seeds)",
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
        default="128,128,128,64,64,64,32,32,32",
        help="Hidden dims for the model, comma-separated",
    )
    parser.add_argument(
        "--rep-dim",
        type=int,
        default=8,
        help="Representation dimension for the model",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "dnn_para"],
        help="Execution device for model and SI-DNN path",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="normal",
        choices=["normal", "oc", "bonferonni", "bonferroni"],
        help="Experiment method: normal (si.run), oc (si.run_oc), bonferonni (si.run_bonfer)",
    )
    args = parser.parse_args()

    if args.method == "normal":
        run_fn = run_normal
    elif args.method == "oc":
        from si.run_oc import run as run_fn
    else:
        from si.run_bonfer import run as run_fn

    os.makedirs(args.results_dir, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "dnn_para":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")

    h_dims = [int(x) for x in args.h_dims.split(",")]
    deepsad_encoder, deepsad_c, _ = load_models(
        device=device,
        model_dir=args.model_dir,
        model_name=args.model_name,
        d=args.d,
        h_dims=h_dims,
        rep_dim=args.rep_dim,
    )

    p_values = []
    times = []

    if args.seed is None:
        seeds = range(args.n_seeds)
    else:
        if args.seed < 0:
            raise ValueError("--seed must be >= 0")
        seeds = [args.seed]

    for seed in seeds:
        start = time.time()
        p_value = run_fn(
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
            device=args.device,
        )
        if p_value is None:
            continue
        p_values.append(p_value)
        times.append(time.time() - start)

    dir_name = f"{args.model_name}_delta_{args.delta}_n_{args.n}_method_{args.method}"
    print(f"Results for {dir_name}:")
    # Create result dir if it doesn't exist
    os.makedirs(os.path.join(args.results_dir, dir_name), exist_ok=True)

    print(f"Total time taken: {sum(times)} seconds")
    print(f"Avg time per run: {sum(times) / len(p_values)} seconds")
    with open(os.path.join(args.results_dir, dir_name, f"timing.txt"), "w") as f:
        for t in times:
            f.write(f"{t}\n")

    # Save results
    result_prefix = f"delta_{args.delta}_n_{args.n}"

    with open(os.path.join(args.results_dir, dir_name, f"p_values.txt"), "w") as f:
        for p in p_values:
            f.write(f"{p}\n")

    # np.save(
    #     os.path.join(args.results_dir, dir_name, f"{result_prefix}_p_values.npy"),
    #     np.array(p_values),
    # )

    # Plot histogram
    plt.hist(p_values)
    plt.title(f"Histogram of p-values from SI (delta={args.delta}, n={args.n})")
    plt.xlabel("p-value")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(args.results_dir, dir_name, f"histogram.png"))
    plt.close()

    # KS test
    ks_result = kstest(p_values, "uniform")
    print(f"KS test against uniform distribution: {ks_result}")

    with open(os.path.join(args.results_dir, dir_name, f"ks_test.txt"), "w") as f:
        f.write(f"KS test against uniform distribution: {ks_result}\n")


if __name__ == "__main__":
    import torch

    main()
