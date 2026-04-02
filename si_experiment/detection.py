import numpy as np
import torch


def anomaly_detection(
    X: np.ndarray, top_k_percent: float, deepsad_encoder, deepsad_c
) -> np.ndarray:
    model_device = next(deepsad_encoder.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=model_device)
        x_transformed = deepsad_encoder(x_tensor).cpu().numpy()

    distances = np.linalg.norm(x_transformed - deepsad_c, axis=1)
    top_k = int(top_k_percent * len(distances))
    anomalies_index = np.argpartition(distances, -top_k)[-top_k:]
    return sorted(anomalies_index)


def get_ad_intervals(intervals, top_k_percent, deepsad_c):
    quad_intervals = []
    for left, right, a, b in intervals:
        u = a - deepsad_c
        A = np.sum(b * b, axis=1)
        B = 2 * np.sum(u * b, axis=1)
        C = np.sum(u * u, axis=1)
        quad_intervals.append((left, right, A, B, C))

    new_intervals = []
    n_pairs_idx = None

    for left, right, A, B, C in quad_intervals:
        n = len(A)

        if n_pairs_idx is None or n_pairs_idx[0].shape[0] != n * (n - 1) // 2:
            i_idx, j_idx = np.triu_indices(n, k=1)

        a_p = A[i_idx] - A[j_idx]
        b_p = B[i_idx] - B[j_idx]
        c_p = C[i_idx] - C[j_idx]

        lo, hi = left - 1e-5, right + 1e-5

        lin_mask = np.abs(a_p) < 1e-16
        safe_b = np.where(lin_mask & (np.abs(b_p) > 1e-16), b_p, 1.0)
        z_lin = np.where(lin_mask & (np.abs(b_p) > 1e-16), -c_p / safe_b, np.inf)

        disc = b_p**2 - 4 * a_p * c_p
        quad_mask = (~lin_mask) & (disc > 1e-16)
        safe_denom = np.where(quad_mask, 2 * a_p, 1.0)
        sqrt_disc = np.where(quad_mask, np.sqrt(np.maximum(0.0, disc)), 0.0)
        z1 = np.where(quad_mask, (-b_p + sqrt_disc) / safe_denom, np.inf)
        z2 = np.where(quad_mask, (-b_p - sqrt_disc) / safe_denom, np.inf)

        z_values = []
        for mask, zs in [
            (lin_mask & (np.abs(b_p) > 1e-16) & (z_lin >= lo) & (z_lin <= hi), z_lin),
            (quad_mask & (z1 >= lo) & (z1 <= hi), z1),
            (quad_mask & (z2 >= lo) & (z2 <= hi), z2),
        ]:
            hits = np.where(mask)[0]
            z_values.extend(zip(zs[hits], i_idx[hits], j_idx[hits]))

        z0 = left
        distances = A * z0**2 + B * z0 + C
        indices = np.argsort(distances)
        top_k = max(1, int(top_k_percent * n))

        position = np.empty(n, dtype=np.intp)
        position[indices] = np.arange(n)

        if not z_values:
            new_intervals.append((left, right, sorted(indices[-top_k:])))
            continue

        z_values.sort(key=lambda x: x[0])
        previous = left

        for z_val, i, j in z_values:
            new_intervals.append((previous, z_val, sorted(indices[-top_k:])))
            previous = z_val

            idx_i = position[i]
            idx_j = position[j]

            # if abs(idx_i - idx_j) > 1:
            #     print(f"Warning: swapping non-adjacent {i} and {j} at z={z_val:.4f}")

            indices[idx_i], indices[idx_j] = j, i
            position[i], position[j] = idx_j, idx_i

        new_intervals.append((previous, right, sorted(indices[-top_k:])))

    return new_intervals
