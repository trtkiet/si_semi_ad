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

def top_k_normal_indices(X: np.ndarray, top_k_percent: float, deepsad_encoder, deepsad_c) -> np.ndarray:
    model_device = next(deepsad_encoder.parameters()).device
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32, device=model_device)
        x_transformed = deepsad_encoder(x_tensor).cpu().numpy()

    distances = np.linalg.norm(x_transformed - deepsad_c, axis=1)
    top_k = int(top_k_percent * len(distances))
    normal_indices = np.argpartition(distances, top_k)[:top_k]
    return sorted(normal_indices)


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

import heapq
import math
import numpy as np


def get_ad_intervals_fast(intervals, top_k_percent, deepsad_c):
    quad_intervals = []
    for left, right, a, b in intervals:
        u = a - deepsad_c
        A = np.sum(b * b, axis=1)
        B = 2 * np.sum(u * b, axis=1)
        C = np.sum(u * u, axis=1)
        quad_intervals.append((left, right, A, B, C))

    new_intervals = []

    for left, right, A, B, C in quad_intervals:
        n = len(A)
        top_k = max(1, int(top_k_percent * n))
        boundary = n - top_k  # items at rank >= boundary are in top-k

        def find_next_crossing(i, j, z_min):
            """
            First z in (z_min, right] where d_i crosses from below d_j to above.
            Precondition: d_i(z_min) <= d_j(z_min).
            Each quadratic difference has at most 2 roots, checked in O(1).
            """
            da = float(A[i] - A[j])
            db = float(B[i] - B[j])
            dc = float(C[i] - C[j])
            eps = 1e-10

            if abs(da) < 1e-16:
                roots = [-dc / db] if abs(db) > 1e-16 else []
            else:
                disc = db * db - 4.0 * da * dc
                if disc <= eps:
                    return None
                sq = math.sqrt(max(0.0, disc))
                roots = [(-db - sq) / (2 * da), (-db + sq) / (2 * da)]

            for z in sorted(roots):
                if z <= z_min + eps or z > right + 1e-9:
                    continue
                # Confirm direction: d_i - d_j goes negative → positive here
                z_t = z + eps
                if da * z_t * z_t + db * z_t + dc > 0.0:
                    return z
            return None

        # --- Initial sort at z = left ---
        dist0 = A * left * left + B * left + C
        sorted_order = list(np.argsort(dist0))   # sorted_order[r] = item at rank r (asc. distance)
        rank = np.empty(n, dtype=np.intp)
        for r, item in enumerate(sorted_order):
            rank[item] = r

        # --- Seed heap with crossing times for each adjacent pair ---
        # O(n log n) — only n-1 pairs, not O(n²)
        heap = []  # (z_crossing, i, j)  where rank[i]+1 == rank[j] at insertion time
        for r in range(n - 1):
            i, j = sorted_order[r], sorted_order[r + 1]
            z_cross = find_next_crossing(i, j, left - 1e-9)
            if z_cross is not None:
                heapq.heappush(heap, (z_cross, i, j))

        previous = left
        current_topk = set(sorted_order[boundary:])

        # --- Kinetic sweep ---
        while heap:
            z_val, i, j = heapq.heappop(heap)
            if z_val > right + 1e-9:
                break

            if rank[i] + 1 != rank[j]:
                continue

            r = rank[i]

            if r == boundary - 1:
                # i is crossing the boundary with j
                # before swap: sorted_order[r]=i, sorted_order[r+1]=j
                outgoing = sorted_order[boundary]
                incoming = sorted_order[boundary - 1]
                
                new_intervals.append((previous, z_val, current_topk))
                previous = z_val
                current_topk = (current_topk - {outgoing}) | {incoming}

            sorted_order[r], sorted_order[r + 1] = j, i
            rank[j] = r
            rank[i] = r + 1

            for r2 in (r - 1, r, r + 1):
                if 0 <= r2 < n - 1:
                    ii, jj = sorted_order[r2], sorted_order[r2 + 1]
                    z_cross = find_next_crossing(ii, jj, z_val)
                    if z_cross is not None and z_cross < right + 1e-9:
                        heapq.heappush(heap, (z_cross, ii, jj))

        new_intervals.append((previous, right, current_topk))

    return new_intervals

def get_top_k_normal_intervals(intervals, top_k_percent, deepsad_c):
    quad_intervals = []
    for left, right, a, b in intervals:
        u = a - deepsad_c
        A = np.sum(b * b, axis=1)
        B = 2 * np.sum(u * b, axis=1)
        C = np.sum(u * u, axis=1)
        quad_intervals.append((left, right, A, B, C))

    new_intervals = []

    for left, right, A, B, C in quad_intervals:
        n = len(A)
        top_k = max(1, int(top_k_percent * n))
        boundary = top_k  # items at rank < boundary are in top-k normal

        def find_next_crossing(i, j, z_min):
            """
            First z in (z_min, right] where d_i crosses from below d_j to above.
            Precondition: d_i(z_min) <= d_j(z_min).
            Each quadratic difference has at most 2 roots, checked in O(1).
            """
            da = float(A[i] - A[j])
            db = float(B[i] - B[j])
            dc = float(C[i] - C[j])
            eps = 1e-10

            if abs(da) < 1e-16:
                roots = [-dc / db] if abs(db) > 1e-16 else []
            else:
                disc = db * db - 4.0 * da * dc
                if disc <= eps:
                    return None
                sq = math.sqrt(max(0.0, disc))
                roots = [(-db - sq) / (2 * da), (-db + sq) / (2 * da)]

            for z in sorted(roots):
                if z <= z_min + eps or z > right + 1e-9:
                    continue
                # Confirm direction: d_i - d_j goes negative → positive here
                z_t = z + eps
                if da * z_t * z_t + db * z_t + dc > 0.0:
                    return z
            return None

        # --- Initial sort at z = left ---
        dist0 = A * left * left + B * left + C
        sorted_order = list(np.argsort(dist0))   # sorted_order[r] = item at rank r (asc. distance)
        rank = np.empty(n, dtype=np.intp)
        for r, item in enumerate(sorted_order):
            rank[item] = r

        # --- Seed heap with crossing times for each adjacent pair ---
        # O(n log n) — only n-1 pairs, not O(n²)
        heap = []  # (z_crossing, i, j)  where rank[i]+1 == rank[j] at insertion time
        for r in range(n - 1):
            i, j = sorted_order[r], sorted_order[r + 1]
            z_cross = find_next_crossing(i, j, left - 1e-9)
            if z_cross is not None:
                heapq.heappush(heap, (z_cross, i, j))

        previous = left
        current_topk = set(sorted_order[:boundary])

        # --- Kinetic sweep ---
        while heap:
            z_val, i, j = heapq.heappop(heap)
            if z_val > right + 1e-9:
                break

            if rank[i] + 1 != rank[j]:
                continue

            r = rank[i]

            if r == boundary - 1:                      # i is last in top-k, j is first outside
                outgoing = sorted_order[boundary - 1]  # = i, leaving top-k
                incoming = sorted_order[boundary]      # = j, entering top-k

                new_intervals.append((previous, z_val, current_topk))
                previous = z_val
                current_topk = (current_topk - {outgoing}) | {incoming}

            sorted_order[r], sorted_order[r + 1] = j, i
            rank[j] = r
            rank[i] = r + 1

            for r2 in (r - 1, r, r + 1):
                if 0 <= r2 < n - 1:
                    ii, jj = sorted_order[r2], sorted_order[r2 + 1]
                    z_cross = find_next_crossing(ii, jj, z_val)
                    if z_cross is not None and z_cross < right + 1e-9:
                        heapq.heappush(heap, (z_cross, ii, jj))

        new_intervals.append((previous, right, current_topk))

    return new_intervals