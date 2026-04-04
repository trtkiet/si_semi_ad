import numpy as np

from .util import parse_model


def get_model_intervals(model, intervals, eps=1e-9):
    layers = parse_model(model)

    for layer_type, params in layers:
        # print(f"Processing layer: {layer_type}")
        if layer_type == "Linear":
            W, bias = params
            new_intervals = []
            for left, right, a_curr, b_curr in intervals:
                a_new = a_curr @ W
                if bias is not None:
                    a_new = a_new + bias
                b_new = b_curr @ W
                new_intervals.append((left, right, a_new, b_new))
            intervals = new_intervals

        elif layer_type == "ReLU":
            new_intervals = []

            num_intervals = len(intervals)
            if num_intervals == 0:
                continue

            lefts = np.array([left for left, _, _, _ in intervals])
            rights = np.array([right for _, right, _, _ in intervals])
            A_curr = np.array([a_curr for _, _, a_curr, _ in intervals])
            B_curr = np.array([b_curr for _, _, _, b_curr in intervals])

            with np.errstate(divide="ignore", invalid="ignore"):
                z_star_matrix = np.where(
                    np.abs(B_curr) >= 1e-12, -A_curr / B_curr, np.inf
                )

            in_range_matrix = (z_star_matrix > lefts[:, np.newaxis, np.newaxis]) & (
                z_star_matrix < rights[:, np.newaxis, np.newaxis]
            )

            new_intervals = []
            for idx in range(num_intervals):
                left = lefts[idx]
                right = rights[idx]
                a_curr = A_curr[idx]
                b_curr = B_curr[idx]

                interior_splits = z_star_matrix[idx][in_range_matrix[idx]]

                if interior_splits.size > 0:
                    splits = np.unique(np.concatenate([[left, right], interior_splits]))
                else:
                    splits = np.array([left, right])

                num_splits = len(splits) - 1
                sub_lefts = splits[:-1]
                sub_rights = splits[1:]
                z_mids = (sub_lefts + sub_rights) * 0.5

                pre_activations = a_curr + b_curr * z_mids[:, np.newaxis, np.newaxis]
                active_masks = (pre_activations > 0).astype(np.float64)

                a_news = a_curr * active_masks
                b_news = b_curr * active_masks

                for i in range(num_splits):
                    new_intervals.append(
                        (sub_lefts[i], sub_rights[i], a_news[i], b_news[i])
                    )

            intervals = new_intervals

        elif layer_type == "LeakyReLU":
            alpha = params
            new_intervals = []

            for left, right, a_curr, b_curr in intervals:
                with np.errstate(divide="ignore", invalid="ignore"):
                    z_stars = np.where(
                        np.abs(b_curr) >= 1e-12, -a_curr / b_curr, np.inf
                    )

                in_range = (z_stars > left) & (z_stars < right)
                interior_splits = z_stars[in_range]

                if interior_splits.size > 0:
                    splits = np.unique(np.concatenate([[left, right], interior_splits]))
                else:
                    splits = np.array([left, right])

                for i in range(len(splits) - 1):
                    sub_left = splits[i]
                    sub_right = splits[i + 1]
                    z_mid = (sub_left + sub_right) * 0.5

                    pre_activation = a_curr + b_curr * z_mid
                    active_mask = np.where(pre_activation > 0, 1.0, 0.0)
                    inactive_mask = 1.0 - active_mask

                    a_new = a_curr * active_mask + alpha * a_curr * inactive_mask
                    b_new = b_curr * active_mask + alpha * b_curr * inactive_mask

                    new_intervals.append((sub_left, sub_right, a_new, b_new))

            intervals = new_intervals
        elif layer_type == "BatchNorm1d":
            gamma, beta, running_mean, running_var, bn_eps = params
            new_intervals = []
            
            if running_var is None or running_mean is None:
                print("Warning: BatchNorm1d layer is missing running statistics. Skipping normalization.")
            # Precompute the effective scale and shift
            # scale[i] = gamma[i] / sqrt(var[i] + eps)   (or 1.0 if gamma is None)
            # shift[i] = beta[i] - mean[i] * scale[i]    (or 0.0 if beta/mean are None)
            scale = np.ones_like(running_var) if running_var is None else 1.0 / np.sqrt(running_var + bn_eps)
            if gamma is not None:
                scale = gamma * scale

            shift = np.zeros_like(scale) if running_mean is None else -running_mean * scale
            if beta is not None:
                shift = shift + beta

            for left, right, a_curr, b_curr in intervals:
                # BN is element-wise: output_i = scale_i * input_i + shift_i
                # a_new = scale * a_curr + shift  (bias term absorbs shift)
                # b_new = scale * b_curr          (slope is unaffected by shift)
                a_new = scale * a_curr + shift
                b_new = scale * b_curr
                new_intervals.append((left, right, a_new, b_new))
            
            intervals = new_intervals

    intervals = sorted(intervals, key=lambda x: x[0])
    return intervals
