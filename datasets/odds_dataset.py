from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import os
import torch
import numpy as np
import urllib.request

CATEGORICAL_UNIQUE_THRESHOLD = {
    "arrhythmia": 250,
    "satimage-2": 0,
    "thyroid": 0,
    "annthyroid": 0,
    "creditcard": 0
}

class ODDSDataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """

    urls = {
        'arrhythmia': 'https://www.dropbox.com/s/lmlwuspn1sey48r/arrhythmia.mat?dl=1',
        'cardio': 'https://www.dropbox.com/s/galg3ihvxklf0qi/cardio.mat?dl=1',
        'satellite': 'https://www.dropbox.com/s/dpzxp8jyr9h93k5/satellite.mat?dl=1',
        'satimage-2': 'https://www.dropbox.com/s/hckgvu9m6fs441p/satimage-2.mat?dl=1',
        'shuttle': 'https://www.dropbox.com/s/mk8ozgisimfn3dw/shuttle.mat?dl=1',
        'thyroid': 'https://www.dropbox.com/s/bih0e15a0fukftb/thyroid.mat?dl=1',
        'musk': 'https://www.dropbox.com/scl/fi/o9nk6gv7pgeouop/musk.mat?rlkey=cjp9trvd1sulnt5u1apg2j46p&e=1&dl=1',
        'annthyroid': 'https://www.dropbox.com/scl/fi/a0sgvfst8gwvty8/annthyroid.mat?rlkey=t8z9z7jn06a9vsp1muhylteqz&e=1&dl=1',
        'vowels': 'https://www.dropbox.com/scl/fi/ikz2if1yyz8zbly/vowels.mat?rlkey=8z5ce4g2frcx22xqqko7d7vvk&e=1&dl=1'
    }

    # Integer-valued features with low cardinality are treated as categorical and removed.
    

    def __init__(self, root: str, dataset_name: str, train=True, split: str = None,
                 random_state=None, download=False, reference_ratio: float = 0.3, test_ratio: float = 0.4,
                 known_normal_ratio: float = 0.7):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # backward-compatible flag
        self.split = split if split is not None else ('train' if train else 'test')
        self.file_name = self.dataset_name + '.mat'
        self.data_file = self.root / self.file_name

        if download:
            self.download()

        mat = loadmat(self.data_file)
        X = mat['X']
        X = self._keep_numerical_only_features(X, dataset_name=self.dataset_name)
        y = mat['y'].ravel().astype(np.int64)

        idx_train, idx_test, idx_reference = self._split_indices(
            y=y,
            known_normal_ratio=known_normal_ratio,
            reference_ratio=reference_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

        X_train = X[idx_train]
        y_train = y[idx_train]
        X_test = X[idx_test]
        y_test = y[idx_test]
        X_reference = X[idx_reference]
        y_reference = y[idx_reference]
        
        # robust_scaler = RobustScaler().fit(X_train)
        # X_train = robust_scaler.transform(X_train)
        # X_test = robust_scaler.transform(X_test)
        # X_reference = robust_scaler.transform(X_reference)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)
        X_reference_stand = scaler.transform(X_reference)

        # Scale to range [0,1]
        # minmax_scaler = MinMaxScaler().fit(X_train_stand)
        # X_train_scaled = minmax_scaler.transform(X_train_stand)
        # X_test_scaled = minmax_scaler.transform(X_test_stand)
        
        X_train_scaled = X_train_stand
        X_test_scaled = X_test_stand
        X_reference_scaled = X_reference_stand

        if self.split == 'train':
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        elif self.split == 'test':
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)
        elif self.split == 'reference':
            self.data = torch.tensor(X_reference_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_reference, dtype=torch.int64)
        else:
            raise ValueError(f"Unknown split '{self.split}'. Expected 'train', 'test', or 'reference'.")

        self.semi_targets = torch.zeros_like(self.targets)
        if self.split == 'reference':
            self.semi_targets = torch.ones_like(self.targets)

    @staticmethod
    def _split_indices(y: np.ndarray, known_normal_ratio: float, reference_ratio: float, test_ratio: float, random_state=None):
        """Create reference split from known normal first, then split remaining data into train/test."""
        if not (0.0 <= reference_ratio < 1.0):
            raise ValueError(f"reference_ratio must be in [0, 1), got {reference_ratio}.")
        if not (0.0 < test_ratio < 1.0):
            raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}.")

        y = np.asarray(y).ravel()
        idx_all = np.arange(y.shape[0])
        idx_normal = np.flatnonzero(y == 0)

        if idx_normal.size == 0:
            raise ValueError("No known normal samples (label 0) found for reference split.")

        split_random_state = 0 if random_state is None else random_state
        rng = np.random.RandomState(split_random_state)

        n_reference = int(np.floor(known_normal_ratio * reference_ratio * idx_normal.size))
        if n_reference > 0:
            idx_reference = np.sort(rng.choice(idx_normal, size=n_reference, replace=False))
        else:
            idx_reference = np.array([], dtype=np.int64)

        keep_mask = np.ones(y.shape[0], dtype=bool)
        keep_mask[idx_reference] = False
        idx_remaining = idx_all[keep_mask]
        y_remaining = y[idx_remaining]

        unique_labels, label_counts = np.unique(y_remaining, return_counts=True)
        stratify_labels = y_remaining if (unique_labels.size > 1 and np.all(label_counts >= 2)) else None

        idx_train, idx_test = train_test_split(
            idx_remaining,
            test_size=test_ratio,
            random_state=split_random_state,
            stratify=stratify_labels,
        )

        return idx_train, idx_test, idx_reference

    @classmethod
    def _keep_numerical_only_features(self, X, dataset_name):
        mask = np.array([
            np.unique(X[:, i]).size > CATEGORICAL_UNIQUE_THRESHOLD[dataset_name]
            for i in range(X.shape[1])
        ])
        return X[:, mask]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)

    def download(self):
        """Download the ODDS dataset if it doesn't exist in root already."""

        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        base_url = self.urls[self.dataset_name]
        fallback_url = base_url
        if "dropbox.com" in base_url:
            fallback_url = (
                base_url.replace("https://www.dropbox.com", "https://dl.dropboxusercontent.com")
                .replace("?dl=1", "")
            )

        last_error = None
        for url in [base_url, fallback_url]:
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/123.0.0.0 Safari/537.36"
                        )
                    },
                )
                with urllib.request.urlopen(request) as response, open(
                    self.data_file, "wb"
                ) as out_file:
                    out_file.write(response.read())
                print("Done!")
                return
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to download dataset '{self.dataset_name}' from all known URLs. "
            f"Last error: {last_error}"
        )

        print('Done!')
