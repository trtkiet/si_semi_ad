from torch.utils.data import DataLoader, Subset
from datasets.base_dataset import BaseADDataset
from datasets.odds_dataset import ODDSDataset

import torch
import numpy as np
from typing import Tuple

def create_semisupervised_setting(labels, normal_classes, outlier_classes, known_outlier_classes,
                                  ratio_known_normal, ratio_known_outlier, ratio_pollution):
    """
    Create a semi-supervised data setting. 
    :param labels: np.array with labels of all dataset samples
    :param normal_classes: tuple with normal class labels
    :param outlier_classes: tuple with anomaly class labels
    :param known_outlier_classes: tuple with known (labeled) anomaly class labels
    :param ratio_known_normal: the desired ratio of known (labeled) normal samples
    :param ratio_known_outlier: the desired ratio of known (labeled) anomalous samples
    :param ratio_pollution: the desired pollution ratio of the unlabeled data with unknown (unlabeled) anomalies.
    :return: tuple with list of sample indices, list of original labels, and list of semi-supervised labels
    """
    idx_normal = np.argwhere(np.isin(labels, normal_classes)).flatten()
    idx_outlier = np.argwhere(np.isin(labels, outlier_classes)).flatten()
    idx_known_outlier_candidates = np.argwhere(np.isin(labels, known_outlier_classes)).flatten()
    
    # print(f"Total samples: {len(labels)}, Normal samples: {len(idx_normal)}, Outlier samples: {len(idx_outlier)}, Known outlier candidates: {len(idx_known_outlier_candidates)}")

    n_normal = len(idx_normal)

    # Solve system of linear equations to obtain respective number of samples
    x = np.array([
        ratio_known_normal * n_normal,  # known normal
        n_normal - ratio_known_normal * n_normal,
        (1 - ratio_known_outlier) * len(idx_outlier),  # unlabeled outlier
        ratio_known_outlier * len(idx_outlier)  # known outlier
    ])

    # Get number of samples
    n_known_normal = int(x[0])
    n_unlabeled_normal = len(idx_normal) - n_known_normal
    n_unlabeled_outlier = int(x[2])
    n_known_outlier = int(x[3])
    
    # print(f"Calculated sample sizes - Known normal: {n_known_normal}, Unlabeled normal: {n_unlabeled_normal}, Unlabeled outlier: {n_unlabeled_outlier}, Known outlier: {n_known_outlier}")

    # Sample indices
    perm_normal = np.random.permutation(n_normal)
    perm_outlier = np.random.permutation(len(idx_outlier))
    perm_known_outlier = np.random.permutation(len(idx_known_outlier_candidates))

    idx_known_normal = idx_normal[perm_normal[:n_known_normal]].tolist()
    idx_unlabeled_normal = idx_normal[perm_normal[n_known_normal:n_known_normal+n_unlabeled_normal]].tolist()
    idx_unlabeled_outlier = idx_outlier[perm_outlier[:n_unlabeled_outlier]].tolist()
    idx_known_outlier = idx_known_outlier_candidates[perm_known_outlier[:n_known_outlier]].tolist()

    # Get original class labels
    labels_known_normal = labels[idx_known_normal].tolist()
    labels_unlabeled_normal = labels[idx_unlabeled_normal].tolist()
    labels_unlabeled_outlier = labels[idx_unlabeled_outlier].tolist()
    labels_known_outlier = labels[idx_known_outlier].tolist()

    # Get semi-supervised setting labels
    semi_labels_known_normal = np.ones(n_known_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_normal = np.zeros(n_unlabeled_normal).astype(np.int32).tolist()
    semi_labels_unlabeled_outlier = np.zeros(n_unlabeled_outlier).astype(np.int32).tolist()
    semi_labels_known_outlier = (-np.ones(n_known_outlier).astype(np.int32)).tolist()

    # Create final lists
    list_idx = idx_known_normal + idx_unlabeled_normal + idx_unlabeled_outlier + idx_known_outlier
    list_labels = labels_known_normal + labels_unlabeled_normal + labels_unlabeled_outlier + labels_known_outlier
    list_semi_labels = (semi_labels_known_normal + semi_labels_unlabeled_normal + semi_labels_unlabeled_outlier
                        + semi_labels_known_outlier)

    return list_idx, list_labels, list_semi_labels


class ODDSADDataset(BaseADDataset):

    def __init__(self, root: str, dataset_name: str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.4,
                 ratio_known_outlier: float = 0.05, ratio_pollution: float = 0.1, random_state=None):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = (1,)

        # Get train set
        train_set = ODDSDataset(
            root=self.root,
            dataset_name=dataset_name,
            split='train',
            random_state=random_state,
            download=True,
            known_normal_ratio=ratio_known_normal,
        )

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = ODDSDataset(
            root=self.root,
            dataset_name=dataset_name,
            split='test',
            random_state=random_state,
            known_normal_ratio=ratio_known_normal,
        )

        # Get reference set (20% of known normal sampled before train/test split)
        self.reference_set = ODDSDataset(
            root=self.root,
            dataset_name=dataset_name,
            split='reference',
            random_state=random_state,
            known_normal_ratio=ratio_known_normal,
        )
        
        # idx, _, semi_targets = create_semisupervised_setting(self.test_set.targets.cpu().data.numpy(), self.normal_classes,
        #                                                      self.outlier_classes, self.known_outlier_classes,
        #                                                         ratio_known_normal, ratio_known_outlier, ratio_pollution)
        # self.test_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
    
    def train_loader(self, batch_size: int, shuffle_train=True, num_workers: int = 0) -> DataLoader:
        return DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                          num_workers=num_workers, drop_last=True)

    def test_loader(self, batch_size: int, shuffle_test=False, num_workers: int = 0) -> DataLoader:
        return DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                          num_workers=num_workers, drop_last=False)

    def reference_loader(self, batch_size: int, shuffle_reference=False, num_workers: int = 0) -> DataLoader:
        return DataLoader(dataset=self.reference_set, batch_size=batch_size, shuffle=shuffle_reference,
                          num_workers=num_workers, drop_last=False)

    def get_train_set(self):
        X = self.train_set.dataset.data.cpu().data.numpy()
        y = self.train_set.dataset.targets.cpu().data.numpy()
        semi_y = self.train_set.dataset.semi_targets.cpu().data.numpy()
        return X, y, semi_y
    
    def get_test_set(self):
        X = self.test_set.data.cpu().data.numpy()
        y = self.test_set.targets.cpu().data.numpy()
        # Replace 1 with -1, and 0 with 1 to match label
        mask_outliers = (y == 1)
        y[mask_outliers] = -1
        y[~mask_outliers] = 1
        return X, y
    
    def get_reference_set(self):
        X = self.reference_set.data.cpu().data.numpy()
        y = self.reference_set.targets.cpu().data.numpy()
        
        mask_outliers = (y == 1)
        y[mask_outliers] = -1
        y[~mask_outliers] = 1
        return X, y
    
    def get_split_set(self, split: str):
        if split == 'train':
            return self.get_train_set()
        elif split == 'test':
            return self.get_test_set()
        elif split == 'reference':
            return self.get_reference_set()