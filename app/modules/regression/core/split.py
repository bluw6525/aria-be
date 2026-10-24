"""
Utilitas pemisahan data latih dan data uji (train/test split) dengan RNG NumPy.
"""

from typing import Tuple, Union
import numpy as np


def split_data(
    X: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    test_size: float = 0.3,
    random_state: Union[int, None] = None,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Membagi data matriks fitur X dan target y menjadi data latih (train) dan data uji (test).

    Parameters
    ----------
    X : array-like
        Matriks fitur independen.
    y : array-like
        Vektor target dependen.
    test_size : float, default=0.3
        Proporsi data uji (misal 0.3 = 30% data uji, 70% data latih).
    random_state : int or None, default=None
        Seed pengacakan untuk reprodusibilitas dengan default_rng.
    shuffle : bool, default=True
        Apakah data diacak sebelum dibagi.

    Returns
    -------
    X_train, X_test, y_train, y_test : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples = len(y)

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
