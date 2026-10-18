"""
Utilitas bersama untuk skrip perintah CLI dan benchmarking regresi di direktori commands/.
"""

import numpy as np


def split_data(X, y, test_size=0.3, random_state=None, shuffle=True):
    """
    Membagi data fitur X dan target y menjadi data latih (train) dan data uji (test).

    Parameters
    ----------
    X : array-like
        Matriks fitur independen.
    y : array-like
        Vektor target dependen.
    test_size : float, default=0.3
        Proporsi data uji (misal 0.3 = 30% data uji, 70% data latih).
    random_state : int or None, default=None
        Seed pengacakan untuk reprodusibilitas. Jika None, pengacakan bersifat non-deterministik.
    shuffle : bool, default=True
        Apakah data diacak sebelum dibagi.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
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


# Alias fungsi untuk kompatibilitas
split_train_test = split_data


def predict_model(model, X, fit_intercept=False):
    """
    Menghitung prediksi nilai y berdasarkan objek estimator dan matriks fitur X.

    Parameters
    ----------
    model : Estimator object
        Objek estimator yang memiliki atribut params_.
    X : array-like
        Matriks fitur independen.
    fit_intercept : bool, default=False
        Apakah model menggunakan konstanta (intercept).

    Returns
    -------
    y_pred : np.ndarray
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if fit_intercept:
        X_design = np.column_stack([np.ones(len(X)), X])
    else:
        X_design = X

    return X_design @ model.params_
