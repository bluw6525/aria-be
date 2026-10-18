"""
Modul Deteksi Pencilan (Outlier) dan Titik Pengaruh (Influential Points):
1. Leverage Values (Hii) dari Hat Matrix dengan ambang batas 2p/n.
2. DFFITS (Difference in Fits) dengan ambang batas 2*sqrt(p/n).
3. Studentized Deleted Residuals (t_i)
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


from .algorithms.ols_estimator import OLSEstimator


def calculate_leverage(
    X: np.ndarray,
    fit_intercept: bool = True
) -> Tuple[np.ndarray, float, int, int]:
    """
    Menghitung nilai Leverage (diagonal matriks topi Hii) dan ambang batasnya.

    Formula Hat Matrix:
        H = X (X^T X)^(-1) X^T
        h_ii = diag(H)
        Ambang Batas = 2p / n

    Parameters
    ----------
    X : np.ndarray
        Matriks fitur independen berukuran (n, k).
    fit_intercept : bool, default=True
        Apakah model menyertakan konstanta/intercept (menambah kolom 1 pada X).

    Returns
    -------
    h_ii : np.ndarray
        Nilai leverage untuk setiap observasi ke-i.
    threshold : float
        Nilai ambang batas leverage (2p / n).
    p : int
        Jumlah parameter model (k + 1 jika fit_intercept=True, atau k jika False).
    n : int
        Jumlah total observasi sampel.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, k = X.shape

    if fit_intercept:
        X_design = np.column_stack([np.ones(n), X])
        p = k + 1
    else:
        X_design = X
        p = k

    # (X^T X)^(-1) menggunakan pseudo-inverse untuk stabilitas numerik
    XtX = X_design.T @ X_design
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    # Matriks Topi H = X (X^T X)^(-1) X^T persis sesuai rumus Gambar 1
    H = X_design @ XtX_inv @ X_design.T
    h_ii = np.diag(H)

    threshold = (2.0 * p) / n

    return h_ii, threshold, p, n


def calculate_dffits(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
    model: Optional[OLSEstimator] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Menghitung statistik DFFITS dan Studentized Deleted Residuals (t_i).

    Formula (Persamaan 2.13):
        DFFITS_i = t_i * ( h_ii / (1 - h_ii) )^(1/2)
        Ambang Batas = 2 * sqrt(p / n)

    Formula Studentized Residual (t_i):
        t_i = e_i * sqrt( (n - p - 1) / ( SSE * (1 - h_ii) - e_i^2 ) )

    Parameters
    ----------
    X : np.ndarray
        Matriks fitur independen berukuran (n, k).
    y : np.ndarray
        Vektor target dependen berukuran (n,).
    fit_intercept : bool, default=True
        Apakah menyertakan konstanta/intercept.
    model : OLSEstimator, optional
        Objek OLSEstimator yang sudah di-fit atau akan di-fit.

    Returns
    -------
    dffits : np.ndarray
        Nilai DFFITS untuk setiap observasi ke-i.
    t_studentized : np.ndarray
        Nilai residual terstudentisasi (t_i) untuk setiap observasi ke-i.
    threshold : float
        Nilai ambang batas DFFITS (2 * sqrt(p / n)).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, k = X.shape

    if fit_intercept:
        X_design = np.column_stack([np.ones(n), X])
        p = k + 1
    else:
        X_design = X
        p = k

    # 1. Hitung Leverage (h_ii)
    h_ii, _, _, _ = calculate_leverage(X, fit_intercept=fit_intercept)

    # 2. Estimasi Regresi OLS menggunakan objek OLSEstimator
    if model is None:
        model = OLSEstimator(fit_intercept=fit_intercept)
        model.fit(X, y)
    elif model.params_ is None:
        model.fit(X, y)

    y_pred = X_design @ model.params_
    e = y - y_pred
    sse = float(np.sum(e ** 2))

    # 3. Hitung Studentized Deleted Residual (t_i) sesuai Gambar 3
    # t_i = e_i * sqrt( (n - p - 1) / ( SSE * (1 - h_ii) - e_i^2 ) )
    denom = sse * (1.0 - h_ii) - (e ** 2)
    # Hindari pembagian dengan nol atau angka negatif akibat pembulatan numerik
    denom = np.maximum(denom, 1e-12)

    df_residual = n - p - 1
    if df_residual <= 0:
        raise ValueError(f"Derajat bebas residual tidak mencukupi: n - p - 1 = {df_residual}")

    t_studentized = e * np.sqrt(df_residual / denom)

    # 4. Hitung DFFITS sesuai Gambar 2
    # DFFITS_i = t_i * ( h_ii / (1 - h_ii) )^(1/2)
    leverage_factor = np.maximum(h_ii / np.maximum(1.0 - h_ii, 1e-12), 0.0)
    dffits = t_studentized * np.sqrt(leverage_factor)

    threshold = 2.0 * np.sqrt(p / n)

    return dffits, t_studentized, threshold


def detect_outliers(
    X: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
    feature_names: Optional[List[str]] = None,
    target_name: Optional[str] = None,
    model: Optional[OLSEstimator] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Melakukan deteksi outlier menyeluruh menggunakan Leverage Values dan DFFITS.

    Kategori Diagnostik:
    - Leverage Outlier: h_ii > 2p/n (Titik yang memiliki nilai fitur X ekstrem).
    - DFFITS Outlier  : |DFFITS_i| > 2*sqrt(p/n) (Titik yang berpengaruh besar pada prediksi).
    - Influential Point: Memenuhi KEDUA kriteria sekaligus (paling berbahaya bagi OLS).

    Parameters
    ----------
    X : np.ndarray
        Matriks fitur independen.
    y : np.ndarray
        Vektor target dependen.
    fit_intercept : bool, default=True
        Apakah menyertakan konstanta/intercept.
    feature_names : list of str, optional
        Nama kolom fitur X.
    target_name : str, optional
        Nama variabel target y (misal: 'O3').
    model : OLSEstimator, optional
        Objek OLSEstimator yang sudah di-fit atau akan di-fit.

    Returns
    -------
    df_diagnostics : pd.DataFrame
        DataFrame lengkap dengan metrik diagnostik setiap baris observasi beserta nilai X dan y.
    summary : dict
        Ringkasan ambang batas, jumlah, dan indeks outlier yang terdeteksi.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    h_ii, lev_thresh, p, _ = calculate_leverage(X, fit_intercept=fit_intercept)
    dffits, t_studentized, dffits_thresh = calculate_dffits(X, y, fit_intercept=fit_intercept, model=model)

    is_lev_outlier = h_ii > lev_thresh
    is_dffits_outlier = np.abs(dffits) > dffits_thresh
    is_both = is_lev_outlier & is_dffits_outlier
    is_any = is_lev_outlier | is_dffits_outlier

    # Klasifikasi status
    status = np.full(n, "Normal", dtype=object)
    status[is_lev_outlier & ~is_dffits_outlier] = "Leverage Outlier"
    status[~is_lev_outlier & is_dffits_outlier] = "DFFITS Outlier"
    status[is_both] = "Influential & Leverage"

    data_dict = {
        "observation_id": np.arange(1, n + 1),
    }

    # Memasukkan nilai fitur prediktor (X)
    if feature_names is not None and len(feature_names) == X.shape[1]:
        for idx, col_name in enumerate(feature_names):
            data_dict[col_name] = X[:, idx]
    else:
        for idx in range(X.shape[1]):
            data_dict[f"X{idx+1}"] = X[:, idx]

    y_col = target_name if target_name else "actual_y"
    data_dict[y_col] = y
    data_dict["leverage"] = np.round(h_ii, 6)
    data_dict["leverage_outlier"] = is_lev_outlier
    data_dict["studentized_residual"] = np.round(t_studentized, 4)
    data_dict["dffits"] = np.round(dffits, 4)
    data_dict["abs_dffits"] = np.round(np.abs(dffits), 4)
    data_dict["dffits_outlier"] = is_dffits_outlier
    data_dict["status"] = status

    df_diagnostics = pd.DataFrame(data_dict)

    summary = {
        "n_samples": n,
        "n_parameters": p,
        "fit_intercept": fit_intercept,
        "leverage_threshold": float(lev_thresh),
        "dffits_threshold": float(dffits_thresh),
        "total_leverage_outliers": int(np.sum(is_lev_outlier)),
        "total_dffits_outliers": int(np.sum(is_dffits_outlier)),
        "total_influential_both": int(np.sum(is_both)),
        "total_unique_outliers": int(np.sum(is_any)),
        "leverage_outlier_indices": [int(idx + 1) for idx in np.where(is_lev_outlier)[0]],
        "dffits_outlier_indices": [int(idx + 1) for idx in np.where(is_dffits_outlier)[0]],
        "both_outlier_indices": [int(idx + 1) for idx in np.where(is_both)[0]],
    }

    return df_diagnostics, summary
