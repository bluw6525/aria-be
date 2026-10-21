"""
Modul perhitungan metrik evaluasi model regresi:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 (Koefisien Determinasi / R-Squared)
"""

import numpy as np


def calculate_rmse(y_true, y_pred) -> float:
    """
    Menghitung Root Mean Squared Error (RMSE):
        RMSE = sqrt( sum( (y_hat_i - y_i)^2 ) / n )

    Parameters
    ----------
    y_true : array-like
        Nilai aktual target.
    y_pred : array-like
        Nilai prediksi dari model.

    Returns
    -------
    float:
        Nilai RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def calculate_mae(y_true, y_pred) -> float:
    """
    Menghitung Mean Absolute Error (MAE) Standar:
        MAE = sum( |y_hat_i - y_i| ) / n

    Parameters
    ----------
    y_true : array-like
        Nilai aktual target.
    y_pred : array-like
        Nilai prediksi dari model.

    Returns
    -------
    float:
        Nilai MAE standar.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def calculate_r2(y_true, y_pred) -> float:
    """
    Menghitung Koefisien Determinasi (R^2) Standar:
        R^2 = 1 - ( SS_res / SS_tot )
        SS_res = sum( (y_i - y_hat_i)^2 )
        SS_tot = sum( (y_i - y_bar)^2 )

    Parameters
    ----------
    y_true : array-like
        Nilai aktual target.
    y_pred : array-like
        Nilai prediksi dari model.

    Returns
    -------
    float:
        Nilai R-squared standar.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_reg = np.sum((y_pred - y_mean) ** 2)
    ss_tot = ss_res + ss_reg

    if ss_tot == 0.0:
        return 0.0
    return float(ss_reg/ss_tot)
