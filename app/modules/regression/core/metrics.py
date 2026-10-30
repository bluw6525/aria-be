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
    # ss_tot = ss_res + ss_reg
    ss_tot = np.sum((y_true - y_mean) ** 2)

    if ss_tot == 0.0:
        return 0.0
    return float(1 -(ss_res/ss_tot))


def calculate_f_significance(
    X,
    y,
    intercept: float,
    coefs,
    alpha: float = 0.05,
) -> dict:
    """
    Menghitung uji signifikansi simultan (Uji F / ANOVA Regresi) persis sesuai rumus matriks MATLAB:
        x0 = ones(n, 1);
        xModel1 = [x0 x];
        b = [intercept, coefs...]';
        yhat = xModel1 * b;
        error = y - yhat;
        ssReg = b' * (xModel1' * y) - n*(mean(y))^2;
        ssRes = ((y' * y) - (b' * (xModel1' * y)));
        ssTot = y' * y - n*(mean(y))^2;
        [nx, px] = size(xModel1);
        dfReg = px-1;
        msReg = ssReg/dfReg;
        dfRes = nx-px;
        msRes = ssRes/dfRes;
        fHitung = msReg/msRes;
        fTabel = finv(0.95, dfReg, dfRes);
    """
    from scipy import stats

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).flatten()
    n = len(y)

    x0 = np.ones((n, 1), dtype=float)
    xModel1 = np.hstack([x0, X])
    nx, px = xModel1.shape

    coef_list = [float(intercept)] + [float(c) for c in np.asarray(coefs, dtype=float).flatten()]
    b = np.asarray(coef_list, dtype=float).reshape(-1, 1)

    y_col = y.reshape(-1, 1)
    mean_y = float(np.mean(y))

    # b' * (xModel1' * y)
    b_x_y = float((b.T @ (xModel1.T @ y_col)).item())

    # ssReg = b' * (xModel1' * y) - n*(mean(y))^2;
    ssReg = float(b_x_y - n * (mean_y ** 2))

    # ssRes = ((y' * y) - (b' * (xModel1' * y)));
    y_dot_y = float((y_col.T @ y_col).item())
    ssRes = float(y_dot_y - b_x_y)

    # Antisipasi kestabilan numerik jika terjadi sedikit deviasi pembulatan
    if ssReg < 0:
        yhat = (xModel1 @ b).flatten()
        ssReg = float(np.sum((yhat - mean_y) ** 2))
    if ssRes <= 0:
        yhat = (xModel1 @ b).flatten()
        ssRes = float(np.sum((y - yhat) ** 2))

    dfReg = px - 1
    msReg = ssReg / dfReg if dfReg > 0 else 0.0

    dfRes = nx - px
    msRes = ssRes / dfRes if dfRes > 0 else 1e-12

    fHitung = float(msReg / msRes) if msRes > 0 else 0.0

    if dfReg > 0 and dfRes > 0:
        fTabel = float(stats.f.ppf(1.0 - alpha, dfReg, dfRes))
        pVal = float(stats.f.sf(fHitung, dfReg, dfRes))
    else:
        fTabel = 0.0
        pVal = 1.0

    isSignificant = bool(fHitung > fTabel)

    return {
        "f_statistic": float(round(fHitung, 4)),
        "f_table": float(round(fTabel, 4)),
        "p_value": float(round(pVal, 6)),
        "is_significant": isSignificant,
        "df_reg": int(dfReg),
        "df_res": int(dfRes),
        "ss_reg": float(round(ssReg, 4)),
        "ss_res": float(round(ssRes, 4)),
    }

