"""
Modul Penaksir OLS (Ordinary Least Squares).
"""

import numpy as np

from app.modules.regression.algorithms.base_estimator import BaseRegressionEstimator


class OLSEstimator(BaseRegressionEstimator):
    """
    Penaksir Ordinary Least Squares (OLS).
    """
    def __init__(self, fit_intercept=True):
        """
        Inisialisasi konfigurasi dan atribut penaksir OLS.

        Parameters:
        -----------
        fit_intercept : bool, default=True
            Menentukan apakah model menyertakan konstanta/titik potong (beta_0).
            Jika True, model mengestimasi konstanta beta_0 dan slope beta_1..p.
            Jika False, garis regresi dipaksa melewati titik asal (0, 0).
        """
        super().__init__(fit_intercept=fit_intercept)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        # 2. Penyesuaian dimensi:
        # Jika X berupa array 1 dimensi (regresi linear sederhana dengan 1 variabel bebas),
        # ubah bentuknya menjadi matriks kolom 2 dimensi (n, 1) agar dapat dioperasikan secara aljabar matriks.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 3. Pembentukan matriks desain:
        # Jika fit_intercept=True, tambahkan satu kolom bernilai 1 di kolom pertama untuk mengestimasi konstanta (beta_0).
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X

        # 4. Estimasi koefisien dengan persamaan normal: (X'X) beta = X'y
        try:
            # Menyelesaikan sistem persamaan linier (X'X) * beta = X'y untuk memperoleh nilai parameter beta
            params = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
        except np.linalg.LinAlgError:
            # Jika matriks (X'X) singular (tidak memiliki invers), gunakan invers semu (pseudo-inverse) untuk menghitung beta
            params = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y)

        # 5. Penyimpanan hasil estimasi ke dalam atribut objek
        self.params_ = params
        if self.fit_intercept:
            self.intercept_ = params[0]  # Elemen pertama adalah konstanta beta_0
            self.coef_ = params[1:]      # Elemen berikutnya adalah koefisien variabel bebas
        else:
            self.intercept_ = 0.0        # Tanpa konstanta
            self.coef_ = params

        # Mengembalikan parameter hasil estimasi
        return self.params_


def ols_estimator(X, y, fit_intercept=True):
    """
    Fungsi penaksir OLS yang mengembalikan parameter regresi.
    """
    return OLSEstimator(fit_intercept=fit_intercept).fit(X, y)
