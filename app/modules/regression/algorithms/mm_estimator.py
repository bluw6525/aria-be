import numpy as np

from app.modules.regression.algorithms.base_estimator import BaseRegressionEstimator
from app.modules.regression.core.robust_utils import calculate_mad_scale, weight_tukey
from app.modules.regression.algorithms.s_estimator import SEstimator



class MMEstimator(BaseRegressionEstimator):
    """
    Estimasi-MM.
    
    Tahapan Algoritma:
    1. Estimasi koefisien regresi pada data menggunakan OLS.
    2. Deteksi pencilan (outlier) pada data.
    3. Hitung nilai residual e_i = y_i - y_hat_i dan skala sigma_sn dari Estimasi-S.
    4. Ambil nilai skala konvergen sigma_MM = sigma_sn (fixed scale).
    5. Hitung u_i = e_i / sigma_sn.
    6. Hitung nilai pembobot w_i:
       w_i = [1 - (u_i / 1.547)^2]^2 jika |u_i| <= 1.547, 0 jika > 1.547.
    7. Hitung beta_MM menggunakan metode WLS dengan pembobot w_i:
       beta = (X' W X)^(-1) X' W Y.
    8. Ulangi langkah 5-7 hingga diperoleh nilai beta_MM yang konvergen.
    9. Mengembalikan hanya parameter regresi saja.
    """
    def __init__(self, k=1.547, max_iter=100, tol=1e-5, fit_intercept=True, **kwargs):
        super().__init__(fit_intercept=fit_intercept)
        self.k = float(k)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.scale_s_ = None  # Fixed scale dari Estimasi-S (sigma_sn)
        self.weights_ = None
        self.residuals_ = None
        self.fittedvalues_ = None
        self.iteration_history_ = []
        self.converged_ = False
        self.n_iter_ = 0
        self.s_estimator_ = None

    def fit(self, X, y, initial_s_estimator=None, scale=None):
        """
        Menaksir dan mengembalikan parameter regresi Estimasi-MM dari data X dan y.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        # Jika X berupa array 1 dimensi (regresi linear sederhana dengan 1 variabel bebas),
        # ubah bentuknya menjadi matriks kolom 2 dimensi (n, 1) agar kompatibel dengan operasi aljabar matriks.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X

        # Dapatkan estimasi awal beta_S dan skala konvergen sigma_sn dari Estimasi-S
        if initial_s_estimator is None:
            s_est = SEstimator(fit_intercept=False)
            s_params = s_est.fit(X_design, y)
            fixed_scale = float(s_est.scale_)
            params = np.asarray(s_params, dtype=float).copy()
        elif isinstance(initial_s_estimator, SEstimator):
            s_est = initial_s_estimator
            fixed_scale = float(s_est.scale_)
            params = np.asarray(s_est.params_, dtype=float).copy()
        else:
            s_est = None
            params = np.asarray(initial_s_estimator, dtype=float).copy()
            residuals = y - X_design @ params
            if scale is not None:
                fixed_scale = float(scale)
            else:
                fixed_scale = calculate_mad_scale(residuals, divisor=0.6745)

        self.s_estimator_ = s_est
        self.scale_s_ = fixed_scale

        self.iteration_history_ = []
        self.converged_ = False

        residuals = y - X_design @ params

        for it in range(1, self.max_iter + 1):
            # Nilai u_i = e_i / sigma_sn
            u = residuals / fixed_scale

            # Nilai pembobot w_i dengan k = 1.547
            weights = weight_tukey(u, k=self.k)

            self.iteration_history_.append({
                'iteration': it,
                'params': params.copy(),
                'scale': fixed_scale,
                'weights': weights.copy(),
            })

            # Hitung beta_MM dengan WLS: (X' W X)^(-1) X' W Y
            XtWX = X_design.T @ (weights[:, None] * X_design)
            XtWy = X_design.T @ (weights * y)

            try:
                params_new = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                params_new = np.linalg.pinv(XtWX) @ XtWy

            # Residual baru dan cek konvergensi
            residuals_new = y - X_design @ params_new
            param_diff = np.max(np.abs(params_new - params) / (np.abs(params) + 1e-8))
            res_diff = np.abs(np.sum(np.abs(residuals_new)) - np.sum(np.abs(residuals)))

            params = params_new
            residuals = residuals_new

            if param_diff < self.tol or res_diff < 0.001:
                self.converged_ = True
                break

        # Simpan atribut hasil akhir
        self.n_iter_ = it
        self.params_ = params
        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

        self.residuals_ = residuals
        self.fittedvalues_ = X_design @ params
        u_final = residuals / fixed_scale
        self.weights_ = weight_tukey(u_final, k=self.k)

        return self.params_


def mm_estimator(X, y, k=1.547, max_iter=100, tol=1e-5, fit_intercept=True, initial_s_estimator=None, **kwargs):
    """
    Fungsi penaksir Estimasi-MM yang hanya mengembalikan parameter regresi.
    """
    estimator = MMEstimator(
        k=k,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        **kwargs
    )
    return estimator.fit(X, y, initial_s_estimator=initial_s_estimator)
