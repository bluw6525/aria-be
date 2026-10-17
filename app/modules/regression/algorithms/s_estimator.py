import numpy as np
from scipy import optimize

try:
    from .robust_utils import (
        rho_tukey,
        weight_tukey,
        weight_s_scale,
        calculate_mad_scale
    )
    from .ols_estimator import OLSEstimator
except ImportError:
    from robust_utils import (
        rho_tukey,
        weight_tukey,
        weight_s_scale,
        calculate_mad_scale
    )
    from ols_estimator import OLSEstimator



def solve_s_scale(residuals, c=1.547, K=0.199, max_iter=100, tol=1e-6):
    """
    Menyelesaikan persamaan skala M implisit untuk Estimasi-S (Susanti et al. 2014):
    (1/n) sum rho_tukey(e / s, c=1.547) = 0.199
    """
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)
    
    s = calculate_mad_scale(residuals, divisor=0.6745)
    if s < 1e-8:
        s = np.std(residuals) if np.std(residuals) > 1e-8 else 1.0

    rho_eval = rho_tukey

    for _ in range(max_iter):
        u = residuals / s
        rho_val = rho_eval(u, c=c)
        mean_rho = np.mean(rho_val)
        if mean_rho < 1e-12:
            break
        s_new = s * np.sqrt(mean_rho / K)
        if np.abs(s_new - s) / (s + 1e-10) < tol:
            s = s_new
            break
        s = s_new

    def f_scale(log_s):
        curr_s = np.exp(log_s)
        return np.mean(rho_eval(residuals / curr_s, c=c)) - K

    try:
        log_s_min = np.log(max(1e-8, s * 0.01))
        log_s_max = np.log(max(1e-4, s * 100.0))
        if f_scale(log_s_min) * f_scale(log_s_max) < 0:
            sol = optimize.brentq(f_scale, log_s_min, log_s_max, xtol=tol)
            s = np.exp(sol)
    except Exception:
        pass

    return max(float(s), 1e-8)


class SEstimator:
    """
    Estimasi-S.
    
    Tahapan Algoritma:
    1. Estimasi koefisien regresi pada data menggunakan OLS (beta^0).
    2. Hitung residual e_i = y_i - y_hat_i.
    3. Hitung skala sigma:
       - Iterasi 1 : sigma = median(|e_i - median(e_i)|) / 0.6745 = 1.4826 * MAD
       - Iterasi > 1: sigma = sqrt( (1 / (n * K)) * sum(w_i * e_i^2) ), dengan K = 0.199
    4. Hitung u_i = e_i / sigma_i.
    5. Hitung nilai pembobot w_i:
       - Iterasi 1 : w_i = [1 - (u_i / 1.547)^2]^2 jika |u_i| <= 1.547, 0 jika > 1.547
       - Iterasi > 1: w_i = rho(u_i) / u_i^2 (fungsi Tukey Bisquare, c = 1.547)
    6. Hitung beta_S dengan metode WLS (Weighted Least Squares) dengan pembobot w_i:
       beta = (X' W X)^(-1) X' W Y
    7. Ulangi langkah 2-6 hingga diperoleh nilai beta_S yang konvergen.
    8. Mengembalikan hanya parameter regresi saja.
    """
    def __init__(self, c=1.547, K=0.199, max_iter=100, tol=1e-5, fit_intercept=True, **kwargs):
        self.c = float(c)
        self.K = float(K)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_intercept = fit_intercept

        self.params_ = None
        self.intercept_ = None
        self.coef_ = None
        self.scale_ = None
        self.weights_ = None
        self.residuals_ = None
        self.fittedvalues_ = None
        self.iteration_history_ = []
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, X, y):
        """
        Menaksir dan mengembalikan parameter regresi Estimasi-S dari data X dan y.
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

        # Estimasi parameter awal beta^0 dengan OLS
        if OLSEstimator is not None:
            ols = OLSEstimator(fit_intercept=False).fit(X_design, y)
            params = ols.params_.copy() if hasattr(ols, 'params_') else np.asarray(ols, dtype=float).copy()
        else:
            try:
                params = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
            except np.linalg.LinAlgError:
                params = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y)

        # Residual awal e_i = y_i - y_hat_i
        residuals = y - X_design @ params

        # Iterasi 1: sigma_1 = MAD / 0.6745
        scale = calculate_mad_scale(residuals, divisor=0.6745)

        self.iteration_history_ = []
        self.converged_ = False

        weights = None

        for it in range(1, self.max_iter + 1):
            # Iterasi > 1: sigma_i = sqrt( (1 / (n * K)) * sum(w_i * e_i^2) )
            if it > 1:
                scale = np.sqrt((1.0 / (n * self.K)) * np.sum(weights * (residuals**2)))
                scale = max(float(scale), 1e-8)

            # u_i = e_i / sigma_i
            u = residuals / scale

            # Nilai pembobot w_i
            if it == 1:
                weights = weight_tukey(u, c=self.c)
            else:
                weights = weight_s_scale(u, c=self.c)

            self.iteration_history_.append({
                'iteration': it,
                'params': params.copy(),
                'scale': float(scale),
                'weights': weights.copy(),
            })

            # Hitung beta_S dengan WLS: (X' W X)^(-1) X' W Y
            W = np.diag(weights)
            XtWX = X_design.T @ W @ X_design
            XtWy = X_design.T @ W @ y

            try:
                params_new = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                params_new = np.linalg.pinv(XtWX) @ XtWy

            # Residual baru
            residuals_new = y - X_design @ params_new

            # Cek konvergensi
            param_diff = np.max(np.abs(params_new - params) / (np.abs(params) + 1e-8))
            res_diff = np.abs(np.sum(np.abs(residuals_new)) - np.sum(np.abs(residuals)))

            params = params_new
            residuals = residuals_new

            if param_diff < self.tol or res_diff < 0.001:
                self.converged_ = True
                break

        # Simpan hasil akhir
        self.n_iter_ = it
        self.params_ = params
        if self.fit_intercept:
            self.intercept_ = params[0]
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

        # Skala akhir
        final_scale = np.sqrt((1.0 / (n * self.K)) * np.sum(weights * (residuals**2)))
        self.scale_ = max(float(final_scale), 1e-8)
        self.residuals_ = residuals
        self.fittedvalues_ = X_design @ params

        u_final = residuals / self.scale_
        self.weights_ = weight_s_scale(u_final, c=self.c)

        # Hanya mengembalikan parameter regresi saja
        return self.params_


def s_estimator(X, y, c=1.547, K=0.199, max_iter=100, tol=1e-5, fit_intercept=True, **kwargs):
    """
    Fungsi penaksir Estimasi-S yang hanya mengembalikan parameter regresi.
    """
    estimator = SEstimator(
        c=c,
        K=K,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        **kwargs
    )
    return estimator.fit(X, y)
