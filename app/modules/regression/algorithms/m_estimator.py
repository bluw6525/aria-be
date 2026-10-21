import numpy as np

from app.modules.regression.algorithms.base_estimator import BaseRegressionEstimator
from app.modules.regression.algorithms.ols_estimator import OLSEstimator
from app.modules.regression.algorithms.robust_utils import calculate_mad_scale, weight_tukey



class MEstimator(BaseRegressionEstimator):
    """
    Kelas Penaksir Estimasi-M (Tukey Bisquare).
    
    Tahapan Algoritma:
    1. Estimasi koefisien regresi awal menggunakan OLS (beta^0).
    2. Hitung residual e_i = y_i - y_hat_i.
    3. Hitung nilai skala sigma = median(|e_i - median(e)|) / 0.6745 (skala MAD).
    4. Hitung residual terstandarisasi u_i = e_i / sigma.
    5. Hitung pembobot Tukey Bisquare:
       w_i = [1 - (u_i / c)^2]^2 jika |u_i| <= c, dan 0 jika |u_i| > c (c = 4.685).
    6. Hitung pembaruan koefisien beta_M menggunakan WLS:
       beta = (X' W X)^(-1) X' W y.
    7. Ulangi langkah 2-6 hingga parameter beta_M konvergen.
    8. Mengembalikan parameter regresi hasil estimasi.
    """

    def __init__(self, c=4.685, tuning_constant=None, max_iter=100, tol=1e-5, fit_intercept=True, **kwargs):
        """
        Inisialisasi konfigurasi dan atribut penaksir Estimasi-M.

        Parameters:
        -----------
        c : float, default=4.685
            Konstanta tuning pembobot Tukey Bisquare (4.685 menghasilkan efisiensi 95% pada distribusi normal).
        tuning_constant : float, optional
            Alias untuk parameter c jika pengguna mengoper nama parameter alternatif.
        max_iter : int, default=100
            Batas maksimum jumlah iterasi pembobotan ulang (IRLS).
        tol : float, default=1e-5
            Batas toleransi perubahan nilai koefisien untuk kriteria konvergensi.
        fit_intercept : bool, default=True
            Menentukan apakah model menyertakan konstanta/titik potong (beta_0).
        """
        super().__init__(fit_intercept=fit_intercept)

        # Konfigurasi parameter input
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Penentuan nilai konstanta tuning c
        if tuning_constant is not None:
            self.c = float(tuning_constant)
        else:
            self.c = float(c)
        self.scale_ = None               # Nilai skala residual akhir (sigma MAD)
        self.weights_ = None             # Array bobot akhir tiap observasi w_i
        self.residuals_ = None           # Array residual akhir: y - y_hat
        self.fittedvalues_ = None        # Nilai prediksi model: y_hat
        self.iteration_history_ = []     # Riwayat riil tiap iterasi (iterasi, parameter, skala, bobot)
        self.converged_ = False          # Status apakah iterasi mencapai konvergensi
        self.n_iter_ = 0                 # Jumlah iterasi yang dijalankan hingga berhenti

    def fit(self, X, y):
        """
        Menaksir dan mengembalikan parameter regresi Estimasi-M dari data X dan y.
        """
        # 1. Konversi data masukan ke format array numerik (float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)

        # 2. Penyesuaian dimensi:
        # Jika X berupa array 1 dimensi (regresi linear sederhana dengan 1 variabel bebas),
        # ubah bentuknya menjadi matriks kolom 2 dimensi (n, 1) agar kompatibel dengan operasi aljabar matriks.
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 3. Pembentukan matriks desain:
        # Jika fit_intercept=True, tambahkan satu kolom bernilai 1 di awal untuk mengestimasi konstanta (beta_0).
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n), X])
        else:
            X_design = X

        # 4. Estimasi koefisien awal (beta^0) menggunakan OLS
        if OLSEstimator is not None:
            ols = OLSEstimator(fit_intercept=False).fit(X_design, y)
            params = ols.params_.copy() if hasattr(ols, 'params_') else np.asarray(ols, dtype=float).copy()
        else:
            try:
                # Menyelesaikan sistem persamaan linier (X'X) * beta = X'y untuk memperoleh koefisien awal OLS
                params = np.linalg.solve(X_design.T @ X_design, X_design.T @ y)
            except np.linalg.LinAlgError:
                # Jika matriks (X'X) singular, gunakan invers semu (pseudo-inverse)
                params = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y)

        # 5. Residual awal dan penaksir skala MAD (sigma)
        residuals = y - X_design @ params
        scale = calculate_mad_scale(residuals, divisor=0.6745)

        self.iteration_history_ = []
        self.converged_ = False

        # 6. Iterasi WLS (Iteratively Reweighted Least Squares / IRLS)
        for it in range(1, self.max_iter + 1):
            # Hitung residual terstandarisasi u_i = e_i / sigma
            u = residuals / scale

            # Hitung nilai pembobot Tukey Bisquare w_i
            weights = weight_tukey(u, c=self.c)

            # Catat riwayat proses iterasi
            self.iteration_history_.append({
                'iteration': it,
                'params': params.copy(),
                'scale': float(scale),
                'weights': weights.copy(),
            })

            # Pembentukan matriks pembobot diagonal W
            W = np.diag(weights)
            XtW = X_design.T @ W
            XtWX = XtW @ X_design
            XtWy = XtW @ y

            # Menyelesaikan sistem persamaan linier terboboti: (X'WX) * beta = X'Wy
            try:
                # Menyelesaikan sistem persamaan linier terboboti untuk mencari pembaruan koefisien beta
                params_new = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                # Jika matriks (X'WX) singular, gunakan invers semu (pseudo-inverse)
                params_new = np.linalg.pinv(XtWX) @ XtWy

            # Hitung residual baru berdasarkan koefisien baru
            residuals_new = y - X_design @ params_new

            # Perbarui estimasi skala MAD dari residual baru
            scale_new = calculate_mad_scale(residuals_new, divisor=0.6745)

            # Cek kriteria konvergensi (perubahan relatif parameter atau perubahan total residual)
            param_diff = np.max(np.abs(params_new - params) / (np.abs(params) + 1e-8))
            res_diff = np.abs(np.sum(np.abs(residuals_new)) - np.sum(np.abs(residuals)))

            # Perbarui nilai untuk iterasi berikutnya
            params = params_new
            residuals = residuals_new
            scale = scale_new

            # Hentikan iterasi jika kriteria konvergensi telah terpenuhi
            if param_diff < self.tol or res_diff < 0.001:
                self.converged_ = True
                break

        # 7. Penyimpanan hasil estimasi akhir ke dalam atribut objek
        self.n_iter_ = it
        self.params_ = params
        if self.fit_intercept:
            self.intercept_ = params[0]  # Elemen pertama adalah konstanta beta_0
            self.coef_ = params[1:]      # Elemen berikutnya adalah koefisien variabel bebas
        else:
            self.intercept_ = 0.0        # Tanpa konstanta
            self.coef_ = params

        self.residuals_ = residuals
        self.fittedvalues_ = X_design @ params
        self.scale_ = scale
        u_final = residuals / scale
        self.weights_ = weight_tukey(u_final, c=self.c)

        # Mengembalikan parameter hasil estimasi
        return self.params_


def m_estimator(X, y, c=4.685, tuning_constant=None, max_iter=100, tol=1e-5, fit_intercept=True, **kwargs):
    """
    Fungsi pembantu (helper) untuk langsung menaksir parameter regresi Estimasi-M (Tukey Bisquare).

    Fungsi ini memudahkan pengguna yang ingin mendapatkan parameter regresi
    secara langsung dalam satu baris kode tanpa perlu membuat instance class secara manual.

    Parameters:
    -----------
    X : array-like (n, p) atau (n,)
        Variabel independen.
    y : array-like (n,)
        Variabel dependen.
    c : float, default=4.685
        Konstanta tuning Tukey Bisquare.
    tuning_constant : float, optional
        Alias untuk parameter c.
    max_iter : int, default=100
        Batas iterasi maksimum.
    tol : float, default=1e-5
        Toleransi konvergensi.
    fit_intercept : bool, default=True
        Menyertakan konstanta/intersep atau tidak.

    Returns:
    --------
    params : np.ndarray
        Array koefisien parameter regresi hasil estimasi Estimasi-M.
    """
    estimator = MEstimator(
        c=c,
        tuning_constant=tuning_constant,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        **kwargs
    )
    return estimator.fit(X, y)
